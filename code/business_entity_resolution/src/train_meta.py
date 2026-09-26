"""Train Ensemble Meta-Learner (LightGBM + CatBoost) on Multimodal Hybrid Features with Isotonic Calibration."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import joblib
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
import numpy as np

from .blocking import extract_prefix_keys
from .config import OPTIMAL_THRESHOLD, SEED, SOURCE_COLUMNS, TRUTH_COLUMNS
from .evaluate import evaluate_predictions
from .hybrid_blocker import HybridCountryIndex
from .io_utils import parse_id_list, read_tsv
from .meta_features import META_FEATURE_NAMES, extract_meta_pair_features
from .normalize import NormalizedEntity, normalize_record


from .ensemble import CalibratedMetaEnsemble


def train_meta_pipeline(
    data_root: Path,
    output_model_path: Path,
    sample_s1_limit: int = 25000,
    val_ratio: float = 0.20,
    max_competitors: int = 50000,
) -> None:
    print("=" * 70)
    print("TRAINING MULTIMODAL ENSEMBLE META-LEARNER (LIGHTGBM + CATBOOST)")
    print(f"Data root:        {data_root}")
    print(f"Output model:     {output_model_path}")
    print(f"S1 sample limit:  {sample_s1_limit}")
    print("=" * 70)
    t0 = time.time()

    # 1. Load S1 reference entities
    print(f"\n1. Loading {sample_s1_limit} Source 1 reference records...")
    s1_records: list[NormalizedEntity] = []
    s1_path = data_root / "train" / "train_source1.tsv"

    for i, row in enumerate(read_tsv(s1_path, SOURCE_COLUMNS)):
        s1_records.append(
            normalize_record(row["entity_id"], row["business_name"], row["business_address"], row["country"])
        )
        if i + 1 >= sample_s1_limit:
            break

    n_val = int(len(s1_records) * val_ratio)
    n_train = len(s1_records) - n_val
    train_s1 = s1_records[:n_train]
    val_s1 = s1_records[n_train:]
    all_s1_ids = {r.entity_id for r in s1_records}
    train_s1_ids = {r.entity_id for r in train_s1}
    print(f"Split: {len(train_s1)} train S1, {len(val_s1)} validation S1.")

    # 2. Ground Truth
    print("\n2. Loading ground truth for sample entities...")
    truth_path = data_root / "train" / "train_ground_truth.tsv"
    train_truth: dict[str, set[str]] = {}
    val_truth: dict[str, set[str]] = {}
    needed_targets: set[str] = set()

    for row in read_tsv(truth_path, TRUTH_COLUMNS):
        sid = row["source1_entity_id"]
        if sid in all_s1_ids:
            matched = parse_id_list(row["matched_entity_ids"])
            if sid in train_s1_ids:
                train_truth[sid] = matched
            else:
                val_truth[sid] = matched
            needed_targets.update(matched)

    for sid in train_s1_ids:
        if sid not in train_truth:
            train_truth[sid] = set()
    for s in val_s1:
        if s.entity_id not in val_truth:
            val_truth[s.entity_id] = set()

    print(f"Needed ground-truth target entities: {len(needed_targets)}.")

    # 3. Locational and Lexical competitor anchors
    target_postal_codes: set[str] = {s.postal_code for s in s1_records if s.postal_code}
    target_prefix_keys: set[str] = set()
    for s in s1_records:
        for pkey in extract_prefix_keys(s.root_name):
            target_prefix_keys.add(pkey)

    # 4. Stream Source 2 and Source 3 into HybridCountryIndex
    print("\n4. Building Hybrid Country Indexes (Lexical + BGE-M3 Dense)...")
    country_indexes: dict[str, HybridCountryIndex] = {}

    def get_index(c: str) -> HybridCountryIndex:
        c = c.strip()
        if c not in country_indexes:
            country_indexes[c] = HybridCountryIndex(c, use_gpu_dense=True)
        return country_indexes[c]

    for src_filename in ("train_source2.tsv", "train_source3.tsv"):
        p = data_root / "train" / src_filename
        is_s2 = "source2" in src_filename
        needed_for_file = {t for t in needed_targets if (t.startswith("S2-") if is_s2 else t.startswith("S3-"))}
        found_needed = 0
        comp_count = 0
        t_src = time.time()

        for row in read_tsv(p, SOURCE_COLUMNS):
            eid = row["entity_id"]
            is_needed = eid in needed_for_file

            is_competitor = False
            if not is_needed and comp_count < max_competitors:
                rec = normalize_record(eid, row["business_name"], row["business_address"], row["country"])
                if rec.postal_code and rec.postal_code in target_postal_codes:
                    is_competitor = True
                else:
                    for pkey in extract_prefix_keys(rec.root_name):
                        if pkey in target_prefix_keys:
                            is_competitor = True
                            break

            if is_needed:
                rec = normalize_record(eid, row["business_name"], row["business_address"], row["country"])
                get_index(rec.country).add_target(rec)
                found_needed += 1
            elif is_competitor:
                get_index(rec.country).add_target(rec)
                comp_count += 1

            if found_needed >= len(needed_for_file) and comp_count >= max_competitors:
                break

        print(f"  {src_filename}: indexed {found_needed} true targets + {comp_count} hard competitors in {time.time()-t_src:.1f}s.", flush=True)

    print("Finalizing indices & encoding dense target vectors on RTX 4090...", flush=True)
    for idx in country_indexes.values():
        idx.finalize_index()

    # 5. Retrieve Hybrid Candidates and Extract 25 Meta-Features for Training
    print("\n5. Generating Hybrid Candidates via Reciprocal Rank Fusion & extracting 25 meta-features...")
    X_train_list: list[list[float]] = []
    y_train_list: list[int] = []

    # Batch by 500 for GPU retrieval
    batch_size = 500
    for i in range(0, len(train_s1), batch_size):
        batch = train_s1[i : i + batch_size]
        # Partition by country
        c_groups = defaultdict(list)
        for s in batch:
            c_groups[s.country].append(s)

        for country, s1_group in c_groups.items():
            idx = country_indexes.get(country)
            if not idx:
                continue
            hybrid_cands_batch = idx.retrieve_hybrid_batch(s1_group, max_candidates=35)

            for s1_rec, cands in zip(s1_group, hybrid_cands_batch):
                true_matches = train_truth.get(s1_rec.entity_id, set())
                for tid, rrf_sc, d_sim in cands:
                    cand_rec = idx.entity_store.get(tid)
                    if not cand_rec:
                        continue
                    feats = extract_meta_pair_features(s1_rec, cand_rec, rrf_score=rrf_sc, dense_sim=d_sim)
                    label = 1 if tid in true_matches else 0
                    X_train_list.append(feats)
                    y_train_list.append(label)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    positives = int(np.sum(y_train))
    negatives = len(y_train) - positives
    print(f"Training dataset: {len(y_train):,} pairs (Positives: {positives:,}, Negatives: {negatives:,}, Ratio: {negatives/max(positives,1):.1f}:1).")

    # 6. Train LightGBM + CatBoost Meta-Learner
    print("\n6. Training LightGBM Meta-Learner...")
    t_lgb = time.time()
    model_lgb = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.06,
        num_leaves=35,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb.fit(X_train, y_train)
    print(f"  LightGBM trained in {time.time()-t_lgb:.1f}s.")

    print("\n7. Training CatBoost Meta-Learner...")
    t_cat = time.time()
    model_cat = CatBoostClassifier(
        iterations=300,
        learning_rate=0.06,
        depth=6,
        random_seed=SEED,
        thread_count=-1,
        verbose=0,
    )
    model_cat.fit(X_train, y_train)
    print(f"  CatBoost trained in {time.time()-t_cat:.1f}s.")

    # 7. Validation Evaluation & Isotonic Probability Calibration
    print("\n8. Evaluating on Validation Split & Fitting Isotonic Probability Calibration...")
    val_X_list: list[list[float]] = []
    val_y_list: list[int] = []
    val_meta: list[tuple[str, str]] = []

    for i in range(0, len(val_s1), batch_size):
        batch = val_s1[i : i + batch_size]
        c_groups = defaultdict(list)
        for s in batch:
            c_groups[s.country].append(s)

        for country, s1_group in c_groups.items():
            idx = country_indexes.get(country)
            if not idx:
                continue
            hybrid_cands_batch = idx.retrieve_hybrid_batch(s1_group, max_candidates=35)

            for s1_rec, cands in zip(s1_group, hybrid_cands_batch):
                true_matches = val_truth.get(s1_rec.entity_id, set())
                for tid, rrf_sc, d_sim in cands:
                    cand_rec = idx.entity_store.get(tid)
                    if not cand_rec:
                        continue
                    feats = extract_meta_pair_features(s1_rec, cand_rec, rrf_score=rrf_sc, dense_sim=d_sim)
                    label = 1 if tid in true_matches else 0
                    val_X_list.append(feats)
                    val_y_list.append(label)
                    val_meta.append((s1_rec.entity_id, tid))

    X_val = np.array(val_X_list, dtype=np.float32)
    y_val = np.array(val_y_list, dtype=np.int32)

    raw_val_lgb = model_lgb.predict_proba(X_val)[:, 1]
    raw_val_cat = model_cat.predict_proba(X_val)[:, 1]

    # Fit Isotonic Regressors
    iso_lgb = IsotonicRegression(out_of_bounds="clip")
    iso_lgb.fit(raw_val_lgb, y_val)

    iso_cat = IsotonicRegression(out_of_bounds="clip")
    iso_cat.fit(raw_val_cat, y_val)

    cal_val_lgb = np.clip(iso_lgb.predict(raw_val_lgb), 0.0, 1.0)
    cal_val_cat = np.clip(iso_cat.predict(raw_val_cat), 0.0, 1.0)
    cal_val_ens = 0.5 * cal_val_lgb + 0.5 * cal_val_cat

    # Reconstruct predictions per S1
    val_preds_map = defaultdict(list)
    for (sid, tid), prob in zip(val_meta, cal_val_ens):
        val_preds_map[sid].append((tid, float(prob)))

    # Grid search decision thresholds for Macro F0.5
    print("\n9. Scanning Decision Policy Thresholds on Validation Split:")
    best_thresh = OPTIMAL_THRESHOLD
    best_f05 = -1.0
    best_res = None

    for thresh in np.arange(0.50, 0.85, 0.05):
        eval_preds = {}
        for s1 in val_s1:
            sid = s1.entity_id
            c_p = val_preds_map.get(sid, [])
            if not c_p:
                eval_preds[sid] = set()
                continue
            c_p.sort(key=lambda x: -x[1])
            top_tid, top_prob = c_p[0]

            # Singleton gate
            if top_prob < (thresh - 0.10):
                eval_preds[sid] = set()
                continue

            matches = set()
            for tid, prob in c_p:
                if prob >= thresh and prob >= (0.75 * top_prob):
                    matches.add(tid)
                elif prob >= 0.85:
                    matches.add(tid)
            eval_preds[sid] = matches

        res = evaluate_predictions(val_truth, eval_preds)
        print(f"  Thresh {thresh:.2f} -> Macro F0.5 = {res.macro_f05:.6f} | Exact lists = {res.exact_lists} | Singleton false merges = {res.false_merges_on_singletons}")
        if res.macro_f05 > best_f05:
            best_f05 = res.macro_f05
            best_thresh = float(thresh)
            best_res = res

    print("\n" + "=" * 60)
    print("HYBRID ENSEMBLE META-LEARNER VALIDATION REPORT:")
    print(f"Selected Threshold: {best_thresh:.2f}")
    print(f"Macro F_0.5:        {best_res.macro_f05:.6f}")
    print(f"Exact match lists:  {best_res.exact_lists} / {best_res.entities} ({best_res.exact_lists/best_res.entities*100:.2f}%)")
    print(f"Correct singletons: {best_res.correct_singletons} / {best_res.true_singletons} ({best_res.correct_singletons/best_res.true_singletons*100:.2f}%)")
    print("=" * 60)

    # 8. Save Model Artifact
    ensemble_artifact = CalibratedMetaEnsemble(
        lgb_model=model_lgb,
        cat_model=model_cat,
        iso_lgb=iso_lgb,
        iso_cat=iso_cat,
        weight_lgb=0.5,
    )
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": ensemble_artifact,
        "feature_names": META_FEATURE_NAMES,
        "threshold": best_thresh,
        "val_macro_f05": best_f05,
    }
    joblib.dump(payload, output_model_path)
    print(f"\nSaved calibrated Meta-Learner model artifact to {output_model_path} in {time.time()-t0:.1f}s.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Hybrid Multimodal Meta-Learner.")
    parser.add_argument("--data-root", type=Path, default=Path("../../dataset").resolve())
    parser.add_argument("--output-model", type=Path, default=Path("artifacts/matcher_meta_ensemble.joblib").resolve())
    parser.add_argument("--sample-limit", type=int, default=25000)
    args = parser.parse_args()

    train_meta_pipeline(
        data_root=args.data_root,
        output_model_path=args.output_model,
        sample_s1_limit=args.sample_limit,
    )


if __name__ == "__main__":
    main()
