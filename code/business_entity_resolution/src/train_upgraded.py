"""Upgraded training pipeline with Locational & Lexical Hard-Negative Mining,
Dual-Model LightGBM + CatBoost Ensemble, and Isotonic Probability Calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import joblib
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
import numpy as np

from .blocking import CountryIndex, extract_prefix_keys
from .config import OPTIMAL_THRESHOLD, SEED, SOURCE_COLUMNS, TRUTH_COLUMNS
from .evaluate import evaluate_predictions, f05_for_entity
from .features import FEATURE_NAMES, extract_pair_features
from .io_utils import parse_id_list, read_tsv
from .normalize import NormalizedEntity, normalize_record


class CalibratedEnsemble:
    """Ensemble of LightGBM and CatBoost with Isotonic Probability Calibration."""

    def __init__(self, lgb_model, cat_model, iso_lgb, iso_cat, weight_lgb: float = 0.5):
        self.lgb_model = lgb_model
        self.cat_model = cat_model
        self.iso_lgb = iso_lgb
        self.iso_cat = iso_cat
        self.weight_lgb = weight_lgb
        self.weight_cat = 1.0 - weight_lgb

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_lgb = self.lgb_model.predict_proba(X)[:, 1]
        raw_cat = self.cat_model.predict_proba(X)[:, 1]

        cal_lgb = self.iso_lgb.predict(raw_lgb)
        cal_cat = self.iso_cat.predict(raw_cat)

        cal_lgb = np.clip(cal_lgb, 0.0, 1.0)
        cal_cat = np.clip(cal_cat, 0.0, 1.0)

        p1 = self.weight_lgb * cal_lgb + self.weight_cat * cal_cat
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


def build_upgraded_training_pipeline(
    data_root: Path,
    output_model_path: Path,
    sample_s1_limit: int = 50000,
    val_ratio: float = 0.20,
    max_competitors_per_file: int = 150000,
) -> None:
    print("=" * 70)
    print("STARTING UPGRADED HARD-NEGATIVE TRAINING PIPELINE")
    print(f"Data root:        {data_root}")
    print(f"Output model:     {output_model_path}")
    print(f"Sample limit S1:  {sample_s1_limit}")
    print(f"Validation ratio: {val_ratio}")
    print("=" * 70)
    t0 = time.time()

    # 1. Load S1 entities (stratified stream)
    print(f"\n1. Loading up to {sample_s1_limit} Source 1 reference records...")
    s1_records: list[NormalizedEntity] = []
    s1_path = data_root / "train" / "train_source1.tsv"

    for i, row in enumerate(read_tsv(s1_path, SOURCE_COLUMNS)):
        s1_records.append(
            normalize_record(row["entity_id"], row["business_name"], row["business_address"], row["country"])
        )
        if i + 1 >= sample_s1_limit:
            break

    total_s1 = len(s1_records)
    n_val = int(total_s1 * val_ratio)
    n_train = total_s1 - n_val
    train_s1 = s1_records[:n_train]
    val_s1 = s1_records[n_train:]
    all_s1_ids = {r.entity_id for r in s1_records}
    train_s1_ids = {r.entity_id for r in train_s1}

    print(f"Split: {len(train_s1)} train S1, {len(val_s1)} validation S1.")

    # 2. Load ground truth
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

    print(f"Ground truth loaded: {len(needed_targets)} true target entities in sample.")

    # 3. Collect Locational & Lexical competitor anchors from train S1
    print("\n3. Building locational & lexical competitor filter...")
    target_postal_codes: set[str] = {s.postal_code for s in s1_records if s.postal_code}
    target_prefix_keys: set[str] = set()
    for s in s1_records:
        for pkey in extract_prefix_keys(s.root_name):
            target_prefix_keys.add(pkey)

    print(f"Target filter: {len(target_postal_codes)} postal codes, {len(target_prefix_keys)} prefix keys.")

    # 4. Stream S2 and S3: index true matches + real hard competitor distractors
    print("\n4. Streaming Source 2 & 3: indexing true matches and locational/lexical hard negatives...")
    country_indexes: dict[str, CountryIndex] = defaultdict(lambda: CountryIndex(""))

    def get_index(country: str) -> CountryIndex:
        c = country.strip()
        if c not in country_indexes:
            country_indexes[c] = CountryIndex(c)
        return country_indexes[c]

    for src_filename in ("train_source2.tsv", "train_source3.tsv"):
        p = data_root / "train" / src_filename
        is_s2 = "source2" in src_filename
        needed_for_file = {t for t in needed_targets if (t.startswith("S2-") if is_s2 else t.startswith("S3-"))}
        found_needed = 0
        competitor_count = 0
        t_src = time.time()

        for row in read_tsv(p, SOURCE_COLUMNS):
            eid = row["entity_id"]
            is_needed = eid in needed_for_file

            # Check if this row is a locational or lexical competitor
            is_competitor = False
            if not is_needed and competitor_count < max_competitors_per_file:
                target_rec = normalize_record(eid, row["business_name"], row["business_address"], row["country"])
                if target_rec.postal_code and target_rec.postal_code in target_postal_codes:
                    is_competitor = True
                else:
                    for pkey in extract_prefix_keys(target_rec.root_name):
                        if pkey in target_prefix_keys:
                            is_competitor = True
                            break

            if is_needed:
                target_rec = normalize_record(eid, row["business_name"], row["business_address"], row["country"])
                idx = get_index(target_rec.country)
                idx.add_target(target_rec)
                found_needed += 1
            elif is_competitor:
                idx = get_index(target_rec.country)
                idx.add_target(target_rec)
                competitor_count += 1

        print(f"  {src_filename}: indexed {found_needed} true targets + {competitor_count} hard competitors in {time.time()-t_src:.1f}s.")

    print("Finalizing inverted token indices...")
    for idx in country_indexes.values():
        idx.finalize_tokens()

    total_indexed = sum(len(idx.entity_store) for idx in country_indexes.values())
    print(f"Total target records indexed: {total_indexed} across {len(country_indexes)} countries.")

    # 5. Generate Candidate Training Pairs
    print("\n5. Generating training candidate pairs and extracting pairwise features...")
    X_train_list: list[list[float]] = []
    y_train_list: list[int] = []

    for s1 in train_s1:
        idx = country_indexes.get(s1.country)
        if not idx:
            continue
        cands = idx.retrieve_candidates(s1, max_candidates=20)
        true_matches = train_truth.get(s1.entity_id, set())

        for tid, r_sc in cands:
            cand_rec = idx.entity_store.get(tid)
            if not cand_rec:
                continue
            feats = extract_pair_features(s1, cand_rec, r_sc)
            label = 1 if tid in true_matches else 0
            X_train_list.append(feats)
            y_train_list.append(label)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    positives = int(np.sum(y_train))
    negatives = len(y_train) - positives
    print(f"Generated {len(y_train)} training candidate pairs (Positives: {positives:,}, Negatives: {negatives:,}, Ratio: {negatives/max(positives,1):.1f}:1).")

    # 6. Train LightGBM + CatBoost Models
    print("\n6. Training LightGBM Pair Matcher...")
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

    print("\n7. Training CatBoost Pair Matcher...")
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

    # 8. Validation Feature Extraction & Probability Calibration
    print(f"\n8. Scoring validation set ({len(val_s1)} S1 entities) for Isotonic Probability Calibration...")
    val_cand_meta: dict[str, list[tuple[str, float]]] = {}
    val_pairs_X: list[list[float]] = []
    val_pairs_y: list[int] = []
    val_pair_map: list[tuple[str, str]] = []

    for s1 in val_s1:
        idx = country_indexes.get(s1.country)
        if not idx:
            val_cand_meta[s1.entity_id] = []
            continue
        cands = idx.retrieve_candidates(s1, max_candidates=20)
        true_matches = val_truth.get(s1.entity_id, set())

        for tid, r_sc in cands:
            cand_rec = idx.entity_store.get(tid)
            if not cand_rec:
                continue
            feats = extract_pair_features(s1, cand_rec, r_sc)
            label = 1 if tid in true_matches else 0
            val_pairs_X.append(feats)
            val_pairs_y.append(label)
            val_pair_map.append((s1.entity_id, tid))

    X_val = np.array(val_pairs_X, dtype=np.float32)
    y_val = np.array(val_pairs_y, dtype=np.int32)
    print(f"Validation candidate pairs: {len(y_val)} (Positives: {np.sum(y_val):,}, Negatives: {len(y_val)-np.sum(y_val):,}).")

    raw_val_lgb = model_lgb.predict_proba(X_val)[:, 1]
    raw_val_cat = model_cat.predict_proba(X_val)[:, 1]

    # Fit Isotonic Regressors
    print("Fitting Isotonic Regressors for probability calibration...")
    iso_lgb = IsotonicRegression(out_of_bounds="clip")
    iso_lgb.fit(raw_val_lgb, y_val)

    iso_cat = IsotonicRegression(out_of_bounds="clip")
    iso_cat.fit(raw_val_cat, y_val)

    cal_val_lgb = np.clip(iso_lgb.predict(raw_val_lgb), 0.0, 1.0)
    cal_val_cat = np.clip(iso_cat.predict(raw_val_cat), 0.0, 1.0)
    cal_val_ens = 0.5 * cal_val_lgb + 0.5 * cal_val_cat

    # Reconstruct per-S1 candidate predictions
    val_ens_scored: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (sid, tid), p in zip(val_pair_map, cal_val_ens):
        val_ens_scored[sid].append((tid, float(p)))

    # 9. Evaluate & Tune Decision Policy on Validation Split
    print("\n9. Tuning Decision Policy for Macro F_0.5 on Validation Split...")
    best_thresh = OPTIMAL_THRESHOLD
    best_f05 = -1.0
    best_res = None

    for thresh in np.arange(0.50, 0.85, 0.05):
        val_preds: dict[str, set[str]] = {}
        for s1 in val_s1:
            sid = s1.entity_id
            c_p = val_ens_scored.get(sid, [])
            if not c_p:
                val_preds[sid] = set()
                continue
            c_p.sort(key=lambda x: -x[1])
            top_tid, top_prob = c_p[0]

            # Singleton gate
            if top_prob < (thresh - 0.10):
                val_preds[sid] = set()
                continue

            matches = set()
            for tid, prob in c_p:
                if prob >= thresh and prob >= (0.75 * top_prob):
                    matches.add(tid)
                elif prob >= 0.85:
                    matches.add(tid)
            val_preds[sid] = matches

        res = evaluate_predictions(val_truth, val_preds)
        print(f"  Thresh {thresh:.2f} -> Macro F0.5 = {res.macro_f05:.6f} | Exact lists = {res.exact_lists} | Singleton false merges = {res.false_merges_on_singletons}")
        if res.macro_f05 > best_f05:
            best_f05 = res.macro_f05
            best_thresh = float(thresh)
            best_res = res

    print("\n" + "=" * 60)
    print("UPGRADED ENSEMBLE VALIDATION REPORT:")
    print(f"Best Decision Threshold: {best_thresh:.2f}")
    print(f"Macro F_0.5:             {best_res.macro_f05:.6f}")
    print(f"Exact match lists:       {best_res.exact_lists} / {best_res.entities} ({best_res.exact_lists/best_res.entities*100:.2f}%)")
    print(f"Correct singletons:      {best_res.correct_singletons} / {best_res.true_singletons} ({best_res.correct_singletons/best_res.true_singletons*100:.2f}%)")
    print(f"Singleton False Merges:  {best_res.false_merges_on_singletons}")
    print("=" * 60)

    # 10. Save Calibrated Ensemble Model Artifact
    calibrated_ensemble = CalibratedEnsemble(
        lgb_model=model_lgb,
        cat_model=model_cat,
        iso_lgb=iso_lgb,
        iso_cat=iso_cat,
        weight_lgb=0.5,
    )

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": calibrated_ensemble,
        "feature_names": FEATURE_NAMES,
        "threshold": best_thresh,
        "val_macro_f05": best_f05,
    }
    joblib.dump(payload, output_model_path)
    print(f"\nSaved calibrated ensemble artifact to {output_model_path} in {time.time()-t0:.1f}s.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Upgraded LightGBM + CatBoost Calibrated Ensemble.")
    parser.add_argument("--data-root", type=Path, default=Path("../../dataset").resolve())
    parser.add_argument("--output-model", type=Path, default=Path("artifacts/matcher_model_upgraded.joblib").resolve())
    parser.add_argument("--sample-limit", type=int, default=50000)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    args = parser.parse_args()

    build_upgraded_training_pipeline(
        data_root=args.data_root,
        output_model_path=args.output_model,
        sample_s1_limit=args.sample_limit,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
