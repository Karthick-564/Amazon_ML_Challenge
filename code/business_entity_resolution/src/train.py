"""Train supervised pairwise matching classifier and optimize threshold for Macro F_0.5."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from .blocking import CountryIndex
from .config import OPTIMAL_THRESHOLD, SEED, SOURCE_COLUMNS, TRUTH_COLUMNS
from .evaluate import evaluate_predictions
from .features import FEATURE_NAMES, extract_pair_features
from .io_utils import parse_id_list, read_tsv
from .normalize import NormalizedEntity, normalize_record


def build_training_pipeline(
    data_root: Path,
    output_model_path: Path,
    sample_s1_limit: int = 25000,
    val_ratio: float = 0.25,
) -> None:
    print(f"Starting training pipeline with seed {SEED}...")
    t0 = time.time()

    # 1. Load S1 entities
    print(f"Loading up to {sample_s1_limit} Source 1 records from {data_root}...")
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
    truth_path = data_root / "train" / "train_ground_truth.tsv"
    train_truth: dict[str, set[str]] = {}
    val_truth: dict[str, set[str]] = {}
    needed_targets: set[str] = set()

    for row in read_tsv(truth_path, TRUTH_COLUMNS):
        s1_id = row["source1_entity_id"]
        if s1_id in all_s1_ids:
            matched = parse_id_list(row["matched_entity_ids"])
            if s1_id in train_s1_ids:
                train_truth[s1_id] = matched
            else:
                val_truth[s1_id] = matched
            needed_targets.update(matched)

    print(f"Loaded ground truth. Needed true target records: {len(needed_targets)}.")

    # 3. Stream S2 and S3 to build country indexes
    print("Building country indexes from Source 2 and Source 3...")
    country_indexes: dict[str, CountryIndex] = defaultdict(lambda: CountryIndex(""))

    def get_index(country: str) -> CountryIndex:
        c = country.strip()
        if c not in country_indexes:
            country_indexes[c] = CountryIndex(c)
        return country_indexes[c]

    # Target pool: all needed ground truth targets + background distractors
    for src_filename in ("train_source2.tsv", "train_source3.tsv"):
        p = data_root / "train" / src_filename
        distractor_count = 0
        found_needed = 0
        needed_for_file = {t for t in needed_targets if (t.startswith("S2-") if "source2" in src_filename else t.startswith("S3-"))}

        for row in read_tsv(p, SOURCE_COLUMNS):
            eid = row["entity_id"]
            is_needed = eid in needed_for_file
            if is_needed or distractor_count < 30000:
                target_rec = normalize_record(eid, row["business_name"], row["business_address"], row["country"])
                idx = get_index(target_rec.country)
                idx.add_target(target_rec)
                if is_needed:
                    found_needed += 1
                else:
                    distractor_count += 1
            if found_needed >= len(needed_for_file) and distractor_count >= 30000:
                break

    for idx in country_indexes.values():
        idx.finalize_tokens()

    total_targets_loaded = sum(len(idx.entity_store) for idx in country_indexes.values())
    print(f"Indexed {total_targets_loaded} target records across {len(country_indexes)} countries in {time.time()-t0:.1f}s.")

    # 4. Generate candidate training pairs
    print("Generating candidate pairs and extracting pairwise features for training...")
    X_train: list[list[float]] = []
    y_train: list[int] = []

    for s1 in train_s1:
        idx = country_indexes.get(s1.country)
        if not idx:
            continue
        cands = idx.retrieve_candidates(s1, max_candidates=15)
        true_matches = train_truth.get(s1.entity_id, set())

        for tid, r_sc in cands:
            cand_rec = idx.entity_store.get(tid)
            if not cand_rec:
                continue
            feats = extract_pair_features(s1, cand_rec, r_sc)
            label = 1 if tid in true_matches else 0
            X_train.append(feats)
            y_train.append(label)

    X_train_arr = np.array(X_train, dtype=np.float32)
    y_train_arr = np.array(y_train, dtype=np.int32)
    positives = int(sum(y_train_arr))
    negatives = len(y_train_arr) - positives
    print(f"Generated {len(y_train_arr)} training candidate pairs (Positives: {positives}, Negatives: {negatives}).")

    # 5. Train LightGBM model
    print("Training LightGBM pair matcher...")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.08,
        num_leaves=31,
        min_child_samples=20,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train_arr, y_train_arr)

    # 6. Evaluate and tune threshold on validation set
    print(f"Evaluating on {len(val_s1)} held-out validation Source 1 entities...")
    val_cand_predictions: dict[str, list[tuple[str, float]]] = {}

    for s1 in val_s1:
        idx = country_indexes.get(s1.country)
        if not idx:
            val_cand_predictions[s1.entity_id] = []
            continue
        cands = idx.retrieve_candidates(s1, max_candidates=15)
        if not cands:
            val_cand_predictions[s1.entity_id] = []
            continue

        pair_feats: list[list[float]] = []
        c_ids: list[str] = []
        for tid, r_sc in cands:
            cand_rec = idx.entity_store.get(tid)
            if not cand_rec:
                continue
            pair_feats.append(extract_pair_features(s1, cand_rec, r_sc))
            c_ids.append(tid)

        if pair_feats:
            probs = model.predict_proba(np.array(pair_feats, dtype=np.float32))[:, 1]
            val_cand_predictions[s1.entity_id] = list(zip(c_ids, probs))
        else:
            val_cand_predictions[s1.entity_id] = []

    best_thresh = OPTIMAL_THRESHOLD
    best_f05 = -1.0
    best_res = None

    print("\nScanning decision thresholds for Macro F_0.5:")
    for thresh in np.arange(0.50, 0.90, 0.05):
        val_pred = {
            s1_id: {cid for cid, p in cand_probs if p >= thresh}
            for s1_id, cand_probs in val_cand_predictions.items()
        }
        res = evaluate_predictions(val_truth, val_pred)
        print(f"  Threshold {thresh:.2f} -> Macro F_0.5 = {res.macro_f05:.6f} | Exact lists = {res.exact_lists} | Singleton false merges = {res.false_merges_on_singletons}")
        if res.macro_f05 > best_f05:
            best_f05 = res.macro_f05
            best_thresh = float(thresh)
            best_res = res

    print("\n" + "=" * 50)
    print(f"VALIDATION REPORT:")
    print(f"Selected Threshold: {best_thresh:.2f}")
    print(f"Macro F_0.5: {best_res.macro_f05:.6f}")
    print(f"Entities: {best_res.entities}")
    print(f"Exact match lists: {best_res.exact_lists}")
    print(f"Singletons: {best_res.correct_singletons}/{best_res.true_singletons} (False merges: {best_res.false_merges_on_singletons})")
    print("=" * 50)

    # 7. Save model and metadata
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "threshold": best_thresh,
        "val_macro_f05": best_f05,
    }
    joblib.dump(payload, output_model_path)
    print(f"Saved trained model artifact to {output_model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM matching model.")
    parser.add_argument("--data-root", type=Path, default=Path("../../dataset").resolve())
    parser.add_argument("--output-model", type=Path, default=Path("artifacts/matcher_model.joblib").resolve())
    parser.add_argument("--sample-limit", type=int, default=25000)
    args = parser.parse_args()

    build_training_pipeline(args.data_root, args.output_model, sample_s1_limit=args.sample_limit)


if __name__ == "__main__":
    main()
