"""Benchmark flat threshold vs per-segment calibration vs adaptive rank+prefix decision policies."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from .blocking import CountryIndex
from .config import SEED, SOURCE_COLUMNS, TRUTH_COLUMNS
from .evaluate import f05_for_entity
from .features import extract_pair_features
from .io_utils import parse_id_list, read_tsv
from .normalize import NormalizedEntity, normalize_record


def tune_policies(
    data_root: Path,
    model_path: Path,
    sample_s1: int = 10000,
    val_ratio: float = 0.333,
    distractor_limit: int = 50000,
) -> None:
    t0 = time.time()
    print("=" * 70)
    print("OPTIMIZING DECISION POLICIES (FLAT vs PER-SEGMENT vs RANK+PREFIX)")
    print("=" * 70)

    # 1. Load S1 validation split
    s1_records: list[NormalizedEntity] = []
    for i, row in enumerate(read_tsv(data_root / "train/train_source1.tsv", SOURCE_COLUMNS)):
        s1_records.append(
            normalize_record(row["entity_id"], row["business_name"], row["business_address"], row["country"])
        )
        if i + 1 >= sample_s1:
            break

    n_val = int(len(s1_records) * val_ratio)
    val_s1 = s1_records[-n_val:]
    val_s1_ids = {r.entity_id for r in val_s1}

    # 2. Ground truth
    val_truth: dict[str, set[str]] = {}
    needed_targets: set[str] = set()
    for row in read_tsv(data_root / "train/train_ground_truth.tsv", TRUTH_COLUMNS):
        sid = row["source1_entity_id"]
        if sid in val_s1_ids:
            matched = parse_id_list(row["matched_entity_ids"])
            val_truth[sid] = matched
            needed_targets.update(matched)
    for sid in val_s1_ids:
        if sid not in val_truth:
            val_truth[sid] = set()

    # 3. Stream target records
    country_indexes: dict[str, CountryIndex] = defaultdict(lambda: CountryIndex(""))
    for src_filename in ("train_source2.tsv", "train_source3.tsv"):
        p = data_root / "train" / src_filename
        distractor_count = 0
        found_needed = 0
        is_s2 = "source2" in src_filename
        needed_for_file = {t for t in needed_targets if (t.startswith("S2-") if is_s2 else t.startswith("S3-"))}

        for row in read_tsv(p, SOURCE_COLUMNS):
            eid = row["entity_id"]
            is_needed = eid in needed_for_file
            if is_needed or distractor_count < distractor_limit:
                target_rec = normalize_record(eid, row["business_name"], row["business_address"], row["country"])
                c = target_rec.country.strip()
                if c not in country_indexes:
                    country_indexes[c] = CountryIndex(c)
                country_indexes[c].add_target(target_rec)
                if is_needed:
                    found_needed += 1
                else:
                    distractor_count += 1
            if found_needed >= len(needed_for_file) and distractor_count >= distractor_limit:
                break

    for idx in country_indexes.values():
        idx.finalize_tokens()

    # 4. Load Model
    payload = joblib.load(model_path)
    model: lgb.LGBMClassifier = payload["model"]

    # 5. Extract features & model raw probabilities for all validation pairs
    print("\nScoring all validation pairs with model...")
    # Map from sid -> list of (tid, raw_prob, segment_key)
    entity_candidates: dict[str, list[dict[str, object]]] = {}

    for s1 in val_s1:
        sid = s1.entity_id
        idx = country_indexes.get(s1.country)
        if not idx:
            entity_candidates[sid] = []
            continue

        cands = idx.retrieve_candidates(s1, max_candidates=15)
        if not cands:
            entity_candidates[sid] = []
            continue

        feats_list = []
        c_meta = []
        for tid, r_sc in cands:
            cand_rec = idx.entity_store.get(tid)
            if not cand_rec:
                continue
            feats = extract_pair_features(s1, cand_rec, r_sc)
            feats_list.append(feats)
            src_tag = "S2" if tid.startswith("S2-") else "S3"
            seg_key = f"{s1.country}_{src_tag}"
            c_meta.append((tid, seg_key))

        if feats_list:
            probs = model.predict_proba(np.array(feats_list, dtype=np.float32))[:, 1]
            scored = []
            for (tid, seg_key), p in zip(c_meta, probs):
                scored.append({"tid": tid, "prob": float(p), "seg": seg_key})
            # sort descending by prob
            scored.sort(key=lambda x: -x["prob"])
            entity_candidates[sid] = scored
        else:
            entity_candidates[sid] = []

    print(f"Scored candidates for {len(entity_candidates)} validation entities.")

    # 6. Evaluate Policies
    def evaluate_policy(decision_fn) -> tuple[float, float, int]:
        scores = []
        total_preds = 0
        singletons = 0
        for s1 in val_s1:
            sid = s1.entity_id
            truth = val_truth[sid]
            cands = entity_candidates[sid]
            preds = decision_fn(s1, cands)
            scores.append(f05_for_entity(truth, preds))
            total_preds += len(preds)
            if len(preds) == 0:
                singletons += 1
        return float(np.mean(scores)), total_preds / len(val_s1), singletons

    print("\n--- A. FLAT THRESHOLD GRID SEARCH ---")
    best_flat_f05 = 0.0
    best_flat_th = 0.70
    for th in np.arange(0.50, 0.90, 0.05):
        f05, avg_l, s_cnt = evaluate_policy(lambda s1, cands, th=th: {c["tid"] for c in cands if c["prob"] >= th})
        print(f"Flat Th = {th:.2f} | Macro F0.5 = {f05:.4f} | Avg Links = {avg_l:.2f} | Singletons = {s_cnt}")
        if f05 > best_flat_f05:
            best_flat_f05 = f05
            best_flat_th = th

    print(f"\nBest Flat Threshold: {best_flat_th:.2f} -> Macro F0.5 = {best_flat_f05:.4f}")

    print("\n--- B. SEGMENT-SPECIFIC THRESHOLDS (US vs India) ---")
    # Grid search (th_us, th_in)
    best_seg_f05 = 0.0
    best_seg_params = (0.70, 0.70)
    for th_us in (0.60, 0.65, 0.70, 0.75):
        for th_in in (0.50, 0.55, 0.60, 0.65, 0.70):
            def seg_policy(s1, cands, u=th_us, i=th_in):
                t = u if s1.country == "US" else i
                return {c["tid"] for c in cands if c["prob"] >= t}
            f05, avg_l, s_cnt = evaluate_policy(seg_policy)
            if f05 > best_seg_f05:
                best_seg_f05 = f05
                best_seg_params = (th_us, th_in)

    print(f"Best Segment Policy (US={best_seg_params[0]:.2f}, IN={best_seg_params[1]:.2f}) -> Macro F0.5 = {best_seg_f05:.4f}")

    print("\n--- C. ADAPTIVE RANK + PREFIX DECISION RULE ---")
    # Rule: Keep candidate c_i if:
    # 1. c_i["prob"] >= min_prob (base floor)
    # 2. c_i["prob"] >= alpha * c_0["prob"] (relative gap from top candidate)
    # 3. Stop if probability gap (c_i - c_{i+1}) > drop_threshold (natural boundary)
    best_rank_f05 = 0.0
    best_rank_params = None

    for min_th in (0.45, 0.50, 0.55, 0.60):
        for alpha in (0.60, 0.70, 0.80):
            for max_keep in (3, 5, 8, 15):
                def rank_policy(s1, cands, mth=min_th, a=alpha, mk=max_keep):
                    if not cands:
                        return set()
                    top_prob = cands[0]["prob"]
                    # If even the top candidate is weak, emit singleton!
                    base_floor = 0.65 if s1.country == "US" else 0.55
                    if top_prob < base_floor:
                        return set()
                    kept = set()
                    for c in cands[:mk]:
                        if c["prob"] >= mth and c["prob"] >= (a * top_prob):
                            kept.add(c["tid"])
                        else:
                            break  # stop at prefix boundary
                    return kept

                f05, avg_l, s_cnt = evaluate_policy(rank_policy)
                if f05 > best_rank_f05:
                    best_rank_f05 = f05
                    best_rank_params = (min_th, alpha, max_keep, avg_l, s_cnt)

    print(f"Best Rank+Prefix Policy: MinTh={best_rank_params[0]:.2f}, RelGap={best_rank_params[1]:.2f}, MaxKeep={best_rank_params[2]}")
    print(f"  -> Macro F0.5 = {best_rank_f05:.4f} | Avg Links = {best_rank_params[3]:.2f} | Singletons = {best_rank_params[4]}")

    print("\n" + "=" * 70)
    print("POLICY COMPARISON SUMMARY:")
    print(f"1. Baseline Flat Threshold (0.70):  F0.5 = {best_flat_f05:.4f}")
    print(f"2. Country-Segmented Thresholds:    F0.5 = {best_seg_f05:.4f}")
    print(f"3. Adaptive Rank+Prefix Policy:     F0.5 = {best_rank_f05:.4f}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune decision policies on validation split.")
    parser.add_argument("--data-root", type=Path, default=Path("../../dataset").resolve())
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/matcher_model.joblib").resolve())
    parser.add_argument("--sample-s1", type=int, default=10000)
    parser.add_argument("--distractors", type=int, default=50000)
    args = parser.parse_args()

    tune_policies(
        data_root=args.data_root,
        model_path=args.model_path,
        sample_s1=args.sample_s1,
        distractor_limit=args.distractors,
    )


if __name__ == "__main__":
    main()
