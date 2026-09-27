"""Train supervised pairwise matching classifier and optimize threshold for Macro F_0.5.

Gap 1 Fix: Training scale is now fully configurable via --sample-limit (default: None = all S1 data).
Gap 2 Fix: Sequential [:n_train] split replaced with a reproducible, S1-level stratified
           train/validation split that preserves country x match-count-bucket distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import tracemalloc
from collections import Counter, defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np

from .blocking import CountryIndex
from .config import OPTIMAL_THRESHOLD, SEED, SOURCE_COLUMNS, TRUTH_COLUMNS
from .decision_policy import adaptive_cap, evidence_aware_match
from .device import get_cuda_status, print_cuda_info
from .evaluate import evaluate_predictions, f05_for_entity
from .features import FEATURE_NAMES, extract_pair_features
from .io_utils import parse_id_list, read_tsv
from .metrics import compute_all_metrics
from .normalize import NormalizedEntity, normalize_country, normalize_record


# ---------------------------------------------------------------------------
# Stratified split helpers (Gap 2)
# ---------------------------------------------------------------------------

def _match_count_bucket(n: int) -> str:
    """Coarse bucket for match-count stratification: singleton / low / mid / high."""
    if n == 0:
        return "singleton"
    if n <= 2:
        return "low"
    if n <= 5:
        return "mid"
    return "high"


def stratified_train_val_split(
    s1_records: list[NormalizedEntity],
    full_truth: dict[str, set[str]],
    val_ratio: float,
    seed: int,
) -> tuple[list[NormalizedEntity], list[NormalizedEntity]]:
    """Stratified S1-level split preserving (country, match_count_bucket) distribution.

    Each stratum is shuffled independently with a fixed seed so the split is
    reproducible and country + multiplicity balanced.  Train/val are strictly disjoint.
    """
    rng = np.random.default_rng(seed)

    strata: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(s1_records):
        match_count = len(full_truth.get(rec.entity_id, set()))
        bucket = _match_count_bucket(match_count)
        key = f"{rec.country}|{bucket}"
        strata[key].append(i)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for key, indices in sorted(strata.items()):
        arr = np.array(indices, dtype=np.int64)
        rng.shuffle(arr)
        n_val = max(1, int(len(arr) * val_ratio))
        val_indices.extend(arr[:n_val].tolist())
        train_indices.extend(arr[n_val:].tolist())

    train_indices.sort()
    val_indices.sort()

    train_s1 = [s1_records[i] for i in train_indices]
    val_s1 = [s1_records[i] for i in val_indices]
    return train_s1, val_s1


# ---------------------------------------------------------------------------
# Detailed validation metrics helper
# ---------------------------------------------------------------------------

def compute_detailed_metrics(
    val_s1: list[NormalizedEntity],
    val_truth: dict[str, set[str]],
    val_cand_predictions: dict[str, list[tuple[str, float, list[float]]]],
    thresh: float,
    use_evidence_policy: bool = False,
) -> dict:
    """Return macro F0.5, precision, recall, and candidate recall for a given threshold.

    If use_evidence_policy is True, applies evidence_aware_match policy (Gap 10).
    Otherwise, applies simple flat threshold p >= thresh (Baseline).
    """
    if use_evidence_policy:
        val_pred = {}
        for s1 in val_s1:
            sid = s1.entity_id
            scored = val_cand_predictions.get(sid, [])
            val_pred[sid] = evidence_aware_match(s1, scored, thresh)
    else:
        val_pred = {
            s1_id: {cid for cid, p, _ in cand_probs if p >= thresh}
            for s1_id, cand_probs in val_cand_predictions.items()
        }

    f05_scores, prec_scores, rec_scores = [], [], []
    cand_recall_hits = 0
    cand_recall_total = 0

    for s1 in val_s1:
        sid = s1.entity_id
        truth = val_truth.get(sid, set())
        predicted = val_pred.get(sid, set())
        candidates = {cid for cid, _, _ in val_cand_predictions.get(sid, [])}

        score = f05_for_entity(truth, predicted)
        f05_scores.append(score)

        if truth:
            tp = len(truth & predicted)
            prec_scores.append(tp / len(predicted) if predicted else 0.0)
            rec_scores.append(tp / len(truth))
            cand_recall_total += 1
            if truth & candidates:
                cand_recall_hits += 1

    macro_f05 = sum(f05_scores) / len(f05_scores) if f05_scores else 0.0
    mean_prec = sum(prec_scores) / len(prec_scores) if prec_scores else 0.0
    mean_rec = sum(rec_scores) / len(rec_scores) if rec_scores else 0.0
    cand_recall = cand_recall_hits / cand_recall_total if cand_recall_total > 0 else 0.0

    return {
        "macro_f05": macro_f05,
        "mean_precision": mean_prec,
        "mean_recall": mean_rec,
        "candidate_recall": cand_recall,
        "n_val_entities": len(val_s1),
    }


# ---------------------------------------------------------------------------
# Core training pipeline
# ---------------------------------------------------------------------------

def build_training_pipeline(
    data_root: Path,
    output_model_path: Path,
    sample_s1_limit: Optional[int] = None,
    val_ratio: float = 0.25,
    bg_per_country: int = 50000,
    verbose: bool = True,
) -> dict:
    """Train LightGBM pairwise matcher and evaluate on stratified validation split.

    Args:
        data_root:         Path to dataset root (must have a train/ sub-directory).
        output_model_path: Where to save the trained model artifact (.joblib).
        sample_s1_limit:   Number of S1 entities to sample for training+validation.
                           ``None`` (default) = use the entire S1 training file.
        val_ratio:         Fraction of selected S1 entities held out for validation.
        bg_per_country:    Maximum background (non-true-target) records loaded per
                           country from S2/S3 (Gap 3 fix).  Default 50,000.
                           These become the hard-negative candidate pool for
                           blocking; a per-country cap ensures that every country
                           in the training split has a rich distractor pool
                           regardless of its absolute share of the data.
        verbose:           Print progress to stdout when True.

    Returns:
        dict with keys: macro_f05, mean_precision, mean_recall, candidate_recall,
        n_train, n_val, train_time_s, peak_memory_mb, best_threshold.
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    tracemalloc.start()
    t0 = time.time()

    limit_desc = f"{sample_s1_limit:,}" if sample_s1_limit else "ALL"
    log("=" * 60)
    log(f"Training pipeline  |  S1 limit: {limit_desc}  |  val_ratio: {val_ratio}")
    log(f"Data root: {data_root}")
    log(print_cuda_info().strip())
    log("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load S1 entities  (Gap 1: None = full dataset, no hardcoded cap)
    # ------------------------------------------------------------------
    log(f"[1/7] Loading Source 1 records (limit={limit_desc})...")
    s1_records: list[NormalizedEntity] = []
    s1_path = data_root / "train" / "train_source1.tsv"
    for i, row in enumerate(read_tsv(s1_path, SOURCE_COLUMNS)):
        s1_records.append(
            normalize_record(
                row["entity_id"], row["business_name"],
                row["business_address"], row["country"],
            )
        )
        if sample_s1_limit and (i + 1) >= sample_s1_limit:
            break

    total_s1 = len(s1_records)
    log(f"    Loaded {total_s1:,} S1 records in {time.time() - t0:.1f}s.")

    # ------------------------------------------------------------------
    # 2. Load ground truth for all selected S1 entities
    #    (needed before the split so stratification uses real match counts)
    # ------------------------------------------------------------------
    log("[2/7] Loading ground truth for selected S1 entities...")
    all_s1_ids = {r.entity_id for r in s1_records}
    full_truth: dict[str, set[str]] = {}
    needed_targets: set[str] = set()

    truth_path = data_root / "train" / "train_ground_truth.tsv"
    for row in read_tsv(truth_path, TRUTH_COLUMNS):
        s1_id = row["source1_entity_id"]
        if s1_id in all_s1_ids:
            matched = parse_id_list(row["matched_entity_ids"])
            full_truth[s1_id] = matched
            needed_targets.update(matched)
            if len(full_truth) >= len(all_s1_ids):
                break

    log(f"    Ground truth loaded. Unique true target IDs needed: {len(needed_targets):,}.")

    # ------------------------------------------------------------------
    # 3. Stratified train / validation split  (Gap 2)
    # ------------------------------------------------------------------
    log("[3/7] Performing stratified train/validation split...")
    train_s1, val_s1 = stratified_train_val_split(s1_records, full_truth, val_ratio, SEED)

    train_s1_ids = {r.entity_id for r in train_s1}
    val_s1_ids = {r.entity_id for r in val_s1}

    assert not (train_s1_ids & val_s1_ids), "BUG: Train and val sets overlap!"

    train_truth = {sid: full_truth.get(sid, set()) for sid in train_s1_ids}
    val_truth = {sid: full_truth.get(sid, set()) for sid in val_s1_ids}

    train_strata: Counter = Counter()
    val_strata: Counter = Counter()
    for r in train_s1:
        n = len(train_truth.get(r.entity_id, set()))
        train_strata[f"{r.country}|{_match_count_bucket(n)}"] += 1
    for r in val_s1:
        n = len(val_truth.get(r.entity_id, set()))
        val_strata[f"{r.country}|{_match_count_bucket(n)}"] += 1

    log(f"    Train: {len(train_s1):,}  |  Val: {len(val_s1):,}")
    log("    Stratum (country|bucket)    train       val")
    for s in sorted(set(list(train_strata) + list(val_strata))):
        log(f"      {s:<30}  {train_strata.get(s, 0):>7,}  {val_strata.get(s, 0):>6,}")

    # ------------------------------------------------------------------
    # 4. Stream S2/S3 into country indexes  (Gap 3: per-country background)
    # ------------------------------------------------------------------
    log(f"[4/7] Building country indexes (bg_per_country={bg_per_country:,})...")
    log("      GAP 3: Per-country background pool replaces global 30K distractor cap.")
    log("      All needed true targets are always loaded; additional records fill")
    log("      the background pool per country so blocking retrieves hard negatives.")
    country_indexes: dict[str, CountryIndex] = {}

    def get_index(country: str) -> CountryIndex:
        c = country.strip()
        if c not in country_indexes:
            country_indexes[c] = CountryIndex(c)
        return country_indexes[c]

    for src_filename in ("train_source2.tsv", "train_source3.tsv"):
        p = data_root / "train" / src_filename
        prefix = "S2-" if "source2" in src_filename else "S3-"
        needed_for_file = {t for t in needed_targets if t.startswith(prefix)}

        # Per-country background counters (Gap 3 & Gap 12 fix)
        needed_countries = {r.country for r in s1_records}
        unsaturated_countries = set(needed_countries)
        country_bg_counts: dict[str, int] = defaultdict(int)
        found_needed = 0
        total_bg = 0
        t_src = time.time()

        row_count = 0
        for row in read_tsv(p, SOURCE_COLUMNS):
            row_count += 1
            if row_count % 500_000 == 0:
                elapsed = time.time() - t_src
                log(f"      ... {row_count:,} rows scanned ({elapsed:.0f}s) | "
                    f"found={found_needed:,}/{len(needed_for_file):,} bg={total_bg:,}")

            eid = row["entity_id"]
            is_needed = eid in needed_for_file
            bg_saturated = not unsaturated_countries

            if not is_needed and bg_saturated:
                continue

            canon_c = normalize_country(row["country"], row["business_address"])

            # Always load true targets; load background until per-country cap hit
            if is_needed or (canon_c in unsaturated_countries):
                target_rec = normalize_record(
                    eid, row["business_name"], row["business_address"], canon_c
                )
                idx = get_index(target_rec.country)
                idx.add_target(target_rec)
                if is_needed:
                    found_needed += 1
                else:
                    country_bg_counts[canon_c] += 1
                    total_bg += 1
                    if country_bg_counts[canon_c] >= bg_per_country:
                        unsaturated_countries.discard(canon_c)

            # Early-exit: all needed targets found AND all needed countries saturated
            if found_needed >= len(needed_for_file) and not unsaturated_countries:
                log(f"      Early-exit at row {row_count:,}: all targets found & bg saturated.")
                break

        bg_summary = ", ".join(f"{c}={n:,}" for c, n in sorted(country_bg_counts.items()))
        log(f"    {src_filename}: {found_needed:,} true targets + {total_bg:,} bg "
            f"({bg_summary}) in {time.time() - t_src:.1f}s.")

    for idx in country_indexes.values():
        idx.finalize_tokens()

    total_targets_loaded = sum(len(idx.entity_store) for idx in country_indexes.values())
    log(f"    Indexed {total_targets_loaded:,} target records across {len(country_indexes)} countries.")

    # ------------------------------------------------------------------
    # 5. Generate candidate training pairs (Gap 3: hard negatives from blocking)
    # ------------------------------------------------------------------
    log("[5/7] Generating candidate pairs from blocking (hard negatives)...")
    log("      GAP 3: Negatives are records returned by the actual blocker, not")
    log("             random distractors.  Per-country pool ensures same-country hard negs.")
    X_train: list[list[float]] = []
    y_train: list[int] = []

    # Hard-negative quality tracking
    hard_neg_count = 0          # blocking-retrieved negatives (the hard ones)
    train_cand_recall_hits = 0  # S1 entities where blocking found >= 1 true match
    train_cand_recall_total = 0
    total_train_cands = 0

    # candidate_pairs.tsv written here (training split blocking output)
    cand_pairs_path = output_model_path.parent / "candidate_pairs_train.tsv"
    cand_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with cand_pairs_path.open("w", encoding="utf-8", newline="") as f_cp:
        cp_writer = csv.writer(f_cp, delimiter="\t", lineterminator="\n")
        cp_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        total_fixed_train_cands = 0
        train_fixed_cand_recall_hits = 0

        total_s1 = len(train_s1)
        t_pair_gen = time.time()
        for i, s1 in enumerate(train_s1, 1):
            if i % 2500 == 0 or i == total_s1:
                elapsed = time.time() - t_pair_gen
                rate = i / max(elapsed, 0.001)
                log(f"      ... [{i:,}/{total_s1:,}] entities processed ({i/total_s1*100:.1f}%) | "
                    f"{rate:.1f} s1/s | pairs={len(y_train):,} ({elapsed:.0f}s elapsed)")

            idx = country_indexes.get(s1.country)
            if not idx:
                cp_writer.writerow([s1.entity_id, ""])
                continue

            # Gap 9: Adaptive candidate cap
            cap = adaptive_cap(s1)
            raw_cands = idx.retrieve_candidates_full(s1, max_candidates=max(15, cap))
            cands = raw_cands[:cap]
            true_matches = train_truth.get(s1.entity_id, set())

            # Write candidate pairs row (the actual adaptive candidates passed to the matcher)
            cand_ids = [cr.entity_id for cr in cands]
            cp_writer.writerow([s1.entity_id, ",".join(cand_ids)])

            # Candidate recall tracking (adaptive vs fixed-15)
            total_train_cands += len(cands)
            fixed_cands = raw_cands[:15]
            fixed_cand_ids = [cr.entity_id for cr in fixed_cands]
            total_fixed_train_cands += len(fixed_cands)

            if true_matches:
                train_cand_recall_total += 1
                if set(cand_ids) & true_matches:
                    train_cand_recall_hits += 1
                if set(fixed_cand_ids) & true_matches:
                    train_fixed_cand_recall_hits += 1

            # Context feature values for this S1
            n_cands = len(cands)
            max_rs = cands[0].retrieval_score if cands else 1.0

            for rank, cr in enumerate(cands, 1):
                cand_rec = idx.entity_store.get(cr.entity_id)
                if not cand_rec:
                    continue
                rs_norm   = cr.retrieval_score / max_rs
                rs_margin = max_rs - cr.retrieval_score  # 0 for rank-1
                feats = extract_pair_features(
                    s1, cand_rec, cr.retrieval_score,
                    n_candidates=n_cands,
                    retrieval_rank=rank,
                    retrieval_score_norm=rs_norm,
                    retrieval_margin=rs_margin,
                    n_routes=cr.n_routes,
                )
                label = 1 if cr.entity_id in true_matches else 0
                X_train.append(feats)
                y_train.append(label)
                if label == 0:
                    hard_neg_count += 1  # every retrieved non-match = hard negative

    X_train_arr = np.array(X_train, dtype=np.float32)
    y_train_arr = np.array(y_train, dtype=np.int32)
    positives = int(y_train_arr.sum())
    negatives = len(y_train_arr) - positives
    train_cand_recall = (
        train_cand_recall_hits / train_cand_recall_total if train_cand_recall_total else 0.0
    )
    fixed_train_cand_recall = (
        train_fixed_cand_recall_hits / train_cand_recall_total if train_cand_recall_total else 0.0
    )
    avg_cands_per_s1 = total_train_cands / max(len(train_s1), 1)
    fixed_avg_cands = total_fixed_train_cands / max(len(train_s1), 1)

    log(f"    {len(y_train_arr):,} training pairs  (pos={positives:,}  neg={negatives:,})")
    log(f"    Hard negatives (blocking-retrieved): {hard_neg_count:,} / {negatives:,} = "
        f"{100*hard_neg_count/max(negatives,1):.1f}% of all negatives")
    log(f"    Train candidate recall (Adaptive cap): {train_cand_recall:.4f} (avg {avg_cands_per_s1:.2f} cands/S1)")
    log(f"    Train candidate recall (Fixed-15 cap): {fixed_train_cand_recall:.4f} (avg {fixed_avg_cands:.2f} cands/S1)")
    log(f"    Candidate pairs written: {cand_pairs_path}")

    # ------------------------------------------------------------------
    # 6. Train LightGBM
    # ------------------------------------------------------------------
    log("[6/7] Training LightGBM pair matcher...")
    t_lgb = time.time()
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
    log(f"    LightGBM done in {time.time() - t_lgb:.1f}s.")

    # ------------------------------------------------------------------
    # 7. Evaluate on stratified validation split & tune threshold
    # ------------------------------------------------------------------
    log(f"[7/7] Evaluating on {len(val_s1):,} stratified validation entities...")
    val_cand_predictions: dict[str, list[tuple[str, float, list[float]]]] = {}
    val_cand_recall_hits = 0
    val_fixed_cand_recall_hits = 0
    val_cand_recall_total = 0
    val_total_cands = 0
    val_total_fixed_cands = 0

    total_val = len(val_s1)
    t_val_start = time.time()
    for i, s1 in enumerate(val_s1, 1):
        if i % 1000 == 0 or i == total_val:
            elapsed = time.time() - t_val_start
            rate = i / max(elapsed, 0.001)
            log(f"      ... [{i:,}/{total_val:,}] validation entities evaluated ({i/total_val*100:.1f}%) | {rate:.1f} s1/s")

        idx = country_indexes.get(s1.country)
        if not idx:
            val_cand_predictions[s1.entity_id] = []
            continue

        # Gap 9: Adaptive candidate cap
        cap = adaptive_cap(s1)
        raw_cands = idx.retrieve_candidates_full(s1, max_candidates=max(15, cap))
        cands = raw_cands[:cap]
        true_matches = val_truth.get(s1.entity_id, set())

        val_total_cands += len(cands)
        fixed_cands = raw_cands[:15]
        val_total_fixed_cands += len(fixed_cands)

        if true_matches:
            val_cand_recall_total += 1
            if set(cr.entity_id for cr in cands) & true_matches:
                val_cand_recall_hits += 1
            if set(cr.entity_id for cr in fixed_cands) & true_matches:
                val_fixed_cand_recall_hits += 1

        if not cands:
            val_cand_predictions[s1.entity_id] = []
            continue

        pair_feats: list[list[float]] = []
        c_ids: list[str] = []
        n_cands  = len(cands)
        max_rs   = cands[0].retrieval_score

        for rank, cr in enumerate(cands, 1):
            cand_rec = idx.entity_store.get(cr.entity_id)
            if not cand_rec:
                continue
            rs_norm   = cr.retrieval_score / max_rs
            rs_margin = max_rs - cr.retrieval_score
            pair_feats.append(extract_pair_features(
                s1, cand_rec, cr.retrieval_score,
                n_candidates=n_cands,
                retrieval_rank=rank,
                retrieval_score_norm=rs_norm,
                retrieval_margin=rs_margin,
                n_routes=cr.n_routes,
            ))
            c_ids.append(cr.entity_id)

        if pair_feats:
            probs = model.predict_proba(np.array(pair_feats, dtype=np.float32))[:, 1]
            val_cand_predictions[s1.entity_id] = [
                (cid, float(p), feats) for cid, p, feats in zip(c_ids, probs, pair_feats)
            ]
        else:
            val_cand_predictions[s1.entity_id] = []

    val_adaptive_cand_recall = val_cand_recall_hits / val_cand_recall_total if val_cand_recall_total else 0.0
    val_fixed_cand_recall = val_fixed_cand_recall_hits / val_cand_recall_total if val_cand_recall_total else 0.0
    val_avg_cands = val_total_cands / max(len(val_s1), 1)
    val_fixed_avg_cands = val_total_fixed_cands / max(len(val_s1), 1)

    log(f"    Val candidate recall (Adaptive cap): {val_adaptive_cand_recall:.4f} (avg {val_avg_cands:.2f} cands/S1)")
    log(f"    Val candidate recall (Fixed-15 cap): {val_fixed_cand_recall:.4f} (avg {val_fixed_avg_cands:.2f} cands/S1)")

    # Scan 1: Baseline flat threshold
    best_flat_thresh = OPTIMAL_THRESHOLD
    best_flat_f05 = -1.0
    best_flat_metrics: dict = {}
    for thresh in np.arange(0.35, 0.92, 0.05):
        m = compute_detailed_metrics(val_s1, val_truth, val_cand_predictions, float(thresh), use_evidence_policy=False)
        if m["macro_f05"] > best_flat_f05:
            best_flat_f05 = m["macro_f05"]
            best_flat_thresh = float(thresh)
            best_flat_metrics = m

    # Scan 2: Evidence-aware decision policy (Gap 10)
    best_evidence_thresh = OPTIMAL_THRESHOLD
    best_evidence_f05 = -1.0
    best_evidence_metrics: dict = {}
    log("\n    Evidence-Aware Policy Threshold scan (Macro F0.5):")
    for thresh in np.arange(0.35, 0.92, 0.05):
        m = compute_detailed_metrics(val_s1, val_truth, val_cand_predictions, float(thresh), use_evidence_policy=True)
        log(
            f"      thresh={thresh:.2f}  "
            f"F0.5={m['macro_f05']:.6f}  "
            f"prec={m['mean_precision']:.4f}  "
            f"rec={m['mean_recall']:.4f}"
        )
        if m["macro_f05"] > best_evidence_f05:
            best_evidence_f05 = m["macro_f05"]
            best_evidence_thresh = float(thresh)
            best_evidence_metrics = m

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024 ** 2)
    total_time = time.time() - t0

    # Compute final predictions on validation set using the best evidence policy threshold
    val_final_preds = {
        s1.entity_id: evidence_aware_match(s1, val_cand_predictions.get(s1.entity_id, []), best_evidence_thresh)
        for s1 in val_s1
    }

    # Compute comprehensive evaluation metrics
    comp_metrics = compute_all_metrics(
        val_s1, val_truth, val_cand_predictions, val_final_preds, total_target_pool_size=10_000_000
    )

    cuda_info = get_cuda_status()
    device_label = f"{cuda_info['device_name']} ({cuda_info['memory_total_mb']:.0f} MB VRAM)" if cuda_info["cuda_available"] else "CPU (Multi-core)"

    log("\n" + "=" * 80)
    log("                       END-TO-END VALIDATION REPORT")
    log("=" * 80)
    log(f"  Hardware Device       : {device_label}")
    log(f"  Training S1 Entities  : {len(train_s1):,} ({len(y_train_arr):,} pairwise pairs)")
    log(f"  Validation S1 Entities: {len(val_s1):,} (Disjoint held-out stratified split)")
    log(f"  Total Targets Loaded  : {total_targets_loaded:,}")
    log(f"  Execution Time        : {total_time:.1f}s | Peak Memory: {peak_mb:.1f} MB")
    log("-" * 80)
    log("  1. COMPETITION BENCHMARK EVALUATION (Official Metric):")
    log(f"     S1 Macro F0.5           : {comp_metrics['macro_f05']:.6f}  (Baseline Flat: {best_flat_metrics.get('macro_f05', 0):.6f}, Gain: {comp_metrics['macro_f05'] - best_flat_metrics.get('macro_f05', 0):+.6f})")
    log(f"     Mean Precision          : {comp_metrics['mean_precision']:.6f}")
    log(f"     Mean Recall             : {comp_metrics['mean_recall']:.6f}")
    log(f"     Exact Match Lists       : {comp_metrics['exact_matches']:,} / {len(val_s1):,} ({comp_metrics['exact_match_pct']:.2f}%)")
    log(f"     True Singletons         : {comp_metrics['total_singletons']:,} ({comp_metrics['singleton_accuracy']*100:.2f}% correctly predicted, {comp_metrics['false_merges_on_singletons']:,} false merges)")
    log("-" * 80)
    log("  2. PAIRWISE CLASSIFICATION & CONFUSION MATRIX:")
    log(f"     Accuracy                : {comp_metrics['accuracy']:.6f}")
    log(f"     Precision               : {comp_metrics['precision']:.6f}  |  Recall (Sensitivity): {comp_metrics['recall']:.6f}")
    log(f"     Specificity             : {comp_metrics['specificity']:.6f}  |  Balanced Accuracy  : {comp_metrics['balanced_accuracy']:.6f}")
    log(f"     F1 Score                : {comp_metrics['f1']:.6f}  |  F0.5 Score          : {comp_metrics['f05']:.6f}  |  F2 Score: {comp_metrics['f2']:.6f}")
    log(f"     MCC (Matthews Coeff)    : {comp_metrics['mcc']:.6f}  |  Cohen's Kappa       : {comp_metrics['cohen_kappa']:.6f}")
    log(f"     ROC-AUC                 : {comp_metrics['roc_auc']:.6f}  |  PR-AUC (Avg Prec)   : {comp_metrics['pr_auc']:.6f}")
    log(f"     Log Loss                : {comp_metrics['log_loss']:.6f}  |  Brier Score         : {comp_metrics['brier_score']:.6f}")
    log(f"     Confusion Matrix        : TP = {comp_metrics['tp']:,}  |  FP = {comp_metrics['fp']:,}  |  FN = {comp_metrics['fn']:,}  |  TN = {comp_metrics['tn']:,}")
    log("-" * 80)
    log("  3. RETRIEVAL & RANKING QUALITY:")
    log(f"     Candidate Recall        : {comp_metrics['candidate_recall']:.6f} ({comp_metrics['candidate_recall']*100:.2f}%)")
    log(f"     All-Target Recall       : {comp_metrics['all_target_candidate_recall']:.6f}")
    log(f"     MRR (Mean Recip Rank)   : {comp_metrics['mrr']:.6f}  |  MAP@K: {comp_metrics['map_at_k']:.6f}")
    log(f"     Avg Candidates / S1     : {comp_metrics['avg_candidates_per_s1']:.2f}  (Candidate Reduction: {comp_metrics['candidate_reduction_pct']:.4f}%)")
    rec_k_str = ", ".join(f"{k}: {v:.4f}" for k, v in comp_metrics["recall_at_k"].items())
    hit_k_str = ", ".join(f"{k}: {v:.4f}" for k, v in comp_metrics["hit_at_k"].items())
    log(f"     Recall@K                : {rec_k_str}")
    log(f"     Hit@K                   : {hit_k_str}")
    log("-" * 80)
    log("  4. PER-COUNTRY SUBGROUP PERFORMANCE:")
    for c, cinfo in comp_metrics["country_breakdown"].items():
        log(f"     Country '{c:<10}' ({cinfo['n_entities']:>6,} entities): Macro F0.5 = {cinfo['macro_f05']:.6f} | Prec = {cinfo['precision']:.4f} | Rec = {cinfo['recall']:.4f}")
    log("-" * 80)
    log("  5. MATCH-COUNT MULTIPLICITY PERFORMANCE:")
    for b, binfo in comp_metrics["bucket_breakdown"].items():
        log(f"     Bucket '{b:<15}' ({binfo['n_entities']:>6,} entities): Macro F0.5 = {binfo['macro_f05']:.6f} | Prec = {binfo['precision']:.4f} | Rec = {binfo['recall']:.4f}")
    log("=" * 80)

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "threshold": best_evidence_thresh,
        "val_macro_f05": comp_metrics["macro_f05"],
        "baseline_flat_macro_f05": best_flat_f05,
        "n_train": len(train_s1),
        "n_val": len(val_s1),
        "metrics": comp_metrics,
        "device_info": cuda_info,
    }
    joblib.dump(payload, output_model_path)
    log(f"Saved model artifact: {output_model_path}")

    # Export structured JSON evaluation report
    report_json_path = output_model_path.parent / "evaluation_report.json"
    full_report = {
        "hardware": {
            "device": device_label,
            "cuda_available": cuda_info["cuda_available"],
            "vram_mb": cuda_info["memory_total_mb"],
        },
        "dataset_split": {
            "sample_s1_limit": sample_s1_limit,
            "n_train_s1": len(train_s1),
            "n_val_s1": len(val_s1),
            "n_train_pairs": len(y_train_arr),
            "hard_neg_count": hard_neg_count,
            "hard_neg_pct": 100 * hard_neg_count / max(negatives, 1),
            "bg_per_country": bg_per_country,
        },
        "runtime": {
            "train_time_s": total_time,
            "peak_memory_mb": peak_mb,
        },
        "optimal_policy": {
            "decision_threshold": best_evidence_thresh,
            "baseline_flat_threshold": best_flat_thresh,
        },
        "evaluation_metrics": comp_metrics,
    }
    with report_json_path.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    log(f"Saved evaluation report JSON: {report_json_path}")

    return {
        "macro_f05": comp_metrics["macro_f05"],
        "baseline_flat_macro_f05": best_flat_f05,
        "mean_precision": comp_metrics["mean_precision"],
        "mean_recall": comp_metrics["mean_recall"],
        "val_candidate_recall": comp_metrics["candidate_recall"],
        "val_fixed_cand_recall": val_fixed_cand_recall,
        "train_candidate_recall": train_cand_recall,
        "avg_cands_per_s1": comp_metrics["avg_candidates_per_s1"],
        "hard_neg_count": hard_neg_count,
        "hard_neg_pct": 100 * hard_neg_count / max(negatives, 1),
        "n_train": len(train_s1),
        "n_val": len(val_s1),
        "train_time_s": total_time,
        "peak_memory_mb": peak_mb,
        "best_threshold": best_evidence_thresh,
        "sample_s1_limit": sample_s1_limit,
        "bg_per_country": bg_per_country,
        "full_metrics": comp_metrics,
    }


# ---------------------------------------------------------------------------
# Benchmark mode
# ---------------------------------------------------------------------------

def run_benchmark(
    data_root: Path,
    output_dir: Path,
    sizes: list[Optional[int]],
    val_ratio: float = 0.25,
) -> None:
    """Run training at each S1 size and print/save a comparative results table."""
    import traceback

    results: list[dict] = []
    for size in sizes:
        size_label = f"{size:,}" if size else "ALL"
        safe_label = size_label.replace(",", "k")
        model_path = output_dir / f"matcher_model_s1_{safe_label}.joblib"
        print(f"\n{'#' * 60}")
        print(f"# BENCHMARK: sample_limit={size_label}")
        print(f"{'#' * 60}")
        try:
            r = build_training_pipeline(
                data_root=data_root,
                output_model_path=model_path,
                sample_s1_limit=size,
                val_ratio=val_ratio,
                verbose=True,
            )
            r["label"] = size_label
            results.append(r)
        except Exception as exc:
            print(f"  ERROR for size={size_label}: {exc}")
            traceback.print_exc()

    print("\n")
    print("=" * 115)
    print("BENCHMARK SUMMARY")
    header = (
        f"{'S1 Limit':>10}  {'Train':>9}  {'Val':>7}  "
        f"{'F0.5':>8}  {'Prec':>8}  {'Recall':>8}  "
        f"{'CandRec':>8}  {'Time(s)':>8}  {'Mem(MB)':>8}  {'Thresh':>7}"
    )
    print(header)
    print("-" * 115)
    for r in results:
        print(
            f"{r.get('label', '?'):>10}  "
            f"{r.get('n_train', 0):>9,}  "
            f"{r.get('n_val', 0):>7,}  "
            f"{r.get('macro_f05', 0):>8.6f}  "
            f"{r.get('mean_precision', 0):>8.6f}  "
            f"{r.get('mean_recall', 0):>8.6f}  "
            f"{r.get('candidate_recall', 0):>8.6f}  "
            f"{r.get('train_time_s', 0):>8.1f}  "
            f"{r.get('peak_memory_mb', 0):>8.1f}  "
            f"{r.get('best_threshold', 0):>7.2f}"
        )
    print("=" * 115)

    results_path = output_dir / "benchmark_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nBenchmark results saved to {results_path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train LightGBM entity matching model.\n"
            "Gap 1 fix: configurable S1 limit (default=None=all).\n"
            "Gap 2 fix: stratified random train/val split."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("../../dataset").resolve(),
        help="Dataset root directory (default: ../../dataset).",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("artifacts/matcher_model.joblib").resolve(),
        help="Output path for the trained model artifact.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        metavar="N",
        help="S1 records to use (Gap 1 fix). Default: None = ALL records.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.25,
        help="Fraction of S1 held out for stratified validation (default: 0.25).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark mode: train at multiple S1 sizes and compare results.",
    )
    parser.add_argument(
        "--benchmark-sizes",
        type=str,
        default="25000,100000,250000,500000,None",
        help="Comma-separated S1 sizes for benchmark. Use 'None' for full dataset.",
    )
    parser.add_argument(
        "--bg-per-country",
        type=int,
        default=50000,
        help="Maximum background records loaded per country (default: 50000).",
    )
    args = parser.parse_args()

    if args.benchmark:
        raw_sizes = [s.strip() for s in args.benchmark_sizes.split(",")]
        sizes: list[Optional[int]] = []
        for s in raw_sizes:
            if s.lower() == "none" or s == "":
                sizes.append(None)
            else:
                sizes.append(int(s))
        run_benchmark(
            data_root=args.data_root,
            output_dir=args.benchmark_output_dir,
            sizes=sizes,
            val_ratio=args.val_ratio,
        )
    else:
        build_training_pipeline(
            data_root=args.data_root,
            output_model_path=args.output_model,
            sample_s1_limit=args.sample_limit,
            val_ratio=args.val_ratio,
            bg_per_country=args.bg_per_country,
            verbose=True,
        )


if __name__ == "__main__":
    main()
