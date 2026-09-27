"""Statistically representative validation tuning and open-set country policy calibration.

Gap 11 Fix: Evaluates on a statistically representative stratified validation split
            (default: 25,000 S1 sample -> 6,250 held-out entities).
Gap 12 Fix: Generic country normalization, partitioning, and fallback handling
            ensures unseen countries work automatically without hardcoded assumptions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import joblib
import lightgbm as lgb
import numpy as np

from .blocking import CountryIndex
from .config import OPTIMAL_THRESHOLD, SEED, SOURCE_COLUMNS, TRUTH_COLUMNS
from .decision_policy import adaptive_cap, evidence_aware_match
from .device import get_cuda_status, print_cuda_info
from .evaluate import f05_for_entity
from .features import extract_pair_features
from .io_utils import parse_id_list, read_tsv
from .normalize import NormalizedEntity, normalize_country, normalize_record
from .train import stratified_train_val_split


def tune_decision_policies(
    data_root: Path,
    model_path: Path,
    sample_s1: int = 25000,
    val_ratio: float = 0.25,
    bg_per_country: int = 5000,
    output_json_path: Path | None = None,
) -> dict:
    """Tune thresholds and evidence-aware decision policies on a statistically representative validation split.

    Reports: Macro F0.5, Precision, Recall, Candidate Recall, Runtime, Memory, and Open-Set Country behavior.
    """
    t0 = time.time()
    tracemalloc.start()
    cuda_status = get_cuda_status()

    print("=" * 75)
    print("REPRESENTATIVE DECISION POLICY TUNING & OPEN-SET CALIBRATION")
    print(print_cuda_info().strip())
    print(f"Data root:       {data_root}")
    print(f"Model path:      {model_path}")
    print(f"S1 sample limit: {sample_s1:,} records (val_ratio={val_ratio})")
    print(f"Bg per country:  {bg_per_country:,} distractors")
    print("=" * 75)

    # 1. Load S1 records with canonical country normalization (Gap 12)
    print(f"\n[1/6] Loading {sample_s1:,} Source 1 records...")
    s1_records: list[NormalizedEntity] = []
    for i, row in enumerate(read_tsv(data_root / "train" / "train_source1.tsv", SOURCE_COLUMNS)):
        s1_records.append(
            normalize_record(row["entity_id"], row["business_name"], row["business_address"], row["country"])
        )
        if sample_s1 and (i + 1) >= sample_s1:
            break

    total_s1 = len(s1_records)
    print(f"    Loaded {total_s1:,} S1 records in {time.time() - t0:.1f}s.")

    # 2. Load Ground Truth
    print("[2/6] Loading ground truth for selected entities...")
    all_s1_ids = {r.entity_id for r in s1_records}
    full_truth: dict[str, set[str]] = {}
    needed_targets: set[str] = set()

    for row in read_tsv(data_root / "train" / "train_ground_truth.tsv", TRUTH_COLUMNS):
        sid = row["source1_entity_id"]
        if sid in all_s1_ids:
            matched = parse_id_list(row["matched_entity_ids"])
            full_truth[sid] = matched
            needed_targets.update(matched)
            if len(full_truth) >= len(all_s1_ids):
                break

    for sid in all_s1_ids:
        if sid not in full_truth:
            full_truth[sid] = set()

    # 3. Stratified Train / Validation Split (Gap 11)
    print(f"[3/6] Splitting into stratified train/val (ratio={val_ratio})...")
    train_s1, val_s1 = stratified_train_val_split(s1_records, full_truth, val_ratio, SEED)
    val_truth = {s.entity_id: full_truth[s.entity_id] for s in val_s1}

    # Discover validation countries dynamically (Gap 12 open-set)
    val_countries = sorted({s.country for s in val_s1})
    needed_val_countries = set(val_countries)
    val_targets = set()
    for sid in (s.entity_id for s in val_s1):
        val_targets.update(val_truth.get(sid, set()))

    print(f"    Held-out validation size: {len(val_s1):,} S1 entities ({len(val_targets):,} true targets).")
    print(f"    Observed country segments: {', '.join(val_countries)}")

    # 4. Stream and Index Target Database
    print(f"\n[4/6] Building country target indexes (bg_per_country={bg_per_country:,})...")
    country_indexes: dict[str, CountryIndex] = defaultdict(lambda: CountryIndex(""))

    def get_index(country_str: str) -> CountryIndex:
        c = normalize_country(country_str)
        if c not in country_indexes:
            country_indexes[c] = CountryIndex(c)
        return country_indexes[c]

    for src_filename in ("train_source2.tsv", "train_source3.tsv"):
        p = data_root / "train" / src_filename
        prefix = "S2-" if "source2" in src_filename else "S3-"
        needed_for_file = {t for t in val_targets if t.startswith(prefix)}

        country_bg_counts: dict[str, int] = defaultdict(int)
        found_needed = 0
        total_bg = 0
        t_src = time.time()

        for row in read_tsv(p, SOURCE_COLUMNS):
            eid = row["entity_id"]
            is_needed = eid in needed_for_file
            bg_saturated = all(country_bg_counts[c] >= bg_per_country for c in needed_val_countries)

            if not is_needed and bg_saturated:
                continue

            canon_c = normalize_country(row["country"], row["business_address"])

            if is_needed or (canon_c in needed_val_countries and country_bg_counts[canon_c] < bg_per_country):
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

            # Early exit: all needed targets found and needed countries saturated
            if found_needed >= len(needed_for_file) and bg_saturated:
                break

        bg_summary = ", ".join(f"{c}={n:,}" for c, n in sorted(country_bg_counts.items()))
        print(f"    {src_filename}: {found_needed:,} targets + {total_bg:,} bg ({bg_summary}) in {time.time()-t_src:.1f}s.")

    for idx in country_indexes.values():
        idx.finalize_tokens()

    total_targets_indexed = sum(len(idx.entity_store) for idx in country_indexes.values())
    print(f"    Total target records indexed: {total_targets_indexed:,} across {len(country_indexes)} countries.")

    # 5. Load Trained Model
    print(f"\n[5/6] Loading matcher model from {model_path}...")
    payload = joblib.load(model_path)
    model: lgb.LGBMClassifier = payload["model"]

    # 6. Candidate Retrieval & Scoring on Validation Split
    print("\n[6/6] Retrieving candidates & scoring validation pairs...")
    t_score = time.time()
    val_cand_predictions: dict[str, list[tuple[str, float, list[float]]]] = {}

    val_cand_recall_hits = 0
    val_fixed_cand_recall_hits = 0
    val_cand_recall_total = 0
    val_total_cands = 0
    val_total_fixed_cands = 0

    for s1 in val_s1:
        idx = country_indexes.get(s1.country)
        # Open-set country fallback: check case-insensitive match
        if not idx:
            c_lower = s1.country.lower()
            for k, v in country_indexes.items():
                if k.lower() == c_lower:
                    idx = v
                    break
        if not idx and len(country_indexes) == 1:
            idx = next(iter(country_indexes.values()))

        if not idx:
            val_cand_predictions[s1.entity_id] = []
            continue

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

    dt_score = time.time() - t_score
    val_cand_rec_adaptive = val_cand_recall_hits / val_cand_recall_total if val_cand_recall_total else 0.0
    val_cand_rec_fixed = val_fixed_cand_recall_hits / val_cand_recall_total if val_cand_recall_total else 0.0
    val_avg_cands = val_total_cands / max(len(val_s1), 1)
    val_fixed_avg = val_total_fixed_cands / max(len(val_s1), 1)

    print(f"    Scoring completed in {dt_score:.1f}s ({len(val_s1)/dt_score:.0f} S1/s).")

    # ------------------------------------------------------------------
    # Policy Evaluation Helper
    # ------------------------------------------------------------------
    def evaluate_predictions(predict_fn) -> dict:
        f05_scores, prec_scores, rec_scores = [], [], []
        total_links = 0
        singleton_count = 0

        for s1 in val_s1:
            sid = s1.entity_id
            truth = val_truth.get(sid, set())
            pred = predict_fn(s1, val_cand_predictions.get(sid, []))

            f05 = f05_for_entity(truth, pred)
            f05_scores.append(f05)
            total_links += len(pred)
            if not pred:
                singleton_count += 1

            if truth:
                tp = len(truth & pred)
                prec_scores.append(tp / len(pred) if pred else 0.0)
                rec_scores.append(tp / len(truth))

        return {
            "macro_f05": float(np.mean(f05_scores)) if f05_scores else 0.0,
            "mean_precision": float(np.mean(prec_scores)) if prec_scores else 0.0,
            "mean_recall": float(np.mean(rec_scores)) if rec_scores else 0.0,
            "avg_links": total_links / max(len(val_s1), 1),
            "singletons": singleton_count,
        }

    # ------------------------------------------------------------------
    # A. GLOBAL FLAT THRESHOLD GRID SEARCH
    # ------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("A. GLOBAL FLAT THRESHOLD GRID SEARCH (Baseline)")
    print("-" * 75)
    best_flat_th = OPTIMAL_THRESHOLD
    best_flat_metrics = {"macro_f05": -1.0}

    for th in np.arange(0.40, 0.92, 0.05):
        m = evaluate_predictions(lambda s1, cands, t=th: {c[0] for c in cands if c[1] >= t})
        print(f"  Threshold {th:.2f} | Macro F0.5 = {m['macro_f05']:.6f} | "
              f"Prec = {m['mean_precision']:.4f} | Rec = {m['mean_recall']:.4f} | Avg Links = {m['avg_links']:.2f}")
        if m["macro_f05"] > best_flat_metrics["macro_f05"]:
            best_flat_metrics = m
            best_flat_th = float(th)

    print(f"\n  >> Best Global Flat Threshold: {best_flat_th:.2f} -> Macro F0.5 = {best_flat_metrics['macro_f05']:.6f}")

    # ------------------------------------------------------------------
    # B. DYNAMIC COUNTRY-SEGMENTED CALIBRATION (Gap 12 Open-Set)
    # ------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("B. DYNAMIC COUNTRY-SEGMENTED THRESHOLD CALIBRATION (Open-Set)")
    print("-" * 75)

    tuned_segment_thresholds: dict[str, float] = {}

    for country in val_countries:
        country_s1 = [s for s in val_s1 if s.country == country]
        best_c_th = best_flat_th
        best_c_f05 = -1.0
        best_c_prec = 0.0
        best_c_rec = 0.0

        for th in np.arange(0.45, 0.90, 0.05):
            c_f05s, c_precs, c_recs = [], [], []
            for s1 in country_s1:
                sid = s1.entity_id
                truth = val_truth.get(sid, set())
                cands = val_cand_predictions.get(sid, [])
                pred = {c[0] for c in cands if c[1] >= th}
                c_f05s.append(f05_for_entity(truth, pred))
                if truth:
                    tp = len(truth & pred)
                    c_precs.append(tp / len(pred) if pred else 0.0)
                    c_recs.append(tp / len(truth))

            mean_f = float(np.mean(c_f05s))
            if mean_f > best_c_f05:
                best_c_f05 = mean_f
                best_c_th = float(th)
                best_c_prec = float(np.mean(c_precs)) if c_precs else 0.0
                best_c_rec = float(np.mean(c_recs)) if c_recs else 0.0

        tuned_segment_thresholds[country] = best_c_th
        print(f"  Segment '{country}' ({len(country_s1):,} S1): Calibrated Threshold = {best_c_th:.2f} | "
              f"Macro F0.5 = {best_c_f05:.6f} | Prec = {best_c_prec:.4f} | Rec = {best_c_rec:.4f}")

    # Evaluate dynamic segmented policy across full validation set
    seg_m = evaluate_predictions(
        lambda s1, cands: {c[0] for c in cands if c[1] >= tuned_segment_thresholds.get(s1.country, best_flat_th)}
    )
    print(f"\n  >> Dynamic Segmented Policy: Macro F0.5 = {seg_m['macro_f05']:.6f} | "
          f"Prec = {seg_m['mean_precision']:.4f} | Rec = {seg_m['mean_recall']:.4f}")

    # ------------------------------------------------------------------
    # C. EVIDENCE-AWARE DECISION POLICY TUNING (Gap 10 + 12)
    # ------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("C. EVIDENCE-AWARE DECISION POLICY TUNING (Multi-Attribute)")
    print("-" * 75)

    best_policy_th = best_flat_th
    best_policy_metrics = {"macro_f05": -1.0}

    for base_th in np.arange(0.45, 0.90, 0.05):
        def policy_fn(s1, cands, bth=base_th):
            # Pass segment threshold with fallback to base_th for unseen countries
            th = tuned_segment_thresholds.get(s1.country, bth)
            return evidence_aware_match(s1, cands, th)

        m = evaluate_predictions(policy_fn)
        print(f"  Base Th {base_th:.2f} | Macro F0.5 = {m['macro_f05']:.6f} | "
              f"Prec = {m['mean_precision']:.4f} | Rec = {m['mean_recall']:.4f} | Avg Links = {m['avg_links']:.2f}")
        if m["macro_f05"] > best_policy_metrics["macro_f05"]:
            best_policy_metrics = m
            best_policy_th = float(base_th)

    print(f"\n  >> Best Evidence-Aware Policy: Base Th = {best_policy_th:.2f} -> Macro F0.5 = {best_policy_metrics['macro_f05']:.6f}")

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024 ** 2)
    total_time = time.time() - t0

    # ------------------------------------------------------------------
    # Comprehensive Comparison Summary Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("REPRESENTATIVE TUNING & OPEN-SET VALIDATION REPORT")
    print(f"  Total S1 Entities Analyzed : {len(s1_records):,}")
    print(f"  Held-out Validation Split  : {len(val_s1):,} S1 entities (Disjoint)")
    print(f"  Total Background Targets   : {total_targets_indexed:,}")
    print("-" * 75)
    print("  1. CANDIDATE RETRIEVAL BUDGET (Gap 9):")
    print(f"     Fixed-15 Cap Candidate Recall : {val_cand_rec_fixed:.4f} (avg {val_fixed_avg:.2f} cands/S1)")
    print(f"     Adaptive Cap Candidate Recall : {val_cand_rec_adaptive:.4f} (avg {val_avg_cands:.2f} cands/S1)")
    print(f"     Recall Gain from Adaptive Cap : {val_cand_rec_adaptive - val_cand_rec_fixed:+.4f}")
    print("-" * 75)
    print("  2. OPEN-SET COUNTRY CALIBRATION (Gap 12):")
    for c, th in sorted(tuned_segment_thresholds.items()):
        print(f"     Known Segment '{c:<10}' : Calibrated Threshold = {th:.2f}")
    print(f"     Unseen Country Fallback   : Optimal Threshold    = {best_policy_th:.2f}")
    print("-" * 75)
    print("  3. DECISION POLICIES COMPARISON (Gap 10 & 11):")
    print(f"     Baseline Flat (th={best_flat_th:.2f})       : F0.5 = {best_flat_metrics['macro_f05']:.6f} | "
          f"Prec = {best_flat_metrics['mean_precision']:.4f} | Rec = {best_flat_metrics['mean_recall']:.4f}")
    print(f"     Country-Segmented (Open-Set)    : F0.5 = {seg_m['macro_f05']:.6f} | "
          f"Prec = {seg_m['mean_precision']:.4f} | Rec = {seg_m['mean_recall']:.4f}")
    print(f"     Evidence-Aware Policy           : F0.5 = {best_policy_metrics['macro_f05']:.6f} | "
          f"Prec = {best_policy_metrics['mean_precision']:.4f} | Rec = {best_policy_metrics['mean_recall']:.4f}")
    print("-" * 75)
    gpu_info_str = f" | GPU VRAM: {cuda_status['memory_total_mb']:.0f} MB ({cuda_status['device_name']})" if cuda_status["cuda_available"] else ""
    print(f"  Runtime: {total_time:.1f}s | Peak Memory: {peak_mb:.1f} MB{gpu_info_str}")
    print("=" * 75)

    results = {
        "n_val_s1": len(val_s1),
        "val_candidate_recall_fixed": val_cand_rec_fixed,
        "val_candidate_recall_adaptive": val_cand_rec_adaptive,
        "avg_cands_adaptive": val_avg_cands,
        "best_flat_threshold": best_flat_th,
        "best_flat_metrics": best_flat_metrics,
        "calibrated_segment_thresholds": tuned_segment_thresholds,
        "fallback_threshold": best_policy_th,
        "segmented_policy_metrics": seg_m,
        "evidence_aware_metrics": best_policy_metrics,
        "runtime_s": total_time,
        "peak_memory_mb": peak_mb,
    }

    if output_json_path:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with output_json_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nTuning results exported to {output_json_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune decision policies on representative validation split.")
    parser.add_argument("--data-root", type=Path, default=Path("../../dataset").resolve())
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/matcher_model.joblib").resolve())
    parser.add_argument("--sample-s1", type=int, default=25000, help="S1 records to sample (default: 25000).")
    parser.add_argument("--val-ratio", type=float, default=0.25, help="Held-out validation ratio (default: 0.25).")
    parser.add_argument("--bg-per-country", type=int, default=5000, help="Background records per country (default: 5000).")
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/tuned_policy.json").resolve())
    args = parser.parse_args()

    tune_decision_policies(
        data_root=args.data_root,
        model_path=args.model_path,
        sample_s1=args.sample_s1,
        val_ratio=args.val_ratio,
        bg_per_country=args.bg_per_country,
        output_json_path=args.output_json,
    )


if __name__ == "__main__":
    main()
