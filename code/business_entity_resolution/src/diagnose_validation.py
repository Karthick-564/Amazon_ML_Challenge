"""Comprehensive diagnostic tool to dissect model and blocking performance on validation split."""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from .blocking import CountryIndex
from .config import OPTIMAL_THRESHOLD, SEED, SOURCE_COLUMNS, TRUTH_COLUMNS
from .evaluate import f05_for_entity
from .features import extract_pair_features
from .io_utils import parse_id_list, read_tsv
from .normalize import NormalizedEntity, normalize_record


def is_indic_script(text: str) -> bool:
    """Return True if text contains characters in Indic Unicode blocks."""
    for ch in text:
        code = ord(ch)
        # Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam
        if 0x0900 <= code <= 0x0D7F:
            return True
    return False


def run_diagnostics(
    data_root: Path,
    model_path: Path,
    sample_s1_limit: int = 15000,
    val_ratio: float = 0.333,
    distractor_limit: int = 100000,
) -> None:
    t0 = time.time()
    print("=" * 70)
    print("STARTING COMPREHENSIVE ERROR DIAGNOSTIC PIPELINE")
    print(f"Data root:        {data_root}")
    print(f"Model path:       {model_path}")
    print(f"S1 sample limit:  {sample_s1_limit}")
    print("=" * 70)

    # 1. Load S1 entities
    print(f"\n1. Loading {sample_s1_limit} Source 1 records...")
    s1_records: list[NormalizedEntity] = []
    s1_path = data_root / "train" / "train_source1.tsv"
    for i, row in enumerate(read_tsv(s1_path, SOURCE_COLUMNS)):
        s1_records.append(
            normalize_record(row["entity_id"], row["business_name"], row["business_address"], row["country"])
        )
        if i + 1 >= sample_s1_limit:
            break

    n_val = int(len(s1_records) * val_ratio)
    val_s1 = s1_records[-n_val:]
    val_s1_ids = {r.entity_id for r in val_s1}
    print(f"Validation set size: {len(val_s1)} Source 1 entities.")

    # 2. Load ground truth
    print("\n2. Loading ground truth for validation entities...")
    truth_path = data_root / "train" / "train_ground_truth.tsv"
    val_truth: dict[str, set[str]] = {}
    needed_targets: set[str] = set()

    for row in read_tsv(truth_path, TRUTH_COLUMNS):
        sid = row["source1_entity_id"]
        if sid in val_s1_ids:
            matched = parse_id_list(row["matched_entity_ids"])
            val_truth[sid] = matched
            needed_targets.update(matched)

    # Fill singletons if any missing
    for sid in val_s1_ids:
        if sid not in val_truth:
            val_truth[sid] = set()

    true_singleton_count = sum(1 for sid, t in val_truth.items() if len(t) == 0)
    total_val_links = sum(len(t) for t in val_truth.values())
    print(f"Validation ground truth loaded: {total_val_links} total links.")
    print(f"True singletons: {true_singleton_count} / {len(val_s1)} ({true_singleton_count/len(val_s1)*100:.2f}%)")
    print(f"Average true links per S1: {total_val_links / len(val_s1):.2f}")

    # 3. Stream S2 and S3 target records
    print("\n3. Indexing target records (all true validation targets + background distractors)...")
    country_indexes: dict[str, CountryIndex] = defaultdict(lambda: CountryIndex(""))
    target_raw_names: dict[str, str] = {}

    def get_index(country: str) -> CountryIndex:
        c = country.strip()
        if c not in country_indexes:
            country_indexes[c] = CountryIndex(c)
        return country_indexes[c]

    for src_filename in ("train_source2.tsv", "train_source3.tsv"):
        p = data_root / "train" / src_filename
        distractor_count = 0
        found_needed = 0
        is_s2 = "source2" in src_filename
        needed_for_file = {t for t in needed_targets if (t.startswith("S2-") if is_s2 else t.startswith("S3-"))}

        print(f"  Streaming {src_filename} (needs {len(needed_for_file)} true targets)...")
        for row in read_tsv(p, SOURCE_COLUMNS):
            eid = row["entity_id"]
            is_needed = eid in needed_for_file
            if is_needed or distractor_count < distractor_limit:
                target_rec = normalize_record(eid, row["business_name"], row["business_address"], row["country"])
                target_raw_names[eid] = row["business_name"]
                idx = get_index(target_rec.country)
                idx.add_target(target_rec)
                if is_needed:
                    found_needed += 1
                else:
                    distractor_count += 1
            if found_needed >= len(needed_for_file) and distractor_count >= distractor_limit:
                break
        print(f"    Indexed {found_needed} true targets + {distractor_count} distractors.")

    for idx in country_indexes.values():
        idx.finalize_tokens()
    print(f"Total target corpus indexed: {sum(len(idx.entity_store) for idx in country_indexes.values()):,} records.")

    # 4. Load model
    print(f"\n4. Loading matcher model from {model_path}...")
    payload = joblib.load(model_path)
    model: lgb.LGBMClassifier = payload["model"]
    threshold: float = payload.get("threshold", OPTIMAL_THRESHOLD)
    print(f"Using threshold: {threshold:.2f}")

    # 5. Run Candidate Retrieval & Model Inference
    print("\n5. Running candidate retrieval and inference...")
    val_predictions: dict[str, set[str]] = {}
    val_candidates: dict[str, list[str]] = {}
    s1_scores: list[float] = []

    # Segment tracking
    seg_links_total: Counter[str] = Counter()
    seg_links_in_cands: Counter[str] = Counter()
    seg_links_in_preds: Counter[str] = Counter()

    # Multiplicity tracking
    mult_cap_hit = 0
    mult_cap_total = 0

    for s1 in val_s1:
        sid = s1.entity_id
        true_targets = val_truth[sid]
        num_true = len(true_targets)
        c_country = s1.country

        # Multiplicity bucket
        if num_true == 0:
            m_bucket = "0_singletons"
        elif num_true == 1:
            m_bucket = "1_match"
        elif num_true == 2:
            m_bucket = "2_matches"
        elif 3 <= num_true <= 5:
            m_bucket = "3-5_matches"
        else:
            m_bucket = "6-11_matches"

        idx = country_indexes.get(s1.country)
        if not idx:
            val_candidates[sid] = []
            val_predictions[sid] = set()
            s1_scores.append(f05_for_entity(true_targets, set()))
            continue

        cands = idx.retrieve_candidates(s1, max_candidates=15)
        cand_ids = [tid for tid, _ in cands]
        val_candidates[sid] = cand_ids

        if num_true >= 6:
            mult_cap_total += 1
            if len(cand_ids) >= 15:
                mult_cap_hit += 1

        cand_set = set(cand_ids)

        # Segment-level candidate recall check
        for tid in true_targets:
            seg_links_total["Overall"] += 1
            seg_links_total[f"Country: {c_country}"] += 1
            seg_links_total[f"Multiplicity: {m_bucket}"] += 1
            src_tag = "Source 2" if tid.startswith("S2-") else "Source 3"
            seg_links_total[f"Target: {src_tag}"] += 1

            raw_name = target_raw_names.get(tid, "")
            is_indic = is_indic_script(raw_name)
            script_tag = "Cross-Script (Indic)" if is_indic else "Same-Script (Latin)"
            seg_links_total[f"Script: {script_tag}"] += 1

            if tid in cand_set:
                seg_links_in_cands["Overall"] += 1
                seg_links_in_cands[f"Country: {c_country}"] += 1
                seg_links_in_cands[f"Multiplicity: {m_bucket}"] += 1
                seg_links_in_cands[f"Target: {src_tag}"] += 1
                seg_links_in_cands[f"Script: {script_tag}"] += 1

        # Model pair scoring
        predicted_matches: set[str] = set()
        if cands:
            feats_list = []
            cand_meta = []
            for tid, r_sc in cands:
                cand_rec = idx.entity_store.get(tid)
                if not cand_rec:
                    continue
                feats = extract_pair_features(s1, cand_rec, r_sc)
                feats_list.append(feats)
                cand_meta.append(tid)

            if feats_list:
                probs = model.predict_proba(np.array(feats_list, dtype=np.float32))[:, 1]
                for tid, prob in zip(cand_meta, probs):
                    if prob >= threshold:
                        predicted_matches.add(tid)

        val_predictions[sid] = predicted_matches
        score = f05_for_entity(true_targets, predicted_matches)
        s1_scores.append(score)

        # Track final matches per segment
        for tid in true_targets:
            if tid in predicted_matches:
                seg_links_in_preds["Overall"] += 1
                seg_links_in_preds[f"Country: {c_country}"] += 1
                seg_links_in_preds[f"Multiplicity: {m_bucket}"] += 1
                src_tag = "Source 2" if tid.startswith("S2-") else "Source 3"
                seg_links_in_preds[f"Target: {src_tag}"] += 1
                raw_name = target_raw_names.get(tid, "")
                is_indic = is_indic_script(raw_name)
                script_tag = "Cross-Script (Indic)" if is_indic else "Same-Script (Latin)"
                seg_links_in_preds[f"Script: {script_tag}"] += 1

    # =========================================================================
    # DIAGNOSTIC RESULTS REPORT
    # =========================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC REPORT SUMMARY")
    print("=" * 70)

    # 1. Overall Metrics
    macro_f05 = float(np.mean(s1_scores))
    print(f"Overall Macro F0.5 on Validation: {macro_f05:.4f}")
    pred_links_total = sum(len(p) for p in val_predictions.values())
    print(f"Average predicted links per S1:   {pred_links_total / len(val_s1):.2f} (Ground truth: {total_val_links / len(val_s1):.2f})")

    # 2. Score Decomposition
    exact_zeros = sum(1 for s in s1_scores if s == 0.0)
    exact_ones = sum(1 for s in s1_scores if s == 1.0)
    in_betweens = len(s1_scores) - exact_zeros - exact_ones

    print("\n--- 1. SCORE DECOMPOSITION (Bimodal vs Partial Errors) ---")
    print(f"Score == 0.0 (Total Failure):  {exact_zeros:5d} / {len(s1_scores)} ({exact_zeros/len(s1_scores)*100:5.2f}%)")
    print(f"Score == 1.0 (Perfect Match):  {exact_ones:5d} / {len(s1_scores)} ({exact_ones/len(s1_scores)*100:5.2f}%)")
    print(f"0.0 < Score < 1.0 (Partial):   {in_betweens:5d} / {len(s1_scores)} ({in_betweens/len(s1_scores)*100:5.2f}%)")

    # 3. Singleton Diagnostics
    pred_singletons = sum(1 for sid, p in val_predictions.items() if len(p) == 0)
    correct_singletons = sum(1 for sid, p in val_predictions.items() if len(p) == 0 and len(val_truth[sid]) == 0)
    false_merges_on_singletons = sum(1 for sid, p in val_predictions.items() if len(p) > 0 and len(val_truth[sid]) == 0)
    false_singletons = sum(1 for sid, p in val_predictions.items() if len(p) == 0 and len(val_truth[sid]) > 0)

    print("\n--- 2. SINGLETON ANALYSIS ---")
    print(f"True singletons in validation:     {true_singleton_count} ({true_singleton_count/len(val_s1)*100:.2f}%)")
    print(f"Predicted singletons:              {pred_singletons} ({pred_singletons/len(val_s1)*100:.2f}%)")
    print(f"Singleton Precision:               {correct_singletons} / {pred_singletons} ({correct_singletons/max(1,pred_singletons)*100:.2f}%)")
    print(f"Singleton Recall:                  {correct_singletons} / {true_singleton_count} ({correct_singletons/max(1,true_singleton_count)*100:.2f}%)")
    print(f"False Merges on True Singletons:   {false_merges_on_singletons} (Each dropped F0.5 directly to 0.0!)")
    print(f"False Singletons (Missed Matches): {false_singletons} (True matches had all candidates rejected)")

    # 4. Segment-Level Candidate Recall vs Final Recall
    print("\n--- 3. CANDIDATE RECALL VS FINAL RECALL BY SEGMENT ---")
    print(f"{'Segment':<32} | {'True Links':<10} | {'Cand Recall':<12} | {'Final Recall':<12} | {'Loss in Classifier':<15}")
    print("-" * 90)

    segments_ordered = (
        ["Overall"]
        + [k for k in seg_links_total if k.startswith("Country:")]
        + [k for k in seg_links_total if k.startswith("Script:")]
        + [k for k in seg_links_total if k.startswith("Target:")]
        + [k for k in seg_links_total if k.startswith("Multiplicity:")]
    )

    for seg in segments_ordered:
        tot = seg_links_total[seg]
        if tot == 0:
            continue
        c_rec = (seg_links_in_cands[seg] / tot) * 100
        f_rec = (seg_links_in_preds[seg] / tot) * 100
        loss = c_rec - f_rec
        print(f"{seg:<32} | {tot:<10} | {c_rec:10.2f}% | {f_rec:10.2f}% | {loss:13.2f}%")

    # 5. High-Multiplicity Candidate Cap Analysis
    print("\n--- 4. HIGH-MULTIPLICITY CANDIDATE CAP ANALYSIS ---")
    print(f"Entities with >= 6 true matches:   {mult_cap_total}")
    if mult_cap_total > 0:
        print(f"Hit candidate cap (15 candidates): {mult_cap_hit} / {mult_cap_total} ({mult_cap_hit/mult_cap_total*100:.2f}%)")

    print("\n" + "=" * 70)
    print(f"Diagnostics completed in {time.time()-t0:.1f}s.")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run detailed validation error diagnostics.")
    parser.add_argument("--data-root", type=Path, default=Path("../../dataset").resolve())
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/matcher_model.joblib").resolve())
    parser.add_argument("--sample-s1", type=int, default=15000)
    parser.add_argument("--distractors", type=int, default=100000)
    args = parser.parse_args()

    run_diagnostics(
        data_root=args.data_root,
        model_path=args.model_path,
        sample_s1_limit=args.sample_s1,
        distractor_limit=args.distractors,
    )


if __name__ == "__main__":
    main()
