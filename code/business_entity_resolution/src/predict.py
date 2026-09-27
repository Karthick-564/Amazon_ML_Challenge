"""High-performance batched inference engine for entity resolution."""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from .blocking import CountryIndex, CandidateResult
from .config import (
    CANDIDATE_COLUMNS,
    MATCHING_COLUMNS,
    MAX_CANDIDATES_PER_S1,
    OPTIMAL_THRESHOLD,
    SEGMENT_THRESHOLDS,
    SOURCE_COLUMNS,
    get_segment_threshold,
)
from .decision_policy import adaptive_cap, evidence_aware_match
from .features import extract_pair_features
from .io_utils import read_tsv
from .normalize import NormalizedEntity, normalize_record


def run_inference(
    data_root: Path,
    model_path: Path,
    output_dir: Path,
    threshold_override: float | None = None,
    sample_limit: int | None = None,
    batch_size: int = 5000,
) -> None:
    t0 = time.time()
    print("=" * 60)
    print("STARTING OPTIMIZED BATCHED TEST INFERENCE PIPELINE")
    print(f"Data root:   {data_root}")
    print(f"Model path:  {model_path}")
    print(f"Output dir:  {output_dir}")
    print(f"Batch size:  {batch_size}")
    print("=" * 60)

    # 1. Load model artifact
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found at {model_path}. Run training first.")
    payload = joblib.load(model_path)
    model: lgb.LGBMClassifier = payload["model"]
    decision_threshold = threshold_override if threshold_override is not None else payload.get("threshold", OPTIMAL_THRESHOLD)
    print(f"Loaded matcher model. Using decision threshold: {decision_threshold:.2f}")

    # 2. Build target indexes from test_source2 and test_source3
    print("\nStreaming test_source2 and test_source3 to build country indexes...")
    country_indexes: dict[str, CountryIndex] = defaultdict(lambda: CountryIndex(""))

    def get_index(country: str) -> CountryIndex:
        c = country.strip()
        if c not in country_indexes:
            country_indexes[c] = CountryIndex(c)
        return country_indexes[c]

    for src_filename in ("test_source2.tsv", "test_source3.tsv"):
        p = data_root / "test" / src_filename
        if not p.is_file():
            raise FileNotFoundError(f"Test file not found: {p}")
        print(f"  Indexing {src_filename}...")
        t_src = time.time()
        count = 0
        for row in read_tsv(p, SOURCE_COLUMNS):
            target_rec = normalize_record(
                row["entity_id"], row["business_name"], row["business_address"], row["country"]
            )
            idx = get_index(target_rec.country)
            idx.add_target(target_rec)
            count += 1
            if count % 500_000 == 0:
                elapsed = time.time() - t_src
                rate = count / max(elapsed, 0.001)
                print(f"    ... {count:,} records indexed ({elapsed:.0f}s elapsed, {rate:,.0f} rec/s)")
            if sample_limit and count >= sample_limit:
                break
        print(f"  --> Completed {src_filename}: {count:,} records in {time.time()-t_src:.1f}s.")

    print("Finalizing inverted token indices...")
    for idx in country_indexes.values():
        idx.finalize_tokens()
    total_targets = sum(len(idx.entity_store) for idx in country_indexes.values())
    print(f"Total target records indexed: {total_targets} across {len(country_indexes)} countries.")

    # 3. Stream test_source1 and perform batched blocking + scoring
    test_s1_path = data_root / "test" / "test_source1.tsv"
    if not test_s1_path.is_file():
        raise FileNotFoundError(f"Source 1 test file not found: {test_s1_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_tsv_path = output_dir / "candidate_pairs.tsv"
    matching_tsv_path = output_dir / "matching_results.tsv"

    print(f"\nProcessing {test_s1_path} in batches of {batch_size}...")
    s1_processed = 0
    total_candidates_emitted = 0
    total_matches_emitted = 0

    with candidate_tsv_path.open("w", encoding="utf-8", newline="") as f_cand, \
         matching_tsv_path.open("w", encoding="utf-8", newline="") as f_match:

        cand_writer = csv.writer(f_cand, delimiter="\t", lineterminator="\n")
        match_writer = csv.writer(f_match, delimiter="\t", lineterminator="\n")

        # Headers
        cand_writer.writerow(CANDIDATE_COLUMNS)
        match_writer.writerow(MATCHING_COLUMNS)

        # Batch accumulator
        s1_batch: list[NormalizedEntity] = []

        def process_batch(batch: list[NormalizedEntity]) -> None:
            nonlocal total_candidates_emitted, total_matches_emitted

            batch_pairs_feats: list[list[float]] = []
            # Map from pair index to (s1_id, candidate_target_id)
            pair_meta: list[tuple[str, str]] = []
            # Keep candidate IDs per S1
            batch_cand_ids: dict[str, list[str]] = {}

            for s1_rec in batch:
                idx = country_indexes.get(s1_rec.country)
                if not idx:
                    c_lower = s1_rec.country.lower()
                    for k, v in country_indexes.items():
                        if k.lower() == c_lower:
                            idx = v
                            break
                if not idx and len(country_indexes) == 1:
                    idx = next(iter(country_indexes.values()))
                if not idx:
                    batch_cand_ids[s1_rec.entity_id] = []
                    continue

                # Gap 9: adaptive per-S1 candidate budget
                cap = adaptive_cap(s1_rec)
                cands = idx.retrieve_candidates_full(s1_rec, max_candidates=cap)
                if not cands:
                    batch_cand_ids[s1_rec.entity_id] = []
                    continue

                # Sorted unique candidate IDs for candidate_pairs.tsv
                cand_ids = sorted({cr.entity_id for cr in cands})
                batch_cand_ids[s1_rec.entity_id] = cand_ids

                # Context values shared across all candidates of this S1
                n_cands = len(cands)
                max_rs  = cands[0].retrieval_score  # list is sorted descending

                for rank, cr in enumerate(cands, 1):
                    cand_rec = idx.entity_store.get(cr.entity_id)
                    if not cand_rec:
                        continue
                    rs_norm   = cr.retrieval_score / max_rs
                    rs_margin = max_rs - cr.retrieval_score  # 0 for rank-1
                    feats = extract_pair_features(
                        s1_rec, cand_rec, cr.retrieval_score,
                        n_candidates=n_cands,
                        retrieval_rank=rank,
                        retrieval_score_norm=rs_norm,
                        retrieval_margin=rs_margin,
                        n_routes=cr.n_routes,
                    )
                    batch_pairs_feats.append(feats)
                    pair_meta.append((s1_rec.entity_id, cr.entity_id))

            # Batched model inference -> probability scores
            # Store (tid, prob, feats) triples so decision policy can access
            # evidence without re-computation (Gap 10).
            s1_cands_full_scored: dict = defaultdict(list)
            if batch_pairs_feats:
                X_batch = np.array(batch_pairs_feats, dtype=np.float32)
                probs = model.predict_proba(X_batch)[:, 1]
                for (s1_id, tid), prob, feats in zip(pair_meta, probs, batch_pairs_feats):
                    s1_cands_full_scored[s1_id].append((tid, float(prob), feats))

            # Gap 10: evidence-aware decision policy
            batch_matches: dict = defaultdict(set)
            for s1_rec in batch:
                sid = s1_rec.entity_id
                scored = s1_cands_full_scored.get(sid, [])
                if not scored:
                    continue
                seg_th = threshold_override if threshold_override is not None else get_segment_threshold(s1_rec.country)
                batch_matches[sid] = evidence_aware_match(s1_rec, scored, seg_th)

            # Write results in exact S1 order
            for s1_rec in batch:
                sid = s1_rec.entity_id
                c_list = batch_cand_ids.get(sid, [])
                m_list = sorted(batch_matches.get(sid, set()))

                cand_writer.writerow((sid, ",".join(c_list)))
                match_writer.writerow((sid, ",".join(m_list)))

                total_candidates_emitted += len(c_list)
                total_matches_emitted += len(m_list)

        t_inf_start = time.time()
        for row in read_tsv(test_s1_path, SOURCE_COLUMNS):
            s1_processed += 1
            s1_batch.append(
                normalize_record(row["entity_id"], row["business_name"], row["business_address"], row["country"])
            )

            if len(s1_batch) >= batch_size:
                process_batch(s1_batch)
                s1_batch = []
                if s1_processed % 10_000 == 0:
                    elapsed = time.time() - t_inf_start
                    rate = s1_processed / max(elapsed, 0.001)
                    avg_c = total_candidates_emitted / max(s1_processed, 1)
                    avg_m = total_matches_emitted / max(s1_processed, 1)
                    if sample_limit:
                        pct = (s1_processed / sample_limit) * 100
                        eta_s = (sample_limit - s1_processed) / max(rate, 0.001)
                        print(f"  [{s1_processed:,}/{sample_limit:,}] ({pct:.1f}%) | {rate:.0f} S1/s | ETA: {eta_s/60:.1f}m | avg cands: {avg_c:.1f} | matches: {total_matches_emitted:,} ({avg_m:.2f}/S1)")
                    else:
                        print(f"  [{s1_processed:,}] | {rate:.0f} S1/s | elapsed: {elapsed/60:.1f}m | avg cands: {avg_c:.1f} | matches: {total_matches_emitted:,} ({avg_m:.2f}/S1)")

            if sample_limit and s1_processed >= sample_limit:
                break

        # Process any remaining records
        if s1_batch:
            process_batch(s1_batch)

    dt = time.time() - t0
    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY:")
    print(f"Processed S1 entities:       {s1_processed:,}")
    print(f"Total candidates emitted:    {total_candidates_emitted:,} (avg {total_candidates_emitted/s1_processed:.2f}/S1)")
    print(f"Total matches emitted:       {total_matches_emitted:,} (avg {total_matches_emitted/s1_processed:.2f}/S1)")
    print(f"Wrote candidates to:         {candidate_tsv_path}")
    print(f"Wrote final matches to:      {matching_tsv_path}")
    print(f"Total elapsed time:          {dt:.1f}s ({s1_processed/dt:.0f} S1/s)")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run entity resolution candidate generation and matching inference.")
    parser.add_argument("--data-root", type=Path, default=Path("../../dataset").resolve())
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/matcher_model.joblib").resolve())
    parser.add_argument("--output-dir", type=Path, default=Path("../../output").resolve())
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    run_inference(
        data_root=args.data_root,
        model_path=args.model_path,
        output_dir=args.output_dir,
        threshold_override=args.threshold,
        sample_limit=args.sample_limit,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
