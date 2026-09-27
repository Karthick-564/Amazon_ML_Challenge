"""Full Hybrid Multimodal Inference Engine combining BGE-M3 Dense Retrieval,
Lexical Inverted Index, LightGBM + CatBoost Meta-Learner, and Global Graph Consistency.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from .config import (
    CANDIDATE_COLUMNS,
    MATCHING_COLUMNS,
    OPTIMAL_THRESHOLD,
    SEGMENT_THRESHOLDS,
    SOURCE_COLUMNS,
)
from .ensemble import CalibratedMetaEnsemble
from .graph_matching import resolve_candidate_graph_consistency
from .hybrid_blocker import HybridCountryIndex
from .io_utils import read_tsv
from .meta_features import extract_meta_pair_features
from .normalize import NormalizedEntity, normalize_record


def run_hybrid_inference(
    data_root: Path,
    model_path: Path,
    output_dir: Path,
    batch_size: int = 2000,
    sample_limit: int | None = None,
    max_candidates: int = 35,
) -> None:
    t0 = time.time()
    print("=" * 70)
    print("STARTING HYBRID MULTIMODAL TEST INFERENCE PIPELINE")
    print(f"Data root:       {data_root}")
    print(f"Model path:      {model_path}")
    print(f"Output dir:      {output_dir}")
    print(f"Batch size:      {batch_size}")
    print(f"Max candidates:  {max_candidates}")
    print("=" * 70)

    # 1. Load trained Meta-Learner Model Artifact
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {model_path}.")
    payload = joblib.load(model_path)
    model = payload["model"]
    decision_threshold = payload.get("threshold", OPTIMAL_THRESHOLD)
    print(f"Loaded Meta-Learner model. Default decision threshold: {decision_threshold:.2f}")

    # 2. Build Target Indexes from test_source2 and test_source3
    print("\nStreaming test_source2.tsv and test_source3.tsv to build Hybrid Country Indexes...")
    country_indexes: dict[str, HybridCountryIndex] = {}

    def get_index(country_str: str) -> HybridCountryIndex:
        c = country_str.strip()
        if c not in country_indexes:
            country_indexes[c] = HybridCountryIndex(c, use_gpu_dense=True)
        return country_indexes[c]

    for src_filename in ("test_source2.tsv", "test_source3.tsv"):
        p = data_root / "test" / src_filename
        if not p.is_file():
            raise FileNotFoundError(f"Test file not found: {p}")
        t_src = time.time()
        count = 0
        for row in read_tsv(p, SOURCE_COLUMNS):
            target_rec = normalize_record(
                row["entity_id"], row["business_name"], row["business_address"], row["country"]
            )
            get_index(target_rec.country).add_target(target_rec)
            count += 1
            if sample_limit and count >= sample_limit:
                break
        print(f"  Indexed {count:,} records from {src_filename} in {time.time()-t_src:.1f}s.")

    print("\nFinalizing inverted indices & encoding dense target vectors on RTX 4090...")
    t_fin = time.time()
    for country, idx in country_indexes.items():
        print(f"  Encoding {country} partition ({len(idx.entity_store):,} records)...")
        idx.finalize_index()
    print(f"All target indices ready in {time.time()-t_fin:.1f}s.")

    # 3. Stream test_source1 and perform batched hybrid retrieval + scoring + graph resolution
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

        cand_writer.writerow(CANDIDATE_COLUMNS)
        match_writer.writerow(MATCHING_COLUMNS)

        s1_batch: list[NormalizedEntity] = []

        def process_batch(batch: list[NormalizedEntity]) -> None:
            nonlocal total_candidates_emitted, total_matches_emitted

            # Group batch by country
            c_groups = defaultdict(list)
            for s1 in batch:
                c_groups[s1.country].append(s1)

            batch_cand_ids: dict[str, list[str]] = {}
            batch_pairs_feats: list[list[float]] = []
            pair_meta: list[tuple[str, str, str]] = []  # (sid, tid, country)

            for country, s1_group in c_groups.items():
                idx = country_indexes.get(country)
                if not idx:
                    for s1 in s1_group:
                        batch_cand_ids[s1.entity_id] = []
                    continue

                # Hybrid Batch Retrieval on RTX 4090
                hybrid_batch = idx.retrieve_hybrid_batch(s1_group, max_candidates=max_candidates)

                for s1_rec, cands in zip(s1_group, hybrid_batch):
                    sid = s1_rec.entity_id
                    c_ids = [tid for tid, _, _ in cands]
                    batch_cand_ids[sid] = sorted(set(c_ids))

                    for tid, rrf_sc, d_sim in cands:
                        cand_rec = idx.entity_store.get(tid)
                        if not cand_rec:
                            continue
                        feats = extract_meta_pair_features(s1_rec, cand_rec, rrf_score=rrf_sc, dense_sim=d_sim)
                        batch_pairs_feats.append(feats)
                        pair_meta.append((sid, tid, country))

            # Batch model inference
            s1_cands_scored: dict[str, list[tuple[str, float]]] = defaultdict(list)
            if batch_pairs_feats:
                X_batch = np.array(batch_pairs_feats, dtype=np.float32)
                probs = model.predict_proba(X_batch)[:, 1]
                for (sid, tid, _), prob in zip(pair_meta, probs):
                    s1_cands_scored[sid].append((tid, float(prob)))

            # Decision policy & Graph consistency resolution
            batch_final_matches: dict[str, set[str]] = {}

            for s1_rec in batch:
                sid = s1_rec.entity_id
                cands_p = s1_cands_scored.get(sid, [])
                if not cands_p:
                    batch_final_matches[sid] = set()
                    continue

                cands_p.sort(key=lambda x: -x[1])
                top_tid, top_prob = cands_p[0]

                base_th = decision_threshold
                if s1_rec.country == "India":
                    seg_th = max(0.65, base_th - 0.05)
                else:
                    seg_th = base_th

                # Singleton protection gate: if top candidate is weak, emit singleton
                if top_prob < (seg_th - 0.10):
                    batch_final_matches[sid] = set()
                    continue

                # Preliminary candidate selection
                preliminary_matches: list[tuple[str, float]] = []
                for tid, prob in cands_p:
                    if prob >= seg_th and prob >= (0.75 * top_prob):
                        preliminary_matches.append((tid, prob))
                    elif prob >= 0.85:
                        preliminary_matches.append((tid, prob))

                # Global Graph Disjoint-Set & Structural Consistency Resolution
                idx = country_indexes.get(s1_rec.country)
                entity_store = idx.entity_store if idx else {}
                consistent_matches = resolve_candidate_graph_consistency(
                    s1_rec, preliminary_matches, entity_store
                )
                batch_final_matches[sid] = consistent_matches

            # Write in exact order
            for s1_rec in batch:
                sid = s1_rec.entity_id
                c_list = batch_cand_ids.get(sid, [])
                m_list = sorted(batch_final_matches.get(sid, set()))

                cand_writer.writerow((sid, ",".join(c_list)))
                match_writer.writerow((sid, ",".join(m_list)))

                total_candidates_emitted += len(c_list)
                total_matches_emitted += len(m_list)

        # Stream test_source1
        for row in read_tsv(test_s1_path, SOURCE_COLUMNS):
            s1_processed += 1
            s1_batch.append(
                normalize_record(row["entity_id"], row["business_name"], row["business_address"], row["country"])
            )

            if len(s1_batch) >= batch_size:
                process_batch(s1_batch)
                s1_batch = []
                if s1_processed % 20000 == 0:
                    elapsed = time.time() - t0
                    print(f"  Processed {s1_processed:,} Source-1 entities ({elapsed:.1f}s, {s1_processed/elapsed:.0f} S1/s)...")

            if sample_limit and s1_processed >= sample_limit:
                break

        if s1_batch:
            process_batch(s1_batch)

    dt = time.time() - t0
    print("\n" + "=" * 70)
    print("HYBRID INFERENCE COMPLETE:")
    print(f"Processed S1 entities:       {s1_processed:,}")
    print(f"Total candidates emitted:    {total_candidates_emitted:,} (avg {total_candidates_emitted/s1_processed:.2f}/S1)")
    print(f"Total matches emitted:       {total_matches_emitted:,} (avg {total_matches_emitted/s1_processed:.2f}/S1)")
    print(f"Wrote candidates to:         {candidate_tsv_path}")
    print(f"Wrote final matches to:      {matching_tsv_path}")
    print(f"Total elapsed time:          {dt:.1f}s ({s1_processed/dt:.0f} S1/s)")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hybrid Multimodal Inference.")
    parser.add_argument("--data-root", type=Path, default=Path("../../dataset").resolve())
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/matcher_meta_ensemble.joblib").resolve())
    parser.add_argument("--output-dir", type=Path, default=Path("../../output").resolve())
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=35)
    args = parser.parse_args()

    run_hybrid_inference(
        data_root=args.data_root,
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        sample_limit=args.sample_limit,
        max_candidates=args.max_candidates,
    )


if __name__ == "__main__":
    main()
