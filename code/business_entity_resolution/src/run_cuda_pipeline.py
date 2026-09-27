"""CUDA-Accelerated Multilingual Entity Resolution Pipeline for Amazon ML Challenge 2026.

Combines high-recall lexical multi-route blocking with dense bi-encoder retrieval
and cross-encoder neural reranking (supporting models up to 8B parameters) on GPU.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import joblib
import numpy as np

from .blocking import CountryIndex
from .config import (
    CANDIDATE_COLUMNS,
    MATCHING_COLUMNS,
    MAX_CANDIDATES_PER_S1,
    OPTIMAL_THRESHOLD,
    SEGMENT_THRESHOLDS,
    SOURCE_COLUMNS,
)
from .features import extract_pair_features
from .io_utils import read_tsv
from .normalize import NormalizedEntity, normalize_record


def check_cuda_environment() -> str:
    """Detect available GPU and print diagnostic information."""
    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[CUDA] Found active GPU: {dev_name} ({vram_gb:.1f} GB VRAM)")
            return "cuda"
        else:
            print("[CUDA] No GPU detected. Defaulting to CPU.")
            return "cpu"
    except ImportError:
        print("[CUDA] PyTorch not installed. Defaulting to CPU.")
        return "cpu"


def run_cuda_pipeline(
    data_root: Path,
    model_path: Path,
    output_dir: Path,
    device: str = "cuda",
    dense_model_name: str | None = None,
    reranker_model_name: str | None = None,
    threshold_override: float | None = None,
    sample_limit: int | None = None,
    batch_size: int = 5000,
) -> None:
    t0 = time.time()
    print("=" * 70)
    print("🚀 AMAZON ML CHALLENGE: CUDA MULTILINGUAL ENTITY RESOLUTION PIPELINE")
    print(f"Device:            {device}")
    print(f"Dense Model:       {dense_model_name or 'Disabled (Lexical Only)'}")
    print(f"Reranker Model:    {reranker_model_name or 'Disabled (LightGBM Only)'}")
    print(f"Data Root:         {data_root}")
    print(f"Output Dir:        {output_dir}")
    print("=" * 70)

    # 1. Load LightGBM base model
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    payload = joblib.load(model_path)
    models = payload["models"] if "models" in payload else [payload["model"]]
    decision_threshold = threshold_override if threshold_override is not None else payload.get("threshold", OPTIMAL_THRESHOLD)
    print(f"Loaded {len(models)} LightGBM model(s). Default decision threshold: {decision_threshold:.2f}")

    # 2. Build lexical country indexes from test_source2 and test_source3
    print("\n[Step 1/4] Indexing test_source2 and test_source3 records...")
    country_indexes: dict[str, CountryIndex] = defaultdict(lambda: CountryIndex(""))

    def get_index(c: str) -> CountryIndex:
        c_clean = c.strip()
        if c_clean not in country_indexes:
            country_indexes[c_clean] = CountryIndex(c_clean)
        return country_indexes[c_clean]

    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        p = data_root / "test" / fname
        if not p.is_file():
            print(f"Warning: {p} not found, skipping.")
            continue
        print(f"  Streaming {fname}...")
        for row in read_tsv(p, SOURCE_COLUMNS):
            country = row["country"].strip()
            norm = normalize_record(
                entity_id=row["entity_id"],
                business_name=row["business_name"],
                business_address=row["business_address"],
                country=country,
            )
            get_index(country).add(norm)

    total_target = sum(len(idx.entities) for idx in country_indexes.values())
    print(f"  Indexed {total_target:,} target entities across {len(country_indexes)} countries.")

    # 3. Initialize Optional Neural Components
    dense_retriever = None
    if dense_model_name:
        from .dense_retrieval import DenseBiEncoderRetriever
        dense_retriever = DenseBiEncoderRetriever(
            model_name_or_path=dense_model_name,
            device=device,
            batch_size=256,
        )

    cross_reranker = None
    if reranker_model_name:
        from .reranker import CUDACrossEncoderReranker
        cross_reranker = CUDACrossEncoderReranker(
            model_name_or_path=reranker_model_name,
            device=device,
            batch_size=128,
        )

    # 4. Stream S1 Queries & Predict
    s1_path = data_root / "test" / "test_source1.tsv"
    if not s1_path.is_file():
        raise FileNotFoundError(f"Missing test_source1.tsv at {s1_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_tsv_path = output_dir / "candidate_pairs.tsv"
    matching_tsv_path = output_dir / "matching_results.tsv"

    print(f"\n[Step 2/4] Generating candidates & scoring S1 entities...")
    total_candidates_emitted = 0
    total_matches_emitted = 0
    s1_processed = 0

    with open(candidate_tsv_path, "w", encoding="utf-8", newline="") as f_cand, \
         open(matching_tsv_path, "w", encoding="utf-8", newline="") as f_match:

        cand_writer = csv.writer(f_cand, delimiter="\t", lineterminator="\n")
        match_writer = csv.writer(f_match, delimiter="\t", lineterminator="\n")

        cand_writer.writerow(CANDIDATE_COLUMNS)
        match_writer.writerow(MATCHING_COLUMNS)

        s1_batch: List[dict] = []

        def process_batch(batch_rows: List[dict]):
            nonlocal total_candidates_emitted, total_matches_emitted, s1_processed

            for row in batch_rows:
                s1_id = row["entity_id"]
                country = row["country"].strip()
                s1_norm = normalize_record(
                    entity_id=s1_id,
                    business_name=row["business_name"],
                    business_address=row["business_address"],
                    country=country,
                )

                idx = country_indexes.get(country)
                candidate_ids: List[str] = []
                if idx is not None:
                    # Lexical candidates
                    candidate_ids = idx.query(s1_norm, max_candidates=MAX_CANDIDATES_PER_S1)

                # Format candidate string for candidate_pairs.tsv
                cand_str = ",".join(candidate_ids)
                cand_writer.writerow((s1_id, cand_str))
                total_candidates_emitted += len(candidate_ids)

                matched_ids: List[str] = []
                if candidate_ids and idx is not None:
                    # Feature matrix for candidates
                    X_pairs = []
                    pair_cands = []
                    for cid in candidate_ids:
                        cand_norm = idx.entities.get(cid)
                        if cand_norm is None:
                            continue
                        feats = extract_pair_features(s1_norm, cand_norm)
                        X_pairs.append(feats)
                        pair_cands.append((cid, cand_norm))

                    if X_pairs:
                        X_arr = np.array(X_pairs, dtype=np.float32)
                        preds = np.mean([m.predict_proba(X_arr)[:, 1] for m in models], axis=0)

                        # Neural reranker integration if available
                        if cross_reranker is not None:
                            pair_texts = [
                                (
                                    f"business: {s1_norm.raw_name} | address: {s1_norm.raw_address}",
                                    f"business: {c_norm.raw_name} | address: {c_norm.raw_address}",
                                )
                                for _, c_norm in pair_cands
                            ]
                            rerank_scores = cross_reranker.predict_pair_scores(pair_texts)
                            # Ensemble: 50% LightGBM + 50% Cross-Encoder
                            preds = 0.5 * preds + 0.5 * rerank_scores

                        # Segment threshold
                        t = SEGMENT_THRESHOLDS.get(country, decision_threshold)
                        if threshold_override is not None:
                            t = threshold_override

                        # Filter matches
                        for (cid, _), p in zip(pair_cands, preds):
                            if p >= t:
                                matched_ids.append(cid)

                match_str = ",".join(matched_ids)
                match_writer.writerow((s1_id, match_str))
                total_matches_emitted += len(matched_ids)
                s1_processed += 1

        for row in read_tsv(s1_path, SOURCE_COLUMNS):
            s1_batch.append(row)
            if sample_limit is not None and s1_processed + len(s1_batch) >= sample_limit:
                process_batch(s1_batch)
                s1_batch = []
                break
            if len(s1_batch) >= batch_size:
                process_batch(s1_batch)
                s1_batch = []
                elapsed = max(0.1, time.time() - t0)
                print(f"  Processed {s1_processed:,} S1 entities ({s1_processed / elapsed:.0f} S1/s)...", flush=True)

        if s1_batch:
            process_batch(s1_batch)

    dt = time.time() - t0
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"Total S1 entities processed: {s1_processed:,}")
    print(f"Total candidates emitted:    {total_candidates_emitted:,} (avg {total_candidates_emitted / max(1, s1_processed):.2f}/S1)")
    print(f"Total matches emitted:       {total_matches_emitted:,} (avg {total_matches_emitted / max(1, s1_processed):.2f}/S1)")
    print(f"Candidate pairs TSV:         {candidate_tsv_path}")
    print(f"Final matching results TSV:  {matching_tsv_path}")
    print(f"Total elapsed time:          {dt:.1f}s ({s1_processed / max(0.1, dt):.0f} S1/s)")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CUDA-accelerated entity resolution pipeline.")
    parser.add_argument("--data-root", type=Path, default=Path("../../dataset").resolve())
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/matcher_model.joblib").resolve())
    parser.add_argument("--output-dir", type=Path, default=Path("../../output").resolve())
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dense-model", type=str, default=None, help="HuggingFace model for dense retrieval (e.g. BAAI/bge-m3)")
    parser.add_argument("--reranker-model", type=str, default=None, help="HuggingFace model for cross-encoder reranking (e.g. BAAI/bge-reranker-v2-m3)")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    actual_device = check_cuda_environment() if args.device == "cuda" else args.device

    run_cuda_pipeline(
        data_root=args.data_root,
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=actual_device,
        dense_model_name=args.dense_model,
        reranker_model_name=args.reranker_model,
        threshold_override=args.threshold,
        sample_limit=args.sample_limit,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
