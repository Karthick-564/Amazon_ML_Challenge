"""Hybrid Country Index combining Lexical Inverted Indices with GPU-Accelerated Dense BGE-M3 Bi-Encoder Retrieval."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from typing import Optional

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

from .blocking import extract_prefix_keys
from .normalize import NormalizedEntity

# Ensure cache is strictly inside workspace
os.environ["HF_HOME"] = r"E:\ML_challange\.cache\huggingface"


_SHARED_TOKENIZER = None
_SHARED_DENSE_MODEL = None


def get_shared_dense_model():
    global _SHARED_TOKENIZER, _SHARED_DENSE_MODEL
    if _SHARED_DENSE_MODEL is None and torch.cuda.is_available():
        model_name = "BAAI/bge-m3"
        _SHARED_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _SHARED_DENSE_MODEL = AutoModel.from_pretrained(model_name, dtype=torch.float16).cuda()
        _SHARED_DENSE_MODEL.eval()
    return _SHARED_TOKENIZER, _SHARED_DENSE_MODEL


class HybridCountryIndex:
    """High-recall hybrid candidate retrieval index combining Lexical rules and Dense BGE-M3 embeddings on GPU."""

    def __init__(self, country: str, use_gpu_dense: bool = True, batch_size: int = 256):
        self.country = country
        self.use_gpu_dense = use_gpu_dense and torch.cuda.is_available()
        self.batch_size = batch_size

        # Lexical Indices
        self.name_index: dict[str, list[str]] = defaultdict(list)
        self.prefix_index: dict[str, list[str]] = defaultdict(list)
        self.token_index: dict[str, list[str]] = defaultdict(list)
        self.postal_prefix_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.addr_num_token_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.token_doc_counts: Counter[str] = Counter()
        self.entity_store: dict[str, NormalizedEntity] = {}

        # Dense Storage
        self.target_eids: list[str] = []
        self.target_texts: list[str] = []
        self.target_embeddings: Optional[torch.Tensor] = None

        # GPU Model
        if self.use_gpu_dense:
            self.tokenizer, self.dense_model = get_shared_dense_model()
        else:
            self.tokenizer = None
            self.dense_model = None

    def _init_dense_model(self) -> None:
        if self.use_gpu_dense and self.dense_model is None:
            self.tokenizer, self.dense_model = get_shared_dense_model()

    def add_target(self, target: NormalizedEntity) -> None:
        """Register a target entity (S2 or S3) in both lexical and dense stores."""
        eid = target.entity_id
        self.entity_store[eid] = target
        self.target_eids.append(eid)

        # Dense text representation: Name | Address | Country
        rep = f"{target.clean_name} | {target.clean_address} | {target.country}"
        self.target_texts.append(rep)

        # Lexical Route 1: Exact root name
        if len(target.root_name) >= 3:
            self.name_index[target.root_name].append(eid)

        # Lexical Route 2: Prefix bigram keys
        for pkey in extract_prefix_keys(target.root_name):
            if len(self.prefix_index[pkey]) < 80:
                self.prefix_index[pkey].append(eid)

        # Lexical Route 3: Token frequencies for rare token detection
        tokens = set(target.root_name.split())
        for tok in tokens:
            if len(tok) >= 3:
                self.token_doc_counts[tok] += 1

        # Lexical Route 4: Postal code + first token prefix
        if target.postal_code and tokens:
            first_tok = target.root_name.split()[0][:4]
            if len(first_tok) >= 3:
                self.postal_prefix_index[(target.postal_code, first_tok)].append(eid)

        # Lexical Route 5: Address number + token indexing
        addr_words = [w for w in target.clean_address.split() if len(w) >= 4]
        for num in target.address_numbers[:2]:
            for w in addr_words[:3]:
                if len(self.addr_num_token_index[(num, w)]) < 40:
                    self.addr_num_token_index[(num, w)].append(eid)

    def finalize_index(self, max_token_freq: int = 1500) -> None:
        """Build the inverted index for distinctive tokens and encode all dense embeddings on RTX 4090."""
        # 1. Finalize Lexical Tokens
        for eid, target in self.entity_store.items():
            tokens = set(target.root_name.split())
            for tok in tokens:
                if len(tok) >= 3 and self.token_doc_counts[tok] <= max_token_freq:
                    if len(self.token_index[tok]) < 50:
                        self.token_index[tok].append(eid)

        # 2. Encode Dense Embeddings in Batches on GPU (with disk cache)
        if self.use_gpu_dense and self.target_texts:
            from pathlib import Path
            cache_dir = Path("E:/ML_challange/.cache/embeddings")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"targets_{self.country}_{len(self.target_texts)}.pt"

            if cache_file.exists():
                print(f"    Loading cached embeddings from {cache_file}...", flush=True)
                self.target_embeddings = torch.load(cache_file, map_location="cuda")
            else:
                self._init_dense_model()
                emb_chunks = []
                n_texts = len(self.target_texts)

                with torch.no_grad():
                    for i in range(0, n_texts, self.batch_size):
                        batch_texts = self.target_texts[i : i + self.batch_size]
                        inputs = self.tokenizer(
                            batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=64,
                            return_tensors="pt",
                        ).to("cuda")
                        outputs = self.dense_model(**inputs)
                        embs = outputs.last_hidden_state[:, 0]  # CLS token
                        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                        emb_chunks.append(embs)

                self.target_embeddings = torch.cat(emb_chunks, dim=0)
                try:
                    torch.save(self.target_embeddings, cache_file)
                except Exception as e:
                    print(f"    Could not save cache: {e}", flush=True)

    def retrieve_lexical_candidates(
        self, s1: NormalizedEntity, max_cands: int = 35
    ) -> list[tuple[str, float]]:
        """Retrieve candidates using the multi-route lexical inverted index."""
        scores: dict[str, float] = defaultdict(float)

        # Exact root name (weight 5.0)
        if len(s1.root_name) >= 3:
            for tid in self.name_index.get(s1.root_name, ()):
                scores[tid] += 5.0

        # Prefix bigrams (weight 3.5)
        for pkey in extract_prefix_keys(s1.root_name):
            for tid in self.prefix_index.get(pkey, ())[:40]:
                scores[tid] += 3.5

        # Distinctive tokens (weight 2.5 / 1.5)
        tokens = set(s1.root_name.split())
        for tok in tokens:
            if len(tok) >= 3:
                df = self.token_doc_counts.get(tok, 0)
                if 0 < df <= 1500:
                    w = 2.5 if df < 100 else 1.5
                    for tid in self.token_index.get(tok, ())[:35]:
                        scores[tid] += w

        # Postal code + prefix (weight 4.0)
        if s1.postal_code and tokens:
            first_tok = s1.root_name.split()[0][:4]
            if len(first_tok) >= 3:
                for tid in self.postal_prefix_index.get((s1.postal_code, first_tok), ())[:30]:
                    scores[tid] += 4.0

        # Address number + locality (weight 2.5)
        addr_words = [w for w in s1.clean_address.split() if len(w) >= 4]
        for num in s1.address_numbers[:2]:
            for w in addr_words[:3]:
                for tid in self.addr_num_token_index.get((num, w), ())[:25]:
                    scores[tid] += 2.5

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:max_cands]
        return [(tid, sc) for tid, sc in ranked if sc >= 1.5]

    def retrieve_hybrid_batch(
        self, s1_batch: list[NormalizedEntity], max_candidates: int = 35
    ) -> list[list[tuple[str, float, float]]]:
        """Retrieve hybrid candidates (tid, rrf_score, dense_sim) for an S1 batch via RRF."""
        batch_results = []
        n_queries = len(s1_batch)

        # 1. Lexical Retrieval
        lexical_cands_list = [
            self.retrieve_lexical_candidates(s1, max_cands=max_candidates) for s1 in s1_batch
        ]

        # 2. Dense GPU Batch Retrieval (if available)
        dense_results: list[dict[str, tuple[int, float]]] = [{} for _ in range(n_queries)]

        if self.use_gpu_dense and self.target_embeddings is not None and n_queries > 0:
            query_texts = [f"{s.clean_name} | {s.clean_address} | {s.country}" for s in s1_batch]
            with torch.no_grad():
                inputs = self.tokenizer(
                    query_texts,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                ).to("cuda")
                q_embs = self.dense_model(**inputs).last_hidden_state[:, 0]
                q_embs = torch.nn.functional.normalize(q_embs, p=2, dim=1)

                # Cosine similarity matrix: (n_queries, n_targets)
                sim_matrix = torch.matmul(q_embs, self.target_embeddings.T)
                top_k = min(max_candidates, self.target_embeddings.size(0))
                top_scores, top_indices = torch.topk(sim_matrix, k=top_k, dim=1)

                top_scores_cpu = top_scores.cpu().numpy()
                top_indices_cpu = top_indices.cpu().numpy()

                for qi in range(n_queries):
                    for rank, (tid_idx, score) in enumerate(
                        zip(top_indices_cpu[qi], top_scores_cpu[qi])
                    ):
                        tid = self.target_eids[tid_idx]
                        dense_results[qi][tid] = (rank, float(score))

        # 3. Reciprocal Rank Fusion (RRF) Merge
        for qi, s1 in enumerate(s1_batch):
            lex_cands = lexical_cands_list[qi]
            dense_map = dense_results[qi]

            rrf_scores: dict[str, float] = defaultdict(float)
            dense_sims: dict[str, float] = defaultdict(float)

            # RRF from Lexical
            for rank, (tid, _) in enumerate(lex_cands):
                rrf_scores[tid] += 1.0 / (60.0 + rank)

            # RRF from Dense
            for tid, (rank, score) in dense_map.items():
                rrf_scores[tid] += 1.0 / (60.0 + rank)
                dense_sims[tid] = score

            # Sort descending by RRF score
            ranked_merged = sorted(rrf_scores.items(), key=lambda x: -x[1])[:max_candidates]
            merged_with_meta = [
                (tid, rrf_sc, dense_sims.get(tid, 0.0)) for tid, rrf_sc in ranked_merged
            ]
            batch_results.append(merged_with_meta)

        return batch_results
