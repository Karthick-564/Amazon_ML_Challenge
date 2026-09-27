"""CUDA-accelerated dense semantic retrieval module for multilingual entity resolution.

Supports any HuggingFace or Sentence-Transformers embedding model up to 8B parameters
(e.g., BAAI/bge-m3, intfloat/multilingual-e5-large, Alibaba-NLP/gte-Qwen2-7B-instruct).
"""

from __future__ import annotations

import gc
from typing import Dict, List, Tuple
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def format_entity_text(name: str, address: str, country: str) -> str:
    """Format structured record into canonical representation for embedding."""
    name_clean = (name or "").strip()
    addr_clean = (address or "").strip()
    country_clean = (country or "").strip()
    return f"business: {name_clean} | address: {addr_clean} | country: {country_clean}"


class DenseBiEncoderRetriever:
    """CUDA-accelerated dense retriever with country-partitioned vector indexing."""

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-m3",
        device: str | None = None,
        batch_size: int = 256,
        use_fp16: bool = True,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch and sentence-transformers are required for dense retrieval.")

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[DenseRetriever] Initializing {model_name_or_path} on device: {self.device} (FP16={use_fp16})")
        self.batch_size = batch_size
        self.use_fp16 = use_fp16

        # Load embedding model
        model_kwargs = {}
        if self.device == "cuda" and use_fp16:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = SentenceTransformer(
            model_name_or_path,
            device=self.device,
            model_kwargs=model_kwargs if model_kwargs else None,
        )
        self.model.eval()

    def encode_texts(self, texts: List[str], show_progress: bool = True) -> torch.Tensor:
        """Encode a list of text strings into normalized embedding tensors."""
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=True,
            )
        return embeddings

    def retrieve_candidates_by_country(
        self,
        s1_ids: List[str],
        s1_texts: List[str],
        target_ids: List[str],
        target_texts: List[str],
        top_k: int = 15,
        min_similarity: float = 0.55,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Perform GPU batched matrix multiplication to retrieve top-k candidates for each S1 entity.

        Returns mapping: s1_id -> list of (target_id, cosine_similarity)
        """
        if not s1_ids or not target_ids:
            return {}

        print(f"[DenseRetriever] Encoding {len(target_ids)} target records (S2+S3)...")
        target_embeddings = self.encode_texts(target_texts, show_progress=True)

        print(f"[DenseRetriever] Encoding {len(s1_ids)} query records (S1)...")
        results: Dict[str, List[Tuple[str, float]]] = {}

        # Query in chunks to avoid GPU OOM on large similarity matrices
        chunk_size = 2048
        for i in range(0, len(s1_ids), chunk_size):
            s1_chunk_ids = s1_ids[i : i + chunk_size]
            s1_chunk_texts = s1_texts[i : i + chunk_size]

            s1_chunk_emb = self.encode_texts(s1_chunk_texts, show_progress=False)

            # Cosine similarity matrix: (chunk_size, num_targets)
            sim_matrix = torch.matmul(s1_chunk_emb, target_embeddings.T)

            # Retrieve top_k on GPU
            k = min(top_k, sim_matrix.shape[1])
            top_vals, top_indices = torch.topk(sim_matrix, k=k, dim=1)

            top_vals_cpu = top_vals.cpu().numpy()
            top_indices_cpu = top_indices.cpu().numpy()

            for idx, s1_id in enumerate(s1_chunk_ids):
                candidates = []
                for val, tidx in zip(top_vals_cpu[idx], top_indices_cpu[idx]):
                    if val >= min_similarity:
                        candidates.append((target_ids[tidx], float(val)))
                results[s1_id] = candidates

            del s1_chunk_emb, sim_matrix, top_vals, top_indices
            if self.device == "cuda":
                torch.cuda.empty_cache()

        del target_embeddings
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        return results
