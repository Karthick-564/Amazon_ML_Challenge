"""CUDA-accelerated Cross-Encoder reranker module for pairwise verification.

Uses full cross-attention between S1 and counterpart pairs to detect subtle
contradictions and output high-precision match scores, protecting Macro F0.5.
Supports models up to 8B parameters (e.g. BAAI/bge-reranker-v2-m3, Qwen-based rerankers).
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

try:
    import torch
    from sentence_transformers import CrossEncoder
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class CUDACrossEncoderReranker:
    """CUDA-accelerated Cross-Encoder scoring engine."""

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        batch_size: int = 128,
        max_length: int = 256,
        use_fp16: bool = True,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch and sentence-transformers are required for Cross-Encoder reranker.")

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[Reranker] Initializing CrossEncoder {model_name_or_path} on device: {self.device}")
        self.batch_size = batch_size
        self.max_length = max_length

        model_kwargs = {}
        if self.device == "cuda" and use_fp16:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = CrossEncoder(
            model_name_or_path,
            max_length=self.max_length,
            device=self.device,
            automodel_args=model_kwargs if model_kwargs else None,
        )

    def predict_pair_scores(
        self,
        pairs: List[Tuple[str, str]],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Score pairwise tuples (s1_text, candidate_text) and return sigmoid probabilities in [0.0, 1.0]."""
        if not pairs:
            return np.array([], dtype=np.float32)

        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        # Apply sigmoid if logits are unnormalized
        if raw_scores.ndim > 0 and (raw_scores.min() < 0.0 or raw_scores.max() > 1.0):
            probs = 1.0 / (1.0 + np.exp(-raw_scores))
        else:
            probs = raw_scores

        return probs.astype(np.float32)
