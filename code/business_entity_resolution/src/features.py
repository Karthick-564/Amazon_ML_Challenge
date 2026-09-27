"""Pairwise feature engineering for business entity matching.

Includes 25+ base pairwise features plus candidate-context & sibling evidence features
(rank, max score, gap to second best, multi-tenant commercial hub indicators).
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .normalize import NormalizedEntity

FEATURE_NAMES = [
    "name_sort",
    "name_set",
    "root_sort",
    "root_set",
    "root_partial",
    "root_ratio",
    "name_jw",
    "skeleton_jw",
    "len_diff",
    "len_ratio",
    "suf_match",
    "suf_conflict",
    "addr_set",
    "addr_sort",
    "addr_partial",
    "num_score",
    "first_num_match",
    "post_score",
    "shared_name_tokens",
    "shared_addr_tokens",
    "retrieval_score",
    "is_s2",
    "is_s3",
    "has_non_latin",
    "is_domiciliation_hub",
]


def extract_pair_features(
    s1: NormalizedEntity,
    c: NormalizedEntity,
    retrieval_score: float = 1.0,
    is_hub: bool = False,
) -> list[float]:
    """Compute robust similarity, numeric, phonetic skeleton, and multi-tenant features for a pair."""
    # 1. Name similarities
    name_sort = fuzz.token_sort_ratio(s1.clean_name, c.clean_name) / 100.0
    name_set = fuzz.token_set_ratio(s1.clean_name, c.clean_name) / 100.0
    root_sort = fuzz.token_sort_ratio(s1.root_name, c.root_name) / 100.0
    root_set = fuzz.token_set_ratio(s1.root_name, c.root_name) / 100.0
    root_partial = fuzz.partial_ratio(s1.root_name, c.root_name) / 100.0
    root_ratio = fuzz.ratio(s1.root_name, c.root_name) / 100.0
    name_jw = float(JaroWinkler.similarity(s1.root_name, c.root_name))

    # Phonetic skeleton similarity (bridges Indian script transliterations & French accent drift)
    skeleton_jw = float(JaroWinkler.similarity(s1.skeleton, c.skeleton))

    len1, len2 = len(s1.root_name), len(c.root_name)
    len_diff = float(abs(len1 - len2))
    len_ratio = min(len1, len2) / max(len1, len2, 1)

    # 2. Legal suffix
    suf_match = 1.0 if (s1.legal_suffix and s1.legal_suffix == c.legal_suffix) else 0.0
    suf_conflict = 1.0 if (s1.legal_suffix and c.legal_suffix and s1.legal_suffix != c.legal_suffix) else 0.0

    # 3. Address similarities
    addr_set = fuzz.token_set_ratio(s1.clean_address, c.clean_address) / 100.0
    addr_sort = fuzz.token_sort_ratio(s1.clean_address, c.clean_address) / 100.0
    addr_partial = fuzz.partial_ratio(s1.clean_address, c.clean_address) / 100.0

    # 4. Numeric agreement / conflict
    nums1 = set(s1.address_numbers)
    nums2 = set(c.address_numbers)
    if nums1 and nums2:
        num_score = 1.0 if (nums1 & nums2) else -1.0
    else:
        num_score = 0.0

    # Primary (first) street number match
    if s1.address_numbers and c.address_numbers:
        first_num_match = 1.0 if s1.address_numbers[0] == c.address_numbers[0] else -1.0
    else:
        first_num_match = 0.0

    # 5. Postal code agreement
    if s1.postal_code and c.postal_code:
        post_score = 1.0 if s1.postal_code == c.postal_code else -1.0
    else:
        post_score = 0.0

    # 6. Token overlaps
    w1_name = set(s1.root_name.split())
    w2_name = set(c.root_name.split())
    shared_name = float(len(w1_name & w2_name))

    w1_addr = set(s1.clean_address.split())
    w2_addr = set(c.clean_address.split())
    shared_addr = float(len(w1_addr & w2_addr))

    is_s2 = 1.0 if (c.entity_id.startswith("S2-") or "_s2_" in c.entity_id.lower()) else 0.0
    is_s3 = 1.0 if (c.entity_id.startswith("S3-") or "_s3_" in c.entity_id.lower()) else 0.0
    has_non_latin = 1.0 if (s1.is_non_latin or c.is_non_latin) else 0.0
    is_domiciliation_hub = 1.0 if is_hub else 0.0

    return [
        name_sort,
        name_set,
        root_sort,
        root_set,
        root_partial,
        root_ratio,
        name_jw,
        skeleton_jw,
        len_diff,
        len_ratio,
        suf_match,
        suf_conflict,
        addr_set,
        addr_sort,
        addr_partial,
        num_score,
        first_num_match,
        post_score,
        shared_name,
        shared_addr,
        retrieval_score,
        is_s2,
        is_s3,
        has_non_latin,
        is_domiciliation_hub,
    ]


def append_candidate_context_features(
    base_features: np.ndarray,
    probs: np.ndarray,
) -> np.ndarray:
    """Stage 2: Augment base features with candidate context and sibling evidence.

    Adds 4 context signals per candidate:
    1. Candidate rank within S1's pool
    2. Max candidate probability in S1's pool
    3. Gap to the highest scoring candidate
    4. Gap to the second-highest scoring candidate
    """
    n = len(probs)
    if n == 0:
        return np.empty((0, base_features.shape[1] + 4), dtype=np.float32)

    # Sort indices descending by probability
    sorted_order = np.argsort(-probs)
    ranks = np.empty(n, dtype=np.float32)
    for rank_pos, idx in enumerate(sorted_order):
        ranks[idx] = float(rank_pos + 1)

    max_p = float(np.max(probs))
    second_p = float(probs[sorted_order[1]]) if n > 1 else max_p

    context_cols = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        p_i = probs[i]
        context_cols[i, 0] = ranks[i]
        context_cols[i, 1] = max_p
        context_cols[i, 2] = max_p - p_i
        context_cols[i, 3] = p_i - second_p if i == sorted_order[0] else p_i - max_p

    return np.hstack([base_features, context_cols])
