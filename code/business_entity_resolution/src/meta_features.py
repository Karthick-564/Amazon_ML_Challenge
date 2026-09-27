"""Multimodal Pairwise Feature Engineering combining handcrafted string/alphanumeric features with neural dense similarity."""

from __future__ import annotations

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .normalize import NormalizedEntity

META_FEATURE_NAMES = [
    # 1. Name string similarities
    "name_sort",
    "name_set",
    "root_sort",
    "root_set",
    "root_partial",
    "root_ratio",
    "name_jw",
    "len_diff",
    "len_ratio",
    # 2. Legal suffix
    "suf_match",
    "suf_conflict",
    # 3. Address similarities
    "addr_set",
    "addr_sort",
    "addr_partial",
    # 4. Alphanumeric & Postal
    "num_score",
    "post_score",
    # 5. Overlaps
    "shared_name_tokens",
    "shared_addr_tokens",
    "retrieval_score",
    "is_s2",
    "has_non_latin",
    # 6. Dense & Hybrid features
    "dense_sim",
    "rrf_score",
    "dense_sim_cubed",
    "num_conflict_strict",
]


def extract_meta_pair_features(
    s1: NormalizedEntity,
    c: NormalizedEntity,
    rrf_score: float = 0.0,
    dense_sim: float = 0.0,
) -> list[float]:
    """Extract comprehensive 25-dimensional feature vector combining RapidFuzz and Neural signals."""
    # 1. Name similarities
    name_sort = fuzz.token_sort_ratio(s1.clean_name, c.clean_name) / 100.0
    name_set = fuzz.token_set_ratio(s1.clean_name, c.clean_name) / 100.0
    root_sort = fuzz.token_sort_ratio(s1.root_name, c.root_name) / 100.0
    root_set = fuzz.token_set_ratio(s1.root_name, c.root_name) / 100.0
    root_partial = fuzz.partial_ratio(s1.root_name, c.root_name) / 100.0
    root_ratio = fuzz.ratio(s1.root_name, c.root_name) / 100.0
    name_jw = float(JaroWinkler.similarity(s1.root_name, c.root_name))

    len1, len2 = len(s1.root_name), len(c.root_name)
    len_diff = float(abs(len1 - len2))
    len_ratio = min(len1, len2) / max(len1, len2, 1)

    # 2. Legal suffix agreement/conflict
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
        # Strict conflict flag: both present, zero intersection, on same street
        num_conflict_strict = 1.0 if (not (nums1 & nums2) and addr_set > 0.65) else 0.0
    else:
        num_score = 0.0
        num_conflict_strict = 0.0

    # 5. Postal code agreement
    if s1.postal_code and c.postal_code:
        post_score = 1.0 if s1.postal_code == c.postal_code else -1.0
    else:
        post_score = 0.0

    # 6. Distinctive token overlaps
    w1_name = set(s1.root_name.split())
    w2_name = set(c.root_name.split())
    shared_name = float(len(w1_name & w2_name))

    w1_addr = set(s1.clean_address.split())
    w2_addr = set(c.clean_address.split())
    shared_addr = float(len(w1_addr & w2_addr))

    is_s2 = 1.0 if c.entity_id.startswith("S2-") else 0.0
    has_non_latin = 1.0 if (s1.is_non_latin or c.is_non_latin) else 0.0

    # 7. Dense & RRF signals
    dense_sim_val = float(dense_sim)
    dense_sim_cubed = float(dense_sim ** 3)
    rrf_score_val = float(rrf_score)

    return [
        name_sort,
        name_set,
        root_sort,
        root_set,
        root_partial,
        root_ratio,
        name_jw,
        len_diff,
        len_ratio,
        suf_match,
        suf_conflict,
        addr_set,
        addr_sort,
        addr_partial,
        num_score,
        post_score,
        shared_name,
        shared_addr,
        rrf_score_val,
        is_s2,
        has_non_latin,
        dense_sim_val,
        rrf_score_val,
        dense_sim_cubed,
        num_conflict_strict,
    ]
