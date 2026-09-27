"""Pairwise feature engineering for business entity matching.

Feature set (34 total):
  - Original 21 similarity, numeric, and source features (unchanged).
  - 8 new pairwise features: char n-gram Jaccard, prefix JaroWinkler,
    long-token overlap, digit sequence Jaccard, name-address cross-similarity,
    root-prefix equality (Gap 6).
  - 5 candidate-context features: n_candidates, retrieval_rank,
    retrieval_score_norm, retrieval_margin, n_routes (Gap 5/6).

All features are leakage-free: context features are derived from blocking
retrieval scores (available before the classifier runs), not from model probs.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .normalize import NormalizedEntity


# ---------------------------------------------------------------------------
# Character n-gram helpers (Gap 5/6 -- defined locally to avoid circular import)
# ---------------------------------------------------------------------------

def _char_ngrams_set(text: str, n: int) -> set:
    """Return the set of character n-grams of length n in text.

    Word boundaries are marked with an underscore so that word-initial and
    word-final positions are distinguishable from interior positions.
    """
    if not text:
        return set()
    marked = "_" + "_".join(text.split()) + "_"
    length = len(marked)
    return {marked[i: i + n] for i in range(length - n + 1)}


def _char_ngram_jaccard(a: str, b: str, n: int = 4) -> float:
    """Jaccard similarity of character n-gram sets for strings a and b."""
    sa = _char_ngrams_set(a, n)
    sb = _char_ngrams_set(b, n)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / (len(sa) + len(sb) - inter)


# ---------------------------------------------------------------------------
# Digit-sequence helper
# ---------------------------------------------------------------------------

_DIGIT_RE = re.compile(r"\d+")


def _digit_seq_jaccard(a: str, b: str) -> float:
    """Jaccard similarity of the sets of digit sub-sequences in a and b."""
    da = set(_DIGIT_RE.findall(a))
    db = set(_DIGIT_RE.findall(b))
    if not da and not db:
        return 1.0
    if not da or not db:
        return 0.0
    inter = len(da & db)
    return inter / (len(da) + len(db) - inter)


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # ---- original 21 features (order preserved for backward compat) ----
    "name_sort",            # 0  token_sort_ratio on full clean name
    "name_set",             # 1  token_set_ratio  on full clean name
    "root_sort",            # 2  token_sort_ratio on root name (no legal suffix)
    "root_set",             # 3  token_set_ratio  on root name
    "root_partial",         # 4  partial_ratio    on root name
    "root_ratio",           # 5  simple ratio     on root name
    "name_jw",              # 6  JaroWinkler on root name
    "len_diff",             # 7  |len(root_a) - len(root_b)|
    "len_ratio",            # 8  min/max(len) of root names
    "suf_match",            # 9  legal suffix equality flag
    "suf_conflict",         # 10 legal suffix conflict flag (both set, different)
    "addr_set",             # 11 token_set_ratio  on clean address
    "addr_sort",            # 12 token_sort_ratio on clean address
    "addr_partial",         # 13 partial_ratio    on clean address
    "num_score",            # 14 building-number match (+1) / missing (0) / conflict (-1)
    "post_score",           # 15 postal/PIN code agreement (+1) / conflict (-1)
    "shared_name_tokens",   # 16 count of shared root-name word tokens
    "shared_addr_tokens",   # 17 count of shared address word tokens
    "retrieval_score",      # 18 combined multi-route blocking score
    "is_s2",                # 19 1.0 if candidate is from Source 2
    "has_non_latin",        # 20 1.0 if either entity has non-Latin characters

    # ---- new pairwise features (Gap 6) ----
    "name_jw_prefix",       # 21 JW on first min(8, len) chars of root name
    "name_4gram_jaccard",   # 22 char 4-gram Jaccard of root names
    "addr_4gram_jaccard",   # 23 char 4-gram Jaccard of clean addresses
    "shared_long_tokens",   # 24 count of shared root-name tokens with len >= 5
    "long_token_jaccard",   # 25 Jaccard of long (>=5-char) token sets
    "digit_overlap",        # 26 Jaccard of digit sub-sequence sets in addresses
    "name_addr_cross",      # 27 partial_ratio(s1.root_name, c.clean_address)
    "root_prefix4_eq",      # 28 1.0 if root names share first 4 chars

    # ---- candidate-context features (Gap 5) -- leakage-free ----
    "n_candidates",         # 29 total candidates retrieved for this S1 entity
    "retrieval_rank",       # 30 rank of this candidate by score (1 = highest)
    "retrieval_score_norm", # 31 this score / max score for this S1 (0-1)
    "retrieval_margin",     # 32 max score - this score (0 = rank 1)
    "n_routes",             # 33 distinct blocking routes that found this candidate

    # ---- address-structure features (Gap 7) ----
    "house_num_exact",      # 34 exact house/building number match (0 or 1)
    "street_jaccard",       # 35 token Jaccard of parsed street portion
    "locality_jaccard",     # 36 token Jaccard of parsed locality portion
    "city_sim",             # 37 exact / fuzzy match of city proxy token
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_pair_features(
    s1: NormalizedEntity,
    c: NormalizedEntity,
    retrieval_score: float = 1.0,
    n_candidates: int = 1,
    retrieval_rank: int = 1,
    retrieval_score_norm: float = 1.0,
    retrieval_margin: float = 0.0,
    n_routes: int = 1,
) -> list:
    """Compute 34 similarity, numeric, and context features for a candidate pair.

    Args:
        s1:                   Normalised Source-1 query entity.
        c:                    Normalised candidate (S2/S3) entity.
        retrieval_score:      Combined multi-route blocking score.
        n_candidates:         Total candidates retrieved for s1 this run.
        retrieval_rank:       1-based rank by retrieval score (1 = highest).
        retrieval_score_norm: retrieval_score / max_score_for_s1 in [0, 1].
        retrieval_margin:     max_score - retrieval_score (0 for rank-1 candidate).
        n_routes:             Number of the 6 blocking routes that found this candidate.
    """
    # ---- original 21 -------------------------------------------------------

    name_sort    = fuzz.token_sort_ratio(s1.clean_name, c.clean_name) / 100.0
    name_set     = fuzz.token_set_ratio(s1.clean_name, c.clean_name) / 100.0
    root_sort    = fuzz.token_sort_ratio(s1.root_name, c.root_name) / 100.0
    root_set     = fuzz.token_set_ratio(s1.root_name, c.root_name) / 100.0
    root_partial = fuzz.partial_ratio(s1.root_name, c.root_name) / 100.0
    root_ratio   = fuzz.ratio(s1.root_name, c.root_name) / 100.0
    name_jw      = float(JaroWinkler.similarity(s1.root_name, c.root_name))

    len1, len2 = len(s1.root_name), len(c.root_name)
    len_diff  = float(abs(len1 - len2))
    len_ratio = min(len1, len2) / max(len1, len2, 1)

    suf_match = 1.0 if (s1.legal_suffix and s1.legal_suffix == c.legal_suffix) else 0.0
    suf_conflict = (
        1.0 if (s1.legal_suffix and c.legal_suffix and s1.legal_suffix != c.legal_suffix)
        else 0.0
    )

    addr_set     = fuzz.token_set_ratio(s1.clean_address, c.clean_address) / 100.0
    addr_sort    = fuzz.token_sort_ratio(s1.clean_address, c.clean_address) / 100.0
    addr_partial = fuzz.partial_ratio(s1.clean_address, c.clean_address) / 100.0

    nums1, nums2 = set(s1.address_numbers), set(c.address_numbers)
    if nums1 and nums2:
        num_score = 1.0 if (nums1 & nums2) else -1.0
    else:
        num_score = 0.0

    if s1.postal_code and c.postal_code:
        post_score = 1.0 if s1.postal_code == c.postal_code else -1.0
    else:
        post_score = 0.0

    w1_name = set(s1.root_name.split())
    w2_name = set(c.root_name.split())
    shared_name_tokens = float(len(w1_name & w2_name))

    w1_addr = set(s1.clean_address.split())
    w2_addr = set(c.clean_address.split())
    shared_addr_tokens = float(len(w1_addr & w2_addr))

    is_s2         = 1.0 if c.entity_id.startswith("S2-") else 0.0
    has_non_latin = 1.0 if (s1.is_non_latin or c.is_non_latin) else 0.0

    # ---- new pairwise 8 (Gap 6) --------------------------------------------

    # 21: JaroWinkler on first 8 chars (prefix discrimination)
    p1, p2 = s1.root_name[:8], c.root_name[:8]
    name_jw_prefix = float(JaroWinkler.similarity(p1, p2)) if (p1 and p2) else 0.0

    # 22-23: Character 4-gram Jaccard for name and address
    name_4gram_jaccard = _char_ngram_jaccard(s1.root_name, c.root_name, 4)
    addr_4gram_jaccard = _char_ngram_jaccard(s1.clean_address, c.clean_address, 4)

    # 24-25: Long-token overlap (tokens >= 5 chars are discriminative)
    long1 = {t for t in w1_name if len(t) >= 5}
    long2 = {t for t in w2_name if len(t) >= 5}
    shared_long_tokens = float(len(long1 & long2))
    union_long = len(long1 | long2)
    long_token_jaccard = (len(long1 & long2) / union_long) if union_long > 0 else 1.0

    # 26: Digit sub-sequence Jaccard (house numbers, PIN mismatches)
    digit_overlap = _digit_seq_jaccard(s1.clean_address, c.clean_address)

    # 27: Cross-field: does S1 root name appear in candidate's address?
    name_addr_cross = fuzz.partial_ratio(s1.root_name, c.clean_address) / 100.0

    # 28: Root name shares first 4 chars
    pref1, pref2 = s1.root_name[:4], c.root_name[:4]
    root_prefix4_eq = 1.0 if (pref1 and pref2 and pref1 == pref2) else 0.0

    # ---- candidate-context 5 (Gap 5) passed in from blocking loop ----------
    # 29-33: n_candidates, retrieval_rank, retrieval_score_norm,
    #        retrieval_margin, n_routes

    # ---- address-structure 4 (Gap 7) ---------------------------------------

    # 34: exact house/building number equality
    house_num_exact = (
        1.0 if (s1.addr_house and c.addr_house and s1.addr_house == c.addr_house)
        else 0.0
    )

    # 35: street-token Jaccard (first portion of address, road/lane name)
    ws1 = set(s1.addr_street)
    wc1 = set(c.addr_street)
    union_s = len(ws1 | wc1)
    street_jaccard = len(ws1 & wc1) / union_s if union_s > 0 else 1.0

    # 36: locality-token Jaccard (area/colony/neighbourhood portion)
    ws2 = set(s1.addr_locality)
    wc2 = set(c.addr_locality)
    union_l = len(ws2 | wc2)
    locality_jaccard = len(ws2 & wc2) / union_l if union_l > 0 else 1.0

    # 37: city proxy token similarity (last substantive token; exact or fuzz)
    if s1.addr_city and c.addr_city:
        city_sim = 1.0 if s1.addr_city == c.addr_city else fuzz.ratio(s1.addr_city, c.addr_city) / 100.0
    else:
        city_sim = 0.0

    return [
        # original 21
        name_sort, name_set, root_sort, root_set, root_partial, root_ratio,
        name_jw, len_diff, len_ratio, suf_match, suf_conflict,
        addr_set, addr_sort, addr_partial, num_score, post_score,
        shared_name_tokens, shared_addr_tokens, retrieval_score, is_s2, has_non_latin,
        # new pairwise 8 (Gap 6)
        name_jw_prefix, name_4gram_jaccard, addr_4gram_jaccard,
        shared_long_tokens, long_token_jaccard, digit_overlap,
        name_addr_cross, root_prefix4_eq,
        # candidate-context 5 (Gap 5)
        float(n_candidates), float(retrieval_rank), retrieval_score_norm,
        retrieval_margin, float(n_routes),
        # address-structure 4 (Gap 7)
        house_num_exact, street_jaccard, locality_jaccard, city_sim,
    ]
