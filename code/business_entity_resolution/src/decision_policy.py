"""Evidence-aware adaptive candidate cap and match decision policy.

Gap 9 -- Adaptive candidate cap:
  adaptive_cap() returns a per-S1 candidate budget based on name ambiguity,
  presence of address anchors (postal, house number), and retrieval confidence.
  Strong/clear cases use a compact budget; ambiguous S1s expand to a larger
  budget to maintain high candidate recall.

Gap 10 -- Evidence-aware decision policy:
  evidence_aware_match() replaces the flat probability-threshold rule with a
  policy that explicitly weighs match probability, probability margin,
  candidate rank, name/address similarity, numeric conflicts, and postal/house-
  number agreement.  Cases handled:

    * strong name + strong address -> high-confidence match (relaxed threshold)
    * strong name + numeric conflict -> conservative (tighter threshold)
    * weak name + postal + house confirmed -> consider match (relaxed threshold)
    * name-only (no address anchor) -> conservative (strict threshold)
    * default -> segment threshold + relative-to-top check
"""

from __future__ import annotations

from .features import FEATURE_NAMES
from .normalize import NormalizedEntity

# ---------------------------------------------------------------------------
# Pre-computed feature-index lookup (avoids repeated list scans at runtime)
# ---------------------------------------------------------------------------

_F: dict = {name: i for i, name in enumerate(FEATURE_NAMES)}

# Name similarity indices
_I_NAME_SET  = _F["name_set"]
_I_ROOT_SORT = _F["root_sort"]
_I_ROOT_SET  = _F["root_set"]
_I_NAME_JW   = _F["name_jw"]
_I_ROOT_RATIO = _F["root_ratio"]

# Address similarity indices
_I_ADDR_SET     = _F["addr_set"]
_I_ADDR_SORT    = _F["addr_sort"]
_I_ADDR_PARTIAL = _F["addr_partial"]

# Numeric / postal agreement indices
_I_NUM_SCORE   = _F["num_score"]    # +1=match, 0=absent, -1=conflict
_I_POST_SCORE  = _F["post_score"]   # +1=match, 0=absent, -1=conflict

# Parsed address structure indices (Gap 7)
_I_HOUSE_EXACT   = _F["house_num_exact"]
_I_STREET_JAC    = _F["street_jaccard"]
_I_LOCALITY_JAC  = _F["locality_jaccard"]
_I_CITY_SIM      = _F["city_sim"]

# Digit overlap
_I_DIGIT_OVL = _F["digit_overlap"]

# Candidate-context indices (Gap 5)
_I_N_CANDS  = _F["n_candidates"]
_I_RANK     = _F["retrieval_rank"]
_I_RS_NORM  = _F["retrieval_score_norm"]
_I_MARGIN   = _F["retrieval_margin"]
_I_N_ROUTES = _F["n_routes"]


# ---------------------------------------------------------------------------
# Gap 9 -- Adaptive candidate cap
# ---------------------------------------------------------------------------

# Tuning constants (all overridable via adaptive_cap kwargs)
_BASE_CAP = 15
_MIN_CAP  = 8
_MAX_CAP  = 30


def adaptive_cap(
    s1: NormalizedEntity,
    base: int = _BASE_CAP,
    min_cap: int = _MIN_CAP,
    max_cap: int = _MAX_CAP,
) -> int:
    """Compute a per-S1 candidate budget.

    The budget grows for ambiguous / under-anchored queries and shrinks for
    well-anchored ones.  Budget is always clamped to [min_cap, max_cap].

    Signals that *increase* the cap (ambiguity indicators):
      - Very short root name (1 token): +10
      - Short root name (2 tokens): +5
      - No postal code: +3
      - No address numbers: +2
      - Short root name (< 5 chars total): +5

    Signals that *decrease* the cap (anchor indicators):
      - Has postal code AND address numbers: -3
      - Root name is long (>= 4 tokens): -2

    Args:
        s1:      Normalised S1 query entity.
        base:    Starting budget before adjustments.
        min_cap: Hard minimum returned budget.
        max_cap: Hard maximum returned budget.

    Returns:
        Integer candidate budget in [min_cap, max_cap].
    """
    cap = base
    tokens = s1.root_name.split()
    n_toks = len(tokens)

    # Name ambiguity adjustments
    if n_toks <= 1:
        cap += 10
    elif n_toks == 2:
        cap += 5
    elif n_toks >= 4:
        cap -= 2

    # Short overall root name is inherently ambiguous
    if len(s1.root_name) < 5:
        cap += 5

    # Address anchor adjustments
    if not s1.postal_code:
        cap += 3
    if not s1.address_numbers:
        cap += 2

    # Dual anchor: postal + house number = strong anchor
    if s1.postal_code and s1.address_numbers:
        cap -= 3

    return max(min_cap, min(max_cap, cap))


# ---------------------------------------------------------------------------
# Gap 10 -- Evidence-aware decision policy
# ---------------------------------------------------------------------------

def evidence_aware_match(
    s1: NormalizedEntity,
    scored_cands: list,
    base_threshold: float,
) -> set:
    """Select matched target entity IDs using an evidence-aware decision policy.

    Args:
        s1:              Normalised S1 query entity (used for country, address
                         anchor signals).
        scored_cands:    list of (entity_id, probability, feature_vector)
                         tuples for all scored candidates of this S1.
                         May be in any order; sorted internally.
        base_threshold:  The segment-specific decision threshold.

    Returns:
        set[str] of matched entity IDs (may be empty for singletons).
    """
    if not scored_cands:
        return set()

    # Sort descending by model probability
    scored_cands = sorted(scored_cands, key=lambda x: -x[1])

    top_tid, top_prob, top_feats = scored_cands[0]
    n = len(scored_cands)

    # Probability margin: gap between 1st and 2nd candidate
    prob_margin = top_prob - scored_cands[1][1] if n > 1 else top_prob

    # ---- Singleton gate -----------------------------------------------
    # If even the top candidate is very weak, emit nothing (singleton).
    # Gap: original used fixed -0.10; now aware of address anchors.
    singleton_floor = base_threshold - 0.12
    if _has_address_anchors(top_feats):
        singleton_floor -= 0.04   # relax gate when address evidence is present
    if top_prob < singleton_floor:
        return set()

    # ---- Evaluate each candidate --------------------------------------
    matches: set = set()
    for rank, (tid, prob, feats) in enumerate(scored_cands, 1):
        rel_to_top = prob / max(top_prob, 1e-6)
        if _accept(
            prob, top_prob, prob_margin, rank, rel_to_top, feats, base_threshold
        ):
            matches.add(tid)

    return matches


def _has_address_anchors(feats: list) -> bool:
    """Return True if the feature vector shows strong address anchoring."""
    return (
        feats[_I_POST_SCORE] == 1.0
        or feats[_I_HOUSE_EXACT] == 1.0
        or feats[_I_NUM_SCORE] == 1.0
    )


def _accept(
    prob: float,
    top_prob: float,
    prob_margin: float,
    rank: int,
    rel_to_top: float,
    feats: list,
    thresh: float,
) -> bool:
    """Core per-candidate accept/reject logic.

    Implements evidence-weighted cases tailored for macro F_0.5 optimization
    with high true match multiplicity (71% of entities have 3+ true matches):

    1) Numeric conflict               -> reject / conservative (+0.10 threshold)
    2) Strong name + strong address   -> high-confidence match (-0.08 threshold)
    3) Weak name + postal + house + locality -> consider match (-0.10 threshold)
    4) Name-only (no address anchor)  -> conservative (+0.06 threshold)
    5) Default                        -> segment threshold with multi-match support
    """
    # Extract evidence
    name_sim    = feats[_I_NAME_SET]
    root_sort   = feats[_I_ROOT_SORT]
    root_set    = feats[_I_ROOT_SET]
    name_jw     = feats[_I_NAME_JW]
    addr_sim    = feats[_I_ADDR_SET]
    num_score   = feats[_I_NUM_SCORE]
    post_score  = feats[_I_POST_SCORE]
    house_exact = feats[_I_HOUSE_EXACT]
    locality_j  = feats[_I_LOCALITY_JAC]
    city_sim    = feats[_I_CITY_SIM]
    digit_ovl   = feats[_I_DIGIT_OVL]

    # Absolute floor: never match below this regardless of evidence
    if prob < thresh - 0.18:
        return False

    # ---- Case 1: Numeric / house conflict -> reject / very conservative ----
    # House/building numbers explicitly conflict
    if num_score == -1.0:
        if name_sim < 0.88 and root_sort < 0.90:
            return False
        return prob >= thresh + 0.10 and rel_to_top >= 0.70

    # ---- Case 2: Strong name + strong address -> high-confidence match ----
    strong_name = (name_sim >= 0.80 or root_sort >= 0.85 or name_jw >= 0.88)
    strong_addr = (
        addr_sim >= 0.60
        or (post_score == 1.0 and house_exact == 1.0)
        or (post_score == 1.0 and locality_j >= 0.40)
        or (house_exact == 1.0 and city_sim >= 0.65)
    )
    if strong_name and strong_addr:
        return prob >= thresh - 0.08 and rel_to_top >= 0.55

    # ---- Case 3: Weak name + exact postal + house number + locality -> consider match ----
    has_strong_location = (
        post_score == 1.0
        and (house_exact == 1.0 or num_score == 1.0)
        and (locality_j >= 0.30 or addr_sim >= 0.35 or city_sim >= 0.55)
    )
    if has_strong_location:
        return prob >= thresh - 0.10 and rel_to_top >= 0.50

    # ---- Case 4: Name-only (zero address anchor) -> conservative ---------
    no_address_evidence = (
        addr_sim < 0.20
        and post_score <= 0.0
        and num_score <= 0.0
        and house_exact == 0.0
    )
    if no_address_evidence:
        # Require higher threshold when unanchored
        return prob >= thresh + 0.06 and rel_to_top >= 0.65

    # ---- Case 5: Moderate postal evidence alone --------------------------
    if post_score == 1.0:
        return prob >= thresh - 0.04 and rel_to_top >= 0.58

    # ---- Default case ----------------------------------------------------
    # Standard threshold with multi-match support for high-multiplicity data
    if rank == 1:
        return prob >= thresh - 0.02
    else:
        # Multi-match support: keep candidates above threshold that have solid relative confidence
        return prob >= thresh and rel_to_top >= 0.58
