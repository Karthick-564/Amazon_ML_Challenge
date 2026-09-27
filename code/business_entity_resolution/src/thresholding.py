"""High-precision threshold calibration and pool-side 1-to-1 bipartite exclusivity.

Ensures that every candidate record in Source 2 or Source 3 is linked to AT MOST ONE
Source 1 entity (the globally highest-scoring match), eliminating duplicate false merges
and safeguarding singletons (zero-match entities) under Macro F0.5.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple


# Default calibrated plateau thresholds by country and vendor source
DEFAULT_SEGMENT_THRESHOLDS = {
    "US": {"S2": 0.70, "S3": 0.75},
    "India": {"S2": 0.68, "S3": 0.72},
    "France": {"S2": 0.65, "S3": 0.70},
}


def apply_bipartite_exclusivity(
    candidate_scores: Dict[str, List[Tuple[str, float]]],
    s1_countries: Dict[str, str],
    thresholds: Dict[str, Dict[str, float]] | None = None,
    domiciliation_hubs: Set[str] | None = None,
) -> Dict[str, List[str]]:
    """Apply calibrated decision cutoffs followed by a global 1-to-1 target exclusivity pass.

    Args:
        candidate_scores: Mapping s1_id -> list of (target_id, model_probability).
        s1_countries: Mapping s1_id -> country string (US, India, France).
        thresholds: Nested dict: country -> {'S2': threshold, 'S3': threshold}.
        domiciliation_hubs: Set of target_ids that belong to high-density commercial centers.

    Returns:
        Mapping s1_id -> list of final deduplicated matched target_ids.
    """
    if thresholds is None:
        thresholds = DEFAULT_SEGMENT_THRESHOLDS

    if domiciliation_hubs is None:
        domiciliation_hubs = set()

    # 1. Collect all valid candidate links passing source & country thresholds
    candidate_edges = []
    for s1_id, scores in candidate_scores.items():
        country = s1_countries.get(s1_id, "US")
        country_cfg = thresholds.get(country, thresholds.get("US", {"S2": 0.70, "S3": 0.75}))

        for target_id, prob in scores:
            src_key = "S3" if target_id.startswith("S3-") or "_s3_" in target_id.lower() else "S2"
            cutoff = country_cfg.get(src_key, 0.70)

            # Apply stricter threshold on multi-tenant / domiciliation commercial addresses
            if target_id in domiciliation_hubs:
                cutoff = max(cutoff, 0.82)

            if prob >= cutoff:
                candidate_edges.append((prob, s1_id, target_id))

    # 2. Sort candidate links globally in descending order of confidence
    candidate_edges.sort(key=lambda x: x[0], reverse=True)

    # 3. Greedy maximum-weight matching: each target_id is consumed by at most one S1
    assigned_targets: Set[str] = set()
    matches_by_s1: Dict[str, List[str]] = defaultdict(list)

    for prob, s1_id, target_id in candidate_edges:
        if target_id not in assigned_targets:
            assigned_targets.add(target_id)
            matches_by_s1[s1_id].append(target_id)

    # 4. Guarantee all S1 entities exist in output mapping (empty list for singletons)
    result = {}
    for s1_id in candidate_scores.keys():
        m_list = sorted(matches_by_s1.get(s1_id, []))
        result[s1_id] = m_list

    return result
