"""Global Graph Disjoint-Set & Consistency Resolution to enforce structural entity constraints."""

from __future__ import annotations

from collections import defaultdict
from .normalize import NormalizedEntity


class DisjointSet:
    """Disjoint-set union-find data structure."""

    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, i: str) -> str:
        if i not in self.parent:
            self.parent[i] = i
            return i
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: str, j: str) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def resolve_candidate_graph_consistency(
    s1: NormalizedEntity,
    matched_candidates: list[tuple[str, float]],
    entity_store: dict[str, NormalizedEntity],
) -> set[str]:
    """Resolve intra-cluster contradictions (e.g., conflicting building numbers or postal codes)."""
    if len(matched_candidates) <= 1:
        return {tid for tid, _ in matched_candidates}

    # Group into mutually consistent clusters
    s1_nums = set(s1.address_numbers)
    s1_post = s1.postal_code

    valid_matches: list[tuple[str, float]] = []

    # Priority 1: Check against S1 ground facts
    for tid, prob in matched_candidates:
        cand_rec = entity_store.get(tid)
        if not cand_rec:
            continue

        c_nums = set(cand_rec.address_numbers)
        c_post = cand_rec.postal_code

        # If both have postal codes and they conflict heavily (different cities):
        if s1_post and c_post and s1_post != c_post and s1.country == "US":
            # In US, 5-digit zip codes are strict geographic boundaries
            if prob < 0.85:
                continue

        # If both have building numbers and they strictly disagree:
        if s1_nums and c_nums and not (s1_nums & c_nums):
            if prob < 0.88:
                continue

        valid_matches.append((tid, prob))

    if not valid_matches:
        return set()

    # Priority 2: Intra-target consistency check
    # Sort descending by probability
    valid_matches.sort(key=lambda x: -x[1])
    top_tid, top_prob = valid_matches[0]
    top_rec = entity_store.get(top_tid)

    if not top_rec:
        return {tid for tid, _ in valid_matches}

    top_nums = set(top_rec.address_numbers)
    top_post = top_rec.postal_code

    final_kept: set[str] = {top_tid}

    for tid, prob in valid_matches[1:]:
        cand_rec = entity_store.get(tid)
        if not cand_rec:
            continue

        c_nums = set(cand_rec.address_numbers)
        c_post = cand_rec.postal_code

        # If candidate strictly contradicts the top confident match:
        if top_nums and c_nums and not (top_nums & c_nums) and prob < 0.85:
            continue
        if top_post and c_post and top_post != c_post and s1.country == "US" and prob < 0.85:
            continue

        final_kept.add(tid)

    return final_kept
