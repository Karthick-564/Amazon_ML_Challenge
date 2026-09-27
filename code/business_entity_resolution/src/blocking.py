"""Country-partitioned multi-route blocking and candidate generation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

from .config import MAX_CANDIDATES_PER_S1
from .normalize import NormalizedEntity


def extract_prefix_keys(root_name: str) -> list[str]:
    """Generate discriminative phonetic prefix keys to bridge spelling & transliteration drift."""
    tokens = [t for t in root_name.split() if len(t) >= 2]
    if not tokens:
        return []
    keys = []
    # 1. Exact first 2 tokens
    if len(tokens) >= 2:
        keys.append(f"{tokens[0]}_{tokens[1]}")
        keys.append("_".join(sorted(tokens[:2])))
        # 4-char prefix bigram (handles suffix drift e.g. surya_om vs sury_om)
        p1 = tokens[0][:4]
        p2 = tokens[1][:4]
        keys.append(f"{p1}_{p2}")
        keys.append("_".join(sorted([p1, p2])))
    if len(tokens) >= 3:
        p1 = tokens[0][:4]
        p3 = tokens[2][:4]
        keys.append(f"{p1}_{p3}")
    elif len(tokens) == 1 and len(tokens[0]) >= 5:
        keys.append(tokens[0][:5])
    return list(set(keys))


class CountryIndex:
    """High-recall dual-view candidate retrieval index with phonetic skeleton & multi-tenant detection."""

    def __init__(self, country: str):
        self.country = country
        self.name_index: dict[str, list[str]] = defaultdict(list)
        self.prefix_index: dict[str, list[str]] = defaultdict(list)
        self.token_index: dict[str, list[str]] = defaultdict(list)
        self.skeleton_3gram_index: dict[str, list[str]] = defaultdict(list)
        self.postal_prefix_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.addr_num_token_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.token_doc_counts: Counter[str] = Counter()
        self.address_doc_counts: Counter[str] = Counter()
        self.entity_store: dict[str, NormalizedEntity] = {}

    def add_target(self, target: NormalizedEntity) -> None:
        """Register a target entity (S2 or S3) in the country index."""
        eid = target.entity_id
        self.entity_store[eid] = target

        # Track address density (flags multi-tenant / domiciliation commercial centers)
        if target.clean_address:
            self.address_doc_counts[target.clean_address] += 1

        # View 1: Exact root name
        if len(target.root_name) >= 3:
            self.name_index[target.root_name].append(eid)

        # View 1b: Prefix bigram keys
        for pkey in extract_prefix_keys(target.root_name):
            if len(self.prefix_index[pkey]) < 80:
                self.prefix_index[pkey].append(eid)

        # View 2: Consonant skeleton 3-grams (bridges Indic transliteration and French accent drift)
        for trigram in target.skeleton_3grams:
            if len(self.skeleton_3gram_index[trigram]) < 100:
                self.skeleton_3gram_index[trigram].append(eid)

        # Count token frequencies for rare token detection
        tokens = set(target.root_name.split())
        for tok in tokens:
            if len(tok) >= 3:
                self.token_doc_counts[tok] += 1

        # View 1c: Postal code + first token prefix
        if target.postal_code and tokens:
            first_tok = target.root_name.split()[0][:4]
            if len(first_tok) >= 3:
                self.postal_prefix_index[(target.postal_code, first_tok)].append(eid)

        # View 1d: Address number + token indexing
        addr_words = [w for w in target.clean_address.split() if len(w) >= 4]
        for num in target.address_numbers[:2]:
            for w in addr_words[:3]:
                if len(self.addr_num_token_index[(num, w)]) < 50:
                    self.addr_num_token_index[(num, w)].append(eid)

    def finalize_tokens(self, max_token_freq: int = 1500) -> None:
        """Build the inverted index for distinctive tokens."""
        for eid, target in self.entity_store.items():
            tokens = set(target.root_name.split())
            for tok in tokens:
                if len(tok) >= 3 and self.token_doc_counts[tok] <= max_token_freq:
                    if len(self.token_index[tok]) < 50:
                        self.token_index[tok].append(eid)

    def is_high_density_hub(self, address: str) -> bool:
        """Returns True if the address hosts >= 20 entities (commercial plaza / domiciliation hub)."""
        return self.address_doc_counts.get(address, 0) >= 20

    def retrieve_candidates(
        self, s1: NormalizedEntity, max_candidates: int = 50
    ) -> list[tuple[str, float]]:
        """Retrieve top candidate IDs using dual views (word tokens + consonant skeleton 3-grams)."""
        candidate_scores: dict[str, float] = defaultdict(float)

        # 1. Exact root name match (weight 5.0)
        if len(s1.root_name) >= 3:
            for tid in self.name_index.get(s1.root_name, ()):
                candidate_scores[tid] += 5.0

        # 2. Prefix bigram matches (weight 3.5)
        for pkey in extract_prefix_keys(s1.root_name):
            for tid in self.prefix_index.get(pkey, ())[:40]:
                candidate_scores[tid] += 3.5

        # 3. Consonant skeleton 3-gram match (weight 2.5 — recovers transliterations)
        for trigram in s1.skeleton_3grams:
            for tid in self.skeleton_3gram_index.get(trigram, ())[:30]:
                candidate_scores[tid] += 2.5

        # 4. Distinctive token match in root name (weight 2.0)
        tokens = set(s1.root_name.split())
        for tok in tokens:
            if len(tok) >= 3:
                df = self.token_doc_counts.get(tok, 0)
                if 0 < df <= 1500:
                    weight = 2.5 if df < 100 else 1.5
                    for tid in self.token_index.get(tok, ())[:35]:
                        candidate_scores[tid] += weight

        # 5. Postal code + first token prefix (weight 4.0)
        if s1.postal_code and tokens:
            first_tok = s1.root_name.split()[0][:4]
            if len(first_tok) >= 3:
                for tid in self.postal_prefix_index.get((s1.postal_code, first_tok), ())[:30]:
                    candidate_scores[tid] += 4.0

        # 6. Address number + street word (weight 2.5)
        addr_words = [w for w in s1.clean_address.split() if len(w) >= 4]
        for num in s1.address_numbers[:2]:
            for w in addr_words[:3]:
                for tid in self.addr_num_token_index.get((num, w), ())[:25]:
                    candidate_scores[tid] += 2.5

        if not candidate_scores:
            return []

        # Sort descending by score, retain top candidates up to max_candidates (50)
        ranked = sorted(candidate_scores.items(), key=lambda x: -x[1])[:max_candidates]
        return [(tid, score) for tid, score in ranked if score >= 1.5]
