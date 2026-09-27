"""Country-partitioned multi-route blocking and candidate generation.

Route additions:
  Route 2 -- Token bigram prefix keys (renamed from 'phonetic prefix' -- was
             misleading; now correctly named extract_token_bigram_keys).
  Route 6 -- Character 3-5 gram TF-IDF inverted index (Gap 4).
  Route 7 -- Genuine phonetic blocking using consonant-skeleton codes (Gap 8).
             Catches transliteration drift, misspellings, and phonetically-
             similar business names that Routes 1-6 miss entirely.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import NamedTuple, Optional

from .address_parser import phonetic_keys_for_name
from .config import MAX_CANDIDATES_PER_S1
from .normalize import NormalizedEntity

# ---------------------------------------------------------------------------
# Route 6 tuning constants
# ---------------------------------------------------------------------------

# Maximum posting-list entries stored per character n-gram (memory guard).
_NGRAM_POST_CAP: int = 200

# Only n-grams with IDF >= this value contribute during retrieval.
# Equivalent to keeping n-grams that appear in <= N / e^1.8 ~= 16.5% of docs.
_NGRAM_MIN_IDF: float = 1.8

# Maximum posting-list entries consulted per n-gram at query time.
_NGRAM_LOOKUP_CAP: int = 60

# Multiplier applied to TF-IDF score before accumulating into candidate_scores.
# Kept < 1 so Route 6 supplements rather than dominates Routes 1-5.
_NGRAM_ROUTE_WEIGHT: float = 0.55


# ---------------------------------------------------------------------------
# Public result type (Gap 5/6: exposes n_routes per candidate)
# ---------------------------------------------------------------------------

class CandidateResult(NamedTuple):
    """Structured result from retrieve_candidates_full.

    Attributes:
        entity_id:       Target entity ID (S2-xxx or S3-xxx).
        retrieval_score: Combined multi-route heuristic score.
        n_routes:        Number of distinct blocking routes (1-7) that found
                         this candidate.  Used as a candidate-context feature.
    """
    entity_id: str
    retrieval_score: float
    n_routes: int


# ---------------------------------------------------------------------------
# Shared key-generation helpers
# ---------------------------------------------------------------------------


def extract_token_bigram_keys(root_name: str) -> list:
    """Generate token-bigram prefix keys for Route 2 (order-invariant prefix pairs).

    These are *not* phonetic keys -- they are exact prefix substrings of token
    pairs, providing high-precision recall for names whose spelling matches
    within the first 4 characters.  Genuine phonetic retrieval is Route 7.
    """
    tokens = [t for t in root_name.split() if len(t) >= 2]
    if not tokens:
        return []
    keys = []
    if len(tokens) >= 2:
        keys.append(f"{tokens[0]}_{tokens[1]}")
        keys.append("_".join(sorted(tokens[:2])))
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


# Backward-compatible alias (used by predict.py if it calls directly -- safe to keep)
extract_prefix_keys = extract_token_bigram_keys


def extract_char_ngrams(text: str, min_n: int = 3, max_n: int = 5) -> list[str]:
    """Extract character n-grams (min_n..max_n) from *text* for TF-IDF retrieval.

    Word boundaries are marked with an underscore before extraction so that
    word-initial and word-final character positions are distinguishable, e.g.
    'abc' at a word start produces '_ab' and '_abc' in addition to 'abc'.
    This improves discrimination between words that share interior substrings.

    Returns a flat list with duplicates (suitable for Counter-based TF counting).
    """
    if not text:
        return []
    marked = "_" + "_".join(text.split()) + "_"
    length = len(marked)
    ngrams: list[str] = []
    for n in range(min_n, max_n + 1):
        for i in range(length - n + 1):
            ngrams.append(marked[i: i + n])
    return ngrams


# ---------------------------------------------------------------------------
# Country-partitioned blocking index
# ---------------------------------------------------------------------------


class CountryIndex:
    """High-recall multi-route candidate retrieval index for a single country partition."""

    def __init__(self, country: str):
        self.country = country

        # Routes 1-5 (original, unchanged)
        self.name_index: dict[str, list[str]] = defaultdict(list)
        self.prefix_index: dict[str, list[str]] = defaultdict(list)
        self.token_index: dict[str, list[str]] = defaultdict(list)
        self.postal_prefix_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.addr_num_token_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.token_doc_counts: Counter[str] = Counter()
        self.entity_store: dict[str, NormalizedEntity] = {}

        # Route 6: character n-gram TF-IDF (Gap 4)
        self.ngram_doc_counts: Counter = Counter()
        self.ngram_idf: dict = {}
        self.ngram_posting: dict = defaultdict(list)

        # Route 7: phonetic token index (Gap 8)
        # phonetic_index maps phonetic_key -> list[entity_id].
        # Keys are generated by phonetic_keys_for_name() which produces both
        # single-word codes ("ph1_srm00") and sorted bigram codes
        # ("ph2_srm00_rjst0") -- order-invariant and noise-tolerant.
        self.phonetic_index: dict = defaultdict(list)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def add_target(self, target: NormalizedEntity) -> None:
        """Register a target entity (S2 or S3) in the country index."""
        eid = target.entity_id
        self.entity_store[eid] = target

        # Route 1: Exact root name
        if len(target.root_name) >= 3:
            self.name_index[target.root_name].append(eid)

        # Route 2: Prefix bigram keys
        for pkey in extract_prefix_keys(target.root_name):
            if len(self.prefix_index[pkey]) < 80:
                self.prefix_index[pkey].append(eid)

        # Word-token document-frequency counts (Routes 3 and 6 IDF)
        name_tokens = set(target.root_name.split())
        for tok in name_tokens:
            if len(tok) >= 3:
                self.token_doc_counts[tok] += 1

        # Route 3 (postal+prefix) and Route 4 (addr num+token) are built here
        # Route 3: Postal code + first token prefix
        if target.postal_code and name_tokens:
            first_tok = target.root_name.split()[0][:4]
            if len(first_tok) >= 3:
                self.postal_prefix_index[(target.postal_code, first_tok)].append(eid)

        # Route 4: Address number + street/locality word
        addr_words = [w for w in target.clean_address.split() if len(w) >= 4]
        for num in target.address_numbers[:2]:
            for w in addr_words[:3]:
                if len(self.addr_num_token_index[(num, w)]) < 40:
                    self.addr_num_token_index[(num, w)].append(eid)

        # Route 6 (Gap 4): accumulate n-gram document-frequency counts.
        combined = (target.root_name + " " + target.clean_address).strip()
        for ng in set(extract_char_ngrams(combined, 3, 5)):
            self.ngram_doc_counts[ng] += 1

        # Route 7 (Gap 8): phonetic key indexing.
        # Posting list per phonetic key is capped at 60 to bound memory use.
        for pkey in phonetic_keys_for_name(target.root_name):
            pl = self.phonetic_index[pkey]
            if len(pl) < 60:
                pl.append(eid)

    def finalize_tokens(self, max_token_freq: int = 1500) -> None:
        """Build Route 3 token index and Route 6 char n-gram TF-IDF structures.

        Must be called once after **all** target records for this country have
        been added via add_target().  Routes 1, 2, 4, and 5 are built
        incrementally in add_target() and need no further work here.
        """
        N = len(self.entity_store)
        if N == 0:
            return

        # Route 3: distinctive word-token inverted index
        for eid, target in self.entity_store.items():
            tokens = set(target.root_name.split())
            for tok in tokens:
                if len(tok) >= 3 and self.token_doc_counts[tok] <= max_token_freq:
                    if len(self.token_index[tok]) < 50:
                        self.token_index[tok].append(eid)

        # Route 6 (Gap 4): compute IDF values and build posting lists.
        #
        # We only keep n-grams whose IDF >= _NGRAM_MIN_IDF, which translates to
        # df <= N / exp(_NGRAM_MIN_IDF).  This discards very common substrings
        # such as "_the", "_pvt", "_ltd" that would flood posting lists.
        max_df_allowed = int(N / math.exp(_NGRAM_MIN_IDF))
        self.ngram_idf = {
            ng: math.log(N / df)
            for ng, df in self.ngram_doc_counts.items()
            if 1 < df <= max_df_allowed
        }

        for eid, target in self.entity_store.items():
            combined = (target.root_name + " " + target.clean_address).strip()
            for ng in set(extract_char_ngrams(combined, 3, 5)):
                if ng in self.ngram_idf:
                    pl = self.ngram_posting[ng]
                    if len(pl) < _NGRAM_POST_CAP:
                        pl.append(eid)

        # Free intermediate frequency counters to keep RAM lean
        self.token_doc_counts.clear()
        self.ngram_doc_counts.clear()
        # Route 7 phonetic index is built incrementally in add_target() --
        # no further work needed here.

    # ------------------------------------------------------------------
    # Candidate retrieval
    # ------------------------------------------------------------------

    def retrieve_candidates_full(
        self, s1: NormalizedEntity, max_candidates: int = MAX_CANDIDATES_PER_S1
    ) -> list[CandidateResult]:
        """Retrieve top candidates with per-candidate route-count metadata.

        Returns a list of :class:`CandidateResult` sorted descending by
        ``retrieval_score``.  Only candidates with ``score >= 1.5`` are
        returned.  ``n_routes`` indicates how many of the six blocking routes
        contributed to that candidate's score -- a feature used by the
        LightGBM classifier (Gap 5/6).
        """
        candidate_scores: dict[str, float] = defaultdict(float)

        # Per-route hit sets (7 routes: Routes 1-7) for n_routes counting.
        rh: list = [set() for _ in range(7)]

        # Route 1: Exact root name match (weight 5.0)
        if len(s1.root_name) >= 3:
            for tid in self.name_index.get(s1.root_name, ()):
                candidate_scores[tid] += 5.0
                rh[0].add(tid)

        # Route 2: Token bigram prefix matches (weight 3.5)
        for pkey in extract_token_bigram_keys(s1.root_name):
            for tid in self.prefix_index.get(pkey, ())[:40]:
                candidate_scores[tid] += 3.5
                rh[1].add(tid)

        # Route 3: Distinctive token match in root name
        tokens = set(s1.root_name.split())
        for tok in tokens:
            if len(tok) >= 3:
                df = self.token_doc_counts.get(tok, 0)
                if 0 < df <= 1500:
                    weight = 2.5 if df < 100 else 1.5
                    for tid in self.token_index.get(tok, ())[:35]:
                        candidate_scores[tid] += weight
                        rh[2].add(tid)

        # Route 4: Postal code + first token prefix (weight 4.0)
        if s1.postal_code and tokens:
            first_tok = s1.root_name.split()[0][:4]
            if len(first_tok) >= 3:
                for tid in self.postal_prefix_index.get((s1.postal_code, first_tok), ())[:30]:
                    candidate_scores[tid] += 4.0
                    rh[3].add(tid)

        # Route 5: Address number + street/locality word (weight 2.5)
        addr_words = [w for w in s1.clean_address.split() if len(w) >= 4]
        for num in s1.address_numbers[:2]:
            for w in addr_words[:3]:
                for tid in self.addr_num_token_index.get((num, w), ())[:25]:
                    candidate_scores[tid] += 2.5
                    rh[4].add(tid)

        # Route 6: Character 3-5 gram TF-IDF (Gap 4)
        query_combined = (s1.root_name + " " + s1.clean_address).strip()
        query_ng_tf: Counter = Counter(extract_char_ngrams(query_combined, 3, 5))
        for ng, tf in query_ng_tf.items():
            idf = self.ngram_idf.get(ng, 0.0)
            if idf < _NGRAM_MIN_IDF:
                continue
            contrib = min(tf, 4) * idf * _NGRAM_ROUTE_WEIGHT
            for tid in self.ngram_posting.get(ng, ())[:_NGRAM_LOOKUP_CAP]:
                candidate_scores[tid] += contrib
                rh[5].add(tid)

        # Route 7: Phonetic token index (Gap 8)
        # Weight 2.0 -- supplementary: corroborates Routes 1-6 without
        # dominating; posting list limited to 30 per key at query time.
        for pkey in phonetic_keys_for_name(s1.root_name):
            for tid in self.phonetic_index.get(pkey, ())[:30]:
                candidate_scores[tid] += 2.0
                rh[6].add(tid)

        if not candidate_scores:
            return []

        # Sort descending by score, retain top-N candidates with score >= 1.5
        ranked = sorted(candidate_scores.items(), key=lambda x: -x[1])[:max_candidates]
        results: list[CandidateResult] = []
        for tid, score in ranked:
            if score < 1.5:
                break  # sorted descending, so no further candidates qualify
            n_r = sum(1 for route_set in rh if tid in route_set)
            results.append(CandidateResult(tid, score, n_r))
        return results

    def retrieve_candidates(
        self, s1: NormalizedEntity, max_candidates: int = MAX_CANDIDATES_PER_S1
    ) -> list[tuple[str, float]]:
        """Backward-compatible wrapper: returns (entity_id, retrieval_score) pairs."""
        return [(r.entity_id, r.retrieval_score)
                for r in self.retrieve_candidates_full(s1, max_candidates)]
