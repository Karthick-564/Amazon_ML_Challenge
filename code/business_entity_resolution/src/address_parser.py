"""Lightweight address field extractor and phonetic code generator.

Gap 7 -- Address parsing:
  Extracts approximate structural fields (house number, street, locality, city)
  from a pre-normalised address string using regex heuristics.  No external
  geocoding or NLP library is required.

Gap 8 -- Phonetic coding:
  phonetic_code() produces a language-agnostic consonant-skeleton code that
  collapses common phonetic variants (ph/f, ck/k, th/t ...) and removes
  vowels after the first character.  Used by CountryIndex Route 7 to retrieve
  candidates whose business names *sound* like the query even when spelling
  diverges (transliteration noise, misspellings, abbreviation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_HOUSE_TOKEN_RE = re.compile(r"^\d+[a-z]?$")

_SKIP_TOKENS = frozenset(
    ["road", "rd", "street", "st", "lane", "ln", "avenue", "ave",
     "nagar", "marg", "gali", "cross", "main", "block", "phase",
     "sector", "area", "colony", "layout", "ext", "extn"]
)


@dataclass(slots=True)
class ParsedAddress:
    house_number: str
    street_tokens: list
    locality_tokens: list
    city_token: str


def parse_address(clean_address: str) -> ParsedAddress:
    """Decompose a pre-normalised address into approximate structural fields."""
    tokens = clean_address.split()
    if not tokens:
        return ParsedAddress("", [], [], "")

    house_toks: list = []
    rest: list = []
    i = 0
    # Only collect initial digit-based tokens as house/building number
    while i < len(tokens):
        t = tokens[i]
        if _HOUSE_TOKEN_RE.match(t):
            house_toks.append(t)
            i += 1
        else:
            break
    rest = tokens[i:]

    n = len(rest)
    if n == 0:
        street_toks: list = []
        locality_toks: list = []
    elif n == 1:
        street_toks = rest
        locality_toks = []
    elif n <= 3:
        street_toks = rest[:1]
        locality_toks = rest[1:]
    else:
        split = max(1, n * 2 // 5)
        street_toks = rest[:split]
        locality_toks = rest[split:]

    city_tok = ""
    search_pool = locality_toks or street_toks
    for t in reversed(search_pool):
        if len(t) >= 3 and not t.isdigit():
            city_tok = t
            break

    return ParsedAddress(
        house_number=" ".join(house_toks),
        street_tokens=street_toks,
        locality_tokens=locality_toks,
        city_token=city_tok,
    )


_PHONETIC_SUBS: list = [
    ("ph",  "f"),
    ("ck",  "k"),
    ("qu",  "k"),
    ("th",  "t"),
    ("gh",  "g"),
    ("kh",  "k"),
    ("sh",  "s"),
    ("ch",  "c"),
    ("wh",  "w"),
    ("tz",  "ts"),
    ("x",   "ks"),
    ("z",   "s"),
    ("v",   "b"),
    ("w",   ""),
    ("y",   "i"),
    ("j",   "dz"),
]

_VOWELS = frozenset("aeiou")


def phonetic_code(word: str, length: int = 5) -> str:
    """Compute a consonant-skeleton phonetic code for word."""
    if not word:
        return "0" * length
    s = word.lower()
    for old, new in _PHONETIC_SUBS:
        s = s.replace(old, new)

    result = s[0]
    for ch in s[1:]:
        if ch not in _VOWELS:
            result += ch

    deduped = result[0] if result else "0"
    for ch in result[1:]:
        if ch != deduped[-1]:
            deduped += ch

    return (deduped[:length]).ljust(length, "0")


def phonetic_keys_for_name(root_name: str, min_word_len: int = 4) -> list:
    """Generate phonetic key pairs from root_name for Route 7 blocking index."""
    words = [w for w in root_name.split() if len(w) >= min_word_len]
    if not words:
        return []
    codes = [phonetic_code(w) for w in words]
    keys: list = []

    for code in codes:
        if code != "00000":
            keys.append("ph1_" + code)

    if len(codes) >= 2:
        for i in range(len(codes) - 1):
            pair = tuple(sorted([codes[i], codes[i + 1]]))
            keys.append("ph2_" + pair[0] + "_" + pair[1])

    return list(set(keys))
