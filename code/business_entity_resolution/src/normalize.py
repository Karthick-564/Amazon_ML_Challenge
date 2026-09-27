"""Deterministic, language-aware normalization for business entity resolution.

Handles:
1. Indic script transliteration to Latin ASCII (Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali, etc.)
2. Unicode NFKD accent/diacritic stripping (French accents é, è, ê, ç, à, etc. and noisy Latin accents)
3. Legal entity suffix normalization and extraction of core root business name
4. Indian state/city Indic name standardization
5. Address number and postal/PIN code extraction
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

import anyascii


# Common Indian state names in native scripts mapped to standard Latin names
STATE_SCRIPT_MAP = {
    "महाराष्ट्र": "maharashtra",
    "ગુજરાત": "gujarat",
    "दिल्ली": "delhi",
    "मध्य प्रदेश": "madhya pradesh",
    "उत्तर प्रदेश": "uttar pradesh",
    "தமிழ்நாடு": "tamil nadu",
    "कर्नाटक": "karnataka",
    "ఆంధ్రప్రదేశ్": "andhra pradesh",
    "తెలంగాణ": "telangana",
    "पश्चिम बंगाल": "west bengal",
    "ਪੰਜਾਬ": "punjab",
    "राजस्थान": "rajasthan",
    "बिहार": "bihar",
    "हरियाणा": "haryana",
    "केरल": "kerala",
    "ଓଡ଼ିଶା": "odisha",
    "অসম": "assam",
}

# Regex to find any of these Indic state names
STATE_SCRIPT_RE = re.compile(
    "|".join(re.escape(k) for k in STATE_SCRIPT_MAP.keys()), re.UNICODE
)

# Common legal suffixes mapped to canonical category tokens
LEGAL_SUFFIX_MAP = {
    # Private limited variants
    "private limited": "PVT_LTD",
    "pvt ltd": "PVT_LTD",
    "pvt limited": "PVT_LTD",
    "private ltd": "PVT_LTD",
    "praivet limited": "PVT_LTD",
    "piraivet limitet": "PVT_LTD",
    "praivet limitet": "PVT_LTD",
    "pra li": "PVT_LTD",
    "pte ltd": "PVT_LTD",
    "prvt ltd": "PVT_LTD",
    
    # Limited variants
    "limited": "LTD",
    "limitet": "LTD",
    "ltd": "LTD",
    
    # Corporate / Incorporated / LLC
    "corporation": "CORP",
    "corp": "CORP",
    "incorporated": "INC",
    "inc": "INC",
    "llc": "LLC",
    "l l c": "LLC",
    "llp": "LLP",
    
    # French legal entities
    "sarl": "SARL",
    "s a r l": "SARL",
    "sas": "SAS",
    "s a s": "SAS",
    "sasu": "SASU",
    "s a s u": "SASU",
    "sci": "SCI",
    "s c i": "SCI",
    "sa": "SA",
    "s a": "SA",
    "eurl": "EURL",
    "snc": "SNC",
}

# Compiled regex to match legal suffixes at word boundaries (usually near end or middle of string)
LEGAL_SUFFIX_PATTERNS = [
    (re.compile(rf"\b{re.escape(suffix)}\b", re.IGNORECASE), cat)
    for suffix, cat in sorted(LEGAL_SUFFIX_MAP.items(), key=lambda x: -len(x[0]))
]

# Numeric patterns
PIN_CODE_RE = re.compile(r"\b[1-9][0-9]{5}\b")  # 6-digit Indian PIN
US_ZIP_RE = re.compile(r"\b[0-9]{5}(?:-[0-9]{4})?\b")  # 5-digit US ZIP
FR_ZIP_RE = re.compile(r"\b[0-9]{5}\b")  # 5-digit French postal code
NUMBERS_RE = re.compile(r"\b\d+\b")  # standalone numbers


def strip_accents(text: str) -> str:
    """Normalize unicode and strip diacritical marks (e.g., é -> e, ç -> c)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def contains_non_latin(text: str) -> bool:
    """Check if string contains any non-Latin alphabetic character."""
    for ch in text:
        if ch.isalpha():
            o = ord(ch)
            if not ((65 <= o <= 90) or (97 <= o <= 122)):
                return True
    return False


def standardize_indic_states(text: str) -> str:
    """Replace common native script Indian state names with standard Latin names."""
    if not text:
        return text
    return STATE_SCRIPT_RE.sub(lambda m: STATE_SCRIPT_MAP.get(m.group(0), m.group(0)), text)


def normalize_text_base(text: str) -> str:
    """Clean, transliterate if non-latin, strip accents, and lowercase."""
    if not text:
        return ""
    # 1. State substitutions for Indic scripts
    text = standardize_indic_states(text)
    # 2. Transliterate to ASCII if non-Latin characters are present
    if contains_non_latin(text):
        text = anyascii.anyascii(text)
    # 3. Strip accents / diacritics
    text = strip_accents(text)
    # 4. Standardize whitespace & lowercase
    text = text.lower()
    # 5. Clean punctuation to spaces while preserving alphanumeric
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_legal_suffix(clean_name: str) -> tuple[str, str]:
    """Extract canonical legal suffix category and return (root_name, suffix_category)."""
    if not clean_name:
        return "", ""
    suffix_cat = ""
    root = clean_name
    for pattern, cat in LEGAL_SUFFIX_PATTERNS:
        if pattern.search(root):
            suffix_cat = cat
            root = pattern.sub(" ", root)
            break
    root = re.sub(r"\s+", " ", root).strip()
    return root if root else clean_name, suffix_cat


def extract_numbers(text: str) -> list[str]:
    """Extract standalone integer strings from text."""
    if not text:
        return []
    return NUMBERS_RE.findall(text)


def extract_postal_code(text: str, country: str) -> Optional[str]:
    """Extract country-appropriate postal code from address if present."""
    if not text:
        return None
    c_lower = country.lower()
    if "india" in c_lower:
        match = PIN_CODE_RE.search(text)
        return match.group(0) if match else None
    elif "us" in c_lower:
        match = US_ZIP_RE.search(text)
        return match.group(0)[:5] if match else None
    elif "france" in c_lower:
        match = FR_ZIP_RE.search(text)
        return match.group(0) if match else None
    return None


def consonant_skeleton(text: str) -> str:
    """Extract consonant skeleton by dropping vowels to normalize phonetic/accent variations."""
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return re.sub(r"[aeiouy]", "", cleaned).strip()


def extract_3grams(text: str) -> list[str]:
    """Extract character 3-grams for robust fuzzy/phonetic indexing."""
    s = re.sub(r"\s+", "", text.lower())
    if len(s) < 3:
        return [s] if s else []
    return [s[i : i + 3] for i in range(len(s) - 2)]


@dataclass(slots=True)
class NormalizedEntity:
    entity_id: str
    country: str
    clean_name: str
    root_name: str
    legal_suffix: str
    clean_address: str
    address_numbers: list[str]
    postal_code: Optional[str]
    is_non_latin: bool
    skeleton: str
    skeleton_3grams: list[str]


def normalize_record(
    entity_id: str, business_name: str, business_address: str, country: str
) -> NormalizedEntity:
    """Normalize a business entity record into a structured normalized representation."""
    is_non_latin = contains_non_latin(business_name) or contains_non_latin(business_address)
    clean_name = normalize_text_base(business_name)
    root_name, legal_suffix = extract_legal_suffix(clean_name)
    clean_address = normalize_text_base(business_address)
    address_numbers = extract_numbers(clean_address)
    postal_code = extract_postal_code(business_address, country)
    skel = consonant_skeleton(root_name)
    skel_3g = extract_3grams(skel)

    return NormalizedEntity(
        entity_id=entity_id,
        country=country.strip(),
        clean_name=clean_name,
        root_name=root_name,
        legal_suffix=legal_suffix,
        clean_address=clean_address,
        address_numbers=address_numbers,
        postal_code=postal_code,
        is_non_latin=is_non_latin,
        skeleton=skel,
        skeleton_3grams=skel_3g,
    )
