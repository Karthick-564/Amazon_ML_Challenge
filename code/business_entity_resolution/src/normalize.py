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

from .address_parser import ParsedAddress, parse_address


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

# Canonical country name aliases mapping to standard representation
COUNTRY_ALIASES = {
    "us": "US",
    "usa": "US",
    "u s": "US",
    "u s a": "US",
    "united states": "US",
    "united states of america": "US",
    "in": "India",
    "ind": "India",
    "india": "India",
    "bharat": "India",
    "fr": "France",
    "fra": "France",
    "france": "France",
    "uk": "UK",
    "gbr": "UK",
    "gb": "UK",
    "united kingdom": "UK",
    "great britain": "UK",
    "england": "UK",
    "de": "Germany",
    "deu": "Germany",
    "germany": "Germany",
    "deutschland": "Germany",
    "ca": "Canada",
    "can": "Canada",
    "canada": "Canada",
    "au": "Australia",
    "aus": "Australia",
    "australia": "Australia",
    "jp": "Japan",
    "jpn": "Japan",
    "japan": "Japan",
    "cn": "China",
    "chn": "China",
    "china": "China",
    "br": "Brazil",
    "bra": "Brazil",
    "brazil": "Brazil",
    "mx": "Mexico",
    "mex": "Mexico",
    "mexico": "Mexico",
    "es": "Spain",
    "esp": "Spain",
    "spain": "Spain",
    "it": "Italy",
    "ita": "Italy",
    "italy": "Italy",
    "sg": "Singapore",
    "singapore": "Singapore",
    "nl": "Netherlands",
    "nld": "Netherlands",
    "netherlands": "Netherlands",
}

# Regex to detect country name mentioned near the tail of an address if country is missing
COUNTRY_TAIL_RE = re.compile(
    r"\b(united states|usa|u\.s\.a\.|u\.s\.|india|bharat|france|united kingdom|uk|great britain|germany|deutschland|canada|australia|japan|china|brazil|spain|italy|singapore|netherlands)\s*$",
    re.IGNORECASE,
)

# Known country-specific postal code regexes
COUNTRY_POSTAL_REGEX = {
    "India": re.compile(r"\b[1-9][0-9]{5}\b"),                                # 6-digit PIN
    "US": re.compile(r"\b[0-9]{5}(?:-[0-9]{4})?\b"),                          # 5-digit ZIP or ZIP+4
    "France": re.compile(r"\b[0-9]{5}\b"),                                    # 5-digit code
    "Germany": re.compile(r"\b[0-9]{5}\b"),                                   # 5-digit code
    "Spain": re.compile(r"\b[0-9]{5}\b"),                                     # 5-digit code
    "Italy": re.compile(r"\b[0-9]{5}\b"),                                     # 5-digit code
    "UK": re.compile(r"\b[A-Z]{1,2}[0-9][A-Z0-9]?\s*[0-9][A-Z]{2}\b", re.I),  # UK alphanumeric
    "Canada": re.compile(r"\b[A-Z][0-9][A-Z]\s*[0-9][A-Z][0-9]\b", re.I),    # Canadian alphanumeric
    "Australia": re.compile(r"\b[0-9]{4}\b"),                                 # 4-digit code
    "Japan": re.compile(r"\b[0-9]{3}-?[0-9]{4}\b"),                           # 7-digit code
    "Brazil": re.compile(r"\b[0-9]{5}-?[0-9]{3}\b"),                          # 8-digit code
    "Singapore": re.compile(r"\b[0-9]{6}\b"),                                 # 6-digit code
    "Netherlands": re.compile(r"\b[1-9][0-9]{3}\s?[A-Z]{2}\b", re.I),         # 4-digit + 2 letters
}

# Generic open-set fallback postal regexes (applied in priority order if country is unknown/unseen)
GENERIC_POSTAL_PATTERNS = [
    re.compile(r"\b[A-Z]{1,2}[0-9][A-Z0-9]?\s*[0-9][A-Z]{2}\b", re.I),       # Alphanumeric UK/Commonwealth
    re.compile(r"\b[A-Z][0-9][A-Z]\s*[0-9][A-Z][0-9]\b", re.I),               # Alphanumeric Canadian
    re.compile(r"\b[1-9][0-9]{4,5}\b"),                                       # 5 to 6 digits (global standard in ~80% of nations)
    re.compile(r"\b\d{4}\b"),                                                 # 4-digit near end of address
]

NUMBERS_RE = re.compile(r"\b\d+\b")  # standalone numbers


def normalize_country(raw_country: str, address_fallback: str = "") -> str:
    """Normalize a raw country string into a canonical partition key.

    Handles:
    - Known aliases, ISO codes, and full country names.
    - Case variations, punctuation, and leading/trailing whitespace.
    - Fallback country extraction from address if raw_country is missing/empty.
    - Open-set unseen countries: canonicalizes to title-case/upper-case so unseen
      countries partition consistently across sources without hardcoding.
    """
    if raw_country:
        c_strip = raw_country.strip()
        c_low = c_strip.lower()
        if c_low in COUNTRY_ALIASES:
            return COUNTRY_ALIASES[c_low]
        cleaned = re.sub(r"[^\w\s]", " ", c_strip).strip().lower()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if cleaned in COUNTRY_ALIASES:
            return COUNTRY_ALIASES[cleaned]
        # Open-set unseen country: format nicely as Title Case or uppercase code
        if len(cleaned) <= 3:
            return cleaned.upper()
        return cleaned.title()

    # Fallback: check if address ends with a known country name
    if address_fallback:
        m = COUNTRY_TAIL_RE.search(address_fallback.strip())
        if m:
            detected = m.group(1).lower().replace(".", "")
            if detected in COUNTRY_ALIASES:
                return COUNTRY_ALIASES[detected]

    return "UNKNOWN"


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


def extract_postal_code(text: str, country: str = "") -> Optional[str]:
    """Extract country-appropriate postal code from address, with generic open-set fallback.

    If the country has a known postal standard, applies that pattern first.
    If the country is unknown or unseen, sequentially evaluates standard international
    postal patterns (alphanumeric, 5-6 digit, 4-digit) to capture postal anchors globally.
    """
    if not text:
        return None

    # 1. Known country-specific pattern
    canon_country = normalize_country(country) if country else ""
    if canon_country in COUNTRY_POSTAL_REGEX:
        pat = COUNTRY_POSTAL_REGEX[canon_country]
        match = pat.search(text)
        if match:
            raw = match.group(0).strip()
            # For US ZIP+4, use first 5 digits
            if canon_country == "US" and len(raw) > 5 and "-" in raw:
                return raw[:5]
            return raw

    # 2. Generic open-set fallback: evaluate international patterns
    for pat in GENERIC_POSTAL_PATTERNS:
        match = pat.search(text)
        if match:
            return match.group(0).strip()

    return None


@dataclass(slots=True)
class NormalizedEntity:
    entity_id: str
    country: str
    clean_name: str
    root_name: str
    legal_suffix: str
    clean_address: str
    address_numbers: list
    postal_code: Optional[str]
    is_non_latin: bool
    # Parsed address fields (Gap 7)
    addr_house: str             # house / building number string
    addr_street: list           # tokens for street / road portion
    addr_locality: list         # tokens for area / colony / locality
    addr_city: str              # city proxy token (last substantive word)


def normalize_record(
    entity_id: str, business_name: str, business_address: str, country: str
) -> NormalizedEntity:
    """Normalize a business entity record into a structured normalized representation."""
    is_non_latin = contains_non_latin(business_name) or contains_non_latin(business_address)
    clean_name = normalize_text_base(business_name)
    root_name, legal_suffix = extract_legal_suffix(clean_name)
    clean_address = normalize_text_base(business_address)
    address_numbers = extract_numbers(clean_address)
    canon_country = normalize_country(country, business_address)
    postal_code = extract_postal_code(business_address, canon_country)
    parsed = parse_address(clean_address)

    return NormalizedEntity(
        entity_id=entity_id,
        country=canon_country,
        clean_name=clean_name,
        root_name=root_name,
        legal_suffix=legal_suffix,
        clean_address=clean_address,
        address_numbers=address_numbers,
        postal_code=postal_code,
        is_non_latin=is_non_latin,
        addr_house=parsed.house_number,
        addr_street=parsed.street_tokens,
        addr_locality=parsed.locality_tokens,
        addr_city=parsed.city_token,
    )
