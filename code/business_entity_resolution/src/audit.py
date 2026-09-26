"""Memory-safe audit of challenge sources and labels."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

from .config import SOURCE_COLUMNS, TRUTH_COLUMNS
from .io_utils import parse_id_list, read_tsv


def unicode_script_hint(char: str) -> str:
    """Return a script-like Unicode label without assuming a language inventory.

    Python's standard library exposes Unicode character *names*, but not the full
    Unicode Script property. Alphabetic names conventionally begin with their script
    (for example ``TAMIL LETTER``, ``BENGALI LETTER``, ``ARABIC LETTER``,
    ``CYRILLIC CAPITAL LETTER``). Using that leading token lets the audit discover
    scripts present in the supplied data rather than hard-coding Tamil, Hindi/
    Devanagari, or a fixed country-language mapping. This is reporting metadata only;
    the matching pipeline will retain the original Unicode text.
    """
    name = unicodedata.name(char, "")
    if not name:
        return f"UNNAMED_U+{ord(char):04X}"
    first_word = name.split(maxsplit=1)[0]
    # CJK ideographs have names beginning with CJK; retaining this distinct family is
    # more informative than the generic leading token alone.
    return first_word


def script_group(text: str) -> str:
    """Return all discovered Unicode-script hints for audit slicing, not matching."""
    groups = {unicode_script_hint(char) for char in text if char.isalpha()}
    return "+".join(sorted(groups)) if groups else "None"


def source_summary(path: Path) -> dict[str, object]:
    countries: Counter[str] = Counter()
    name_scripts: Counter[str] = Counter()
    address_scripts: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    name_lengths: list[int] = []
    address_lengths: list[int] = []
    rows = 0
    for row in read_tsv(path, SOURCE_COLUMNS):
        rows += 1
        countries[row["country"]] += 1
        for field in ("business_name", "business_address", "country"):
            if not row[field]:
                missing[field] += 1
        name_scripts[script_group(row["business_name"])] += 1
        address_scripts[script_group(row["business_address"])] += 1
        name_lengths.append(len(row["business_name"]))
        address_lengths.append(len(row["business_address"]))

    def lengths(values: list[int]) -> dict[str, float]:
        values.sort()
        if not values:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "mean": round(sum(values) / len(values), 2),
            "p50": values[int(0.50 * (len(values) - 1))],
            "p95": values[int(0.95 * (len(values) - 1))],
            "max": values[-1],
        }

    return {
        "rows": rows,
        "countries": dict(countries.most_common()),
        "missing": dict(missing),
        "name_scripts": dict(name_scripts.most_common()),
        "address_scripts": dict(address_scripts.most_common()),
        "name_length": lengths(name_lengths),
        "address_length": lengths(address_lengths),
    }


def truth_summary(path: Path) -> dict[str, object]:
    multiplicity: Counter[int] = Counter()
    s2_links = s3_links = rows = 0
    for row in read_tsv(path, TRUTH_COLUMNS):
        rows += 1
        links = parse_id_list(row["matched_entity_ids"])
        multiplicity[len(links)] += 1
        s2_links += sum(value.startswith("S2-") for value in links)
        s3_links += sum(value.startswith("S3-") for value in links)
    links = s2_links + s3_links
    return {
        "source1_rows": rows,
        "total_links": links,
        "source2_links": s2_links,
        "source3_links": s3_links,
        "links_per_source1": round(links / rows, 5) if rows else 0.0,
        "singletons": multiplicity[0],
        "singleton_rate": round(multiplicity[0] / rows, 5) if rows else 0.0,
        "match_count_distribution": dict(sorted(multiplicity.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit challenge data without loading all rows into memory.")
    parser.add_argument("--data-root", required=True, type=Path, help="Folder containing train/ and test/.")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    files = {
        "train_source1": args.data_root / "train" / "train_source1.tsv",
        "train_source2": args.data_root / "train" / "train_source2.tsv",
        "train_source3": args.data_root / "train" / "train_source3.tsv",
        "test_source1": args.data_root / "test" / "test_source1.tsv",
        "test_source2": args.data_root / "test" / "test_source2.tsv",
        "test_source3": args.data_root / "test" / "test_source3.tsv",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    truth_path = args.data_root / "train" / "train_ground_truth.tsv"
    if not truth_path.is_file():
        missing.append(str(truth_path))
    if missing:
        raise FileNotFoundError("Missing required data files: " + ", ".join(missing))

    report: dict[str, object] = {name: source_summary(path) for name, path in files.items()}
    report["train_ground_truth"] = truth_summary(truth_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "data_audit.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
