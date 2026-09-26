"""Streaming TSV readers and strict submission-file helpers."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from pathlib import Path

from .config import DELIMITER, TSV_ENCODING


def read_tsv(path: str | Path, expected_columns: tuple[str, ...]) -> Iterator[dict[str, str]]:
    """Yield rows from a UTF-8 TSV and reject unexpected headers early."""
    path = Path(path)
    with path.open("r", encoding=TSV_ENCODING, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=DELIMITER)
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise ValueError(
                f"{path}: expected columns {expected_columns}, got {reader.fieldnames}. "
                "All challenge files must be tab-separated."
            )
        for row in reader:
            yield {key: (value or "").strip() for key, value in row.items()}


def parse_id_list(value: str) -> set[str]:
    """Parse a challenge comma-separated list, returning an empty set for blanks."""
    if not value:
        return set()
    values = [item.strip() for item in value.split(",") if item.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate ID in list: {value!r}")
    return set(values)


def read_id_lists(path: str | Path, expected_value_column: str) -> dict[str, set[str]]:
    """Read an S1-to-ID-list file such as truth, predictions, or candidates."""
    rows = read_tsv(path, ("source1_entity_id", expected_value_column))
    result: dict[str, set[str]] = {}
    for row in rows:
        source_id = row["source1_entity_id"]
        if source_id in result:
            raise ValueError(f"{path}: duplicate source1_entity_id {source_id}")
        result[source_id] = parse_id_list(row[expected_value_column])
    return result


def write_id_lists(
    path: str | Path,
    value_column: str,
    source1_ids: Iterable[str],
    values_by_source1: dict[str, set[str]],
) -> None:
    """Write a format-safe, deterministic TSV with one row per supplied S1 ID."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=TSV_ENCODING, newline="") as handle:
        writer = csv.writer(handle, delimiter=DELIMITER, lineterminator="\n")
        writer.writerow(("source1_entity_id", value_column))
        for source_id in source1_ids:
            values = sorted(values_by_source1.get(source_id, set()))
            writer.writerow((source_id, ",".join(values)))
