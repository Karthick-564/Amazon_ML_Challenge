"""Streaming TSV readers and strict submission-file helpers."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from pathlib import Path

from .config import DELIMITER, TSV_ENCODING


def read_tsv(path: str | Path, expected_columns: tuple[str, ...]) -> Iterator[dict[str, str]]:
    """Yield rows from a UTF-8 TSV and reject unexpected headers early (fast buffered line splitting)."""
    path = Path(path)
    with path.open("r", encoding=TSV_ENCODING, newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            return
        actual_columns = tuple(header_line.rstrip("\r\n").split(DELIMITER))
        if actual_columns != expected_columns:
            raise ValueError(
                f"{path}: expected columns {expected_columns}, got {actual_columns}. "
                "All challenge files must be tab-separated."
            )
        num_cols = len(expected_columns)
        for line in handle:
            parts = line.rstrip("\r\n").split(DELIMITER)
            if len(parts) == num_cols:
                yield dict(zip(expected_columns, parts))
            else:
                pad = parts + [""] * (num_cols - len(parts))
                yield dict(zip(expected_columns, pad[:num_cols]))


def read_tsv_tuples(path: str | Path, expected_columns: tuple[str, ...]) -> Iterator[list[str]]:
    """Ultra-fast TSV line reader yielding raw column string lists without dict creation overhead."""
    path = Path(path)
    with path.open("r", encoding=TSV_ENCODING, newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            return
        actual_columns = tuple(header_line.rstrip("\r\n").split(DELIMITER))
        if actual_columns != expected_columns:
            raise ValueError(
                f"{path}: expected columns {expected_columns}, got {actual_columns}. "
                "All challenge files must be tab-separated."
            )
        num_cols = len(expected_columns)
        for line in handle:
            parts = line.rstrip("\r\n").split(DELIMITER)
            if len(parts) == num_cols:
                yield parts
            else:
                pad = parts + [""] * (num_cols - len(parts))
                yield pad[:num_cols]


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
