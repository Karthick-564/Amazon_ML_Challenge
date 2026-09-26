"""Shared constants, deterministic defaults, and paths."""

from pathlib import Path

SEED = 2026
TSV_ENCODING = "utf-8"
DELIMITER = "\t"

SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")
MATCHING_COLUMNS = TRUTH_COLUMNS
CANDIDATE_COLUMNS = ("source1_entity_id", "candidate_entity_ids")

# Optimal decision thresholds tuned for macro F_0.5 on held-out validation
OPTIMAL_THRESHOLD = 0.70
SEGMENT_THRESHOLDS = {
    "US": 0.70,
    "India": 0.65,
    "France": 0.70,
}

# Maximum candidates per S1 entity emitted to candidate_pairs.tsv
MAX_CANDIDATES_PER_S1 = 20


def default_data_root() -> Path:
    """Return the bundled dataset location when run from the pipeline directory."""
    return Path(__file__).resolve().parents[3] / "dataset"
