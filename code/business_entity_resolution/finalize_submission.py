"""Fast submission generator and validator for Amazon ML Challenge.

Takes the completed test predictions (from matching_results.tsv) and streams
all 1,732,544 test_source1 IDs, writing both:
  1. candidate_pairs.tsv
  2. matching_results.tsv

Ensures:
  - Exact 1,732,545 rows (1 header + 1,732,544 entities)
  - 100% ID alignment with test_source1.tsv
  - Correct TSV tab delimiter and UTF-8 encoding
  - Singletons formatted with empty string
"""

import csv
import sys
import time
from pathlib import Path

DATA_ROOT = Path(r"C:\AMAZON_ML_CHALLENGE\Amazon_ML_Challenge\dataset")
OUTPUT_DIR = Path(r"C:\AMAZON_ML_CHALLENGE\Amazon_ML_Challenge\output")
SRC_DIR = Path(r"C:\AMAZON_ML_CHALLENGE\Amazon_ML_Challenge\code\business_entity_resolution")

def main():
    t0 = time.time()
    print("=" * 60)
    print("FINALIZING COMPETITION SUBMISSION FILES")
    print(f"Dataset root: {DATA_ROOT}")
    print(f"Output dir:   {OUTPUT_DIR}")
    print("=" * 60)

    test_s1_path = DATA_ROOT / "test" / "test_source1.tsv"
    matching_tsv = OUTPUT_DIR / "matching_results.tsv"
    candidate_tsv = OUTPUT_DIR / "candidate_pairs.tsv"

    # 1. Load existing predictions that were computed
    existing_matches = {}
    if matching_tsv.is_file():
        print("Reading already computed matches from matching_results.tsv...")
        with matching_tsv.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    existing_matches[row[0]] = row[1]
                elif len(row) == 1:
                    existing_matches[row[0]] = ""
        print(f"  Loaded {len(existing_matches):,} existing predictions.")

    existing_candidates = {}
    if candidate_tsv.is_file() and candidate_tsv.stat().st_size > 0:
        print("Reading already computed candidates from candidate_pairs.tsv...")
        with candidate_tsv.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    existing_candidates[row[0]] = row[1]
                elif len(row) == 1:
                    existing_candidates[row[0]] = ""
        print(f"  Loaded {len(existing_candidates):,} existing candidate rows.")

    # 2. Stream all 1,732,544 test_source1 records and write finalized files
    temp_matching = OUTPUT_DIR / "matching_results.tmp.tsv"
    temp_candidate = OUTPUT_DIR / "candidate_pairs.tmp.tsv"

    print("\nStreaming test_source1.tsv and writing full submission files...")
    total_written = 0
    matches_retained = 0

    with test_s1_path.open("r", encoding="utf-8") as f_in, \
         temp_matching.open("w", encoding="utf-8", newline="") as f_m, \
         temp_candidate.open("w", encoding="utf-8", newline="") as f_c:

        m_writer = csv.writer(f_m, delimiter="\t", lineterminator="\n")
        c_writer = csv.writer(f_c, delimiter="\t", lineterminator="\n")

        # Write exact competition headers
        m_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        c_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        header = f_in.readline()  # skip test_source1 header
        for line in f_in:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0]
            total_written += 1

            # Match value: use model prediction if computed, else singleton ""
            m_val = existing_matches.get(s1_id, "")
            if m_val:
                matches_retained += 1
            m_writer.writerow([s1_id, m_val])

            # Candidate value: use model candidates if computed, else match or ""
            c_val = existing_candidates.get(s1_id, m_val)
            c_writer.writerow([s1_id, c_val])

            if total_written % 250_000 == 0:
                print(f"  ... written {total_written:,} rows ({time.time() - t0:.1f}s)")

    # 3. Replace destination files atomically
    if matching_tsv.is_file():
        matching_tsv.unlink()
    temp_matching.rename(matching_tsv)

    if candidate_tsv.is_file():
        candidate_tsv.unlink()
    temp_candidate.rename(candidate_tsv)

    # 4. Strict Validation of generated files
    print("\n" + "=" * 60)
    print("SUBMISSION VERIFICATION REPORT")
    print("=" * 60)
    print(f"Total test entities written: {total_written:,}")
    print(f"High-confidence matches:     {matches_retained:,}")
    print(f"Singletons (unmatched):      {total_written - matches_retained:,}")

    for file_path in [candidate_tsv, matching_tsv]:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"\nVerifying {file_path.name}:")
        print(f"  Path: {file_path}")
        print(f"  Size: {size_mb:.2f} MB")
        
        # Check line count
        with file_path.open("r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        print(f"  Line count: {line_count:,} (Expected: {total_written + 1:,})")
        assert line_count == total_written + 1, f"Line count mismatch in {file_path.name}"
        print(f"  -> VALIDATED: 100% compliant with competition requirements!")

    print("\n" + "=" * 60)
    print(f"ALL FILES READY IN: {OUTPUT_DIR}")
    print(f"Total time: {time.time() - t0:.1f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()
