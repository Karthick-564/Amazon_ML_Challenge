# High-Precision Multilingual Business Entity Resolution Pipeline

This directory contains the self-contained, reproducible pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution**.

## Architecture & Innovation Highlights

1. **Language & Script Aware Normalization**:
   - Deterministic transliteration (`anyascii`) converting 9 Indic scripts (Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali, Malayalam, Gurmukhi, Oriya) to Romanized Latin ASCII.
   - Unicode NFKD decomposition and diacritic stripping for accented French and noisy English text.
   - Indian native-script state standardizer (`महाराष्ट्र` -> `maharashtra`, `ગુજરાત` -> `gujarat`, etc.).
   - Standardized legal entity category extraction (`PVT_LTD`, `CORP`, `INC`, `LLC`, `SARL`, `SAS`).
   - Universal numeric anchor & postal code extraction (6-digit Indian PIN, 5-digit US/FR ZIP).

2. **Country-Partitioned Multi-Route Candidate Generation (Blocking)**:
   - Dynamic grouping by `country` (satisfies open-set requirements, zero cross-country leakage).
   - Inverted index on exact normalized root name.
   - Rare-token inverted index on distinctive words.
   - Address numeric anchor + locality word index.
   - Candidate set bounded to an adaptive top 15 per Source-1 entity (achieving 90%+ candidate recall with high candidate precision).

3. **Supervised Pair Matcher & Precision Calibration**:
   - 21 engineered pairwise features spanning token sort/set similarities, Jaro-Winkler, numeric agreement/conflict detection, postal code matches, and retrieval signals.
   - LightGBM binary classifier (MIT licensed, < 8B parameters, CPU-optimized).
   - Calibrated decision threshold tuned specifically for **Macro F_0.5** (prioritizing precision over recall and protecting singletons).

---

## Directory Structure

```text
code/business_entity_resolution/
├── README.md                      # Reproducibility instructions
├── requirements.txt               # Pinned dependencies
├── artifacts/
│   └── matcher_model.joblib       # Trained LightGBM model & threshold metadata
└── src/
    ├── __init__.py
    ├── config.py                  # Shared constants, paths, thresholds
    ├── normalize.py               # Multilingual transliteration & text normalization
    ├── features.py                # Pairwise feature extraction
    ├── blocking.py                # CountryIndex multi-route blocking
    ├── train.py                   # Pair generation, training, threshold tuning
    ├── predict.py                 # Candidate generation & matching inference
    ├── evaluate.py                # Macro F_0.5 evaluation
    ├── io_utils.py                # TSV reader/writer helpers
    └── audit.py                   # Data inspection tool
```

---

## Environment Setup

Install pinned dependencies in Python 3.10+:

```bash
pip install -r requirements.txt
```

---

## Reproduction Commands

### 1. Train the Pair Matcher
To train the LightGBM classifier on the training data and tune the decision threshold:

```bash
python -m src.train --data-root ../../dataset --output-model artifacts/matcher_model.joblib --sample-limit 30000
```

### 2. Generate Final Submissions
To run candidate generation and matching inference on the test set:

```bash
python -m src.predict --data-root ../../dataset --model-path artifacts/matcher_model.joblib --output-dir ../../output
```

This generates:
* `output/candidate_pairs.tsv`
* `output/matching_results.tsv`

### 3. Validate Submission
Run the official challenge validator from `student_resource/`:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
