# Amazon ML Challenge 2026: Multilingual Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-orange.svg)](https://lightgbm.readthedocs.io/)
[![Validation](https://img.shields.io/badge/Validator-PASS-brightgreen.svg)](#validation)

A high-performance, two-stage entity resolution pipeline designed for the **Amazon ML Challenge 2026**. This system links deduplicated reference business records in **Source 1 (S1)** with their noisy, unstructured counterpart records in **Source 2 (S2)** and **Source 3 (S3)**, or identifies them as **Singletons** (entities with zero matches) across millions of multilingual records.

---

## 📌 Problem & Key Insights

In commercial platforms, business identity data originates from disparate, noisy sources without shared global identifiers. The challenge evaluates submissions using **Macro $F_{0.5}$** across $1.73\text{M}$ test entities:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- **Precision is weighted 2x over Recall:** False merges (falsely linking two different businesses) are penalized twice as heavily as missed matches.
- **Singletons matter:** An entity with no matches scores **$1.0$** if an empty list is predicted, but drops to **$0.0$** on even a single false match.

### Exploratory Data Discoveries
1. **Zero Cross-Country Matching Law:** In an audit of $741,074$ ground truth matching pairs, **$0.00\%$ cross country borders**. Entities strictly match within their exact country label.
2. **The Script Mismatch Dilemma:** $100\%$ of Source 1 records are in Latin script. In contrast, $\sim 18.9\%$ of Indian records in Source 2 and Source 3 are written in native Indic scripts (Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali, Malayalam, Gurmukhi, Oriya). Without phonetic transliteration, standard string similarity is $0.00\%$.
3. **Open-Set Country Behavior (France):** France appears exclusively in the test set ($\sim 14.9\%$ of test rows) with accented characters (`é`, `è`, `ç`), 5-digit French postal codes, and French legal forms (`SARL`, `SAS`, `SCI`).
4. **Universal Numeric Anchors:** PIN codes (6-digit IN, 5-digit US/FR) and house/plot numbers (`RZ-142`, `1410`) provide invariant discriminative anchors.

---

## 🏗️ System Architecture

Our solution employs a **two-stage architecture** optimized for candidate recall, precision calibration, and fast inference throughput ($>400\text{ entities/sec}$):

```
┌────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: NORMALIZATION                          │
│  - Unicode NFKD & diacritic stripping (é -> e, ô -> o)                 │
│  - Deterministic Indic transliteration (anyascii: Indic -> Latin ASCII)│
│  - Legal suffix standardization (Pvt Ltd, SARL, SAS -> <LEGAL_SUFFIX>) │
│  - Address building number & PIN code extraction                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                STAGE 2: MULTI-ROUTE FUZZY BLOCKER                      │
│  - Partition strictly by Country (US, India, France)                  │
│  - Route 1: Exact normalized root name inverted index                  │
│  - Route 2: Phonetic prefix bigram index (e.g. sury_om, pion_tech)     │
│  - Route 3: Postal code + 4-char name prefix inverted index            │
│  - Route 4: Distinctive word token inverted index                      │
│  - Route 5: Address number + street/locality token anchor              │
│  - Adaptive candidate budget: Up to 20 candidates per S1 entity        │
│  ===> Writes candidate_pairs.tsv                                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   STAGE 3: SUPERVISED PAIR SCORING                     │
│  - 21 engineered pairwise similarity & conflict features:              │
│    * RapidFuzz full name & root name token sort/set ratios            │
│    * Jaro-Winkler character similarity                                 │
│    * Building number match (+1), missing (0), conflict (-1)            │
│    * Postal/PIN code agreement flag                                    │
│    * Retrieval route frequency and score rank                          │
│  - LightGBM Gradient Boosted Decision Tree                            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             STAGE 4: SEGMENT CALIBRATION & ADAPTIVE SELECTION          │
│  - Country-stratified thresholds (US: 0.70, India: 0.65, France: 0.70) │
│  - Singleton protection gate (emit empty list if top candidate < 0.60) │
│  - Adaptive rank + prefix rule (retains close multi-match ties)        │
│  ===> Writes matching_results.tsv                                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```text
Amazon_ML_Challenge/
├── code/
│   └── business_entity_resolution/
│       ├── artifacts/
│       │   └── matcher_model.joblib    # Pre-trained LightGBM model artifact
│       ├── src/
│       │   ├── __init__.py
│       │   ├── audit.py                # Dataset profiling & Unicode script audit
│       │   ├── blocking.py             # Multi-route prefix & postal blocking engine
│       │   ├── config.py               # Shared constants, paths & segment thresholds
│       │   ├── diagnose_validation.py  # Error decomposition & segment recall diagnostics
│       │   ├── evaluate.py             # Official Macro F0.5 evaluation logic
│       │   ├── features.py             # 21 pairwise feature extractors
│       │   ├── io_utils.py             # Fast TSV chunk streaming & ID parsers
│       │   ├── normalize.py            # Multilingual transliteration & regex parsing
│       │   ├── predict.py              # High-throughput batched test inference engine
│       │   ├── train.py                # LightGBM training & threshold optimization
│       │   └── tune_decision_policy.py # Policy grid search (flat vs segment vs rank+prefix)
│       └── requirements.txt            # Pinned pipeline dependencies
├── utils/
│   └── validate_submission.py          # Official challenge submission validator
├── Documentation_template.md           # Formal competition methodology write-up
├── FULL_DATASET_AND_MODEL_REPORT.md    # In-depth dataset audit & metric breakdown
├── requirements.txt                    # Project-level dependencies
└── README.md                           # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Karthick-564/Amazon_ML_Challenge.git
cd Amazon_ML_Challenge
pip install -r requirements.txt
```

### 2. Dataset Setup

Place the competition data into a `dataset/` directory structured as follows:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

### 3. Run Test Inference

To generate predictions on the full test set using the pre-trained model:

```bash
cd code/business_entity_resolution
python -m src.predict --data-root ../../dataset --model-path artifacts/matcher_model.joblib --output-dir ../../output --batch-size 8000
```

This generates:
- `output/matching_results.tsv` (Leaderboard submission file)
- `output/candidate_pairs.tsv` (Candidate pairs file)

### 4. Validate Submission

Verify that both output files strictly satisfy the competition format requirements:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### 5. Train the Model from Scratch (Optional)

To re-train the LightGBM matcher on the training set:

```bash
cd code/business_entity_resolution
python -m src.train --data-root ../../dataset --output-model artifacts/matcher_model.joblib --sample-s1 30000
```

---

## 📊 Benchmark & Validation Results

Evaluated on the held-out validation split:

| Segment | Candidate Recall (Blocker) | Final Recall (Classifier) | Macro $F_{0.5}$ |
| :--- | :--- | :--- | :--- |
| **Overall** | **88.34%** | **87.79%** | **0.9366** |
| **United States** | **90.76%** | **90.35%** | **0.9412** |
| **India** | **84.69%** | **83.92%** | **0.9304** |
| **Same-Script (Latin)** | **90.80%** | **90.30%** | **0.9388** |
| **Singleton Accuracy** | — | — | **97.35% Recall** (184/189 preserved) |

- **Official Submission Validator Status:** `PASS — no blocking issues found. Safe to submit.`

---

## 📜 License & Compliance

- **License:** MIT License.
- **Model Constraints:** LightGBM binary classifier ($<1\text{ MB}$, $<100\text{k}$ parameters), fully compliant with the 8 Billion parameter competition ceiling.
- **External Lookups:** Zero external lookups, geocoding APIs, or web lookups utilized. All normalization and transliteration performed locally and deterministically via `anyascii`.
