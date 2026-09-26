# Implementation Plan

## Operating principle

Every change must be evaluated in this sequence:

```text
candidate recall and candidate volume -> pair-classification quality -> macro F_0.5 -> format validation
```

Never improve the final model while silently reducing candidate recall. Never add a model or external package without recording its version and license.

## Milestone 0 — Environment and project skeleton

**Deliverables**

- `code/business_entity_resolution/src/` project structure.
- Pinned `requirements.txt` for the local classical baseline.
- One command entry point for train, validation, and test prediction.
- A run manifest containing seed, source paths, package versions, thresholds, candidate counts, and metrics.

**Baseline dependencies**

- `numpy`, `pandas`, `scipy`, `scikit-learn`, `rapidfuzz`, `lightgbm`, and `joblib`.
- Add a dense retrieval/model package only after its exact model licence has been checked.

**Exit gate**

The code loads a small row-limited sample of every TSV with `sep="\t"` and writes a valid empty-format submission with every test S1 ID.

## Milestone 1 — Data audit and local evaluator

**Tasks**

1. Read all source files in chunks, retaining entity IDs as strings and preserving UTF-8 Unicode.
2. Parse `matched_entity_ids` into lists/sets.
3. Build S1-level train/validation splits, stratified by dynamic country label and match-count bucket.
4. Implement the challenge's exact macro F_0.5 calculation, including singleton behavior.
5. Produce an audit report: null rates, country counts, script distribution, text lengths, label multiplicity, and target-ID uniqueness across S1 records.

**Exit gate**

The local evaluator reproduces hand-checked outcomes for empty predictions, exact predictions, and a prediction containing one extra ID. The audit report identifies cross-script examples rather than assuming scripts from country.

## Milestone 2 — Normalization and exact/rare blocking baseline

**Tasks**

1. Implement non-destructive Unicode normalization and field-specific tokenization.
2. Extract digits, likely postal/PIN sequences, script categories, and token frequencies.
3. Build disk-backed indices for strict keys: country + normalized full name; country + rare name token; country + address number + rare address token.
4. Generate training validation candidates using only retrieval structures built from S2/S3 data.
5. Measure candidate recall, total pairs, and candidates-per-S1 distribution.

**Exit gate**

Candidate recall and size are reported overall and by S2/S3, country, script group, and number of true links. No pair outside the final candidate table reaches the matcher.

## Milestone 3 — Sparse fuzzy retrieval baseline

**Tasks**

1. Fit character 3–5 gram TF-IDF vectorizers on combined source text in memory-safe batches.
2. Create separate name, address, and combined-text retrieval routes for S2 and S3.
3. Retrieve limited top candidates per route; union and deduplicate with strict-block candidates.
4. Add retrieval score/rank/route-count fields.
5. Tune candidate pruning on validation data, including an adaptive rather than one-size-fits-all candidate budget.

**Exit gate**

The union improves validation candidate recall over Milestone 2 without unbounded candidate growth. The exact retained candidates are emitted as a valid `candidate_pairs.tsv`.

## Milestone 4 — Supervised high-precision matcher

**Tasks**

1. Label candidate pairs from the train ground truth.
2. Construct hard-negative training data from high-ranking nonmatching candidates.
3. Implement pair features: name/address character similarity, token overlap, edit similarity, numeric agreement/conflict, country equality, missingness, and retrieval context.
4. Train a LightGBM model with deterministic seed and validation-based early stopping.
5. Calibrate probability/threshold decisions against macro F_0.5 at the S1 list level.

**Exit gate**

Report pairwise precision/recall only as diagnostics; promote a model only when held-out **macro F_0.5** improves. Record false positive examples, especially singleton false merges.

## Milestone 5 — Cross-script enhancement

**Tasks**

1. Add deterministic romanization as a reversible additional representation.
2. Re-run sparse character retrieval and pair features on romanized name/address text.
3. Compare candidate recall and F_0.5 specifically on script-different positive pairs.
4. If still needed, evaluate one small multilingual embedding model with verified compatible licence and fully documented version.

**Exit gate**

Keep an enhancement only if it increases held-out macro F_0.5 or cross-script candidate recall at an acceptable candidate-volume cost.

## Milestone 6 — Full inference and submission

**Tasks**

1. Refit approved components on all training labels.
2. Build test retrieval indices and produce final candidates.
3. Score exactly the final candidate pairs.
4. Write both TSVs with every test S1 row, including empty lists.
5. Run `utils/validate_submission.py` locally, and use `--check-ids` if available memory permits.
6. Package source, pinned requirements, reproducibility README, outputs, and completed methodology template.

**Exit gate**

The validator passes, final matches are a subset of candidate pairs, code regenerates both files from supplied data, and the methodology accurately reports observed metrics.

## Experiment ledger

Each experiment is logged with:

| Field | Example |
|---|---|
| Run ID | `m3_char_tfidf_v2` |
| Split seed | `2026` |
| Candidate routes | strict + name-char + address-char |
| Candidate recall | overall and slices |
| Mean / P95 candidates | per S1 |
| Matcher model | version and parameters |
| Decision rule | threshold / prefix policy |
| Validation macro F_0.5 | primary selection score |
| Failure notes | singleton false merges, script misses |

## Immediate next task

Implement Milestone 0 and Milestone 1 before selecting a matcher or cloud resource. They establish the evaluator, data facts, and reproducible baseline required to make every later choice evidence-based.
