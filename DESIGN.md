# High-Precision Business Entity Resolution: Technical Design

## Decision summary

Build a **CPU-first, two-stage entity-resolution pipeline**:

1. Multi-view candidate retrieval: exact/rare-token rules plus sparse character n-gram search.
2. Supervised pair classifier: a precision-calibrated gradient-boosted tree over name, address, country, numeric, and retrieval features.

The pipeline produces `candidate_pairs.tsv` from the *final* retrieved-and-pruned candidates, then scores exactly those pairs to produce `matching_results.tsv`. It uses only the supplied data and local text transformations. No business lookup, geocoding, or external entity data is used.

## Why this design

- Training has 2,206,821 S1 entities and 7,638,365 positive links; test has 1,732,544 S1 entities and almost 10 million S2/S3 records. Exhaustive comparison is infeasible.
- Labels are one-to-many: mean 3.46 links per S1 and up to 11. The system must independently accept several candidates, not select one nearest neighbour.
- The metric is macro F_0.5 and penalises false merges. Decisions must be conservative and validated at the S1-list level.
- The data can include different scripts (for example Tamil versus Latin), so raw character similarity alone is insufficient.

## Resource budget and execution mode

Detected local hardware: 16 GB RAM, GTX 1650 Ti (4 GB VRAM), 819 GB free on D:.

**Local baseline: supported.** All large source data are read in chunks and all intermediate tables are persisted on D:. Sparse TF-IDF, rule blocks, feature extraction, and LightGBM training run on CPU.

**Local multilingual embedding experiment: limited.** The 4 GB GPU is enough for batched encoding with a small multilingual model, but not for comfortably holding/searching a full dense index together with the model. Run it CPU/batched or move only this optional experiment to a cloud GPU.

Do not use cloud for business-data lookup or augmentation. If a cloud GPU is used, upload only the supplied challenge data and project code; record the model name, exact version, license, and commands for reproducibility.

## Architecture

```text
TSV sources
  -> chunked ingestion and Unicode-safe normalization
  -> disk-backed retrieval indices over S2 and S3
  -> union of retrieval routes per S1
  -> lightweight candidate rank/prune
  -> candidate_pairs.tsv (exact ML inference set)
  -> pair feature extraction
  -> calibrated LightGBM classifier
  -> S1-level conservative decision policy
  -> matching_results.tsv
  -> official validator
```

## 1. Text representations

Retain raw text and produce several non-destructive views for every name and address.

| View | Purpose |
|---|---|
| `raw` | Auditability and script identification. |
| `unicode_normalized` | NFKC/case-fold/whitespace normalization. |
| `latin_folded` | Accent-insensitive Latin matching. |
| `romanized` | Cross-script phonetic bridge; derived locally via deterministic transliteration. |
| `tokenized` | Token overlap, rare-token retrieval, number extraction. |
| `char_text` | Character 3-5 gram TF-IDF retrieval. |

Legal suffixes and common address abbreviations are represented in an additional normalized view; the original tokens remain available. Do not destructively replace ambiguous tokens such as `St`.

Extract and preserve digit sequences, likely postal/PIN patterns, and address-number tokens. Numeric disagreement is strong negative evidence; agreement is not sufficient on its own.

## 2. Candidate generation

Create retrieval indices independently for Source 2 and Source 3. A candidate is retained if returned by any route below, then duplicate IDs are removed.

1. Exact strict blocks: dynamic country equality plus normalized name, rare name token, or address-number + rare address token.
2. Sparse character TF-IDF: top-k nearest neighbours for name, address, and combined text.
3. Romanized sparse retrieval: character TF-IDF over romanized name/address for cross-script matches.
4. Optional multilingual dense retrieval: a small explicitly MIT/Apache-2.0 licensed multilingual model, only after the sparse baseline is measured. Retrieve in batches and persist candidate IDs/scores.

Candidate union is ranked cheaply using retrieval ranks, route count, name/address sparse scores, country agreement, and numeric compatibility. Keep a small **adaptive** final candidate set—not a single fixed top-k. Its size must allow multi-match S1 records, while low-scoring generic candidates are removed.

The final retained IDs become `candidate_pairs.tsv`; the scorer never sees any other pair.

### Blocking evaluation

For every validation run report:

- positive-link candidate recall (overall, S2/S3, country, script group, and match-count group);
- mean, median, P95, and maximum candidates per S1;
- total candidates and reduction ratio;
- candidate precision;
- candidate recall at each retrieval route and at their union.

## 3. Supervised matching model

Generate training pairs using the same candidate pipeline. Positives are ground-truth links. Negatives are retrieved but unlabeled pairs; oversample difficult negatives with high name or address similarity. Never train primarily on random cross-dataset pairs.

Initial model: LightGBM binary classifier (MIT license), trained with S1-group-aware train/validation split.

### Features

- name: exact flags, character cosine, edit/Jaro ratios, token Jaccard/containment, rare-token overlap, length and digit agreement;
- address: equivalent features plus house-number, postal/PIN, locality-token agreement and conflict flags;
- cross-script: raw-script indicator, romanized character similarity, optional multilingual retrieval score;
- context: country equality as an open-set string feature, source (S2/S3), missingness;
- retrieval: rank and score in every route, number of routes returning the pair, candidate-score margin to next competitors.

## 4. Validation and decision policy

Split at the S1 entity level, stratifying dynamically by country and number of true links. All threshold selection is performed on the held-out S1 set.

Rank each S1's scored candidates and test output prefixes (`[]`, top 1, top 2, ...). Select the calibrated rule that maximises **macro F_0.5 per S1**, not global pairwise F_0.5. Start with a conservative global threshold, then evaluate source-specific and evidence-aware thresholds only if they improve held-out performance.

Required error slices:

- true singleton false merges;
- cross-script and transliteration pairs;
- same-name/different-address false positives;
- incomplete-address false negatives;
- France readiness proxy: held-out, non-US/India-specific preprocessing tests and script-based slices.

## 5. Reproducible project layout

```text
code/business_entity_resolution/
  src/
    config.py
    normalize.py
    build_indices.py
    generate_candidates.py
    features.py
    train.py
    predict.py
    evaluate.py
    io_utils.py
  requirements.txt
  README.md
output/
  candidate_pairs.tsv
  matching_results.tsv
```

All commands accept explicit input/output paths and random seeds. Record package versions, model license, validation metrics, candidate counts, threshold, and git/source revision in a run manifest.

## Implementation phases

1. **Foundation:** chunked TSV reader, normalization, label parser, S1-level evaluator, and format-safe writers.
2. **Baseline blocker:** strict blocks plus raw character TF-IDF; measure candidate recall and size.
3. **Baseline matcher:** hard-negative generation, features, LightGBM, macro F_0.5 threshold tuning.
4. **Cross-script upgrade:** romanization retrieval and features; quantify improvement on script-different validation links.
5. **Optional dense upgrade:** multilingual embedding retrieval only if it improves candidate recall/F_0.5 enough to justify resources.
6. **Finalization:** full-train/full-test inference, official validation, reproducibility package, and methodology document.

## Cloud decision gate

Stay local through phases 1-4. Use cloud only if phase 5 is justified by validation evidence or local batch inference is too slow. A rented 16 GB+ GPU is sufficient for the optional dense experiment; no multi-GPU or large-LLM infrastructure is required.
