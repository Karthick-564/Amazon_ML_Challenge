# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolvers  
**Team Members:** Amazon ML Challenge Participant  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We developed a high-precision, multilingual, two-stage entity resolution pipeline designed specifically for the macro $F_{0.5}$ evaluation metric. Our approach couples **script-aware canonical normalization** (handling 9 Indic scripts and accented French text) with **country-partitioned multi-route blocking**, followed by a **precision-calibrated LightGBM pairwise classifier**. On held-out validation Source 1 entities, our pipeline achieves a **Macro $F_{0.5}$ score of 0.9285** while maintaining an ultra-compact candidate set of ~11.5 candidates per Source 1 entity (99.999% reduction ratio) and achieving near-perfect singleton classification (421/429 true singletons correctly identified).

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis across 12.3 million training rows and 11.6 million test rows revealed critical structural insights:
1. **Zero Cross-Country Matching Law:** In an audit of 741,074 ground truth links, 0% crossed country boundaries. Entities only match within their exact country label.
2. **The Script Mismatch Dilemma:** 100% of Source 1 records are in Latin script. In contrast, ~18.9% of Indian records in Sources 2 & 3 are written in native Indic scripts (Devanagari, Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Gurmukhi, Oriya). Without cross-script transliteration, standard string similarity metrics yield 0.00% overlap, causing an immediate ~19% recall ceiling drop.
3. **Open-Set Country Behavior (France):** France appears only in the test set (~14.9% of test rows) with Latin text, French diacritics (`é`, `è`, `ê`, `à`, `ç`), 5-digit postal codes, and French legal forms (`SARL`, `SAS`, `SCI`, `EURL`). Hard-coding a US/India categorical schema causes total failure on test.
4. **Universal Numeric Anchors:** Regardless of language or script, digits (PIN codes, house/plot numbers like `RZ-142`, `1410`, `3121`) are script-invariant and provide high-discriminative evidence. Disagreement between numeric address tokens is a decisive indicator of a false merge.

### 2.2 Solution Strategy
We structured the solution into a four-stage sequential pipeline:
1. **Multilingual Normalization Layer:** Deterministic transliteration (`anyascii`) of non-Latin scripts to Latin ASCII, Unicode NFKD diacritic folding, Indian state name harmonization, and legal suffix standardization (`pvt ltd`, `private limited`, `sarl`, etc. $\to$ `<LEGAL_SUFFIX>`).
2. **Country-Partitioned Multi-Route Blocker:** Retrieval executed independently within country partitions using inverted indexes on exact name roots, rare name tokens, and address numeric anchors.
3. **Gradient-Boosted Pairwise Matcher:** A 21-feature LightGBM model trained on candidate pairs with hard negative sampling from the blocking stage.
4. **Precision-First Threshold Policy:** Global and country-stratified decision threshold tuning optimized directly for macro $F_{0.5}$ list scoring, with conservative abstention on ambiguous records.

**Approach Type:** Hybrid Multi-Route Blocking + Calibrated Gradient-Boosted Decision Tree  
**Core Innovation:** Script-aware transliteration bridging Latin Reference entities to Indic and accented variants, paired with an address numeric conflict detection feature that penalizes false merges under $F_{0.5}$.

---

## 3. Candidate Generation (Blocking)

To scale entity resolution without comparing $1.7 \times 10^{13}$ Cartesian pairs, we partition candidates dynamically by `country` (open set) and retrieve candidates via three complementary inverted-index routes:

- **Route 1 (Exact Normalized Root Name):** Hashes the core business root name after stripping legal suffixes and punctuation.
- **Route 2 (Rare Token Inverted Index):** Indexes tokens appearing in fewer than 250 records across the corpus (high IDF), capturing distinctive business words even when surrounding tokens differ.
- **Route 3 (Address Number + Distinctive Token Anchor):** Indexes tuples of `(address_number, locality_word)` to capture businesses where names are heavily corrupted but physical locations match.

### Blocking Performance
- **Candidate pairs generated per S1 entity:** Mean = 11.56, Median = 11, P95 = 12 (Adaptive cap at top 15).
- **Candidate Recall:** 89.4% – 90.5% on validation entities.
- **How true matches were preserved:**
  - Multi-route union ensures that missing one signal (e.g., name typo) is rescued by another (e.g., address digits or rare token).
  - Transliteration ensures Indic script records share n-grams and tokens with Latin reference records.

---

## 4. Matching Model

### Features Used (21 engineered features)
- **Name Similarities:**
  - RapidFuzz `token_sort_ratio` & `token_set_ratio` on full names.
  - RapidFuzz `token_sort_ratio`, `token_set_ratio`, `partial_ratio`, and basic `ratio` on stripped root names.
  - Jaro-Winkler similarity on root names.
  - Character length difference and length ratio.
  - Canonical legal suffix exact match flag and suffix conflict flag.
  - Shared name token count.
- **Address Similarities:**
  - RapidFuzz `token_set_ratio`, `token_sort_ratio`, and `partial_ratio` on normalized addresses.
  - Shared address token count.
- **Numeric & Postal Features:**
  - **Numeric Agreement/Conflict Flag:** `+1.0` if address numbers intersect; `-1.0` if both records contain numbers but have zero intersection (strong negative evidence); `0.0` if numbers are absent.
  - **Postal/PIN Code Match Flag:** `+1.0` on exact postal code match (6-digit IN, 5-digit US/FR); `-1.0` on postal code contradiction; `0.0` if absent.
- **Retrieval & Script Context:**
  - Heuristic blocking retrieval score.
  - Source indicator (`S2` vs `S3`).
  - Cross-script flag (indicating whether non-Latin script was present).

### Model Architecture & Tuning
- **Model Type:** LightGBM Binary Classifier (MIT License, < 8B parameters, 200 trees, learning rate 0.08, num_leaves 31).
- **Threshold Selection Method:** Threshold search on held-out S1 validation entities maximizing the official competition Macro $F_{0.5}$ metric:
  $$\text{Selected Optimal Threshold} = 0.70 - 0.80$$
  Because $F_{0.5}$ weights precision twice as much as recall, the decision boundary is pushed higher than the default 0.50, rejecting marginal candidates and safeguarding singletons.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** **0.9285** (Validation on 7,500 held-out S1 entities).
- **Exact List Matches:** 4,857 out of 7,500 validation entities (64.8% exact list match).
- **Singleton Performance:** 421 out of 429 true singletons correctly identified (98.1% singleton accuracy; only 8 false merges).
- **Common False Positives (Wrong Merges):**
  - Common chain franchises or multi-branch businesses (e.g., generic retail, pharmacies) sharing identical root names located in the same city or on the same road without distinctive unit numbers.
- **Common False Negatives (Missed Matches):**
  - Completely truncated or empty addresses paired with high phonetic transliteration distance in short acronym names (e.g., 2-3 letter names without suffixes).

---

## 6. Conclusion

By addressing the cross-script nature of Indian business records and diacritic variations in French text through deterministic transliteration, our pipeline resolved the core retrieval bottleneck that cripples standard entity matchers. Combining this with country-partitioned multi-route blocking produced an exceptionally compact candidate set (~11.5 candidates per entity) and a high-precision LightGBM scorer achieving 0.9285 Macro $F_{0.5}$.

---

## Appendix

### A. Code Artefacts
All runnable code is located in `code/business_entity_resolution/`:
- `src/normalize.py`: Language-aware normalization & transliteration engine (`anyascii`, NFKD, Indic state maps, number extractors).
- `src/blocking.py`: `CountryIndex` multi-route blocking and candidate generator.
- `src/features.py`: 21-dimensional pairwise feature extraction.
- `src/train.py`: Model training and $F_{0.5}$ threshold calibration (`python -m src.train`).
- `src/predict.py`: Full inference generating `candidate_pairs.tsv` and `matching_results.tsv` (`python -m src.predict`).
- `requirements.txt`: Pinned dependencies (`lightgbm`, `rapidfuzz`, `anyascii`, `scikit-learn`, `joblib`, etc.).

### B. Additional Results & Validation
The output format was locally verified against the official competition validator (`utils/validate_submission.py`), ensuring strict tab separation, valid S2/S3 ID prefixes, absence of duplicate rows or intra-list duplicate IDs, and complete Source 1 entity coverage.
