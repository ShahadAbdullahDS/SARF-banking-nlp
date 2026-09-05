# Corpus Card: SARF Synthetic Saudi Arabic Corpus (v1)

## Purpose
Synthetic Saudi Arabic text corpus for continued pre-training, supporting cross-dialect Saudi banking intent classification. Two pools: a general (non-financial) pool for broad Saudi-dialect language exposure, and a banking-domain pool for intent-relevant vocabulary and phrasing.

## Composition
| Pool | Final size | Nested subsets |
|---|---|---|
| General | 12,000 | 3,000 ⊂ 6,000 ⊂ 12,000 (fixed-seed stable-hash slicing, reproducible regardless of build order) |
| Banking | 3,000 | — |

Each row: `id, text, pool, topic, genre, length_bucket`. Topics, genres, and length buckets follow the project's pre-approved topic plan. Generated via the Gemini API (`gemini-3.5-flash-lite`) using frozen prompt templates and schema.

## Generation & Cleaning Pipeline
1. Batch production against frozen prompts/schema (504 general batches, 146 banking batches, plus targeted retries and top-ups for API failures and shortfalls).
2. Cross-batch deduplication via normalized-text matching (whitespace-collapsed comparison).
3. Schema/domain validation: banking pool checked for named-entity/product leakage (word-boundary regex matching against a curated term list; ambiguous terms soft-flagged rather than hard-rejected).
4. Latin-character contamination scan and removal (see decision_log.md, 2026-08-30 entries) — two known generation artifacts (pool-label leakage, garbled word splicing) detected and removed from both pools; general pool topped up with new production after removal to restore 12,000.
5. Reproducible final selection via fixed-seed SHA-256 stable-hash sort + slice (seeds: `sarf_general_corpus_v1`, `sarf_banking_corpus_v1`).

## Quality Assurance
- **Manual audit** (stratified by topic × genre × length_bucket, fixed random seed): 300 general rows (2.5% of pool) and 60 banking rows (2% of pool), reviewed by Person B against a calibrated Pass/Fail/Unsure standard (Pass = understandable, natural enough, logical, on-topic even with occasional plain MSA usage; Fail reserved for broken/incomprehensible phrasing, logical contradiction, nonsensical meaning, off-topic content, banking-term leakage in the general pool, PII, or duplication; Unsure = genuine unresolved judgment call, recorded but not auto-rejected).
  - General: 268 Pass / 30 Fail / 2 Unsure (89.3% pass rate).
  - Banking: 60 Pass / 0 Fail / 0 Unsure (100% pass rate).
- **Test-set leakage check**: exact-match and normalized-match comparison against the frozen Saudi test set (`saudi_test_frozen_v1.csv`, 3,580 rows). Result: 0 exact / 0 normalized matches in either pool.

## Known Limitations
- The manual audit is sample-based (2–2.5% coverage), not exhaustive; the general pool's ~10.7% Fail/Unsure rate observed in-sample should be treated as an estimated defect-rate ceiling for the full 12,000, not a guarantee that all remaining rows are clean.
- The Latin-character contamination check catches ASCII Latin letters only; it will not catch every possible generation artifact (e.g., non-Latin stray characters, subtler word-level errors not involving foreign script).
- Root cause of the Latin-character contamination was not fully isolated (suspected model-generation artifact); no prompt-level fix was applied, only post-hoc detection and removal.
- All content is synthetic (LLM-generated), not naturally occurring text; it may under-represent some real-world register or topic variation despite following the approved topic plan.

## Freeze Status
Frozen 2026-08-30. See `corpus_manifest.json` for file hashes (SHA-256), row counts, and audit/leakage summary.
