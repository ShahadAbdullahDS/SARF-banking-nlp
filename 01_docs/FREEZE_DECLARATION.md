# Corpus Freeze Declaration

**Corpus:** SARF Synthetic Saudi Arabic Corpus (v1)
**Frozen by:** Person B (SARF)
**Freeze date (UTC):** 2026-08-30

## Declaration

This declares the following files as the final, frozen deliverable for the general and banking synthetic corpora. No further edits, regenerations, or row-level changes are to be made to these files without creating a new versioned release (v2+) and a corresponding new entry in `decision_log.md`.

## Frozen Artifacts

| File | Rows | SHA-256 |
|---|---|---|
| `general_3k_final.csv` | 3,000 | see `corpus_manifest.json` |
| `general_6k_final.csv` | 6,000 | see `corpus_manifest.json` |
| `general_12k_final.csv` | 12,000 | see `corpus_manifest.json` |
| `banking_3k_final.csv` | 3,000 | see `corpus_manifest.json` |

Authoritative hashes and row counts are recorded in `06_final_audit/corpus_manifest.json`, generated at freeze time. Any future user of these files should verify file hashes against this manifest before use, to confirm the file has not been altered since freeze.

## Conditions Satisfied Prior to Freeze

- [x] Generation completed against pre-approved, unmodified prompts/schema/topic plan.
- [x] Cross-batch deduplication applied (both pools).
- [x] Banking-domain validation filter applied and bug-fixed (see `decision_log.md`, 2026-08-30).
- [x] Latin-character contamination detected, removed, and shortfall backfilled (see `decision_log.md`, 2026-08-30).
- [x] Manual quality audit completed: general 300/12,000 (2.5%) sampled, 89.3% pass rate; banking 60/3,000 (2%) sampled, 100% pass rate.
- [x] Zero leakage confirmed against frozen Saudi test set (`saudi_test_frozen_v1.csv`), exact and normalized matching, both pools.
- [x] Corpus card (`CORPUS_CARD.md`) published documenting purpose, composition, pipeline, and known limitations.

## Known Residual Risk

Manual audit is sample-based, not exhaustive; an estimated defect rate (~10.7% for general, based on sample) may exist among unaudited rows. This is disclosed in `CORPUS_CARD.md` and accepted as within tolerance for this project's purposes.

**Status: FROZEN — v1**
