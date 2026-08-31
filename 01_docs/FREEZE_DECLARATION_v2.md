# Corpus Freeze Declaration — v2

**Corpus:** SARF Synthetic Saudi Arabic Corpus (v2)
**Supersedes:** v1 (frozen 2026-08-30) — **for model training purposes only**
**Frozen by:** Person B (SARF)
**Freeze date (UTC):** 2026-08-31
**Reason for v2:** Targeted post-audit correction. The original 300-row general manual audit identified rows marked Fail or Unsure that remained inside the v1 final selection. After a formal recalibration pass (documented below), 8 rows retained a genuine, specific defect and were excluded and replaced from the existing clean reserve pool. No new Gemini API generation was used.

## Frozen Artifacts (v2)

| File | Rows | SHA-256 |
|---|---|---|
| `general_3k_final_v2.csv` | 3,000 | see `corpus_manifest_v2.json` |
| `general_6k_final_v2.csv` | 6,000 | see `corpus_manifest_v2.json` |
| `general_12k_final_v2.csv` | 12,000 | see `corpus_manifest_v2.json` |
| `banking_3k_final_v2.csv` | 3,000 | see `corpus_manifest_v2.json` |

## Conditions Satisfied Prior to v2 Freeze

- [x] `general_audit_review_needed.csv` (32 rows) re-reviewed against the official recalibration standard; results saved to `general_audit_recalibrated.csv`.
- [x] 8 rows with a genuine, specific defect excluded from the general candidate pool; 8 replacement rows selected via the same fixed-seed stable-hash method (`sarf_general_corpus_v1`) and individually manually reviewed (all Pass).
- [x] Nesting property re-verified: `general_3k_v2 ⊆ general_6k_v2 ⊆ general_12k_v2`.
- [x] File-level checks on v2 (both pools): 0 blank texts, 0 duplicate IDs, 0 duplicate texts, 0 Latin-character rows.
- [x] Saudi frozen-test leakage check re-run on v2: 0 exact / 0 normalized matches, both pools.
- [x] `audit_status` / `cleaning_status` metadata corrected on every row (no row remains marked "pending").
- [x] `general_audit_correction_log.csv` documents every excluded row and its replacement, plus the 3 rows dropped from v1 by stable-hash chance (not a quality decision).
- [x] SHA-256 hashes recorded in `corpus_manifest_v2.json`.

## Status

**No Fail or Unsure row remains in any v2 final file.**

v1 files (`general_3k_final.csv`, `general_6k_final.csv`, `general_12k_final.csv`, `banking_3k_final.csv`, `corpus_manifest.json`, `FREEZE_DECLARATION.md`) are retained unmodified for provenance and are **not** to be used for training.

**Status: FROZEN — v2 (supersedes v1 for training)**
