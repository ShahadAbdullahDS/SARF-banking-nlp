# Decision Log

| Date       | Decision                                               | Why                                       | Owner |
|------------|--------------------------------------------------------|-------------------------------------------|---|
| 2026-08-25 | Use AraBERTv2-base as the main model.                  | It is the main model for the CPT study.   | A |
| 2026-08-25 | Use MSA train/dev and a frozen Saudi evaluation split. | Saudi data must not be used for tuning.   | A |
| 2026-08-25 | Run E0, E1 (3K), E2 (6K), E3 (12K), and EB.            | This is the approved experimental design. | A |
| 2026-08-25 | Use 3 seeds per experiment.                            | To report mean ± standard deviation.      | A |
| 2026-08-25 | Keep the scope to intent classification only.          | No chatbot, RAG, or extra dialects.       | A |
| 2026-08-25 | Use the verified processed MSA files: 10,732 train rows and 1,229 validation rows; retain the Saudi test as a frozen 3,580-row evaluation split. | Audit confirmed 77 matching labels across all MSA and Saudi files; PAL is excluded from the project pipeline. | A |
| 2026-08-25 | Accept the synthetic Saudi pilot with revisions; keep all 120 pilot rows as raw audit artifacts only and exclude them from the final CPT corpus. | Manual review passed language, privacy, genre, and diversity checks; prompts were revised for ambiguous general wording, unique IDs, and generic banking content. Production generation must use API batches with local metadata and automatic checks. | A |
| 2026-08-27 | Close the initial EDA and data-quality audit; retain all approved processed splits unchanged. | The SARF EDA verified the audited split counts, the shared 77-intent label set, no missing labels or empty texts, duplicate status by split, class-support variation, and text-length patterns. The frozen Saudi split remains unchanged and is reserved for planned final evaluation. The EDA findings support Macro-F1 as the primary metric. | A |

## [2026-08-30] Banking validation filter bug fix (named-term false positives)

**Issue:** The banking-pool validation filter (`validate_banking_rows`) was hard-rejecting valid banking texts due to naive substring matching against `BANKING_NAMED_TERMS`. The term "ساب" (intended to catch "بنك ساب") matched inside unrelated words like "الحساب" (account), causing false rejections.

**Fix:**
- Replaced substring (`in`) matching with word-boundary regex matching (`\b` + `re.escape(term)`, `re.UNICODE`) via a new `term_hits()` helper.
- Removed the bare term "ساب" from `BANKING_NAMED_TERMS` (redundant — "بنك ساب" already covers the real bank name).
- Moved the ambiguous term "مدى" (mada card network vs. the common word "extent") from hard-reject to a new soft-flag list `BANKING_SOFT_FLAG_TERMS` — kept in the corpus but flagged in `notes` for review, rather than auto-rejected.

**Verification:** Re-tested against the 4 originally falsely-rejected production texts — all 4 now correctly pass. Full 144-job banking production run proceeded after this fix with no further false positives of this kind observed.

**Impact:** No data loss — fix applied before full-scale production.

---

## [2026-08-30] Latin-character contamination discovered and removed (both pools)

**Issue:** Manual audit surfaced synthetic rows containing stray Latin/ASCII characters embedded in otherwise-Arabic text — two patterns: (1) a literal pool-label leak (`،general` / `،banking` appended to text), and (2) garbled word-level corruption (English fragments/words spliced into Arabic words, e.g. "تكhelا", "قبلDeadline", "tonight"). Root cause not fully isolated — likely a generation-side artifact rather than a deterministic code bug.

**Detection method:** Full-corpus regex scan (`[a-zA-Z]` against the `text` field) on both final pools.

**Scope found:** General: 128 / 12,003 rows (~1.07%). Banking: 14 / 3,036 rows (~0.46%).

**Fix:**
- Contaminated rows removed from both pools' candidate sets before final slicing.
- General: shortfall of 125 rows required 8 additional production jobs (job_id 8000–8007) to restore ≥12,000 clean candidates.
- Banking: sufficient surplus existed (3,022 clean) — no new production needed.
- Final `general_3k/6k/12k` and `banking_3k` re-sliced from cleaned pools using the same fixed-seed stable-hash method, preserving nesting (3k ⊂ 6k ⊂ 12k).
- Saudi frozen-test leakage check re-run on cleaned files: 0 exact / 0 normalized matches, both pools (unchanged).
- Contaminated rows logged: `general_latin_contamination_log.csv`, `banking_latin_contamination_log.csv`.

**Impact on manual audit:** 8 of the 300 general-audit rows fell out of the final 12,000 — 5 were already independently flagged Fail for this exact issue (confirms manual audit accuracy); 3 were Pass-rated rows excluded only by normal hash-cutoff reshuffling (no quality concern). The banking 60-row sample was regenerated from the cleaned pool before audit began, so unaffected.

**Recommendation:** Add this Latin-character check as a standard automated validation step for any future production runs.

| 2026-08-31 | Issue SARF Synthetic Saudi Arabic Corpus v2 after targeted post-audit correction. | The first manual general audit identified rows marked Fail or Unsure that remained in v1 final selection. After recalibration, unresolved flagged rows were excluded and replaced from the existing clean reserve pool without new generation. General nested subsets, metadata, leakage checks, hashes, and freeze documentation were regenerated for v2. | B / A |
