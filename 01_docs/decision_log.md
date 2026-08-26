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
