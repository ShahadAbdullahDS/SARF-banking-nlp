# Weekly Status

## Week 1

| Completed workstream | Owner | Completed work | Evidence / Link |
| --- | --- | --- | --- |
| No completed item recorded | — | The original Week 1 table recorded the initial plan; no workstream was marked complete in that week. | — |

## Status Update — 2026-08-27

| Workstream | Owner | Completed work | Verification / Evidence |
| --- | --- | --- | --- |
| Project identity and scope | A | Updated the project identity to **SARF — صَرْف** and fixed the scope to banking intent classification. | `decision_log.md`; scope limited to intent classification. |
| MSA and Saudi data audit | A | Inspected the processed MSA and Saudi splits. | Verified the approved split counts, 77 shared intent labels, no missing labels, and no empty texts. |
| EDA | A | Completed the initial EDA for split sizes, intent support, duplicates, and text-length patterns. | `00_eda_and_data_audit.ipynb`; Macro-F1 selected as the primary metric. |
| Synthetic-data preparation | B | Prepared the prompts, topic plans, schema, pilot audit, and production workflow. | `02_configs/`; generation notebook; pilot rows excluded from the final CPT corpus. |
| Baseline preparation | C | Prepared the SVM reference skeleton and shared metrics contract. | `baseline_svm.py`; `metrics.py`. |
| AraBERT preparation | D | Prepared the AraBERT experiment materials and CPT evidence. | `experiment_plan.md`; AraBERT notes; E0, E1, E2, E3, and EB conditions with three seeds approved. |

## Current Status — 2026-09-05

| Workstream | Owner | Current status | Current evidence |
| --- | --- | --- | --- |
| Synthetic corpus | B | Complete — v2 frozen | `CORPUS_CARD_v2.md`; `FREEZE_DECLARATION_v2.md`; `corpus_manifest_v2.json`. |
| v2 quality and leakage checks | A / B | Complete | 0 blank texts, 0 duplicate IDs, 0 duplicate texts, 0 Latin-character rows, and 0 exact or normalized Saudi-test matches. |
| Presentation | A / Team | Complete | Final proposal deck and speaking script. |
| GitHub organization | A / Team | In progress | Duplicate numbering, nested `02_configs`, notebooks outside `03_notebooks`, and mixed document versions are being cleaned. |
| Drive organization | A / B | In progress | v1, v2, and draft files are being separated so v2 is the active release. |
| Person C model implementation | C | Not started | SVM skeleton and metrics contract are ready. |
| Person D AraBERT implementation | D | Not started | AraBERT experiment plan and setup materials are ready. |

## Next Milestones — This Week

| Order | Milestone | Owner | Completion condition |
| --- | --- | --- | --- |
| 1 | Finish GitHub and Drive organization | A / Team | v2 is clearly active, v1 and drafts are archived, duplicate folders are removed, and both notebooks are under `03_notebooks/`. |
| 2 | Complete the Person C handoff | A / B / C | C receives the exact v2 file paths and the training-use rule. |
| 3 | Run the SVM baseline | C | MSA training and validation metrics are logged using the shared metrics contract. |
| 4 | Begin TextCNN implementation | C | The model structure and preprocessing protocol are documented and ready for the first run. |
| 5 | Run the AraBERT E0 smoke test | D | The configuration, runtime, checkpoint saving, and MSA validation procedure are verified. |
| 6 | Finalize the AraBERT run schedule | D | The approved conditions and three seeds have assigned run IDs and storage paths. |
