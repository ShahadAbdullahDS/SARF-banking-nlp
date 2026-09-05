# Weekly Status

## Week 1 — Kick-off and Initial Planning

| Workstream | Owner | Status | Evidence / Link | Blocker | Next step |
| --- | --- | --- | --- | --- | --- |
| Data audit | A | Not started | — | — | Download and inspect ArBanking77 files. |
| Synthetic corpus pilot | B | Not started | — | — | Draft prompts and generate a small pilot. |
| Baselines | C | Not started | — | — | Prepare SVM/TextCNN notebook structure. |
| AraBERT setup | D | Not started | — | — | Create Colab smoke-test notebook. |
| Documentation | A | In progress | GitHub files created | — | Share kickoff guide with team. |

## Week 2 — Data Audit, EDA, and Experiment Foundation

| Workstream | Owner | Completed work | Verification / Approval recorded that week | Evidence / Link | Remaining next step |
| --- | --- | --- | --- | --- | --- |
| Project identity and scope | A | Updated the project identity to **SARF — صَرْف** and fixed the project scope to banking intent classification. | Scope decision recorded: no chatbot, RAG, extra dialects, or unrelated tasks. | `decision_log.md` | Keep future changes within the approved scope. |
| MSA and Saudi data audit | A | Inspected the processed MSA and Saudi splits and checked the shared intent structure. | Verified 77 matching intent labels, approved split counts, no missing labels, and no empty texts. | EDA notebook; data manifest | Preserve the Saudi split for final evaluation only. |
| EDA | A | Completed the initial EDA covering split sizes, intent support, duplicates, and text-length patterns. | Macro-F1 was selected as the primary metric because intent support varies across classes. | `00_eda_and_data_audit.ipynb`; EDA outputs | Use the EDA findings in the experiment report. |
| Experimental design | A / D | Defined E0, E1, E2, E3, and EB and the three-seed AraBERT plan. | The controlled conditions and three-seed design were approved and recorded. | `experiment_plan.md`; `decision_log.md` | Prepare the CPT and fine-tuning runs. |
| Project infrastructure | C / D | Prepared the SVM reference skeleton, shared metrics contract, and AraBERT setup materials. | The implementation artifacts were reviewed as ready for the next phase. | `baseline_svm.py`; `metrics.py`; AraBERT notes | Begin model smoke tests after the data foundation is complete. |

## Week 3 — Synthetic Corpus Generation and Initial Freeze

| Workstream | Owner | Completed work | Verification / Approval recorded that week | Evidence / Link | Remaining next step |
| --- | --- | --- | --- | --- | --- |
| Synthetic data generation | B | Generated the general Saudi-style pool and the Saudi banking pool using the approved prompts, schema, and topic plans. | The production pools reached the approved target sizes after cleaning and top-up work. | Synthetic-data Drive folder; generation notebook | Complete the final audit and release documentation. |
| Banking validation | B | Fixed the false-positive named-term filter and continued banking generation. | The affected rows were re-tested and the corrected validation logic was recorded. | `decision_log.md`; cleaning logs | Use the corrected filter for future production runs. |
| Latin-character cleaning | B | Detected and removed Latin-character contamination from both pools and re-sliced the cleaned candidates. | Full-corpus scan completed; contaminated rows and replacement production were logged. | `04_cleaning_logs/` | Re-run final integrity and leakage checks. |
| v1 corpus release | B | Created the v1 general and banking final files with the required corpus sizes and nested general subsets. | v1 integrity and Saudi-test leakage checks were completed. | v1 manifest and freeze declaration | Recalibrate the manually flagged general rows before training. |
| Manual audit | B | Completed stratified manual samples for general and banking text. | The audit results and limitations were recorded; the audit was recognized as sample-based, not exhaustive. | Audit sample files | Resolve genuine defects before the training freeze. |

## Week 4 — v2 Correction, Final Data Freeze, and Presentation Completion

| Workstream | Owner | Completed work | Verification / Approval recorded that week | Evidence / Link | Remaining next step |
| --- | --- | --- | --- | --- | --- |
| Targeted v2 correction | B | Re-reviewed the originally flagged general rows, removed the eight genuine defective rows, and replaced them from the clean reserve pool. | All eight replacements were individually reviewed and marked Pass; no new API generation was used. | `general_audit_recalibrated.csv`; `general_audit_correction_log.csv` | Use only v2 for model training. |
| v2 metadata and freeze | B | Corrected row-level metadata and issued the v2 general and banking files. | No v2 row remains marked `pending`; v2 manifest hashes and row counts were checked. | `CORPUS_CARD_v2.md`; `FREEZE_DECLARATION_v2.md`; `corpus_manifest_v2.json` | Archive v1 and draft files away from the active training folder. |
| v2 integrity and leakage | B / A | Re-checked blank texts, duplicate IDs, duplicate texts, Latin-character rows, nested subsets, and Saudi-test leakage. | v2 passed all checks with 0 exact and 0 normalized Saudi-test matches in both pools. | v2 manifest; leakage report | Record the v2 release as the only training release. |
| Proposal presentation | A / Team | Completed the proposal presentation and the short speaking script. | Presentation structure was aligned with the required separation between Solution Architecture and Implementation Plan. | Final presentation and script files | Rehearse if needed; keep presentation artifacts separate from implementation files. |
| Current handoff | A / B / C / D | Closed the synthetic-data task and prepared the transition to model implementation. | v2 is ready for training; C and D can begin after the repository folders are cleaned. | `team_roles.md`; `weekly_status.md` | Start the model implementation phase. |

## Current Week — Repository Cleanup Before Model Implementation

| Workstream | Owner | Status | Current issue | Next step |
| --- | --- | --- | --- | --- |
| GitHub structure | A / Team | In progress | Duplicate numbering, nested `02_configs`, notebooks outside `03_notebooks`, and mixed document versions. | Move files to the agreed folders and archive v1 documents. |
| Synthetic-data Drive structure | A / B | In progress | v1, v2, and draft files are mixed in `05_final_corpora/` and `06_final_audit/`. | Keep v2 active; move v1 and drafts to `archive/v1/`. |
| Decision log | A | In progress | Some entries were converted from a table into paragraphs. | Replace with the corrected all-table version. |
| Project context | A | In progress | `00_project_context/` contains only a basic README. | Add the project overview, scope, and links to the active documents. |
| Person C handoff | C | Waiting for repository cleanup | C must not accidentally use v1 or draft files. | Start with SVM and TextCNN after the v2 paths are confirmed. |
| Person D handoff | D | Waiting for repository cleanup | E0 configuration and GPU schedule still need to be fixed. | Run the E0 smoke test, then schedule the AraBERT runs. |

## Next Milestones

| Order | Milestone | Owner | Completion condition |
| --- | --- | --- | --- |
| 1 | Finish GitHub and Drive organization | A / Team | v2 is active; v1 and drafts are archived; duplicate folders are removed; notebooks are under `03_notebooks/`. |
| 2 | Start the SVM baseline | C | MSA training and validation metrics are logged with the approved metrics contract. |
| 3 | Implement and run TextCNN | C | MSA training and validation metrics are logged reproducibly. |
| 4 | Run the AraBERT E0 smoke test | D | Configuration, runtime, checkpoint saving, and MSA validation are verified. |
| 5 | Finalize the GPU schedule | D | All 15 AraBERT runs have assigned conditions, seeds, and storage paths. |
| 6 | Run E1, E2, E3, and EB | D | All approved CPT runs complete with logs and checkpoints. |
| 7 | Select checkpoints using MSA validation | C / D | One fixed checkpoint is selected for every condition before Saudi evaluation. |
| 8 | Evaluate on the frozen Saudi test | C / D | Final Macro-F1, weighted F1, and accuracy are reported once after all decisions are fixed. |
