# Weekly Status

| Field | Current status |
| --- | --- |
| Project | SARF — صَرْف: Cross-Dialect Banking Intent Classification |
| Report date | 2026-09-05 |
| Current phase | Model Implementation |
| Completed phase | Data Foundation: EDA, synthetic corpus generation, cleaning, audit, and freeze |
| Active corpus release | v2 only |
| Superseded release | v1 — provenance only; not for training |
| Main blocker | GitHub and Drive folder organization must be cleaned before Person C starts |
| Saudi test policy | Frozen; final evaluation only |

## Workstream Status

| Workstream | Owner | Status | Evidence / Link | Blocker | Next step |
| --- | --- | --- | --- | --- | --- |
| Project context and documentation | A | In progress | `00_project_context/`, `01_docs/` | GitHub folders and document versions are being reorganized | Move notebooks, flatten the duplicated config folder, archive v1 documents, and keep `decision_log.md` as one table |
| MSA and Saudi data audit | A | Complete | EDA notebook; processed data notes | None | Keep the Saudi test frozen and use it only for final evaluation |
| Synthetic corpus generation and audit | B | Complete — v2 frozen | `CORPUS_CARD_v2.md`, `FREEZE_DECLARATION_v2.md`, `corpus_manifest_v2.json` | None for training data | Use v2 only; keep v1 and drafts in the archive for provenance |
| SVM baseline | C | Not started | `04_src/baseline_svm.py` reference skeleton | Training run and result logging remain to be done | Run the MSA baseline and save reproducible metrics |
| TextCNN baseline | C | Not started | `02_configs/experiment_plan.md` | Implementation remains to be done | Implement and evaluate using the same MSA protocol |
| AraBERT E0 | D | Setup stage | AraBERT evidence and experiment plan | GPU schedule and run configuration remain to be finalized | Run the E0 smoke test and finalize the schedule |
| AraBERT CPT conditions | D | Not started | `02_configs/experiment_plan.md` | E0 setup must be validated first | Run E1, E2, E3, and EB with three approved seeds each |
| Shared metrics and evaluation contract | C / D | Prepared | `04_src/metrics.py` | Final result files do not exist yet | Use Macro-F1 as primary, with weighted F1 and accuracy as secondary metrics |
| Presentation and proposal | A | Complete | Final proposal deck and team script | Rehearsal may still be needed | Keep presentation artifacts separate from implementation artifacts |

## Verified Data Foundation

| Item | Verified status |
| --- | --- |
| MSA training examples | 10,732 |
| MSA validation examples | 1,229 |
| Saudi frozen test examples | 3,580 |
| Shared intent labels | 77 |
| MSA missing or empty values | None reported in the approved audit |
| Synthetic general v2 corpora | 3K, 6K, and 12K |
| Synthetic banking v2 corpus | 3K |
| General subset nesting | `3K ⊆ 6K ⊆ 12K` |
| v2 blank texts | 0 |
| v2 duplicate IDs | 0 |
| v2 duplicate texts | 0 |
| v2 Latin-character rows | 0 |
| v2 Saudi-test leakage | 0 exact and 0 normalized matches |
| Manual audit coverage | Sample-based, not exhaustive |

## Approved Experimental Conditions

| Condition | Saudi CPT input | Supervised training | Validation | Final evaluation |
| --- | --- | --- | --- | --- |
| E0 | None | Same labelled MSA banking training set | Same MSA validation set | Frozen Saudi test only at the end |
| E1 | `general_3k_final_v2.csv` | Same labelled MSA banking training set | Same MSA validation set | Frozen Saudi test only at the end |
| E2 | `general_6k_final_v2.csv` | Same labelled MSA banking training set | Same MSA validation set | Frozen Saudi test only at the end |
| E3 | `general_12k_final_v2.csv` | Same labelled MSA banking training set | Same MSA validation set | Frozen Saudi test only at the end |
| EB | `general_12k_final_v2.csv` + `banking_3k_final_v2.csv` | Same labelled MSA banking training set | Same MSA validation set | Frozen Saudi test only at the end |

## Non-Negotiable Rules

| Rule | Requirement |
| --- | --- |
| Training release | Use v2 only; do not use v1 or draft files |
| Base checkpoint | Start all AraBERT conditions from the same base checkpoint |
| Fine-tuning protocol | Keep the MSA fine-tuning and validation protocol the same across conditions |
| Random seeds | Use the three approved seeds for every AraBERT condition |
| Checkpoint selection | Use MSA validation only |
| Saudi test | Do not use for CPT, prompt changes, hyperparameter selection, checkpoint selection, debugging, or tuning |
| Final evaluation | Open the Saudi test only after all condition-level decisions are fixed |
| Synthetic data | Use for CPT only; do not treat it as labelled supervised data |
| Storage | Store code and notebooks in GitHub; store large datasets, checkpoints, outputs, metrics, and figures in Drive |
| Scope | Do not add a chatbot, RAG system, extra dialects, new datasets, or new experiments without approval from A |

## Next Milestones

| Order | Milestone | Owner | Completion condition |
| --- | --- | --- | --- |
| 1 | Clean the GitHub and Drive structure | A / team | v2 is active; v1 and drafts are in archive; duplicate folders are removed |
| 2 | Run the SVM baseline | C | MSA train and validation metrics are logged reproducibly |
| 3 | Implement and run TextCNN | C | MSA train and validation metrics are logged reproducibly |
| 4 | Run the AraBERT E0 smoke test | D | Configuration, runtime, checkpoint, and validation procedure are verified |
| 5 | Finalize the GPU schedule | D | The 15 AraBERT runs have assigned seeds, conditions, and storage paths |
| 6 | Run E1, E2, E3, and EB | D | All approved runs complete with logs and checkpoints |
| 7 | Select checkpoints using MSA validation | C / D | One fixed checkpoint is selected for every condition before Saudi evaluation |
| 8 | Evaluate on the frozen Saudi test | C / D | Final Macro-F1, weighted F1, and accuracy are reported once |
