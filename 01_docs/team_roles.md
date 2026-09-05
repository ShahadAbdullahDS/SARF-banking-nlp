# Team Roles and Handoff Rules

## Team Roles

| Role | Owner | Main responsibility | Current handoff status |
| --- | --- | --- | --- |
| Project Lead | A | Research question, scope, data integrity, decisions, audits, documentation review, and final approval. | Receives the v2 corpus handoff and approves the transition to model implementation. |
| Synthetic Data Owner | B | Prompts, generation batches, automatic cleaning, manual sample audit, corpus manifests, and freeze documentation. | Complete: v2 is frozen and supersedes v1 for training. |
| Baselines and Analysis Owner | C | SVM, TextCNN, shared metrics, baseline figures, validation logs, and error analysis. | Next owner to begin model implementation after the repository structure is cleaned. |
| AraBERT Owner | D | AraBERT setup, E0, CPT runs, checkpoints, run configurations, GPU scheduling, and model artifacts. | Setup stage; begin with the E0 smoke test and schedule the CPT runs. |

## Source-of-Truth Rules

- The active synthetic training release is **v2 only**.

- The approved files are `general_3k_final_v2.csv`, `general_6k_final_v2.csv`, `general_12k_final_v2.csv`, and `banking_3k_final_v2.csv`.

- v1 files and draft files are retained for provenance in the archive and must not be used for training.

- The v2 manifest and freeze declaration are the source of truth for file hashes, row counts, correction history, and freeze status.

- The Saudi test set is frozen and may be used only once for final evaluation after training and checkpoint decisions are complete.

## Handoff Requirements

Every completed task must include the following information in the relevant GitHub or Drive record:

| Required item | Description |
| --- | --- |
| Input | Exact file name, version, path, or link used by the owner. |
| Output | Exact output file name, version, path, or link produced by the owner. |
| Reproduction | Command, notebook, configuration, seed, or ordered steps needed to reproduce the output. |
| Integrity checks | At least two checks completed, with their results recorded. |
| Experimental status | Whether the result is a draft, smoke test, final run, or frozen artifact. |
| Open issues | Any blocker, limitation, failed run, or decision still requiring approval. |

## Model Handoff Rules

- C and D must use the same approved MSA train and validation files.

- C must record the SVM and TextCNN preprocessing and metrics contract before comparing results.

- D must use the same AraBERT base checkpoint across E0, E1, E2, E3, and EB.

- Each AraBERT condition must use the three approved random seeds.

- MSA validation may be used for checkpoint selection; the Saudi test may not.

- No result may be described as a final Saudi evaluation result until all condition-level training and checkpoint decisions are fixed.

## Scope Rule

Do not add a chatbot, RAG system, extra dialects, new datasets, or new experiments without written approval from A and an entry in `decision_log.md`.
