# Weekly Status

## Week 1

| Workstream | Owner | Status | Evidence / Link | Blocker | Next step |
|---|---|---|---|---|---|
| Data audit | A | Not started | — | — | Download and inspect ArBanking77 files. |
| Synthetic corpus pilot | B | Not started | — | — | Draft prompts and generate a small pilot. |
| Baselines | C | Not started | — | — | Prepare SVM/TextCNN notebook structure. |
| AraBERT setup | D | Not started | — | — | Create Colab smoke-test notebook. |
| Documentation | A | In progress | GitHub files created | — | Share kickoff guide with team. |


## Status Update — 2026-08-27

### Completed
- Project identity updated to **SARF — صَرْف**.
- The final EDA and data-audit notebook was completed and uploaded to `04_notebooks/00_eda_and_data_audit.ipynb`.
- EDA outputs were saved in Drive: audit tables in `04_audits/eda/` and figures in `08_figures/eda/`.
- The approved processed splits were verified: MSA train (10,732), MSA validation (1,229), MSA test (3,574), and frozen Saudi test (3,580).
- All splits use the same 77 intent labels, with no missing labels or empty texts.
- No duplicate texts were found in the MSA splits. The frozen Saudi test contains 38 duplicate rows representing 37 repeated text values and remains unchanged.
- The MSA training distribution varies across intents (31–291 examples per intent), so Macro-F1 remains the primary evaluation metric.
- Saudi queries are slightly shorter than MSA queries; this is documented as descriptive EDA only.

### Current State
- Prompts, topic plans, schema, pilot audit, EDA, AraBERT environment smoke test, shared metrics, and SVM reference skeleton are ready.
- No synthetic production corpus has been generated yet.
- No baseline or AraBERT performance result has been reported yet.

### Next Step
- Person B starts the Gemini API generation notebook with small smoke batches only.
- Person C runs the SVM reference baseline on MSA train → MSA validation only.
- Person D runs a small AraBERT fine-tuning smoke test before E0.
- Person A reviews evidence and records only real decisions or blockers.

### Non-Negotiable Rules
- The frozen Saudi test split is never used for prompt changes, hyperparameter selection, checkpoint selection, or other tuning decisions.
- Production synthetic data must be generated through the API workflow with local metadata and validation logs; pilot rows do not enter the final corpus.
- Code and notebooks are versioned in GitHub. Data, run outputs, checkpoints, metrics, and figures are stored in Drive.
