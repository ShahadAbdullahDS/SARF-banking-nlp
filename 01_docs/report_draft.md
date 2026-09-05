# SARF — Project Report Draft

> This is a living report. Add only completed work and verified results. Do not add model-performance claims until the corresponding runs are complete and logged.

## 1. Problem and Motivation

Saudi banking customers may express the same intent in Saudi-style Arabic rather than Modern Standard Arabic (MSA). A banking intent classifier trained mainly on labelled MSA queries may therefore face a dialectal language gap when it processes Saudi-style wording. Labelled Saudi banking data are limited and expensive to create, so the project studies whether unlabelled Saudi-style text can improve adaptation before the supervised banking task.

The practical task is banking intent classification: assigning a customer query to one of 77 banking intents. The project focuses on language adaptation and does not include a chatbot, retrieval-augmented generation, or additional dialects.

## 2. Research Question and Scope

The main question is:

> To what extent does increasing synthetic Saudi Arabic exposure through Continued Pre-Training affect Macro-F1 when an AraBERT-based classifier is fine-tuned on labelled MSA banking queries and evaluated on a frozen Saudi test set?

The controlled AraBERT conditions are E0, E1, E2, E3, and EB. E0 receives no Saudi CPT exposure. E1, E2, and E3 use general Saudi-style corpora of 3K, 6K, and 12K texts. EB tests the approved general-plus-banking exposure setup using the 12K general corpus and the 3K banking corpus.

The project also includes two non-AraBERT baselines: TF-IDF with SVM and TextCNN. These baselines provide reference points for the intent-classification task; they are not CPT conditions.

## 3. Data Sources and Data Audit

The labelled supervised data come from the verified MSA banking splits associated with the ArBanking77 task. The approved counts are 10,732 MSA training examples and 1,229 MSA validation examples, with 77 intent labels. The Saudi evaluation split contains 3,580 rows and remains frozen for final evaluation only.

The initial EDA verified split sizes, shared labels, missing and empty values, duplicate status, class-support variation, and text-length patterns. The MSA training distribution varies across intents, so Macro-F1 is the primary metric. Weighted F1 and accuracy are secondary metrics.

The synthetic data are separate unlabelled corpora created for CPT. They are not labelled banking training data and must not be treated as an additional evaluation set.

## 4. Synthetic Corpus Generation and Quality Audit

The active synthetic release is **SARF Synthetic Saudi Arabic Corpus v2**. It contains the following frozen files:

| Corpus | Size | Intended use |
| --- | --- | --- |
| `general_3k_final_v2.csv` | 3,000 | E1 general Saudi-style CPT exposure |
| `general_6k_final_v2.csv` | 6,000 | E2 general Saudi-style CPT exposure |
| `general_12k_final_v2.csv` | 12,000 | E3 general Saudi-style CPT exposure and EB general component |
| `banking_3k_final_v2.csv` | 3,000 | EB banking-domain CPT component |

The general subsets are nested using the approved reproducible selection method: 3K is a subset of 6K, and 6K is a subset of 12K. The general pool is non-financial for the isolated dialect-exposure conditions. The banking pool is separate so the experiment can distinguish the effect of general Saudi-style exposure from the additional effect of banking-domain content.

The corpus was generated as synthetic Arabic text through the Gemini API using the approved prompts, schema, and topic plans. Version v2 was produced through targeted post-audit correction using clean reserve rows; no new generation was used for the v2 replacements.

The v2 release passed the documented integrity checks: correct row counts, matching SHA-256 hashes, no blank texts, no duplicate IDs, no duplicate texts, no Latin-character rows, correct nested subsets, and zero exact or normalized matches against the frozen Saudi test set. Manual review remains sample-based rather than exhaustive; the v2 metadata distinguishes manually reviewed rows from rows that were not individually sampled.

The v1 files and draft files remain available for provenance only and must not be used for training.

## 5. Experimental Setup

The experiment has two stages. First, AraBERT receives the assigned unlabelled Saudi-style corpus through Continued Pre-Training with a masked-language-model objective. E0 skips this stage. Second, every AraBERT condition is fine-tuned on the same labelled MSA banking training data.

The MSA validation set is used to select the best saved checkpoint within each condition. The Saudi test set is not used during CPT, fine-tuning decisions, prompt changes, hyperparameter selection, or checkpoint selection. After all condition-level decisions are fixed, the final models are evaluated once on the frozen Saudi test set.

The AraBERT conditions start from the same base checkpoint, follow the same MSA fine-tuning and validation protocol, and use three approved random seeds per condition. The baselines use the same labelled MSA task and the shared evaluation contract but do not receive Saudi CPT.

## 6. Results

This section will be completed after the baseline and AraBERT runs finish. Report Macro-F1 as the primary metric, followed by weighted F1 and accuracy. For the AraBERT conditions, report the mean and standard deviation across the three seeds, together with the fixed final evaluation results on the Saudi test set.

No model-performance result should be added to this section before it is produced by a recorded run and checked against the experiment configuration.

## 7. Error Analysis and Interpretation

This section will be completed after the final predictions are generated. The analysis should compare representative errors across the baselines and AraBERT conditions, with special attention to Saudi-style wording, intent confusions, and whether errors change as the CPT corpus size or domain changes.

## 8. Limitations and Ethical Notes

The synthetic corpora are LLM-generated and may not represent the full variation of naturally occurring Saudi Arabic. The manual audit is sample-based, so it does not guarantee that every row is linguistically perfect. The corpus is intended for unlabelled CPT only and does not replace naturally collected labelled Saudi banking data.

The project uses a frozen Saudi test set to reduce evaluation leakage. The test set must remain outside training and tuning. Generated texts should not contain personal data, named banks, or named financial products. Any future change to the prompts, schema, corpus composition, or experimental conditions must be documented in `decision_log.md` before execution.

## 9. Reproducibility and Artifacts

The active project structure separates context, documentation, configurations, notebooks, source code, and data notes. Large datasets, checkpoints, run outputs, metrics, and figures are stored in Drive. Code and notebooks are versioned in GitHub.

The primary reproducibility artifacts are:

| Artifact | Purpose |
| --- | --- |
| `03_notebooks/00_eda_and_data_audit.ipynb` | EDA and data-audit workflow |
| `03_notebooks/sarf_corpus_generation_cleaning_final_20260830.ipynb` | Synthetic corpus generation and cleaning workflow |
| `04_src/baseline_svm.py` | SVM baseline reference implementation |
| `04_src/metrics.py` | Shared evaluation metrics contract |
| `02_configs/experiment_plan.md` | Approved experimental conditions and protocol |
| `01_docs/CORPUS_CARD_v2.md` | v2 corpus purpose, composition, QA, and limitations |
| `01_docs/FREEZE_DECLARATION_v2.md` | v2 freeze declaration and training-use rule |
| `05_data_notes/data_manifest.csv` | Source dataset manifest and verification status |
| `06_final_audit/corpus_manifest_v2.json` | v2 row counts, hashes, quality summary, and leakage summary |

Version v1 and draft artifacts are retained in an archive for provenance and are excluded from the active training workflow.

## 10. References

Add the final verified links for the ArBanking77 dataset, the AraBERT model, the Continued Pre-Training research foundation, and the project repository before submission. Do not add a reference unless the linked source was actually used in the report or experiment.
