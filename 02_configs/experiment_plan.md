# Experiment Plan

## Main Question

How does increasing synthetic Saudi Arabic exposure affect Macro-F1 when AraBERT is first adapted through Continued Pre-Training, then fine-tuned on labelled MSA banking queries, and finally evaluated on the frozen Saudi evaluation split?

## Fixed Data Roles

| Data split | Role | Use |
| --- | --- | --- |
| MSA train | Labelled banking queries | Fine-tuning every AraBERT condition and training the baseline models |
| MSA validation | Labelled banking queries | Hyperparameter selection, early stopping, and checkpoint selection only |
| Saudi evaluation split | Frozen Saudi variants | Final evaluation only, after all training and checkpoint decisions are fixed |

The MSA training split contains 10,732 examples, the MSA validation split contains 1,229 examples, and the frozen Saudi evaluation split contains 3,580 examples across 77 intent labels.

## Approved Synthetic CPT Inputs

| ID | CPT corpus | Size | Approved v2 file(s) |
| --- | --- | --- | --- |
| E0 | No Saudi CPT | 0 | None |
| E1 | General synthetic Saudi-style Arabic | 3K | `general_3k_final_v2.csv` |
| E2 | General synthetic Saudi-style Arabic | 6K | `general_6k_final_v2.csv` |
| E3 | General synthetic Saudi-style Arabic | 12K | `general_12k_final_v2.csv` |
| EB | 12K general synthetic Saudi-style Arabic + 3K Saudi banking text | 15K total | `general_12k_final_v2.csv` + `banking_3k_final_v2.csv` |

The v2 files are the only synthetic files approved for training. v1 files and draft files are retained for provenance and must not be used in the experiments.

## AraBERT Protocol

1. Start every condition from the same AraBERT base checkpoint.

1. For E1, E2, E3, and EB, perform Continued Pre-Training on the assigned unlabelled Saudi-style corpus using the masked-language-model objective.

1. Skip CPT for E0.

1. Fine-tune every condition on the same labelled MSA banking training set.

1. Use the same MSA validation procedure for all five conditions.

1. Run every condition with the same three approved random seeds.

1. Use the MSA validation split to select the best checkpoint within each condition.

1. Keep the Saudi evaluation split closed until all condition-level training and checkpoint decisions are fixed.

## Fixed AraBERT Random Seeds

Use the same fixed random seeds for every official AraBERT condition:

```python
SEEDS = [42, 123, 2026]
```

## Baseline Models

The project also includes two non-CPT baselines:

| Baseline | Input and training |
| --- | --- |
| TF-IDF + SVM | Train on labelled MSA banking queries using the shared MSA train and validation protocol. |
| TextCNN | Train on labelled MSA banking queries using the shared MSA train and validation protocol. |

The baselines provide reference points for intent classification. They are separate from the five AraBERT CPT conditions.

## Evaluation

| Metric | Role |
| --- | --- |
| Macro-F1 | Primary metric because the 77 intent classes have different support sizes. |
| Weighted F1 | Secondary metric. |
| Accuracy | Secondary metric. |

For each AraBERT condition, report the mean and standard deviation across the three random seeds. Evaluate the final fixed models on the Saudi evaluation split only after all MSA-based checkpoint selections and training decisions are complete.

The Saudi evaluation split must not be used for prompt changes, corpus selection, hyperparameter tuning, early stopping, checkpoint selection, debugging decisions, or any other model-development decision. No hyperparameters or model choices may be changed after viewing Saudi evaluation results.
