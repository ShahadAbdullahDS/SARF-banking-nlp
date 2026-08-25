# Experiment Plan

## Main question
How does increasing synthetic Saudi Arabic exposure affect Macro-F1 on the frozen Saudi evaluation split?

## Fixed data roles
- MSA train: labeled banking queries for fine-tuning.
- MSA dev: hyperparameter selection and early stopping only.
- Saudi evaluation split: frozen planned evaluation only.

## Experiments
| ID | CPT corpus                                 | Size      |
|----|--------------------------------------------|-----------|
| E0 | No CPT                                     | 0         |
| E1 | General synthetic Saudi Arabic             | 3K        |
| E2 | General synthetic Saudi Arabic             | 6K        |
| E3 | General synthetic Saudi Arabic             | 12K       |
| EB | 12K general + synthetic Saudi banking text | 15K total |

## Evaluation
- Primary metric: Macro-F1.
- Secondary metrics: weighted F1 and accuracy.
- Run every experiment with 3 random seeds.
- Report mean ± standard deviation.
- Do not change hyperparameters after observing Saudi evaluation results.
