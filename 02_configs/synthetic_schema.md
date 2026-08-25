# Synthetic Saudi Corpus Schema

## Purpose
Create controlled, unlabeled Saudi-style Arabic text for continued pre-training (CPT).

## Required columns
| Column | Meaning |
|---|---|
| id | Unique text ID. |
| text | Generated Saudi-style Arabic text. |
| pool | `general` or `banking`. |
| topic | Planned content topic. |
| genre | `question`, `statement`, or `comment`. |
| length_bucket | `short`, `medium`, or `long`. |
| generator | Generator/provider used. |
| prompt_version | Prompt version, starting with `v1`. |
| batch_id | Generation batch ID. |
| created_at | Generation date. |
| audit_status | `pending`, `pass`, or `fail`. |

## General pool
- Target size: 12,000 texts.
- Saudi-style Arabic across balanced everyday topics.
- No labels are used by the model.
- Do not include banking, cards, transfers, balances, payments, currencies, or ArBanking77 intent names.

## Banking pool
- Target size: 3,000 texts.
- Saudi-style banking-related text without intent labels.
- Used only in the EB secondary experiment after the 12K general corpus.

## Length mix
- Short: 5–9 words, about 35%.
- Medium: 10–20 words, about 50%.
- Long: 21–30 words, about 15%.

## Audit rule
No bulk generation starts before a small pilot is reviewed for Saudi style, clarity, duplicates, PII, and banking-term leakage in the general pool.
