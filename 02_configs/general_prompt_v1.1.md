# General Saudi Corpus — Prompt v1

## Role
You generate short, natural Saudi-style Arabic texts for a research corpus.

## Goal
Create unlabeled general Saudi Arabic text for continued pre-training. The text will be used only to expose a language model to Saudi-style wording before a separate banking intent-classification task.

## Output format
Return CSV rows only, with these columns:

```csv
id,text,pool,topic,genre,length_bucket,generator,prompt_version,batch_id,created_at,audit_status
```

Use `general` for pool, `v1` for prompt_version, and `pending` for audit_status.

## Content requirements
- Write natural Saudi-style Arabic, not formal MSA and not deliberately broken Arabic.
- Produce only one self-contained text per row.
- Use the requested topic, genre, and length bucket.
- Use everyday, non-financial topics only.
- Do not include personal names, phone numbers, account numbers, addresses, IDs, or private information.
- Avoid repeated templates, repeated openings, and near-duplicate texts.
- Use globally unique IDs with the batch prefix, for example: `general_pilot_01_001`.
- Avoid ambiguous standalone words that may be banking-related, such as `branch`, unless the non-financial context is explicit in the same text.


## Strict exclusions
Do NOT mention or imply banks, banking applications, cards, transfers, balances, payments, invoices, money, currencies, loans, accounts, ATMs, or any banking intent from ArBanking77.

## Genre guidance
- `question`: a natural question someone may ask in daily life.
- `statement`: a short everyday statement or observation.
- `comment`: a casual comment or review about a non-financial everyday topic.

## Length guidance
- `short`: 5–9 Arabic words.
- `medium`: 10–20 Arabic words.
- `long`: 21–30 Arabic words.

## Batch request template
Generate [N] texts for:
- topic: [TOPIC]
- genre: [GENRE]
- length_bucket: [LENGTH_BUCKET]
- batch_id: [BATCH_ID]
- created_at: [DATE]

Return valid CSV only. Do not add explanations, headings, markdown fences, numbering, or extra text.
