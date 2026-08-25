# Banking Saudi Corpus — Prompt v1

## Role
You generate short, natural Saudi-style Arabic texts for a research corpus.

## Goal
Create unlabeled Saudi-style banking-related text for a secondary continued pre-training experiment. These texts must not contain intent labels or copy examples from ArBanking77.

## Output format
Return CSV rows only, with these columns:

```csv
id,text,pool,topic,genre,length_bucket,generator,prompt_version,batch_id,created_at,audit_status
```

Use `banking` for pool, `v1` for prompt_version, and `pending` for audit_status.

## Content requirements
- Write natural Saudi-style Arabic.
- Produce only one self-contained text per row.
- Use general banking situations such as cards, transfers, account access, payments, cash machines, or banking-app experience.
- Do not assign, mention, or imply an intent label.
- Do not copy or paraphrase any known ArBanking77 example.
- Do not include personal names, phone numbers, account numbers, addresses, IDs, or private information.
- Avoid repeated templates, repeated openings, and near-duplicate texts.
- Use generic banking concepts only. Do not mention real bank names, named card products, payment networks, loyalty programs, wallets, or product-specific fees, rewards, or benefits.


## Genre guidance
- `question`: a natural customer question.
- `statement`: a short customer observation.
- `comment`: a casual customer comment or app review.

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
