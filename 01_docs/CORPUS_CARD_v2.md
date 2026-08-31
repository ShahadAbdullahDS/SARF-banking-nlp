# Corpus Card: SARF Synthetic Saudi Arabic Corpus (v2)

> This card describes **v2**, which supersedes v1 for training purposes. See `CORPUS_CARD.md` for the original v1 description; the pipeline, purpose, and composition are unchanged except where noted below.

## What changed from v1

An independent review (Person A) found that the original 300-row general manual audit had correctly flagged 32 rows as Fail/Unsure, but 27 of them (after 5 were already removed by the earlier Latin-contamination cleaning) were still present in the v1 final 12,000. Person B re-reviewed all 32 rows against the project's official Pass/Fail/Unsure standard:

- 19 rows were recalibrated to Pass (the original review had been over-strict — an occasional plain-MSA word or minor stylistic quirk is not disqualifying on its own).
- 8 rows retained a genuine, specific defect (a wrong or garbled word, fused words, or a broken sentence) and were excluded.
- The 8 excluded rows were replaced using the existing clean reserve pool via the same reproducible stable-hash selection — no new generation.
- Each of the 8 replacement rows was individually read and manually judged; all 8 passed.

Full row-by-row reasoning: `general_audit_recalibrated.csv`. Full excluded → replacement mapping: `general_audit_correction_log.csv`.

## Quality Assurance (v2)

- General final file: 292 rows manually reviewed and Pass (265 from the original 300-sample survivors + 19 recalibrated + 8 replacements); remaining 11,708 rows were not individually manually sampled (`audit_status = not_manually_sampled`).
- Banking final file: unchanged from v1 — 60/60 manually reviewed rows Pass; remaining 2,940 rows not individually sampled.
- **No row marked Fail or Unsure by manual audit remains in either v2 final file.**
- Saudi frozen-test leakage check re-run on v2: 0 exact / 0 normalized matches, both pools.

## Known Limitations (unchanged from v1)

Manual audit remains sample-based (292/12,000 general and 60/3,000 banking directly reviewed), not exhaustive; rows outside the sampled/corrected set were not individually re-checked. See `CORPUS_CARD.md` for the full list of original limitations, which still apply.

## Freeze Status

Frozen 2026-08-31. See `corpus_manifest_v2.json` for file hashes, row counts, and audit/leakage summary. See `FREEZE_DECLARATION_v2.md` for the full sign-off checklist.
