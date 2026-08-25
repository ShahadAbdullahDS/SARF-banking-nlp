# Decision Log

| Date       | Decision                                               | Why                                       | Owner |
|------------|--------------------------------------------------------|-------------------------------------------|---|
| 2026-08-25 | Use AraBERTv2-base as the main model.                  | It is the main model for the CPT study.   | A |
| 2026-08-25 | Use MSA train/dev and a frozen Saudi evaluation split. | Saudi data must not be used for tuning.   | A |
| 2026-08-25 | Run E0, E1 (3K), E2 (6K), E3 (12K), and EB.            | This is the approved experimental design. | A |
| 2026-08-25 | Use 3 seeds per experiment.                            | To report mean ± standard deviation.      | A |
| 2026-08-25 | Keep the scope to intent classification only.          | No chatbot, RAG, or extra dialects.       | A |
