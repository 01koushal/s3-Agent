# Evaluation protocol

The `/api/metrics` endpoint reports a holdout evaluation for the included intent dataset.

For a defensible academic evaluation:
- Add at least 150–300 varied, manually labelled examples across all four intents.
- Keep a separate test set that is never used during model fitting.
- Report macro precision, macro recall, macro F1, per-class scores, and confusion matrix.
- Test SQL generation on a manually curated set of questions with expected results.
- Evaluate RAG retrieval using manually labelled relevant passages and Precision@K / Recall@K.
- Do not claim a metric as a result until the experiment has actually been run and recorded.
