# mellea-grounded-qa

## Grounded Q&A Assistant
Ask questions over a real set of documents and get answers that cite their source — and an honest "I don't know" when the answer isn't there, instead of a made-up one. Built with [Mellea](https://docs.mellea.ai).

**Setup:** Python 3.11+ pip install mellea A backend: Ollama running an IBM Granite model locally, or a hosted Granite endpoint

**Pick a document set:** the FIU course catalog / academic policies · a public product manual · government benefits or tax rules (e.g. IRS publications at `irs.gov`).

## Baseline ("before") pipeline

`grounded_qa/baseline.py` is a naive retrieve-then-answer pipeline with **no grounding**. It is the reference point the grounded assistant is measured against. It retrieves the top-k policy facts with BM25 (`grounded_qa/retrieval.py`), pastes them into the prompt, and takes a single generation. It does **not** instruct the model to stay within the documents, cite sources, or refuse. It also applies no retrieval threshold and uses no Mellea requirements, validation, or `grounding_context`.

```bash
pip install -r requirements.txt
ollama pull granite4.1:3b
python -m grounded_qa.baseline "What GPA triggers academic warning?"
python scripts/run_benchmark.py            # all 60 questions -> results/
```

`run_benchmark.py` writes per-question results to `results/<pipeline>_<model>.jsonl` and metrics to `.summary.json`. The metrics are retrieval recall, citation rate, false-refusal rate, and refusal/hallucination rate on unanswerables. Answer accuracy needs semantic grading: each record has a `correct` field to fill in. Refusals are detected with a regex heuristic, so spot-check the `refused` flag.

The grounded pipeline should reuse the same retriever, model, and `k`, so any score difference comes from grounding alone.

**Contribute:** fork, pick an issue, open a PR. Interface (CLI / notebook / web) is your team's choice.
