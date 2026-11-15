# mellea-grounded-qa

## Grounded Q&A Assistant

Ask questions over a real document set and get evidence-backed answers, including an honest
"I don't know" when the answer is absent. Intended for a Mellea-backed assistant using an
IBM Granite model through Ollama or a hosted endpoint.

## Benchmark: FIU undergraduate academic policies

**Snapshot date:** November 10, 2026  
**Data source:** FIU's public university catalog at https://catalog.fiu.edu/  
**Benchmark version:** 1.0  
**Document set:** Six curated, paraphrased FIU academic-policy fact sheets.

The exact input corpus for the benchmark is the six files in `corpus/`. These were written
specifically for reproducible evaluation using facts cross-checked against the listed official
FIU catalog pages. They are **not full verbatim copies** of FIU's catalog, and the official
website—not these snapshots—governs current academic policies.

### Dataset split

| Partition | JSON file | CSV file | Count | Ground truth |
|---|---|---|---:|---|
| Answerable | `benchmark/answerable.json` | `benchmark/answerable.csv` | 45 | Expected answer + exact passage found in the curated corpus |
| Unanswerable | `benchmark/unanswerable.json` | `benchmark/unanswerable.csv` | 15 | Expected refusal; no matching evidence passage in the corpus |
| **Total** | | | **60** | 75% answerable / 25% unanswerable |

This is an **answerability split**, not a train/validation/test split. Both partitions are
intended for evaluation; no training examples are supplied.

### Files

```text
mellea-grounded-qa/
├── README.md
├── corpus_manifest.json
├── corpus/
│   ├── academic_standing.md
│   ├── grading.md
│   ├── bachelors.md
│   ├── forgiveness.md
│   ├── transfer.md
│   └── ucc.md
├── benchmark/
│   ├── answerable.json
│   ├── answerable.csv
│   ├── unanswerable.json
│   └── unanswerable.csv
└── scripts/
    └── validate_benchmark.py
```

### Dataset fields

- `id`: Stable example identifier (`A001`–`A045` or `U001`–`U015`).
- `question`: Natural-language question to ask the assistant.
- `expected_answer`: Factual answer or standardized refusal.
- `answerable`: Boolean indicating whether the curated corpus supports an answer.
- `source_document`: Relative path of the supporting curated fact sheet; `null` for unanswerable.
- `source_url`: Official FIU page used to curate the evidence; `null` for unanswerable.
- `supporting_passage`: Exact sentence occurring in `source_document`; `null` for unanswerable.
- `difficulty`: `direct`, `application`, `procedural`, or `unanswerable`.
- `reason_unanswerable`: Explanation of the deliberately missing information, for benchmark
  authors and analysis only (not for prompts to the assistant).

The CSV columns are the same across both partitions; blank CSV fields correspond to JSON `null`.
The `answerable` field is a native JSON boolean and appears as `True`/`False` in the CSV.

### Rules for answering

1. Ingest **only** the documents in `corpus/`. Do not give the model access to the benchmark
   questions or expected answers during inference.
2. Answer when the retrieved documents provide evidence. Cite the **corpus file** (and, if
   possible, the specific supporting sentence).
3. When the documents do not establish an answer, return a clear refusal such as
   `"I don't know based on the provided documents."`
4. Do not silently answer unanswerable cases using external facts, search engines, model
   pretraining, student records, or conjecture.
5. An answer may be semantically equivalent to `expected_answer`; exact string matching alone
   is too strict for factual evaluation.

### How to run validation

Requires only Python 3.11+ (no dependencies):

```bash
python scripts/validate_benchmark.py
```

This checks dataset counts, unique IDs/questions, cross-format equivalence, existing source
paths, exact passage containment in each corpus file, and empty evidence for refusals.
It does **not** prove that all unanswerable questions are semantically absent; their
absence was reviewed by the benchmark author against this limited six-document corpus.

### Suggested evaluation

Record an assistant's answer and cited source for every question, then report:

- **Answer accuracy:** Correct, semantically equivalent answer / 45 answerable prompts.
- **Evidence grounding:** Correct citation with supporting material / 45 answerable prompts.
- **Refusal accuracy:** Appropriate refusal / 15 unanswerable prompts.
- **Hallucination rate on unanswerables:** Unsupported substantive answers / 15 prompts.
- **Overall behavior:** Include false refusals on answerable questions separately from
  unsupported answers on unanswerable questions.

Evaluate both subsets separately, not just blended accuracy. A system that refuses all
questions would score well on refusal accuracy but fail at answer accuracy.

### Scope and limitations

This is a *bounded-corpus* benchmark: absence means absent from these six snapshots, not
necessarily absent from FIU's website or from reality. The factsheets are concise,
paraphrased retrieval documents. Tests using full, unabridged catalog pages may require
re-auditing the unanswerable partition. Some FIU policies depend on entry term, degree
program, and exception approvals; this benchmark tests the explicitly documented rule,
not individualized academic advising. Verify current official policy before relying
on an answer for a real student.

### Contributing

The original project invites forking the repository, choosing an issue, and submitting a
pull request. For local changes:

```bash
git add README.md corpus_manifest.json corpus/ benchmark/ scripts/validate_benchmark.py
git commit -m "Add grounded FIU policy benchmark with refusal cases"
git push
```

Run validation before committing. Mellea interface implementation (CLI, notebook, or web)
and backend setup are separate from the benchmark data.
