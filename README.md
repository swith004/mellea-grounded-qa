# mellea-grounded-qa

## Grounded Q&A Assistant
Ask questions over a real set of documents and get answers that cite their source — and an honest "I don't know" when the answer isn't there, instead of a made-up one. Built with [Mellea](https://docs.mellea.ai).

**Setup:** Python 3.11+ pip install mellea A backend: Ollama running an IBM Granite model locally, or a hosted Granite endpoint

**Pick a document set:** the FIU course catalog / academic policies · a public product manual · government benefits or tax rules (e.g. IRS publications at `irs.gov`).

**Contribute:** fork, pick an issue, open a PR. Interface (CLI / notebook / web) is your team's choice.

## Answer stability via majority voting (#14)

Each question is asked 5 times at temperature 0.7 on `granite4.1:3b`, once with plain sampling and once with
Mellea's `MBRDRougeLStrategy` (5 votes per answer). The whole syllabus is the grounding context.

**Drift** (our definition of variance) = 1 − mean pairwise RougeL F-measure of a question's 5 answers.
0 means all 5 answers matched. Caveat: the voting strategy also picks answers by RougeL, so this metric slightly favours it.

```bash
ollama pull granite4.1:3b
python -m grounded_qa.majority_voting   # needs ./CSCourseSyllabi/, ~3 min on a laptop
pytest -q                               # no Ollama needed
```

| Question | Voting off | Voting on |
|---|---|---|
| CIS_3590: What are the prerequisite courses? | 0.000 | 0.000 |
| CIS_3590: What score must students get on assignments to earn the micro-credential badge? | 0.024 | 0.024 |
| COP_3337: What textbook does the course use? | 0.165 | 0.000 |
| COP_4338: What are the corequisite courses? | 0.000 | 0.000 |
| COP_4338: What topics does the concurrency unit cover? | 0.044 | 0.000 |
| **Mean drift** | **0.047** | **0.005** |

One run, Nov 16 2026. Voting removed the drift on the two questions that varied most; the others were already stable.
