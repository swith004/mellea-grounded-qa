#!/usr/bin/env python3
"""Run a pipeline over all 60 benchmark questions and report automatic metrics.

    python scripts/run_benchmark.py                      # baseline, granite4.1:3b
    python scripts/run_benchmark.py --model granite4.1:8b --limit 5

Writes one JSON line per question to results/<pipeline>_<model>.jsonl and a
summary to results/<pipeline>_<model>.summary.json.

Automatic metrics (no judge model needed):
  * retrieval doc recall@k   gold source_document among retrieved chunks (answerable)
  * retrieval passage recall@k  exact gold supporting_passage retrieved (answerable)
  * citation rate             answer names the gold corpus file (answerable)
  * false refusal rate        answer looks like a refusal (answerable)
  * refusal accuracy          answer looks like a refusal (unanswerable)
  * hallucination rate        1 - refusal accuracy (unanswerable)

Answer accuracy needs semantic judgement; each record carries expected_answer and an
empty "correct" field so it can be graded by hand or by a judge later.
Refusal detection is a regex heuristic - spot-check the "refused" flag in the JSONL.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grounded_qa.baseline import DEFAULT_K, DEFAULT_MODEL, BaselinePipeline  # noqa: E402

PIPELINES = {"baseline": BaselinePipeline}

# A refusal is a statement about the *source material* ("the context does not
# mention X"), not a negative policy fact ("FIU does not include transfer grades",
# "grade forgiveness is not available after graduation"). Negations only count when
# their subject or location is the context/documents; bare "I don't know" and
# "cannot determine/answer" count anywhere. Accepts straight and curly apostrophes.
_SOURCE = r"(?:context|documents?|passages?|text|sources?|information)"
_NOT = r"(?:\s+not|n['’]t)"
_VERB = r"(?:include|contain|mention|provide|specify|state|say|have|cover|address|give)"
_PARTICIPLE = r"(?:mentioned|provided|specified|included|available|stated|given|addressed|covered)"
REFUSAL_RE = re.compile(
    r"\b(?:"
    rf"(?:i\s+)?do{_NOT}\s+know"
    # "the (provided) context does not mention ...", "the documents don't include ..."
    rf"|{_SOURCE}(?:\s+(?:provided|given|above|supplied))?\s+(?:does|do|did){_NOT}\s+{_VERB}"
    # "... is not mentioned in the (provided) context"
    rf"|(?:is|are|was|were){_NOT}\s+{_PARTICIPLE}\s+(?:in|by)\s+(?:the\s+)?"
    rf"(?:provided\s+|given\s+)?{_SOURCE}"
    # "there is no information about ...", "no mention of ..."
    r"|no\s+(?:specific\s+)?(?:information|mention|details|data)\s+(?:about|on|regarding|of|in|for)"
    r"|(?:cannot|can['’]t|can\s+not|unable\s+to)\s+(?:determine|answer|be\s+(?:determined|answered))"
    r"|not\s+possible\s+to\s+(?:determine|answer|say)"
    r"|insufficient\s+information"
    r")\b",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL_RE.search(answer))


def load_questions() -> list[dict]:
    rows = []
    for name in ("answerable", "unanswerable"):
        rows += json.loads((ROOT / "benchmark" / f"{name}.json").read_text(encoding="utf-8"))
    return rows


def rate(num: int, den: int) -> dict:
    return {"n": num, "of": den, "rate": round(num / den, 3) if den else None}


def summarize(records: list[dict]) -> dict:
    ans = [r for r in records if r["answerable"]]
    una = [r for r in records if not r["answerable"]]
    refused_una = sum(r["refused"] for r in una)
    return {
        "answerable": {
            "retrieval_doc_recall": rate(sum(r["gold_doc_retrieved"] for r in ans), len(ans)),
            "retrieval_passage_recall": rate(sum(r["gold_passage_retrieved"] for r in ans), len(ans)),
            "citation_rate": rate(sum(r["cites_gold_doc"] for r in ans), len(ans)),
            "false_refusal_rate": rate(sum(r["refused"] for r in ans), len(ans)),
        },
        "unanswerable": {
            "refusal_accuracy": rate(refused_una, len(una)),
            "hallucination_rate": rate(len(una) - refused_una, len(una)),
        },
        "mean_latency_s": round(sum(r["latency_s"] for r in records) / len(records), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the grounded-QA benchmark.")
    ap.add_argument("--pipeline", choices=PIPELINES, default="baseline")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("-k", type=int, default=DEFAULT_K)
    ap.add_argument("--limit", type=int, help="only run the first N of each split")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    questions = load_questions()
    if args.limit:
        questions = ([q for q in questions if q["answerable"]][: args.limit]
                     + [q for q in questions if not q["answerable"]][: args.limit])

    pipeline = PIPELINES[args.pipeline](model=args.model, backend=args.backend, k=args.k)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.pipeline}_{re.sub(r'[^A-Za-z0-9._-]+', '-', args.model)}"
    out_path = args.out_dir / f"{stem}.jsonl"

    records = []
    with out_path.open("w", encoding="utf-8") as out:
        for i, q in enumerate(questions, 1):
            r = pipeline.answer(q["question"])
            docs = [c.doc for c in r.retrieved]
            gold_doc = q["source_document"]
            rec = {
                "id": q["id"],
                "answerable": q["answerable"],
                "difficulty": q["difficulty"],
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                "answer": r.answer,
                "refused": is_refusal(r.answer),
                "correct": None,  # fill in by hand / judge
                "cites_gold_doc": bool(gold_doc) and Path(gold_doc).name in r.answer,
                "gold_doc_retrieved": bool(gold_doc) and gold_doc in docs,
                "gold_passage_retrieved": bool(q["supporting_passage"])
                    and q["supporting_passage"] in [c.text for c in r.retrieved],
                "retrieved": [{"doc": c.doc, "text": c.text, "score": s}
                              for c, s in zip(r.retrieved, r.scores)],
                "latency_s": r.latency_s,
                "pipeline": args.pipeline,
                "model": args.model,
                "k": args.k,
            }
            records.append(rec)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            flag = "REFUSED" if rec["refused"] else "answered"
            print(f"[{i:2d}/{len(questions)}] {q['id']} {flag:8s} {r.answer[:90]!r}",
                  flush=True)

    summary = {"pipeline": args.pipeline, "model": args.model, "k": args.k,
               "questions": len(records), **summarize(records)}
    (args.out_dir / f"{stem}.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
