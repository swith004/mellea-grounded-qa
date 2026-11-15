"""Baseline: naive retrieve-then-answer with NO grounding.

This is the "before" system the grounded assistant is measured against. It
retrieves the top-k corpus facts and pastes them into the prompt, then takes a
single generation. Deliberately absent:

  * no instruction to answer only from the documents
  * no instruction to cite sources
  * no "I don't know" / refusal instruction
  * no retrieval-score threshold (top-k is always passed, even if irrelevant)
  * no Mellea requirements, validation, or rejection sampling (strategy=None)
  * no use of Mellea's ``grounding_context``

Usage:
    python -m grounded_qa.baseline "What GPA triggers academic warning?"
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

from mellea import start_session
from mellea.backends import ModelOption

from .retrieval import BM25Retriever, Chunk

DEFAULT_MODEL = "granite4.1:3b"
DEFAULT_K = 4

PROMPT = """Use the following context to answer the question.

Context:
{context}

Question: {question}"""


@dataclass
class Result:
    question: str
    answer: str
    retrieved: list[Chunk] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    latency_s: float = 0.0


class BaselinePipeline:
    name = "baseline"

    def __init__(self, model: str = DEFAULT_MODEL, backend: str = "ollama",
                 k: int = DEFAULT_K, seed: int = 0):
        self.k = k
        self.retriever = BM25Retriever()
        self.session = start_session(
            backend, model,
            model_options={ModelOption.TEMPERATURE: 0.0, ModelOption.SEED: seed},
        )

    def answer(self, question: str) -> Result:
        hits = self.retriever.search(question, k=self.k)
        context = "\n".join(f"- {c.text}" for c, _ in hits)
        start = time.perf_counter()
        out = self.session.instruct(
            PROMPT.format(context=context, question=question),
            strategy=None,
        )
        # Each question is independent; don't let earlier Q&A leak into later prompts.
        self.session.reset()
        return Result(
            question=question,
            answer=str(out.value).strip(),
            retrieved=[c for c, _ in hits],
            scores=[round(s, 3) for _, s in hits],
            latency_s=round(time.perf_counter() - start, 2),
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("question")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("-k", type=int, default=DEFAULT_K)
    args = ap.parse_args()
    r = BaselinePipeline(args.model, args.backend, args.k).answer(args.question)
    print(r.answer)
    print("\n[retrieved]")
    for c, s in zip(r.retrieved, r.scores):
        print(f"  {s:6.2f}  {c.doc}: {c.text}")


if __name__ == "__main__":
    main()
