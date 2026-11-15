"""Lexical retrieval over the curated corpus.

Shared by every pipeline (baseline and grounded) so that differences in benchmark
scores come from how the answer is generated, not from what was retrieved.

Each corpus file is a list of one-sentence policy facts under "## Policy facts";
every bullet becomes one chunk. Ranking is plain BM25 (no dependencies).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "corpus"

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its "
    "my of on or that the their there they this to was what when where which who "
    "will with would".split()
)


@dataclass(frozen=True)
class Chunk:
    doc: str      # repo-relative path, e.g. "corpus/grading.md"
    title: str    # document H1
    text: str     # the bullet sentence


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def load_chunks(corpus_dir: Path = CORPUS_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(corpus_dir.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        title = next((l[2:].strip() for l in lines if l.startswith("# ")), path.stem)
        doc = path.relative_to(ROOT).as_posix()
        for line in lines:
            if line.startswith("- "):
                chunks.append(Chunk(doc=doc, title=title, text=line[2:].strip()))
    if not chunks:
        raise RuntimeError(f"No bullet facts found under {corpus_dir}")
    return chunks


class BM25Retriever:
    def __init__(self, chunks: list[Chunk] | None = None, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks if chunks is not None else load_chunks()
        # Title is indexed with the sentence so topical words ("forgiveness") help ranking.
        self._docs = [Counter(tokenize(f"{c.title} {c.text}")) for c in self.chunks]
        self._lens = [sum(d.values()) for d in self._docs]
        self._avg = sum(self._lens) / len(self._lens)
        n = len(self._docs)
        df = Counter(term for d in self._docs for term in d)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        scored = []
        for chunk, tf, length in zip(self.chunks, self._docs, self._lens):
            s = 0.0
            for term in q:
                if term in tf:
                    f = tf[term]
                    s += self._idf[term] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * length / self._avg))
            scored.append((chunk, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        # Always return top-k, even zero-score chunks: a naive pipeline does not
        # gate on retrieval confidence.
        return scored[:k]
