#!/usr/bin/env python3
"""Validate the FIU Q&A benchmark's JSON, CSV, citations and split invariants."""
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark"
FIELDS = ["id", "question", "expected_answer", "answerable",
          "source_document", "source_url", "supporting_passage",
          "difficulty", "reason_unanswerable"]
REFUSAL = "I don't know based on the provided documents."

def fail(message):
    raise AssertionError(message)

def load(name):
    with (BENCHMARK / (name + ".json")).open(encoding="utf-8") as f:
        data = json.load(f)
    with (BENCHMARK / (name + ".csv")).open(encoding="utf-8", newline="") as f:
        table = list(csv.DictReader(f))
    if len(data) != len(table):
        fail(f"{name}: CSV/JSON lengths differ")
    for idx, (record, row) in enumerate(zip(data, table), start=1):
        if set(row) != set(FIELDS):
            fail(f"{name} CSV column mismatch")
        if set(record) - set(FIELDS):
            fail(f"{name} JSON contains unexpected fields")
        for key in FIELDS:
            raw = record.get(key)
            expected_csv = "" if raw is None else str(raw)
            if row[key] != expected_csv:
                fail(f"{name} row {idx}, field {key}: CSV/JSON mismatch")
    return data

def validate():
    answerable = load("answerable")
    unanswerable = load("unanswerable")
    if not 30 <= len(answerable) <= 50:
        fail("Answerable partition must have 30–50 questions")
    if not 10 <= len(unanswerable) <= 15:
        fail("Unanswerable partition must have 10–15 questions")
    if len(answerable) != 45 or len(unanswerable) != 15:
        fail("Benchmark v1.0 expects 45 answerable and 15 unanswerable questions")

    all_rows = answerable + unanswerable
    ids = [x["id"] for x in all_rows]
    questions = [x["question"].strip().casefold() for x in all_rows]
    if len(set(ids)) != len(ids) or len(set(questions)) != len(questions):
        fail("Duplicate ID or question")

    for i, record in enumerate(answerable, 1):
        if record["id"] != f"A{i:03d}" or record.get("answerable") is not True:
            fail(f"Invalid answerable record at row {i}")
        for field in ("question", "expected_answer", "source_document",
                      "source_url", "supporting_passage"):
            if not record.get(field):
                fail(f"{record['id']}: missing {field}")
        path = (ROOT / record["source_document"]).resolve()
        if ROOT.resolve() not in path.parents or not path.is_file():
            fail(f"{record['id']}: invalid source document path")
        text = path.read_text(encoding="utf-8")
        if record["supporting_passage"] not in text:
            fail(f"{record['id']}: support passage not found in {path.name}")
        if record["source_url"] not in text:
            fail(f"{record['id']}: source URL not cited in corpus fact sheet")
        if record.get("reason_unanswerable") is not None:
            fail(f"{record['id']}: unexpected refusal reason")
        if record["expected_answer"] == REFUSAL:
            fail(f"{record['id']}: answerable row uses refusal answer")

    for i, record in enumerate(unanswerable, 1):
        if record["id"] != f"U{i:03d}" or record.get("answerable") is not False:
            fail(f"Invalid unanswerable record at row {i}")
        if record["expected_answer"] != REFUSAL:
            fail(f"{record['id']}: unexpected refusal reference response")
        for field in ("source_document", "source_url", "supporting_passage"):
            if record.get(field) is not None:
                fail(f"{record['id']}: invented supporting evidence in {field}")
        if not record.get("reason_unanswerable"):
            fail(f"{record['id']}: missing author-only explanation")

    manifest = json.loads((ROOT / "corpus_manifest.json").read_text(encoding="utf-8"))
    for src in manifest["documents"]:
        path = ROOT / src["path"]
        if not path.is_file():
            fail(f"Missing corpus document {path}")
        counts = sum(r["source_document"] == src["path"] for r in answerable)
        if counts != src["qa_count"]:
            fail(f"Manifest question count mismatch for {path.name}")
    print(f"PASS: {len(answerable)} answerable + {len(unanswerable)} unanswerable questions")
    print(f"PASS: {len(manifest['documents'])} corpus documents with exact supporting passages")
    print("PASS: JSON/CSV parity, ID uniqueness, split and refusal invariants")

if __name__ == "__main__":
    try:
        validate()
    except (AssertionError, OSError, ValueError, KeyError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
