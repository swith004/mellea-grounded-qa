from itertools import combinations
from pathlib import Path
from statistics import mean

import mellea
from docx import Document
from mellea.backends import ModelOption
from mellea.stdlib.sampling.majority_voting import MBRDRougeLStrategy
from rouge_score.rouge_scorer import RougeScorer

SYLLABI_DIR = Path("CSCourseSyllabi")
MODEL = "granite4.1:3b"  # same model as Group 6's baseline; Mellea's default differs
TEMPERATURE = 0.7  # at 0 every run is identical and there is nothing to measure
RUNS = 5
VOTES = 5

QUESTIONS = [
    ("CIS_3590", "What are the prerequisite courses?"),
    ("CIS_3590", "What score must students get on assignments to earn the micro-credential badge?"),
    ("COP_3337", "What textbook does the course use?"),
    ("COP_4338", "What are the corequisite courses?"),
    ("COP_4338", "What topics does the concurrency unit cover?"),
]

_rouge = RougeScorer(["rougeL"], use_stemmer=True)


def read_file(course_code: str) -> str:
    path = SYLLABI_DIR / f"{course_code}.docx"
    if not path.is_file():
        raise FileNotFoundError(f"Syllabus not found: {path}")

    document = Document(path)
    sections = []

    sections.extend(
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    for table in document.tables:
        for row in table.rows:
            cells = []

            for cell in row.cells:
                text = cell.text.strip()

                if text and text not in cells:
                    cells.append(text)

            if cells:
                sections.append(" | ".join(cells))

    return "\n".join(sections)


def ask_question(course_code: str, question: str, voting: bool = False) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question cannot be empty")

    syllabus_text = read_file(course_code)
    session = mellea.start_session(model_id=MODEL)
    prompt = (
        "Answer the question briefly, using only the provided syllabus context. "
        "If the answer is not in the syllabus, say 'I don't know.'\n\n"
        f"Question:\n{question.strip()}"
    )
    response = session.instruct(
        prompt,
        grounding_context={course_code: syllabus_text},
        # None, not the default: instruct() otherwise adds its own retry loop
        strategy=MBRDRougeLStrategy(number_of_samples=VOTES) if voting else None,
        model_options={ModelOption.TEMPERATURE: TEMPERATURE},
    )

    return str(response)


def drift(answers: list[str]) -> float:
    """1 - mean pairwise RougeL F-measure: 0 means every answer matched."""
    return 1 - mean(
        _rouge.score(a, b)["rougeL"].fmeasure for a, b in combinations(answers, 2)
    )


def main() -> None:
    for voting in (False, True):
        print(f"\nvoting={'on' if voting else 'off'}")
        drifts = []
        for course_code, question in QUESTIONS:
            answers = [ask_question(course_code, question, voting) for _ in range(RUNS)]
            drifts.append(drift(answers))
            print(f"  {drifts[-1]:.3f}  {course_code}: {question}")
        print(f"  mean drift: {mean(drifts):.3f}")


if __name__ == "__main__":
    main()
