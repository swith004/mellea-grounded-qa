import pytest
from mellea.stdlib.sampling.majority_voting import MBRDRougeLStrategy

from grounded_qa import majority_voting as mv


def test_drift_is_zero_for_identical_answers_and_positive_otherwise():
    assert mv.drift(["COP 2210", "COP 2210", "COP 2210"]) == 0
    assert mv.drift(["COP 2210", "COP 2210", "Big Java"]) > 0


@pytest.mark.parametrize("voting, strategy_type", [(False, type(None)), (True, MBRDRougeLStrategy)])
def test_ask_question_uses_voting_strategy_only_when_asked(monkeypatch, voting, strategy_type):
    calls = []

    class FakeSession:
        def instruct(self, prompt, **kwargs):
            calls.append(kwargs)
            return "COP 2210"

    monkeypatch.setattr(mv, "read_file", lambda code: "syllabus text")
    monkeypatch.setattr(mv.mellea, "start_session", lambda **kwargs: FakeSession())

    assert mv.ask_question("CIS_3590", "Prerequisites?", voting=voting) == "COP 2210"
    assert isinstance(calls[0]["strategy"], strategy_type)


def test_ask_question_rejects_empty_question():
    with pytest.raises(ValueError):
        mv.ask_question("CIS_3590", "  ")
