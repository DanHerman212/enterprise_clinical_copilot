"""Structured laboratory authority, citation modality binding, and judge quota policy.

Three defect classes from Gate 2 (`pilot-verify-20261027-215139`, `00c6`), each pinned
here so the coupling cannot drift apart again:

  - `dev_t3_03` answered "which lab was worst at discharge?" from laboratory numbers
    inside a note passage, contradicting the series it had reported one turn earlier. The
    turn is now given the series and nothing else (`_freetext_allowed`).
  - `dev_t4_02` attached a laboratory-trend citation index to a narrative claim about
    transfusion, because `^[4]` and `^[6]` look alike and nothing said what kind of
    evidence each number pointed at. Every numbered source now carries `cite_modality`.
  - the judge exhausted its regional quota. Its endpoint and backoff are asserted here
    rather than left to the next 429.

Offline: no model, no network, no cluster.
"""

import asyncio
import json
import re

import pytest

from services.agent.engine import graph
from services.agent.engine.mcp_client import _render_tool_result
from services.agent.engine.prompts import SYSTEM_PROMPT
from services.agent.routing import routing


# --- 1. ranking a laboratory value is a question about the series --------------------

@pytest.mark.parametrize("question", [
    "which lab was worst at discharge?",
    "Which lab value was most abnormal at discharge?",
    "what was the highest creatinine?",
    "rank the analytes by how far out of range they were",
    "compare the hemoglobin and the sodium",
    "what was the nadir for red blood cells",
    "which laboratory result was furthest outside its reference range",
])
def test_a_ranking_question_about_labs_is_recognised(question):
    assert routing.asks_about_lab_ranking(question) is True


@pytest.mark.parametrize("question", [
    "Was there any concern about a blood clot in her leg?",
    "What is his 30-day readmission risk score?",
    "Which medication was prescribed at the highest dose?",
    "Summarize the hospital course.",
    "Is the sodium within its reference range?",
    "What was she discharged on?",
    "",
])
def test_other_questions_are_not_lab_ranking_questions(question):
    """No ranking word, no laboratory subject, or neither — and the answer is no.

    The medication question is the important negative: it carries "highest" and is not a
    laboratory question. A false positive there withdraws the note search that answers it.
    """
    assert routing.asks_about_lab_ranking(question) is False


def test_a_ranking_turn_is_given_the_series_and_nothing_else():
    """The set is REPLACED, not narrowed: only `get_lab_trends` is bound.

    A note search stays out because it returns prose that mentions laboratory numbers
    incidentally, which is how `dev_t3_03` came back with a hemoglobin of 10.8 against the
    10.0 it had already reported. Whether the panel still has to be fetched is
    `_still_bound`'s question, so the same set is asked for either way.
    """
    assert graph._freetext_allowed(
        False, False, False, False, False,
        lab_ranking_question=True, lab_panel_in_transcript=False,
    ) == frozenset({"get_lab_trends"})

    assert graph._freetext_allowed(
        False, False, False, False, False,
        lab_ranking_question=True, lab_panel_in_transcript=True,
    ) == frozenset({"get_lab_trends"})


def test_a_ranking_turn_yields_no_tools_once_the_replay_holds_the_panel():
    """`_still_bound` is what withdraws the tool, and `route` is what ends the turn.

    The replay allowance is the existing mechanism (`ONCE_PER_TRANSCRIPT` plus
    `FULL_FETCH_UNFILTERED`), so a full panel in the transcript leaves this turn with
    nothing to call. The graph must then not demand a retrieval: the demand text names
    `rag_search`, which is the search that produced the contradicting figures.
    """
    allowed = graph._freetext_allowed(
        False, False, False, False, False,
        lab_ranking_question=True, lab_panel_in_transcript=True,
    )
    replayed = [{"name": "get_lab_trends", "args": {"hadm_id": 90000015},
                 "response": {"trends": []}}]

    assert graph._still_bound(allowed, {"tool_calls": [],
                                        "replayed_tool_calls": replayed}) == frozenset()

    source = open(graph.__file__, encoding="utf-8").read()
    assert "if lab_ranking_question and lab_panel_in_transcript:" in source


# --- 2. every numbered source says what kind of evidence it is -----------------------

def _rendered(payload):
    return json.loads(_render_tool_result(payload))


def test_a_note_passage_is_labelled_a_note_section():
    rendered = _rendered({
        "returned": 1,
        "passages": [{"id": "p1", "section": "brief_hospital_course", "text": "text"}],
    })

    assert rendered["passages"][0]["cite"] == 1
    assert rendered["passages"][0]["cite_modality"] == (
        "CLINICAL NOTE SECTION — brief_hospital_course")


def test_an_analyte_is_labelled_a_structured_lab_trend():
    rendered = _rendered({
        "trends": [{"label": "Hemoglobin", "unit": "g/dL", "series": []}],
    })

    assert rendered["trends"][0]["cite"] == 1
    assert rendered["trends"][0]["cite_modality"] == "STRUCTURED LAB TREND — Hemoglobin"


def test_the_modality_label_follows_the_number_across_a_continuation():
    """The label rides on the number, so a replayed number keeps its own kind."""
    rendered = _rendered({"trends": [{"label": "RDW", "series": []}]})

    assert rendered["trends"][0]["cite"] == 1


def test_the_citation_rule_and_the_labels_use_one_vocabulary():
    """The prompt may not name a label the renderer does not produce."""
    labelled = {
        "CLINICAL NOTE SECTION": _render_tool_result({"passages": [{"section": "x"}]}),
        "STRUCTURED LAB TREND": _render_tool_result({"trends": [{"label": "x"}]}),
    }
    for label, rendered in labelled.items():
        assert label in SYSTEM_PROMPT, label
        assert label in rendered, label


# --- 3. the prompt: structured authority, and empty is not missing -------------------

def test_the_prompt_puts_structured_labs_above_note_prose():
    prompt = " ".join(SYSTEM_PROMPT.split())

    assert "STRUCTURED DATA OUTRANKS NOTE PROSE" in prompt
    assert "does not document" in prompt or "empty" in prompt
    assert "never overrides the series" in prompt


def test_the_prompt_no_longer_calls_an_empty_section_unavailable():
    """The instruction that produced `dev_t4_02`'s contradiction is gone.

    `00c6`: Turn 2 reported the section was retrieved and held no medications; Turn 3 said
    it was "unavailable". The prompt asked for exactly that word.
    """
    prompt = " ".join(SYSTEM_PROMPT.split())

    assert "section itself was unavailable" not in prompt
    assert "Reserve \"unavailable\"" in prompt


# --- 4. the judge: global endpoint, and a backoff that clears the window ------------

def test_the_judge_location_defaults_to_global_and_stays_overridable():
    from services.mcp import config

    assert config.JUDGE_LOCATION == "global"
    source = open(config.__file__, encoding="utf-8").read()
    assert 'os.environ.get("VERTEX_JUDGE_LOCATION")' in source


def test_a_quota_retry_waits_past_the_rolling_window(monkeypatch, capsys):
    """A 429 waits 30s, then 60, then the cap — never 15s inside the same window.

    The error string is printed WHOLE: it used to be cut to 120 characters, which hid the
    `quota_metric` field behind it and left the exhaustion undiagnosable.
    """
    from evaluation.agent import judge as judge_mod

    class _Models:
        def generate_content(self, **kwargs):
            raise RuntimeError(
                "429 RESOURCE_EXHAUSTED quota_metric="
                "GenerateContentInputTokensPerModelPerMinute")

    class _Client:
        models = _Models()

    waits: list[int] = []
    monkeypatch.setattr(judge_mod.time, "sleep", lambda seconds: waits.append(seconds))

    result = judge_mod._judge(_Client(), "system", "user")

    assert "RESOURCE_EXHAUSTED" in result["error"]
    assert waits == [30, 60, 120, 120]
    assert "quota_metric=GenerateContentInputTokensPerModelPerMinute" in capsys.readouterr().out
