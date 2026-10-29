"""Structured laboratory authority, citation modality binding, and judge quota policy.

Two defect classes from Gate 2, each pinned here so the coupling cannot drift apart again:

  - a laboratory-trend citation index was attached to a narrative claim about
    transfusion, because `^[4]` and `^[6]` look alike and nothing said what kind of
    evidence each number pointed at. Every numbered source now carries `cite_modality`.
  - the judge exhausted its regional quota. Its endpoint and backoff are asserted here
    rather than left to the next 429.

Offline: no model, no network, no cluster.
"""

import asyncio
import json
import re

from services.agent.engine.mcp_client import _render_tool_result
from services.agent.engine.prompts import SYSTEM_PROMPT
from services.agent.routing import routing


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
    from evaluation.platform import judge_client

    class _Models:
        def generate_content(self, **kwargs):
            raise RuntimeError(
                "429 RESOURCE_EXHAUSTED quota_metric="
                "GenerateContentInputTokensPerModelPerMinute")

    class _Client:
        models = _Models()

    waits: list[int] = []
    monkeypatch.setattr(judge_client.time, "sleep",
                        lambda seconds: waits.append(seconds))

    result = judge_client.judge(_Client(), "system", "user")

    assert "RESOURCE_EXHAUSTED" in result["error"]
    assert waits == [30, 60, 120, 120]
    assert "quota_metric=GenerateContentInputTokensPerModelPerMinute" in capsys.readouterr().out
