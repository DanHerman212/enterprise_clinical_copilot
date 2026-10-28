"""The four repairs `00d1` asked for, at the layer each one lives in.

  * a suppressed chip left the row EMPTY (`holdout_t2_01`, unscored) — the pool now falls
    back to a protective factor and, failing that, to questions the record can answer;
  * a series that returned to an earlier value was described as unchanged (`t1_02`,
    `t2_02`) — a prompt rule now forbids the words;
  * "which one worried you most?" after `/labs` was routed to a note search (`t3_02`) —
    the predicate now reads the conversation as well as the sentence;
  * a passage id carried a local position beside the global citation number (`t4_01`,
    `t4_02`) — the id no longer travels to the model.
"""

from __future__ import annotations

import asyncio

import pytest

from services.agent.engine.mcp_client import _render_tool_result
from services.agent.engine.prompts import SYSTEM_PROMPT
from services.agent.postprocess import followups
from services.agent.routing import admission_context, routing


# --- 1. the chip row is never empty on a record with something to ask about ---------

def _risk(factors):
    return {"probability": 0.112419, "threshold": 0.11,
            "top_factors": factors}


def _chips(state: dict) -> list[str]:
    return followups.suggest("What does the note say?", "An answer.", state=state)


def test_the_upward_factor_is_offered_ahead_of_a_protective_one():
    """`90000067` on Gate 3: the only "increases" factor leads the chip row.

    This test used to assert the reverse — that the row skipped the upward factor because
    the patient's profile recorded zero of it, and fell through to a protective one. The
    manifest cross-reference that decision needed has been removed, so the chip follows
    the payload's own ranking again.
    """
    state = {"tool_calls": [{"name": "predict_readmission", "args": {},
                             "response": _risk([
                                 {"feature": "recent_ed_visits", "direction": "increases"},
                                 {"feature": "prior_inpatient_days",
                                  "direction": "decreases"},
                             ])}],
             "command": "summarize"}

    chips = _chips(state)

    assert chips, "the row came back empty"
    assert any("increase this patient's readmission risk" in chip for chip in chips), chips


def test_a_record_with_a_course_gets_a_question_about_it_even_with_no_chips():
    """The last resort, for a `/ehr` turn whose note overview was its own answer."""
    state = {
        "command": "summarize",
        "tool_calls": [{"name": "rag_search_sections", "args": {}, "response": {
            "returned": 2,
            "passages": [{"id": "p1", "section": "brief_hospital_course",
                          "text": "Treated with IV antibiotics."},
                         {"id": "p2", "section": "discharge_diagnosis",
                          "text": "Pneumonia."}],
        }}],
    }

    chips = _chips(state)

    assert chips, "the row came back empty despite a hospital course"
    assert any("hospital course" in chip for chip in chips), chips


def test_an_empty_record_still_offers_nothing():
    """The fallback is anchored to evidence, so a turn with none offers none."""
    assert _chips({"command": "summarize", "tool_calls": []}) == []


# --- 2. a series that returns to an earlier value was not flat ---------------------

def test_the_prompt_forbids_calling_a_fluctuating_series_flat():
    prompt = " ".join(SYSTEM_PROMPT.split())

    assert "Do not describe a value as holding a level across timepoints" in prompt
    assert "every measurement in between carries that exact value" in prompt
    assert "is a trajectory, not a set of endpoints" in prompt


# --- 4. one index on a passage, and it is the global one --------------------------

def test_a_passage_carries_no_local_identifier():
    """`MT-…_discharge_medications_2` prints a 2 beside a global `^[3]`."""
    import json  # noqa: PLC0415

    rendered = json.loads(_render_tool_result({"passages": [
        {"id": "MT-2616-DS_discharge_medications_2", "section": "discharge_medications",
         "text": "Prednisone 40 mg."},
    ]}))

    passage = rendered["passages"][0]
    assert "id" not in passage
    assert passage["cite"] == 1
    assert passage["cite_header"].startswith("[CITATION ^[1] |")


def test_the_global_number_survives_the_offset_of_a_later_turn():
    import json  # noqa: PLC0415

    rendered = json.loads(_render_tool_result(
        {"passages": [{"id": "x_2", "section": "brief_hospital_course", "text": "t"}]},
        offset=2))

    assert rendered["passages"][0]["cite"] == 3
    assert "[CITATION ^[3] |" in rendered["passages"][0]["cite_header"]


def test_the_prompt_requires_a_marker_on_every_synthesis_paragraph():
    prompt = " ".join(SYSTEM_PROMPT.split())

    assert "every paragraph that states a fact carries the marker" in prompt
    assert "including a paragraph restating an estimate" in prompt
