"""A laboratory trajectory is evidence a turn can cite, and it is read once.

Two failures on one path, reported from the production UI on 2026-10-26:

  * A lab follow-up after `/labs` spent eighteen steps, three of them more
    `get_lab_trends(analyte=…)` calls against a panel that was already in the transcript,
    and re-rendered the `LabTrends` card each time the panel came back.
  * And the answer grew a second paragraph drawn from the discharge note that nobody
    asked for. The cause was not narration: `TF-10` requires every clinical answer to
    carry a `^[n]`, and `get_lab_trends` returned a series the citation layer did not
    number. The model had no citable evidence in front of it, so it went and found some —
    a note passage it could cite for having described the labs.

The rule both come back to: evidence a turn is expected to cite has to be citable, and a
tool whose result is a fact about the admission does not get a second turn to say it
again. Offline — no MCP, no BigQuery, no model.
"""

from __future__ import annotations

import json

from services.agent import citations, graph
from services.agent.a2ui import compose_presentation
from services.agent.note_presentation import resolve_sources

#: The panel as the lab tool returns it: one entry per analyte, with the series, the
#: reference interval and the anchors the generator reproduced.
LABS = {
    "hadm_id": 90000023,
    "synthetic": True,
    "note": "Values are synthetic.",
    "trends": [
        {"key": "creatinine", "label": "Creatinine", "unit": "mg/dL",
         "reference_low": 0.5, "reference_high": 1.1, "decimals": 2,
         "anchored": ["creatinine_max"],
         "series": [{"day": 0.0, "value": 1.05, "flag": "normal"},
                    {"day": 4.0, "value": 1.62, "flag": "high"}]},
        {"key": "sodium", "label": "Sodium", "unit": "mmol/L",
         "reference_low": 135.0, "reference_high": 145.0, "decimals": 0,
         "anchored": ["sodium_min"],
         "series": [{"day": 0.0, "value": 133.0, "flag": "low"},
                    {"day": 4.0, "value": 139.0, "flag": "normal"}]},
    ],
}

LAB_CALL = {"name": "get_lab_trends", "args": {"hadm_id": 90000023},
            "response": LABS}

NOTES = {
    "query": "sections",
    "returned": 1,
    "passages": [{"id": "p1", "section": "discharge_diagnosis",
                  "text": "Discharge Diagnosis: Acute decompensated heart failure"}],
}

NOTE_CALL = {"name": "rag_search_sections", "args": {"hadm_id": 90000023},
             "response": NOTES}

PREDICT_CALL = {"name": "predict_readmission", "args": {"hadm_id": 90000023},
                "response": {"hadm_id": 90000023, "probability": 0.2,
                             "threshold": 0.11, "decision": 1,
                             "model_version": "test-v1"}}


def _types(out: dict) -> list[str]:
    return [c["component"]
            for c in out["a2ui"]["messages"][1]["updateComponents"]["components"]]


# --- the numbering: one analyte, one number -----------------------------------

class TestTheAnalytesCarryNumbers:
    def test_each_analyte_gets_a_citation_number_the_model_can_write(self):
        """The defect, at the layer that caused it: the model was shown a series with no
        numbers on it and told every clinical claim must cite one."""
        rendered = json.loads(graph._render_tool_result(LABS))
        assert [trend["cite"] for trend in rendered["trends"]] == [1, 2]

    def test_the_numbering_continues_across_tool_results(self):
        """Four note sections then a panel: the panel starts at 5, not at 1."""
        rendered = json.loads(graph._render_tool_result(LABS, offset=4))
        assert [trend["cite"] for trend in rendered["trends"]] == [5, 6]

    def test_one_definition_of_a_citable_unit(self):
        """Two layers count these and must agree, so there is one function and both call
        it. A retrieval payload contributes a unit per passage; a panel one per analyte; a
        prediction none at all."""
        assert citations.passage_total(LABS) == 2
        assert citations.passage_total(NOTES) == 1
        assert citations.passage_total(PREDICT_CALL["response"]) == 0
        assert graph._passage_total(LABS) == citations.passage_total(LABS)


# --- the citation: opening a lab marker shows the trajectory ------------------

class TestALabCitationOpens:
    def test_a_marker_resolves_to_the_analyte_with_its_range(self):
        sources = resolve_sources("Creatinine rose ^[1]", {}, None, (),
                                  evidence=[LAB_CALL])

        assert len(sources) == 1
        source = sources[0]
        assert source["section"] == citations.LAB_SECTION
        assert source["cite_title"] == "Creatinine trajectory"
        # The values, the days and the interval they were read against — a marker that
        # opened onto a label would be a citation to a heading.
        assert "1.05" in source["text"] and "1.62" in source["text"]
        assert "Reference interval 0.50-1.10 mg/dL" in source["text"]

    def test_the_second_marker_resolves_to_the_second_analyte(self):
        """The offset is the claim: `^[2]` is sodium, because the numbering the model was
        given is the array this resolves against."""
        sources = resolve_sources("Sodium fell ^[2]", {}, None, (), evidence=[LAB_CALL])

        assert sources[0]["section"] == citations.LAB_SECTION
        assert sources[0]["cite_title"] == "Sodium trajectory"
        assert "133" in sources[0]["text"]

    def test_a_lab_turn_is_a_verified_chart_answer_with_no_note_passage(
            self):
        """What the second paragraph was for: without this, a lab answer had no citable
        evidence, so the badge was withheld too — a chart answer reported as resting on
        nothing."""
        out = compose_presentation(
            "How do the discharge red blood cells compare to the reference range?",
            "Creatinine rose from 1.05 to 1.62 mg/dL.^[1]",
            [LAB_CALL],
        )

        assert out["verified"] is True
        assert out["scope"] == "patient_chart"
        assert out["scope_suppressed"] is False
        assert out["sources"][0]["section"] == citations.LAB_SECTION
        # The drawer row names the trajectory rather than calling it a discharge note.
        assert out["references"][0]["source"] == citations.LAB_SOURCE
        assert out["references"][0]["title"] == "Creatinine trajectory"
        # And the contract's own requirement for a non-literal source.
        assert isinstance(out["sources"][0]["query"], str)

    def test_a_follow_up_that_quotes_the_panel_cites_it_without_redrawing_it(self):
        """Both halves of the report, in one turn: the marker resolves, and the card that
        was already drawn three turns ago is not drawn again."""
        out = compose_presentation(
            "What was the peak creatinine?",
            "It peaked at 1.62 mg/dL.^[2]",
            [NOTE_CALL],
            evidence=[LAB_CALL, NOTE_CALL],
        )

        assert "LabTrendCard" not in _types(out)
        assert any(section["section"] == citations.LAB_SECTION
                   for section in out["sources"])


# --- the cap: a panel already read is not read again --------------------------

class TestThePanelIsReadOnce:
    def test_a_full_panel_in_the_transcript_leaves_nothing_to_bind(self):
        """The eighteen steps: after `/labs` the whole panel was in the transcript and the
        tool was still offered."""
        state = {"tool_calls": [], "replayed_tool_calls": [LAB_CALL]}

        assert "get_lab_trends" not in graph._still_bound(
            frozenset({"get_lab_trends", "rag_search"}), state)

    def test_an_analyte_read_alone_leaves_the_rest_of_the_panel(self):
        """The qualification that keeps this a bound rather than a ban: a turn shown only
        creatinine has not seen sodium, and a question about sodium is a new question."""
        state = {"tool_calls": [], "replayed_tool_calls": [
            {"name": "get_lab_trends", "args": {"hadm_id": 90000023,
                                                "analyte": "creatinine"},
             "response": LABS},
        ]}

        assert "get_lab_trends" in graph._still_bound(
            frozenset({"get_lab_trends"}), state)

    def test_the_panel_is_spent_after_one_call_in_the_turn(self):
        """`ONCE_PER_TURN` is the second lock: a single model turn can ask twice, and a
        refusal is something a model can retry. The absence of the tool is not."""
        spent = {"tool_calls": [LAB_CALL], "replayed_tool_calls": []}

        assert "get_lab_trends" not in graph._still_bound(
            frozenset({"get_lab_trends"}), spent)

    def test_the_medication_list_is_read_once_per_turn_and_per_conversation(self):
        """`discharge_medications` returns the whole discharge list whatever it is asked,
        so a second call cannot return anything new — in the turn or after `/meds`."""
        meds = {"name": "discharge_medications", "args": {"hadm_id": 90000023},
                "response": {"medications": []}}
        allowed = frozenset({"discharge_medications"})

        assert "discharge_medications" not in graph._still_bound(
            allowed, {"tool_calls": [meds], "replayed_tool_calls": []})
        assert "discharge_medications" not in graph._still_bound(
            allowed, {"tool_calls": [], "replayed_tool_calls": [meds]})

    def test_the_bounds_do_not_touch_a_tool_that_takes_a_new_question(self):
        """The regression guard for the other direction: the transcript rule is about
        FIXED results, and a tool that answers the question it was asked stays available.

        A follow-up may search the note again — it is asking something new of it — and a
        turn that is about published records may search the literature whatever an earlier
        turn searched for. Only this turn's own spending closes `rag_search`, which is the
        per-turn cap and not the transcript one.
        """
        allowed = frozenset({"rag_search", "search_literature", "predict_readmission"})
        replayed_only = {
            "tool_calls": [],
            "replayed_tool_calls": [
                {"name": "rag_search", "args": {"query": "a"}},
                {"name": "search_literature", "args": {"query": "b"}},
            ],
        }

        assert graph._still_bound(allowed, replayed_only) == allowed

        spent_this_turn = {
            "tool_calls": [{"name": "rag_search", "args": {"query": "c"}}],
            "replayed_tool_calls": [],
        }
        assert "rag_search" not in graph._still_bound(allowed, spent_this_turn)
