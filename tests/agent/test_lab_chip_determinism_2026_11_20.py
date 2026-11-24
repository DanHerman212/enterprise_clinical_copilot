"""Deterministic lab driver chips, bounded model attempts, and clean value formatting.

Each block here corresponds to one thing the 2026-11-20 audit found on the live session:

1. two consecutive driver clicks on the same panel drifted apart — the first read its
   analyte and drew a sparkline, the next had the tool withdrawn because a DIFFERENT
   analyte was already in the record, so it answered with no series and no card;
2. that first click paid two model rounds for one answer, and a stalled generation held a
   single span for 21.7 s and 41.4 s with nothing in our logs to explain it;
3. values carried a robotic `(patient value: …)` prefix, a bare `0 days` beside a label
   that already said days, an ASCII cell count, and a one-hot level that `/ehr` dropped
   altogether.

Everything here is offline: no LLM, no MCP, no Vertex, no BigQuery.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessageChunk

from services.agent.engine import graph
from services.agent.engine import model as model_turn
from services.agent.postprocess import guardrail
from services.agent.postprocess.evidence_state import SessionEvidenceState
from services.agent.presentation.a2ui import compose_presentation
from services.agent.routing import admission_context
from shared.labels import analytes_canonical, canonical_analyte


# ---------------------------------------------------------------------------------------
# 1a. The vocabulary a question uses, mapped to the key the panel draws
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("written, key", [
    ("lowest monocyte count", "monocytes"),
    ("red cell distribution width (RDW)", "rdw"),
    ("red blood cells", "rbc"),
    ("hemoglobin", "hemoglobin"),
    ("haemoglobin", "hemoglobin"),
    ("creatinine", "creatinine"),
    ("prior inpatient days", None),
])
def test_a_question_is_mapped_to_the_panels_own_key(written, key):
    assert canonical_analyte(written) == key


def test_two_analytes_in_one_question_are_two_reads():
    assert analytes_canonical(
        "How do the lowest monocyte count and the RDW affect risk?") == ("monocytes", "rdw")


# ---------------------------------------------------------------------------------------
# 1b. A driver click keeps the tool for ITS OWN analyte
# ---------------------------------------------------------------------------------------


def _driver_state(question: str, read: tuple[str, ...]) -> dict:
    record = SimpleNamespace(
        items={}, prediction=None, literature={},
        checked_labs={analyte: "present" for analyte in read},
    )
    return {"question": question, "evidence_state": record}


MONOCYTE_QUESTION = "How does lowest monocyte count affect this patient's readmission risk?"
RDW_QUESTION = "How does red cell distribution width (RDW) affect this patient's risk?"


def test_a_second_driver_click_keeps_the_tool_for_a_different_analyte():
    """The defect: RDW in the record withdrew the tool from a monocyte question.

    Live, the first click bound `get_lab_trends` and read RDW, and the next bound nothing
    at all — `tools: 1` then `tools: 0` on two turns whose `lab_driver` was true, so the
    second answer had no series and drew no card (2026-11-20, 90000069).
    """
    state = _driver_state(MONOCYTE_QUESTION, read=("rdw",))

    assert graph._redundant_on_declared_turn("get_lab_trends", state) is False


def test_the_tool_is_withdrawn_when_the_named_analyte_is_already_read():
    state = _driver_state(RDW_QUESTION, read=("rdw",))

    assert graph._redundant_on_declared_turn("get_lab_trends", state) is True


def test_a_question_naming_no_analyte_is_settled_by_any_laboratory_item():
    """The panel question: the card exists because the course was already drawn."""
    state = _driver_state("Which of his discharge labs was worst?", read=("sodium",))

    assert graph._redundant_on_declared_turn("get_lab_trends", state) is True


# ---------------------------------------------------------------------------------------
# 1c. The analyte is read before the model runs, so the answer is one pass
# ---------------------------------------------------------------------------------------


class _LabBox:
    """A toolbox that records dispatch and answers with one trend."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, dict(arguments)))
        analyte = str(arguments.get("analyte") or "monocytes")
        return {
            "hadm_id": arguments.get("hadm_id"), "returned": 1, "synthetic": True,
            "trends": [{
                "key": analyte, "label": analyte.title(), "unit": "×10³/µL", "decimals": 1,
                "reference_low": 0.2, "reference_high": 0.8, "anchored": [],
                "series": [{"day": 0.0, "value": 0.6, "flag": "normal"},
                           {"day": 2.0, "value": 0.33, "flag": "normal"}],
                "cite": 1, "cite_modality": "STRUCTURED LAB TREND", "cite_header": "^[1]",
            }],
        }


def test_the_named_analyte_is_read_before_the_model_runs():
    box = _LabBox()
    session = SessionEvidenceState(hadm_id=90000069)

    recorded = asyncio.run(graph._prefetch_lab_series(
        box, session, question=MONOCYTE_QUESTION, hadm_id=90000069,
        on_event=None, cite_offset=0,
    ))

    assert box.calls == [("get_lab_trends", {"analyte": "monocytes", "hadm_id": 90000069})]
    # The record holds it, which is what lets `_still_bound` bind no tool at all.
    assert "monocytes" in session.checked_labs
    # And the call is the turn's own evidence, so the card and the guardrails see it.
    assert recorded and recorded[0]["args"]["analyte"] == "monocytes"


def test_an_analyte_the_record_already_holds_is_not_read_again():
    box = _LabBox()
    session = SessionEvidenceState(hadm_id=90000069)
    session.checked_labs["monocytes"] = "present"

    recorded = asyncio.run(graph._prefetch_lab_series(
        box, session, question=MONOCYTE_QUESTION, hadm_id=90000069,
        on_event=None, cite_offset=0,
    ))

    assert recorded == []
    assert box.calls == []


# ---------------------------------------------------------------------------------------
# 1d. The card and the citation channel do not both carry the same series
# ---------------------------------------------------------------------------------------

_MONOCYTE_SERIES = {
    "hadm_id": 90000069, "returned": 2, "synthetic": True,
    "trends": [
        {"key": "rdw", "label": "RDW", "unit": "%", "decimals": 1,
         "reference_low": 11.5, "reference_high": 14.5, "anchored": ["rdw_max"],
         "series": [{"day": 0.0, "value": 11.6, "flag": "normal"},
                    {"day": 30.0, "value": 17.1, "flag": "high"}],
         "cite": 1, "cite_modality": "STRUCTURED LAB TREND", "cite_header": "^[1]"},
        {"key": "monocytes", "label": "Monocytes", "unit": "×10³/µL", "decimals": 1,
         "reference_low": 0.2, "reference_high": 0.8, "anchored": ["monocytes_min"],
         "series": [{"day": 0.0, "value": 0.6, "flag": "normal"},
                    {"day": 2.0, "value": 0.33, "flag": "normal"}],
         "cite": 2, "cite_modality": "STRUCTURED LAB TREND", "cite_header": "^[2]"},
    ],
}
_PANEL_CALL = {"name": "get_lab_trends", "args": {"hadm_id": 90000069},
               "response": _MONOCYTE_SERIES}
_DRIVER_ANSWER = (
    "According to the readmission risk model, lowest monocyte count (0.33 ×10³/µL) "
    "decreases this patient's 30-day readmission risk. It held at 0.33 ×10³/µL on day 2 "
    "^[2].")


def _drawn(out: dict) -> list[dict]:
    components: list[dict] = []
    for message in (out.get("a2ui") or {}).get("messages") or []:
        update = (message or {}).get("updateComponents") or {}
        components.extend(update.get("components") or [])
    return components


def test_a_driver_click_draws_the_series_its_thread_already_holds():
    """No laboratory call this turn, and the card still appears — one analyte wide."""
    out = compose_presentation(
        MONOCYTE_QUESTION, _DRIVER_ANSWER, [], None, None,
        evidence=[_PANEL_CALL], driver_question=True, is_outpatient=False,
    )

    card = next(component for component in _drawn(out)
                if component["component"] == "LabTrendCard")
    assert [trend["key"] for trend in card["trends"]] == ["monocytes"]


def test_the_driver_turn_does_not_also_publish_its_trajectory_as_a_citation():
    """The series is drawn in the card, so the pill and the drawer row would repeat it.

    Live, the RDW driver answer carried `^[1]` and the drawer listed
    "1. RDW trajectory · Clinical Note" under a sparkline that already drew that course
    (2026-11-20, 90000069).
    """
    out = compose_presentation(
        MONOCYTE_QUESTION, _DRIVER_ANSWER, [], None, None,
        evidence=[_PANEL_CALL], driver_question=True, is_outpatient=False,
    )

    assert out["sources"] == []
    assert out["references"] == []
    assert "^[2]" not in out["answer"]
    # Still a chart answer, and still verified: the card IS the evidence.
    assert out["scope"] == "patient_chart"
    assert out["verified"] is True


def test_a_note_citation_beside_a_trajectory_survives():
    """Only the trajectory's markers go; a real note citation is untouched."""
    passage_call = {"name": "rag_search", "args": {"hadm_id": 90000069},
                    "response": {"hadm_id": 90000069, "passages": [
                        {"section": "brief_hospital_course", "text": "Monocytes were low."}]}}
    answer = ("According to the readmission risk model, lowest monocyte count "
              "(0.33 ×10³/µL) decreases this patient's 30-day readmission risk. The course "
              "records a low count ^[1]. The series fell to 0.33 ×10³/µL ^[2].")

    out = compose_presentation(
        MONOCYTE_QUESTION, answer, [passage_call], None, None,
        evidence=[passage_call, _PANEL_CALL], driver_question=True, is_outpatient=False,
    )

    assert "^[1]" in out["answer"]
    assert "^[2]" not in out["answer"]
    assert [source["section"] for source in out["sources"]] == ["brief_hospital_course"]


def test_a_free_text_lab_answer_keeps_its_trajectory_citation():
    """Out of scope, and pinned: a chart answer whose evidence IS the series cites it."""
    out = compose_presentation(
        "How do the discharge red blood cells compare to the reference range?",
        "Monocytes fell to 0.33.^[2]", [_PANEL_CALL], None, None, is_outpatient=False,
    )

    assert any(source.get("section") == "laboratory_trends" for source in out["sources"])


# ---------------------------------------------------------------------------------------
# 2b. With nothing bound, the prose is announced as it arrives
# ---------------------------------------------------------------------------------------


class _StreamingLLM:
    """A model whose chunks arrive one at a time, with a trace of when."""

    def __init__(self, pieces: tuple[str, ...], log: list[tuple[str, str]]) -> None:
        self._pieces = pieces
        self._log = log

    async def astream(self, messages):
        for piece in self._pieces:
            self._log.append(("chunk", piece))
            yield AIMessageChunk(content=piece)


def test_an_unbound_turn_announces_each_chunk_as_it_arrives():
    log: list[tuple[str, str]] = []
    llm = _StreamingLLM(("one ", "two"), log)

    turn = asyncio.run(model_turn.streamed_turn(
        llm, [], lambda event: log.append(("emit", event["text"])), immediate=True))

    assert turn.content == "one two"
    # Each chunk reached the client before the model produced the next one.
    assert log == [("chunk", "one "), ("emit", "one "),
                   ("chunk", "two"), ("emit", "two")]


def test_a_bound_turn_still_holds_its_prose_to_the_end():
    """The deliberation rule is unchanged: a tool call may still withdraw the prose."""
    log: list[tuple[str, str]] = []
    llm = _StreamingLLM(("one ", "two"), log)

    asyncio.run(model_turn.streamed_turn(
        llm, [], lambda event: log.append(("emit", event["text"]))))

    assert [kind for kind, _ in log] == ["chunk", "chunk", "emit", "emit"]


# ---------------------------------------------------------------------------------------
# 3. Values a reader can read: no label, no repeated unit, clinical counts
# ---------------------------------------------------------------------------------------


def test_the_sentence_template_carries_the_value_with_no_label():
    from services.agent.engine import prompts

    spoke = " ".join(prompts.RISK_SPOKE.split())

    assert "<factor> (<value>) increases/decreases this patient's 30-day" in spoke
    assert "patient value:" not in spoke
    # A present flag is named alone; only absence is marked.
    assert "(not present)" in spoke


def test_a_count_whose_label_says_days_carries_no_unit():
    """"prior inpatient days (0 days)" repeated the label; "length of stay" needs one.

    Both are durations, so both keep the served fraction: the label that already says days
    carries no unit, and the one that does not carries it.
    """
    assert admission_context.unit_for("prior_inpatient_days") == ""
    assert admission_context.clinical_measure("prior_inpatient_days", 4.75) == "4.75"
    assert admission_context.clinical_measure("index_los_days", 42.46) == "42.46 days"


@pytest.mark.parametrize("feature, value, stated", [
    ("monocytes_min", 0.33, "0.33 ×10³/µL"),
    ("rbc_last", 3.09, "3.09 ×10⁶/µL"),
    ("wbc", 6.5, "6.5 ×10³/µL"),
])
def test_a_cell_count_is_written_the_way_a_laboratory_writes_it(feature, value, stated):
    from services.agent.engine.mcp_client import _presented_factor

    assert _presented_factor(
        {"feature": feature, "direction": "increases", "value": value})["value"] == stated


def test_a_present_flag_is_stated_by_the_factors_own_name():
    from services.agent.engine.mcp_client import _presented_factor

    assert "value" not in _presented_factor(
        {"feature": "oncology_flag", "direction": "increases", "value": 1})
    assert _presented_factor(
        {"feature": "oncology_flag", "direction": "increases", "value": 0})["value"] \
        == "not present"


def test_the_guardrail_accepts_both_spellings_of_a_cell_count():
    """The prose may carry either form; neither may be stripped or flagged."""
    for unit in ("×10³/µL", "x10^3/uL"):
        answer = f"Her monocytes nadired at 0.33 {unit} on day 2."
        out, flags = guardrail.verify_lab_values(answer, "", "", answer)

        assert out == answer
        assert flags == []


def test_a_cell_count_sentence_is_checked_against_the_record():
    """The Unicode form is a measurement, so an unsupported one is still caught."""
    answer = "Her monocytes nadired at 0.21 ×10³/µL on day 2."

    out, flags = guardrail.verify_lab_values(answer, "", "", "monocytes 0.33 ×10³/µL")

    assert out.strip() == ""
    assert flags == ["lab_value_unsupported:0.21 ×10³/µL"]


# ---------------------------------------------------------------------------------------
# 3b. `/ehr` states the one-hot level the block holds
# ---------------------------------------------------------------------------------------


class _CommandBox:
    """A toolbox that answers the estimate and nothing else."""

    def __init__(self, predict: dict) -> None:
        self.predict = predict
        self.calls: list[tuple[str, dict]] = []

    async def call(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, dict(arguments)))
        if name == "predict_readmission":
            return self.predict
        return {"hadm_id": arguments.get("hadm_id"), "passages": []}


class _FormattingModel:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def bind_tools(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the formatting turn was given tools")

    async def ainvoke(self, messages, config=None):
        from langchain_core.messages import AIMessage

        return AIMessage(content=self.reply)


def test_the_ehr_briefing_projects_the_estimate_with_the_blocks_levels(monkeypatch):
    """`discharge destination` is a one-hot parent, so its value exists only in the block.

    The free-text loop has always handed the projection those values; the command path did
    not, so a briefing listed "discharge destination" with no value beside factors that all
    had one (measured 2026-11-20, 90000069).
    """
    from services.agent.engine import deterministic
    from services.agent.routing import routing

    seen: dict = {}
    real = deterministic._presented_for_model

    def spy(name, payload, administrative=None):
        if name == "predict_readmission":
            seen["administrative"] = administrative
        return real(name, payload, administrative)

    monkeypatch.setattr(deterministic, "_presented_for_model", spy)
    monkeypatch.setattr(model_turn, "build_llm",
                        lambda name, **_kw: _FormattingModel("briefing"))

    block = admission_context.render(
        90000069, {"admission_type": "Emergency (EW EMER.)", "discharge_location": "Home"})
    box = _CommandBox({"probability": 0.09, "threshold": 0.11, "top_factors": [
        {"feature": "discharge_location", "direction": "decreases"}]})

    asyncio.run(deterministic.run_command(
        box, routing.command_for("summarize"), 90000069,
        question="Summarize the recent discharge notes for this patient.",
        model="test-model", admission_context=block,
    ))

    assert seen["administrative"]["discharge_location"] == "Home"
    projected = real("predict_readmission", {"top_factors": [
        {"feature": "discharge_location", "direction": "decreases"}]},
        seen["administrative"])
    assert projected["top_factors"][0]["value"] == "Home"
