"""The four global fixes of 2026-11-06, each pinned at the seam it changed.

Round two of the remediation (`master_remediation_plan.md`). Every test names the live
failure it was written for, because each of these was a behaviour that passed a
shape-level check and failed a reader:

  * a verifier deleting a sentence the corpus could support,
  * a citation deleted from a sentence that made its own claim,
  * a tool description naming an argument the schema had withheld,
  * a turn thrown away by the closing of a socket.

Offline — no MCP, no model, no network.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from services.agent.engine import mcp_client
from services.agent.engine import prompts
from services.agent.postprocess import citations
from services.agent.postprocess import guardrail
from services.agent.postprocess.evidence_state import SessionEvidenceState

HADM = 90000069

MANIFEST = (
    '<admission_metadata source="feature store row scored by the readmission model">\n'
    "- age: 41\n"
    "- discharge_location: Home\n"
    "- length_of_stay_days: 42.4583\n"
    "</admission_metadata>"
)


# --------------------------------------------------------------------------------------
# Fix 1 — one session corpus, and the manifest read as a FIELD.
# --------------------------------------------------------------------------------------


def test_the_manifest_field_is_read_rather_than_searched_for_a_phrase():
    assert guardrail._manifest_field(MANIFEST, "discharge_location") == "Home"
    assert guardrail._manifest_field(MANIFEST, "length_of_stay_days") == "42.4583"
    assert guardrail._manifest_field(MANIFEST, "absent_field") == ""


@pytest.mark.parametrize("value,expected", [
    ("Home", {"home"}),
    ("Home with home health", {"home"}),
    ("Skilled Nursing Facility", {"skilled_nursing"}),
    ("Inpatient Rehabilitation Facility", {"rehabilitation"}),
    ("Hospice", {"hospice"}),
    ("Long-term Acute Care Hospital", {"long_term_acute"}),
])
def test_a_stated_destination_is_a_supported_disposition(value, expected):
    block = f"- discharge_location: {value}"

    assert guardrail._manifest_dispositions(block) == expected


def test_the_disposition_sentence_survives_when_the_manifest_states_it():
    """The 2026-11-06 follow-up: `disposition_unsupported:home` deleted the discharge
    paragraph while `discharge_location: Home` sat in the manifest."""
    answer = (
        "**Clinical course.** She underwent an anterior cervical discectomy ^[2].\n\n"
        "**Discharge.** The patient was discharged to home ^[4]."
    )

    kept, flags = guardrail.verify_dispositions(
        answer, MANIFEST, guardrail._manifest_dispositions(MANIFEST))

    assert "discharged to home" in kept
    assert flags == []


def test_a_disposition_the_corpus_does_not_state_is_still_removed():
    """Unifying the corpus is not a licence to stop checking: a destination nobody
    stated is still an invented claim."""
    answer = "**Plan.** She was discharged to a skilled nursing facility ^[4]."

    kept, flags = guardrail.verify_dispositions(
        answer, MANIFEST, guardrail._manifest_dispositions(MANIFEST))

    assert "skilled nursing facility" not in kept
    # And the section that lost its only sentence is reported as one, not silently
    # dropped: the paragraph is gone and the reader is owed the reason.
    assert set(flags) == {"disposition_unsupported:skilled_nursing",
                          "section_without_sentences"}


def _prediction(**overrides) -> dict:
    payload = {
        "hadm_id": HADM,
        "probability": 0.206685,
        "probability_display": "20.7%",
        "threshold": 0.11,
        "threshold_display": "11.0%",
        "decision": 1,
        "top_factors": [
            {"feature": "prior_inpatient_days", "contribution": 0.31,
             "direction": "increases", "value": 4.75},
        ],
    }
    payload.update(overrides)
    return payload


def test_the_score_and_threshold_survive_a_turn_with_no_tool_calls():
    """The record is the corpus: a follow-up answering from the session's own estimate
    is not inventing a number. `tool_calls` is empty here on purpose."""
    record = SessionEvidenceState(hadm_id=HADM)
    record.ingest_tool_output("predict_readmission", {"hadm_id": HADM}, _prediction())
    answer = "Her risk is 0.206685, above the 0.11 operating threshold."

    result = guardrail.guard_answer(
        answer, [], [], literature_replay=False, finish_reason=None,
        state={"hadm_id": HADM, "evidence_state": record,
               "admission_context": MANIFEST, "tool_calls": []},
    )

    assert "0.206685" in result["answer"]
    assert "0.11" in result["answer"]


def test_a_number_no_channel_states_is_still_stripped():
    """The control. If the corpus change made the risk guard inert, this passes."""
    record = SessionEvidenceState(hadm_id=HADM)
    record.ingest_tool_output("predict_readmission", {"hadm_id": HADM}, _prediction())

    result = guardrail.guard_answer(
        "Her risk is 0.72.", [], [], literature_replay=False, finish_reason=None,
        state={"hadm_id": HADM, "evidence_state": record,
               "admission_context": MANIFEST, "tool_calls": []},
    )

    assert "0.72" not in result["answer"]


def test_a_record_unit_rounds_out_the_corpus_where_the_call_lost_its_payload():
    """The record keeps the unit when a replay arrives without one. A medication dose it
    holds is a dose the answer may state."""
    record = SessionEvidenceState(hadm_id=HADM)
    record.ingest_tool_output("discharge_medications", {"hadm_id": HADM}, {
        "hadm_id": HADM, "returned": 1,
        "passages": [{"id": "m1", "section": "discharge_medications",
                      "text": "Celebrex 200 mg daily. Colace 100 mg b.i.d."}],
    })

    result = guardrail.guard_answer(
        "She continues Celebrex 200 mg daily.", [], [], literature_replay=False,
        finish_reason=None,
        state={"hadm_id": HADM, "evidence_state": record,
               "admission_context": MANIFEST, "tool_calls": []},
    )

    assert "200 mg" in result["answer"]


# --------------------------------------------------------------------------------------
# Fix 2 — a citation belongs to a SENTENCE.
# --------------------------------------------------------------------------------------


def test_two_sentences_on_one_passage_each_keep_their_marker():
    answer = "She had shoulder pain ^[4]. It was managed with morphine ^[4]."

    collapsed, flags = citations.collapse_repeated_citations(answer)

    assert collapsed == answer
    assert flags == []


def test_a_double_stamp_inside_one_sentence_still_collapses():
    answer = "The discharge was uneventful ^[4] and uncomplicated ^[4]."

    collapsed, flags = citations.collapse_repeated_citations(answer)

    assert collapsed == "The discharge was uneventful and uncomplicated ^[4]."
    assert flags == ["citation_repeat_collapsed:^4"]


# --------------------------------------------------------------------------------------
# Fix 3 — the withheld argument, out of the description and out of the schema's silence.
# --------------------------------------------------------------------------------------


def test_the_description_stops_naming_the_withheld_argument():
    text = "Read the discharge medication list. Requires hadm_id. Returns rows."

    assert mcp_client._model_description(text) == (
        "Read the discharge medication list. Returns rows.")


class _AdmittedTool:
    name = "discharge_medications"
    description = "Read the discharge medication list. Requires hadm_id."
    input_schema = {"type": "object",
                    "properties": {"hadm_id": {"type": "integer"}},
                    "required": ["hadm_id"]}


def test_the_model_schema_can_carry_a_closed_object(monkeypatch):
    """The capability, and the reason it is behind a flag.

    `additionalProperties` is not in the SDK's Schema message, so whether Gemini accepts
    it is one live call away. The decision this test protects is that the flag closes
    EVERY branch — including the tool left with no properties, which is the one the model
    reads for `discharge_medications`.
    """
    monkeypatch.setattr(mcp_client, "_CLOSE_MODEL_SCHEMA", True)
    monkeypatch.setattr(mcp_client, "_ALLOWED_SCHEMA_KEYS",
                        mcp_client._ALLOWED_SCHEMA_KEYS | {"additionalProperties"})

    empty = mcp_client._model_schema("discharge_medications", _AdmittedTool.input_schema)
    partial = mcp_client._model_schema("get_lab_trends", {
        "type": "object",
        "properties": {"hadm_id": {"type": "integer"}, "analyte": {"type": "string"}},
    })

    assert empty == {"type": "object", "properties": {}, "additionalProperties": False}
    assert partial["additionalProperties"] is False
    assert "hadm_id" not in partial["properties"]


def test_the_flag_is_off_until_the_live_call_is_made():
    """Sending a keyword the SDK cannot carry would 400 every bound tool at once."""
    assert mcp_client._CLOSE_MODEL_SCHEMA is False


def test_the_hub_forbids_writing_the_admission_even_as_null():
    assert "not even as `null`" in prompts.CORE_HUB_PROMPT


# --------------------------------------------------------------------------------------
# Fix 4 — the transport, at the session seam.
# --------------------------------------------------------------------------------------


def test_a_group_of_transport_failures_is_transient():
    group = BaseExceptionGroup("stream", [RuntimeError("closed"), OSError("reset")])

    assert mcp_client._transient_transport(group) is True


def test_a_group_holding_a_bug_is_not_transient():
    group = BaseExceptionGroup("mixed", [RuntimeError("closed"), ValueError("schema")])

    assert mcp_client._transient_transport(group) is False


def test_a_cancellation_is_never_transient():
    assert mcp_client._transient_transport(asyncio.CancelledError()) is False
    assert mcp_client._transient_transport(
        BaseExceptionGroup("cancel", [asyncio.CancelledError()])) is False


class _FakeSession:
    def __init__(self, *args, **kwargs):
        pass

    async def initialize(self):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        raise BaseExceptionGroup(
            "teardown", [RuntimeError("generator didn't stop after athrow()")])


@contextlib.asynccontextmanager
async def _fake_stream(*args, **kwargs):
    yield ("read", "write")


class _FakeBox:
    async def load(self):
        return self


def _run_toolbox(monkeypatch, body):
    monkeypatch.setattr(mcp_client, "_cached_id_token", lambda audience: "token")
    monkeypatch.setattr(mcp_client, "streamable_http_client", _fake_stream)
    monkeypatch.setattr(mcp_client, "ClientSession", _FakeSession)

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(mcp_client.httpx2, "AsyncClient", _Client)
    monkeypatch.setattr(mcp_client, "MCPToolbox", lambda session: _FakeBox())
    return asyncio.run(body())


def test_a_teardown_after_a_completed_body_does_not_lose_the_turn(monkeypatch):
    async def body():
        async with mcp_client.http_toolbox("https://example.invalid"):
            pass
        return "turn survived"

    assert _run_toolbox(monkeypatch, body) == "turn survived"


def test_a_failure_inside_the_body_is_still_raised(monkeypatch):
    """The teardown suppression must not become a blanket swallow."""
    async def body():
        async with mcp_client.http_toolbox("https://example.invalid"):
            raise ValueError("the turn itself failed")

    with pytest.raises(BaseException):
        _run_toolbox(monkeypatch, body)
