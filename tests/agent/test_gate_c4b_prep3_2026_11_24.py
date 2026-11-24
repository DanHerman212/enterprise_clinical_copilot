"""Gate C4b-preparation 3: the four fixes the latency diagnostic asked for.

The C4b slice against `93b7066` produced four complaints, and the latency pass found the
cause of the worst of them: a model call bounded at 60 s with three SDK retries is a ladder
up to four minutes long, and the ask deadline is 110 s, so a provider `504
DEADLINE_EXCEEDED` was served to the caller as a timeout that the retry itself had caused.

Four behaviours, one file:

1. the retry budget fits inside the ask deadline, and a timed-out turn can still name its
   trace;
2. a served fraction is stated as served, in the manifest, the record and the guardrail;
3. a free-text turn keeps its laboratory markers beside the laboratory records it
   publishes, and only a driver turn trades them for the card;
4. the medication extraction is told what to do with an arrival list, so a drug the record
   stopped is stated as stopped rather than dropped.
"""

from __future__ import annotations

from services.agent import turn
from services.agent.engine import graph
from services.agent.engine import model as model_gateway
from services.agent.engine import prompts
from services.agent.postprocess.guardrail import normalise_factor_values
from services.agent.routing import admission_context
from services.agent.telemetry import observability
from services.mcp.runtime import timeout_chain

# ------------------------------------------------------- 1. the retry budget and the trace id


def test_the_retry_ladder_finishes_inside_the_ask_deadline():
    """Three attempts at 30 s, and room to spare before the question is cut off.

    The numbers are a budget, not a preference. At the values this replaced — 60 s and
    three retries — the worst case was 240 s inside a 110 s deadline, which is how
    `dev_d4_risk_01` turns 3 and 4 and `dev_d5_fup_01` turn 4 came to be served an HTTP 504
    by their own retry on 2026-11-24.
    """
    chain = timeout_chain()

    assert chain.model == 30.0
    assert model_gateway.MODEL_MAX_RETRIES == 2
    ladder = (model_gateway.MODEL_MAX_RETRIES + 1) * chain.model
    assert ladder < chain.ask
    assert ladder <= chain.ask - 10


def test_the_turn_trace_id_is_taken_from_the_span_before_anything_is_awaited():
    """A turn cancelled by the deadline never reached the old assignment.

    The id is a property of the span, so it exists the moment the span opens. Taking it
    there — rather than after the answer settles — is what lets a timed-out turn report
    which trace to read.
    """
    span = type("Span", (), {"trace_id": "9a1c2b3d4e5f"})()
    assert graph._turn_trace_id(span, None) == "9a1c2b3d4e5f"
    # No span at all is not an error: the ambient context is the last resort, and an empty
    # string is how "no trace" is spelled.
    assert graph._turn_trace_id(None, None) == observability.current_trace_id()


def test_a_failure_record_carries_the_trace_the_turn_was_written_on(monkeypatch):
    """The record for a timeout used to say `"langfuse_trace_id": ""`.

    Every exit from a request records one line, and the FAILURE exits never had a state to
    read the id off. The route calls the recorder while the turn span is still open, so the
    ambient context still knows it — which is where the recorder now looks.
    """
    monkeypatch.setattr(observability, "current_trace_id", lambda: "0f1e2d3c4b5a")

    record = turn.StageLog(intent="risk_driver").record(
        "trace-tag", "what is her risk", "timeout", error="timeout")

    assert record["langfuse_trace_id"] == "0f1e2d3c4b5a"
    assert record["outcome"] == "timeout"


def test_a_success_record_keeps_the_id_it_was_handed(monkeypatch):
    """The fallback must not overwrite the id the turn already resolved from its span."""
    monkeypatch.setattr(observability, "current_trace_id", lambda: "context-id")

    record = turn.StageLog().record(
        "trace-tag", "question", "ok", langfuse_trace_id="span-id")

    assert record["langfuse_trace_id"] == "span-id"


# -------------------------------------------- 2. a served fraction is stated as it was served


def test_a_whole_served_value_is_written_whole_and_a_fraction_keeps_its_decimals():
    """The rule follows the VALUE, not the feature's name.

    `medication_order_count` was rounded half up because the name promises a tally. The
    encoder holds an order rate there, so a served 13.64 was stated as 14 and
    `dev_d2_risk_01` was docked for it on 2026-11-24: the answer disagreed with the risk
    result drawn beside it.
    """
    assert admission_context.clinical_value("medication_order_count", 14.0) == 14
    assert admission_context.clinical_value("medication_order_count", 0.0) == 0
    assert admission_context.clinical_value("medication_order_count", 13.64) == 13.64
    assert admission_context.clinical_value("medication_order_count", 4.85) == 4.85
    assert admission_context.clinical_value("prior_inpatient_days", 4.75) == 4.75
    assert admission_context.clinical_value("index_los_days", 42.458333333333336) == 42.46


def test_the_manifest_states_a_count_at_the_precision_it_was_served():
    """`prior_admissions` 2 is "2" and 2.5 stays "2.5": the block states the row.

    The manifest is the ground truth the judge holds the answer against, so it is the first
    place a rounded value stops agreeing with the record.
    """
    block = admission_context.render(
        90000020, {"prior_admission_count": "2", "index_los_days": "13.64"})

    assert "prior_admissions: 2" in block
    assert "length_of_stay_days: 13.64" in block


def test_the_guardrail_leaves_a_two_decimal_fraction_alone_and_trims_a_long_float():
    """Two decimals is the published precision; more than that is the encoder's float."""
    calls = [{"name": "predict_readmission", "response": {"top_factors": [
        {"feature": "medication_order_count", "direction": "decreases", "value": 13.64},
        {"feature": "index_los_days", "direction": "increases",
         "value": 42.458333333333336},
    ]}}]
    answer = "Her 13.64 medication orders and 42.458333333333336 days both counted."

    out, flags = normalise_factor_values(answer, calls)

    assert "13.64 medication orders" in out
    assert "42.46 days" in out
    assert flags == ["factor_value_rounded:42.458333333333336"]


def test_a_whole_served_value_is_not_rewritten_at_all():
    calls = [{"name": "predict_readmission", "response": {"top_factors": [
        {"feature": "recent_ed_visits", "direction": "increases", "value": 4.0}]}}]
    answer = "She had 4.0 emergency visits, on day 4.0 of the stay."

    out, flags = normalise_factor_values(answer, calls)

    assert out == answer
    assert flags == []


# ------------------------------------------------- 3. a laboratory marker and its record stay


def test_a_free_text_turn_keeps_its_markers_beside_its_laboratory_records():
    """The markers and the records are one statement, so they are published together.

    A revision stripped the markers into a trajectory on every turn that was not a driver
    turn, and left the `LAB_SECTION` records in `shown_sources`. The judge saw sentences
    citing nothing beside three laboratory sources and reported exactly that split
    (`dev_d5_slash_01` turn 5, 2026-11-24).
    """
    import inspect

    from services.agent.presentation import a2ui

    source = inspect.getsource(a2ui.compose_presentation)

    # The stripping is inside the driver branch, and there is no second branch doing it.
    assert "stray_lab_cites" not in source
    assert source.count("lab_cites") >= 1
    driver_at = source.index("if labs is not None and driver_question:")
    strip_at = source.index("_without_markers(answer, lab_cites)")
    assert driver_at < strip_at


# --------------------------------------------------------- 4. an arrival list, and what it means


def test_the_medication_instruction_says_what_to_do_with_a_transfer_list():
    text = " ".join(prompts.MEDS_FORMATTING.split())

    assert "ARRIVED on" in text
    assert '"setting" "inpatient"' in text
    assert "stopped or discontinued" in text
    # Still inside the word ceiling the contract test owns.
    assert len(prompts.MEDS_FORMATTING.split()) <= 750


def test_the_medication_instruction_still_refuses_ward_treatment():
    text = " ".join(prompts.MEDS_FORMATTING.split())

    assert "Exclude what was only given in hospital" in text
    assert "a drug on the arrival list" in text.lower() or "arrival list" in text.lower()
