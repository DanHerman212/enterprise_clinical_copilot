"""Stage 4a — a capacity refusal must cost the reader time, not the answer.

`gemini-3.1-flash-lite` is reached on the multi-region endpoint, where
`429 RESOURCE_EXHAUSTED` arrives in bursts. `langchain_google_genai` retries 5xx and raises
429 immediately, so a burst reached the clinician as "the agent failed to answer" — observed
2026-11-08, on `/ehr` and on `/meds`, while a free-text turn between them answered normally.

Two behaviours are pinned here, and they are different promises:

  * the call that produces the answer (the formatting turn) is retried, because a slower
    answer is still an answer;
  * the two calls that only enrich it (target and comparator extraction) degrade to their
    defined empty state once retries are spent, because the briefing, the estimate and the
    narrative do not depend on them.

Everything is offline: the model doubles raise the provider's own refusal shape.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from services.agent.engine import deterministic
from services.agent.routing import routing


class _ResourceExhausted(Exception):
    """The refusal as `langchain_google_genai` raises it: a status code and a message."""

    code = 429

    def __init__(self) -> None:
        super().__init__(
            "Error calling model (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
            "{'error': {'code': 429, 'status': 'RESOURCE_EXHAUSTED'}}"
        )


@pytest.fixture(autouse=True)
def _no_real_waiting(monkeypatch):
    """Record the pauses instead of taking them, and prove they happened."""
    pauses: list[tuple[int, str]] = []

    async def _pause(attempt: int, what: str) -> None:
        pauses.append((attempt, what))

    monkeypatch.setattr(deterministic, "_pause_before_retry", _pause)
    return pauses


class _Refusing:
    """A model that refuses for capacity `failures` times, then answers."""

    def __init__(self, failures: int, reply: str = '{"condition": "cervical spondylosis"}',
                 error: Exception | None = None) -> None:
        self.failures = failures
        self.reply = reply
        self.error = error or _ResourceExhausted()
        self.calls = 0

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return AIMessage(content=self.reply)


# --------------------------------------------------------------------------------------
# What counts as a capacity refusal
# --------------------------------------------------------------------------------------


def test_the_provider_refusal_is_recognised_by_its_status():
    assert deterministic._rate_limited(_ResourceExhausted())


def test_the_provider_refusal_is_recognised_by_its_message_alone():
    class _Bare(Exception):
        pass

    assert deterministic._rate_limited(_Bare("429 RESOURCE_EXHAUSTED"))


def test_a_content_refusal_is_not_a_capacity_refusal():
    """A safety block must surface as itself; retrying it only makes the failure slower."""
    assert not deterministic._rate_limited(ValueError("blocked by safety filters"))
    assert not deterministic._rate_limited(KeyError("comparators"))


# --------------------------------------------------------------------------------------
# Option 1 — the structured reply retries, then reports rather than raises
# --------------------------------------------------------------------------------------


def test_a_throttled_structured_reply_is_retried_and_succeeds(_no_real_waiting):
    llm = _Refusing(failures=1)

    payload, why = asyncio.run(deterministic._structured_reply(llm, "sys", "human", None))

    assert why == ""
    assert payload == {"condition": "cervical spondylosis"}
    # One refusal costs one attempt, and the second attempt is the one that lands. The
    # ceiling is two (Stage 8, 2026-11-09): this step is an enrichment whose failure is
    # already handled, so it is not worth four attempts and ten seconds of the reader's
    # time — measured at 34.7 s of a 67.6 s `/ehr` on `90000069`.
    assert llm.calls == 2, "the refusal did not cost an attempt"
    assert [attempt for attempt, _ in _no_real_waiting] == [1]


def test_a_structured_reply_that_never_gets_through_reports_capacity(_no_real_waiting):
    """It reports; it does not raise. The callers below have a defined answer for this."""
    llm = _Refusing(failures=99)

    payload, why = asyncio.run(deterministic._structured_reply(llm, "sys", "human", None))

    assert payload is None
    assert why == deterministic.REPLY_RATE_LIMITED
    assert llm.calls == deterministic.STRUCTURED_MAX_ATTEMPTS
    assert "refused for capacity" in deterministic.reply_problem(why)


def test_a_non_capacity_failure_is_not_retried(_no_real_waiting):
    llm = _Refusing(failures=99, error=ValueError("the request was malformed"))

    with pytest.raises(ValueError):
        asyncio.run(deterministic._structured_reply(llm, "sys", "human", None))

    assert llm.calls == 1, "a malformed request was retried"
    assert _no_real_waiting == []


# --------------------------------------------------------------------------------------
# Option 1 — the formatting turn is the answer, so it is retried harder
# --------------------------------------------------------------------------------------


def test_the_formatting_turn_is_retried_before_it_answers(_no_real_waiting):
    llm = _Refusing(failures=1, reply="**Presentation & diagnosis.** Stable.")

    response = asyncio.run(deterministic._formatting_turn(
        llm, [], config=None, announce=None))

    assert "Stable" in response.content
    assert llm.calls == 2


class _StreamsThenRefuses:
    """Streams `chunks_before_failure` pieces, then raises the provider's refusal.

    A later call streams `chunks_on_success` pieces and succeeds, so the same double can
    express both a refusal at the front of the turn and a refusal part-way through it.
    """

    def __init__(self, chunks_before_failure: int = 0, chunks_on_success: int = 2,
                 fail_on_call: int = 1) -> None:
        self.chunks_before_failure = chunks_before_failure
        self.chunks_on_success = chunks_on_success
        self.fail_on_call = fail_on_call
        self.calls = 0

    async def astream(self, prompt, config=None):
        self.calls += 1
        failing = self.calls == self.fail_on_call
        count = self.chunks_before_failure if failing else self.chunks_on_success
        for index in range(count):
            yield AIMessageChunk(content=f"piece{index} ")
        if failing:
            raise _ResourceExhausted()


def test_a_stream_failure_before_the_first_frame_is_retried(_no_real_waiting):
    """Nothing has been announced, so a second attempt cannot duplicate anything."""
    llm = _StreamsThenRefuses(chunks_before_failure=0, fail_on_call=1)
    announced: list[str] = []

    response = asyncio.run(deterministic._formatting_turn(
        llm, [], config=None, announce=announced.append))

    assert llm.calls == 2
    assert "".join(announced) == "piece0 piece1 "
    assert _no_real_waiting == [(1, "the formatting turn")]


def test_a_stream_failure_after_a_frame_is_not_retried(_no_real_waiting):
    """Re-running would print the prose twice on the reader's screen."""
    llm = _StreamsThenRefuses(chunks_before_failure=2, fail_on_call=1)
    announced: list[str] = []

    with pytest.raises(_ResourceExhausted):
        asyncio.run(deterministic._formatting_turn(
            llm, [], config=None, announce=announced.append))

    assert llm.calls == 1
    assert "".join(announced) == "piece0 piece1 "
    assert _no_real_waiting == [], "a retry was scheduled after prose was announced"


# --------------------------------------------------------------------------------------
# Option 2 — the enrichment steps degrade, they do not fail the turn
# --------------------------------------------------------------------------------------

_CERVICAL = deterministic.ClinicalTarget(
    condition="cervical spondylosis",
    procedure="anterior cervical discectomy and fusion",
    anchors=("cervical spondylosis",),
)


def test_a_throttled_extraction_degrades_the_comparison_rather_than_the_turn(
        _no_real_waiting):
    """"The extraction did not complete" is tellable apart from "no cohort matched"."""
    rows, rejected, outcome = asyncio.run(deterministic._extract_pass(
        _Refusing(failures=99),
        _CERVICAL,
        [{"pmid": "1", "title": "A cohort", "abstract": "30-day readmission was 4%."}],
        None,
    ))

    assert rows == []
    assert outcome == routing.OUTCOME_EXTRACTION_DEGRADED
    assert rejected and "capacity" in rejected[0]["reason"]


def test_a_throttled_target_falls_back_to_the_chart_headers(_no_real_waiting):
    """The briefing still gets a condition when the extraction model cannot be reached."""
    result = asyncio.run(deterministic.derive_target(_Refusing(failures=99), {
        "discharge_diagnosis": "1. Cervical spondylosis with herniated nucleus pulposus.",
    }))

    assert result.source == "chart_headers"
    assert result.target is not None
    assert result.target.condition.startswith("Cervical spondylosis")


# --------------------------------------------------------------------------------------
# The observability the blind spot was missing
# --------------------------------------------------------------------------------------


def test_the_extraction_records_what_it_proposed_and_what_survived(caplog):
    """The line that makes an empty table diagnosable.

    On `826289ea` twenty-one abstracts were retrieved, extraction ran for 8.6 s, and the
    table drew nothing. Nothing recorded whether the model had answered
    `{"comparators": []}` or proposed rows the verifier refused.
    """
    caplog.set_level(logging.INFO, logger=deterministic.__name__)
    rejected_rows: list[dict] = []

    verified = deterministic._reply_rows(
        {"comparators": [{"pmid": "12345678", "rate_sentence": "30-day readmission 4%."}]},
        [],
        _CERVICAL,
        rejected_rows,
    )

    assert verified == []
    text = caplog.text
    assert "1 proposed, 0 verified, 1 refused, over 0 abstract(s)" in text
    assert "refused 12345678" in text


def test_an_empty_reply_is_recorded_as_nothing_proposed(caplog):
    caplog.set_level(logging.INFO, logger=deterministic.__name__)

    deterministic._reply_rows({"comparators": []}, [], _CERVICAL, [])

    assert "0 proposed, 0 verified, 0 refused" in caplog.text
