"""Stage 6C: the five full-Dev failures, each pinned where it broke.

Five items failed the 27-item Dev gate on `89f2cf3`, and each has one cause:

* `dev_t2_02` — the model wrote `^[2], [5]`. Only the first bracket was a marker, so the
  others stayed raw prose and the reader saw a number that opened nothing.
* `dev_t2_03` — turn 3 read a replayed `/ehr` answer whose two published cohort rates
  carried `^[7]` and `^[8]`, the numbers they had on THAT turn's page. On this turn 7 and
  8 were two note sections, so the model addressed two published rates to two note
  passages, and cited them.
* `dev_t4_02` — the guardrail stripped `-0.0768` and left `(contribution -)`: a
  contribution the turn had been served, deleted because the sign in the prose was not
  the sign in the payload.
* `dev_t5_01` — `find_neighbors` raised `ServiceUnavailable: 503 Stream removed` once and
  `rag_search` answered `search_failed`; the harness then scored the empty answer as a
  structural failure of the agent.
* `dev_t1_05` — an all-normal panel answered without naming Hemoglobin, because nothing
  in it was abnormal and the rule only asked about what was.

Each test names the defect, so a regression points at its own cause.
"""

from __future__ import annotations

import asyncio
import re

import pytest
from google.api_core import exceptions as google_exceptions
from langchain_core.messages import AIMessage

import httpx

from services.agent.engine import graph, prompts
from services.agent.postprocess import answer as answer_text
from services.agent.postprocess import citations
from services.agent.postprocess import guardrail as guard
from services.agent.presentation import a2ui
from services.mcp.tools import literature
from services.mcp.tools import retrieval


# --- Workstream 1: a caret-less bracket inside a cluster is a marker too ------------

@pytest.mark.parametrize(("written", "expected"), [
    ("The rate was 4.4% ^[2], [5].", "The rate was 4.4% ^[2], ^[5]."),
    ("The rates were 4.4% and 1.1% ^[1], [2], [7].", "The rates were 4.4% and 1.1% ^[1], ^[2], ^[7]."),
    ("Both cohorts ^[2] [5] agree.", "Both cohorts ^[2] ^[5] agree."),
    ("A semicolon separates them ^[3]; [4].", "A semicolon separates them ^[3]; ^[4]."),
])
def test_a_bare_bracket_after_a_marker_becomes_a_marker(written, expected):
    assert citations.normalise_carets(written) == expected


def test_a_bracket_that_is_not_a_citation_is_untouched():
    """The reason the repair is anchored on a marker: `[95% CI …]` is a statistic."""
    written = ("The pooled rate was 2.3% [95% CI 0.4-1.9] ^[1], and the cohort was "
               "uniform [n=812].")

    assert citations.normalise_carets(written) == written


def test_normalising_twice_changes_nothing():
    once = citations.normalise_carets("Rates ^[1], [2], [7].")
    assert citations.normalise_carets(once) == once


#: Five note sections, so a cluster of markers has five positions to address.
SECTIONS = {
    "name": "rag_search_sections",
    "args": {"hadm_id": 90000010},
    "response": {"returned": 5, "passages": [
        {"cite": number, "section": section, "text": f"The {section} section."}
        for number, section in enumerate(
            ["brief_hospital_course", "discharge_instructions", "admission_diagnosis",
             "past_medical_history", "physical_exam"], start=1)
    ]},
}


def test_settlement_serves_both_markers_of_a_cluster():
    """`dev_t2_02` turn 3, verbatim shape: `^[2], [5]` reached the reader with `[5]` as
    prose, so one of the two claims opened nothing."""
    prose = "The published rates were 4.4% and 1.1% ^[1], [4]."

    settled = answer_text.settle_answer(
        prose=prose, deterministic=None, tool_calls=[SECTIONS],
        replayed_tool_calls=[])

    assert citations.cited_numbers(settled.prose) == [1, 2]
    assert re.search(r"(?<!\^)\[\d", settled.prose) is None, (
        f"a bracket survived settlement as prose: {settled.prose!r}")


def test_both_markers_of_a_cluster_resolve_to_their_own_passage():
    """The consequence the reader sees: two pills, two different sections."""
    prose = "The rate was 4.4% and the interval ^[2], [4]."

    settled = answer_text.settle_answer(
        prose=prose, deterministic=None, tool_calls=[SECTIONS],
        replayed_tool_calls=[])
    resolved = a2ui.resolve_sources(
        settled.prose, settled.display, None, (), evidence=[SECTIONS])

    assert [row["cite"] for row in resolved] == [1, 2]
    assert resolved[0]["section"] != resolved[1]["section"], (
        "two markers on one sentence opened the same passage, so one of them was lost")


# --- Workstream 2: a published record is addressed by identifier, not by number -----

PMID_A = "35464578"
PMID_B = "29026380"


def test_has_literature_attaches_the_literature_rules():
    """A follow-up holding retrieved cohorts needs the rules for citing them, whatever
    its own wording asked for."""
    composed = prompts.compose_system_prompt(
        allowed_tools={"rag_search"}, has_literature=True)

    assert prompts.LITERATURE_SPOKE.strip() in composed


def test_has_literature_is_off_by_default():
    composed = prompts.compose_system_prompt(allowed_tools={"rag_search"})
    assert prompts.LITERATURE_SPOKE.strip() not in composed


def _replayed_ehr_turn(*, cited_markers: dict | None) -> dict:
    """Turn 1's `/ehr` answer as the site replays it: note passages 1-5, then two
    published cohorts numbered 9 and 10 on that turn's page."""
    response = {"query": "cervical fusion readmission", "returned": 2, "articles": [
        {"pmid": PMID_A, "title": "Readmission after ACDF.",
         "abstract": "The 30-day readmission rate was 4.4%.", "url": "u1"},
        {"pmid": PMID_B, "title": "Readmission after revision fusion.",
         "abstract": "The 30-day readmission rate was 1.1%.", "url": "u2"},
    ], "cited_pmids": [PMID_A, PMID_B]}
    if cited_markers is not None:
        response["cited_markers"] = cited_markers
    return {
        "question": "/ehr For admission 90000010.",
        "answer": (
            "**Readmission risk & trajectory.** Her estimate is 23.4%, above the 4.4% "
            f"published for this procedure ^[9] and the 1.1% published for the "
            f"revision cohort ^[10]. Her course is in the note ^[1], ^[2]."
        ),
        "tool_calls": [
            # The site stores a tool result as `payload` — `turn.py` calls it `response`
            # on the wire and the caller translates it — and a replay reads it there.
            {"name": SECTIONS["name"], "args": SECTIONS["args"],
             "payload": SECTIONS["response"]},
            {"name": "search_literature",
             "args": {"query": "cervical fusion readmission", "scope": "readmission"},
             "payload": response},
        ],
        # Presentation republishes a published record's number as its array position, so
        # the map is the identity over 9 and 10 and the note markers.
        "citation_map": {"1": 1, "2": 2, "9": 9, "10": 10},
    }


def _replay(turn: dict):
    class _NoFetch:
        async def call(self, name, arguments):  # pragma: no cover - asserted absence
            raise AssertionError(f"re-executed {name!r} although its payload was stored")

    return asyncio.run(graph.replay_messages(_NoFetch(), [turn], hadm_id=90000010))


def test_a_replayed_cohort_rate_keeps_its_identifier():
    """`dev_t2_03` turn 3: the replayed page's `^[9]` and `^[10]` were positions in a
    page that no longer existed, and position 9 on the next turn was a note section."""
    messages, _calls, _articles = _replay(_replayed_ehr_turn(
        cited_markers={"9": PMID_A, "10": PMID_B}))

    answer = [m for m in messages if isinstance(m, AIMessage)][-1].content

    assert f"^[PMID: {PMID_A}]" in answer
    assert f"^[PMID: {PMID_B}]" in answer
    assert citations.cited_pmids(answer) == [PMID_A, PMID_B]


def test_the_replayed_note_markers_are_still_positions():
    """Only the published records move to identifiers; a note passage is a position and
    stays one, restored into the conversation's own numbering."""
    messages, _calls, _articles = _replay(_replayed_ehr_turn(
        cited_markers={"9": PMID_A, "10": PMID_B}))

    answer = [m for m in messages if isinstance(m, AIMessage)][-1].content

    assert "the note ^[1], ^[2]" in answer


def test_a_turn_stored_before_the_pairs_existed_replays_as_it_did():
    """An older stored turn carries no `cited_markers`, and a replay must not invent
    one: the numbers are restored against the map, which is what it did before."""
    messages, _calls, _articles = _replay(_replayed_ehr_turn(cited_markers=None))

    answer = [m for m in messages if isinstance(m, AIMessage)][-1].content

    assert "PMID" not in answer
    assert "^[9]" in answer


def test_the_pairs_are_read_from_the_stored_payload_key():
    assert citations.literature_marker_rewrites([{
        "name": "search_literature",
        "payload": {"cited_markers": {"9": PMID_A}},
    }]) == {"9": PMID_A}


def test_a_cluster_splits_into_one_marker_per_claim():
    """A number that is not a record becomes a position marker, so a mixed cluster is
    served as two markers rather than one badge standing for two claims."""
    rewritten = citations.rewrite_literature_markers(
        "The rate is 4.4% ^[9], ^[2].", {"9": PMID_A})

    assert rewritten == f"The rate is 4.4% ^[PMID: {PMID_A}], ^[2]."


# --- Turn 3 of a thread: the comparators it names, and nothing else ------------------

def _comparator(pmid: str, rate: str) -> dict:
    return {"pmid": pmid, "title": f"Cohort study {pmid}",
            "abstract": f"The 30-day readmission rate was {rate}.",
            "journal": "Spine", "pub_date": "2022", "authors": "A B"}


def test_a_follow_up_publishes_the_comparators_it_names_and_no_others():
    """`dev_t2_03` turn 3, as the composer sees it.

    The turn asks a literature question and runs no search of its own: the earlier
    `/ehr` search is already in the transcript, and the replay pruned it to the cohorts
    that turn cited. The answer names two of them by identifier. What the reader is shown
    must be those two — not the candidate pool the `/ehr` search drew from, by then
    neither read nor named by anything.
    """
    answer = (f"The 4.4% published for this procedure ^[PMID: {PMID_A}] is below her "
              f"estimate, and the 1.1% published for the revision cohort "
              f"^[PMID: {PMID_B}] is further below it.")
    # The thread's records are the two the earlier turn rested on, plus a pool candidate
    # that turn's answer never named — the shape a pruned search would leave behind if the
    # prune ever failed.
    thread_literature = [
        _comparator(PMID_A, "4.4%"),
        _comparator(PMID_B, "1.1%"),
        _comparator("35120405", "9.2%"),
    ]

    out = a2ui.compose_presentation(
        "How does she compare with the published cohorts?", answer, [],
        None, None, literature_replay=True, thread_literature=thread_literature)

    published = [str(source.get("pmid")) for source in out["sources"]
                 if source.get("pmid")]
    drawer = [str(row["pmid"]) for row in out["references"] if row.get("pmid")]

    assert published == [PMID_A, PMID_B]
    assert drawer == [PMID_A, PMID_B]
    assert "35120405" not in published + drawer
    # The page numbers the two records are served under are the numbers the prose cites,
    # and each one opens the row that carries its identifier.
    assert citations.cited_numbers(out["answer"]) == [1, 2]
    assert [row["index"] for row in out["references"]] == [1, 2]
    assert [row["pmid"] for row in out["references"]] == [PMID_A, PMID_B]


# --- Workstream 3: a signed contribution is the number the payload served -----------

#: `dev_t4_02`'s shape: the model's own attribution weights, one of them negative.
PREDICT = [{
    "name": "predict_readmission",
    "args": {"hadm_id": 90000015},
    "response": {
        "probability": 0.234,
        "threshold": 0.11,
        "top_factors": [
            {"feature": "hemoglobin", "contribution": -0.0768},
            {"feature": "prior_admission", "contribution": 0.0824},
        ],
    },
}]


def test_a_negative_contribution_survives_with_its_sign():
    answer = "These factors pushed the estimate down (contribution -0.0768)."

    cleaned, flags = guard.verify_risk_numbers(answer, PREDICT)

    assert cleaned == answer
    assert flags == []


def test_the_magnitude_of_a_negative_contribution_survives():
    """`the factor decreased it by 0.0768` states the size, and the payload served it."""
    answer = "Prior admission decreased the estimate by 0.0768."

    cleaned, flags = guard.verify_risk_numbers(answer, PREDICT)

    assert cleaned == answer
    assert flags == []


def test_a_fabricated_negative_decimal_leaves_no_stranded_sign():
    """`(contribution -)` was the served artefact: the number went and the sign stayed."""
    answer = ("These factors pushed the estimate down (contribution -0.9999) and that "
              "is the whole story.")

    cleaned, flags = guard.verify_risk_numbers(answer, PREDICT)

    assert "-0.9999" not in cleaned
    assert "-)" not in cleaned, "the sign was left behind by the removed number"
    assert "(contribution)" in cleaned
    assert any(flag.startswith("risk_number_unsupported:") for flag in flags)


# --- Workstream 4: a dropped stream is retried on a fresh channel --------------------

class _Flaky:
    """A cached client handle whose channel fails a stated number of times."""

    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0
        self.cleared = 0
        # `_embed_with_retry` reaches the embedding through `.models`; the index handle
        # is used directly. One object serves both so the record is the same shape.
        self.models = self

    def cache_clear(self):
        self.cleared += 1

    def __call__(self):
        return self

    def embed_content(self, **_kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise google_exceptions.ServiceUnavailable(
                "503 Stream removed (recvmsg:Connection reset by peer (104))")
        return "embedded"

    def find_neighbors(self, **_kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise google_exceptions.ServiceUnavailable(
                "503 Stream removed (recvmsg:Connection reset by peer (104))")
        return "neighbours"


@pytest.fixture
def no_sleep(monkeypatch):
    pauses: list[float] = []
    monkeypatch.setattr(retrieval.time, "sleep", lambda seconds: pauses.append(seconds))
    return pauses


def test_a_dropped_stream_on_the_index_is_retried(monkeypatch, no_sleep):
    """`dev_t5_01`: `find_neighbors` failed once and the whole turn answered
    `search_failed`."""
    handle = _Flaky(failures=1)
    monkeypatch.setattr(retrieval, "_index_endpoint", handle)

    assert retrieval._find_neighbors_with_retry([0.1], 5, 90000010) == "neighbours"
    assert handle.calls == 2
    assert handle.cleared == 1, (
        "the cached handle was not cleared, so the retry reused the broken channel")
    assert no_sleep == [0.5]


def test_a_dropped_stream_on_the_embedding_is_retried(monkeypatch, no_sleep):
    handle = _Flaky(failures=1)
    monkeypatch.setattr(retrieval, "_embed_client", handle)

    assert retrieval._embed_with_retry("readmission") == "embedded"
    assert handle.calls == 2
    assert handle.cleared == 1


def test_a_third_failure_is_raised_rather_than_hidden(monkeypatch, no_sleep):
    """Bounded: the tool's own error path still reports a backend it could not reach."""
    handle = _Flaky(failures=3)
    monkeypatch.setattr(retrieval, "_index_endpoint", handle)

    with pytest.raises(google_exceptions.ServiceUnavailable):
        retrieval._find_neighbors_with_retry([0.1], 5, 90000010)
    assert handle.calls == 3


# --- and the same retry on the PubMed client ----------------------------------------

class _Status:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _PubMed:
    """A client that answers in the order it was given, or raises and then answers."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.attempts = 0

    async def get(self, _url, params=None):
        outcome = self.outcomes[min(self.attempts, len(self.outcomes) - 1)]
        self.attempts += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def no_async_sleep(monkeypatch):
    pauses: list[float] = []

    async def _sleep(seconds):
        pauses.append(seconds)

    monkeypatch.setattr(literature.asyncio, "sleep", _sleep)
    return pauses


def test_a_busy_pubmed_is_retried(no_async_sleep):
    client = _PubMed(_Status(503), _Status(200))

    assert asyncio.run(literature._get(client, "u", {})).status_code == 200
    assert client.attempts == 2
    assert no_async_sleep == [0.6]


def test_a_dropped_connection_is_retried(no_async_sleep):
    client = _PubMed(httpx.ReadError("connection reset"), _Status(200))

    assert asyncio.run(literature._get(client, "u", {})).status_code == 200
    assert client.attempts == 2


def test_a_refused_request_is_not_retried(no_async_sleep):
    """A 400 is the request, and it fails identically the second time."""
    client = _PubMed(_Status(400))

    assert asyncio.run(literature._get(client, "u", {})).status_code == 400
    assert client.attempts == 1
    assert no_async_sleep == []


def test_a_persistent_outage_still_returns_the_response_to_report(no_async_sleep):
    """The caller's error branch reports a 503 as an outage; the retry must not swallow
    the response it is there to read."""
    client = _PubMed(_Status(503))

    assert asyncio.run(literature._get(client, "u", {})).status_code == 503
    assert client.attempts == literature._RATE_LIMIT_RETRIES + 1


def test_an_argument_error_is_not_retried(monkeypatch, no_sleep):
    """A request the service refuses fails identically the second time."""
    class _Refuses(_Flaky):
        def find_neighbors(self, **_kwargs):
            self.calls += 1
            raise google_exceptions.InvalidArgument("bad namespace")

    handle = _Refuses(failures=0)
    monkeypatch.setattr(retrieval, "_index_endpoint", handle)

    with pytest.raises(google_exceptions.InvalidArgument):
        retrieval._find_neighbors_with_retry([0.1], 5, 90000010)
    assert handle.calls == 1


# --- Workstream 5: a whole-panel answer names the whole panel ------------------------

def _flatten(text: str) -> str:
    return " ".join(text.split())


def test_the_labs_rules_say_a_whole_panel_names_every_analyte():
    """`dev_t1_05`: an all-normal panel, and the answer named the movers only."""
    text = _flatten(prompts.LABS_FORMATTING)

    assert "THE WHOLE PANEL MEANS EVERY ANALYTE IN IT" in text
    assert "no analyte in it may be silently omitted" in text
    assert "stayed within its reference interval" in text
    assert "hemoglobin moving 14.0 to 13.4" in text


def test_the_labs_rules_still_answer_only_what_was_asked():
    """The named-analyte rule is the one this must not weaken: two names out of ten."""
    text = _flatten(prompts.LABS_FORMATTING)

    assert "those analytes are the answer" in text
    assert "say nothing about the rest of the panel" in text


def test_the_ehr_rules_refuse_an_arrival_list_as_a_discharge_regimen():
    """`dev_t2_03` turn 1 presented the "On transfer" list as what she went home on."""
    text = _flatten(prompts.EHR_FORMATTING)

    assert "WHAT THE PATIENT GOES HOME ON IS NOT WHAT THEY ARRIVED ON" in text
    assert "On transfer" in text
    assert "Medications on transfer" in text
    assert "discharge_medications" in text
    assert "brief_hospital_course" in text
