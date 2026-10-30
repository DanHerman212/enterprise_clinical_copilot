"""A reply the provider withheld is not the same statement as an answer that is absent.

`/pubmed Hypothermia` on 2026-10-30 retrieved three records and then answered with
`SILENT_TURN_SENTENCE` — "I don't have an answer for that. I can summarise this
admission, …" — above three PMIDs it had just fetched. The trace shows why:

    finish_reason "SAFETY"  filtered ["HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT"]

The formatting turn's text was withheld, the empty-text branch served the sentence
written for a model that said nothing, and nothing logged the difference. The reader was
told the service had no answer when the truth was that its answer had been taken away.

Two causes, two sentences. A model that finished cleanly and said nothing has no answer;
a model whose reply was filtered never got to give one, and the records it retrieved are
on the page underneath. These tests hold the branch, and hold the log line that makes the
withheld case visible.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from services.agent.engine import model as model_gateway
from services.agent.engine import graph
from services.agent.engine.deterministic import run_command
from services.agent.postprocess import answer
from services.agent.routing import encounter, routing

PUBMED = routing.command_for("literature")

ARTICLES = [
    {"pmid": "35010760", "title": "Accidental hypothermia",
     "journal": "N Engl J Med", "pub_date": "2022 Jan", "abstract": "Rewarming.",
     "url": "https://pubmed.ncbi.nlm.nih.gov/35010760/"},
    {"pmid": "37797337", "title": "Rewarming strategies",
     "journal": "Resuscitation", "pub_date": "2023 Sep", "abstract": "Methods.",
     "url": "https://pubmed.ncbi.nlm.nih.gov/37797337/"},
    {"pmid": "33485655", "title": "Hypothermia in the field",
     "journal": "Wilderness Environ Med", "pub_date": "2021 Mar",
     "abstract": "Field care.", "url": "https://pubmed.ncbi.nlm.nih.gov/33485655/"},
]


class _SearchToolbox:
    """`search_literature` returns three records; nothing returns passages."""

    def __init__(self, articles=None) -> None:
        self.articles = ARTICLES if articles is None else articles
        self.calls: list[str] = []

    async def call(self, name, args):
        self.calls.append(name)
        if name == "search_literature":
            return {"query": args.get("query", ""), "returned": len(self.articles),
                    "articles": list(self.articles)}
        return {"returned": 0, "passages": []}


class _WithholdingModel:
    """A formatting turn whose reply the provider filtered away.

    Shaped like the live response: empty content, `finish_reason=SAFETY`, and the
    per-category rating that says which filter fired.
    """

    def __init__(self, *, finish_reason="SAFETY", blocked=True) -> None:
        self.reason = finish_reason
        self.blocked = blocked
        self.calls = 0

    def bind_tools(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the formatting turn was given tools")

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        return AIMessage(
            content="",
            response_metadata={
                "finish_reason": self.reason,
                "safety_ratings": [
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                     "blocked": self.blocked, "probability": "MEDIUM"},
                ],
            },
        )


class _SilentModel:
    """A formatting turn that finished cleanly and said nothing at all."""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the formatting turn was given tools")

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        return AIMessage(content="", response_metadata={"finish_reason": "STOP"})


def _run(model, toolbox=None):
    encounter.set_current(encounter.INPATIENT)
    box = toolbox if toolbox is not None else _SearchToolbox()
    with patch.object(model_gateway, "build_llm", lambda name: model):
        state = asyncio.run(run_command(
            box, PUBMED, 90000059, question="/pubmed Hypothermia", model="test-model",
            topic="Hypothermia",
        ))
    return state, box, model


def test_a_withheld_reply_over_retrieved_records_is_not_served_as_no_answer():
    state, box, model = _run(_WithholdingModel())

    assert model.calls == 1, 'the formatting turn never ran'
    assert "search_literature" in box.calls
    assert state["deterministic_answer"] == answer.SAFETY_WITHHELD_SUMMARY
    assert state["deterministic_answer"] != answer.SILENT_TURN_SENTENCE


def test_the_withheld_sentence_points_at_what_survived():
    """It says what happened and where the records are, and invents nothing else.

    No promise about a retry: the same input is filtered the same way, so "try again"
    would be advice the product cannot stand behind.
    """
    state, _, _ = _run(_WithholdingModel())
    sentence = state["deterministic_answer"].lower()

    assert "safety filter withheld" in sentence
    assert "are listed with this answer" in sentence
    assert "retry" not in sentence and "try again" not in sentence


def test_the_withheld_turn_says_so_in_the_log(caplog):
    """The trace recorded `outcome: ok` and nothing else, which is how this was found
    by a reader rather than by an operator."""
    with caplog.at_level(logging.WARNING, logger="services.agent.engine.deterministic"):
        _run(_WithholdingModel())

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "withheld" in messages
    assert "SAFETY" in messages
    assert "HARM_CATEGORY_DANGEROUS_CONTENT" in messages
    assert "3" in messages, 'the log line has to say how many records survived'


def test_a_turn_that_said_nothing_keeps_the_original_sentence():
    """The genuinely empty reply is unchanged. This branch is why the fix is a branch
    rather than a rewrite: a silent model has no records to point at in the same way,
    and the sentence written for it was correct."""
    state, _, model = _run(_SilentModel())

    assert model.calls == 1
    assert state["deterministic_answer"] == answer.SILENT_TURN_SENTENCE


def test_a_withheld_reply_with_nothing_retrieved_is_not_told_it_has_records():
    """The sentence promises records on the page, so it may only be served when there
    are some. A filtered reply over an empty search keeps the sentence that claims
    nothing."""
    state, _, _ = _run(_WithholdingModel(), toolbox=_SearchToolbox(articles=[]))

    assert state["deterministic_answer"] == answer.SILENT_TURN_SENTENCE


@pytest.mark.parametrize("reason", ["RECITATION", "PROHIBITED_CONTENT"])
def test_the_other_refusals_are_not_treated_as_safety(reason):
    """Only SAFETY names the mechanism the sentence describes. The other refusals keep
    the neutral sentence rather than borrowing a claim about a filter."""
    state, _, _ = _run(_WithholdingModel(finish_reason=reason))

    assert state["deterministic_answer"] == answer.SILENT_TURN_SENTENCE


def test_the_withheld_answer_still_reaches_the_trace():
    """The trace row reads its output off the state, and a dash is what TF-6 removed."""
    state, _, _ = _run(_WithholdingModel())

    assert graph.command_trace_output(state).strip() == answer.SAFETY_WITHHELD_SUMMARY


def test_the_records_are_still_published_with_the_withheld_summary():
    """The contradiction the reader saw — a refusal above three records — is only worth
    fixing if the records stay: they are what the sentence points at."""
    from services.agent.presentation.a2ui import compose_presentation

    state, _, _ = _run(_WithholdingModel())
    out = compose_presentation("/pubmed Hypothermia", state["deterministic_answer"],
                               state["tool_calls"], None, "literature")

    published = [row for row in out["references"] if row.get("pmid")]
    assert [row["pmid"] for row in published] == [a["pmid"] for a in ARTICLES]
