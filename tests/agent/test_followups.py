"""The three questions offered beneath an answer.

This module exists mostly to pin one failure that already happened. The suggester read
the model's reply with `str(reply.content)`, and this model answers with content blocks,
so `_questions` was handed the repr of a list — `[{'type': 'text', 'text': '["…"]'}]`.
That string has a `[` at each end, so it looked like a JSON array and json.loads
rejected it every time. The result was not a crash and not a log line: it was three
missing rows on every turn, which is exactly the kind of absence nobody reports.

The rest of the file covers the promises the module makes in its docstring: any failure
is an empty list, and the wrapper around the JSON does not decide whether good questions
get used.
"""

from __future__ import annotations

import asyncio
import json

from services.agent import followups

ANSWER = "Hospital course. The patient was admitted and intubated."


class _Reply:
    def __init__(self, content):
        self.content = content


class _Model:
    """Stands in for the chat model: one reply, or one raised exception."""

    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.prompts = []

    async def ainvoke(self, messages):
        self.prompts.append(messages)
        if self.error is not None:
            raise self.error
        return _Reply(self.content)


def _bound(monkeypatch, model):
    monkeypatch.setattr(followups, "_build_llm", lambda _id: model)
    return model


def _suggest(*, scope="patient_chart"):
    """Answer one turn through the suggester. `_build_llm` is always stubbed by
    `_bound`, so the model id passed here is only ever a placeholder."""
    return asyncio.run(
        followups.suggest("Summarise this admission.", ANSWER, model="test-model", scope=scope)
    )


def test_content_blocks_are_read_as_text(monkeypatch):
    """The regression: blocks are unwrapped before the text is parsed."""
    questions = ["What was the oxygen requirement?", "Which antibiotics were started?", "When is follow-up?"]
    blocks = [{"type": "text", "text": json.dumps(questions)}]
    _bound(monkeypatch, _Model(content=blocks))

    assert _suggest() == questions


def test_a_plain_string_reply_still_works(monkeypatch):
    """The other shape the same call can return; unwrapping must not break it."""
    questions = ["What was the oxygen requirement?", "Which antibiotics were started?", "When is follow-up?"]
    _bound(monkeypatch, _Model(content=json.dumps(questions)))

    assert _suggest() == questions


def test_a_fenced_reply_still_yields_its_questions(monkeypatch):
    """A model that fences its JSON is forgiven; the questions are still good."""
    questions = ["What was the oxygen requirement?", "Which antibiotics were started?", "When is follow-up?"]
    content = "Here are the questions:\n```json\n" + json.dumps(questions) + "\n```"
    _bound(monkeypatch, _Model(content=content))

    assert _suggest() == questions


def test_more_than_three_are_cut_to_three(monkeypatch):
    """The count is the contract, not the model's arithmetic."""
    four = ["One?", "Two?", "Three?", "Four?"]
    _bound(monkeypatch, _Model(content=json.dumps(four)))

    assert len(_suggest()) == followups.FOLLOW_UP_COUNT


def test_an_over_long_question_is_dropped_not_truncated(monkeypatch):
    """A question past the cap would render as a paragraph pretending to be a control,
    and a truncated one is a question the reader did not ask."""
    long_question = "What did the discharge summary say about " + "the plan " * 30 + "?"
    assert len(long_question) > followups._MAX_QUESTION_CHARS
    _bound(monkeypatch, _Model(content=json.dumps([long_question, "A real question?"])))

    assert _suggest() == ["A real question?"]


def test_a_question_at_the_cap_is_kept(monkeypatch):
    """The boundary is inclusive, so a question that just fits is still a row."""
    question = "Q" * followups._MAX_QUESTION_CHARS
    _bound(monkeypatch, _Model(content=json.dumps([question])))

    assert _suggest() == [question]


def test_a_model_failure_is_not_a_failed_turn(monkeypatch):
    """Whatever the model does — including raising — the caller gets a list."""
    _bound(monkeypatch, _Model(error=RuntimeError("upstream is down")))

    assert _suggest() == []


def test_unparseable_output_yields_nothing(monkeypatch):
    _bound(monkeypatch, _Model(content="I would rather not."))

    assert _suggest() == []


def test_an_empty_answer_skips_the_model_entirely(monkeypatch):
    """A redirect has no answer to build on, so it must not spend a call."""
    model = _bound(monkeypatch, _Model(content="[]"))

    result = asyncio.run(followups.suggest("Anything?", "   ", model="test-model"))

    assert result == []
    assert model.prompts == []


def test_the_scope_decides_what_a_useful_next_question_is(monkeypatch):
    """A chart turn asks about the record; a literature turn about the evidence."""
    patient = _bound(monkeypatch, _Model(content="[]"))
    _suggest(scope="patient_chart")
    general = _bound(monkeypatch, _Model(content="[]"))
    _suggest(scope="general_evidence")

    patient_instruction = patient.prompts[0][0].content
    general_instruction = general.prompts[0][0].content

    assert followups._PATIENT_SCOPE in patient_instruction
    assert followups._GENERAL_SCOPE in general_instruction
    assert patient_instruction != general_instruction
