"""Gap 4 / plan item 3 — earlier turns are replayed as messages, not as text.

The contract now accepts the turns a caller wants its question considered with,
and the chain turns each one into the pair of messages a conversation would have
produced: the question as a human message, the answer as the assistant message it
was. Two properties are load-bearing.

The first is that the material arrives as messages. Flattening a conversation
into a transcript string, or appending it to the prompt, would put a clinician's
question and the agent's answer — and, in a fuller replay, retrieved clinical
text — in the same position as the instruction that frames the request. The
prompt's only claim about provenance is the tool-result wrapper, so text that
enters outside a message role has no provenance at all.

The second is order and fidelity: oldest turn first, verbatim, so the question
being asked now is last and the model reads the conversation in the order it
happened.
"""

import asyncio

import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from services.agent import graph
from services.agent.contracts import (
    MAX_REPLAYED_TURNS,
    AgentRequestError,
    parse_agent_request,
)

TURNS = [
    {"question": "Why was this patient flagged?", "answer": "She was flagged at 0.31."},
    {"question": "And the potassium?", "answer": "Potassium was 3.2, low."},
]


def test_each_turn_becomes_a_question_and_an_answer_message():
    messages = graph.replayed_turns(TURNS)

    assert [type(m) for m in messages] == [HumanMessage, AIMessage,
                                           HumanMessage, AIMessage]
    assert [m.content for m in messages] == [
        "Why was this patient flagged?", "She was flagged at 0.31.",
        "And the potassium?", "Potassium was 3.2, low.",
    ]


def test_no_turns_produces_no_messages():
    assert graph.replayed_turns(None) == []
    assert graph.replayed_turns([]) == []


def test_the_text_is_replayed_verbatim():
    """Re-encoding a stored answer would change what the model is told happened."""
    turn = {"question": "Why?  ^[1]", "answer": "See ^[1] and the 3.2 value."}

    messages = graph.replayed_turns([turn])

    assert messages[0].content == turn["question"]
    assert messages[1].content == turn["answer"]


def test_a_turn_carrying_an_unexpected_field_never_reaches_the_chain():
    """The validation is the reason `replayed_turns` can index the fields."""
    with pytest.raises(AgentRequestError):
        parse_agent_request(
            {"question": "why?", "turns": [{**TURNS[0], "sources": []}]},
            max_question_chars=2000,
        )


def test_the_cap_is_enforced_before_the_chain_builds_a_prompt():
    with pytest.raises(AgentRequestError) as caught:
        parse_agent_request(
            {"question": "why?", "turns": [TURNS[0]] * (MAX_REPLAYED_TURNS + 1)},
            max_question_chars=2000,
        )

    assert caught.value.code == "too_many_turns"


def test_the_prompt_stays_a_template_and_the_history_is_data(monkeypatch):
    """The chain's seeded messages: prompt, then the conversation, then the ask.

    The system prompt is one message and the replayed conversation is others, so
    the template cannot absorb the transcript even by accident.
    """
    captured: dict = {}

    class _Graph:
        async def ainvoke(self, state, config=None):  # noqa: D401 - test double
            captured["state"] = state
            return state

    monkeypatch.setattr(graph, "build_graph", lambda *a, **k: _Graph())

    parsed = parse_agent_request(
        {"question": "and now?", "hadm_id": 90000009, "turns": TURNS},
        max_question_chars=2000,
    )
    state = asyncio.run(
        graph.ask(toolbox=None, question=parsed["question"], turns=parsed["turns"])
    )

    messages = state["messages"]
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == graph.SYSTEM_PROMPT
    assert [m.content for m in messages[1:-1]] == [
        "Why was this patient flagged?", "She was flagged at 0.31.",
        "And the potassium?", "Potassium was 3.2, low.",
    ]
    assert isinstance(messages[-1], HumanMessage)
    # The composed question, admission suffix and all: the replay must not
    # change what the model is asked about now.
    assert messages[-1].content == parsed["question"]
    assert captured["state"]["messages"][-1].content == parsed["question"]
