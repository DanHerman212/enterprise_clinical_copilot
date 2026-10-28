"""Citation numbering across turns: the offset `tool_node` hands the renderer.

`replay_messages` numbers every citable unit a replayed turn showed — note passages AND
laboratory analytes alike — because the conversation's citable units are one array and a
marker addresses a position in it.

`tool_node` used to compute its own offset from the replayed turns' `passages` only:

    _passage_total({"passages": [p for call in replayed_tool_calls
                                 for p in call["response"]["passages"]]})

A `get_lab_trends` payload has no `passages` key — it carries `trends` — so a replayed
`/labs` turn contributed zero, and the next tool result was stamped from `^[1]` again,
beside a panel already holding `^[1]`..`^[9]`. On Gate 3 (`00e`) `holdout_t4_02` cited the
chemotherapy regimen to `^[1]`, which the reader resolves to RDW.

The offset is `_citable_unit_total(state)` now: one sum, used by the replay and the tool
node both, over `passages` and `trends` together.
"""

from __future__ import annotations

import asyncio

from services.agent.engine import graph as gr

#: Nine analytes — the panel a `/labs` turn shows, and the units that turn consumes.
PANEL = {
    "hadm_id": 90000101,
    "returned": 9,
    "trends": [
        {"key": key, "label": label, "reference_low": 0, "reference_high": 1,
         "decimals": 1, "series": [{"day": 0.0, "value": 0.5, "flag": "normal"}]}
        for key, label in (
            ("rdw", "RDW"), ("rbc", "Red blood cells"), ("hemoglobin", "Hemoglobin"),
            ("sodium", "Sodium"), ("monocytes", "Monocytes"),
            ("creatinine", "Creatinine"), ("egfr", "eGFR"),
            ("potassium", "Potassium"), ("wbc", "White blood cells"),
        )
    ],
}

#: One note passage, the unit a follow-up's retrieval adds next.
RAG = {
    "query": "chemotherapy regimen",
    "passages": [
        {"id": "MT-2616-DS_brief_hospital_course_1",
         "section": "brief_hospital_course",
         "text": "Cycle seven of chemotherapy with VP-16, ifosfamide and mesna."},
    ],
}


class _Box:
    """A toolbox whose `call` returns the stored retrieval payload."""

    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    async def call(self, name, arguments):  # noqa: ANN001 - mirrors MCPToolbox.call
        self.calls.append((name, arguments))
        return self._payload


def _state(*, replayed=(), this_turn=()):
    return {"replayed_tool_calls": list(replayed), "tool_calls": list(this_turn)}


def test_a_replayed_lab_panel_consumes_nine_citation_numbers():
    """The panel is nine citable units, not zero — `citable_passages` gives one per analyte."""
    state = _state(replayed=[{"name": "get_lab_trends", "args": {}, "response": PANEL}])

    assert gr._citable_unit_total(state) == 9


def test_the_offset_counts_replayed_labs_and_this_turns_calls_together():
    state = _state(
        replayed=[{"name": "get_lab_trends", "args": {}, "response": PANEL}],
        this_turn=[{"name": "rag_search", "args": {}, "response": RAG}],
    )

    assert gr._citable_unit_total(state) == 10


def test_the_offset_after_a_replayed_lab_panel_is_ten_not_one():
    """The regression: a follow-up's first passage is `^[10]`, not a second `^[1]`.

    Nine analytes were already numbered `^[1]`..`^[9]` in the replayed transcript. A
    retrieval whose own unit restarts at `^[1]` gives the model two different things
    called `^[1]` and no way to tell which one it is writing about.
    """
    state = _state(replayed=[{"name": "get_lab_trends", "args": {}, "response": PANEL}])
    box = _Box(RAG)

    messages, _ = asyncio.run(gr._execute_tool_calls(
        box,
        [{"name": "rag_search", "id": "c1",
          "args": {"hadm_id": 90000101, "query": "chemotherapy regimen"}}],
        cite_offset=gr._citable_unit_total(state),
    ))

    body = messages[0].content
    assert '"cite": 10' in body, body
    assert '"cite": 9' not in body.replace('"cite": 10', ""), body


def test_without_a_replayed_panel_the_first_passage_is_still_one():
    """The offset is additive, not a floor: a thread with no prior units starts at 1."""
    box = _Box(RAG)

    messages, _ = asyncio.run(gr._execute_tool_calls(
        box,
        [{"name": "rag_search", "id": "c1",
          "args": {"hadm_id": 90000101, "query": "chemotherapy regimen"}}],
        cite_offset=gr._citable_unit_total(_state()),
    ))

    assert '"cite": 1' in messages[0].content


def test_the_next_turn_continues_from_where_this_one_stopped():
    """Across turns the running number keeps advancing: 10, then 11.

    Each turn's offset is the whole conversation's unit count, so the second retrieval
    starts one past the passage the first one added.
    """
    box = _Box(RAG)
    call = {"name": "rag_search", "id": "c1",
            "args": {"hadm_id": 90000101, "query": "chemotherapy regimen"}}

    messages, _ = asyncio.run(gr._execute_tool_calls(
        box, [call],
        cite_offset=gr._citable_unit_total(
            _state(replayed=[{"name": "get_lab_trends", "args": {}, "response": PANEL}])
        ),
    ))
    assert '"cite": 10' in messages[0].content

    next_turn = _state(
        replayed=[{"name": "get_lab_trends", "args": {}, "response": PANEL},
                  {"name": "rag_search", "args": {}, "response": RAG}],
    )
    assert gr._citable_unit_total(next_turn) == 10

    messages, _ = asyncio.run(gr._execute_tool_calls(
        box, [dict(call, id="c2")], cite_offset=gr._citable_unit_total(next_turn),
    ))
    assert '"cite": 11' in messages[0].content
