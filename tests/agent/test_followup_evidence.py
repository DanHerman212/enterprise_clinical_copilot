"""What a follow-up turn is allowed to see, and what it may draw.

Four defects were found in the production UI on 2026-10-25, all of them on the same
path: a followed-up question in a thread whose first turn was `/labs`.

1. The replay **re-ran** `get_lab_trends` on every subsequent turn. The tool was
   classified as re-derivable, a re-derivable result is not stored, and `replay_messages`
   resolves what it was not given — so a thread that had ever answered `/labs` paid for
   a live MCP round trip before the model was asked anything, on every turn, invisibly.
2. The **lab card** was composed from the whole conversation's evidence, so a question
   about the discharge diagnosis redrew the sparklines of a `/labs` turn from three
   turns earlier. The comparative matrix beside it had a command gate; the card did
   not.
3. The graph **demanded a retrieval** of a model that had just answered from the
   transcript, which `prompts.py` explicitly tells it to do. The forced round then had
   to call something, and `get_lab_trends` was among the things it could call.
4. Prose announced before the run went back for evidence was drawn and then replaced,
   and `http._prose_settled` compared it against the served answer — reporting a
   correction to text the reader was no longer holding.

The rule all four come back to: a conversation's evidence is not a turn's evidence.
Citations and the scope pill answer "what does this answer rest on", and the transcript
is part of that. A chart, a tool call and a swap warning answer "what did this turn do".
"""

import asyncio

from langchain_core.messages import ToolMessage

from services.agent import graph
from services.agent import http, stages
from services.agent.a2ui import compose_presentation
from services.agent.contracts import RE_DERIVABLE_TOOLS, RESULT_KEPT_TOOLS

LABS = {
    "trends": [
        {"label": "Creatinine", "points": [{"day": 1, "value": 1.1},
                                           {"day": 3, "value": 1.4}]},
    ],
    "synthetic": True,
    "note": "Values are synthetic.",
    "hadm_id": 90000015,
}

LAB_CALL = {"name": "get_lab_trends", "args": {"hadm_id": 90000015},
            "response": LABS}

NOTE_CALL = {"name": "rag_search_sections", "args": {"hadm_id": 90000015},
             "response": {"sections": []}}


def _by_id(canvas):
    components = canvas["messages"][1]["updateComponents"]["components"]
    return {c["id"]: c for c in components}


def _types(out):
    return {c["component"] for c in _by_id(out["a2ui"]).values()}


class _MustNotBeCalled:
    """A toolbox that fails the test if the replay re-resolves anything."""

    async def call(self, *_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("the stored lab series was not used")


# --- RC-1: the series is kept, so the replay does not call the tool -------------

class TestTheLabSeriesIsKept:
    def test_the_series_is_classified_as_a_result_the_caller_keeps(self):
        """`get_lab_trends` is reproducible, and that was read as "free".

        It is reproducible — and the replay re-runs every derivable call to rebuild the
        transcript, which is a live MCP round trip on every turn of the thread. The
        payload is values, intervals and flags generated from the admission's own
        feature row: there is no note prose in it for the store to be careful with.
        """
        assert "get_lab_trends" in RESULT_KEPT_TOOLS
        assert "get_lab_trends" not in RE_DERIVABLE_TOOLS
        # The partition is still a partition.
        assert not (RE_DERIVABLE_TOOLS & RESULT_KEPT_TOOLS)

    def test_a_replayed_lab_turn_reads_its_stored_series(self):
        turns = [{
            "question": "/labs",
            "answer": "Creatinine rose across the admission.",
            "tool_calls": [{
                "name": "get_lab_trends",
                "args": {"hadm_id": 90000015},
                "payload": LABS,
            }],
        }]

        messages, calls = asyncio.run(graph.replay_messages(_MustNotBeCalled(), turns))

        assert [call["name"] for call in calls] == ["get_lab_trends"]
        assert calls[0]["response"]["trends"][0]["label"] == "Creatinine"
        rendered = "\n".join(m.content for m in messages if isinstance(m, ToolMessage))
        assert "Creatinine" in rendered


# --- RC-2: the card is drawn by the turn that read the series -------------------

class TestTheCardIsThisTurnsDrawing:
    def test_a_replayed_series_does_not_redraw_the_card(self):
        """The follow-up that started this: a question about the discharge diagnosis,
        after a `/labs` turn, drew the sparkline panel again."""
        out = compose_presentation(
            "What was the discharge diagnosis?",
            "The discharge diagnosis was heart failure.^[1]",
            [NOTE_CALL],
            evidence=[LAB_CALL, NOTE_CALL],
        )

        assert "LabTrendCard" not in _types(out), (
            "a chart was drawn for evidence this turn only quoted")

    def test_a_turn_that_read_the_series_draws_the_card(self):
        out = compose_presentation(
            "How did the creatinine move?",
            "Creatinine rose.^[1]",
            [LAB_CALL],
            command="labs",
        )

        drawn = _by_id(out["a2ui"]).get("labs")
        assert drawn is not None, "the drill-down turn lost its card"
        assert drawn["component"] == "LabTrendCard"
        assert drawn["trends"][0]["label"] == "Creatinine"

    def test_the_scope_still_reads_the_whole_conversation(self):
        """The ruling kept `evidence` for citations and for the scope, and this is that
        half: a follow-up quoting a replayed chart read is still a chart answer. Only
        the card narrowed."""
        out = compose_presentation(
            "What was the discharge diagnosis?",
            "Heart failure.",
            [NOTE_CALL],
            evidence=[LAB_CALL, NOTE_CALL],
        )

        assert out["scope"] == "patient_chart"


# --- RC-3: the forced retrieval, and what it may reach for ----------------------

class TestTheRetrievalDemandIsNarrowed:
    def test_a_replayed_series_with_trends_is_evidence_a_question_can_read(self):
        state = {"replayed_tool_calls": [LAB_CALL]}
        assert graph._transcript_carries_lab_series(state) is True

    def test_a_failed_or_empty_series_is_not_evidence(self):
        failed = {"name": "get_lab_trends", "args": {"hadm_id": 90000015},
                  "response": {"error": "no labs"}}
        empty = {"name": "get_lab_trends", "args": {"hadm_id": 90000015},
                 "response": {"trends": []}}

        assert graph._transcript_carries_lab_series(
            {"replayed_tool_calls": [failed]}) is False
        assert graph._transcript_carries_lab_series(
            {"replayed_tool_calls": [empty]}) is False
        assert graph._transcript_carries_lab_series(
            {"replayed_tool_calls": []}) is False

    def test_only_a_chart_read_is_not_a_lab_series(self):
        scoring = {"name": "predict_readmission", "args": {"hadm_id": 90000015},
                   "response": {"probability": 0.31}}

        assert graph._transcript_carries_lab_series(
            {"replayed_tool_calls": [scoring]}) is False

    def test_the_forced_round_withholds_the_lab_tool_from_a_question_that_is_not_about_one(self):
        """A question that names no analyte is not answered by a series, so the forced
        round must pull from the chart notes."""
        assert graph._demand_withholds_labs({"replayed_tool_calls": []},
                                            lab_question=False) is True

    def test_the_forced_round_withholds_the_lab_tool_when_the_series_is_already_there(self):
        state = {"replayed_tool_calls": [LAB_CALL]}
        assert graph._demand_withholds_labs(state, lab_question=True) is True

    def test_a_lab_question_with_no_series_may_read_one(self):
        """The one case where the forced round needs the tool: the question is about an
        analyte and the conversation has not answered it yet."""
        assert graph._demand_withholds_labs({"replayed_tool_calls": []},
                                            lab_question=True) is False


# --- RC-4: the announcement belongs to the pass that is served ------------------

class TestTheAnnouncementFollowsTheLoop:
    def test_a_tool_or_a_new_pass_drops_what_was_announced(self):
        """Every frame that means "the run went back for evidence" resets the
        accumulator: what was announced belongs to a pass whose answer is not served."""
        for stage in (stages.STAGE_TOOL, stages.STAGE_RESULT,
                      stages.STAGE_REVIEWING, stages.STAGE_PLANNING):
            assert http._keeps_the_announcement(stage) is False, stage

    def test_prose_and_the_verify_frame_keep_it(self):
        """`verify` is emitted after the chain has finished and before composition.
        Clearing on it would discard every announcement and the verdict would never
        fire at all."""
        assert http._keeps_the_announcement(stages.STAGE_ANSWER_DELTA) is True
        assert http._keeps_the_announcement(stages.STAGE_VERIFY) is True

    def test_a_settled_answer_that_matches_what_was_announced_is_no_swap(self):
        payload = {"answer": "Creatinine rose from 1.1 to 1.4."}
        assert http._prose_settled(["Creatinine rose ", "from 1.1 to 1.4."],
                                   payload, "trace") is None

    def test_nothing_announced_is_nothing_to_reconcile(self):
        assert http._prose_settled([], {"answer": "Anything."}, "trace") is None
