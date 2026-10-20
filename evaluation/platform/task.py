"""The task: one dataset item in, one `TaskOutput` out.

A census item is a two-turn script. Turn 1 runs, and Turn 2 is sent on the *same*
session id with Turn 1 replayed, because a follow-up without its conversation is a
different question. Under concurrency the two turns of one item therefore cannot
interleave: they are sequential by construction, one item per worker slot.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from evaluation.platform.schemas import DatasetItemInput, TaskOutput

#: `ask(**kw) -> dict` — the agent round-trip. Injected so the dry run can use a fake.
Ask = Callable[..., dict]


def card_to_text(payload: Any) -> str:
    """Serialise a slash-command card so the trace output is never a dash.

    `/risk` answers with a widget and no prose. Langfuse renders the task's output in
    the trace table, so returning the raw dict (or an empty string) is what made the
    dashboard unreadable. Human-readable JSON keeps it inspectable and non-blank.
    """
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload.strip()
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(payload)


def _answer_text(response: dict) -> str:
    """The readable answer for a turn: prose, else the card, else the refusal.

    A contract refusal (an unknown command answered with HTTP 400) carries no answer
    and no card. Returning "" there would fail `TaskOutput` validation and take the
    whole item down, and leaving it blank in the dashboard is what made refusals
    invisible — so the refusal itself becomes the readable text.
    """
    prose = (response.get("answer") or "").strip()
    if prose:
        return prose
    card = response.get("a2ui") or response.get("ui_components")
    if card:
        return card_to_text(card)
    status = response.get("status")
    if status and int(status) != 200:
        error = str(response.get("error") or "refused")
        message = str(response.get("message") or "").strip()
        return f"[HTTP {status}] {error}" + (f": {message}" if message else "")
    return ""


def _evidence(response: dict) -> list[dict[str, Any]]:
    """Tool payloads with their disclaimers intact.

    `synthetic` and `note` travel with `get_lab_trends` and were dropped by every
    previous collector, which made an answer quoting the tool's own disclaimer look
    fabricated.
    """
    out: list[dict[str, Any]] = []
    for call in (response.get("tool_calls") or []):
        if not isinstance(call, dict):
            continue
        payload = call.get("response")
        item = {"tool": call.get("name"), "result": payload}
        if isinstance(payload, dict):
            for key in ("synthetic", "note"):
                if key in payload:
                    item[key] = payload[key]
        out.append(item)
    return out


def replay_turn(question: str, response: dict) -> dict:
    """Turn 1 as the service will accept it back in `turns`.

    The service's contract (`contracts.REPLAYED_TURN_FIELDS`) accepts exactly
    `question`, `answer` and `tool_calls`, and a tool call is exactly
    `{name, args, payload}` — it refuses anything else with `unsupported_field`
    rather than ignoring it. The agent's own envelope returns `response`, not
    `payload`, so passing `tool_calls` through unchanged would make every Turn 2 fail
    with a 400. Limits are the contract's: 5 calls, 8000 characters of answer.
    """
    answer = (response.get("answer") or "")[:8000]
    calls: list[dict[str, Any]] = []
    for call in (response.get("tool_calls") or [])[:5]:
        if not isinstance(call, dict) or not call.get("name"):
            continue
        payload = call.get("payload", call.get("response"))
        entry: dict[str, Any] = {
            "name": str(call["name"])[:64],
            "args": call.get("args") if isinstance(call.get("args"), dict) else {},
        }
        if isinstance(payload, (dict, str)):
            entry["payload"] = payload
        calls.append(entry)
    return {"question": question, "answer": answer, "tool_calls": calls}


def make_task(ask: Ask, *, run_tag: str,
              new_session_id: Callable[[], str]) -> Callable[..., dict]:
    """Build the Langfuse task function."""

    def task(*, item, **_kwargs) -> dict:
        raw = item.get("input") if isinstance(item, dict) else getattr(item, "input")
        data = DatasetItemInput.model_validate(raw)
        session_id = new_session_id()
        hadm_id = int(data.patient_id)

        first = ask(question=data.turns[0], hadm_id=hadm_id, run_tag=run_tag,
                    session_id=session_id)

        output = TaskOutput(
            turn_1_response=_answer_text(first),
            tool_evidence=_evidence(first),
            session_id=session_id,
            turn_1_status=first.get("status"),
            turn_1_command=first.get("command") or first.get("command_executed"),
            turn_1_components=list(first.get("a2ui_components") or []),
            turn_1_card=first.get("a2ui") if isinstance(first.get("a2ui"), dict) else None,
            turn_1_error=first.get("error"),
        )

        if len(data.turns) > 1:
            replayed = replay_turn(data.turns[0], first)
            second = ask(question=data.turns[1], hadm_id=hadm_id, run_tag=run_tag,
                         session_id=session_id, turns=[replayed])
            output.turn_2_response = _answer_text(second) or None
            output.turn_2_status = second.get("status")
            output.tool_evidence.extend(_evidence(second))

        return output.model_dump()

    return task
