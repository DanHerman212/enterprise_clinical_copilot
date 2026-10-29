"""The task: one dataset item in, one `TaskOutput` out.

A census item is a two-turn script. Turn 1 runs, and Turn 2 is sent on the *same*
session id with Turn 1 replayed, because a follow-up without its conversation is a
different question. Under concurrency the two turns of one item therefore cannot
interleave: they are sequential by construction, one item per worker slot.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from shared import a2ui
from evaluation.platform.schemas import DatasetItemInput, TaskOutput
#: One definition of what consumes a citation number (`00d8` Row 1, charter Law 1).
#: The harness stamps the same markers the agent renders, and a local counter is how the
#: two drifted apart: the harness counted passages-only while the agent counted passages
#: and analytes, which is the `00d6` failure reproduced inside the measurement.
from services.agent.postprocess.citations import passage_total

#: `ask(**kw) -> dict` — the agent round-trip. Injected so the dry run can use a fake.
Ask = Callable[..., dict]


def card_to_text(payload: Any) -> str:
    """Serialise a slash-command card so the trace output is never a dash.

    A command turn can carry a surface as well as prose, and a turn that answers only
    through one — `/risk` was the case until it was retired on 2026-10-26 — has no
    sentence at all. Langfuse renders the task's output in the trace table, so returning
    the raw dict (or an empty string) is what made the dashboard unreadable.
    Human-readable JSON keeps it inspectable and non-blank.
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
    card = response.get("a2ui")
    if card:
        return card_to_text(card)
    status = response.get("status")
    if status and int(status) != 200:
        error = str(response.get("error") or "refused")
        message = str(response.get("message") or "").strip()
        return f"[HTTP {status}] {error}" + (f": {message}" if message else "")
    return ""


def _citation_units(payload: Any) -> int:
    """How many citation numbers one tool payload consumes.

    Delegates to `citations.passage_total`. The local rules — one number per note
    passage, one per analyte, nothing for an estimate, a medication list or a search
    result set — are that function's to state, and this module states none of them.
    """
    return passage_total(payload)


def _numbered(payload: Any, cite_start: int) -> Any:
    """The payload with a `cite` marker on each unit, on a copy.

    The recorded payload is the execution record and is not this function's to rewrite;
    the judge gets its own copy carrying the number the answer must write.
    """
    if not isinstance(payload, dict):
        return payload
    for key in ("passages", "trends"):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        counter = cite_start
        stamped = []
        for row in rows:
            if isinstance(row, dict):
                stamped.append({**row, "cite": f"^[{counter}]"})
                counter += 1
            else:
                stamped.append(row)
        return {**payload, key: stamped}
    return payload


def _evidence(response: dict, cite_start: int = 1) -> list[dict[str, Any]]:
    """Tool payloads with their disclaimers and their citation numbers intact.

    `synthetic` and `note` travel with `get_lab_trends` and were dropped by every
    previous collector, which made an answer quoting the tool's own disclaimer look
    fabricated.

    It also used to be one flat list with no numbering at all, so a marker could only be
    graded by counting array positions in the judge's own head — and the numbering the
    reader sees was rewritten downstream, so the count landed on the wrong unit (`00e`,
    2026-10-28). Each unit now carries the marker the answer must write for it, numbered
    from `cite_start`, which the caller advances by the units this payload consumed.
    """
    out: list[dict[str, Any]] = []
    counter = cite_start
    for call in (response.get("tool_calls") or []):
        if not isinstance(call, dict):
            continue
        payload = call.get("response")
        item = {"tool": call.get("name"),
                "result": _numbered(payload, counter)}
        counter += _citation_units(payload)
        if isinstance(payload, dict):
            for key in ("synthetic", "note"):
                if key in payload:
                    item[key] = payload[key]
        out.append(item)
    return out


def _follow_ups(response: dict) -> list[str]:
    """The next questions the service proposed, when the turn asked for them.

    Kept per turn rather than merged: a loop-back turn consumes the Nth entry of one
    named turn, so the round it came from has to stay identifiable.
    """
    offered = response.get("follow_ups")
    if not isinstance(offered, list):
        return []
    return [text for item in offered if (text := str(item).strip())]


def _components(response: dict) -> list[str]:
    """The component names a turn rendered, in surface order."""
    return (a2ui.components(response.get("a2ui"))
            or list(response.get("a2ui_components") or []))


def _verified(response: dict) -> bool | None:
    """The service's `verified` flag, or None when it did not send one.

    None is not False: a model-graded turn with no flag says nothing about grounding,
    and a check that reads absent as False would fail every envelope predating it.
    """
    flag = response.get("verified")
    return flag if isinstance(flag, bool) else None


def _tool_calls(response: dict) -> list[dict[str, Any]]:
    """This turn's tool calls, with the payload each returned, kept per turn.

    `tool_evidence` flattens every turn's calls into one list, which is what the judge
    needs and not what a post-mortem needs: asked "which passage did `^[2]` point at on
    turn 3", the flattened list cannot say which turn the entry came from. A free-text
    turn's calls are not exported as spans either, so this is the only local record.
    """
    out: list[dict[str, Any]] = []
    for call in response.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        out.append({"name": call.get("name"),
                    "args": call.get("args") or {},
                    "response": call.get("response")})
    return out


def _sources(response: dict) -> list[dict[str, Any]]:
    """The sources the reader can click for this turn, in the numbering they use.

    Kept as `{cite, section, text}`, plus `pmid` when the source carries one: the judge
    needs to know which passage a marker resolves to, and the passage's text is what it
    checks the sentence against — while a literature marker resolves to a record whose
    `pmid` is the identifier the reader opens. `evaluators.structural_check` reads that
    field for the `PMID` citation-marker assertion, so dropping it here left a correctly
    formatted `/pubmed` answer with nothing to resolve against (`dev_t1_06`).
    """
    out: list[dict[str, Any]] = []
    for entry in response.get("sources") or []:
        if not isinstance(entry, dict):
            continue
        kept = {"cite": entry.get("cite"),
                "section": entry.get("section"),
                "text": entry.get("text")}
        if entry.get("pmid") is not None:
            kept["pmid"] = entry.get("pmid")
        out.append(kept)
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


#: A turn whose wording is decided at runtime: the Nth entry of the follow-ups the
#: service proposed on an earlier turn. Trajectory 2 of the Tier 2 dev suite tests the
#: affordance the UI actually renders, so the prompt cannot be typed into the dataset.
FOLLOW_UP_PLACEHOLDER = re.compile(r"\{\{\s*turn_(\d+)\.follow_ups\[\s*(\d+)\s*\]\s*\}\}")


def placeholder_turns(turns: list[str]) -> dict[int, int]:
    """`{turn_index: turn_index}` — which later turn consumes which earlier turn's
    follow-up. Zero-based, as `turns` is."""
    needed: dict[int, int] = {}
    for index, prompt in enumerate(turns):
        for match in FOLLOW_UP_PLACEHOLDER.finditer(prompt):
            source = int(match.group(1)) - 1
            if source >= index:
                raise ValueError(
                    f"turn {index + 1} references turn {source + 1}'s follow-ups, which "
                    "has not run yet"
                )
            needed[source] = source
    return needed


def resolve_prompt(prompt: str, responses: list[dict]) -> str:
    """Replace every loop-back placeholder with the follow-up the service proposed."""

    def _sub(match: re.Match[str]) -> str:
        source = int(match.group(1)) - 1
        position = int(match.group(2))
        if source >= len(responses):
            raise ValueError(f"turn {source + 1} has not run")
        offered = responses[source].get("follow_ups") or []
        if not isinstance(offered, list) or position >= len(offered):
            raise ValueError(
                f"turn {source + 1} proposed no follow_ups[{position}]: "
                f"{len(offered) if isinstance(offered, list) else 0} proposed"
            )
        text = str(offered[position]).strip()
        if not text:
            raise ValueError(f"turn {source + 1} proposed a blank follow_ups[{position}]")
        return text

    return FOLLOW_UP_PLACEHOLDER.sub(_sub, prompt)


def make_task(ask: Ask, *, run_tag: str,
              new_session_id: Callable[[], str]) -> Callable[..., dict]:
    """Build the Langfuse task function."""

    def task(*, item, **_kwargs) -> dict:
        raw = item.get("input") if isinstance(item, dict) else getattr(item, "input")
        data = DatasetItemInput.model_validate(raw)
        session_id = new_session_id()
        hadm_id = int(data.patient_id)
        loop_backs = placeholder_turns(data.turns)

        # Every turn runs in order on one session, each sent with the whole transcript
        # replayed: a follow-up without its conversation is a different question. This
        # is a loop rather than two named calls because Trajectory 4 scripts three and
        # four turns, and turns 3+ used to be dropped without a word.
        #
        # `turn_traces` collects one id per turn rather than two named fields because a
        # turn that ran and left no id is the case the run has to report, and a list
        # can hold that absence where a name cannot.
        responses: list[dict] = []
        turn_traces: list[str] = []
        prompts: list[str] = []
        transcript: list[dict] = []

        for index, scripted in enumerate(data.turns):
            prompt = resolve_prompt(scripted, responses)
            prompts.append(prompt)
            call: dict[str, Any] = {
                "question": prompt,
                "hadm_id": hadm_id,
                "run_tag": run_tag,
                "session_id": session_id,
                "turns": list(transcript) or None,
            }
            # Sent only when the turn has to propose them: `ask` is injected, and a
            # transport that does not know the field still satisfies this task.
            if index in loop_backs:
                call["follow_ups"] = True
            response = ask(**call)
            responses.append(response)
            # The service's own trace id, carried so the experiment can hang this item's
            # scores off the trace that holds the run tag — and the prompts, tool calls
            # and model output that produced the answer (2026-10-21).
            turn_traces.append(str(response.get("langfuse_trace_id") or ""))
            transcript.append(replay_turn(prompt, response))

        first = responses[0]
        answers = [_answer_text(response) for response in responses]
        follow_ups = [_follow_ups(response) for response in responses]
        components = [_components(response) for response in responses]
        verified = [_verified(response) for response in responses]
        sources = [_sources(response) for response in responses]
        tool_calls_per_turn = [_tool_calls(response) for response in responses]
        # Numbered across the WHOLE thread: a follow-up's first passage continues where
        # the previous turn's last unit stopped, which is the arithmetic the agent uses
        # when it renders the same payloads for the model.
        tool_evidence: list[dict[str, Any]] = []
        cite_cursor = 1
        for response in responses:
            items = _evidence(response, cite_cursor)
            tool_evidence.extend(items)
            cite_cursor += sum(_citation_units(call.get("response"))
                               for call in (response.get("tool_calls") or [])
                               if isinstance(call, dict))

        output = TaskOutput(
            turn_1_response=answers[0],
            # The whole thread, so a three- or four-turn item reaches the judge as a
            # thread instead of as its first two turns.
            turn_responses=answers,
            turn_prompts=prompts,
            turn_follow_ups=follow_ups,
            turn_components=components,
            turn_verified=verified,
            turn_sources=sources,
            turn_tool_calls=tool_calls_per_turn,
            turn_1_verified=verified[0],
            tool_evidence=tool_evidence,
            session_id=session_id,
            langfuse_trace_id=turn_traces[0],
            turn_trace_ids=list(turn_traces),
            turn1_trace=turn_traces[0],
            turn_1_status=first.get("status"),
            turn_1_command=first.get("command") or first.get("command_executed"),
            # The service does not send an `a2ui_components` field; the names live
            # inside the messages. Reading the missing field reported [] for every
            # turn, which made three component assertions fail on a card that was
            # actually rendered (Step 8, 2026-10-20).
            turn_1_components=components[0],
            turn_1_card=first.get("a2ui") if isinstance(first.get("a2ui"), dict) else None,
            turn_1_error=first.get("error"),
        )

        if len(responses) > 1:
            second = responses[1]
            output.turn_2_response = answers[1] or None
            output.turn_2_status = second.get("status")
            output.turn_2_trace_id = turn_traces[1]
            output.turn2_trace = turn_traces[1]

        return output.model_dump()

    return task
