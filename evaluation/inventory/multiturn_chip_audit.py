"""Section 5.2: the end-to-end multi-turn chip audit, against the live agent.

`docs/ux_workshop/strategy_and_tactics/05-a1-multi_turn_followup_audit.md` asks two
questions that only a live run can answer:

  1. Does clicking a chip consume it? The follow-up row a reader is shown on Turn n+1
     must not contain the chip they clicked on Turn n, nor any earlier question in the
     thread. That is `followups.suggest(asked=...)`, and `asked` is rebuilt from the
     replayed turns — so the only faithful test is a real thread over the wire.
  2. Is the Turn-2 answer any good? A chip is only offered when the record holds
     something for it, so a Turn-2 answer that says the chart does not document it is a
     defect either in the trigger (the chip should not have fired) or in the answer path
     (the loop did not retrieve or cite what is there).

This script drives `/ask` exactly as the browser does — the same `turns` payload the
site replays, the same `follow_ups: true` — and writes one JSON record per
`(record, turn_1_command, chip)` tuple plus the three-turn progression per
`(record, turn_1_command)`.

Three turns per progression, per §5.2.1: Turn 1 is the command, Turn 2 clicks
`follow_ups[0]`, Turn 3 clicks Turn 2's `follow_ups[0]`. That is the sequence a reader
performs, and the one where a repeated chip would show.

    API_URL=https://agent-jamycsjjzq-ue.a.run.app \
        ./.venv/bin/python evaluation/inventory/multiturn_chip_audit.py \
        --out evaluation/inventory/multiturn_chip_audit.json

Resumable: an existing `--out` file is read and its completed tuples are skipped, so a
run that dies on a transient 502 does not re-bill the turns it already did.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.platform.run import DEFAULT_AGENT, _identity_token, make_ask  # noqa: E402

#: §5.2.1/§5.2.2: the five inpatient Dev archetypes. Outpatients (90000051/90000073)
#: are out of scope here — they have no note sections, so a chip row is the one thing
#: they never offer.
RECORDS = (90000015, 90000108, 90000106, 90000105, 90000023)

#: The two commands that offer chips. `/risk` is a widget turn with no chips by
#: construction, and `/labs` is audited by the lab-trend tests.
COMMANDS = ("/ehr", "/meds")

#: The number of chips to click per Turn 1 — the row is at most three.
MAX_CHIPS = 3

#: What a turn says when the record does not hold the thing it was asked. Matched
#: case-insensitively; these are the wordings the deterministic and prompt layers use.
ABSENT_MARKERS = (
    "not documented",
    "no information",
    "not available",
    "is not available",
    "does not contain",
    "not recorded",
    "no record of",
    "not mentioned",
)

_CITE = re.compile(r"\^\[\d+\]")


def _turn_payload(asked: str, answer: str, calls: list[dict]) -> dict:
    """One replayed turn, in the shape the site stores and sends back.

    The site keeps a tool's response only when the agent says it cannot be produced
    again (`derivable` is false); everything else is re-resolved from the tools on
    replay. Sending a response for a derivable call would make the audit's Turn 2
    cheaper than the product's, and would hide a replay defect behind a payload the
    browser never sends.
    """
    replay = []
    for call in calls or []:
        entry: dict[str, Any] = {
            "name": call.get("name"),
            "args": call.get("args") or {},
        }
        if not call.get("derivable", True):
            entry["payload"] = call.get("response")
        replay.append(entry)
    return {"question": asked, "answer": answer, "tool_calls": replay}


def _tools(calls: list[dict]) -> str:
    return ", ".join(call.get("name") or "?" for call in calls or []) or "none"


def _read(envelope: dict) -> dict:
    """The parts of a `/ask` envelope the audit scores, flattened."""
    return {
        "status": envelope.get("status"),
        "answer": envelope.get("answer") or "",
        "verified": envelope.get("verified"),
        "scope": envelope.get("scope"),
        "sources": envelope.get("sources") or [],
        "follow_ups": envelope.get("follow_ups") or [],
        "tool_calls": envelope.get("tool_calls") or [],
        "guardrail_flags": envelope.get("guardrail_flags") or [],
        "error": envelope.get("error"),
    }


def _verdict(turn: dict, *, chip: str, asked_before: list[str]) -> tuple[str, str]:
    """GOOD / BAD plus the reason, from §5.2.2's four flag conditions.

    Scored on what a reader is given, not on what the pipeline intended: an answer
    counts as cited when it carries a marker AND that marker resolves to a published
    source, which is the pair the console renders together.
    """
    if turn["status"] != 200:
        return "BAD", f"http {turn['status']} {turn['error'] or ''}".strip()

    answer = turn["answer"]
    if not answer.strip():
        return "BAD", "empty answer"

    lowered = answer.lower()
    for marker in ABSENT_MARKERS:
        if marker in lowered:
            return "BAD", f"reports absent: '{marker}'"

    cited = bool(_CITE.search(answer)) and bool(turn["sources"])
    if not cited:
        # A turn can be grounded without a passage: a lab trajectory, or the model's
        # own estimate. `a2ui.compose_presentation` badges those `verified` with no
        # `^[n]` by design, so scoring them as uncited would report a contract the
        # product never made. Read the same channels it does, off the tools the turn
        # called.
        trajectory = any(call.get("name") in ("get_lab_trends", "predict_readmission")
                         for call in turn["tool_calls"])
        if turn["verified"] is True and trajectory:
            return "GOOD", ""
        return "BAD", "no resolving ^[n] citation"
    if turn["verified"] is not True:
        return "BAD", "verified is not true"

    # The chip itself must be gone. A row that offers back the question just answered
    # is the repetition §5.2.1 is about.
    normalise = lambda s: " ".join(str(s).lower().split())  # noqa: E731
    repeated = [q for q in asked_before if normalise(q) in
                [normalise(p) for p in turn["follow_ups"]]]
    if repeated:
        return "BAD", "follow_ups repeat an asked question"

    return "GOOD", ""


def _excerpt(answer: str, limit: int = 240) -> str:
    flat = " ".join((answer or "").split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _ask_turn(ask, *, question: str, hadm_id: int, turns: list[dict],
              run_tag: str) -> dict:
    envelope = ask(question=question, hadm_id=hadm_id, run_tag=run_tag,
                   turns=turns or None, follow_ups=True)
    return _read(envelope)


def _audit_command(ask, *, hadm_id: int, command: str, run_tag: str) -> dict:
    """Turn 1, every chip from it, and the three-turn progression."""
    turn1 = _ask_turn(ask, question=command, hadm_id=hadm_id, turns=[],
                      run_tag=run_tag)
    chips = turn1["follow_ups"][:MAX_CHIPS]

    rows = []
    for chip in chips:
        turn2 = _ask_turn(
            ask, question=chip, hadm_id=hadm_id,
            turns=[_turn_payload(command, turn1["answer"], turn1["tool_calls"])],
            run_tag=run_tag,
        )
        verdict, reason = _verdict(turn2, chip=chip,
                                   asked_before=[command, chip])
        rows.append({
            "record": hadm_id,
            "turn_1_cmd": command,
            "chip": chip,
            "turn_2_tools": _tools(turn2["tool_calls"]),
            "turn_2_answer": _excerpt(turn2["answer"]),
            "turn_2_sources": len(turn2["sources"]),
            "turn_2_verified": turn2["verified"],
            "turn_2_follow_ups": turn2["follow_ups"],
            "turn_2_status": turn2["status"],
            "verdict": verdict,
            "reason": reason,
        })

    # The progression: Turn 1 -> click chips[0] -> click that turn's chips[0].
    progression = {"record": hadm_id, "turn_1_cmd": command,
                   "turn_1_follow_ups": turn1["follow_ups"], "turns": []}
    if chips:
        first = chips[0]
        turn2 = _ask_turn(
            ask, question=first, hadm_id=hadm_id,
            turns=[_turn_payload(command, turn1["answer"], turn1["tool_calls"])],
            run_tag=run_tag,
        )
        progression["turns"].append({"question": first,
                                     "follow_ups": turn2["follow_ups"]})
        if turn2["follow_ups"]:
            second = turn2["follow_ups"][0]
            turn3 = _ask_turn(
                ask, question=second, hadm_id=hadm_id,
                turns=[
                    _turn_payload(command, turn1["answer"], turn1["tool_calls"]),
                    _turn_payload(first, turn2["answer"], turn2["tool_calls"]),
                ],
                run_tag=run_tag,
            )
            progression["turns"].append({"question": second,
                                         "follow_ups": turn3["follow_ups"]})
    return {"rows": rows, "progression": progression, "turn_1": turn1}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--api-url", default=DEFAULT_AGENT)
    ap.add_argument("--out", default=str(HERE / "multiturn_chip_audit.json"))
    ap.add_argument("--only", default=None,
                    help="comma-separated hadm_ids (default: the five archetypes)")
    ap.add_argument("--commands", default=",".join(COMMANDS))
    ap.add_argument("--revision", default="",
                    help="expected serving revision; recorded with the result")
    args = ap.parse_args(argv)

    out = Path(args.out)
    done: dict[str, dict] = {}
    if out.exists():
        done = json.loads(out.read_text())

    records = tuple(int(v) for v in args.only.split(",")) if args.only else RECORDS
    commands = tuple(c.strip() for c in args.commands.split(",") if c.strip())
    token = _identity_token()
    ask = make_ask(args.api_url, token)
    run_tag = f"chip-audit-{time.strftime('%Y%m%d-%H%M%S')}"

    for hadm_id in records:
        for command in commands:
            key = f"{hadm_id}:{command}"
            if key in done:
                print(f"skip {key} (already audited)", flush=True)
                continue
            print(f"audit {key} …", flush=True)
            result = _audit_command(ask, hadm_id=hadm_id, command=command,
                                    run_tag=run_tag)
            done[key] = result
            out.write_text(json.dumps(done, indent=2))
            bad = sum(1 for row in result["rows"] if row["verdict"] == "BAD")
            print(f"  {len(result['rows'])} chip(s), {bad} BAD", flush=True)

    print(f"\nwrote {out} ({len(done)} tuple(s), revision={args.revision or 'not recorded'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
