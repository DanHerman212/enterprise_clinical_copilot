"""The chips each dev record offers, by command — zero LLM calls.

Section 2.3 of `docs/ux_workshop/strategy_and_tactics/05-post_eval_ux_augmentation.md`
asks for a matrix over the seven Dev archetype records rather than a handful of unit
fixtures: the templates are only useful if they fire on the records the product serves,
and only correct if every chip names something the record actually holds.

This script builds, for each record and each command, the `state` the service would hold
at the end of Turn 1 — the same tool calls the command declares, from the same tools —
and runs the deterministic selector over it. It reads the store through the tools and
calls no model.

One accounting note, because it explains a difference a reader might otherwise take for
a bug: `command_payload["medications"]` is the model's extraction, so it cannot be
reproduced here without an LLM call. The matrix therefore runs `/meds` with no verified
rows, which makes the selector fall back to the drug-class table. On the live path the
verified rows are present and are preferred, so the two can differ only in *which* two
drugs a regimen chip names.

    ./.venv/bin/python evaluation/inventory/dev_followup_matrix.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from services.agent.postprocess import followups
from services.agent.routing import routing  # noqa: E402
from evaluation.outpatient_oracle import OUTPATIENT_HADM_IDS  # noqa: E402
from services.mcp.tools import (  # noqa: E402
    discharge_medications,
    get_lab_trends,
    predict_readmission,
    rag_search,
    rag_search_sections,
)

DEV = (90000015, 90000108, 90000106, 90000105, 90000023, 90000051, 90000073)

ARCHETYPE = {
    90000015: "1. Complex meds / tapers",
    90000108: "2. Oncology, high risk, abnormal labs",
    90000106: "3a. Empty meds",
    90000105: "3b. All-normal labs",
    90000023: "4. Irregular / sparse note",
    90000051: "5a. Outpatient boundary",
    90000073: "5b. Outpatient boundary",
}

#: A turn's answer has to be non-empty for the selector to run at all; its text is not
#: an input to any template, so a placeholder is honest here.
ANSWER = "(matrix run — no answer text is read by the selector)"


def _call(name: str, response: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "args": {}, "response": response}


async def _ehr_state(hadm_id: int) -> dict[str, Any]:
    sections = await rag_search_sections(hadm_id)
    calls = [_call("rag_search_sections", sections)]
    # The service adds one record-wide search when the sectioned call returns nothing;
    # without it an outpatient record looks like a record with no sections rather than
    # a record with nothing at all.
    if not (sections or {}).get("passages"):
        calls.append(_call("rag_search", await rag_search(
            hadm_id, routing.EHR_RECORD_QUERY)))
    if hadm_id not in OUTPATIENT_HADM_IDS:
        calls.append(_call("predict_readmission", await predict_readmission(hadm_id)))
    return {"tool_calls": calls, "command": "summarize", "replayed_tool_calls": []}


async def _meds_state(hadm_id: int) -> dict[str, Any]:
    payload = await discharge_medications(hadm_id)
    passages = (payload or {}).get("passages") or []
    return {
        "tool_calls": [_call("discharge_medications", payload)],
        "command": "meds",
        "replayed_tool_calls": [],
        # No verified rows: see the module docstring.
        "command_payload": {"medications": [], "dropped": [],
                            "returned_passages": len(passages)},
    }


async def _labs_state(hadm_id: int) -> dict[str, Any]:
    payload = await get_lab_trends(hadm_id)
    return {"tool_calls": [_call("get_lab_trends", payload)],
            "command": "labs", "replayed_tool_calls": []}


COMMANDS = (("/ehr", _ehr_state), ("/meds", _meds_state), ("/labs", _labs_state))


async def main() -> int:
    print(f"{'record':<10} {'archetype':<34} {'command':<7} chips")
    print("-" * 118)
    violations: list[str] = []
    for hadm_id in DEV:
        outpatient = hadm_id in OUTPATIENT_HADM_IDS
        for question, builder in COMMANDS:
            state = await builder(hadm_id)
            chips = followups.suggest(
                question, ANSWER, scope="patient_chart", state=state,
                outpatient=outpatient,
            )
            print(f"{hadm_id:<10} {ARCHETYPE[hadm_id]:<34} {question:<7} "
                  f"{chips if chips else '(none)'}")
            for chip in chips:
                if routing.asks_for_literature(chip):
                    violations.append(f"{hadm_id} {question}: literature cue: {chip}")
                if len(chip) > followups._MAX_QUESTION_CHARS:
                    violations.append(f"{hadm_id} {question}: over cap: {chip}")
        print()
    print("-" * 118)
    if violations:
        print("VIOLATIONS")
        for line in violations:
            print(" ", line)
        return 1
    print("no chip carried a literature cue; every chip is within the row cap")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
