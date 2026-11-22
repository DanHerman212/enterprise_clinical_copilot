"""The chips and driver questions each dev record offers — zero LLM calls.

Section 2.3 of `docs/ux_workshop/strategy_and_tactics/05-post_eval_ux_augmentation.md`
asks for a matrix over the Dev records rather than a handful of unit fixtures: the
templates are only useful if they fire on the records the product serves, and only
correct if every chip names something the record actually holds.

This script builds, for each record and each command, the `state` the service would hold
at the end of Turn 1 — the same tool calls the command declares, from the same tools —
and runs the deterministic selector over it. It also records, for every ranked driver on
every inpatient chart, the exact question a driver bar asks, so a Tier 2 `risk_driver`
turn can copy it rather than reconstruct `humanize()` and the template by hand. It reads
the store through the tools and calls no model.

It writes `evaluation/dev_followup_matrix.md`. The cohort comes from `dev_suite`, so the
matrix cannot drift from the list the suite runs against.

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

from shared.labels import driver_question, humanize  # noqa: E402
from services.agent.postprocess import followups
from services.agent.engine.mcp_client import _presented_for_model  # noqa: E402
from services.agent.routing import routing  # noqa: E402
from evaluation.outpatient_oracle import OUTPATIENT_HADM_IDS  # noqa: E402
from evaluation.platform import dev_suite  # noqa: E402
from services.mcp.tools import (  # noqa: E402
    discharge_medications,
    get_lab_trends,
    predict_readmission,
    rag_search,
    rag_search_sections,
)

OUT = REPO / "evaluation" / "dev_followup_matrix.md"

DEV = tuple(int(hadm) for hadm in dev_suite.DEV_HADM_IDS)

#: What each record is for. The seven originals are the archetypes the workbook named;
#: the five added on 2026-11-22 are named by the flag that got them in.
ARCHETYPE = {
    90000015: "complex meds, a documented taper, anticoagulated",
    90000108: "oncology, above threshold, critical haemoglobin",
    90000106: "narrative-only medication section",
    90000105: "all-normal lab panel",
    90000023: "irregular / sparse note",
    90000051: "outpatient encounter",
    90000073: "outpatient encounter",
    90000045: "critical haemoglobin, dedicated meds, above threshold",
    90000046: "urgent medical admission, dedicated meds, above threshold",
    90000103: "taper beside a dedicated meds section, below threshold",
    90000065: "taper, no dedicated meds section (narrative fallback)",
    90000035: "no dedicated meds section, below threshold",
}

#: A turn's answer has to be non-empty for the selector to run at all; its text is not
#: an input to any template, so a placeholder is honest here.
ANSWER = "(matrix run — no answer text is read by the selector)"


def _call(name: str, response: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "args": {}, "response": response}


async def _presented_risk(hadm_id: int) -> dict[str, Any]:
    """The estimate as the turn's record holds it, not as the tool returned it.

    The engine projects every tool response through `_presented_for_model` before it
    stores the call, so the record names each driver by its humanised label rather than
    by the model's column key. A state built from the raw payload is a state the service
    never holds: `monocytes_min` there reached the selector as a chip reading "lowest
    monocyte count" where the live row reads "Lowest Monocyte Count".
    """
    payload = await predict_readmission(hadm_id)
    return _presented_for_model("predict_readmission", payload)


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
        calls.append(_call("predict_readmission", await _presented_risk(hadm_id)))
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

#: The row the selector fills. Named once so a change to the constant is visible here.
ROW = followups.FOLLOW_UP_COUNT


def _cells(chips: list[str], row: int) -> list[str]:
    """The first `row` chips, with the empty slots made explicit.

    A short row is the finding, not a rendering detail: the reader has to see that slot
    `[2]` is empty rather than count the pipes.
    """
    padded = list(chips[:row]) + ["—"] * max(0, row - len(chips))
    return [cell.replace("|", "\\|") for cell in padded]


async def _drivers(hadm_id: int) -> list[tuple[str, str, str, str]]:
    """Every ranked driver on this chart: feature, label, direction, exact chip text."""
    payload = await _presented_risk(hadm_id)
    out: list[tuple[str, str, str, str]] = []
    for driver in payload.get("top_factors") or []:
        feature = str(driver.get("feature") or driver.get("name") or "")
        if not feature:
            continue
        label = humanize(feature)
        out.append((feature, label, str(driver.get("direction") or "—"),
                    driver_question(label)))
    return out


async def main() -> int:
    problems: list[str] = []
    short: list[str] = []
    blocks: list[str] = []

    for hadm_id in DEV:
        outpatient = hadm_id in OUTPATIENT_HADM_IDS
        lines = [f"## {hadm_id} — {ARCHETYPE[hadm_id]}", ""]
        if outpatient:
            lines += ["Outpatient encounter: the chart has no admission and no risk "
                      "estimate, so no driver bar and no driver question.", ""]
        lines += ["| command | " + " | ".join(f"`[{i}]`" for i in range(ROW)) + " | chips |",
                  "|---" * (ROW + 2) + "|"]
        for question, builder in COMMANDS:
            state = await builder(hadm_id)
            chips = followups.suggest(
                question, ANSWER, scope="patient_chart", state=state,
                outpatient=outpatient,
            )
            lines.append(f"| `{question}` | " + " | ".join(_cells(chips, ROW))
                         + f" | {len(chips)} |")
            if len(chips) < ROW:
                short.append(f"{hadm_id} {question} → {len(chips)}")
            for chip in chips:
                if routing.asks_for_literature(chip):
                    problems.append(f"{hadm_id} {question}: literature cue: {chip}")
                if len(chip) > followups._MAX_QUESTION_CHARS:
                    problems.append(f"{hadm_id} {question}: over cap: {chip}")
        lines.append("")
        if not outpatient:
            ranked = await _drivers(hadm_id)
            lines += ["| # | feature | label | direction | chip (exact) |",
                      "|---|---|---|---|---|"]
            for index, (feature, label, direction, chip) in enumerate(ranked, start=1):
                lines.append(f"| {index} | `{feature}` | {label} | {direction} "
                             f"| {chip.replace('|', chr(92) + '|')} |")
            lines.append("")
        blocks.append("\n".join(lines))

    header = [
        "# Dev UI Affordances — Phase 04",
        "",
        "Generated by `evaluation/inventory/dev_followup_matrix.py`: the deterministic "
        "follow-up selector and the driver-bar question template, run over the "
        f"{len(DEV)} dev records against the same tool payloads the commands retrieve. "
        "**Zero LLM calls.**",
        "",
        f"The row holds up to {ROW} chips; `—` marks a slot the selector left empty. "
        "`/meds` runs without the model's extracted medication rows (the matrix cannot "
        "produce them without an LLM call), so a regimen chip may name a different pair "
        "of drugs live.",
        "",
        "Driver questions are the exact string a driver bar sends, produced by "
        "`driver_question(humanize(feature))`. A `risk_driver` turn copies one of them "
        "rather than rebuilding it.",
        "",
    ]
    if short:
        header += ["## Command/record pairs offering fewer than "
                   f"{ROW} chips", ""]
        header += [f"- {item}" for item in short]
        header += [""]
    if problems:
        header += ["## Violations", ""] + [f"- {item}" for item in problems] + [""]

    OUT.write_text("\n".join(header) + "\n" + "\n".join(blocks) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(DEV)} records, 0 LLM calls)")
    print(f"{len(short)} command/record pair(s) offered fewer than {ROW} chips")
    if problems:
        print("VIOLATIONS")
        for line in problems:
            print(" ", line)
        return 1
    print("no chip carried a literature cue; every chip is within the row cap")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
