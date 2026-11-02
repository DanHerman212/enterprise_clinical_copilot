"""STAGE GATE 2 — the wired `/ehr` pipeline on live admissions, printed in full.

What was unverified before this ran: the extraction pipeline is proven against real charts
(Gates 1, 1B–1E), the canvas is proven against fixtures, and nothing had driven the two
together. This script drives the production path for one `/ehr` turn — the real MCP
toolbox, the real Vertex client, the real PubMed retrieval — and then composes the
presentation from the state that came back, so the table printed below is the table a
reader would be shown rather than a reconstruction of it.

Six admissions: the three the directive names, and three the extraction work never saw.

    .venv/bin/python scripts/agent/verify_gate2_live.py

Nothing here asserts clinical quality. It prints; a human reads it and decides.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

#: The repository root, as this script sees it: `scripts/agent/` → three levels up.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.agent.engine import deterministic
from services.agent.engine import graph
from services.agent.engine import mcp_client
from services.agent.engine import model as model_gateway
from services.agent.presentation.a2ui import compose_presentation
from services.agent.routing import routing
from services.agent.telemetry import observability

#: The three admissions named in the directive, plus three held out from every diagnostic
#: pass so far: 90000001 has a single diagnosis, 90000019 is the hypertension admission
#: the earlier gates never reached, 90000037 is an untried chart.
COHORT = (90000090, 90000069, 90000081, 90000001, 90000019, 90000037)

SUMMARIZE = routing.command_for("summarize")


def _components(presentation: dict) -> list[str]:
    names = []
    for message in (presentation.get("a2ui") or {}).get("messages") or []:
        update = (message or {}).get("updateComponents") or {}
        names += [c.get("component") for c in update.get("components") or []]
    return [name for name in names if name]


def _matrix(presentation: dict) -> dict | None:
    for message in (presentation.get("a2ui") or {}).get("messages") or []:
        update = (message or {}).get("updateComponents") or {}
        for component in update.get("components") or []:
            if component.get("component") == "ComparativeRiskMatrix":
                return component
    return None


def _print_one(hadm_id: int, state: dict, presentation: dict) -> None:
    print(f"\n{'=' * 78}\nADMISSION {hadm_id}\n{'=' * 78}")

    target = state.get("clinical_target") or {}
    print(f"[1] TARGET: {target.get('condition') or '(none)'}"
          f" | procedure: {target.get('procedure') or '-'}")
    print(f"    ANCHORS: {', '.join(target.get('anchors') or []) or '(none)'}")

    print("[2] TRACED CALLS")
    for call in state.get("tool_calls") or []:
        response = call.get("response") or {}
        if not isinstance(response, dict):  # pragma: no cover - defensive
            continue
        detail = ""
        if call.get("name") == "search_literature":
            detail = (f" query={response.get('query')!r}"
                      f" returned={response.get('returned')}")
        elif isinstance(response.get("passages"), list):
            detail = f" passages={len(response['passages'])}"
        elif isinstance(response.get("probability"), float):
            detail = (f" probability={response['probability']:.4f}"
                      f" threshold={response.get('threshold')}")
        print(f"    - {call.get('name')} {call.get('args')}{detail}")

    for entry in state.get("comparator_passes") or []:
        label = "WIDE" if entry.get("wide") else "PASS 1"
        print(f"[3] {label}: query={entry.get('query')!r}"
              f" candidates={entry.get('candidates')}"
              f" aligned={entry.get('aligned')} verified={entry.get('verified')}")

    matrix = _matrix(presentation)
    print("[4] COMPONENTS:", ", ".join(_components(presentation)) or "(none)")
    if matrix is None:
        print("[5] MATRIX: not drawn")
        return

    baseline = matrix.get("baseline")
    if baseline:
        print(f"[5] BASELINE: {baseline['probability']:.4f} "
              f"(threshold {baseline['threshold']:.2f}, decision {baseline['decision']}, "
              f"{baseline['provenance']})")
    else:
        print("[5] BASELINE: none — no estimate on this turn")
    print(f"    NOTE: {matrix.get('baseline_note') or '(none)'}")

    rows = matrix.get("rows") or []
    print(f"[6] ROWS ({len(rows)})")
    for row in rows:
        print(f"    · {row.get('pmid')} | {row.get('source')} | "
              f"{row.get('reported_rate')} ({row.get('rate_kind')}) | "
              f"{row.get('evidence_tier')}")
        print(f"      cohort: {row.get('cohort_summary')}")
        print(f"      sentence: {row.get('excerpt')}")

    print(f"[7] SOURCES ({len(presentation.get('sources') or [])})")
    for source in presentation.get("sources") or []:
        print(f"    ^{source.get('cite')} {source.get('kind') or source.get('section')}"
              f" | {source.get('label') or source.get('title') or ''}")


async def _drive() -> None:
    llm = model_gateway.build_extraction_llm(observability.MODEL_ID)
    async with mcp_client.toolbox() as box:
        for hadm_id in COHORT:
            try:
                state = await deterministic.run_command(
                    box, SUMMARIZE, hadm_id,
                    question="/ehr",
                    model=observability.MODEL_ID,
                    extraction_llm=llm,
                )
            except Exception as error:  # noqa: BLE001 — one admission must not stop the run
                print(f"\n{'=' * 78}\nADMISSION {hadm_id}\n{'=' * 78}")
                print(f"[!] the turn failed: {type(error).__name__}: {error}")
                continue

            presentation = compose_presentation(
                "/ehr", graph.final_text(state) or "",
                [
                    {"name": call["name"], "args": call.get("args") or {},
                     "response": call.get("response")}
                    for call in state.get("tool_calls") or []
                ],
                state.get("command_payload"),
                state.get("command"),
                comparators=state.get("comparators") or [],
            )
            _print_one(hadm_id, state, presentation)


def main() -> None:
    asyncio.run(_drive())


if __name__ == "__main__":
    main()
