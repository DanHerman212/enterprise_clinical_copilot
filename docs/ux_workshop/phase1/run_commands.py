"""Drive the four commands at the three anchor patients, and record what came back.

UX Phase 1 live pass. Sends the request bodies the composer sends — a `chip` and
a `hadm_id`, exactly as `copilot_canvas.js` builds them — so the pass exercises
the same entry the browser uses, without a browser.

The invariants are asserted here rather than by eye, and the table below is
written out longhand rather than imported from `services.agent.routing`: a check
that reads the table it is checking cannot fail when the table changes, and the
change is exactly what this pass is looking for.

Writes, under this directory:
  live/<hadm_id>.md      every turn verbatim, in the order it was run
  live/results.json      the same turns, machine-readable, for the diff

Usage:
  AGENT_URL=http://127.0.0.1:8010 python docs/ux_workshop/phase1/run_commands.py
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

AGENT = os.environ.get("AGENT_URL", "http://127.0.0.1:8010").rstrip("/")
OUT = Path(__file__).resolve().parent / "live"

# The three anchor patients, from the UX capture.
PATIENTS = [
    (90000023, "Bruce Hollingsworth"),
    (90000019, "Susan Okafor"),
    (90000015, "Cynthia Petrov"),
]

# wire name -> (expected tool, expected component, the topic to send, or None)
COMMANDS = [
    ("risk", "predict_readmission", "risk_visualizer", None),
    ("meds", "discharge_medications", "medication_table", None),
    ("summarize", "rag_search_sections", "narrative_with_sources", None),
    # `/pubmed` needs the clinician's own words: the tool takes a search string
    # and no admission. These are the questions the baseline's model derived for
    # each patient from the note; a clinician would type them.
    ("literature", "search_literature", "literature_cards", None),
]

# The topic to send with the literature command, per patient.
TOPICS = {
    90000023: "coronary artery disease bare metal stent dual antiplatelet therapy",
    90000019: "total hip replacement postoperative constipation management",
    90000015: "total knee replacement rehabilitation CPM machine pain management",
}

ALL_TOOLS = {
    "predict_readmission", "rag_search", "rag_search_sections",
    "discharge_medications", "search_literature",
}

TIMEOUT_SECONDS = 300


def ask(chip: str, hadm_id: int, topic: str | None = None) -> tuple[int, dict]:
    body: dict = {"chip": chip, "hadm_id": hadm_id}
    if topic:
        body["question"] = topic
    request = urllib.request.Request(
        f"{AGENT}/ask",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def verdicts(chip: str, tool: str, component: str, payload: dict) -> list[str]:
    """The boundary claims this turn either supports or refutes."""
    calls = payload.get("tool_calls") or []
    names = [call.get("name") for call in calls]
    components = [c.get("component") for c in payload.get("ui_components") or []]
    forbidden = ALL_TOOLS - {tool}

    problems = []
    if len(calls) != 1:
        problems.append(f"expected exactly one tool call, saw {names}")
    elif names[0] != tool:
        problems.append(f"expected {tool}, called {names[0]}")
    if set(names) & forbidden:
        problems.append(f"called a forbidden tool: {sorted(set(names) & forbidden)}")
    if components != [component]:
        problems.append(f"expected component {component}, saw {components}")
    if not (payload.get("narrative_text") or "").strip():
        problems.append("no narrative text")
    if payload.get("citations") != payload.get("sources"):
        problems.append("citations and sources disagree")
    return problems


def render(hadm_id: int, patient: str, turns: list[dict]) -> str:
    lines = [f"# {patient} — admission {hadm_id}", ""]
    for turn in turns:
        lines += [
            f"## /{turn['wire']}",
            "",
            f"- status: {turn['status']}   elapsed: {turn['seconds']}s",
            f"- topic: {turn['topic']}",
            f"- tool calls: {json.dumps(turn['tool_calls'])}",
            f"- components: {json.dumps(turn['components'])}",
            f"- citations: {turn['citations']}   guardrail flags: {turn['flags']}",
        ]
        if turn["medication_table"] is not None:
            table = turn["medication_table"]
            lines += [
                f"- medications verified: {len(table.get('medications') or [])}"
                f"   dropped: {len(table.get('dropped') or [])}"
                f"   passages: {table.get('returned_passages')}",
                f"- dropped rows: {json.dumps(table.get('dropped') or [])}",
            ]
        if turn["literature"] is not None:
            literature = turn["literature"]
            lines.append(
                f"- literature returned: {literature.get('returned')}"
                f"   degraded: {literature.get('degraded')}"
            )
        if turn["problems"]:
            lines.append(f"- **PROBLEMS**: {'; '.join(turn['problems'])}")
        lines += ["", "### narrative", "", turn["narrative"] or "_(empty)_", ""]
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    everything = []
    failures = 0

    for hadm_id, patient in PATIENTS:
        turns = []
        for wire, tool, component, _ in COMMANDS:
            topic = TOPICS[hadm_id] if wire == "literature" else None
            started = time.monotonic()
            status, payload = ask(wire, hadm_id, topic)
            elapsed = round(time.monotonic() - started, 1)

            calls = payload.get("tool_calls") or []
            components = payload.get("ui_components") or []
            medication_table = next(
                (c.get("props") for c in components
                 if c.get("component") == "medication_table"), None)
            turn = {
                "hadm_id": hadm_id,
                "patient": patient,
                "wire": wire,
                "topic": topic,
                "status": status,
                "seconds": elapsed,
                "narrative": payload.get("narrative_text") or payload.get("answer") or "",
                "tool_calls": [
                    {"name": c.get("name"), "args": c.get("args"),
                     "derivable": c.get("derivable")}
                    for c in calls
                ],
                "components": [c.get("component") for c in components],
                "citations": len(payload.get("citations") or []),
                "sources": len(payload.get("sources") or []),
                "flags": payload.get("guardrail_flags"),
                "literature": payload.get("literature"),
                "medication_table": medication_table,
                "code_revision": payload.get("code_revision"),
                "trace": payload.get("langfuse_trace_id"),
                "error": payload.get("error"),
                "message": payload.get("message"),
            }
            turn["problems"] = (
                verdicts(wire, tool, component, payload) if status == 200
                else [f"HTTP {status}: {payload.get('error')}"]
            )
            failures += len(turn["problems"])
            turns.append(turn)
            everything.append(turn)

            flag = "ok  " if not turn["problems"] else "FAIL"
            print(f"{flag} {patient[:18]:18} /{wire:10} {elapsed:6.1f}s "
                  f"tools={[c['name'] for c in turn['tool_calls']]} "
                  f"components={turn['components']}", flush=True)
            for problem in turn["problems"]:
                print(f"       - {problem}", flush=True)

        (OUT / f"{hadm_id}.md").write_text(
            render(hadm_id, patient, turns), encoding="utf-8")

    (OUT / "results.json").write_text(
        json.dumps(everything, indent=2), encoding="utf-8")

    print(f"\n{len(everything)} turns, {failures} problems")

    # The refusal, checked as its own turn: a command that needs the clinician's
    # words must say so rather than answering with something else.
    status, payload = ask("literature", PATIENTS[0][0])
    prompt = payload.get("message") or payload.get("error") or ""
    ok = status == 400 and "clinical topic" in prompt
    print(f"{'ok  ' if ok else 'FAIL'} /pubmed with no topic -> {status} {prompt!r}")
    failures += 0 if ok else 1

    print(f"written to {OUT}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
