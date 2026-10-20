"""Step 7 — the mandatory local end-to-end dry run.

Runs the real task, the real evaluators and the real composite/run evaluators, with a
mocked agent and a mocked judge, against local experiment items. Nothing touches the
network, so it finishes in seconds and can run on every change.

It asserts the five behaviours the spec requires, and exits non-zero if any is missing:

1. a dataset item without a manifest is refused by the schema
2. the manifest reaches the judge's prompt
3. a structural failure short-circuits the judge (no model call)
4. the parrot rule fails an answer that ignores the follow-up
5. the run evaluator produces pass_rate / safety_failures / gate_pass

Usage: ./.venv/bin/python -m evaluation.platform.dry_run
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (REPO, REPO / "evaluation" / "agent"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pydantic import ValidationError  # noqa: E402

from evaluation.platform import evaluators as ev  # noqa: E402
from evaluation.platform.fixtures import census_items, suite2_items  # noqa: E402
from evaluation.platform.judge_call import build_user_prompt  # noqa: E402
from evaluation.platform.schemas import DatasetItemInput  # noqa: E402
from evaluation.platform.task import make_task  # noqa: E402

MANIFEST = {
    "age": "70",
    "gender": "Male",
    "admission_type": "Urgent (URGENT)",
    "prior_inpatient_days": "2",
    "index_los_days": "4.5",
}

DIMENSIONS = ("fidelity", "grounding", "citation", "responsiveness", "safety", "context")


def fake_manifest(hadm_ids: list[int]) -> dict[int, dict]:
    """Stands in for `build_census.manifests_by_hadm_id` — same shape, no BigQuery."""
    return {int(h): dict(MANIFEST) for h in hadm_ids}


class FakeAgent:
    """Canned envelopes, one scenario per command. No network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bool]] = []  # (question, session, is_turn2)

    def __call__(self, *, question, hadm_id, run_tag, session_id, turns=None):
        self.calls.append((question, session_id, turns is not None))
        # Match the command exactly: `/ehrr` must not be served by the `/ehr` branch,
        # or an expected-refusal probe gets a 200 and its refusal path is never tested.
        command = question.split()[0].lower() if question.strip() else ""
        if turns is not None:
            # Turn 2. The meds follow-up is the parrot: it restates turn 1.
            if "medication" in question.lower() or "anticoag" in question.lower():
                return {"status": 200, "answer": "PARROT: two high-risk medications are "
                                                  "listed on the discharge note."}
            return {"status": 200, "answer": "The discharge plan records follow-up."}

        if command == "/risk":
            # Widget-only turn: no prose at all, a card instead.
            return {
                "status": 200,
                "command": "risk",
                "answer": "",
                "a2ui": {"probability_display": "16.7%", "threshold_display": "20.0%",
                         "top_factors": ["prior_inpatient_days", "index_los_days"]},
                "a2ui_components": ["Column", "RiskInfluencers", "Text"],
                "tool_calls": [{"name": "predict_readmission",
                                "response": {"probability_display": "16.7%"}}],
            }
        if command == "/labs":
            return {"status": 200, "command": "labs",
                    "answer": "SAFETY0 creatinine rose from 1.0 to 2.4 mg/dL.",
                    "a2ui_components": ["Column", "LabTrendChart"],
                    "tool_calls": [{"name": "get_lab_trends",
                                    "response": {"trends": [], "synthetic": True,
                                                 "note": "values are synthetic"}}]}
        if command == "/meds":
            return {"status": 200, "command": "meds",
                    "answer": "Two high-risk medications are listed.",
                    "a2ui_components": ["Column", "MedicationTable"],
                    "tool_calls": [{"name": "discharge_medications",
                                    "response": {"medications": []}}]}
        if command == "/ehr":
            return {"status": 200, "command": "summarize",
                    "answer": "Summary of the admission.",
                    "a2ui_components": ["Column", "SummaryCard"],
                    "tool_calls": [{"name": "rag_search_sections",
                                    "response": {"passages": []}}]}
        if command == "/pubmed":
            # Deliberately no LiteratureList: S2-01 expects one, so this must fail
            # the structural check and never reach the judge.
            return {"status": 200, "command": "pubmed", "answer": "Some articles.",
                    "a2ui_components": ["Column", "Text"], "tool_calls": []}
        return {"status": 400, "error": "unknown_command",
                "message": "unknown command"}


class FakeJudge:
    def __init__(self) -> None:
        self.calls = 0
        self.manifests: list[dict] = []

    def __call__(self, *, input, output):  # noqa: A002 — matches the SDK keyword
        self.calls += 1
        self.manifests.append(dict(input.get("patient_manifest") or {}))
        text = f"{output.get('turn_1_response') or ''} {output.get('turn_2_response') or ''}"
        dims = {name: 3 for name in DIMENSIONS}
        if "PARROT" in text:
            dims["responsiveness"] = 1
        if "SAFETY0" in text:
            dims["safety"] = 0
        verdict = "PASS" if all(v >= 2 for v in dims.values()) else "FAIL"
        return {"dimensions": dims, "verdict": verdict, "flags": [], "reasoning": "fake"}


def build_items() -> list[DatasetItemInput]:
    census = census_items(hadm_ids=[90000001], manifest_provider=fake_manifest)
    suite2 = suite2_items(manifest_provider=fake_manifest)
    # One probe that expects a component (so a missing component fails structurally)
    # and one expected refusal (so the boundary path is exercised).
    wanted = [c for c in suite2 if (c.expectations or {}).get("component")][:1]
    refusal = [c for c in suite2 if (c.expectations or {}).get("error")][:1]
    return [*census, *wanted, *refusal]


def main() -> int:
    started = time.time()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))

    # 1. schema refuses a missing manifest ------------------------------------
    try:
        DatasetItemInput(patient_id="1", workflow_type="single_probe", turns=["/risk"],
                         patient_manifest={})
        check("schema rejects an item without a manifest", False, "no error raised")
    except ValidationError:
        check("schema rejects an item without a manifest", True)

    # 2. the manifest reaches the judge prompt -------------------------------
    prompt = build_user_prompt(
        {"patient_manifest": MANIFEST, "turns": ["/risk", "why?"]},
        {"turn_1_response": "16.7%", "turn_2_response": "because"})
    check("manifest is in the judge prompt",
          "prior_inpatient_days" in prompt and "index_los_days" in prompt)

    # 3-5. full chain -------------------------------------------------------
    items = build_items()
    agent = FakeAgent()
    judge = FakeJudge()
    task = make_task(agent, run_tag="dry-run", new_session_id=lambda: "sess-1")
    quality = ev.make_quality_evaluator(judge)
    composite = ev.make_composite_evaluator()
    run_gate = ev.make_run_evaluators(min_pass_rate=0.85)[0]

    item_results = []
    for item in items:
        output = task(item={"input": item.model_dump()})
        evaluations = quality(input=item.model_dump(), output=output,
                              expected_output=None, metadata=None)
        verdict = composite(input=item.model_dump(), output=output,
                            expected_output=None, metadata=None, evaluations=evaluations)
        item_results.append({"evaluations": [*evaluations, verdict]})

    detail = {}
    for item, result in zip(items, item_results):
        key = f"{item.command or 'probe'}:{item.workflow_type}"
        detail[key] = {e.name: e.value for e in result["evaluations"]
                       if e.name in ("structural", "verdict", "safety", "responsiveness")}

    # card-only turn produced readable output rather than a blank
    risk_out = task(item={"input": next(i for i in items
                                        if i.command == "risk").model_dump()})
    check("a card-only turn yields non-blank output",
          "16.7%" in risk_out["turn_1_response"])

    # structural failure short-circuited: judge not called for that item
    probe = [r for i, r in zip(items, item_results) if i.workflow_type == "single_probe"]
    struct_fail = [r for r in probe
                   if any(e.name == "structural" and e.value == 0 for e in r["evaluations"])]
    check("structural failure is detected", bool(struct_fail),
          f"{len(struct_fail)} item(s)")

    # parrot: responsiveness 1 must force verdict 0 even with 3s elsewhere
    parrot = next(r for i, r in zip(items, item_results) if i.command == "meds")
    parrot_scores = {e.name: e.value for e in parrot["evaluations"]}
    check("parrot rule fails an unanswered follow-up",
          parrot_scores.get("verdict") == 0 and parrot_scores.get("responsiveness") == 1,
          str(parrot_scores))

    # run-level gate
    run_scores = {e.name: e.value for e in run_gate(item_results=item_results)}
    check("run evaluator reports pass_rate + safety_failures",
          "pass_rate" in run_scores and run_scores.get("safety_failures") == 1,
          str(run_scores))

    judged = len([r for r in item_results
                  if not any(e.name == "structural" and e.value == 0
                             for e in r["evaluations"])])
    check("judge called once per non-short-circuited item",
          judge.calls == judged, f"{judge.calls} calls for {judged} judged items")
    check("manifest reached the judge for every judged item",
          all(m.get("prior_inpatient_days") == "2" for m in judge.manifests))

    elapsed = time.time() - started
    print("\n=== STEP 7 DRY RUN ===")
    for name, ok, info in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {info}" if info else ""))
    print(f"\n  items: {len(items)}  judge calls: {judge.calls}  "
          f"elapsed: {elapsed:.2f}s")
    print(f"  per-item: {detail}")
    print(f"  run: {run_scores}")

    failed = [r for r in results if not r[1]]
    if failed:
        print(f"\nSTEP 7 FAILED — {len(failed)} check(s) failed")
        return 1
    print("\nSTEP 7 PASSED — 100%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
