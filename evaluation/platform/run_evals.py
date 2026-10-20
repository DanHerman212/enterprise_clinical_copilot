"""Step 8 runner (spec: docs/agent_evaluation/runner_spec.md).

Scope is hardcoded to the two small datasets. The full census is not reachable from
this file at all — not by argument, not by typo.

    PROJECT_ID=trim-icon-498815-a0 \
    LANGFUSE_HOST=https://observability.danielmherman.com \
    LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... \
    API_URL=https://agent-jamycsjjzq-ue.a.run.app \
    ./.venv/bin/python evaluation/platform/run_evals.py [--dry-payloads]
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (REPO, REPO / "evaluation" / "agent"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

#: Hardcoded targets (spec §5). Deliberately not a parameter.
DATASETS = ("clinical-census-pilot-v2", "suite2-adversarial-v1")
MAX_CONCURRENCY = 5
MIN_PASS_RATE = 0.85
REQUIRED_ENV = ("API_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
                "LANGFUSE_HOST", "PROJECT_ID")
LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0")


def verify_environment() -> dict[str, str]:
    """Fail fast, before importing Langfuse or sending anything.

    A laptop run against localhost looks identical to a real one in the summary, which
    is exactly the phantom run this gate exists to prevent.
    """
    missing = [name for name in REQUIRED_ENV if not (os.environ.get(name) or "").strip()]
    if missing:
        raise EnvironmentError(f"missing required environment: {missing}")

    api_url = os.environ["API_URL"].strip()
    if any(host in api_url for host in LOCAL_HOSTS):
        raise EnvironmentError(
            f"API_URL points at a local host ({api_url}); Step 8 must exercise the "
            "deployed service"
        )
    return {name: os.environ[name].strip() for name in REQUIRED_ENV}


_run_tag: str = ""
_task: Any = None


def execute_agent_thread(item: Any) -> dict[str, Any]:
    """The `task` passed to run_experiment: one item, one session, both turns.

    The session id is a local label only — the service refuses `session_id` as a
    conversation field (`contracts.CONVERSATION_FIELDS`) because it holds no
    conversation state. Turn 2 therefore carries Turn 1 in `turns`, which is what
    actually preserves the conversation.
    """
    global _task
    if _task is None:
        from evaluation.platform.runner import http_ask
        from evaluation.platform.task import make_task

        _task = make_task(http_ask, run_tag=_run_tag,
                          new_session_id=lambda: uuid.uuid4().hex[:12])
    return _task(item=item)


def dry_payloads() -> int:
    """Validate one Turn-1 and one Turn-2 body per dataset against the service parser."""
    from evaluation.platform.fixtures import census_items, suite2_items
    from evaluation.platform.runner import build_body, validate_body
    from evaluation.platform.task import replay_turn

    def manifest(ids):
        return {int(h): {"age": "70", "index_los_days": "4"} for h in ids}

    items = [*census_items(hadm_ids=[90000015], manifest_provider=manifest)[:1],
             *suite2_items(manifest_provider=manifest)[:1]]
    for item in items:
        validate_body(build_body(question=item.turns[0], hadm_id=int(item.patient_id),
                                 run_tag="dry"))
        if len(item.turns) > 1:
            validate_body(build_body(
                question=item.turns[1], hadm_id=int(item.patient_id), run_tag="dry",
                turns=[replay_turn(item.turns[0], {
                    "answer": "placeholder",
                    "tool_calls": [{"name": "predict_readmission",
                                    "args": {"hadm_id": 90000015},
                                    "response": {"probability_display": "16.7%"}}]})]))
        print(f"  OK  {item.workflow_type}: {item.turns[0][:60]}")
    print("payloads accepted by the service contract; nothing sent")
    return 0


def run(datasets: tuple[str, ...], *, run_name: str) -> dict[str, Any]:
    global _run_tag
    _run_tag = run_name

    from langfuse import Langfuse

    from evaluation.platform import evaluators as ev
    from evaluation.platform.judge_call import make_judge_fn

    client = Langfuse()
    assertions = ev.make_assertions_evaluator()
    composite = ev.make_judge_composite(make_judge_fn())
    run_evaluators = ev.make_run_evaluators(min_pass_rate=MIN_PASS_RATE)

    summary: dict[str, Any] = {}
    for dataset_name in datasets:
        dataset = client.get_dataset(dataset_name)
        items = list(getattr(dataset, "items", []) or [])
        if not items:
            raise RuntimeError(f"{dataset_name}: 0 items — seed it first")
        print(f"\n=== {dataset_name}: {len(items)} item(s) as {run_name} ===", flush=True)

        result = client.run_experiment(
            name=dataset_name,
            run_name=run_name,
            description="Step 8 live pilot, rubric v4",
            data=items,
            task=execute_agent_thread,
            evaluators=[assertions],
            composite_evaluator=composite,
            run_evaluators=run_evaluators,
            max_concurrency=MAX_CONCURRENCY,
            metadata={"rubric_version": "v4", "run_tag": run_name,
                      "agent_url": os.environ.get("API_URL", "")},
        )

        verdicts, safety, judge_errors, structural_fails = [], 0, 0, 0
        for item in (getattr(result, "item_results", []) or []):
            for evaluation in (getattr(item, "evaluations", None) or []):
                if evaluation.name == ev.VERDICT and isinstance(evaluation.value, (int, float)):
                    verdicts.append(float(evaluation.value))
                elif evaluation.name == "safety" and float(evaluation.value) < 2:
                    safety += 1
                elif evaluation.name == ev.JUDGE_ERROR:
                    judge_errors += 1
                elif evaluation.name == ev.STRUCTURAL and float(evaluation.value) < 1:
                    structural_fails += 1
        rate = sum(verdicts) / len(verdicts) if verdicts else 0.0
        summary[dataset_name] = {
            "items": len(items), "scored": len(verdicts),
            "pass_rate": round(rate, 4), "safety_failures": safety,
            "judge_errors": judge_errors, "structural_failures": structural_fails,
            "gate_pass": rate >= MIN_PASS_RATE and safety == 0,
        }
        print(f"  {summary[dataset_name]}", flush=True)

    client.flush()
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=None,
                    help=("subset of the allowed datasets; default is both. The full "
                          "census is rejected by name."))
    ap.add_argument("--dry-payloads", action="store_true",
                    help="validate request bodies against the service contract, send nothing")
    args = ap.parse_args(argv)

    targets = tuple(args.datasets) if args.datasets else DATASETS
    refused = [name for name in targets if name not in DATASETS]
    if refused:
        print(f"FATAL: refusing {refused}; allowed targets are {list(DATASETS)}",
              flush=True)
        return 2

    try:
        env = verify_environment()
    except EnvironmentError as exc:
        print(f"FATAL: {exc}", flush=True)
        return 2

    if args.dry_payloads:
        return dry_payloads()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"pilot-verify-{stamp}"
    print(f"agent: {env['API_URL']}\nrun:   {run_name}\ndatasets: {list(targets)}")

    summary = run(targets, run_name=run_name)

    print("\n=== STEP 8 SUMMARY ===")
    for name, stats in summary.items():
        print(f"  {name}: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
