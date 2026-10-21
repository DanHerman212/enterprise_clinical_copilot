"""The live runner: dataset -> live agent -> evaluators -> Langfuse run.

Scope is enforced, not remembered. `ALLOWED_DATASETS` holds the two small datasets;
asking for anything else — above all `clinical-census-full-v1` — is refused before a
single request leaves the machine. The full census is a separate, deliberate step.

Usage (from the repo root):

    PROJECT_ID=trim-icon-498815-a0 \
    LANGFUSE_HOST=https://observability.danielmherman.com \
    LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... \
    ./.venv/bin/python -m evaluation.platform.runner --run-tag pilot-$(date +%H%M) \
        clinical-census-pilot-v2 suite2-adversarial-v1

`--dry-payloads` validates every request body against the service's own parser without
sending anything, which is the cheap check to run before spending money.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
AGENT_DIR = REPO / "evaluation" / "agent"
for _path in (REPO, AGENT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

AGENT_URL = "https://agent-jamycsjjzq-ue.a.run.app"

#: The only datasets this runner may execute. Adding a name here is a deliberate act;
#: the full census is not on the list on purpose.
ALLOWED_DATASETS = ("clinical-census-pilot-v2", "suite2-adversarial-v1")


class ScopeError(RuntimeError):
    """Raised when a caller asks the runner for a dataset outside its scope."""


def check_scope(names: list[str]) -> list[str]:
    refused = [n for n in names if n not in ALLOWED_DATASETS]
    if refused:
        raise ScopeError(
            f"refusing to run {refused}: this runner is scoped to {list(ALLOWED_DATASETS)}"
        )
    return names


def build_body(*, question: str, hadm_id: int, run_tag: str,
               turns: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The request body, exactly as the service accepts it.

    `session_id` is deliberately absent: the service refuses it as a conversation
    field, because it holds no conversation state — the caller owns the transcript and
    replays it in `turns`.
    """
    body: dict[str, Any] = {"question": question, "hadm_id": int(hadm_id),
                            "run_tag": run_tag}
    if turns:
        body["turns"] = turns
    return body


def validate_body(body: dict[str, Any]) -> None:
    """Refuse a body the service would refuse, before it costs a request."""
    from services.agent.contracts import parse_agent_request

    parse_agent_request(body, max_question_chars=4000)


def http_ask(*, question: str, hadm_id: int, run_tag: str,
             turns: list[dict[str, Any]] | None = None,
             session_id: str | None = None) -> dict[str, Any]:
    """One `/ask` call. `session_id` is accepted for logging only and not sent."""
    import httpx

    from evaluation.platform.http_auth import identity_token  # noqa: PLC0415

    body = build_body(question=question, hadm_id=hadm_id, run_tag=run_tag, turns=turns)
    validate_body(body)
    headers = {"Authorization": f"Bearer {identity_token(AGENT_URL)}"}
    with httpx.Client(timeout=180.0) as client:
        response = client.post(f"{AGENT_URL}/ask", json=body, headers=headers)
    try:
        payload = response.json()
    except ValueError:
        payload = {"error": f"non-JSON response ({response.status_code})"}
    payload.setdefault("status", response.status_code)
    return payload


def run(names: list[str], *, run_tag: str, max_concurrency: int = 5) -> dict[str, Any]:
    check_scope(names)

    from langfuse import Langfuse  # noqa: PLC0415

    from evaluation.platform import evaluators as ev  # noqa: PLC0415
    from evaluation.platform.experiment import record_run_scores, run_linked  # noqa: PLC0415
    from evaluation.platform.judge_call import make_judge_fn, RUBRIC  # noqa: PLC0415
    from evaluation.platform.task import make_task  # noqa: PLC0415

    client = Langfuse()
    task = make_task(http_ask, run_tag=run_tag,
                     new_session_id=lambda: uuid.uuid4().hex[:12])
    quality = ev.make_quality_evaluator(make_judge_fn())
    composite = ev.make_composite_evaluator()
    run_evaluators = ev.make_run_evaluators(min_pass_rate=0.85)

    summary: dict[str, Any] = {}
    for name in names:
        dataset = client.get_dataset(name)
        items = list(getattr(dataset, "items", []) or [])
        if not items:
            raise RuntimeError(f"{name}: no items — seed it before running")
        print(f"\n=== {name}: {len(items)} item(s), tag {run_tag} ===", flush=True)

        experiment = run_linked(
            client,
            dataset_name=name,
            run_name=run_tag,
            items=items,
            task=task,
            evaluators=[quality],
            composite=composite,
            max_concurrency=max_concurrency,
            description=f"live pilot, rubric {RUBRIC[:0] or 'v4'}",
            metadata={"rubric_version": "v4", "run_tag": run_tag, "agent_url": AGENT_URL},
        )
        record_run_scores(
            client, dataset_run_id=experiment.dataset_run_id,
            outcomes=experiment.item_results, run_evaluators=run_evaluators,
        )
        item_results = experiment.item_results
        verdicts, safety, errors = [], 0, 0
        for item in item_results:
            for evaluation in (getattr(item, "evaluations", None) or []):
                if evaluation.name == ev.VERDICT and isinstance(evaluation.value, (int, float)):
                    verdicts.append(float(evaluation.value))
                elif evaluation.name == "safety" and float(evaluation.value) < 2:
                    safety += 1
                elif evaluation.name == ev.JUDGE_ERROR:
                    errors += 1
        rate = sum(verdicts) / len(verdicts) if verdicts else 0.0
        summary[name] = {"items": len(items), "verdicts": len(verdicts),
                         "pass_rate": round(rate, 4), "safety_failures": safety,
                         "judge_errors": errors}
        print(f"  {summary[name]}", flush=True)

    client.flush()
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--max-concurrency", type=int, default=5)
    ap.add_argument("--dry-payloads", action="store_true",
                    help="validate one request body per dataset and send nothing")
    args = ap.parse_args(argv)

    names = check_scope(args.datasets)

    if args.dry_payloads:
        from evaluation.platform.fixtures import census_items, suite2_items
        from evaluation.platform.task import replay_turn

        examples = (census_items(hadm_ids=[90000015], manifest_provider=lambda ids: {})
                    if "clinical-census-pilot-v2" in names else [])
        probes = (suite2_items(manifest_provider=lambda ids: {})[:1]
                  if "suite2-adversarial-v1" in names else [])
        for item in [*examples[:1], *probes]:
            body = build_body(question=item.turns[0], hadm_id=int(item.patient_id),
                              run_tag=args.run_tag)
            validate_body(body)
            if len(item.turns) > 1:
                second = build_body(
                    question=item.turns[1], hadm_id=int(item.patient_id),
                    run_tag=args.run_tag,
                    turns=[replay_turn(item.turns[0], {
                        "answer": "placeholder", "tool_calls": [
                            {"name": "predict_readmission", "args": {"hadm_id": 1},
                             "response": {"probability": 0.34}}]})])
                validate_body(second)
            print(f"  OK  {item.workflow_type}: {item.turns[0][:50]}")
        print("\nall request bodies accepted by the service contract; nothing sent")
        return 0

    if not os.environ.get("PROJECT_ID"):
        print("PROJECT_ID is required (the judge resolves its project from it)",
              flush=True)
        return 2
    summary = run(names, run_tag=args.run_tag, max_concurrency=args.max_concurrency)
    print("\n=== RUN SUMMARY ===")
    for name, stats in summary.items():
        print(f"  {name}: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
