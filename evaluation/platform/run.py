"""Step 8: run one dataset as a Langfuse experiment against the live agent.

    LANGFUSE_* exported, then:
        ./.venv/bin/python -m evaluation.platform.run --dataset clinical-census-pilot-v2
        ./.venv/bin/python -m evaluation.platform.run --dataset suite2-adversarial-v1

Every item is a real HTTP round-trip to the deployed agent and every item is scored by
the rubric judge, so cost and time scale with the dataset. `--limit` exists so a single
item can prove the wiring before thirty-four do.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (REPO, REPO / "evaluation" / "agent"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import httpx  # noqa: E402

from evaluation.platform import evaluators as ev  # noqa: E402
from evaluation.platform.judge_call import make_judge_fn  # noqa: E402
from evaluation.platform.task import make_task  # noqa: E402

DEFAULT_AGENT = "https://agent-jamycsjjzq-ue.a.run.app"


def _identity_token() -> str:
    return subprocess.run(
        ["gcloud", "auth", "print-identity-token"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def make_ask(agent_url: str, token: str, *, timeout: float = 240.0):
    """One `/ask` round-trip. Returns the envelope plus its HTTP status."""

    def ask(*, question, hadm_id, run_tag, session_id=None, turns=None,
            follow_ups=False, intent=None) -> dict:
        body: dict = {"question": question, "hadm_id": hadm_id, "run_tag": run_tag}
        if turns:
            body["turns"] = turns
        if follow_ups:
            # Only a loop-back item asks: `follow_ups` makes the service propose the
            # next questions, and every other turn has its own wording.
            body["follow_ups"] = True
        if intent:
            # What a rendered control declares about its own question. Only present when
            # the caller is reproducing a click: an evaluation turn that names nothing is
            # an ordinary turn, and the agent treats an absent key as exactly that.
            body["intent"] = intent
        last: dict = {}
        for attempt in range(3):
            try:
                response = httpx.post(f"{agent_url}/ask", json=body, timeout=timeout,
                                      headers={"Authorization": f"Bearer {token}"})
            except httpx.HTTPError as exc:
                last = {"status": 0, "error": type(exc).__name__}
            else:
                try:
                    payload = response.json()
                except json.JSONDecodeError:
                    payload = {"error": "non_json_response"}
                payload["status"] = response.status_code
                if response.status_code < 500:
                    return payload
                last = payload
            time.sleep(2 * (attempt + 1))
        return last

    return ask


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N items (use 1 to prove the wiring)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--agent-url", default=DEFAULT_AGENT)
    ap.add_argument("--concurrency", type=int, default=2,
                    help="judge-call concurrency; 2 is the ceiling the pro model's "
                         "regional quota sustains (5 produced 429s on 2026-10-21)")
    ap.add_argument("--rubric-version", default="v4")
    ap.add_argument("--min-pass-rate", type=float, default=0.85)
    args = ap.parse_args(argv)

    from langfuse import Langfuse  # noqa: PLC0415

    client = Langfuse()
    dataset = client.get_dataset(args.dataset)
    items = list(getattr(dataset, "items", []) or [])
    if args.limit:
        items = items[: args.limit]
    if not items:
        print(f"{args.dataset}: no items", flush=True)
        return 2
    print(f"{args.dataset}: {len(items)} item(s), agent {args.agent_url}", flush=True)

    token = _identity_token()
    ask = make_ask(args.agent_url, token)
    run_tag = args.run_name or f"{args.dataset}-{time.strftime('%Y%m%d-%H%M%S')}"

    task = make_task(ask, run_tag=run_tag,
                     new_session_id=lambda: f"{run_tag}-{time.time_ns()}")

    result = client.run_experiment(
        name=f"{args.dataset}",
        run_name=run_tag,
        description=f"platform-owned evaluation, rubric {args.rubric_version}",
        data=items,
        task=task,
        evaluators=[ev.make_quality_evaluator(make_judge_fn())],
        composite_evaluator=ev.make_composite_evaluator(),
        run_evaluators=ev.make_run_evaluators(min_pass_rate=args.min_pass_rate),
        max_concurrency=args.concurrency,
        metadata={"rubric_version": args.rubric_version, "agent_url": args.agent_url},
    )

    print(f"run complete: {run_tag}", flush=True)
    print(json.dumps(getattr(result, "run_evaluations", None) or {}, default=str,
                     indent=2)[:1200], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
