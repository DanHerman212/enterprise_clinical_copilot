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
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.platform.evaluators import gate_verdict
from evaluation.platform.schemas import MIN_PASS_RATE

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (REPO,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

#: Hardcoded targets (spec §5). Deliberately not a parameter.
DATASETS = ("clinical-census-pilot-v2", "suite2-adversarial-v1")

#: The Tier 2 dev suite (Phase 04). A gated target rather than a routine one: its
#: canary is named, and the full 26 run needs the Architect's sign-off, so it is
#: reached by `--datasets ecc-tier2-dev-suite` and not by the default.
DEV_SUITE_DATASET = "ecc-tier2-dev-suite"

#: The Tier 2 blind holdout suite (Phase 05). Same gate, opposite cohort: 15 scenarios
#: over four records the dev suite never names (`dev_suite.HOLDOUT_HADM_IDS`). Seeded by
#: `evaluation/platform/sync_holdout_suite.py`; reached the same way, by naming it.
HOLDOUT_SUITE_DATASET = "ecc-tier2-holdout-suite"

#: Gate 1's three canary scenarios, and nothing else (Phase 04 §4.2).
CANARY_SCENARIOS = ("dev_t1_01", "dev_t2_03", "dev_t7_01")

#: The Step 9 census: 82 patients x 4 prompts = 328 items, ~2.9 h at concurrency 2.
#:
#: Kept OUT of `DATASETS` on purpose. The pilot set is what a routine check runs, and
#: the census is a deliberate, announced step with its own gate — naming it must be an
#: act, not a typo away from `--datasets`. `--allow-full-census` is that act.
FULL_CENSUS_DATASET = "clinical-census-full-v1"
#: Two, not five. `gemini-2.5-pro` on us-east1 has a markedly tighter per-minute quota
#: than the flash models, and five concurrent items meant five concurrent judge calls in
#: the same window: the 2026-10-21 suite-2 run logged "judge retry 1/6 after ClientError:
#: 429 RESOURCE_EXHAUSTED" and then lost five items to it. Judge concurrency is the knob
#: that matters here — the agent round-trips are unaffected by lowering it.
MAX_CONCURRENCY = int(os.environ.get("EVAL_MAX_CONCURRENCY", "2"))
#: `MIN_PASS_RATE` is `schemas`', and `gate_verdict` is `evaluators`' (`00d8` Rows 3, 4).
#:
#: A local copy of the five-term verdict lived in this module after `structural_failures`
#: was found missing from an inline version — on the 2026-10-28 retest (`00d2`) an item
#: failed `must_contain` on a fact its answer never carried and the run printed
#: `structural_failures: 1` beside `gate_pass: True`. Keeping a second copy here while the
#: Langfuse score used a third rule reproduced the same defect one layer up.

#: The judge ceiling for the end-of-run retry pass, longer than the in-run default.
#:
#: A retry runs alone, so it can afford to wait: the item it is rescuing was already
#: given two 150s attempts inside a concurrent batch, where the call competes for quota
#: and for the model's own queue. If it does not answer in four minutes on its own, the
#: item is genuinely unjudgeable and the run says so.
RETRY_JUDGE_TIMEOUT_SECONDS = int(os.environ.get("EVAL_RETRY_JUDGE_TIMEOUT", "240"))
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


def _item_scenario_id(item: Any) -> str:
    """The scenario id on an item, from its input or its dataset id."""
    data = item.get("input") if isinstance(item, dict) else getattr(item, "input", None)
    if isinstance(data, dict) and data.get("scenario_id"):
        return str(data["scenario_id"])
    return str(item.get("id") if isinstance(item, dict) else getattr(item, "id", ""))


def _select_scenarios(items: list[Any], wanted: tuple[str, ...]) -> list[Any]:
    """The named scenarios, in the order the caller named them.

    Refuses an unknown id rather than quietly running fewer items: a canary that runs
    two of its three scenarios and reports green is the failure this gate exists to
    prevent.
    """
    by_id = {_item_scenario_id(item): item for item in items}
    missing = [name for name in wanted if name not in by_id]
    if missing:
        raise RuntimeError(
            f"scenario(s) not in the dataset: {missing}; have {sorted(by_id)}")
    return [by_id[name] for name in wanted]


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


def served_revision(api_url: str, *, timeout: float = 30.0,
                    token: str | None = None) -> str | None:
    """The `code_revision` the deployed service reports, or None if unreadable.

    The eval never runs this checkout: it POSTs to `api_url`, so the prompts under
    test are whichever revision is deployed. `services/agent/chain.py` resolves that
    identity from `CODE_REVISION` and then Cloud Run's `K_REVISION`, and `/health`
    reports it — which is the only way to tell "the fix is committed" from "the fix
    is serving".

    `/health` sits behind the same Cloud Run IAM as `/ask` (verified: an anonymous
    GET is a 403 from the frontend, not the app), so the probe carries an identity
    token audienced to the service, exactly as `http_ask` does.

    The timeout is generous because the service scales to zero and this probe is
    what wakes it; the first eval POST would pay that cold start anyway.
    """
    base = api_url.rstrip("/")
    url = f"{base}/health"
    import httpx  # noqa: PLC0415 — the same stack `http_ask` uses for /ask

    try:
        if token is None:
            from evaluation.platform.http_auth import identity_token  # noqa: PLC0415

            token = identity_token(base)
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, headers={"Authorization": f"Bearer {token}"})
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, OSError, ValueError, RuntimeError) as exc:
        print(f"WARNING: {url} unreadable ({exc}); this run's code_revision is unknown",
              flush=True)
        return None
    revision = payload.get("code_revision") if isinstance(payload, dict) else None
    revision = (revision or "").strip()
    return revision or None


def rubric_version() -> str:
    """The rubric the run is graded under, read from the rubric itself.

    It was hardcoded to `v4` while the rubric file was at v5, so three runs claimed a
    version that had already been superseded and could not be compared with anything.
    """
    text = (REPO / "evaluation" / "rubric.md").read_text()
    first = text.splitlines()[0] if text else ""
    match = re.search(r"Rubric (v\d+)", first)
    return match.group(1) if match else "unknown"


def run(datasets: tuple[str, ...], *, run_name: str,
        code_revision: str | None = None,
        scenarios: tuple[str, ...] = ()) -> dict[str, Any]:
    global _run_tag
    _run_tag = run_name

    from langfuse import Langfuse

    from evaluation.platform import evaluators as ev
    from evaluation.platform import diagnostics
    from evaluation.platform.experiment import (
        record_run_scores,
        retry_unscored,
        run_linked,
    )
    from evaluation.platform.judge_call import make_judge_fn

    client = Langfuse()
    assertions = ev.make_assertions_evaluator()
    composite = ev.make_judge_composite(make_judge_fn())
    run_evaluators = ev.make_run_evaluators(min_pass_rate=MIN_PASS_RATE)

    summary: dict[str, Any] = {}
    rubric = rubric_version()
    for dataset_name in datasets:
        dataset = client.get_dataset(dataset_name)
        items = list(getattr(dataset, "items", []) or [])
        if not items:
            raise RuntimeError(f"{dataset_name}: 0 items — seed it first")
        if scenarios:
            items = _select_scenarios(items, scenarios)
        print(f"\n=== {dataset_name}: {len(items)} item(s) as {run_name} ===",
              flush=True)

        # One local JSONL per dataset run: every turn's question, answer, sources and
        # tool payloads, every dimension, and the judge's FULL reasoning. The stored
        # score comment is capped at 500 characters and the payloads of a free-text
        # turn are not exported as spans, so without this a post-mortem cannot answer
        # "which passage did `^[2]` point at" (`00e`, 2026-10-28).
        diagnostics_path = diagnostics.configure(
            diagnostics.DEFAULT_DIR / f"{run_name}-{dataset_name}.jsonl")
        if diagnostics_path is not None:
            print(f"  diagnostics: {diagnostics_path}", flush=True)

        # The scores go onto the trace the SERVICE opened for this answer, not onto a
        # wrapper trace of the harness's own. That is what makes the run tag usable as
        # a filter in the Scores view, and what puts a score beside the prompts and
        # tool calls that produced it (`experiment.run_linked`).
        experiment = run_linked(
            client,
            dataset_name=dataset_name,
            run_name=run_name,
            items=items,
            task=execute_agent_thread,
            evaluators=[assertions],
            composite=composite,
            max_concurrency=MAX_CONCURRENCY,
            description=f"Phase 04 Tier 2 dev suite, rubric {rubric}",
            metadata={"rubric_version": rubric, "run_tag": run_name,
                      "agent_url": os.environ.get("API_URL", ""),
                      "code_revision": code_revision or "unknown"},
        )

        stats = experiment.stats
        if stats["unscored"]:
            # One quiet second attempt, at concurrency 1 and with a longer ceiling. A
            # judge call that did not answer is the harness failing to measure, and a
            # single item left unscored blocks the gate — so it is worth half a minute
            # to recover it rather than rerunning a multi-hour census.
            print(f"  retry pass: {stats['unscored']} unscored item(s)", flush=True)
            previous = os.environ.get("JUDGE_TIMEOUT_SECONDS")
            os.environ["JUDGE_TIMEOUT_SECONDS"] = str(RETRY_JUDGE_TIMEOUT_SECONDS)
            try:
                stats = retry_unscored(
                    client, items=items, outcomes=experiment.item_results,
                    evaluators=[assertions], composite=composite,
                )
            finally:
                if previous is None:
                    os.environ.pop("JUDGE_TIMEOUT_SECONDS", None)
                else:
                    os.environ["JUDGE_TIMEOUT_SECONDS"] = previous

        record_run_scores(
            client, dataset_run_id=experiment.dataset_run_id,
            outcomes=experiment.item_results, run_evaluators=run_evaluators,
        )

        _print_item_report(experiment.item_results, dataset_name=dataset_name,
                           run_name=run_name,
                           dataset_run_id=experiment.dataset_run_id)

        summary[dataset_name] = {
            "items": len(items), **stats,
            "dataset_run_id": experiment.dataset_run_id,
            "gate_pass": gate_verdict(stats),
        }
        print(f"  {summary[dataset_name]}", flush=True)

    client.flush()
    return summary


def _print_item_report(outcomes: list[Any], *, dataset_name: str, run_name: str,
                       dataset_run_id: str | None) -> None:
    """One block per item: what was asked, what came back, how it scored.

    The run's aggregate cannot answer the questions a gate asks — which chip was
    resolved, whether an outpatient refusal really carried `verified=False`, which
    dimension lost the item — so each item is reported here rather than re-queried
    from Langfuse afterwards.
    """
    from evaluation.platform import evaluators as ev

    dimensions = ev.DIMENSIONS
    print(f"\n--- per-item report: {dataset_name} / {run_name} "
          f"(run {dataset_run_id or 'unattributed'}) ---", flush=True)
    for outcome in outcomes:
        output = outcome.output or {}
        scores = {e.name: e for e in outcome.evaluations}
        structural = scores.get(ev.STRUCTURAL)
        verdict = scores.get(ev.VERDICT)

        print(f"\n  {outcome.item_id}  "
              f"verdict={'none' if verdict is None else f'{verdict.value:g}'}",
              flush=True)
        prompts = output.get("turn_prompts") or []
        for index, prompt in enumerate(prompts, start=1):
            print(f"    turn {index} asked: {prompt}", flush=True)
        offered = next((turn for turn in output.get("turn_follow_ups") or [] if turn),
                       [])
        if offered:
            print(f"    turn 1 proposed: {offered}", flush=True)
        print("    verified per turn: "
              f"{output.get('turn_verified') or []}", flush=True)
        print("    components per turn: "
              f"{output.get('turn_components') or []}", flush=True)
        if structural is not None:
            print(f"    structural: {structural.value:g} — "
                  f"{structural.comment or 'ok'}", flush=True)
        dims = {name: scores[name].value for name in dimensions if name in scores}
        if dims:
            print(f"    dimensions: {dims}", flush=True)
        if verdict is not None and verdict.comment:
            print(f"    verdict note: {verdict.comment}", flush=True)
        if outcome.error:
            print(f"    ERROR: {outcome.error}", flush=True)
        print(f"    traces: {output.get('turn_trace_ids') or [outcome.trace_id]}",
              flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=None,
                    help=("subset of the allowed datasets; default is both. The full "
                          "census needs --allow-full-census as well."))
    ap.add_argument("--allow-full-census", action="store_true",
                    help=(f"permit {FULL_CENSUS_DATASET} (Step 9, ~2.9h). Without "
                          "this flag the census is refused by name."))
    ap.add_argument("--dry-payloads", action="store_true",
                    help="validate request bodies against the service contract, send nothing")
    ap.add_argument("--expect-revision", default=None,
                    help=("refuse to run unless the deployed service reports this "
                          "code_revision; use the commit you built the image from"))
    ap.add_argument("--canary", action="store_true",
                    help=(f"run only {list(CANARY_SCENARIOS)} (Gate 1, Phase 04 §4.2)"))
    ap.add_argument("--scenarios", default="",
                    help="comma-separated scenario ids to run from the dev suite; "
                         "an unknown id is refused")
    args = ap.parse_args(argv)

    targets = tuple(args.datasets) if args.datasets else DATASETS
    allowed = (*DATASETS, DEV_SUITE_DATASET, HOLDOUT_SUITE_DATASET)
    if args.allow_full_census:
        allowed = (*allowed, FULL_CENSUS_DATASET)
    refused = [name for name in targets if name not in allowed]
    if refused:
        print(f"FATAL: refusing {refused}; allowed targets are {list(allowed)}",
              flush=True)
        return 2

    scenarios: tuple[str, ...] = ()
    if args.canary:
        if DEV_SUITE_DATASET not in targets:
            print(f"FATAL: --canary runs the dev suite; add "
                  f"--datasets {DEV_SUITE_DATASET}", flush=True)
            return 2
        scenarios = CANARY_SCENARIOS
    elif args.scenarios.strip():
        scenarios = tuple(name.strip() for name in args.scenarios.split(",")
                          if name.strip())

    try:
        env = verify_environment()
    except EnvironmentError as exc:
        print(f"FATAL: {exc}", flush=True)
        return 2

    if args.dry_payloads:
        return dry_payloads()

    code_revision = served_revision(env["API_URL"])
    if args.expect_revision and code_revision != args.expect_revision:
        print(f"FATAL: service reports code_revision {code_revision or 'unknown'}, "
              f"not {args.expect_revision}; the image was not built from that commit "
              "(or the deploy has not landed). Refusing to attribute a run to it.",
              flush=True)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = (f"phase04-gate1-canary-{stamp}" if scenarios
                and tuple(scenarios) == CANARY_SCENARIOS else f"pilot-verify-{stamp}")
    print(f"agent:   {env['API_URL']}\n"
          f"serving: {code_revision or 'unknown'}\n"
          f"run:     {run_name}\ndatasets: {list(targets)}")
    if scenarios:
        print(f"scenarios: {list(scenarios)}")

    summary = run(targets, run_name=run_name, code_revision=code_revision,
                  scenarios=scenarios)

    print("\n=== STEP 8 SUMMARY ===")
    for name, stats in summary.items():
        print(f"  {name}: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
