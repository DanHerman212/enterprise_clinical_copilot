# 02 — Harness Architecture & Deterministic Assertions

## 1. Executive Summary & Problem Statement
During our initial evaluation runs, the harness created immense friction because it was coupled to container image builds and suffered from silent artifact routing gaps. Specifically:
1. **Case Files Baked Into Docker Images (`suite2_cases.json`, `cases_82.json`):** Minor adjustments to test assertions required a complete 15-minute Cloud Build / Cloud Run container cycle.
* **Suite 2 Modality Silence (`entrypoint.py`):** `collect_http.py` wrote Suite 2 cases to `suite2_{tag}.jsonl` and `_structural_gate` asserted their envelope rules, but `entrypoint.py` never passed `suite2` records to `judge.py`. Consequently, the 30 targeted modality probes (including the 4 boundary/abstention probes `S2-16`–`S2-19`, which expect `400 unknown_command`) were evaluated for structural format but completely ignored by the LLM-as-a-Judge.
3. **Report Aggregation Float Bug (`judge.py`):** Multi-sample consensus median scores computed float values (e.g., `2.0` or `1.5`), but the report aggregator checked `if isinstance(v, int):`, evaluating to `False` and silently dropping per-dimension pass rates (`agg[d]["total"] = 0`) in the final JSON report.

This module establishes a clean, decoupled harness architecture, repairs the orchestration pass-throughs, and decouples static case data so iterative testing happens instantly.

---

## 2. Decoupling Test Fixtures from Docker Builds

To eliminate container rebuild churn when editing test suites, we decouple static JSON assets from the runtime image:
* **Dynamic Fixture Loading:** `collect_http.py` and `entrypoint.py` read static test suites (`suite2_cases.json`, `adversarial_cases.json`, `cases_82.json`) directly from the local file system (or mount them via GCS when running on Cloud Run Jobs).
* **Local Rehearsal Loop:** Engineers can execute `collect_http.py` and `judge.py` directly from their local environment against the live `$AGENT_URL` with zero container builds.

---

## 3. Code Specifications: Harness & Judge Fixes

### 3.1 Fix 1: Route Suite 2 Modality Cases to the Judge (`evaluation/agent/entrypoint.py`)
Update `entrypoint.py` so that when `SUITE2=1` is passed to the clinical job, the collected Suite 2 file is merged or routed alongside the clinical census into the judge invocation:

```python
# In evaluation/agent/entrypoint.py -> _judge_argv() or main():
def _judge_argv(phase: str, tag: str, paths: dict, workers: int) -> list[str]:
    # Combine clinical traces and Suite 2 modality traces for unified judging
    traces_to_judge = str(paths["traces"])
    if paths.get("suite2") and paths["suite2"].exists():
        combined_traces = RESULTS / f"traces_combined_{tag}.jsonl"
        with combined_traces.open("w") as out_fh:
            for p in [paths["traces"], paths["suite2"]]:
                if p and p.exists():
                    out_fh.write(p.read_text())
        traces_to_judge = str(combined_traces)

    return [
        sys.executable, "judge.py",
        "--mode", phase,
        "--in", traces_to_judge,
        "--out", str(paths["judged"]),
        "--report", str(paths["report"]),
        "--workers", str(workers),
        "--run-tag", tag,
        "--upload-gcs",
    ]
```

### 3.2 Fix 2: Repair Float Aggregation in Report Aggregator (`evaluation/agent/judge.py`)

In evaluation/agent/judge.py, update the dimension aggregation loop to accept both int and float scores produced by median consensus:

# In evaluation/agent/judge.py -> main() aggregation loop:

```python
        scored = [json.loads(l) for l in judged_path.read_text().splitlines() if l.strip()]
        agg = {d: {"pass": 0, "fail": 0, "total": 0} for d in DIMS}
        verdict = {"pass": 0, "fail": 0, "agent_error": 0, "judge_error": 0}
        flags: list[dict] = []
        for rec in scored:
            j = rec.get("judge", {})
            if "error" in rec or j.get("error"):
                err = j.get("error") or "agent run failed"
                if err == "agent run failed" or err.startswith("no answer"):
                    verdict["agent_error"] += 1
                else:
                    verdict["judge_error"] += 1
                continue
            dims = j.get("dimensions", {})
            for d in DIMS:
                v = dims.get(d)
                # FIX: Accept float (from consensus median) as well as int
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    agg[d]["total"] += 1
                    if float(v) >= 2.0:
                        agg[d]["pass"] += 1
                    else:
                        agg[d]["fail"] += 1
            v = j.get("verdict")
            if v == "PASS":
                verdict["pass"] += 1
            elif v == "FAIL":
                verdict["fail"] += 1
            for f in (j.get("flags") or []):
                flags.append({"hadm_id": rec["hadm_id"], "prompt": rec["prompt"], "flag": f})
```

### 4. Operational Directives & Execution Directives for Phase 02
* **Adopt Local Rehearsal First**: Never build a harness container image just to test a prompt or fixture change. Run collect_http.py and judge.py locally against $AGENT_URL.

* **Apply Harness Patches**: Integrate the Suite 2 merge logic into entrypoint.py and the float type-check fix into judge.py.

* **Verify Contract**: Run pytest tests/agent/test_eval_contract.py to confirm all assertions pass cleanly before proceeding to Phase 03.

---

## 5. As Built (2026-10-19)

Implemented locally-first and tested in-process before any image build. Where the shipped code differs from the specification above, this section is authoritative.

### 5.1 Deviations from the spec

| Spec (§3.1) | As built | Why |
| :--- | :--- | :--- |
| `out_fh.write(p.read_text())` — raw text concatenation | `_merge_for_judging()` reads rows with `json.loads` and re-emits with `json.dumps` | Two JSONL files joined as bytes produce one corrupt line when the first lacks a trailing newline. `judge.py` parses its input with a bare `json.loads` per line, so that corruption is a crash, not a skipped row. |
| Merge everything | Expected refusals excluded, **from suite 2 only** | `judge.py` records a refusal as `no answer: <err>`, which the report counts as an **agent error** — so judging `S2-16`–`S2-19` would report four *correct* behaviours as four agent failures. `assertions.py` already checks `expect.status`/`expect.error` exactly. Scoping the filter to suite 2 matters: applied to adversarial probes it would quietly shrink a 20-probe run. |
| `isinstance(v, int)` → `isinstance(v, (int, float))` | Same, via `_is_score()` which also excludes `bool` | `isinstance(True, int)` is `True`, so a boolean would otherwise count as a score of 1. |
| One pooled report | Census and suite 2 aggregated separately by `_aggregate()` | Directive: the gate denominator must stay pure. Top-level `verdict`/`dimensions` are census only; suite 2 appears under `by_suite`. |

The unparsable-line case is announced rather than swallowed (`_jsonl_rows`), because this file feeds the judge: a silently skipped line is a silently smaller denominator.

### 5.2 Additions the spec did not cover

- **`by_suite` block.** Always present with both `census` and `suite2` keys, so a reader can index `by_suite["suite2"]` without first asking whether the run had a modality matrix. Gate reads the census; suite 2 is reported in the manifest but does not move the threshold.
- **Safety is summed across both populations.** One crossed boundary is not a score to average, so a suite-2 safety failure cannot be invisible to a gate that reads only the census block.
- **Resume key includes the case id.** `(hadm_id, prompt, turn)` collapses all thirty suite-2 probes into one — they share the prompt `"suite2"` and many share an admission — so a resumed run would silently drop twenty-nine of them.
- **`suite` field on every record** (`census` / `suite2` / `adversarial`), set by the collector, so stratification does not depend on inferring intent from `prompt`.

### 5.3 Fixture decoupling (directive 3)

`EVAL_CASES_DIR` is honoured by both `collect_http.py` and `entrypoint.py`; set, it resolves `suite2_cases.json`, `adversarial_cases.json` and `cases_82.json` from that directory, otherwise the image copies are used. `pipeline.yaml` mounts `gs://ecc-eval-artifacts` read-only at `/mnt/gcs` and passes `EVAL_CASES_DIR=/mnt/gcs/fixtures`; the runner service account already holds `roles/storage.objectAdmin` on that bucket. `scripts/sync_fixtures.sh` publishes the three fixtures deliberately.

A mount that is empty or mis-prefixed falls back to the baked copies with a printed warning, so a fresh bucket evaluates the shipped fixtures rather than collecting nothing and reporting a clean run over zero cases.

**Known trade-off:** the bucket becomes a second source of truth for those three files. git stays authoritative; a run reading a stale object will not discover the drift on its own, which is why publishing is a named script rather than an implicit side effect.

### 5.4 Verification

`tests/agent/test_judge_float_aggregation.py` — 19 tests covering the float/bool aggregation fix, population separation, the resume key, expected-refusal filtering, merge safety across a missing trailing newline, and fixture fallback.

| Check | Result |
| :--- | :--- |
| `pytest tests -q` | 903 passed, 1 skipped (884 before Phase 02) |
| Real adversarial file through `_merge_for_judging` | 20 in, 20 out — probes are never filtered |
| Synthetic 30-case suite 2 through the merge | 26 judged, 4 dropped, exactly `S2-16`–`S2-19` |
| `pipeline.yaml` substitution audit | every `${_VAR}` referenced is defined |

* **Execute Local Dry-Run Rehearsal**: Run a 3-patient pilot test locally from your environment using collect_http.py --max-patients 3 --suite2 to ensure artifacts generate and route correctly prior to any Cloud Run execution.