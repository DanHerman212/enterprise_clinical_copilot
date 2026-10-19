Here is the complete specification for 01-langfuse_telemetry_and_trace_contract.md, followed by a brief summary of the three exact code bugs we found in your uploaded files that this document resolves.

# 01 — Langfuse Telemetry & Trace Contract

## 1. Executive Summary & Problem Statement
Out of the box, Langfuse renders a flat chronological stream of UUIDs, latency bars, and raw JSON payloads. When an evaluation run fails, hunting through individual trace drawers and unindexed metadata dictionaries to figure out which turn broke creates severe cognitive friction.

Inspection of the current codebase (`services/agent/graph.py`, `services/agent/deterministic.py`, `evaluation/agent/collect_http.py`, and `evaluation/agent/judge.py`) revealed four specific telemetry defects that prevented Langfuse from functioning as an interactive triage control plane:

1. **Turn 2 Drops `run_tag` (`collect_http.py`):** In `_run_set()`, Turn 1 calls `_body()` which attaches `run_tag`, but Turn 2 constructs a raw dictionary `{"question": case["follow_up"], "hadm_id": case.get("hadm_id"), "turns": [replayed]}` and omits `run_tag`. Consequently, every Turn 2 trace in Langfuse is orphaned from the evaluation run filter.
2. **Command Path Orphans Root Tags (`graph.py` & `deterministic.py`):** In `graph.ask()`, deterministic commands open a parent wrapper span via `with observability.observation(f"command.{question_kind}"):` before invoking `run_command()`. However, `trace_metadata` (which holds `langfuse_tags` and `langfuse_session_id`) is only passed into the inner formatting LLM `config`, leaving the root trace (`command.<wire>`) with empty tags (`tags: []`) and `sessionId: None`—or completely untraced on `/risk` turns where `command.prose` is `False` (zero LLM calls).
3. **Missing Session Stitching (`graph.py`):** While `collect_http.py` computes a local `rec["session"]` string for its JSONL file, `_trace_metadata()` in `graph.py` never sets `langfuse_session_id`. As a result, Turn 1 and Turn 2 cannot be viewed together as a chronological admission trajectory in the Langfuse **Sessions** view.
4. **Missing Triage Signal Promotion (`judge.py` & `graph.py`):** Root traces lack filterable operational tags (`turn:1|2`, `tools:<count>`, `verdict:pass|fail`), forcing engineers to open JSONL files in Cloud Storage rather than filtering directly in the Langfuse UI.

This module establishes a strict **Root-Trace Telemetry Contract** so every evaluation turn—whether a 0-LLM `/risk` widget, a 1-LLM `/ehr` synthesis, or a multi-step Turn 2 ReAct loop—lands in Langfuse with complete tags, unified session grouping, and hoverable score rationales.

---

## 2. Canonical Root-Trace Taxonomy

Every trace emitted by `services/agent/` during an evaluation or clinical session must carry the following standardized root properties:

| Property | Format / Key | Example Value | Operational Purpose |
| :--- | :--- | :--- | :--- |
| **Trace Name** | `command.<wire>` or `free_text.turn<n>` | `command.summarize`, `free_text.turn2` | Immediately distinguishes deterministic slash commands from open ReAct turns in the table. |
| **Session ID** | `langfuse_session_id` | `pilot-a1b2c3:hadm_90000015` (eval) or `hadm_90000015` (live) | Groups all turns for a single patient admission into one expandable row in the **Sessions** tab. |
| **Tags (At Birth)** | `langfuse_tags` array | `["run:pilot-a1b2c3", "hadm:90000015", "cmd:summarize", "turn:1"]` | Enables instant 1-click filtering in the Langfuse search bar without parsing JSON metadata. |
| **Tags (Post-Run)** | Appended at completion / judge | `["tools:2", "verdict:pass"]` | Surfaces zero-tool evidence misses (`tools:0`) and failed turns (`verdict:fail`) at the root level. |
| **Metadata** | Top-level dict | `{"hadm_id": "90000015", "command": "summarize", "turn": 1, "run_tag": "pilot-a1b2c3"}` | Structured audit context visible in the trace header. |

---

## 3. Code Specifications: Fixing the Telemetry Seam

### 3.1 Fix 1: Preserve `run_tag` on Turn 2 (`evaluation/agent/collect_http.py`)
In `evaluation/agent/collect_http.py`, update `_run_set()` so Turn 2 uses `_body()` (or explicitly includes `run_tag`) when dispatching the follow-up request:

```python
# In evaluation/agent/collect_http.py -> _run_set() -> one():
if case.get("follow_up") and first.get("status") == 200:
    if _key(case, 2) not in done:
        replayed = replay_turn(case["question"], first)
        turn2_body = _body(case["follow_up"], case)
        turn2_body["turns"] = [replayed]
        produced.append(await _answer(
            client, case, 2, case["follow_up"],
            turn2_body,
            prior=replayed,
        ))
```

### 3.2 Fix 2: Standardize `_trace_metadata` with Turn & Session ID (`services/agent/graph.py`)
Update `_trace_metadata()` in `services/agent/graph.py` so it computes the turn index (`1 + len(turns)`), constructs the deterministic `langfuse_session_id`, and formats the canonical tag set:

```python
# In services/agent/graph.py:
def _trace_metadata(
    hadm_id: int | None,
    question_kind: str | None,
    run_tag: str | None = None,
    turns: list[dict] | None = None,
) -> dict:
    """Build Langfuse root-trace metadata, tags, and sessionId."""
    tags: list[str] = []
    meta: dict[str, Any] = {}
    run_tag = run_tag or observability.current_run_tag()
    turn_index = 1 + len(turns or [])

    tags.append(f"turn:{turn_index}")
    meta["turn"] = turn_index

    if hadm_id is not None:
        tags.append(f"hadm:{hadm_id}")
        meta["hadm_id"] = str(hadm_id)
    if question_kind:
        tags.append(f"command:{question_kind}")
        meta["command"] = str(question_kind)
    else:
        tags.append("command:free_text")
    if run_tag:
        tags.append(f"run:{run_tag}")
        meta["run_tag"] = str(run_tag)

    # Bind multi-turn trajectories to a unified Langfuse Session
    if hadm_id is not None:
        session_id = f"{run_tag}:hadm_{hadm_id}" if run_tag else f"hadm_{hadm_id}"
        meta["langfuse_session_id"] = session_id

    meta["langfuse_tags"] = tags
    return meta
```

### 3.3 Fix 3: Stamp Root Trace Attributes on Deterministic Commands (`services/agent/observability.py` & `graph.py`)
In Langfuse v3/v4, `CallbackHandler` reads `langfuse_tags` and `langfuse_session_id` from `config["metadata"]` during a LangChain run. However, on the deterministic command path (`run_command`), the root trace is opened *outside* LangChain via `with observability.observation(f"command.{question_kind}"):`. Furthermore, `/risk` makes zero LLM calls (`command.prose = False`), so `CallbackHandler` never fires at all.

Update `observability.observation()` in `services/agent/observability.py` to accept `trace_metadata` and stamp the root trace via `update_current_trace` (or `span.update_trace`):

```python
# In services/agent/observability.py:
@contextlib.contextmanager
def observation(
    name: str,
    *,
    as_type: str = "span",
    trace_metadata: dict | None = None,
):
    """Open a Langfuse span and stamp root-trace tags/sessionId if provided."""
    manager = None
    if configured():
        try:
            import langfuse

            manager = langfuse.get_client().start_as_current_observation(
                as_type=as_type, name=name,
            )
        except Exception:  # noqa: BLE001 — rule 1
            logger.warning("could not open the langfuse span %r", name, exc_info=True)
            manager = None
    if manager is None:
        yield None
        return
    with manager as span:
        if trace_metadata and span is not None:
            try:
                span.update_trace(
                    name=name,
                    session_id=trace_metadata.get("langfuse_session_id"),
                    tags=trace_metadata.get("langfuse_tags"),
                    metadata={
                        k: v for k, v in trace_metadata.items()
                        if not k.startswith("langfuse_")
                    },
                )
            except Exception:  # noqa: BLE001 — rule 1
                logger.warning("could not update langfuse root trace", exc_info=True)
        yield span
```

And in `services/agent/graph.py` inside `ask()`:

```python
    trace_meta = _trace_metadata(hadm_id, question_kind, run_tag, turns)
    if routing.is_command(question_kind):
        tracing = observability.handler()
        with observability.observation(
            f"command.{question_kind}", trace_metadata=trace_meta
        ) as root_span:
            command_state = await run_command(
                toolbox,
                routing.command_for(question_kind),
                hadm_id,
                topic=topic,
                question=question,
                model=model,
                on_event=on_event,
                callbacks=[tracing] if tracing is not None else None,
                prose_stream=prose_stream,
                trace_metadata=trace_meta,
            )
            # Stamp tool count on completion
            if root_span is not None:
                try:
                    tool_count = len(command_state.get("tool_calls") or [])
                    root_span.update_trace(
                        tags=[*trace_meta.get("langfuse_tags", []), f"tools:{tool_count}"]
                    )
                except Exception:
                    pass
        command_state["langfuse_trace_id"] = observability.trace_id(tracing)
        return command_state
```
*(Apply the same `tools:<count>` tag update at the end of the free-text `graph.ainvoke` path so `tools:0` on Turn 2 is immediately filterable).*

### 3.4 Fix 4: Score Comments & Verdict Tag Promotion (`evaluation/agent/judge.py`)
In `evaluation/agent/judge.py`, `_attach_scores()` currently passes `flags` as the `comment`, which is empty on passing turns and omits the detailed `reasoning` paragraph when `flags` is empty. Update `_attach_scores()` to:
1. Format a concise, hoverable rationale string combining `flags` and `reasoning`.
2. Stamp `verdict:pass` or `verdict:fail` directly onto the trace's tag list so the Langfuse Traces table can filter by tag as well as by score.

```python
# In evaluation/agent/judge.py:
def _attach_scores(client, trace_id: str, j: dict) -> int:
    """Attach the judge verdict + per-dimension scores and rationale to Langfuse."""
    if client is None or not trace_id:
        return 0
    dims = j.get("dimensions") or {}
    flag_str = " | ".join(j.get("flags") or [])
    reason_str = (j.get("reasoning") or j.get("reason") or "").strip()
    comment = f"[{flag_str}] {reason_str}"[:500] if flag_str else reason_str[:500]

    n = 0
    for dim in DIMS:
        value = dims.get(dim)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        client.create_score(
            trace_id=trace_id, name=dim, value=float(value), comment=comment
        )
        n += 1

    verdict_pass = j.get("verdict") == "PASS"
    client.create_score(
        trace_id=trace_id,
        name="verdict",
        value=1.0 if verdict_pass else 0.0,
        comment=comment,
    )
    return n + 1
```

---

## 4. Langfuse UI Configuration: Turning Traces into a Triage Dashboard

Once the instrumentation contract above is live, configure `https://observability.danielmherman.com` with three persistent **Saved Views** and custom column visibility so nobody ever has to click into raw JSON blobs to diagnose a run.

### 4.1 Table Column Setup
In the **Traces** table, click **Columns** (top right) and enable:
1. `Tags` (shows `run:<tag>`, `hadm:<id>`, `cmd:<wire>`, `turn:<1|2>`, `tools:<k>`)
2. `Session ID` (links directly to the full multi-turn admission conversation)
3. `Scores: verdict`, `Scores: safety`, `Scores: fidelity`, `Scores: grounding`
4. `Score Comment` (surfaces the exact failure flag and judge reasoning on hover)

### 4.2 The 3 Persistent Saved Views
Create and save these three filter presets at the top of the Traces view:

| Saved View Name | Exact Filter Criteria | What It Surfaces in 1 Click |
| :--- | :--- | :--- |
| **🚨 1. Safety & Gate Breaches** | `Score: safety < 2` **OR** `Score: verdict = 0` | Every failed turn or clinical safety violation in the current run, with the judge's defect flag visible in the row. |
| **🔍 2. Turn-2 Evidence Misses** | `Tag = turn:2` **AND** `Tag = tools:0` | Every follow-up turn where the agent answered from memory or manifest short-circuit without calling a retrieval tool (Bucket A). |
| **📉 3. Grounding & Fidelity Drift** | `Score: grounding < 2` **OR** `Score: fidelity < 2` | Numeric discrepancies, lab trend misreadings, or unsupported clinical claims (Buckets B & D). |

### 4.3 Multi-Turn Trajectory Review (Sessions View)
To inspect conversational state propagation between Turn 1 and Turn 2:
1. Click **Sessions** in the left navigation bar.
2. Filter by prefix `pilot-` or `census-`.
3. Each row is one patient admission (`{run_tag}:hadm_{hadm_id}`). Expanding a row displays Turn 1 (`/ehr`) stacked chronologically above Turn 2 (*"What was the discharge disposition..."*), making context retention and tool-call inheritance immediately visible.

---

## 5. Verification Contract (`tests/agent/test_eval_contract.py`)
Before deploying or running a pilot, add two fast in-process tests to `tests/agent/test_eval_contract.py`:
1. `test_trace_metadata_includes_turn_and_session()`: Asserts `_trace_metadata(90000015, "summarize", "test-run", turns=[{...}])` emits `turn:2`, `run:test-run`, `hadm:90000015`, `command:summarize`, and `langfuse_session_id == "test-run:hadm_90000015"`.
2. `test_turn2_collector_preserves_run_tag()`: Asserts that Turn 2 request bodies built by `collect_http.py` always carry `"run_tag"` when `case["run_tag"]` is set.

---

### Key Takeaway from Inspecting Your Code for Phase 01
By reading `services/agent/graph.py`, `services/agent/observability.py`, and `evaluation/agent/collect_http.py` side by side, we uncovered why your Langfuse experience felt so broken yesterday:
- **Why `/risk` traces were invisible or empty:** `/risk` has `prose=False`, so `run_command` returns early before `_build_llm` is ever called. Because `trace_metadata` was only passed to `llm.ainvoke(..., config=config)`, `/risk` never received any tags or metadata at all! Passing `trace_metadata` directly into `observability.observation()` fixes all deterministic commands at the root span.
- **Why Turn 2 traces disappeared when filtering by `run:<tag>`:** `collect_http.py` line 404 hardcoded `{"question": case["follow_up"], "hadm_id": case.get("hadm_id"), "turns": [replayed]}` without calling `_body()`, dropping `run_tag` on every single follow-up turn in the evaluation.

---

## 6. As-Built Notes & Known Gaps

Phase 01 was implemented locally-first, tested in-process, committed, and only then deployed. This section records where the shipped code diverges from the specification above and what remains open.

### 6.1 Gap 1 — `tools:<n>` cannot be a trace tag (SDK 4.14.4)

The spec in §3.3 assumes the root span can write back to its parent trace. It cannot: the installed Langfuse SDK exposes **no post-creation trace attribute updater**. Verified by introspection:

- `LangfuseSpan` has **no** `update_trace` method (first implementation attempt raised `AttributeError`, which was swallowed by the surrounding try/except and produced silently empty traces).
- `LangfuseClient` exposes `create_trace_id`, `get_current_trace_id`, `score_current_trace`, `set_current_trace_io`, `update_current_generation` — none of which mutate trace tags after creation.
- Trace-level attributes can only be set **at trace creation**, via `langfuse.propagate_attributes(trace_name=..., session_id=..., tags=..., metadata=...)`.

**As built:** `observability.observation()` opens the span *and* wraps it in `propagate_attributes(...)` for the four trace-level fields. The tool count, which is only known *after* the graph finishes, is written to **root-span metadata** as `{"tools": n}` by `stamp_trace_tags(span, tags)`.

**Consequence for §4.2:** Saved View **#2 must not be built on `Tag = tools:0`** — that tag will never exist. Build it on span metadata instead.

| Saved View | Corrected filter for SDK 4.14.4 |
| :--- | :--- |
| 🚨 1. Safety & Gate Breaches | `Score: safety < 2` **OR** `Score: verdict = 0` (unchanged) |
| 🔍 2. Turn-2 Evidence Misses | `Tag = turn:2` **AND** `Metadata: tools = 0` |
| 📉 3. Grounding & Fidelity Drift | `Score: grounding < 2` **OR** `Score: fidelity < 2` (unchanged) |

If a true trace-level `tools` tag is required later, the fix is to buffer the tool count and emit the trace's final attributes in a single write at the end of the turn — a Phase 02/04 concern, not a Phase 01 blocker.

### 6.2 Gap 2 — no Cloud Build trigger for the harness

Push triggers currently exist for the deployed services only:

| Trigger | Builds |
| :--- | :--- |
| `deploy-agent-on-push` | `services/agent/cloudbuild.yaml` |
| `deploy-mcp-on-push` | `services/mcp/cloudbuild.yaml` |
| `deploy-on-push` | root `cloudbuild.yaml` |

There is **no** trigger on `evaluation/agent/pipeline.yaml`. The eval harness is still built by hand, so a harness change is not reproducible from a commit. Adding that trigger belongs to Phase 04 (CI/CD); until then, any harness fix must be built explicitly and the image tag recorded.

### 6.3 Operational note — regional triggers

All three triggers are **2nd-gen and regional (`us-east1`)**. `gcloud builds list` without `--region` defaults to `us-central1` and will report *no new build* even when a trigger fired correctly. Always check:

```
gcloud builds list --project trim-icon-498815-a0 --region us-east1 --limit 5 \
  --format="table(id,status,substitutions.TRIGGER_NAME,substitutions.COMMIT_SHA)"
```

This cost a false "the trigger didn't fire" conclusion during Phase 01 verification.

### 6.4 Live verification evidence (commit `e8c587b`, revision `agent-00148-dl4`)

Three new tests were added (the spec asked for two; `test_live_traffic_gets_no_invented_session` guards the deliberate decision that untagged production traffic must **not** be given a fabricated session id). Full suite: 884 passed, 1 skipped.

End-to-end check against the deployed service — `POST /ask {"question":"/risk","hadm_id":90000015,"run_tag":"phase1-verify"}`:

| Field | Observed | Required |
| :--- | :--- | :--- |
| trace name | `command.risk` | ✓ root span now carries attributes without an LLM call |
| tags | `command:risk`, `hadm:90000015`, `run:phase1-verify`, `turn:1` | ✓ |
| sessionId | `phase1-verify:hadm_90000015` | ✓ sessions group per admission |
| returned `langfuse_trace_id` | `2881690a92df910f1375272a7739775f` (non-empty) | ✓ usable as a join key in harness reports |
| root-span metadata | `{"tools": 1}` | ✓ tools recorded (as metadata, per §6.1) |
| child spans | `tool.predict_readmission` | ✓ |

This trace was the first non-empty `/risk` trace in the project. Note it was produced with an authenticated request (`Authorization: Bearer $(gcloud auth print-identity-token)`); an anonymous POST returns 403 from Cloud Run, not from the agent.

Review **`01-langfuse_telemetry_and_trace_contract.md`** above—if this looks solid to you, we can move straight into generating **`02-harness_architecture_and_assertions.md`**!