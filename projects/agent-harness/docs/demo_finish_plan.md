# Demo Finish Plan — 2026-08-11

Goal: **a working end-to-end demo** — the agent answers a clinician's question
about a patient with a risk score *and* cited evidence from the patient's notes.

All data processing stays cloud-only. Everything is already built except the
`rag_search` tool and the wiring that puts it in the demo.

---

## Where we are (done)

| Piece | Status |
|---|---|
| Indexes (tree-AH 555,770 + brute-force 2k) | ✅ built, kept on GCS |
| §9 R1 cross-patient isolation | ✅ PASS (both directions) |
| §9 demo sanity (query → real passages) | ✅ PASS |
| §9 recall@k (0.80 / 0.84 / 0.83) | ✅ done — index quality fine |
| Index endpoint | 🔴 **torn down** (index kept — redeploy when demo runs) |
| Prediction endpoint | 🔴 **torn down** (redeploy via `deploy_cpr.py` when demo runs) |
| `rag_search` MCP tool | ✅ built + 9 offline tests pass |
| Live cross-patient R1 + ML test | ✅ **PASS** (Step 5 gate) — `integration_test_live.py` |
| Agent calls `rag_search` on its own | ✅ **PASS** (Step 6) — smoke test cites `^[n]` |
| Demo UI shows citations | ❌ not wired — next phase (UX redesign) |

---

## The dependency order

```
redeploy endpoint (cost!) → build rag_search tool → offline tests
   → register in MCP server → live cross-patient test
   → agent fusion (prompt + demo) → UI citations → run the demo
```

---

## Step-by-step

### Step 1 — Redeploy the index endpoint  ⛔ cost gate (~$270/mo)

- **Why first:** every later step needs a live index to query.
- **Command:** `deploy_index.py --mode deploy --index-id 2371299135438454784`
- **Verify:** one `find_neighbors` call returns real neighbors (we know it works).
- **Cost:** ~$270/mo meter starts now. **Needs explicit user approval.**
- ⏱ ~10 min (index is kept; only the endpoint is rebuilt)

### Step 2 — Build the `rag_search` MCP tool (§8)

New file `mcp_server/tools/rag_search.py`, mirroring `predict.py`'s structure.

**Contract:** `rag_search(hadm_id: int, query: str, top_k: int = 5) -> dict`

**Behavior:**
1. Embed the query (`gemini-embedding-001`, RETRIEVAL_QUERY, 768 dims).
2. `find_neighbors` on the deployed index with the **`hadm_id` restrict applied
   server-side, always** — never optional, never a post-filter.
3. Map returned datapoint IDs → **text from BigQuery** (the index stores no text).
4. Return plain JSON:
   ```json
   {
     "hadm_id": 20924467,
     "query": "...",
     "returned": 2,
     "passages": [
       {"id": "...", "section": "brief_hospital_course", "text": "...", "score": 0.27}
     ]
   }
   ```
5. **Empty is a real answer:** `{"passages": [], "returned": 0}` — never fabricate.
6. **Structured errors** (same `_error` pattern as `predict.py`): bad hadm_id,
   BigQuery lookup miss, index timeout, embed failure.

**Non-negotiables (from the guide):** the restrict is applied *server-side by the
tool*, the returned text is looked up by ID (never reverse-parsed), and a passage
that's missing from BigQuery errors rather than silently dropping.

- ⏱ ~45–60 min

### Step 3 — Offline tests, no cloud credentials (§8/§9)

New file `tests/test_rag_search.py` with a **fake index client** (like
`test_rag_live.py`'s offline tier):

- restrict is always applied to the query (assert the fake receives the filter)
- empty result → `{"passages": [], "returned": 0}`
- malformed `hadm_id` → structured error
- a returned ID missing from BigQuery → error, not silent drop
- response shape matches the contract

**Verify:** `pytest tests/test_rag_search.py` passes with no GCP credentials.

- ⏱ ~30 min

### Step 4 — Register the tool in the MCP server

- `mcp_server/tools/__init__.py`: export `rag_search`.
- `mcp_server/server.py`: `server.add_tool(rag_search)` + mention in instructions.
- Deploy the MCP server (or run locally on stdio for the next step).

- ⏱ ~15 min

### Step 5 — Live cross-patient test (the gate)

Run the strongest R1 test against the *registered* tool (or reuse the pattern in
`validate_rag.py`'s `run_r1_positive`):

- query with patient A's `hadm_id` using text lifted verbatim from patient B's note
- assert: returns A's passages or nothing — **never B's**

**Gate:** if this fails, stop — everything downstream is unsafe.

- ⏱ ~20 min

### Step 6 — Agent fusion (§11): the agent calls `rag_search` on its own

- **`agent/prompts.py`:** add a section — when the question needs supporting
  evidence from the notes, call `rag_search(hadm_id, query)`, then cite passages
  by section. Conflicts must surface: if notes contain observations warranting
  clinician judgment that the risk score doesn't reflect, say so — **cited, with
  no prediction claim**.
- **`agent/graph.py`:** no structural change needed — tools are auto-discovered
  from the MCP toolbox. Verify `rag_search` appears in `box.names`.
- **`agent/run.py`:** smoke-test a question that *requires* retrieval (e.g., "what
  does the note say about this patient's discharge medications?") and confirm the
  tool is called and cited.
- Add a test: every intervention carries a citation; a constructed
  low-risk/high-concern case produces a conflict callout.

- ⏱ ~60–90 min

### Step 7 — Demo UI: show the citations (§13)

- `danielmherman/demo/`: the Django demo page calls the agent via `agent_client.py`
  and renders the A2UI card. Extend to render the evidence/citations the agent now
  returns (risk + band, drivers, evidence with citations, intervention plan).
- Keep note-text rendering behind one clean boundary (one component).

- ⏱ ~60–90 min

### Step 8 — End-to-end demo run + eval sanity (§12)

- Walk a real demo patient: risk score + drivers + cited evidence + intervention.
- Verify groundedness: every claim in the output traces to a passage or a feature.
- Note: the demo cohort must overlap the test split (which the index covers) or
  `rag_search` will return empty for demo patients — **check this in Step 5/7**.

- ⏱ ~30 min

---

## Cost & teardown

| Resource | Status today | To run the demo |
|---|---|---|
| Index endpoint | torn down | redeploy in Step 1 (~$270/mo) |
| Prediction endpoint | torn down (per earlier notes) | redeploy if demo shows predictions |
| Index / models / storage | kept | free |

Teardown when done: `teardown.py --only vector-index` (keeps the index).

---

## Decisions needed

1. **Step 1 cost gate:** OK to start ~$270/mo for the endpoint? (This is the demo's
   main recurring cost. Teardown is one command.)
2. **Demo cohort:** is the current 32-patient cohort inside the test split? If not,
   `rag_search` returns empty for those patients — we either use test-split patients
   or rebuild the cohort (§10).

---

## Sequencing summary (with the gates that block)

| Step | Blocks on |
|---|---|
| 1. Redeploy endpoint | **Your cost approval** |
| 2. Build rag_search | — |
| 3. Offline tests | — |
| 4. Register tool | — |
| 5. Cross-patient test | **Must pass** |
| 6. Agent fusion | Step 5 pass |
| 7. Demo UI | Step 6 |
| 8. E2E demo + eval | Step 7 |

Estimated remaining work: **~4–6 hours** including one deploy cycle.
