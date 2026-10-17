# UX Audit Assessment & Execution Directives

The `04C-ux_audit_findings_report.md` document isolates the exact failure boundaries required to patch before burning compute on the 82-patient evaluation census. 

## 1. Insights (Wins & Validated Fixes)
* **Recursion & Budget Hardening Hold:** B3 (15 steps, four tool rounds) and D1 (18 steps) ran inside the five-call budget with zero `GraphRecursionError` crashes and zero BFF "unavailable" wipes. Unbinding tools once the budget is spent permanently solved the runaway ReAct loop.
* **Manifest Hydration & Routing Fix:** Administrative questions now skip forced retrieval, preventing the 20-second false-positive `rag_search` loop and preserving manifest-grounded answers (e.g., `17.29 days`).
* **Epistemic Discipline & Sanitization:** Honest abstention held across A2 (RDW driver) and D2/D3 (sparse outpatient notes). All 13 turns settled with zero raw JSON leaks and zero raw `^[` citation markers once non-numeric/non-PMID markers (`^[brief_hospital_course]`) were stripped.
* **Telemetry Polish:** Tool spans no longer detach into orphan root traces, and thumbs feedback now uses signed `1`/`-1` upserts to distinguish a thumbs-down from an unrated turn.

## 2. Risks
* **Production vs. Local Drift (High Risk for Gate 1):** The citation sanitization fix (`be728d5`) and today's patches ran against the local agent (`port 8080`), while Cloud Run still runs an older build. The evaluation census will test stale code if run against `$AGENT_URL` without deploying.
* **Router Hijacking on Multi-Intent Queries (F1):** Matching whole tokens without intent disambiguation caused A5 (*"What does current clinical literature suggest regarding... medications..."*) to route to `/meds` instead of a literature search. If left unpatched, evaluations combining literature terms with `"medications"` or `"labs"` will return the wrong modality.
* **Typo Budget Burn & False Reconciled Signal (F2):** A mistyped command (`/ehrr`) fell through to the reasoning loop, burned 18 steps, and triggered the `🛡️ Reconciled with medical record` badge—signaling a clinical discrepancy when the user simply made a typo.

## 3. Gaps
* **Missing Trace Metadata (F4):** Traces arrive in Langfuse with `sessionId: None`, `userId: None`, and `tags: []`. Without `hadm_id` and `conversation_id` attached at trace creation, Gate 4 triage cannot filter by patient or session in the Langfuse UI.
* **Unverified Partial-Stream Failure Path:** Because the tool budget fix held, no turn failed mid-stream during the 13-turn run. The partial DOM preservation (`⚠️ Copilot run incomplete`) was not exercised live.

## 4. Ambiguities (Product Call Required on F3)
* **Outpatient Note Risk Widget:** On outpatient note `#90000054`, `/ehr` properly refused to invent a discharge diagnosis in prose, but the UI still rendered the 30-day inpatient readmission risk widget (`5.7%, below threshold`). 
  * **Product Call:** The 5 outpatient notes (`90000028, 90000040, 90000051, 90000054, 90000073`) are clinically non-applicable for inpatient readmission risk. Suppress the risk widget when `admission_metadata` indicates an outpatient record, and exclude these 5 records from the standard `summarize`/`meds`/`risk` evaluation census (yielding an **82-admission inpatient census**).

---

## 5. Directives for the Engineer

The 04C audit report is signed off. Execute these fixes in exact sequence before starting the evaluation gates:

1. **Patch F1 (Literature Abstention in `routing.py`):**
   Add a negative lookahead / abstention set to `classify_free_text()`. If the query contains literature/evidence cues ("literature", "evidence", "studies", "trials", "meta-analysis", "pubmed", "guidelines", "efficacy"), abstain from `/meds` and `/labs` routing so the query enters the reasoning loop. Add a test case for the A5 prompt.

2. **Patch F2 (Fast-Fail Unknown Slash Commands in `http.py` / `routing.py`):**
   If the input starts with `/` (`^/[a-zA-Z0-9_-]+`) and the command token does not match a registered command (`/ehr`, `/labs`, `/meds`, `/risk`, `/pubmed`), short-circuit before invoking the graph. Return an immediate, zero-tool response listing the valid slash commands with no `Reconciled` badge.

3. **Patch F3 (Outpatient Risk Suppression & 82-Case Census Alignment):**
   For the 5 outpatient non-admission records, suppress the `RiskInfluencers` widget on `/ehr` (or render a "Readmission risk not applicable" notice). In the eval harness, isolate these 5 records from the standard inpatient census so the main clinical run scores the 82 inpatient admissions.

4. **Patch F4 (Langfuse Trace Metadata in `graph.ask` / `http.py`):**
   Pass `session_id=str(conversation_id)`, `user_id` (if present), `tags=[f"hadm:{hadm_id}"]`, and `metadata={"hadm_id": hadm_id, "command": kind}` into the Langfuse trace/handler configuration so every trace is filterable by patient and conversation in Langfuse.

5. **Complete Step 3 (GCS Artifact Storage) & Deploy to Cloud Run:**
   * Wire `evaluation/agent/storage.py` (`gs://ecc-eval-artifacts/runs/{run_tag}/`) and update `.gitignore`.
   * Deploy the current branch (`be728d5` + F1–F4 fixes) to Cloud Run first so the evaluation harness does not test stale container code.

6. **Execute Step 4 (Evaluation Gates 1 -> 4):**
   Run Gate 1 (`--preflight-only`) and Gate 2 (`--max-patients 3` pilot + judge). Share `report_pilot.json` and `adv_report_pilot.json` before launching the Gate 3 full census.