# Operational Checklist: Execution Agenda

Here is the operational checklist to close out the remaining milestones today in exact order:

## Step 1: Immediate Telemetry & Thumbs Verification (5 mins)
Before running the full UX gauntlet, confirm the trace ingestion fix and the thumbs feedback hook:

1. Open the UI, select Cynthia Petrov (`#90000015`), and send a single turn (`/ehr`).
2. Click **Thumbs Up** (or Thumbs Down).
3. Open `https://observability.danielmherman.com`:
   - Confirm the new trace appears with its full execution graph.
   - Confirm a score entry (e.g., `user_feedback`, `thumb`, or `user_rating`) is attached with value `1` or `0`.
   - Confirm the trace carries the session/hadm metadata cleanly.

## Step 2: Execute the 14-Turn UX Gauntlet (Phase 4C)
With the LangGraph `RECURSION_LIMIT` raised, partial DOM preservation in place, and trace ingestion flowing, run through the 14 turns:

- **Archetype A:** Cynthia Petrov (TKA) — `/ehr`, RDW driver click (abstention + lab panel), free-text labs query (`⚡ Routed to /labs`), `/meds`, and clinical follow-up click.
- **Archetype B:** High-risk medical phenotype (CHF/COPD) — `/ehr` with condition-specific matrix, `/labs`, and free-text intent routing.
- **Archetype C:** Cross-condition literature isolation — `/pubmed congestive heart failure` on Cynthia Petrov to verify matrix suppression and clean citation rendering.
- **Archetype D:** Policy fallbacks and edge resilience — non-admission notes and mistyped slash commands.
- **Deliverable:** Produce `04C-ux_audit_findings_report.md` to verify zero raw JSON leaks, clean Markdown rendering, and strict absence of ungrounded medical advice.

## Step 3: Migrate Evaluation Artifacts to Google Cloud Storage (GCS)
Before running the full eval census, transition away from local `.jsonl` disk storage to prevent repository bloat and accidental PHI/clinical narrative commits.

1. **Provision Bucket:** Ensure `gs://ecc-eval-artifacts` exists in project `trim-icon-498815-a0` (with a 90-day lifecycle rule or archival tier).
2. **Automated Upload Hook:** Create `evaluation/agent/storage.py` with `sync_eval_to_gcs(run_tag: str, files: list[Path])` targeting `gs://ecc-eval-artifacts/runs/{run_tag}/`.
3. **Git & CLI Updates:** Ensure `evaluation/agent/results/*.jsonl` and `*.json` are explicitly ignored in `.gitignore`. Add an `--upload-gcs` flag to `collect_http.py` and `judge.py` that syncs artifacts upon exit.

## Step 4: Run the Eval Harness (`05-agent_eval_agenda.md`)
Once the UX audit report is signed off and GCS is wired:

1. **Gate 1 Preflight:** Run `collect_http.py --preflight-only` to verify transport and retrieval health[cite: 1].
2. **Gate 2 Pilot:** Run `--max-patients 3` (clinical + adversarial)[cite: 1]. Run `judge.py` and inspect `report_pilot.json` to verify the judge correctly handles `"9.0%"` rounded risk displays and untruncated lab payloads without false-positive failures[cite: 1].
3. **Gate 3 Census:** Run the full 82-inpatient census + 20 adversarial cases with `--upload-gcs`.
4. **Gate 4 Langfuse Tagging:** Run `tag_traces.py --tag eval-golden-v3` to stamp the production dataset in ClickHouse.

Once you confirm traces and thumbs scores are landing in the Langfuse dashboard, you are authorized to start the 14-turn browser run.