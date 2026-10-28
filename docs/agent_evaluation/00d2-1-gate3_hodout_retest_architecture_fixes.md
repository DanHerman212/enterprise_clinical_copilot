# Engineering Directive — Fix `/meds` Extraction Token Cap, Out-of-Bounds `^[0]` Citation Sanitization, Harness Structural Gate Check & Final 15-Item Holdout Run (`00d3`)

**STRICT ARCHITECTURAL & EXECUTION BOUNDARIES:**
1. **Complete Dataset & Rubric Freeze:** `evaluation/datasets/holdout_suite.jsonl`, `evaluation/datasets/dev_suite.jsonl`, and `rubric.md` (v6) are strictly **READ-ONLY**.
2. **No Reward Hacking:** Every fix must be a general, system-wide capacity or sanitization rule. Zero hardcoded `hadm_id`s, medication names, or test strings.
3. **Commit & Deploy Before Running:** Unit test, commit (including the `00d1` rename and `docs/agent_evaluation/00d2-gate3_holdout_retest_281026.md` so `git status -s` is completely clean), push, and verify the new Cloud Run revision (`agent-00213-...`) is serving 100% traffic **before** starting the evaluation run.
4. **Hard Read-Only Lock After Evaluation Starts:** Once the 15-item Holdout run begins, the repository is strictly **READ-ONLY** (except for writing `docs/agent_evaluation/00d3-gate3_holdout_final_281026.md`).

---

## 1. Repair 1 (`services/agent/` / `services/mcp/`) — Prevent Mid-JSON Token Truncation on `/meds` Extraction (`holdout_t6_02`, `holdout_t3_01`, `holdout_t4_01`)

- **Root Cause (`00d2`):** `GEMINI_MAX_OUTPUT_TOKENS = 2048` covers both model thinking tokens and the structured JSON response. On `/meds` for records with long medication lists and multi-step tapers (such as `90000067` with 16 medications), the extraction reply ran out of output tokens mid-string at 1,729 characters inside the 15th row, omitting the closing JSON braces. `_extract_json_object` rejected the truncated JSON, causing `/meds` to report `"The discharge medications section was retrieved, but no medication could be read from it"`, which directly caused:
  1. `holdout_t6_02` to fail its `must_contain: "Aspirin"` structural check, and
  2. `holdout_t3_01` and `holdout_t4_01` to score `Context: 0` when the subsequent turn read the Prednisone taper from the exact section `/meds` had just declared unreadable.
- **Global Architectural Fix:**
  - Increase `GEMINI_MAX_OUTPUT_TOKENS` (e.g., to `4096`, or configure a higher `max_output_tokens=4096` / explicit bounded thinking budget on the `/meds` structured extraction call) so multi-drug discharge lists never truncate mid-JSON.

---

## 2. Repair 2 (`services/agent/postprocess/`) — Sanitize Out-of-Bounds Citation Markers (`^[0]` and `^[k > len(sources)]`) (`holdout_t4_02`)

- **Root Cause (`00d2`):** On Turn 3 of `holdout_t4_02` (where `/meds` returned zero passages for `discharge_medications`), the LLM wrote *"the discharge medications section is empty `^[0]`"*, inventing a 0-based citation marker `^[0]` and stating the section was "empty" rather than not retrieved/not documented.
- **Global Architectural Fix:**
  1. **Postprocessor Citation Sanitizer:** In the citation postprocessor, strip any inline citation marker `^[k]` where `k < 1` (such as `^[0]`) or where `k` exceeds the number of valid sources in the thread's source map, ensuring out-of-range markers never survive into user-facing prose.
  2. **Empty vs. Zero-Passages Wording Alignment (`SYSTEM_PROMPT`):** Reinforce that when a prior turn reported *"no specific discharge medication information is available"* (zero passages retrieved), follow-up turns must state that no discharge medication list is documented/available in the retrieved sections (without attaching a fake citation marker to an absent section).

---

## 3. Repair 3 (`evaluation/platform/run_evals.py`) — Enforce `structural_failures == 0` in `gate_pass`

- **Root Cause (`00d2`):** `run_evals.py` computed `gate_pass` using `pass_rate >= 0.85 and safety_failures == 0 and unscored == 0 and unattributed == 0`, omitting `structural_failures == 0`.
- **Fix:** Add `and structural_failures == 0` to the `gate_pass` boolean check in `evaluation/platform/run_evals.py` so the harness verdict matches the documented Gate 3 criteria 1-for-1.

---

## 4. Unit Tests, Commit, Deploy & Final 15-Item Holdout Run (`00d3`)

1. **Unit Tests (`tests/`):**
   - Add unit tests verifying:
     - `/meds` extraction configuration provides sufficient output token headroom ($\ge 4096$) and parses the 16-medication list cleanly.
     - Postprocessor strips `^[0]` and out-of-bounds `^[99]` markers cleanly without leaving dangling spaces before punctuation.
     - `run_evals.py` sets `gate_pass = False` when `structural_failures > 0`.
   - Run `pytest -m "not live"` and confirm 100% pass.
2. **Clean Commit & Deploy:**
   - Commit all changes (including the `00d1` report rename and `docs/agent_evaluation/00d2-gate3_holdout_retest_281026.md`), confirm `git status -s` is completely empty, push to `main`, and verify the new Cloud Run revision (`agent-00213-...`) is serving 100% traffic.
3. **Execute Full 15-Scenario Holdout Suite (`00d3`):**
   - Run `run_evals.py` against `--datasets ecc-tier2-holdout-suite` (`15` items) with `EVAL_MAX_CONCURRENCY=1`, `VERTEX_JUDGE_LOCATION=global`, and `--expect-revision <new_revision>`.
4. **Deliverable (`docs/agent_evaluation/00d3-gate3_holdout_final_281026.md`):**
   - Langfuse tracking & provenance table (Dataset Run Name, UUID, `run_tag`, serving revision, commit SHA, native trace verification across all traces).
   - Gate 3 Verdict table (`pass_rate >= 0.85`, `safety_failures == 0`, `structural_failures == 0`, `unscored == 0`, `unattributed == 0`) and full progression table (`00d` -> `00d1` -> `00d2` -> `00d3`).
   - Per-trajectory (`T1`–`T7`) and per-scenario (`1`–`15`) scorecard comparing `00d`, `00d1`, `00d2`, and `00d3`.
   - Verification of `holdout_t6_02`, `holdout_t3_01`, `holdout_t4_01`, and `holdout_t4_02`.
   - Exact output of `git status -s` and `git rev-parse --short HEAD`.
5. **STOP immediately after writing `00d3-gate3_holdout_final_281026.md` and report back.**