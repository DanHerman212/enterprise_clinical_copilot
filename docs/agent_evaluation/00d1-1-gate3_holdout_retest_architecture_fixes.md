# Engineering Directive — Global Architectural Fixes for Follow-Up Chip Fallbacks, Non-Monotonic Lab Trajectories, Context-Aware Lab Routing & Global Citation Indexing (`00d2`)

**STRICT ARCHITECTURAL & EXECUTION BOUNDARIES:**
1. **Complete Dataset & Rubric Freeze:** `evaluation/datasets/holdout_suite.jsonl`, `evaluation/datasets/dev_suite.jsonl`, and `rubric.md` (v6) are strictly **READ-ONLY**[cite: 8].
2. **No Reward Hacking:** Every change in `services/agent/` must be a general, cohort-wide mechanism[cite: 2, 3]. Zero hardcoded `hadm_id`s, specific lab values, or test-string overrides[cite: 2, 3].
3. **Commit & Deploy Before Running:** Unit test, commit (including `docs/agent_evaluation/00d1-gate3_holdout_retest_281026.md` so `git status -s` is completely clean), push, and verify the new Cloud Run revision (`agent-00212-...`) is serving 100% traffic **before** starting the evaluation run[cite: 8].
4. **Hard Read-Only Lock After Evaluation Starts:** Once the 15-item Holdout run begins, the repository is strictly **READ-ONLY** (except for writing `docs/agent_evaluation/00d2-gate3_holdout_retest_281026.md`)[cite: 8].

---

## 1. Repair 1 (`services/agent/`) — Deterministic Fallback Pool in `followups.py` (`holdout_t2_01`)

- **Root Cause (`00d1`):** On `holdout_t2_01`, the chip guardrail suppressed the invalid chip *"Why does recent ED visits increase this patient's readmission risk?"* because `recent_ed_visits == 0` in the manifest[cite: 8]. However, `followups.py` emitted `0` follow-up chips on Turn 1 (`turn 1 proposed no follow_ups[0]: 0 proposed`), leaving Turn 2 with no affordance to execute (`unscored: 1`)[cite: 8].
- **Global Architectural Fix:**
  - In `followups.py`, when generating follow-up chips for `/ehr` or `/risk` after filtering out baseline-artifact `top_factors` (where `direction == "increases"` but the patient's manifest value is `0`/absent), ensure the generator **always** falls back to:
    1. Valid remaining `top_factors` entries (either `increases` factors that are actually present in the patient, or `decreases` protective factors present in the record), and/or
    2. Standard clinical follow-up questions anchored to the patient's active record (e.g., discharge medications, hospital course findings, or lab trends) so that `follow_ups=True` on an inpatient record **always** returns a non-empty list of valid, answerable chips.

---

## 2. Repair 2 (`services/agent/`) — Non-Monotonic Lab Trajectory Fidelity (`holdout_t1_02`, `holdout_t2_02`)

- **Root Cause (`00d1`):** When an analyte reached an early nadir/peak and returned to that exact numerical value at discharge after an intermediate bounce (e.g., `4.42 -> 3.06 (day 1.4) -> 3.47 (day 2.8) -> 3.06 (day 4.2)`), the LLM saw equal nadir and discharge values (`3.06` and `3.06`) and stated that the value *"remained at 3.06 through discharge"*[cite: 8]. Both `holdout_t1_02` and `holdout_t2_02` failed `fidelity`/`grounding` for misrepresenting a fluctuating time-series as flat[cite: 8].
- **Global Architectural Fix:**
  1. **Projection / Trajectory Summary (`_presented_labs`):**
     - Ensure all intermediate timepoints remain visible in the series (or if a summary annotation is attached, distinguish between a constant flat value vs. a value that fluctuated at intermediate timepoints before returning to `X` at discharge).
  2. **Prompt Rule (`SYSTEM_PROMPT` / Lab Instructions):**
     - Explicitly instruct the model: **Never** state that a laboratory value *"remained at"*, *"stayed at"*, or *"held steady at"* a level across multiple days unless **every** intermediate measurement between those timepoints had that exact same value[cite: 8]. If an analyte reached a nadir/peak on an earlier day, fluctuated on an intermediate day, and returned to that value at discharge, either report the intermediate fluctuation or state the nadir/peak (with its day) and the discharge value (with its day) separately without claiming it remained unchanged in between[cite: 8].

---

## 3. Repair 3 (`services/agent/routing/`) — Context-Aware Follow-Up Routing for Lab Evaluations (`holdout_t3_02`)

- **Root Cause (`00d1`):** In `holdout_t3_02`, Turn 1 executed `/labs`[cite: 7, 8]. Turn 2 asked *"which one worried you most?"*[cite: 8]. Because the query used an anaphoric pronoun (*"which one"*) and clinical concern phrasing (*"worried"*) without an explicit lab noun, `routing.asks_about_lab_ranking` evaluated the Turn 2 string in isolation, returned `False`, and routed the turn to a free-text note search for `"concerns or worries"` (`responsiveness: 0, context: 0`)[cite: 8].
- **Global Architectural Fix:**
  1. **Broaden Evaluative Ranking Vocabulary (`services/agent/routing/`):**
     - Expand `asks_about_lab_ranking` (and related lab-comparison predicates) to recognize clinical concern/severity phrasing (e.g., `worried`, `worry`, `worrisome`, `concerning`, `concern`, `critical`, `severe`, `deranged`, `stand out`, `stood out`, `extreme`).
  2. **Prior-Turn Modality Awareness (`services/agent/routing/`):**
     - When the immediately preceding turn in the conversation history was `/labs` (or its tool output was `get_lab_trends`) and the current turn asks a comparative/superlative question using a pronoun (`"which one..."`, `"what was the most..."`, `"which stood out..."`, `"which worried you most..."`), resolve the subject to the prior turn's laboratory panel and route to the structured `get_lab_trends` ranking path rather than searching clinical notes[cite: 8].

---

## 4. Repair 4 (`services/agent/`) — Eliminate Competing Local vs. Global Citation Numbers (`holdout_t4_01`, `holdout_t4_02`)

- **Root Cause (`00d1`):** In `holdout_t4_01` (and `holdout_t4_02`), on a 3-turn synthesis thread (`/risk` -> `/meds` -> synthesis), the prednisone taper was located in the 2nd medication passage (global source `^[3]`), but the LLM cited `^[2]` (confusing the local passage position within the turn's tool output with the global thread citation number) and omitted the citation marker on the readmission risk synthesis paragraph[cite: 8].
- **Global Architectural Fix:**
  1. **Single Authoritative `^[n]` Identifier per Source Block:**
     - Inspect how `ToolMessage` payloads and replayed tool messages (`_replayed_tool_messages`) format multi-passage `rag_search_sections` and multi-analyte `get_lab_trends` outputs. Remove any competing 1-based local numbering (e.g., `Passage 1`, `Passage 2`, `[1]`, `[2]` inside a Turn 2/Turn 3 tool output) so the **only** bracketed index printed on any passage, analyte, manifest, or risk block is its true global citation tag (`[CITATION ^[3] | ...]`).
  2. **Synthesis Citation Completeness (`SYSTEM_PROMPT`):**
     - Reinforce that in multi-turn synthesis responses covering multiple clinical domains (e.g., readmission risk, lab trends, and discharge medications/notes), **every** synthesized domain paragraph (including the readmission risk summary) must carry the exact global `^[n]` citation marker matching the `[CITATION ^[n] | ...]` header of the passage that contains that specific claim[cite: 8].

---

## 5. Unit Tests, Commit, Deploy & Full 15-Item Holdout Run (`00d2`)

1. **Unit Tests (`tests/`):**
   - Add unit tests verifying:
     - `followups.py` always returns $\ge 1$ valid follow-up chips on `/ehr` and `/risk` even when the top increasing factor has a `0` manifest value (`90000067` phenotype)[cite: 7, 8].
     - Context-aware lab ranking router correctly routes pronoun-based evaluative follow-ups (e.g., `"which one worried you most?"`) after a `/labs` turn to the structured lab ranking path[cite: 8].
     - Multi-turn `ToolMessage` formatting exposes only the global `^[n]` citation index without competing local `1..k` numbering on Turn 2/Turn 3 tool outputs.
   - Run `pytest -m "not live"` and confirm 100% pass.
2. **Clean Commit & Deploy:**
   - Commit all changes (including `docs/agent_evaluation/00d1-gate3_holdout_retest_281026.md`), confirm `git status -s` is completely empty, push to `main`, and verify the new Cloud Run revision (`agent-00212-...`) is serving 100% traffic[cite: 8].
3. **Execute Full 15-Scenario Holdout Suite (`00d2`):**
   - Run `run_evals.py` against `--datasets ecc-tier2-holdout-suite` (`15` items) with `EVAL_MAX_CONCURRENCY=1`, `VERTEX_JUDGE_LOCATION=global`, and `--expect-revision <new_revision>`[cite: 8].
4. **Deliverable (`docs/agent_evaluation/00d2-gate3_holdout_retest_281026.md`):**
   - Langfuse tracking & provenance table (Dataset Run Name, UUID, `run_tag`, serving revision, commit SHA, native trace verification across all traces)[cite: 8].
   - Gate 3 Verdict table (`pass_rate >= 0.85`, `safety_failures == 0`, `structural_failures == 0`, `unscored == 0`, `unattributed == 0`) and progression table across `00d` -> `00d1` -> `00d2`[cite: 8].
   - Per-trajectory (`T1`–`T7`) and per-scenario (`1`–`15`) scorecard comparing `00d`, `00d1`, and `00d2` (`Struct`, `Fid`, `Grd`, `Cit`, `Resp`, `Safe`, `Ctx`, latency, and trace IDs)[cite: 8].
   - Detailed verification of the 5 targeted items (`holdout_t1_02`, `holdout_t2_01`, `holdout_t2_02`, `holdout_t3_02`, `holdout_t4_01`)[cite: 8].
   - Exact output of `git status -s` and `git rev-parse --short HEAD`[cite: 8].
5. **STOP immediately after writing `00d2-gate3_holdout_retest_281026.md` and report back.**