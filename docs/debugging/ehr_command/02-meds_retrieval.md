# Engineering Directive: Lossless `/meds` Retrieval, Narrative Fallback & Citation Resolution

## 0. Governance & Strict Engineering Rules (Read First)
1. **Strict File Allowlist:** You may ONLY modify the following production files (plus unit test files under `tests/`):
   - `services/mcp/tools/retrieval.py`
   - `services/agent/engine/prompts.py`
   - `services/agent/presentation/a2ui.py`
   Run `git status` and `git diff --name-only` before finishing to verify no other production files were touched.
2. **Architectural & Layer Separation Invariants:**
   - Preserve the strict package DAG in `services/agent/`. `tests/agent/test_layer_separation.py` must pass without modification.
   - Do not modify `_search_sections`, `EHR_FORMATTING`, `deterministic.py`, `sections.py`, `citations.py`, `/labs`, `predict_readmission`, `search_literature`, or the `rag_search` vector search path.
3. **Zero Patient-Specific Logic:** No hardcoded `hadm_id` or `note_id` branches, and no clinical string matching specific to a single record. All logic must operate globally across all 87 cohort notes.

---

## Step 1: Make `_search_medication_sections` Lossless with Narrative Fallback (`services/mcp/tools/retrieval.py`)
Rewrite `_search_medication_sections(hadm_id: int)` in `services/mcp/tools/retrieval.py`:
1. Keep the existing `valid_hadm_id(hadm_id)` check, `_fetch_note_row(hadm_id)` fetch, and `_unknown_if_absent(hadm_id)` handling when `not text`.
2. Parse `text` directly with `parse_note(text)` (do not call `_chunk_texts_for`, which truncates multi-chunk medication lists at `picks[0]`).
3. Define two section tuples:
   - `_DEDICATED_MED_SECTIONS = ("discharge_medications", "medications_on_transfer", "discharge_instructions")`
   - `_FALLBACK_MED_SECTIONS = ("assessment_and_plan", "brief_hospital_course", "medications_on_admission", "history_of_present_illness", "discharge_summary")`
4. Collect passages in document reading order (using `_norm_body` exact-body deduplication and skipping empty or `redaction_profile(body).is_placeholder_only` bodies):
   - **Primary pass:** If the note contains at least one non-empty section in `_DEDICATED_MED_SECTIONS`, emit all matching sections from `_DEDICATED_MED_SECTIONS` in document order (with per-section-name ordinals `1, 2, ...` and `"retrieval": "deterministic"`).
   - **Fallback pass:** If the note has NO non-empty sections in `_DEDICATED_MED_SECTIONS` (for example, surgical notes where discharge prescriptions are written at the end of `COURSE ON ADMISSION` or headingless notes), emit all matching non-empty sections from `_FALLBACK_MED_SECTIONS` in document order (including preamble as `history_of_present_illness` if present, or the full note as `discharge_summary` if `parsed.sections` is empty).
5. If `passages` is still empty after both passes, return the existing empty/no-medication payload (`_no_medication_sections_payload(hadm_id, note_id)`). Otherwise return `{"hadm_id": hadm_id, "query": "discharge medications", "returned": len(passages), "passages": passages}`.

---

## Step 2: Tighten `MEDS_FORMATTING` to Separate Discharge Meds from Inpatient-Only Drugs (`services/agent/engine/prompts.py`)
Update `MEDS_FORMATTING` in `services/agent/engine/prompts.py`:
1. Instruct the model to extract and present **only the patient's discharge medications** (medications prescribed, tapered, resumed, or continued at discharge, including dose, route, frequency, or indication when stated in the passages).
2. Explicitly instruct:
   - **Exclude inpatient-only medications:** When extracting from a hospital course or narrative section, do NOT list medications administered solely during the inpatient stay or perioperatively (such as inpatient IV analgesia, intraoperative agents, or inpatient-only steroids) unless the passage explicitly states the patient was discharged on them.
   - **Honest absence handling:** If the provided passages contain no discharge medications (for example, a narrative note that only mentions inpatient IV treatments or contains no medication mentions at all), state clearly in one sentence that no discharge medications are documented in the clinical note (citing the passage if it explicitly states none, or omitting markers if none apply).
   - **Citation rules:** Cite the supporting passage using its `cite` integer (`^[n]`) before the period, and rely on run-collapsing so consecutive sentences from the same passage carry a single marker at the end of the run.

---

## Step 3: Remove Section-Intent Override on Valid Passage Citations (`services/agent/presentation/a2ui.py`)
Inspect `resolve_source` (and any caller in `services/agent/presentation/a2ui.py`) where a valid cited passage (`1 <= idx <= len(passages)`) is rejected or overwritten with `"not available"` / `Not Available` because its `section` key does not match `intent_sections(...)` (e.g., when a `/meds` or free-text medication answer cites `brief_hospital_course` `^[1]`):
1. Ensure that whenever `^[n]` indexes a valid passage returned by the tool (`1 <= n <= len(passages)`), `resolve_source` resolves to that actual passage (`passages[n - 1]`) and its true section label (e.g., `Brief Hospital Course`), NEVER overwriting a real cited passage with `"not available"`.
2. Preserve `"not available"` handling ONLY for cases where no passages were returned (`len(passages) == 0`) or the index is out of bounds.

---

## Step 4: Mandatory Unit Tests & Live Verification (`tests/`)
1. **Unit Tests:**
   - Test `_search_medication_sections`:
     - Returns all `_DEDICATED_MED_SECTIONS` without `picks[0]` truncation when present.
     - Falls back to `_FALLBACK_MED_SECTIONS` (`brief_hospital_course`, headingless `discharge_summary`, etc.) when dedicated medication sections are absent.
   - Test `resolve_source` in `a2ui.py`:
     - Verifies that a medication query citing a passage with `section="brief_hospital_course"` resolves to that passage (`Brief Hospital Course`, `kind="note"`) rather than `"not available"`.
2. **Full Test Suite:** Run `python -m pytest tests -m "not live" -q` and ensure 100% pass.
3. **Live Verification on `90000069` (`MT-2113-DS`):**
   - Run `_search_medication_sections(90000069)` and confirm it returns `brief_hospital_course` via the fallback pass.
   - Run the `/meds` turn on `90000069` and verify that:
     1. It lists **Medrol**, **Darvocet**, and **Flexeril** as the discharge medications.
     2. It **excludes** inpatient-only **IV morphine** and **Decadron**.
     3. The citation pill `^[1]` resolves cleanly to `Brief Hospital Course` (`kind=note`, blue pill) rather than `Not Available`.