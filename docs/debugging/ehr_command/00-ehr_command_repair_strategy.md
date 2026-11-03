# Engineering Directive: Lossless `/ehr` Section Retrieval & 4-Section Summary Contract

## 0. Governance & Strict Engineering Rules (Read First)
1. **Strict File Allowlist:** You may ONLY modify the following production files (plus unit test files under `tests/`):
   - `services/mcp/retrieval/sections.py`
   - `services/mcp/tools/retrieval.py`
   - `services/agent/engine/deterministic.py`
   - `services/agent/engine/prompts.py`
   - `services/agent/presentation/a2ui.py`
   Run `git status` and `git diff --name-only` before finishing to verify no other production files were touched.
2. **Out-of-Scope Freeze:** DO NOT touch `_readmission_comparators`, `derive_target`, `extract_comparators`, `_TARGET_SECTION_ORDER`, `search_literature`, `predict_readmission`, `discharge_medications` (`/meds`), `get_lab_trends` (`/labs`), `citations.py`, `guardrail.py`, `followups.py`, or the `rag_search` vector search path.
3. **Zero Hardcoded Patient IDs or Clinical Band-Aids:** Never branch on a `hadm_id` or `note_id`, and never hardcode clinical text for a single patient. Every change must work globally across all 87 notes in the cohort.
4. **Pure Stdlib Constraint on `sections.py`:** `services/agent/Dockerfile` copies `services/mcp/retrieval/sections.py` into the agent container. `sections.py` MUST only import Python standard library modules (`re`, `dataclasses`, `typing`).
5. **No Dead Code:** Delete `_ordered_summary_payload` in `deterministic.py` rather than commenting it out or leaving it unused.

---

## Step 1: Global Cohort Heading Audit & `KNOWN_HEADINGS` Completion (`services/mcp/retrieval/sections.py`)
1. Run a script against all 87 notes in `readmission.hybrid_notes` (or the local note cache) using `parse_note(text)`.
2. Inspect every entry in `unknown_headings` across all 87 notes and any note where `coverage < 1.0`.
3. Update `KNOWN_HEADINGS` in `services/mcp/retrieval/sections.py` so every genuine clinical section heading across all 87 notes maps to a canonical section constant (e.g., `"Operative Procedures"`, `"Operative Procedure"`, `"Complications"`, preoperative/postoperative diagnoses, etc., while keeping true inline sub-labels like `"Marital status"` or `"Tobacco"` out of `KNOWN_HEADINGS` if they split a section mid-body—EXCEPT when a heading like `"COMPLICATIONS: None."` is a top-level section between `OPERATIVE PROCEDURES` and `COURSE ON ADMISSION`).
4. Separate `ADMISSION_DIAGNOSIS = "admission_diagnosis"` from `DISCHARGE_DIAGNOSIS = "discharge_diagnosis"` in `KNOWN_HEADINGS` so admission/preoperative diagnoses and discharge/postoperative/final diagnoses do not collide under the same canonical key.

---

## Step 2: Make `_search_sections` Lossless (`services/mcp/tools/retrieval.py`)
Rewrite `_search_sections(hadm_id: int)` in `services/mcp/tools/retrieval.py`:
1. Keep the existing `valid_hadm_id(hadm_id)` check, `_fetch_note_row(hadm_id)` fetch, and `_unknown_if_absent(hadm_id)` handling when `not text`.
2. Parse `text` directly with `parse_note(text)`. **Do not call `_chunk_texts_for`, do not filter by `SUMMARY_SECTIONS`, and do not truncate to `picks[0]`.**
3. Build the ordered `passages` list preserving 100% of the note's informative text in natural reading order:
   - If `parsed.sections` is non-empty:
     - Check if any unheaded preamble text precedes `parsed.sections[0].start` (`preamble = normalised[:parsed.sections[0].start].strip()`). If `preamble` is non-empty and `not redaction_profile(preamble).is_placeholder_only`, emit it first with `section="history_of_present_illness"` and `id=f"{note_id}_history_of_present_illness_1"`.
     - Iterate through `parsed.sections` in order, maintaining a per-section-name ordinal counter (`1, 2, ...`). Skip a section only if `not section.body` or `redaction_profile(section.body).is_placeholder_only`.
     - Append `{"id": f"{note_id}_{section.name}_{ordinal}", "section": section.name, "text": section.body, "retrieval": "deterministic"}`.
   - If `parsed.sections` is empty (a headingless narrative note):
     - If `text.strip()` is non-empty and `not redaction_profile(text).is_placeholder_only`, return a single passage: `[{"id": f"{note_id}_discharge_summary_1", "section": "discharge_summary", "text": text.strip(), "retrieval": "deterministic"}]`.
4. Return `{"hadm_id": hadm_id, "query": "discharge notes", "returned": len(passages), "passages": passages}` (or the existing empty-note payload if `passages` is empty).

---

## Step 3: Preserve Natural Reading Order in `/ehr` (`services/agent/engine/deterministic.py`)
1. Delete `_ordered_summary_payload` from `services/agent/engine/deterministic.py`.
2. In `run_command`, pass `order=None` to `_execute_tool_calls` so passages reach the formatting model and the citation ledger in the exact document order returned by `_search_sections`.

---

## Step 4: Update `EHR_FORMATTING` to the 4-Section Global Contract (`services/agent/engine/prompts.py`)
Update `EHR_FORMATTING` in `services/agent/engine/prompts.py`:
1. Specify these exact four section labels, in this order:
   - `**Presentation & diagnosis.**` for an inpatient admission: summarize the presenting symptoms, history of present illness, relevant imaging or diagnostic workup, and the admission and discharge diagnoses documented in the passages. (Where the tool result carries the outpatient encounter caveat, write `**Clinical course.**` instead and omit the other three sections.)
   - `**Hospital course & procedures.**` only if the passages describe the hospital course, procedures, or inpatient treatment: summarize all operative or invasive procedures performed, complications (or documented absence of complications), inpatient treatments and medications administered (such as IV medications or steroids), and symptom progression during the stay.
   - `**Discharge plan & medications.**` only if the passages contain discharge medications, disposition, restrictions, or follow-up instructions: summarize the discharge disposition, prescribed discharge medications, activity restrictions, and follow-up plan.
   - `**Readmission risk & trajectory.**` only where a readmission-risk estimate was given to you. (Keep all existing instructions for this fourth section—`probability_display`, `threshold_display`, qualitative driver explanations, and `<verified_comparators>` `^[PMID: ...]` citation rules—completely unchanged.)
2. At the bottom of `EHR_FORMATTING`, replace `"Write nothing about medications, and append no disclaimer about missing information."` with `"Append no disclaimer about missing information."`

---

## Step 5: Register All Section Labels (`services/agent/presentation/a2ui.py`)
Update `_SECTION_LABELS` and `_PILL_SECTION_LABELS` in `services/agent/presentation/a2ui.py` so every canonical section constant in `services/mcp/retrieval/sections.py` has an explicit human-readable entry in `_SECTION_LABELS` and a Title Case entry in `_PILL_SECTION_LABELS`.

---

## Step 6: Mandatory Harness Invariant Tests (`tests/`)
Add permanent unit tests that enforce:
1. **Label Parity Invariant:** `set(KNOWN_HEADINGS.keys()) <= set(_SECTION_LABELS.keys())` and `set(KNOWN_HEADINGS.keys()) <= set(_PILL_SECTION_LABELS.keys())`.
2. **Lossless `_search_sections` Invariant:** When `_search_sections` runs on a multi-section note (including duplicate diagnosis headers, `OPERATIVE PROCEDURES`, `COMPLICATIONS`, and `COURSE ON ADMISSION`, as well as a headingless narrative note), zero informative sections are dropped, duplicate section names are both preserved with incrementing ordinals (`_1`, `_2`), and the returned passage order matches the document's top-to-bottom reading order.
3. **Full Test Gate:** Run `python -m pytest tests -m "not live" -q` and ensure 100% of tests pass.