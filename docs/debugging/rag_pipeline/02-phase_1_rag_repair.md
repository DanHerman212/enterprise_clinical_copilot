# Engineering Directive: Repair RAG Chunking (`chunking.py`) & Free-Text Prompt Contract (`prompts.py`)

## 0. Governance & Strict Boundaries
1. **Strict File Allowlist:** You may ONLY modify the following production files (plus unit test files under `tests/`):
   - `shared/chunking.py` (or wherever `chunk_note` and `INDEX_SECTIONS` live)
   - `services/agent/engine/prompts.py`
   - `services/mcp/tools/retrieval.py` (ONLY if `_section_for_query` needs a small keyword mapping update for newly indexed sections like `major_procedure` or `complications`; do not alter `_search_sections` or `_search_medication_sections`)
   Run `git status` and `git diff --name-only` before finishing to verify no other production files were touched.
2. **Architectural & Layer Separation Invariants:**
   - `INDEX_SECTIONS` must remain the single source of truth shared between build-time chunking (`chunk_notes.py` / `build_chunks.py`) and serving-time datapoint ID parsing (`retrieval.py`).
   - Ensure `tests/agent/test_layer_separation.py` passes unmodified.
3. **Zero Patient-Specific Logic:** Do not hardcode `hadm_id` values (`90000026`, `90000051`, `90000073`, etc.) or specific note strings.

---

## Step 1: Make `chunk_note` Lossless for Headingless Notes & Preambles (`chunking.py`)
Match the lossless representation already used by `_search_sections` in `retrieval.py`:
1. **Headingless Notes (`parsed.sections == ()`):** When `parse_note(text).sections` is empty and the stripped note text is non-empty, `chunk_note` must chunk the note body under `section="discharge_summary"` (instead of returning `[]`), so admissions with headingless notes (`90000026`, `90000051`, `90000073`) produce indexed chunks rather than 0 chunks.
2. **Unheaded Preamble:** If non-empty text precedes `parsed.sections[0].start`, emit it under `section="history_of_present_illness"` (matching `retrieval.py:650-654`) so no leading narrative is silently dropped.
3. **Identical Diagnosis Body Deduplication:** When `admission_diagnosis` and `discharge_diagnosis` have identical stripped bodies in the same note, skip the duplicate `admission_diagnosis` body (matching `_keep_final_diagnosis` in `retrieval.py`) so the vector index does not embed two identical copies of the same diagnosis list.

---

## Step 2: Expand `INDEX_SECTIONS` to Include Narrative Clinical Sections (`chunking.py`)
1. Expand `INDEX_SECTIONS` to include the clinical narrative sections currently dropped at build time:
   - `major_procedure`
   - `assessment_and_plan`
   - `followup_instructions`
   - `chief_complaint`
   - `complications`
   - `consults`
   - `allergies`
   - `review_of_systems`
   - `admission_diagnosis` (for notes where it differs from `discharge_diagnosis`)
   Keep `pertinent_results` excluded (as deliberate lab-line noise exclusion).
2. Verify that `_SECTION_RE` and `_parse_datapoint_id` in `retrieval.py` cleanly parse every section name in the expanded `INDEX_SECTIONS` tuple (especially multi-underscore names like `assessment_and_plan`, `followup_instructions`, `review_of_systems`, `history_of_present_illness`).

---

## Step 3: Update `SYSTEM_PROMPT` Free-Text Summarization & Tool Contract (`services/agent/engine/prompts.py`)
In `SYSTEM_PROMPT` (`services/agent/engine/prompts.py`, around lines 339–390):
1. **Accurate `rag_search_sections` Description:** Replace the stale pre-doc-00 claim that `rag_search_sections` only returns four sections and *"does NOT return the discharge medications section"*. State accurately that `rag_search_sections` returns all recognized clinical sections of the discharge summary whole and in document order (including `discharge_medications`, `major_procedure`, `complications`, etc.), or `discharge_summary` as a single passage for headingless notes.
2. **Align Free-Text Summary Headings with `/ehr`:** Replace the legacy two-heading instruction (`**Hospital course.**` / `**Clinical course.**` and `**Discharge diagnosis.**`) with the standard clinical summary headings from `EHR_FORMATTING`:
   - `**Presentation & diagnosis.**`
   - `**Hospital course & procedures.**`
   - `**Discharge plan & medications.**`
   (Plus `**Readmission risk & trajectory.**` when `predict_readmission` was called on that turn.)
3. **Consistent Citation Placement Rule:** Ensure the citation placement instructions in `SYSTEM_PROMPT` match `EHR_FORMATTING` (cite the supporting passage `^[n]` at the end of the sentence or claim it supports; do not force unnatural "cite each passage once in returned list order" prose).

---

## Step 4: Verification & Dry-Run Audit of the 87-Note Cohort
1. Add/update unit tests in `tests/` covering:
   - `chunk_note` on a headingless note emits `section="discharge_summary"` chunks.
   - `chunk_note` on a note with unheaded preamble text emits `section="history_of_present_illness"`.
   - `chunk_note` deduplicates identical `admission_diagnosis` / `discharge_diagnosis` bodies while keeping distinct ones.
   - `_parse_datapoint_id` round-trips every section name in the expanded `INDEX_SECTIONS`.
2. Run `python -m pytest tests -m "not live" -q` and ensure 100% pass.
3. Run `chunk_note` (filtered by `INDEX_SECTIONS`, as `chunk_notes.py` / `build_chunks.py` does) over the 87-admission cohort and report:
   - How many of the 87 admissions now have $\ge 1$ chunk (must be **87 / 87**, with `90000026`, `90000051`, `90000073` no longer at 0).
   - Total chunk count before vs. after across the 87 notes.
   - The exact CLI command or pipeline step required to rebuild and deploy the updated chunks/embeddings to the vector index in Phase 2.