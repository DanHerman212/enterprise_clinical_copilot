# Engineering Directive: Global Passage Deduplication, Paragraph Citation Collapsing & Grounding Precision

## 0. Governance & Strict Engineering Rules (Read First)
1. **Strict File Allowlist:** You may ONLY modify the following production files (plus unit test files under `tests/`):
   - `services/mcp/tools/retrieval.py`
   - `services/agent/postprocess/citations.py`
   - `services/agent/engine/prompts.py`
   Run `git status` and `git diff --name-only` before finishing to verify no other production files were touched.
2. **Architectural & Layer Separation Invariants:**
   - Preserve the strict package DAG in `services/agent/` (`ingress` -> `routing` -> `engine` -> `postprocess` -> `presentation` -> `telemetry`). `tests/agent/test_layer_separation.py` must pass without modification.
   - Do not modify `sections.py`, `deterministic.py`, `a2ui.py`, `guardrail.py`, `followups.py`, comparator/literature functions, `/meds`, `/labs`, or the `rag_search` vector search path.
3. **Zero Patient-Specific Logic:** No hardcoded `hadm_id` or `note_id` branches, and no clinical string matching specific to a single record. All logic must operate purely on structural and textual invariants across all 87 cohort notes.
4. **Code Quality:** Keep functions small, pure, and side-effect-free. Do not leave commented-out code or unused helpers.

---

## Step 1: Exact-Body Passage Deduplication (`services/mcp/tools/retrieval.py`)
In `_search_sections(hadm_id: int)` inside `services/mcp/tools/retrieval.py`, eliminate byte-identical duplicate sections within the same note (e.g., when a note repeats the exact same text under both `ADMISSION DIAGNOSES` and `DISCHARGE DIAGNOSES`, or `PREOPERATIVE DIAGNOSIS` and `POSTOPERATIVE DIAGNOSIS`):
1. Define a helper `_norm_body(text: str) -> str` that returns `" ".join(text.split()).lower()`.
2. While iterating through `parsed.sections` to construct `passages`, maintain `seen_bodies: dict[str, int] = {}` mapping each normalized body string to its index in `passages`.
3. When processing a section whose normalized body is already in `seen_bodies`:
   - Do **not** append a duplicate passage.
   - If the existing passage at `passages[existing_idx]["section"]` is `"admission_diagnosis"` and the current duplicate section is `"discharge_diagnosis"`, upgrade the existing passage in place to `section = "discharge_diagnosis"` (recomputing its `id` with the proper ordinal for `"discharge_diagnosis"`), so the canonical final diagnosis label is retained in the single deduplicated passage.
4. Preserve the exact document reading order for all unique passages.

---

## Step 2: Consecutive Same-Source Citation Collapsing (`services/agent/postprocess/citations.py`)
Update `collapse_repeated_citations(text: str) -> str` in `services/agent/postprocess/citations.py` so consecutive sentences within the **same paragraph** that carry the **exact same citation marker(s)** do not stamp duplicate inline pills on back-to-back sentences, while keeping the final marker of the single-source run:
1. Operate independently on each paragraph (`text.split("\n\n")`). Never collapse citations across paragraph breaks.
2. First, preserve existing within-sentence deduplication (if a single sentence repeats `^[4]` multiple times, keep only one `^[4]` at the end of that sentence).
3. Second, across consecutive cited sentences/clauses within the same paragraph:
   - If sentence $i$ ends with citation marker group $C_i$ (e.g., `^[4]`) and the immediately following sentence $i+1$ in the same paragraph has the **exact same** normalized citation marker set $C_{i+1} == C_i$ (and no intervening different citation marker appears between them), remove $C_i$ from the end of sentence $i$ (cleaning up any trailing space before the sentence-ending punctuation) and retain $C_{i+1}$ on sentence $i+1$.
   - Example:
     - Input: `"Postoperatively, the patient had shoulder pain managed with IV morphine ^[4]. The patient also had slight dysphagia treated with Decadron ^[4]."`
     - Output: `"Postoperatively, the patient had shoulder pain managed with IV morphine. The patient also had slight dysphagia treated with Decadron ^[4]."`
   - If sentence $i$ has `^[3]` and sentence $i+1$ has `^[4]`, both `^[3]` and `^[4]` MUST be preserved.
   - If sentence $i$ has `^[4]`, sentence $i+1$ has `^[3]`, and sentence $i+2$ has `^[4]`, all three (`^[4]`, `^[3]`, `^[4]`) MUST be preserved because the run of `^[4]` was interrupted by `^[3]`.
   - Do not collapse `[PMID: ...]` markers across sentences if each sentence describes a distinct cohort study; only collapse identical note-section citations (`^[n]`).

---

## Step 3: Tighten Passage Attribution in `EHR_FORMATTING` (`services/agent/engine/prompts.py`)
In `EHR_FORMATTING` (`services/agent/engine/prompts.py`), tighten the citation attribution instruction so the model does not cite a short diagnosis-list passage for narrative history facts that only exist in the hospital course / HPI passage:
1. In `**Presentation & diagnosis.**`, clarify:
   - Cite the exact passage (`^[n]`) where each specific detail is stated: if presenting history, symptom duration, prior outpatient treatments, or imaging findings come from a narrative passage (such as `history_of_present_illness` or `brief_hospital_course`) rather than a diagnosis-list passage, cite that narrative passage for those sentences and cite the diagnosis passage (`discharge_diagnosis` or `admission_diagnosis`) specifically for the formal diagnoses.

---

## Step 4: Mandatory Unit Tests & Live Verification (`tests/`)
1. **Deduplication Tests (`tests/mcp/test_rag_search.py`):**
   - Add a test verifying that when a note contains byte-identical `ADMISSION DIAGNOSES` and `DISCHARGE DIAGNOSES` sections, `_search_sections` returns a single deduplicated `discharge_diagnosis` passage rather than two identical passages.
   - Add a test verifying that when `ADMISSION DIAGNOSES` and `DISCHARGE DIAGNOSES` have *different* text, both passages are preserved (`admission_diagnosis` and `discharge_diagnosis`).
2. **Paragraph Citation Collapse Tests (`tests/agent/test_citations.py` or equivalent):**
   - Test that back-to-back sentences with the same `^[n]` in the same paragraph collapse to a single `^[n]` on the last sentence of the run.
   - Test that an alternating run (`^[3]` -> `^[4]` -> `^[3]`) preserves all three markers.
   - Test that identical `^[n]` markers in *different* paragraphs (`\n\n`) are both preserved.
   - Test that `[PMID: ...]` markers are never stripped by consecutive note-citation collapsing.
3. **Full Test Suite:** Run `python -m pytest tests -m "not live" -q` and ensure 100% pass.
4. **End-to-End Verification on `90000069`:**
   - Run `_search_sections(90000069)` and confirm it now returns **4** unique passages (`discharge_diagnosis`, `major_procedure`, `complications`, `brief_hospital_course`), with the byte-identical diagnosis block deduplicated.
   - Run the `/ehr` turn on `90000069` and print the full output to verify:
     1. Presenting symptom duration/prior medication management cites `^[4]` (`brief_hospital_course`), while the formal diagnosis cites `^[1]` (`discharge_diagnosis`).
     2. Each paragraph has clean, non-redundant citations with no consecutive duplicate `^[4]` pills in the same paragraph.