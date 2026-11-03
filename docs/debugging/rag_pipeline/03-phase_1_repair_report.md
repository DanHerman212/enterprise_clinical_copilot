# Phase 1 Report — RAG Chunking Repair & Free-Text Prompt Contract

**Date:** 2026-11-03
**Directive:** `02-phase_1_rag_repair.md`
**Files changed:** `services/mcp/retrieval/chunking.py`, `services/agent/engine/prompts.py`,
plus four test files. `services/mcp/tools/retrieval.py` was **not** modified — see
"Why `_section_for_query` was left alone".

---

## 1. `chunk_note` is now lossless at the document's edges

`parse_note` names only the text that follows a recognised heading, so two kinds of note
text produced no chunk at all:

| case | before | after |
|---|---|---|
| text before the first heading | dropped | chunked as `history_of_present_illness`, the same name and text `_search_sections` gives it |
| a note with no recognised heading | `[]` — no chunks, unreachable by any query | the whole note as one `discharge_summary` passage |

Both mirror `_search_sections` (`retrieval.py:650-673`) exactly, including the
redaction-only guard, so the index and the `/ehr` path now read a note the same way.

A preamble takes ordinal 1 of `history_of_present_illness` and a real section of the same
name takes 2, so ids stay unique. Measured on the cohort: **0 of 87 notes have a
preamble**, so no existing datapoint id moves.

## 2. Duplicate diagnosis bodies

A note that prints one list under `ADMISSION DIAGNOSES` and again under `DISCHARGE
DIAGNOSES` now indexes it once, under the discharge heading — the outcome
`_keep_final_diagnosis` produces on the serving side. The comparison folds whitespace and
case (`_norm_body`, the same rule `retrieval._norm_body` applies).

Narrow by construction: only that pair, and only when the bodies agree. A list revised
during the stay is two lists and both are indexed — the difference is a finding.

## 3. `INDEX_SECTIONS` — 13 → 22

Added: `major_procedure`, `complications`, `consults`, `assessment_and_plan`,
`followup_instructions`, `chief_complaint`, `allergies`, `review_of_systems`,
`admission_diagnosis`.

`pertinent_results` stays out (line-per-analyte noise the structured lab tool answers
better). Text outside the whitelist fell from **30,055 chars to 7,859**, and 7,094 of the
remainder is `pertinent_results`.

Every name is a canonical key in `KNOWN_HEADINGS` (the module raises at import otherwise),
and every name already has both presentation labels — `test_section_labels.py` holds
`_SECTION_LABELS` and `_PILL_SECTION_LABELS` to the full registry, and it passes unmodified.

## 4. `SYSTEM_PROMPT`

- The `rag_search_sections` description was the pre-doc-00 contract ("one passage per
  summary section … does NOT return the discharge medications section"). Replaced with
  what the tool does: every recognised section, whole, in document order, plus
  `discharge_summary` for a headingless note. The `/meds` boundary is restated as being
  about a bare request for the medication LIST, not about the section being unavailable.
- The legacy labels (`**Clinical course.**`, `**Discharge diagnosis.**`, `**Hospital
  course.**`) are replaced by the `EHR_FORMATTING` set: `**Presentation & diagnosis.**`,
  `**Hospital course & procedures.**`, `**Discharge plan & medications.**`,
  `**Readmission risk & trajectory.**` (the last only when `predict_readmission` ran).
  `**Clinical course.**` survives as the outpatient label, which is what it was for.
- The placement rule no longer says "cite each passage in returned-list order: ^[1], ^[2]
  …". It says each claim carries the marker of the passage that supports it, taken from
  that passage's `cite_header`, inside the sentence — and no trailing run of markers.
  This is `EHR_FORMATTING`'s rule, in the free-text prompt.
- Also corrected (`TOOL USE`): "There is no separate … 'Admission diagnosis' … section to
  find" was already false — `admission_diagnosis` is a parsed, returned, and now indexed
  section. The sentence now says to read it when the note has one.

Two existing tests pinned the old labels and were retargeted, not weakened:
`test_prompt_contract.py::test_the_summary_headings_are_conditional_on_an_inpatient_stay`
(now asserts the four new labels and that the legacy one is *absent*), and
`test_ux_restoration.py::test_the_ehr_instruction_matches_the_summary_style` (now asserts
every label of one prompt appears in the other).

---

## Cohort measurement (`chunk_note` + `INDEX_SECTIONS`, production parameters)

`python scripts/agent/count_hybrid_chunks.py` — the pipeline's own counting path, over
`hybrid_notes` joined to `hybrid_split` where `split_name='test'`:

| | before | after |
|---|---|---|
| admissions with ≥ 1 chunk | 84 / 87 | **87 / 87** |
| total chunks | 438 | **582** |
| chars outside the whitelist | 30,055 | 7,859 |

The three previously invisible admissions:

| admission | note chars | chunks before | chunks after |
|---|---|---|---|
| 90000026 | 1,153 | 0 | 1 (`discharge_summary`) |
| 90000051 | 1,447 | 0 | 1 (`discharge_summary`) |
| 90000073 | 1,758 | 0 | 1 (`discharge_summary`) |

Per-section counts after the change:

```
pack_to=700 TOTAL chunks: 582
unique ids: 582
by section: {'chief_complaint': 11, 'discharge_diagnosis': 83, 'major_procedure': 39,
'brief_hospital_course': 125, 'history_of_present_illness': 47, 'physical_exam': 24,
'discharge_condition': 24, 'discharge_medications': 45, 'allergies': 11,
'followup_instructions': 20, 'discharge_instructions': 39, 'admission_diagnosis': 28,
'complications': 7, 'past_medical_history': 15, 'family_history': 5, 'social_history': 6,
'review_of_systems': 4, 'assessment_and_plan': 10, 'discharge_summary': 16, 'consults': 9,
'discharge_disposition': 14}
EXPECTED_VECTORS=582
```

---

## Required in Phase 2 (outside this directive's allowlist)

1. **`services/mcp/rag_config.yaml`**, `corpus.demo.expected_vectors: 446` → **582**.
   `eval_ingest` and `build_index` hard-fail on a vector-count mismatch, and 446 is stale
   even against the pre-change count of 438.

2. **Rebuild and deploy.** Code first, then the index: every id the new chunker emits is
   additive (no preamble exists in the cohort, so no existing id moves), so a code-only
   window degrades nothing.

   ```bash
   # chunk -> embed -> eval -> build index -> deploy, per rag_config.yaml
   .venv/bin/python -m services.mcp.pipelines.rag_ingest_pipeline submit
   ```

   Env that legitimately varies per run: `RAG_CORPUS=demo`, `PREVIOUS_INGEST_URI` (leave
   unset for a clean embed — the reuse path verifies the vector space, and the chunk set
   changed), `EMBED_WORKERS`.

   The pipeline ends by deploying the new endpoint; `scripts/agent/rag_endpoint_status.py`
   and `scripts/agent/verify_rag_query.py` confirm the deployed index answers.

---

## Why `_section_for_query` was left alone

`_search`'s anchored retry **replaces** the original hit set when it returns anything
(`retrieval.py:343-352`). Anchoring a query word like "procedure" to `major_procedure`
would therefore substitute that section's passages for a result that may have been
better — and the operation is often narrated in the hospital course as well, which is
the overlap the anchor cannot distinguish. Retrievability was the defect, and the
expanded whitelist fixes it at the index. No keyword was added.

## Test gate

`pytest tests -m "not live"` → **1771 passed, 1 skipped, 44 deselected**.
`test_layer_separation.py` passes unmodified.

New tests: `test_chunking.py` — headingless note, preamble, preamble/H.P.I. ordinal
collision, redaction-only preamble, identical diagnosis bodies, revised diagnosis bodies,
admission diagnosis without a discharge one. `test_rag_search.py` — every name in
`INDEX_SECTIONS` round-trips through `_parse_datapoint_id`.
