# Hybrid-108 Dataset Quality Map — Notes ↔ Features ↔ Golden Sample

**Date:** 2026-08-23 · **Scope:** full 108-patient hybrid readmission cohort
**Purpose:** step-one verification that every note maps to complete features in the warehouse, and that the dataset contains everything we expect.

---

## 1. Method

Joined five sources per `hadm_id` and re-derived note/feature quality locally:

| Source | Role |
|---|---|
| `eval/results/hybrid_notes.json` | note text + variant (MTSamples sid) |
| `eval/results/hybrid_cohort.json` | 49 model features + probability + band |
| `eval/results/golden_sample_hybrid_108.json` | eval contract (hadm, prob, band) |
| `data/hybrid/provenance.json` | per-feature provenance (parsed vs filled) |
| BigQuery `readmission.hybrid_notes` / `hybrid_features` / `hybrid_split` | warehouse of record |

Per patient, locally computed:
- **is_discharge** — regex for discharge-specific content (discharge diagnoses/medications/instructions/condition/disposition/summary, brief hospital course, admission/admitting diagnoses)
- **n_parse_sections** — `parse_note` recognized sections
- **n_whitelist_chunks** — chunks surviving the chunker whitelist (what would be indexed)
- **n_summary_sections** — of the 4 `SUMMARY_SECTIONS` (brief_hospital_course, discharge_diagnosis, discharge_medications, discharge_instructions)
- **grounding** — `_provisional_row` provenance: parsed vs filled features

---

## 2. Mapping completeness — ✅ 108/108

- **All five artifacts agree**: notes=108, cohort=108, golden=108, provenance=108, BQ notes=108, BQ features=108.
- **`hybrid_features` (BQ) == `hybrid_cohort.json`**: **0/108 feature diffs** across all 49 model features. The warehouse is the exact image of the source artifact.
- **Feature completeness**: all 49 model features fully populated for all 108; no NULLs.
- **Only NULL column**: `subject_id` (108/108) — a bookkeeping column never populated for the hybrid cohort; not part of the 49-feature model contract.
- Every hadm_id in the golden sample has a note, a complete feature row, a probability, a band, and provenance. **The mapping is total.**

---

## 3. Note-type quality — the real gaps

### 3a. Not discharge notes: **26/108**
Real MTSamples notes whose document type is not a discharge summary (outpatient therapy/rehab, narrative, consult, obstetrics). Hadm_ids:

`90000010, 90000019, 90000026, 90000028, 90000033, 90000034, 90000039, 90000040, 90000047, 90000050, 90000051, 90000052, 90000054, 90000055, 90000059, 90000064, 90000066, 90000073, 90000075, 90000083, 90000087, 90000094, 90000095, 90000099, 90000101, 90000106`

Most of these still parse and chunk (they get indexed), but their sections are not discharge sections, so they are weak grounding for a discharge-note RAG.

### 3b. Zero whitelisted chunks: **6/108** — ⚠️ exactly the 6 missing from the deployed index
`90000026, 90000041, 90000051, 90000069, 90000073, 90000076`

These produce 0 chunks under the current chunker → never embedded → the exact root cause of the `zero_passage` index gap. **Of these 6, three are genuine discharge notes whose numbered heading format the parser fails on:**
- `90000041` (has `Discharge Summary`, `Recommendations`)
- `90000069` (has `Admission Diagnoses`, `Discharge Diagnoses` numbered lists)
- `90000076` (has `Admitting Diagnoses`, `Discharge Diagnoses` numbered lists)

→ **Parser defect**, independent of the therapy-note question.

### 3c. No summary sections: **10/108** — fail `rag_search_sections` (meds/summarize)
`90000026, 90000028, 90000033, 90000041, 90000051, 90000054, 90000069, 90000073, 90000076, 90000095`

These have none of the 4 summary sections, so section-anchored retrieval returns nothing. This is the second failure mode seen in the eval.

---

## 4. Feature grounding (transparency)

Features are derived from the note where possible and filled from signals/defaults otherwise (documented `story-anchored fill` in `scripts/fill_features.py`). Typical grounding is **4–12 of 23 provenance entries parsed**, the remainder filled. Examples:
- parsed: age, gender, race, med count, LOS, hemoglobin/sodium, prior-admission mentions
- filled: prior_inpatient_days (~14d × prior), insurance (age-based), labs when absent (hgb 12.5, Na 139), rbc family (hgb/3), admission_type (ED→emer else unknown)

This is by design, not a defect, but it means low-grounding notes (esp. the 26 non-discharge notes, which lack most signals) have feature rows that are largely heuristic.

---

## 5. Band distribution

- low 44 · borderline 47 · high 17

---

## 6. Verdict

The **mapping layer is complete and internally consistent**: every note → complete features → golden-sample probability, with the warehouse matching the source artifacts exactly.

The **content layer has two confirmed defects**:
1. **Parser defect (must fix):** real discharge notes `90000041/90000069/90000076` parse to 0 sections → never indexed. `parse_note`'s heading allowlist misses their formats.
2. **Corpus-mix question (product decision):** 26/108 notes are not discharge summaries (therapy/rehab/narrative). Whether they belong in a discharge-note RAG is a data-selection decision, separate from the parser bug.

---

## 7. Artifacts

- Full per-patient mapping table: `projects/agent-harness/data/hybrid/dq_map.csv` *(copy at `/tmp/rag_diag/dq_map.csv`)*
- Regeneration script: `/tmp/rag_diag/dq_map.py`
