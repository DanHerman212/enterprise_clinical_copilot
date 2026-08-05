# Evidence Gate — Session Summary, 2026-08-05

## Goal

Answer one question before spending anything on Vector Search infrastructure:

> Do the discharge notes contain readmission-risk signal that the structured
> model does not already have?

Method: run the A1–A4 evidence gate from `RAG_BUILD_GUIDE.md` §2 — parse note
sections, score the test split, tag risk concepts with negation handling, and
measure **orthogonal lift**: among admissions the model called low-risk, do
concept rates differ between patients who were readmitted anyway (FN) and those
who were not (TN)?

## Operations executed

| Step | Deliverable | Outcome |
|---|---|---|
| Cache | `rag/notes.py`, `scripts/fetch_note_cache.py` | 33,929 test-split notes (378M chars) cached at `~/.cache/` (outside iCloud), one BigQuery scan |
| A1 | `scripts/probe_sections.py` → `docs/probes/sections_probe.json` | Parser validated on full corpus; per-section redaction measured |
| A2 | `scripts/score_test_split.py` → `readmission.test_predictions` | 49,103 admissions scored locally via the production `ReadmissionPredictor`; anchor check reproduced the endpoint-verified 0.131398 exactly |
| A3 | `rag/concepts.py`, `tests/data/concept_sentences.json` (43 labeled sentences), `.venv-nlp` | 48/48 labeled tests pass; negated/hypothetical/family mentions correctly excluded |
| A4 | `scripts/probe_note_signal.py` → `docs/probes/note_signal_probe.json` | Full corpus tagged (46.5 min); lift + coverage measured |

Supporting facts established along the way:

- **Section parsing is reliable**: coverage 0.9998, zero unparsed notes. The
  old "95.5% of Social History redacted" figure was corrected — the truth is
  **98.2% placeholder-only**: de-identification removed entire sections, not
  just names. Narrative sections (Brief Hospital Course, HPI, Discharge
  Instructions) are nearly fully intact.
- **The dry-run model at threshold 0.12 is recall-heavy**: it flags 73% of
  admissions to catch ~90% of readmissions. All results are conditional on
  this model (`model_version` recorded in the predictions table).

## Results

Quadrants among noted test admissions: FN 758, TN 7,656, TP 6,834, FP 18,681.

Lift = concept rate in FN ÷ rate in TN (both below threshold):

| Concept | FN rate | TN rate | Lift | FN events |
|---|---|---|---|---|
| polypharmacy | 0.79% | 0.37% | 2.16 | 6 |
| missed_followup | 0.66% | 0.43% | 1.53 | 5 |
| functional_decline | 15.6% | 14.9% | 1.05 | 118 |
| non_adherence | 3.3% | 3.2% | 1.04 | 25 |
| goals_of_care | 4.6% | 4.5% | 1.02 | 35 |
| hedging (non-gating) | 9.6% | 10.2% | 0.94 | 73 |
| substance_use | 4.4% | 6.9% | 0.63 | 33 |
| ama | 0.26% | 0.63% | 0.42 | 2 |

Coverage: 35.6% of all notes carry ≥1 gating concept (26.0% of FN notes).

## Interpretation

**The gate, as designed, did not pass.**

- The only lifts above 1.5 rest on 5–6 events — statistical noise.
- The decisive row is **functional_decline**: 15% base rate, 118 FN events —
  well-powered — and lift 1.05. Patients whose notes say "lives alone" or
  "requires assistance" were not readmitted more than their low-risk peers.
- Coverage is fine; the demo would have things to *show*. What the probe could
  not find is evidence the concepts *predict* beyond the structured features.

Caveats that keep this from being a final verdict:

1. **Recall is unmeasured.** The phrase lists are narrow seeds. If they catch a
   minority of true mentions, real lift is diluted toward 1.0.
2. **Binary note-level flags are the crudest proxy.** The RAG system would
   operate on passages and semantics, not eight booleans.
3. **30-day all-cause readmission may be substantially unforeseeable** (new
   acute illness). If so, there is no note signal to find — for anyone.

## Optional next step (recommended)

**Read the FN notes before ruling.** Sample ~20 of the 758 model-missed
admissions (notes and tags are already cached) and read what preceded each
readmission.

- Notes visibly contain warning signs our concepts missed → recall problem →
  revise the tagger, re-run A4 once.
- Notes look genuinely unremarkable → the null is real, and the A5 decision is
  made on certainty instead of a judgment call.

Cost: an hour of reading. No cloud spend.

## Potential alternative strategies

| Strategy | Idea | Trade-off |
|---|---|---|
| Reframe the RAG value proposition | Notes as *grounded explanation + cited intervention plan* (already in `rag_requirements.md`), dropping the claim that they improve prediction | Honest and immediately buildable, but a material change to the project thesis — owner's call |
| Refine the measurement | Continuous probability margin instead of hard quadrants; passage-level instead of note-level flags | More rigor, more time; risks becoming a search for a wanted result |
| Better representation | Test embedding-based signal directly (e.g., logistic head on note embeddings vs FN/TN) instead of hand-built concepts | Closest to what RAG actually does; requires embedding spend before the gate has passed |
| Different label | Target a subset of readmissions notes plausibly predict (e.g., non-elective, same-diagnosis) | Smaller cells; needs care to avoid label shopping |

## Status

- All work uncommitted (rag/, scripts/, tests/, probes/, this doc).
- Prediction endpoint: torn down (no hourly billing). Vector Search: not built.
- The A5 pass/fail decision is open and belongs to the project owner.
