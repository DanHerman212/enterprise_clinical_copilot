# RAG Build — Session Update, 2026-08-04

Companion to `RAG_BUILD_GUIDE.md`. The guide describes the destination; this
file records where we actually are and what happens next. Read the "Pick up
here" section first.

---

## Where we are

We are inside **§2 of the build guide: the evidence gate**. Nothing has been
embedded, indexed, or deployed. The gate exists to answer one question before we
spend money on Vector Search:

> Do the discharge notes contain risk signal that the structured model does not
> already have?

If the answer is no, the honest move is to find that out now rather than after
building an index.

---

## What we settled today

### Negation handling — medspaCy (option C)

Verified in a throwaway venv before committing to it. medspaCy 1.3.1 + spaCy
3.8.14 install cleanly on Python 3.12, and ConText classified all five probe
sentences correctly with no tuning:

| Sentence | Expected | Result |
|---|---|---|
| "Patient denies tobacco use." | negated | ✅ |
| "Return to the ED if you miss a dose." | hypothetical | ✅ |
| "Remote history of IV drug use." | historical | ✅ |
| "Father had CHF." | family | ✅ |
| "Patient continues to use tobacco daily." | affirmed | ✅ |

`medspacy.load()` needs no model download — PyRuSH handles sentence splitting.

**Consequence: medspaCy needs its own venv.** It pulls numpy 2.5.1; the harness
venv is pinned at numpy 1.26.4 alongside xgboost and scikit-learn, and
unpickling a model trained under numpy 1.x inside a numpy 2.x process is exactly
the kind of thing that breaks quietly.

```
.venv       numpy 1.26.4   harness, model scoring, MCP server, agent
.venv-nlp   numpy 2.5.1    medspaCy concept tagging (ingestion only)
```

They never share a process; they communicate through BigQuery tables. That
boundary is correct on its own merits — tagging at query time would be slow and
wasteful — so the version conflict costs us nothing architecturally.
`mcp_server/requirements.txt` and the Cloud Run image stay untouched.

### Sample size — full test split

Run the concept probe on all **33,929** test-split admissions with notes, not a
sample. The constraint is not the total; it is the size of the smallest cell in
the 2×2 — the "model said low risk, patient came back anyway" quadrant. Sampling
to ~5,000 shrinks that cell roughly 7×, into the low hundreds, where a modest
but real lift becomes indistinguishable from noise. That is the one ambiguity a
gate must not have.

Still to measure: medspaCy throughput. First step of A4 runs 500 notes, measures
chars/sec, and extrapolates before launching the full run.

### Clinician hedging — keep, but do not let it gate

Scoped to Discharge Condition and Brief Hospital Course, reported in its own
column, excluded from pass/fail.

It is the noisiest concept on the list, and hedging is a tone rather than an
entity. But clinician worry is one of the four things `rag_requirements.md` names
as what notes uniquely contribute, so dropping it outright loses something real.
Reporting it separately lets us learn whether it works without it inflating the
gate. Section scoping matters because Discharge Instructions are saturated with
conditional boilerplate ("call your doctor if you feel worse") that is template
text aimed at the patient, not clinical judgment.

### Concept list

| Concept | Gates? | Section scope |
|---|---|---|
| Non-adherence | ✅ | all |
| Missed / no follow-up scheduled | ✅ | all |
| Left against medical advice | ✅ | all |
| Substance use | ✅ | all |
| Functional decline / needs assistance | ✅ | all |
| Polypharmacy / complex regimen | ✅ | all |
| Goals of care / palliative | ✅ | all |
| Clinician hedging | ❌ reported only | Discharge Condition, Brief Hospital Course |

---

## What we built today

### `rag/sections.py` + `tests/test_sections.py` — A0 complete

22 new tests. Full harness suite: **64 passed**, no regressions.

Parses a discharge note into named sections and reports `coverage` and
`unknown_headings` so a parse failure is countable rather than invisible.

**The design decision to remember:** only headings on an explicit allowlist
split the text. A line that merely *looks* like a heading is recorded in
`unknown_headings` and left in the body where it belongs.

This exists because the earlier approach — "extract from the header to the next
line matching `[A-Z][A-Za-z ]{2,40}:`" — truncated at MIMIC's own sub-fields
(`Marital status:`, `Tobacco:`, `Alcohol:`). A note whose Social History opens
with a redaction placeholder but carries real content below was read as fully
redacted. That is where the **95.5% redaction figure** came from, and why it is
not trustworthy.

The new failure mode is the inverse: a real header we forgot to list gets merged
into the previous section. That is the milder error for a gate — it dilutes
rather than truncates — and `unknown_headings` makes it discoverable.

Regression tests locking this in:

- `test_social_history_is_not_truncated_by_subfields`
- `test_social_history_with_placeholder_first_line_is_not_called_redacted`
- `test_discharge_condition_subfields_are_also_preserved`

`sections.py` is pure standard library on purpose, so both venvs can import it.

---

## Pick up here

**One open question blocks A1.** The probe needs note text; how should it read?

| Option | Trade-off |
|---|---|
| Query BigQuery each run | ~$0.02 per full scan; simple; but A4 is iterative and re-scans 378 MB repeatedly |
| Cache locally on Desktop | Fast iteration, one scan — but Desktop is **iCloud-synced**, and this is MIMIC note text |
| Cache outside iCloud (`~/.cache/`) | Same speed, keeps restricted data out of sync — **recommended** |

`data/` is already gitignored (`data/`, `*.parquet`, `*.csv`), so nothing
PHI-derived can reach GitHub either way. If we cache, `teardown.py` should learn
to delete it.

---

## Remaining plan

### Evidence gate (§2)

| Step | Deliverable | Answers |
|---|---|---|
| ~~A0~~ | ~~`rag/sections.py` + tests~~ | ✅ done |
| A1 | `scripts/probe_sections.py` | Corrected per-section redaction rate — replaces the 95.5% figure |
| A2 | `scripts/score_test_split.py` → `readmission.test_predictions` | Model probability per admission, needed for orthogonality |
| A3 | `rag/concepts.py` + ~40-sentence labeled test set | Concept rules with negation; "denies X" must not match |
| A4 | `scripts/probe_note_signal.py` | Orthogonal lift + coverage |
| A5 | Your pass/fail call | Whether to build the index at all |

**A2 detail (agreed approach: local batch scoring).** No predictions table
exists — `hospital_score` is the baseline heuristic, not the model. The script
loads the CPR bundle from GCS and scores locally: seconds, $0, and it uses the
bundle's own preprocessing rather than a reimplementation.

The safeguard is the **anchor assertion**. Before scoring anything, the script
scores `hadm_id 20924467` and requires probability `0.131398` — the value
verified against the live endpoint. Match to six decimals means the local path
does the same arithmetic as production. Mismatch means it stops before writing a
bad table. Wrong feature ordering, wrong model version, or different missing-value
handling all produce plausible probabilities and no error; the anchor turns that
silent failure into a loud one. The recorded value is enough — this check does
**not** require the endpoint to be live.

**A4 detail.** The lift measurement restricts to admissions the model scored
*below* threshold, then compares concept rates between those who were readmitted
and those who were not. Testing concepts against the label directly would mostly
measure redundancy with the structured features. A **coverage** metric runs
alongside it: lift proves the signal is real, coverage proves the demo has
something to show. That same quadrant is where the best demo cases live.

### After the gate passes

§3 corpus → §4 chunking (uses `sections.py`) → §5 embeddings (~94M tokens, **rate
still unverified**) → §6 index build (~$1.90, BRUTE_FORCE ground truth then
TREE_AH) → **§7 deploy index endpoint — needs your explicit cost approval, ~$68/mo**
→ §8–9 `rag_search` MCP tool with the cross-patient isolation test → §10 rebuild
the 20-patient cohort for your review → §11 agent fusion → §12 eval → §13 UI →
§14 teardown.

---

## Housekeeping

- **All harness work is uncommitted**: `rag/`, `tests/test_sections.py`,
  `teardown.py` rewrite, Feature Store deletions, `RAG_BUILD_GUIDE.md`.
- **`RAG_BUILD_GUIDE.md` §2 is stale** — still cites 95.5% and the older, weaker
  gate design. Amend after A1 produces the corrected number.
- **The prediction endpoint is live and billing ~$2.60/day.** Nothing in A0–A4
  needs it, since the anchor uses the recorded value. `scripts/teardown.py
  --only endpoint` would stop that meter; `deploy_cpr.py` rebuilds it from the
  cached image when §12 evaluation needs it. Your call — flagging it, not acting.
- Build `86637843-50ea-4d86-9fe7-6de3c745d598` (commit `bd6ffd0`) outcome still
  unverified from 2026-07-30.
- Throwaway probe venv at `/tmp/medspacy_probe` can be deleted; it has served
  its purpose.

---

## Standing constraints

- Every step produces a **committed script plus a written artifact**, verified by
  a test or an explicit check. No throwaway terminal queries.
- Ask before anything user-facing, destructive, or that starts an hourly meter.
- Favour legible code over clever code — a detailed code walkthrough is a
  first-class deliverable once the demo works.
- This project's signature failure mode is a **plausible wrong answer rather than
  an error**. The truncated-regex bug is the most recent instance. Design checks
  that make wrongness loud.
