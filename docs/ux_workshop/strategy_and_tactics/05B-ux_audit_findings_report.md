# 04C — UX Audit Findings Report

**Date:** 2026-10-17
**Scope:** Phase 4C pre-flight sanity check, 13 live turns across four archetypes
**Environment:** local agent (`services.agent.http`, port 8080) + local site
(`manage.py runserver`, port 8000), Langfuse tracing on, canonical host
`https://observability.danielmherman.com`
**Verdict:** UX is **ready for evaluation**, with two behavioural defects to fix first
(A5 router hijack, D1 mistyped command) and one non-blocking gap (trace metadata).

---

## 1. Method

Each turn was driven through the real console and captured at settle:

- settled prose (verbatim prefix), character count
- whether a routing pill appeared (`⚡ Routed to /…`)
- whether the `🛡️ Reconciled with medical record` badge appeared
- whether the partial-answer pill (`⚠️ Copilot run incomplete`) appeared
- the run's stage history (length), from the collapsed disclosure
- raw-marker scan: `/\\^\\[/` (unresolved citation syntax)
- raw-JSON scan: `/\\{"[a-z_]+":/` (serialized payload leaking into prose)

Tables inside the A2UI surface live in a shadow root, so a page-level
`querySelectorAll('table')` does **not** see them. Table presence was therefore checked
against the accessibility snapshot, not the DOM count. Any claim in this report about
matrix rendering comes from the snapshot.

---

## 2. Turn-by-turn ledger

### Archetype A — Cynthia Petrov (#90000015), TKA

| # | Turn | Steps | Routed | Reconciled | Incomplete | Raw `^[` | Raw JSON |
|---|------|-------|--------|------------|------------|----------|----------|
| A1 | `/ehr` | 8 | — | no | no | no | no |
| A2 | RDW driver click | 6 | — | no | no | no | no |
| A3 | free text: hemoglobin/sodium | 6 | **`/labs`** | no | no | no | no |
| A4 | `/meds` | 6 | — | no | no | no | no |
| A5 | clinical follow-up click | 6 | **`/meds`** | no | no | no | no |

- **A2** is the flagship epistemic behaviour and it holds: the answer opens with the
  chart not explaining the driver, then gives the RDW/hemoglobin/RBC trajectory with
  the provenance line. The affordance abstention works — clicking a driver is not
  hijacked into `/labs` by the word "RDW" in its own label.
- **A5 is a defect.** The follow-up row asked *"What does current clinical literature
  suggest regarding the efficacy of this specific combination of medications…"* — a
  literature question — and the free-text router matched the word "medications" and
  sent it to `/meds`. See finding **F1**.

### Archetype B — Bruce Hollingsworth (#90000023), COPD

| # | Turn | Steps | Routed | Reconciled | Incomplete | Raw `^[` | Raw JSON |
|---|------|-------|--------|------------|------------|----------|----------|
| B1 | `/ehr` | 8 | — | no | no | no | no |
| B2 | `/labs` | 6 | — | no | no | no | no |
| B3 | free text: kidney function | 15 | — | no | no | no | no |

- **B3** correctly did *not* route: "kidney function" is not a lab keyword, so it went
  to the reasoning loop and used four tool rounds, landing inside the five-call budget
  with no recursion error. This is the exact shape that used to die with
  `GraphRecursionError`.

### Archetype C — cross-condition literature isolation, on Cynthia Petrov

| # | Turn | Steps | Routed | Reconciled | Incomplete | Raw `^[` | Raw JSON |
|---|------|-------|--------|------------|------------|----------|----------|
| C1 | `/pubmed congestive heart failure` | 8 | — | no | no | no | no |
| C2 | `/pubmed total knee arthroplasty readmission risk factors` | 8 | — | no | no | no | no |

- Both rendered as literature answers with no patient-specific matrix injected into the
  prose, and no raw citation syntax.

### Archetype D — policy fallbacks and edge resilience

| # | Turn | Steps | Routed | Reconciled | Incomplete | Raw `^[` | Raw JSON |
|---|------|-------|--------|------------|------------|----------|----------|
| D1 | `/ehrr` (mistyped) | **18** | — | **YES** | no | no | no |
| D2 | `/ehr` on outpatient note (#90000054) | 8 | — | no | no | no | no |
| D3 | "Summarize the note." on outpatient note | 15 | — | no | no | no | no |

- **D2/D3 are good behaviour on a sparse record.** No formal discharge diagnosis is
  invented: *"The provided documentation does not specify a formal discharge diagnosis"*
  and *"The record does not document a formal discharge diagnosis."* No advice.
- **D1 is a defect.** See finding **F2**.
- One more `/meds` turn was attempted here and came back as `/risk`; on inspection the
  command palette correctly showed `/meds` selected, so that was an automation artefact
  and is **not** counted as a turn or reported as a defect.

---

## 3. Findings

### Category 1 — Formatting and DOM

| ID | Finding | Status |
|----|---------|--------|
| 1.1 | Streaming draft showed `## Hospital course` before the answer settled | **Fixed** — heading markers stripped in `drawn()` alongside emphasis and citation markers |
| 1.2 | Comparative matrix rendered a 4th "shared risk factors" column | **Fixed** — 3 columns (Source / rate / cohort), field stripped at the envelope because the zod schema is `.strict()` |
| 1.3 | A turn rendered `^[brief_hospital_course]` as literal prose | **Fixed** — markers that are neither a passage number nor a `PMID:` badge are removed |
| 1.4 | Risk widget was visually dense | **Fixed** — paddings and bar heights compacted |
| 1.5 | Provenance sentence on `/labs` and the matrix caption on `/ehr` were noise | **Removed** at the user's request; "Patient Chart Synthesis" carries provenance |

**Zero raw-JSON leaks across all 13 turns.** **Zero raw citation markers** in the
settled prose of all 13 turns — the only marker defect (1.3) was found before the fix
and did not recur afterwards.

### Category 2 — Epistemic and guardrail

- **Honest abstention holds.** A2 (driver not explained by the chart) and D2/D3 (no
  formal discharge diagnosis in an outpatient note) both refused rather than invented.
- **No ungrounded medical advice observed.** Every captured prose sample reports what
  the record says. *Limitation:* this was a reading of 13 turns, not a classifier run;
  the formal check is the eval harness's `safety` dimension, which is why Gate 3 matters.
- **Risk figures match the model output** (9.0% / 11.0% threshold on Cynthia, 5.7% on
  Clarence), with the threshold shown beside every value.
- **New — F3: a risk score is presented for records that have no admission.** On the
  outpatient speech-therapy note (#90000054) `/ehr` still draws a
  "Readmission risk & trajectory" section and the risk widget ("5.7%, below threshold,
  increasing risk: None"). The agenda calls these records clinically non-applicable, so
  the surface should say the score does not apply rather than show one. **Not fixed**;
  requires a product decision.

### Category 3 — Orchestration and transport

| ID | Finding | Status |
|----|---------|--------|
| 3.1 | `route()` forced a retrieval for any tool-less answer, replacing a correct manifest-grounded answer (17.29 days → "not documented") | **Fixed** — administrative questions skip the demand |
| 3.2 | Tool budget refused calls but never ended the loop; turns died on `GraphRecursionError` at 10 and again at 25 | **Fixed** — once spent, the agent turn binds no tools |
| 3.3 | BFF treated the agent's own `tool_call_limit` refusals as an outage and printed "unavailable" over a correct answer | **Fixed** — `POLICY_REFUSALS` |
| 3.4 | Client `retire()` ran twice on failure and threw `replaceChild` | **Fixed** — idempotent |
| 3.5 | Tool spans became separate root traces in the reasoning loop | **Fixed** — spanned only where a parent exists |
| 3.6 | A thumbs-down scored `0` and was indistinguishable from "no rating" | **Fixed** — signed numeric `1`/`-1`, one row per trace via upsert |
| 3.7 | **F1 — free-text router hijacks literature questions that mention medications** | **Open** |
| 3.8 | **F2 — a mistyped slash command runs a full reasoning turn** | **Open** |
| 3.9 | **F4 — traces carry no patient/session metadata** | **Open** |

#### F1 — router hijacks a literature question (A5)

*"What does current clinical literature suggest regarding the efficacy of this specific
combination of medications…"* → `⚡ Routed to /meds`. The router matches whole tokens from
a fixed vocabulary, and "medications" is in the meds list; nothing in the rule can tell a
request for **the list** from a request for **literature about the list**. The same
question reached the user as a medication extraction instead of an evidence search.

Proposed fix: require the meds vocabulary to appear with an extraction cue ("what was she
discharged on", "medication list", "dosing") and let questions containing literature cues
("literature", "evidence", "studies", "meta-analysis") abstain. Cheap, and testable with
the existing classifier tests.

#### F2 — mistyped slash command (D1)

`/ehrr` produced an 18-step reasoning turn (four tool rounds) and a
`🛡️ Reconciled with medical record` badge. The turn is not wrong, but the reader gets a
"something was corrected" signal for what is a typo, and it costs a full budget. A
command-shaped token that is not a command should get a short, explicit "no such command"
answer with the closest matches, and no tool spend.

#### F4 — no trace metadata (Step 1 gap)

Traces carry `name`, `tags: []`, `sessionId: None`, `userId: None`, and only SDK
instrumentation metadata. There is no `hadm_id` and no conversation id, so a trace cannot
be filtered by patient in Langfuse. For the eval gates — where the question is "show me
every trace for this case" — this matters. Proposed fix: pass `langfuse_session_id`
(conversation) and a `hadm_id` tag through the run config in `graph.ask`.

---

## 4. Definition-of-done check

| Criterion | Result |
|-----------|--------|
| Zero raw JSON in prose | **Pass** — 13/13 turns |
| Clean Markdown (no raw syntax) | **Pass** — 13/13 after fix 1.3 |
| No ungrounded medical advice | **Pass on reading**, not formally classified |
| No forced-retrieval / recursion failures | **Pass** — no `GraphRecursionError`, no "unavailable" in any turn |
| Partial answers preserved on failure | **Not exercised** — no turn failed mid-stream during this run |
| Routing pill renders on routed turns | **Pass** — A3 (`/labs`), A5 (`/meds`, but see F1) |
| Trace + score reachable per turn | **Pass** (Step 1) |

---

## 5. Required before eval census

1. **F1** — stop the router hijacking literature questions. Small, test-backed, and it
   changes an answer's *content*, so it must precede the census.
2. **F2** — answer mistyped commands without spending a turn.
3. **F4** — add `hadm_id`/session metadata to traces, or Gate 3/4 will be unable to
   filter by patient.
4. **F3** — decide the product behaviour for a risk score on a non-admission record
   before the 82/87 cohort question is settled, because it is the same records.
