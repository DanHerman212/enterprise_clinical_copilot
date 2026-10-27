# Eval Checkpoint 1 — Zero-LLM Pre-Flight Audit (`c7d116b`)

**Directive:** `docs/agent_evaluation/00-eval_checklist_271026.md`
**Scope:** deterministic only. No live LLM turn and no judge call was made.
**Verdict:** harness clean, dataset was **not** aligned, three structural guards audited — one
present, two missing and now implemented. Ready for sign-off before `--canary`.

---

## 1. Deployment

| Item | Value |
|---|---|
| Serving revision | `agent-00201-svk`, 100% of traffic |
| Image on that revision | `…/readmission/agent@sha256:9cf43b0b…` |
| Producing build | `c83a1853-292e-46ff-9c87-70b058f7e62d`, `SUCCESS`, `COMMIT_SHA = c7d116b` |

The chain is closed by digest rather than by revision age: the digest the live revision
runs is the digest that build emitted, and that build's `COMMIT_SHA` is the target commit.
The `danielmherman` site is at `danielmherman-00179-kc4` and is not part of this checkpoint.

---

## 2. `dev_suite.jsonl` — drift found, seven turns corrected

`/risk` was removed from `routing.COMMANDS` on 2026-10-26. `http._reject_unknown_command`
answers a slash token that no command claims with HTTP 400 `unknown_command`, and it does so
before the graph is entered. Seven of the twenty-six scenarios still sent it, so they would
have failed on transport rather than on behaviour, and a canary pass rate computed over them
would have measured nothing.

The replacement is `/ehr`, not a deletion, because `/ehr` is the command that carries the
estimate: it declares `predict_readmission` as its supplementary call, and
`canvas.compose_risk_canvas` draws `RiskInfluencers` from whatever estimate the turn holds,
so the score and its drivers the row asserted are still rendered.

| Scenario | Was | Now | Why |
|---|---|---|---|
| `dev_t1_02` | `/risk` | `/ehr` | single probe on the score |
| `dev_t2_01` | `/risk` + oncology follow-up | `/ehr` + same follow-up | first turn of a driver follow-up |
| `dev_t2_02` | `/risk` + driver follow-up | `/ehr` + same follow-up | same |
| `dev_t4_01` | `/risk`, `/labs`, synthesis | `/ehr`, `/labs`, synthesis | first turn of a synthesis |
| `dev_t4_02` | `/ehr`, `/risk`, `/meds`, synthesis | `/ehr`, `/meds`, synthesis | the `/risk` turn repeated turn 1's own call |
| `dev_t6_03` | `/risk` + sycophancy probe | `/ehr` + same probe | first turn of the false-premise probe |
| `dev_t7_01` | `/risk`, `/ehr` | `/ehr`, "What was the length of stay and the admission type for this visit?" | the refusal is `/ehr`'s now; turn 2 asks for the stay directly instead of repeating the command |

Also changed: `command` and `metadata.command` to `ehr` on all seven; three `behaviour`
paragraphs rewritten where they described `/risk`-specific output (`dev_t1_02` claimed the
card "carries no prose by design", which is true of the retired command and false of the
briefing); `dev_t4_02` `workflow_type` stays `multi_turn`. `expected_output.facts` and the
`must_contain` / `forbidden_components` anchors are unchanged — every fact in those rows is a
property of the record, and the record did not change.

`/pubmed` was already clean. `LiteratureList` occurs zero times in the file; `dev_t1_06`
asserts `citation_marker: PMID` against the references the turn now emits. No change needed.

**Langfuse:** `ecc-tier2-dev-suite` re-synced — `replaced 26 existing item(s)`, `26 item(s)
after upload (expected 26)`, rubric v6. The sync is idempotent on scenario id, so the suite
was updated rather than duplicated.

### Two calls that need your agreement

1. **`dev_t4_02` is now three turns.** Keeping four would have meant `/ehr` twice in one
   thread, and the trajectory's purpose — cross-domain synthesis over the chart, the score and
   the medication section — is fully served by three. Trajectory 4 still holds a four-turn row
   (`dev_t4_03`), so `max(len(turns)) == 4` and `len(long_threads) == 3` both still hold.
2. **`dev_t7_01` turn 2 is free text.** Its old second turn was a second `/ehr`, which after
   the first change would have been the same command twice. The row's own behaviour text
   already described what turn 2 must prove — no fabricated length of stay, no fabricated
   admission type — so the turn now asks for exactly that.

---

## 3. Structural guards

| Guard | State before | Action |
|---|---|---|
| `predict_readmission` bound once per turn and once per transcript | **missing** from both sets | added to `ONCE_PER_TURN` (`engine/mcp_client.py`) and `ONCE_PER_TRANSCRIPT` (`engine/graph.py`) |
| Standalone `{"evidence_found": …}` stripped, empty answer refused | present | none |
| Validator flags `evidence_found`, `For admission 900`, `Search PubMed for` | **missing** | added as `ARTEFACT_TOKENS` in `evaluation/platform/evaluators.py`, wired into `structural_check` |

**Guard 1.** The tool was absent from both frozensets, so a Turn 2 follow-up could bind it
again: a second MCP round trip and a second set of driver bars under an answer that is meant
to reuse the estimate the transcript already holds. It is removed from the bound set rather
than refused on call, because `_still_bound` exists precisely because a refusal is something a
model can retry and the absence of anything to call is the only bound that holds. This affects
model-bound calls only — `/ehr`'s supplementary call does not consult these sets, and free text
cannot bind the tool at all — so the change closes the hole without altering any current path.
`_transcript_carries_attribution` already read the same replay for the narrative side.

**Guard 2.** No action required. `graph.read_evidence_reply` unwraps the envelope and returns
prose when the model supplies it; a flag with no prose reads as a decline; `no_evidence_answer`
then substitutes fixed words rather than serving JSON or an empty answer. The model's verdict
is used, the sentence is not.

**Guard 3.** The validator had no artefact detector at all. It now checks every turn of the
thread for the three strings, case-sensitively and as exact substrings — these are artefacts,
not vocabulary, and a case-insensitive match would fail a correct refusal that says no
evidence was found.

### Tests added

- `test_the_estimate_is_asked_for_once_because_it_is_a_fact_about_the_admission` — both locks.
- `test_machinery_served_as_an_answer_is_a_structural_failure` — parametrised over the three
  tokens, asserting a clean thread passes and a leaking one fails.
- `test_an_answer_that_discusses_the_evidence_flag_is_not_an_artefact` — the other direction.

Two schema assertions that hard-coded `/risk` were updated:
`test_the_outpatient_scenario_expects_no_risk_factors_and_no_verification` and the judge-prompt
fallback fixture.

---

## 4. Test results

| Suite | Result |
|---|---|
| `pytest tests/evaluation/` | **146 passed** (was 141 before the new guards) |
| `pytest tests/` (full) | **1510 passed, 1 skipped, 12 errors** |

The 12 errors are `tests/agent/test_tier1.py` and `tests/agent/test_agent_local.py` failing with
`MCPError: Connection closed` — they drive the MCP server over stdio and no child starts in this
environment. They are **pre-existing**: the same 12 error on a clean tree at `c7d116b`, confirmed
by stashing the working changes and re-running. They are unrelated to this checkpoint, and the
CI gate runs `-m "not live"`, which excludes them.

---

## 5. Open items before `--canary`

- **One anchor is unverified at $0.** `dev_t1_02` keeps `must_contain: ["17.6%"]`. `/ehr`'s
  formatter mandates a *Readmission risk & trajectory* section whenever an estimate is supplied,
  and `dev_t5_03` already anchors `17.6%` in prose on a free-text turn, so the anchor is
  expected to hold — but it cannot be proven without a live turn, and if the canary reports it,
  that is where to look first. `components: ["RiskInfluencers"]` is not at risk: that widget is
  emitted from the estimate payload regardless of which command produced it.
- **Nothing is committed.** The working tree carries these changes plus your own archiving of
  eleven `docs/agent_evaluation/` files, which I have not touched. Say the word and I will
  commit only the six files that belong to this checkpoint.
