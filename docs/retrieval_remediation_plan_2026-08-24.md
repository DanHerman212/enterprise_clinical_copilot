# Retrieval & Citation Remediation Plan — 2026-08-24

Status: PLAN — for review. No fixes beyond what is already staged will be
pushed until this plan is approved and each diagnosis is validated with data.

---

## 1. The three symptoms (as reported)

1. **Citations do not enumerate naturally — hard-wired.** A meds-only answer
   cites `^[3]` because `discharge_medications` is the 3rd section in
   `SUMMARY_SECTIONS` order — the "lazy hack" of numbering by the tool's fixed
   array position instead of order of appearance.
2. **Citations do not match the source.** When summarizing meds, the linked
   source is a different section (hospital course, diagnosis) than the claim.
3. **Sources contain out-of-scope content.** The cited passage text pulls in
   other sections (e.g. ALLERGIES, ACTIVITY) unrelated to the claim.

## 2. Working diagnosis (from today's evidence — to be validated in Phase 1)

Four structural facts drive all three symptoms:

### A. The retrieval tool returns the WHOLE NOTE, not the chunk.  [drives #3]
In `agent-harness/mcp_server/tools/rag_search.py`, `_fetch_texts(note_ids)`
fetches the **full note text** from BigQuery by `note_id` and returns it as the
`text` of *every* retrieved passage. The index stores section-level chunk
embeddings (id = `note:section:ordinal`), so the `section` label is real — but
the returned `text` is the whole note. Consequences:
- Every SourceCard shows the entire note (must re-extract the section later).
- The agent reads the full note in **every** passage, so it can cite facts
  that live anywhere in the note — grounded in the note, but not in the cited
  section. This is exactly symptom 3.

### B. Section retrieval is near-tied and drops sections.  [drives #2]
`rag_search_sections` queries each section using that section's body text and
picks the passage whose `section` label matches, with `top_k=3` plus a `seen`
id-dedup. Because section chunks for the same note are highly similar, the
intended section can miss the top-k or be deduplicated (observed: hadm
90000015 returned 2 of 3 sections across runs). When the meds section is
dropped, the display/agent falls back to a different section → symptom 2.

### C. Citation numbers are the model's tool-array positions.  [drives #1]
The prompt tells the model to cite the fixed-position number (meds = `^[3]`).
The numbers are display artifacts, not natural enumeration. (Fix staged:
`renumber_citations` → first-appearance order + collapse of stacked
`^[1]^[2]^[3]` clusters.)

### D. The eval measured answer-groundedness, not citation↔section fidelity.
**[answers "how did we pass yesterday"]**
The LLM judge scores the answer text against the retrieved passages
(faithfulness / groundedness / citation / clinical / safety). Because every
passage's text is the WHOLE note, the meds answer is grounded in text that IS
in the note → high groundedness. The judge cannot see:
- whether the citation *number* maps to the right passage (presentation layer),
- whether the cited *section* matches the claim (section recall),
- that the passage text is a whole-note blob rather than the cited section.
So a high eval pass is compatible with all three symptoms. **The eval gate
needs a retrieval-level check, not just answer-groundedness.**

## 3. Phase 1 — Validate the diagnosis (data, not assumptions)

> **Dependency:** steps 1–3 query the live Vector Search index endpoint, which
> was torn down on 2026-08-24 for cost. Redeploy first with
> `scripts/launch_endpoints.sh` (~30 min, keeps the existing index resource)
> before running the live-measurement steps. Note parsing (step 1, note-level)
> reads BigQuery only and needs no endpoint.

Each step produces an artifact the user can inspect.

1. **Passage-text audit** — run `rag_search`/`rag_search_sections` for a few
   patients; print `id`, `section`, and `text` length vs the note length.
   Confirm `text` == whole note (symptom A) and that `id` names a section
   chunk. → `artifacts/passage_text_audit.md`
2. **Section-recall census (cohort-wide)** — for all 108 demo patients run
   `_search_sections`; tabulate per section (`brief_hospital_course`,
   `discharge_diagnosis`, `discharge_medications`, `discharge_instructions`,
   `discharge_summary`) whether the correct-labeled passage was returned, and
   count patients whose meds/instructions section was dropped. This is the
   precision/recall number for the retrieval layer. → `artifacts/section_recall.csv`
3. **Citation-fidelity audit** — capture raw agent answers for the meds chip
   across a cohort sample; compare each `^[n]` to the section it actually
   supports (ground truth = the note's sections). Classify: correct /
   wrong-section / stacked / fabricated. → `artifacts/citation_audit.md`
4. **Eval gap confirmation** — show a meds turn that scored well on
   groundedness while its citation pointed at the wrong section, to prove the
   judge cannot catch it. → included in citation audit.

## 4. Root-cause sign-off gate

The user reviews Phases 1–4 artifacts and confirms the root cause before any
remediation is built. No production push until then.

## 5. Phase 2 — Remediation (ranked; each local-tested)

1. **Retrieval contract: return the chunk, not the whole note** (fixes #3).
   Make `passages[].text` the actual section-chunk text. Since chunking is
   deterministic (`rag/chunking.py`, id = `note:section:ordinal`) and the
   notes are in BigQuery, the tool can re-derive the exact chunk text
   (offline, cached) — no index rebuild needed. SourceCards then show only the
   cited section.
2. **Deterministic section resolution in `rag_search_sections`** (fixes #2).
   Resolve sections from the parsed note (parse_note already yields every
   section) and return one exact chunk per existing section — 100% recall by
   construction, no top-k luck, no dedup drop.
3. **Natural citation numbering** (fixes #1). Ship the staged
   `renumber_citations` (first-appearance order, collapse stacks) + the
   client-side mirror.
4. **Agent prompt hardening** (staged): meds with no named meds →
   "no discharge medication information is available"; never stack citations;
   cite at most one passage per claim.
5. **Eval gate upgrade** — add retrieval-level checks to the judge/harness:
   citation number ↔ passage section match; section recall over the cohort;
   passage text must equal the section chunk (no whole-note blobs). So this
   class of failure fails the gate next time.

## 6. Local testing before push

- Unit tests (site + harness).
- Cohort-wide retrieval census rerun → section recall must reach the
  parse-based ceiling.
- `demo/verify_live_citations.py` sweep (meds + free-text across 8+ patients,
  incl. no-meds notes).
- Agent sweep classification of meds answers (honest-empty / grounded /
  wrong-section / fabricated) over a cohort sample.

## 7. Deployment sequencing (after approval + local sign-off)

1. Site push (display: renumbering, aliases, unavailable contract).
2. Agent redeploy (prompt rules) — Cloud Run `agent` service.
3. MCP/retrieval change (chunk-text sourcing) — redeploy `mcp-server`.
4. No index rebuild expected (embeddings unchanged); if the census proves
   otherwise, budget for it.

## 8. Open questions for the user

- Accept "whole-note passages" as the current behavior until the chunk-text
  change lands, or gate UAT on the change?
- Should the meds chip resolve to `discharge_instructions` when no
  `discharge_medications` section exists (current), or only to
  `discharge_medications` (stricter)?
- Scope of the eval-gate upgrade: full judge rerun tonight or a targeted
  retrieval check first?

---

## Appendix — why the eval passed (short answer)
The judge scores answer text vs the passages it received. The passages are
whole notes, so meds answers are faithful to the note → high groundedness. The
judge never sees (a) which passage index a citation number points to, or
(b) whether the cited section matches the claim. Those are exactly the three
symptoms above. The fix is to (1) return real section chunks and (2) add a
citation↔section check to the gate.
