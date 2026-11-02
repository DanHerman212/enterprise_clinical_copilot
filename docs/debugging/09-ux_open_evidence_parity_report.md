# OpenEvidence parity — inline citations, cohort synthesis, references, print

2026-11-02, implementing `08-ux_fix_inline_citation_literature_synthesis.md`. Four gaps, two
repos, one backend change that had to exist before any of the prose could be right.

**Gates:** `pytest tests -m "not live"` → **1710 passed**, 1 skipped. `dry_run` → **STEP 7
PASSED**. `npm test` (website) → **110 passed**.

---

## 1. The narrative now places the estimate against the published cohorts

The comparison table drew verified cohorts under an answer that never mentioned them, and the
formatter could not have mentioned them: they are not a tool result. Retrieval returned
twenty-five candidates, an extraction turn verified at most two, and the rows lived in
`command_extras["comparators"]` — read only by the canvas.

- `deterministic.comparator_block(rows)` renders `<verified_comparators>`: PMID, journal ·
  year, the rate with its shape (`a single figure` / `two arms` / `a range`), the cohort, the
  tier. The `rate_sentence` stays out — it is what the row was verified against, not prose to
  quote at a reader.
- `_formatter_input(..., comparators)` appends it after the tool results. Empty rows append
  nothing: a header saying "no cohorts" invites the model to explain an absence the table
  already explains.
- `prompts.EHR_FORMATTING` gains the rule for `Readmission risk & trajectory`: state the
  estimate from `probability_display`, state the published rate(s), say in the same breath
  that a cohort is a population benchmark and not a prediction about this patient, cite each
  as `^[PMID: …]`, and quote a two-arm row as two figures. **Where no block is given, write
  no comparison** — no rate from training, no "the literature shows", and no claim that no
  published data exist.
- `citations.comparator_pmids(state["comparators"])` is unioned into the allowance
  `enforce_citations` uses, so a `^[PMID: …]` badge naming a cohort survives. The allowance is
  the DRAWN rows, not the pool: a PMID the table refused is still an identifier nobody
  verified, and its pill would resolve to no row.

Live, `90000019`:

> **Readmission risk & trajectory.** The patient's estimated readmission risk is 13.6%, which
> is above the 11.0% threshold ^[2]. This estimate is higher than the 3.3% rate observed in a
> cohort of primary THA patients ^[PMID: 33422393] and the 4.9% rate observed in a cohort of
> patients over 80 years old undergoing THA ^[PMID: 32742739]; these published cohorts are
> population benchmarks and not predictions about this patient. …

Both badges survived settlement (`guardrail_flags` empty), and the two PMIDs are the two rows
the table drew.

## 2. Inline pills: one per source per paragraph, and shorter

`cited_narrative.js` decides the pills, and two rules were added where the markers are turned
into controls:

- A repeat of the **same single citation immediately after itself in the same paragraph** is
  drawn once. The marker and the space that carried it go; the prose between markers does not.
  A paragraph that alternates two sources keeps every marker, and one that leaves a source and
  returns keeps the return — the rule is about repetition, not about how often a source may be
  cited. The references list is built server-side from `sources`, so no invariant moves.
- A **chart section's** label is cut at 26 characters with the cut marked
  (`2 · Brief hospital co…`); a **published record's** label is left whole, because it is
  already bounded where it is built. The full label stays in the pill's accessible name and in
  the card it opens.

## 3. References at the foot of the turn

Already built and already visible: `copilot_turn_facts.presentReferences` draws a numbered
`<details>` with the list **open**, before the follow-up row, tagging each row `Clinical Note`
or `PubMed · <year>` and carrying title, citation line (journal, year, authors, PMID) and the
passage. What was missing was paper, not screen — see §4. Nothing in the drawer changed.

## 4. Printing

`demo_shell.css` had no print rules at all. It has one `@media print` block now, and the
components carry their own:

| where | what |
| --- | --- |
| `static/css/demo_shell.css` | hides the sidebar, top bar, context bar, composer, status line, rating controls and menu; releases `height`/`max-height`/`overflow`/`position` from `html` → `.cp-answer` so the browser paginates the whole conversation; turns the row into a column; converts the pinned question into an ordinary heading; forces the references `<details>` open with its summary removed and un-clamps note passages; `break-inside: avoid` on reference rows, follow-ups and blocks; 11pt body, black links, `print-color-adjust: exact`; `@page` margins. |
| `static/js/a2ui/shared.js` (`BLOCK_SETTLE`) | `@media print`: no settle animation (an element caught mid-fade prints at partial opacity), `break-inside: avoid` on every composed block. |
| `static/js/a2ui/cited_narrative.js` | releases the 68ch screen measure on paper; paragraphs avoid page breaks; a heading never ends a page. |
| `static/js/a2ui/comparative_risk_matrix.js` | releases the horizontal scroller (a printed page cannot scroll, so an overflowing table loses its last column); rows and `thead` avoid breaks, the table itself may break between them. |

CSS in `demo_shell.css` cannot reach the A2UI shadow roots, which is why the last three live in
the components. The bundle was rebuilt (`npm run build:clinical`).

## 5. What is verified, and what is not

Verified: the cohort synthesis live on `90000019`; the pill rules and the print rules by test
(`tests/js/citation_pill_density.test.mjs`, `tests/js/print_styles.test.mjs`, both reading the
sources the page loads); the whole backend and frontend suites.

**Not verified here: an actual PDF.** The print block was written against the measured
behaviour (a `100dvh` shell with one scrolling child) and asserted as rules, but no browser was
driven to a printed page from this session. Pagination — `1/3` rather than `1/1` — is the one
thing to eyeball, and the first thing to check is that the sidebar and composer are gone and
the references list is present at the foot of the last page.

**Also outstanding, and in a read-only file:** the two cache-busters. `demo/templates/demo/
app_base.html` loads `demo_shell.css?v=32` and `demo/templates/demo/partials/console_feed.html`
loads `bundled/a2ui_clinical.js?v=16`. Both values are unchanged — those templates are inside
`demo/`, which the workspace contract forbids touching — so a returning browser may serve the
cached stylesheet and bundle. Bumping both is a one-character change each and needs
authorisation; until then a hard reload is required to see any of §2–§4.
