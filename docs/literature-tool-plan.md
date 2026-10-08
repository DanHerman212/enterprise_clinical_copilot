# PubMed literature tool — implementation plan

**Opened:** 2026-10-08 · **Status:** agreed, not started
**Repos:** `enterprise_clinical_copilot` (MCP server, agent) · `danielmherman` (console)

The physician asks for recent literature on the patient's condition. The agent searches
PubMed, answers in the thread citing each article by PMID, and the Context Canvas swaps
from the risk view to the retrieved articles — title, journal, publication date, full
abstract, and a link to the source on NCBI.

---

## The request, in the four steps it was given

1. A chip — "Search PubMed for recent literature on this condition."
2. The agent calls a PubMed tool and gets abstracts back.
3. The thread carries the agent's synthesised answer with citation badges.
4. The canvas shows the unedited abstracts, dates and titles, in place of the risk signal.

## Where that meets the existing architecture

**Step 4 does not match how the console works, and the difference matters.** The browser
does not intercept tool payloads and decide what to render. The agent composes the whole
canvas server-side and sends a finished A2UI envelope; the console mounts it. No code in
`static/js/` maps a tool name to a view — the only name-keyed lookups are for thread
metadata.

So the swap is two pieces of work, neither of them in the browser's logic:

- a **payload selector** in the agent, alongside the ones that already exist for
  prediction and retrieval; and
- a **new A2UI component**, registered in the agent's composer *and* in the console's
  catalogue.

This is the better outcome — no parsing of abstracts in JavaScript — but it means the
work sits in the agent's presentation layer rather than in Django.

A consequence to confirm rather than manage: the canvas is composed per answer, so a
literature answer shows literature and the following answer re-composes from scratch.
"Temporarily replaces the risk signal" is therefore automatic, not state we hold.

---

## Decisions

### D1 — Retrieve via NCBI E-utilities directly, not `PubmedQueryRun`

**Decided 2026-10-08.**

The goal is reliable PubMed capability, not adherence to a particular LangChain helper.
`PubMedAPIWrapper` is designed to dump a text blob into a prompt, not to feed a
structured, multi-component UI. Three reasons:

1. **Structured data over fragile parsing.** The wrapper returns one concatenated string.
   Recovering fields from it means regex or splitting, and PubMed records vary far too
   much for that: structured abstracts (`BACKGROUND`/`METHODS`/`RESULTS`) against single
   blocks, inconsistent author formats, per-journal copyright lines. E-utilities returns
   XML with the fields already separated — `ArticleTitle`, `PubDate`, `AbstractText`,
   `PMID` — so nothing is inferred from text.
2. **Dependency hygiene.** Adding `langchain-community` to the MCP server imports a heavy,
   fast-moving tree to perform one HTTP GET. The MCP server stays a small, focused
   service using `httpx` and `xml.etree.ElementTree`.
3. **Search control.** Query construction can be tuned for clinical decision support —
   restrict to publication types (Clinical Trial, Meta-Analysis, Systematic Review), sort
   by date, and cap the payload with `retmax` so the tool never over-fetches.

The two-call pattern:

- `esearch.fcgi?db=pubmed&term=<query>&retmode=json&retmax=3` → list of PMIDs
- `efetch.fcgi?db=pubmed&id=<pmids>&retmode=xml` → records, parsed into typed dicts

### D2 — Citations by PMID, in a separate `literature[]` array

**Decided 2026-10-08.**

Badges read `[PMID: 3123456]` and link to NCBI. This is deliberately *not* folded into the
existing `sources[]` array, which is the discharge-note citation channel: it carries
`{cite: int, section, text, query}`, is renumbered by the citation pipeline, and a
journal article has no `section` to place in it. Overloading that field to carry a PMID
would couple two unrelated citation schemes and make both harder to reason about.

Reference material is rendered from `literature[]`; note material continues to render from
`sources[]`.

---

## Phase 1 — MCP server: the tool

`enterprise_clinical_copilot`

1. **Declare the return shape** in `services/mcp/contracts.py`, beside `PredictionResult`
   and `RetrievalPassage`. Per article: `pmid`, `title`, `journal`, `pub_date`, `abstract`,
   `url`. Plus the envelope: `query`, `returned`, `articles`. This is not optional
   bookkeeping — the SDK builds a Pydantic output model from the return annotation and
   undocumented keys are dropped before the agent sees them. The file already carries this
   warning in the `note` comment.
2. **Register the validator** in `TOOL_CONTRACTS` (`services/mcp/contracts.py:150`), which
   `validate_tool_result` (`:157`) dispatches through.
3. **Create `services/mcp/tools/literature.py`**, opening with the rule both existing tools
   carry: returns plain JSON, never A2UI, because a tool that returns UI is welded to one
   presentation layer and stops being usable from anywhere else.
4. **Signature.** `async def search_literature(query: str, max_results: Annotated[int, Field(ge=1, le=10)] = 3) -> LiteratureResult | ToolError`, mirroring `rag_search` in
   `services/mcp/tools/retrieval.py`, including its `Field` bounds precedent.
5. **Empty result is not an error.** The console treats *any* `tool_calls[].response.error`
   as an outage: it refunds the request and shows the unavailable message. Reporting "no
   articles matched this query" as an error would tell the clinician the service is down.
   Mirror the retrieval precedent — `returned: 0`, empty `articles`, and a `note` string.
   Reserve `ToolError` for NCBI genuinely being unreachable or malformed.
6. **Configuration** in `services/mcp/config.py`, following its module-level env constants:
   the NCBI contact email and API key, plus a timeout. The key belongs in Secret Manager,
   delivered as a Cloud Run env var. Unauthenticated NCBI allows 3 requests/second against
   10 with a key; both are far above the traffic here, but the email is required by NCBI's
   usage policy regardless.
7. **Register it** — `server.add_tool(search_literature)` at `services/mcp/server.py:38-40`.

## Phase 2 — Agent: classify, label, compose

`enterprise_clinical_copilot`

8. **Classify the tool** in `services/agent/contracts.py:46-47`. It is re-callable, so it
   belongs in `RE_DERIVABLE_TOOLS` and the site does not store its response.
   `tests/agent/test_agent_contract.py` holds the two sets exhaustive over registered
   tools and will fail until this is done — deliberately, so a tool cannot be added
   without someone deciding which it is.
9. **Add a progress label** in `services/agent/stages.py:107`, or the thread reports the
   generic "Consulting a Tool" (`:117`) throughout the search.
   `tests/agent/test_progress_stream.py` enforces this too.
10. **Add a payload selector** in `services/agent/a2ui.py`, beside `predict_payload`
    (`:304`) and `rag_payload` (`:317`), which currently hard-code the three known names.
11. **Compose the literature view** in `compose_risk_canvas` (`services/agent/a2ui.py:180`).
    This is the step that performs the swap: when a literature call is present, emit the
    literature component instead of `RiskBar` and `FactorBars`, which are appended only
    when a prediction estimate is usable. Keep the surface and catalogue identifiers
    consistent with `CATALOG_ID` (`:52`) and `SURFACE_ID` (`:54`).
12. **Extend the response contract** — add `literature` to `AgentSuccess` and to
    `validate_agent_success` in `services/agent/contracts.py`.
13. **Locate the model's instruction source and state the citation convention there.**
    The model must emit `[PMID: <id>]` for literature and leave note citations as they
    are. The instruction file for this agent has not yet been identified in this plan;
    find it before writing the prompt change.

## Phase 3 — Console

`danielmherman`

14. **Add a literature component** to `buildRiskCatalog` (`static/js/a2ui_risk_components.js:310`)
    with a registered custom element alongside the four that exist (`:99`, `:163`, `:238`,
    `:299`) and a zod schema declared `.strict()`, matching the others. `CATALOG_ID`
    (`:303`) must stay identical to the agent's, or nothing renders.
15. **Carry `literature` through Django's contract** — `demo/agent_contract.py`,
    `AgentSuccess` and `validate_agent_response`, which currently rejects any shape it is
    not told about.
16. **Render the badge.** `[PMID: …]` becomes a link to the article on NCBI. Add the new
    tool name to `agentTurnFromResponse` (`static/js/demo_flow.js`) so the thread's "used"
    line names it. Bump the cache-buster — the console test pins those strings.

## Phase 4 — Tests

17. One test per layer: MCP contract validation; agent exhaustiveness
    (`RE_DERIVABLE_TOOLS`, `TOOL_LABELS`); A2UI composition selecting the literature view;
    Django console asserting the chip is present.
18. **A regression test that `[PMID: 12345678]` survives the citation pipeline unchanged.**
    The existing renumbering rewrites note citations to match `sources[]`. Literature
    citations pass through the same text and must not be rewritten, renumbered, or
    stripped. This is the one place the two schemes can silently collide.
19. **An adversarial case: the tool returns nothing.** Assert the clinician sees "no
    recent literature found for this query", the request is not refunded, and no outage
    message appears.

## Phase 5 — Deployment

20. **Site first, then agent.** The two services deploy from separate triggers. If the
    agent emits the literature component before the console knows it, the renderer draws a
    blank box with a console warning; the reverse order is harmless.

---

## To settle before or during implementation

- **Where the user's chip sits.** The requirement says "a button below the chat box"; an
  earlier note says the patient rail, beside the existing chips. Not yet reconciled.
- **Who chooses the search terms.** The model, deriving them from the patient's condition
  in the conversation, or the physician typing them at the chip. The chip's fixed wording
  — "this condition" — implies the first.
- **What leaves the system.** Any query sent to NCBI is transmitted outside our boundary.
  Today's cohort is synthetic, so there is no exposure, but the design should be
  deliberate: constrain the query to the condition or indication rather than permitting
  free text drawn from the note, and record that decision here.
- Whether `httpx` is already available to the MCP service, or must be added.
- Whether the stored turn keeps enough to re-render a literature canvas on reload.
