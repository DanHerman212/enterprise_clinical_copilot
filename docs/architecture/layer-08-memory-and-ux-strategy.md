# Layer 8 — Memory and UX strategy session

This is the agenda for the session that settles the five open entries of
`layer-08-memory-session-state.md`. The layer's gaps are decisions rather than defects, so the
decision is the remediation: each entry is closed by choosing an answer here and then building
to it. Nothing in this document has been implemented.

Two things were never mapped. The first is a memory retention policy: nothing states what a
conversation may keep, for how long, or who may read it, and three holdings of clinical text
already exist without such a rule, so the question is not hypothetical and the answer has to
reconcile with them rather than being written around them. The second is the interaction model.
A conversation is already displayed — the console page keeps one episode per patient and
renders a thread with chapters — while every request is composed as a first question, so the
interface implies a continuity the answers do not have. The owner intends a multi-turn
experience in which a physician asks free-form follow-up questions; the session decides what
that means, and the composer and thread that already exist mean the work is about meaning
rather than about affordance.

**Session held 2026-09-18. Every decision this document raises is now settled.** A
conversation exists for the agent and lasts for the working session, its state lives in the
site's database with the caller sending the turns, it is kept for twenty-four hours and swept
by a job rather than by lazy expiry, a stored citation keeps its identity rather than its
passage text, and each of the three existing holdings has its own stated lifetime. The
interface continues the thread it already renders. A turn ceiling bounds a conversation, and
only the first turn spends a credit. The injection boundary is settled as a requirement and in
its stored shape: the store keeps what is needed to rebuild the tool-result structure, and
retrieved passages are re-derived rather than stored. The decisions are recorded in section 4
and section 5. Nothing here has been implemented; this document is the validated strategy.

---

## 1. What the session has to decide

Seven decisions, in this order, because each constrains the next. D2, D3 and D7 are the ones
the others rest on: without a store there is nowhere for a policy to apply, without a policy the
store should not be built, and the four decisions the orchestrator layer handed over constrain
what a stored turn must carry.

### D1 — What does a second turn mean?

The display already exists, so the options are not "build a thread" but "what may a thread
claim".

| Option | What it means | What it costs |
|---|---|---|
| A. Display only | The thread stays as it is: a record of what was asked, with each answer composed in isolation. | The interface continues to imply continuity it does not have. If this is chosen, the interface has to stop implying it: the thread is labelled as a log of independent questions, not a conversation. |
| B. Session-scoped continuity | The agent receives the earlier turns and answers the follow-up in their context, for as long as the working session lasts. | A store, a retention rule, a contract change, and the four decisions in D7. |
| C. Continuity plus cross-session memory | As B, plus material that outlives the session and is reused later. | Everything in B, plus a personalisation question the product has not asked; Google marks it SHOULD. |

**Recommendation: B, and record C as declined for now.** A follow-up question needs the turns
before it; memory that outlives the working session is a different product decision, and taking
it now would mean holding clinical text for longer than the feature requires.

**Decided 2026-09-18: B, with C declined on the record.** The thread becomes answerable for the
working session, and nothing about a conversation outlives it. If A is chosen
instead, it should be chosen explicitly, with the interface change that makes the thread honest
about what it is, because a thread that looks like a conversation and is not one is worse than
no thread.

### D2 — Where does that state live?

| Option | What it means | What it costs |
|---|---|---|
| A. In the agent or the browser process | The transcript is kept in the memory of one process. | Violates A3, and the tool server already documents why: a later request can land on another instance and find nothing. Both services permit up to three instances (`services/agent/cloudbuild.yaml` 54, `services/mcp/cloudbuild.yaml` 45–46). The browser's episode is already this option, and it is what the session is trying to fix. |
| B. In the site's database, with the caller sending the turns it wants considered | The site stores the thread behind the same authorisation boundary that already holds the account and the counter, and sends the turns with the request. The agent stays stateless. | The site becomes a holder of clinical text on the request path. It is already the component that enforces cohort membership, so the text stays inside the boundary that authorises access to it. The transcript travels on every request and the caller owns the context window. |
| C. A purpose-built store the agent resolves | Firestore, Memorystore or Agent Platform Sessions, written and read by the agent service against a session identifier. | A second store to operate and secure, and the harness gains a storage dependency. It also makes the agent's answer depend on state it does not currently have. |

**Recommendation: B.** It satisfies A3's statelessness clause without interpretation, puts the
data inside the boundary that already authorises it, and keeps the harness without a store. Note
the rejected variant inside B: the site stores the thread and the agent reads it directly. It
looks tidier at the contract, and it is exactly the coupling A3 is written to prevent.

**Decided 2026-09-18: B.** The site stores the conversation and sends the turns it wants
considered; the agent stays stateless and holds no store.

### D3 — What is retained, for how long, and reconciling the holdings that already exist

Six sub-decisions. The last two are the ones a current system has already answered by accident.

| Sub-decision | Options | Recommendation |
|---|---|---|
| What is kept | the turns; the citations shown with each answer; the tool calls and their payloads | the turns and the citation identities. Not the tool payloads, which the data layer can reproduce from the corpus. |
| The stored shape of a citation | the whole resolved source; the identity only | **Decided: identity only** — citation number, section and the query, without the passage text. The answer payload carries the passage text today (`services/agent/contracts.py` 23–27), so a literal implementation of "keep the citations that were shown" would store note text through the field this policy was trying not to store. |
| For how long | the browsing session; a fixed interval; indefinitely | **Decided: a fixed interval of twenty-four hours, swept by a job.** Lazy expiry is not sufficient here, and the recommendation in the first draft of this document was wrong to propose it: a row that is never read again is never swept by a reader, so abandoned conversations would hold clinical text indefinitely. The production database is PostgreSQL (`danielmherman/danielmherman/settings.py` 330–333), which has no native row expiry, so the sweep has to be `pg_cron` in the instance or a scheduled management command. The daily counter's lazy rollover is acceptable for the same reason it is unacceptable here: a counter is not clinical text. |
| Who may read it | the account that created it; administrators; nobody | the account that created it, and administrators through the existing administrative interface. |
| What deletion means | delete with the account; annotate and retain; purge on expiry only | deleting the account deletes the conversations; the end of the demonstration is a purge, not an archive. |
| What the logs hold | the same text; identifiers and timings; nothing | identifiers and timings only, with the lifetime stated in the observability layer. |
| What happens to the three existing holdings | one policy for all; a stated lifetime each; leave them | **Decided: each gets its own stated lifetime, on the record.** The evaluation trace archive, the note cache and the committed fixtures are not conversations, so folding them into this policy would describe them wrongly; stating a lifetime for each is what makes the holdings accounted for. |

**The holdings that already exist, and why this is not a clean-sheet policy.** Three places
already keep clinical text without a stated lifetime: the evaluation harness archives each run
as a JSONL trace file it calls the durable archive, and a record carries the question, the
answer and the tool calls including passage text (`evaluation/agent/collect.py` 1–8); the
retrieval layer caches notes outside the repository with its own manifest
(`services/mcp/retrieval/notes.py` 1–34); and captured fixture payloads containing whole
passages are committed to both repositories. The session should treat these as the precedent
the policy has to reconcile with — either they are covered by the same rule, or each is given
its own stated lifetime — rather than writing a policy for a store that does not exist yet and
leaving three holdings outside it. This is also where requirement F6 (no personal or
confidential data in logs, MUST) and G6 (data minimisation, MUST) apply to this layer.

### D4 — Identity, isolation and what a turn costs

**Decided 2026-09-18: a conversation is bound to exactly one admission when it opens, the
binding is re-asserted on every turn rather than trusted to the transcript, and only the first
turn spends a credit.** A follow-up cannot change the admission; asking about a different
patient starts a new conversation, and the interface says so.

The credit decision decouples spend from the allowance and the consequence has to be recorded
rather than glossed: an account with one credit can now spend as many model calls as the turn
ceiling allows. The turn ceiling decided in D6 is therefore not a tidiness measure, it is the
only bound on what one credit buys, and it has to be chosen small enough to be a real bound.
Two further consequences follow. A conversation whose first turn fails still costs a credit
unless the refund path is extended to close the conversation it opened, and the ceiling has to
say what happens when a turn is refunded after the ceiling was reached.

This is the decision with a clinical consequence. The failure it prevents is a follow-up
answered about a patient other than the one under discussion, which produces a plausible answer
about the wrong person. Pinning the admission in the stored conversation and re-asserting it per
turn is the only defence that does not rely on the model noticing.

### D5 — What the physician sees

The composer and the thread exist; what is undecided is what the thread promises and how a
second turn is rendered into it.

| Option | What it means | What it costs |
|---|---|---|
| A. The thread stays a log | The existing thread and chapters, with each answer composed in isolation. | Honest only if the interface says so. This is D1 option A. |
| B. The thread becomes a conversation | The existing thread continues to grow, each answer composed with the turns above it, with the patient pinned in the header. | The interface is mostly built; the change is what the answer is given, plus the state that answers it. |
| C. Multi-patient conversation | One thread that can move between patients. | Needs the switching act of D4 designed, and makes the wrong-patient failure reachable by ordinary use. |

**Recommendation: B, and decline C for the demonstration.** Two existing behaviours have to be
decided alongside it, because a conversation inherits them: the chapter rendering with its
"earlier messages" control, which decides how much of the transcript a reader can reach, and
the fact that the answer is not streamed token by token — only the chain's stages are streamed,
and the answer arrives as one frame at the end. A conversation interface makes that more
noticeable, and the session should either accept it or decide to change it.

**Decided 2026-09-18: B, with the patient pinned in the header.** The chapter rendering and the
"earlier messages" control stay as they are, and the answer continues to arrive as one frame at
the end of a streamed progress narrative, which is accepted as a limitation of the current
streaming shape rather than changed as part of this work.

### D6 — What bounds a conversation

**Recommendation:** the daily allowance bounds the total, as it does today; the bound that does
not exist is on the shape of a conversation, which is what a ceiling on turns provides. Two
existing limits belong in the decision because a multi-turn interface will meet them first: the
site permits a fixed number of agent calls in flight per instance, enforced by a semaphore, with
a timeout on each call (`danielmherman/demo/agent_client.py` 45,
`danielmherman/danielmherman/settings.py` 101–112). Several conversations open at once against
that limit is the realistic pressure, not one long conversation.

**Decided 2026-09-18: a turn ceiling per conversation, plus the daily allowance, and the
concurrency limit stays as configured.** The ceiling is the control on the credit decision in
D4 and has to be small enough to bound what one credit buys; it is set at implementation time
and recorded here when it is.

### D7 — The four decisions the orchestrator layer handed over

The orchestrator layer recorded four questions as this layer's, because continuity is what makes
them necessary. None has an answer, and three of them constrain what a stored turn must carry
(`docs/architecture/layer-03-orchestrator.md`, section 6).

| Question | Why it is this layer's | Recommendation |
|---|---|---|
| Must the prompt template and the replayed material be separated? | Once history is replayed, what is instruction and what is data cannot be left implicit. | Separate them: the template stays a template, the transcript is data, and the boundary between them is explicit in the assembled context. |
| How does a replayed passage stay outside the tool-result boundary? | Retrieved note text is currently marked as tool output, which is what tells the model it is data and not instruction. Replaying it in the prompt outside that marker reopens the indirect-injection channel the marker closes. | Replay it inside the same boundary, or in an equivalent one. This is a security decision, not a formatting one. |
| Is history replayed as structure or as text? | Structure is cheaper, and it is what keeps a stored turn from duplicating passage text. | Structure: the citation identity and the question, with passage text resolved again on demand, since resolution is deterministic. |
| How does the code revision travel with a stored turn? | A stored answer without the revision that produced it cannot be explained later. | Record the revision with each turn when the turn is stored, which is a field in D3's stored shape. |

**Decided 2026-09-18: the first, third and fourth are accepted; the second is accepted as a
requirement and carried forward as a design question.** The revision is stored with each turn,
history is replayed as structure rather than as text, and the prompt template is separated from
the material replayed into it. What remains open is how the second is implemented, and the
question is narrower than it first appears because the boundary already exists in the code.

A tool result enters the transcript as a `ToolMessage` whose content is wrapped in
`<tool_result name="…">…</tool_result>` (`services/agent/graph.py` 208–215), the system prompt
tells the model that everything inside that wrapper is data rather than instruction
(`services/agent/prompts.py` 31–33), and the wrapper is defended against a payload that closes
it early, by neutralising any occurrence of the closing delimiter inside the payload before it
is rendered (`services/agent/graph.py` 98–102, 130). The boundary is therefore structural and
enforced, not a convention the prompt asks for.

Replaying a historical turn has to reconstruct that structure: the same message roles, in the
same order, with retrieved text back inside a `ToolMessage` carrying the same wrapper and the
same neutralisation. Flattening history into one text block, or replaying it as ordinary user
or assistant text, puts note text where the prompt says instructions live and reopens the
channel the wrapper closes.

The open question is what this requires the store to keep, because it is coupled to D3's
decision to keep citation identities rather than passage text. Reconstructing a `ToolMessage`
needs the tool's rendered payload, so either the rendered payloads are stored, or they are
re-derived on replay. Re-derivation works for retrieval, where the identifier resolves
deterministically to the same passage, and it does not work for the prediction path, whose
payload is a score and its factors and cannot be recomputed to the same value without the
endpoint version that produced it. The shape to decide is therefore: store the tool name and
its arguments, store the payloads that cannot be re-derived (the numeric prediction results),
and re-derive the payloads that can (retrieved passages), so that no passage text is stored but
every turn can be replayed with its tool results in their structural position.

---

## 2. Prerequisites the session should note

Four facts constrain the answers rather than being questions in themselves.

The request contract is closed on purpose at the agent, and the reason is recorded: an unknown
field used to be ignored, so a caller that sent a history received an answer that had silently
dropped it (`services/agent/contracts.py` 41–84). That protection does not extend to the site's
proxy in front of it, which reads three keys and ignores the rest, so a browser posting a
history would have it dropped in silence at the outermost boundary
(`danielmherman/demo/views.py` 41–96). Any field the session adds has to be added to both
closed sets in the same change, and the site's boundary should be closed before continuity work
starts rather than after.

The demonstration has a path that answers without the agent at all: in fixture mode the site
composes the answer from a captured payload, the allowance is reported rather than charged, and
the question wording is the site's own (`danielmherman/demo/views.py` 262–270,
`danielmherman/demo/fixtures.py` 46–52). Whatever the session decides for the live path, the
fixture path has to be either brought along or explicitly left behind, and the difference has
to be visible in the interface rather than inferable from the answers.

Citations remain re-derivable rather than storable for the same reason they resolve today: a
returned identifier is parsed and the passage recomputed by the deterministic chunker that built
the index, so an earlier turn's citation can be reconstructed instead of kept. That is what makes
D3's "identity only" recommendation affordable.

The services run up to three instances each with a minimum of zero
(`services/agent/cloudbuild.yaml` 54, `services/mcp/cloudbuild.yaml` 45–46), and each service
holds caches whose lifetime is per process, including a forty-five-minute identity token
(`services/agent/mcp_client.py` 241, 249). Nothing should be designed on the assumption that the
process that answered the last turn is the process that answers the next one.

---

## 3. What the session produces, and where it is written down

| Output | Where it goes |
|---|---|
| The chosen options for D1 to D7 | the decision record below, then the entries in `layer-08-memory-session-state.md`, which are closed as the work lands |
| The retention policy | this document, and the operations summary, so it is findable without reading the code that implements it |
| The interaction model | the client layer document, which owns what the interface offers |
| The four inherited decisions, and the injection boundary in particular | the orchestrator layer, which owns what is placed into the window and in what order, and the security plane for the boundary |
| The closing of the site's request boundary | the orchestrator layer's record, which currently describes the guarantee as holding at one boundary |
| The isolation rule for conversations | the tools layer, whose isolation guarantee is what each turn has to re-assert |

---

## 4. Decision record

Recorded 2026-09-18. The session put seven decisions; six are settled and the seventh is
settled in part, with its open half stated in the row rather than left blank.

| Decision | Options considered | Chosen | Rationale | Date |
|---|---|---|---|---|
| D1 What a second turn means | A display only / B session-scoped / C cross-session | **B** | A follow-up needs the turns above it; cross-session memory is a decision the product has not taken, and Google marks it SHOULD rather than MUST. | 2026-09-18 |
| D2 Where the state lives | A in-process / B site database with the caller sending turns / C a store the agent resolves | **B** | Satisfies A3's statelessness clause literally, keeps clinical text inside the boundary that already authorises it, and adds no store to the harness. | 2026-09-18 |
| D3 Retention policy and stored shape | six sub-decisions | **twenty-four hours, swept by a job; citation identity only; each existing holding given its own lifetime** | Lazy expiry cannot sweep a row that is never read again, so the sweep is a job, and PostgreSQL has no native row expiry. Keeping citation identity avoids storing passage text through the field meant to avoid it. | 2026-09-18 |
| D4 Identity, isolation and cost per turn | admission pinned per conversation, credits per turn or per conversation | **admission pinned and re-asserted; only the first turn spends a credit** | Continuity must not be able to move between patients implicitly. The credit decision decouples spend from the allowance, so the D6 ceiling carries the bound. | 2026-09-18 |
| D5 Interaction model | A log / B conversation / C multi-patient | **B, patient pinned in the header** | The thread and composer already exist; the change is what the answer is given, not what the interface offers. | 2026-09-18 |
| D6 What bounds a conversation | allowance only / turn ceiling / ceiling and lower concurrency | **a turn ceiling per conversation, plus the daily allowance** | The allowance bounds the total, the ceiling bounds the shape and is the only control on what one credit buys. | 2026-09-18 |
| D7 The four inherited decisions | accept all / accept three and discuss the boundary / discuss all four | **three accepted; the injection boundary accepted as a requirement with its stored shape still to decide** | The boundary exists in the code as a message role plus a wrapper with a delimiter guard, so replay must reconstruct it; what that needs the store to keep is coupled to D3 and is the one question carried forward. | 2026-09-18 |

Once this table is filled, the implementation is ordinary work: a store behind the site's
authorisation boundary, a contract that carries the turns the caller wants considered, an
interface that continues the thread it already renders, and tests that a conversation cannot be
answered about an admission other than the one it was opened for.

---

## 5. Follow-up decisions

Six questions remained after the session. All six were answered on 2026-09-18, and together
they close the open half of D7 and the loose ends the seven decisions left.

| Question | Decision | What it settles |
|---|---|---|
| What does the store keep so that a turn can be replayed? | Tool names, their arguments, and the payloads that cannot be re-derived — the prediction scores. Retrieval payloads are re-derived on demand. | Closes D7's open half: no clinical note text enters the session store, and a replayed turn still carries its tool results in their structural position. **The dialogue half is implemented 2026-09-18**: the contract accepts a bounded, closed turn shape and the chain replays it as messages. **Both halves are implemented 2026-09-18**: the dialogue as messages, and the tool results rebuilt as the assistant call plus its result through the same wrapper and delimiter guard, with retrieval re-resolved rather than stored. The site still has to send a turn's tool calls. |
| What is the turn ceiling, and what happens at the limit? | A hard limit on user turns, provisionally six. At the limit the interface offers a button that starts a new conversation, which spends the next credit. | Bounds what one credit buys, which the D4 credit decision otherwise leaves unbounded. No silent truncation and no dead end. **Implemented 2026-09-18**: the ceiling is enforced in the ask path and refuses with the way forward named; nine tests cover it. |
| What happens when a conversation's first turn fails? | The credit is refunded and the conversation keeps its state. The retry is the first successful turn and costs exactly one credit. | Settles the refund path D4 left open, and removes the incentive to start a fresh conversation to avoid paying twice. **Implemented 2026-09-18**: a failed opening turn refunds its credit and keeps its conversation, and a turn counts only when an answer exists. |
| When is the site's request boundary closed? | Before any continuity work begins. The site refuses an unknown field rather than dropping it, with the same two-wordings refusal the agent uses. | Removes the silence at the boundary a browser reaches, before history fields exist to be dropped. **Implemented and verified 2026-09-18**: the site refuses unknown fields, three tests hold it, and the site's 89 tests pass. |
| What lifetimes do the three existing holdings get? | The evaluation trace archive keeps a fixed, longer lifetime for evaluating and auditing. The note cache is treated as ephemeral and regenerable, with a very short lifetime. The committed fixtures carry no runtime lifetime: they are governed by repository policy and must stay synthetic or de-identified. | Gives each holding a stated lifetime, which is what the D3 decision asked for, and keeps repository content out of a runtime rule. The conversation side of it is **implemented 2026-09-18**: a fixed window from creation, a sweep command, and nine tests. The trace archive and note cache lifetimes are recorded decisions whose harness-side changes are outstanding. |
| Does the offline fixture path follow the live one? | No. It stays explicitly single-turn, and the interface labels it as such. | Prevents the offline demonstration from implying a continuity it cannot produce. **Implemented 2026-09-18**: fixture mode opens no conversation and writes no turn, and the page carries an offline note. |

### The boundary of a session (decided 2026-09-18)

D1 chose session-scoped continuity and declined cross-session memory, but did not say what a
session *is*. Left implicit, that produced a visible gap: the store keeps a conversation for
twenty-four hours while the rendered thread dies with the page, so a clinician who reloads is
answered as though the conversation never happened — a new conversation, and a second credit
for the first turn.

| Option | Behaviour after a reload | Cost | Decision |
|---|---|---|---|
| A session is the page load | The thread is gone and the next question opens a new conversation, **and the interface says so while a conversation is open** | One line of copy and its test | **Chosen** |
| A session is the tab | The thread is re-rendered from the store and the same conversation continues | The store holds no rendered turn: a resumed turn would be a transcript without its A2UI canvas or the passages behind its citations, so a faithful resume needs an agent route that recomposes a stored turn — the only alternative, storing the composed presentation, would put note text and scores in the site's database and contradicts D3 | Deferred, not rejected |
| The conversation continues silently | The pane looks empty while the copilot remembers anyway | Invisible state, which is the failure this layer exists to prevent, and it makes a "start a new conversation" refusal incoherent | Rejected |

The chosen option is recorded as **a session is a page load**: the conversation is the page's,
and the interface states it rather than letting the user discover it from a follow-up that was
answered as a first question. Within a page, nothing is lost — switching patients and back
keeps each thread and each conversation (`danielmherman/static/js/demo_flow.js` 150–168). The
tab-scoped option becomes worth building when the demonstration has a clinician-facing life;
then the explicit form is a "continue your conversation with this patient" affordance rather
than an automatic restore, so nothing is invisible and no credit is spent by surprise.
