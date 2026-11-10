## NON-NEGOTIABLE GOVERNANCE & ANTI-HACKING LAWS

0. **Deploy & Commit Authorization (highest precedence):**
   - NEVER `git commit`, NEVER `git push`, NEVER `gcloud builds submit`, NEVER
     `gcloud builds triggers run`, and NEVER deploy, without express authorization from the
     user for that specific change, in that turn.
   - Authorisation for one stage does not carry to the next. Ask again, every time.
   - Pushing to `main` fires the Cloud Build trigger and deploys to Cloud Run. **A push is a
     deploy.** There is no commit-only path that is safe.
   - When a directive says "report X before committing", produce the report and STOP. Doing
     both in one turn is a violation, even when the report is accurate and the suites pass.
   - The loop is: implement → run both suites → live verification → report the diff, the
     test counts and the evidence → **stop** → wait for authorization → then commit/push.

1. **Strict Scope Enforcement (`Authorized Scope`):**
   - You may ONLY create or modify files explicitly listed in the current prompt's `Authorized Scope`.
   - Modifying any file outside `Authorized Scope` is a hard failure.

2. **Test & Evaluation Harness Immutability:**
   - NEVER modify `evaluation/platform/` (`task.py`, `evaluators.py`, `judge_call.py`) or `tests/agent/test_layer_separation.py` during feature implementation.
   - NEVER weaken, delete, mock out, or alter existing test assertions to make a failing test pass. Fix the production implementation instead.

3. **Zero Hardcoded Clinical Data or Un-Traced UI Shortcuts:**
   - NEVER hardcode PMIDs, clinical study rows, readmission rates, phenotype lookup tables, or `hadm_id` mappings inside `presentation/`, `engine/`, `routing/`, or `mcp/`.
   - Every literature citation or benchmark row rendered in `presentation/` MUST originate from a traced `search_literature` tool call executed in `state["tool_calls"]` during that turn.

4. **Six-Subpackage Layer Purity (`services/agent/`):**
   - `presentation/` (`a2ui.py`, `canvas.py`, `domain.py`) is a pure, stateless projection of settled engine state. It must never perform domain classification, fabricate citations, or mutate `sources` with un-traced data.
   - `telemetry/` (`set_trace_io`) must capture the final settled response and composed presentation envelope so Langfuse traces 100% match the UI output.

---

## COMMUNICATION STYLE — WRITE PLAIN ENGLISH

Everything you report to me is read by a person who did not watch you work. Write prose
that a competent engineer outside this codebase could follow on the first reading.

1. **Describe behaviour first, name code second — or not at all.**
   Say what the thing does in ordinary words. Only then, if the reader needs to find it,
   name the file or function once. Never open a sentence with an identifier.

2. **No internal names as concepts.** A private constant, a state key, a helper function
   or a field on a payload is not a concept; it is a place in the code. Do not use one as
   shorthand the reader must decode. "The evidence record", not `SessionEvidenceState`.
   "Where a turn's citations are renumbered", not "the `compact_display_numbers` path".
   Naming an identifier is allowed only when the point IS that identifier — for example
   reporting which function raised an error.

3. **One idea per sentence, in the order the reader needs them.** No arrow chains
   (`A -> B -> C`), no stacked noun phrases, no invented shorthand ("the hub", "the spoke",
   "the owner module") unless you defined it in plain words in the same message.

4. **Lead with the outcome.** What was wrong, what you did, what is true now. Evidence and
   measurements come after, and only the ones that change the reader's understanding.

5. **Plain does not mean long.** Keep the existing brevity rules: no preamble, no option
   menus, no restating what the reader already knows. A short sentence is easier to write
   plainly than a long one, so cut rather than expand.

6. **Define a term the first time you use it.** "A citation map — the table that turns the
   number a reader sees into a position in the conversation's evidence list — is stored
   with each answer." After that, "citation map" is fine.

7. **In code comments, keep the technical detail but write sentences.** A comment explains
   why the code is shaped that way to someone reading it cold. It is not a place to save
   words.

### Before and after

Cryptic:
> `_ingest_passages` copied only `retrieval`, `granularity` and `query` into an item's
> metadata, so the field Sprint 2 added to `discharge_medications` payloads was dropped on
> ingestion.

Plain:
> The step that files retrieved passages into the session record only kept three of their
> properties, so the provenance label we added last sprint was being thrown away. A
> follow-up turn could not tell a real discharge medication list from text the hospital
> course happened to mention.

Cryptic:
> `turn.py` publishes the map turn-local (`position - turn_start`); `graph.replay_messages`
> adds back `turn_offset`.

Plain:
> Each turn stores its citation numbers relative to that turn's own first piece of
> evidence, and the replay adds back that turn's starting position. That way a later turn
> always lands on the same absolute number, however the earlier turns were numbered.
