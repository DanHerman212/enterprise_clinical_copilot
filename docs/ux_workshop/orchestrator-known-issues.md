# Known issues — the copilot as it behaves today

Purpose: what the UX team needs to know about the current implementation. Two kinds of
entry: observations carried over from reading the system before this survey, and findings
from the survey itself. Each records what was expected, what happened, and the likely
cause.

---

## Carried over from prompt and screen analysis, before the survey

**1. Typed questions are routed by a rule written for the starter commands.**
The assistant classifies a typed question into two buckets: a summary request, or any
question naming one of the fixed note sections, is answered by one deterministic retrieval
that returns one passage per section; everything else goes to relevance search, which the
instructions reserve for questions "that no single section answers". Nothing in the routing
considers intent — looking something up, exploring, checking a fact — or whether the
patient's note contains the section being named. The routing was never exercised by free
text, because until 2026-10-11 the console did not transmit typed text at all; the agent
answered its default question instead.

**2. Citation markers can exceed the source cards shown.**
An answer citing three distinct passages has been seen with a single source card on the
canvas, leaving the other markers resolving to nothing the clinician can open.

**3. Developer-facing copy in a clinician-facing slot.**
On every question that does not ask for a readmission estimate, the model slot reads
"No model — no readmission estimate was requested for this question."

---

## Findings from the survey

Eighteen turns: four commands and two typed questions for each of the three golden-path
patients. Full outputs in `turns/`, screenshots in `screenshots/`. Each finding names the
turns it was observed in.

### A. Which tool runs

**A1. Typed open-ended questions take two different retrieval paths.**
Three typed questions of the same shape — open-ended, no note-section vocabulary — split
between the deterministic section path and ranked relevance search:

| Turn | Question | Path taken |
|---|---|---|
| Bruce T5 | "Was there anything about his recovery that concerned the team?" | note sections (3) |
| Cynthia T5 | "Did anything set back her recovery?" | note sections (4) |
| Susan T5 | "Was there any concern about a blood clot in her leg?" | note passages (5) |

*Expected:* one rule for typed questions. Relevance search is the path the instructions
reserve for open-ended questions; two of the three did not use it.
*Likely cause:* the routing rule keys on words that appear in the question, not on intent.
"Recovery" and "set back" gave it nothing to match, so it fell to the section path.

**A2. A literature request runs tools nobody asked for, including the model.**
Bruce T4 ran note sections, then literature, then the readmission model, and led its answer
with the risk score. Susan T4 ran note sections, note passages, literature and the model, and
answered with a clinical summary, then the risk score, then literature. Cynthia T4 was clean —
literature only, plus one note citation.
*Expected:* literature for a literature request.
*Likely cause:* no rule constrains which tools are admissible for a given command; the
instructions describe each tool's purpose but never restrict the combination.

**A3. A typed question can run the model without asking for it.** Bruce T6, "What is his
prognosis?", returned the 30-day readmission estimate and raised the model card. Nothing in
the answer says the record contains no prognosis.
*Expected:* a statement that the notes do not address prognosis.

**A4. Short factual questions run two retrievals.** Susan T6, "Is she still taking
Coumadin?", ran note sections *and* note passages for a question one of them answers.

### B. Citations and evidence

**B1. The answer cites more passages than the canvas shows cards for.** Four turns:

| Turn | Distinct citations | Cards shown |
|---|---|---|
| Bruce T3 `/ehr` | 3 (`^[1]`, `^[2]`, `^[3]`) | 1 |
| Bruce T6 typed | 3 (`^[1]`, `^[2]`, `^[4]`) | 1 |
| Cynthia T3 `/ehr` | 3 | 1 |
| Susan T5 typed | 2 | 1 |

Bruce T6 also cites `^[4]` with no `^[3]` anywhere in the answer. Markers that resolve to
nothing remove the one thing the source card exists to provide: a trace from claim to note.

**B2. A citation can point at a different section than the claim.** Bruce T2: the sentence is
about the brief hospital course, and `^[1]` is a card labelled *discharge instructions*.

**B3. A model result carries a numeric citation.** Susan T4: the risk sentence ends `^[3]`,
a citation pointing at a model output rather than a note passage.

**B4. One passage is cited repeatedly.** Cynthia T2: `^[1]` appears on all eleven medication
lines.

**B5. Two references are stacked in one marker.** Cynthia T6:
`^[PMID: 35964704, PMID: 35964703]`.

### C. What the answer says

**C1. A medication answer contradicts the card beside it.** Susan T2 and T6: the answer says
no discharge medication information is available while the adjacent card lists the home
medications and the medications on transfer. The clinician sees "none" and a list of eight.

**C2. A medication disclaimer is appended to summaries.** Bruce T3 and Susan T3 end with
"No specific discharge medication information is available for this admission." Neither turn
was a medications question.

**C3. A medication list is bulleted.** Cynthia T2: eleven bullet lines where the instructions
ask for prose.

**C4. A summary can return almost nothing.** Susan T3: the note yields one of the four summary
sections, so `/ehr` answers by listing what is absent. Correct, but the button does not deliver
what its label promises on this patient.

**C5. Retrieved literature can be irrelevant.** Cynthia T6: the third record concerns bleeding
after colorectal endoscopic dissection, offered against a question about dental surgery.

### D. Interface and status

**D1. The run never signals completion.** On all eighteen turns the status line stayed on
"Checking the Answer Against the Evidence" and the completed state was never applied — still
true minutes after the answer had rendered. The caret that expands the stage list appears only
in the completed state, so the stages the user is meant to be able to open stay unreachable,
and the run reads as permanently busy.

**D2. The literature result is labelled as note passages.** Bruce T4, Susan T4 and Cynthia T4
all show "Found N Note Passages" for records that came from PubMed.

**D3. A chip outlives its turn and overrides a typed question.** Observed during this session
before the recorded run: with the `/pubmed` pill still pinned, a typed question returned a
verbatim repeat of the previous literature answer — same 1,514 characters, same PMIDs. The
echoed question showed both the pill and the typed question, and with a chip present the
server uses the chip and discards the question. Not among the eighteen recorded turns only
because the procedure now clears the pill between turns; any clinician who types without
removing the pill will hit it.

**D4. Developer-facing copy sits in a clinician-facing slot.** Every answer that is not a risk
estimate shows "No model — no readmission estimate was requested for this question."

### E. Data

**E1. Live probabilities differ from the recorded golden path.**

| Patient | Recorded | Live | Effect |
|---|---|---|---|
| Bruce Hollingsworth | 0.1148 | 0.316788 | still above threshold |
| Susan Okafor | 0.0977 | 0.135729 | **moved from below to above** |
| Cynthia Petrov | 0.0627 | 0.090459 | still below threshold |

Susan Okafor was chosen as the reassurance case that sits just below the threshold. She is now
above it, which changes her role in the workshop.
