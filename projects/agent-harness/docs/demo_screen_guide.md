# Demo Screen Execution Guide — walk-through test plan

**Purpose:** a personal, step-by-step manual test of the demo skeleton, one
screen at a time. Go through each section in order, tick what you verified, and
note anything that feels wrong, noisy, broken, or missing. Then send me your
comments — I'll take them one at a time.

**Reference:** wireframes at `docs/wireframes.md` · design decisions at
`docs/ux_design_plan.md` §6/§7.

---

## 0. Run it

```bash
cd /Users/danherman/Desktop/danielmherman
.venv/bin/python manage.py runserver 8000
```

Then open **http://127.0.0.1:8000/demo/** and sign in. A temp account exists for
you to test with (change/delete it after):

- username `demo_dev` · password `demo-pass-123`

The page is the **Enterprise Clinical Copilot** shell: an ECC brand header (logo +
wordmark), notifications + user (Dr. Lena Ortiz) top-right, and a left sidebar
(Dashboard · Readmission Risk). The disclaimer about synthetic names is
temporarily removed — you're adding it back later.

**Important — what's real right now (fixture mode):**
- The **risk numbers and dots are real** (computed by running the actual serving
  predictor against every demo patient).
- **Rag passages are only real for Leonard Castellano** (the patient we tested
  live on 2026-08-11). Every other patient will honestly return *"No supporting
  note passage found"* — that is correct behavior, not a bug.
- Free-typed questions are rejected in fixture mode (endpoints are torn down);
  the message tells you to use a starter chip. That is expected.
- **Quota badge** (sidebar footer) only counts down in *live* mode — fixture
  mode makes no real calls, so it stays put. Expected.

**Checklist for section 0:** [ ] server starts · [ ] login works · [ ] page loads

---

## 1. Screen 1 — Patient list (the left rail)

**1.1** Look at the list. Each row has a **small colored dot** and a summary line.
   - [ ] Dots are: green (low), amber (borderline), red/high — plus gray for any
     unscored patient, with a legend at the bottom of the rail.
   - [ ] No text tag overlaps the patient name (tags were removed).

**1.2** The list shows **10 patients per page** with pagination below.
   - [ ] "1–10 of 32" + prev/next; Next advances to 11–20, prev comes back.
   - [ ] Search filters across **all** patients (e.g., type `e`) and re-paginates.

**1.3** Click **Leonard Castellano**. The row highlights and the thread header
   shows his name + meta like `70M · borderline · 19.5%`, and a **back arrow**
   appears at the left of the thread header.
   - [ ] Row highlights · [ ] header meta shows name + band + %
   - [ ] Back arrow visible · [ ] the ask bar (input + Ask) sits right under
     the header, close to the starter chips.

**1.4** Click the **back arrow**.
   - [ ] The thread clears to "Select a patient", the ask bar disables, the row
     de-selects, and the patient rail resets to its starting state (search
     cleared, page 1, all patients from the top). Reopening the same patient
     restores their conversation (history is kept in the browser).

**Questions for you:** Is the list too dense? Is the dot + summary enough to
scan, or do you still want the band label somewhere (e.g., only in the header,
which already has it)?

---

## 2. Screen 2 — Split pane: thread + canvas

### State A — zero-state (empty thread)
**2.1** Select a patient you haven't opened yet (e.g., **Erica Abernathy**).
   - [ ] Thread shows the chips: *Run 30-day readmission risk · Summarize recent
     discharge notes · What medications were they discharged on?*
   - [ ] Canvas shows the empty-state hint.

### State B — risk assessment
**2.2** Click **Run 30-day readmission risk**.
   - [ ] A user turn + agent turn appear; the meta line under the answer reads
     `used: predict_readmission, rag_search · fixture mode · model …`.
   - [ ] Canvas now shows **RISK** (big % , progress bar, threshold marker,
     band label) and **DRIVERS** (top ±5 feature bars with +/− values).
   - [ ] The answer cites a passage with a superscript `^[1]`.
   - [ ] The **SOURCE** widget shows exactly what's cited — one source card per
     footnote, so the count always matches the prose. Each card shows its own
     **section's text** (e.g. "Brief Hospital Course:"), not the whole note.
   - [ ] Below the turns, an **"Ask another question"** chip row appears.

**2.3** Do it again for a low-risk patient (Erica) and a borderline one (Leonard)
   — check the numbers/band colors differ sensibly.
   - [ ] Low patient ≈ green / below threshold · [ ] Leonard ≈ amber / 19.5%

### State C — notes query
**2.4** On **Leonard**, click **What medications were they discharged on?**
   - [ ] Canvas **Source** widget switches to the `medications` source
     (`Source · medications`) with **one card** matching the answer's single
     footnote.
   - [ ] The card shows `[1]` + section `discharge medications` + the **actual
     medications section text**, with a **Show full section** button.

**2.5** Click **Show full section** on a passage.
   - [ ] Text expands to the full section · [ ] button toggles to *Collapse* and
     back.

**2.6** On a patient **other than Leonard** (e.g., Erica), click **What
   medications…**.
   - [ ] The Source widget honestly shows *"No supporting note passage was found
     for this question. An empty result is a real answer…"* (expected in
     fixture mode — see §0).

### State D — citation click
**2.7** On Leonard, after the risk turn, click the `^[1]` superscript in the
   answer.
   - [ ] The Source widget re-renders for that turn's passages and the **cited
     source card highlights** (blue border), showing its section text.

> Note: the "Compare to prior assessment" chip was **removed** (2026-08-11).
> The model scores a fixed feature snapshot, so a second assessment is
> identical — comparing two identical numbers read as broken, not compelling.

**Questions for you:** Is the split (chat left, widgets right) working? Does the
canvas feel like the right place for the widgets, or too much / too little? Is
anything overlapping, clipping, or hard to read?

---

## 3. Screen 3 — Trace view

**3.1** On any patient after a run, click **Show trace** (top right).
   - [ ] The canvas swaps to a **Trace** panel listing tool calls
     (`predict_readmission(…)`, `rag_search(…)`) with their args + returned
     payloads.
   - [ ] A note about **R1 cross-patient isolation** is visible.
   - [ ] The toggle flips to **Hide trace**; clicking again restores the widgets.

**Questions for you:** Is the trace too technical for the default? (It's meant
for the technical-evaluator persona, behind the toggle.)

---

## 4. Cross-cutting checks (R8 — nothing renders empty)

**4.1** Free-typed question in fixture mode (e.g., type *"Why was this patient
   flagged?"* and press Ask).
   - [ ] A clear message appears: *"Free-text questions need the live agent
     (DEMO_FIXTURE_MODE=false). Use a starter chip."*

**4.2** Quota badge (sidebar footer) — counts down by 1 per chip question **in
   live mode only**; fixture mode makes no real calls so it stays put.
   - [ ] Present in the sidebar footer.

**4.3** Switch patients mid-conversation and back.
   - [ ] Each patient keeps their **own** thread and canvas (episodic memory) —
     the conversation does not bleed between patients.
   - [ ] The patient list dot / header always matches the selected patient.

---

## 5. How to send feedback

For each section, send me your notes in whatever granularity you like. Helpful
format per item:

```
Screen # / step:
What you did:
What you expected:
What actually happened (or a screenshot):
How important is it: (blocker / should fix / nice to have / just noting)
```

I'll work through your comments one at a time, in priority order, and we'll
re-test after each round.
