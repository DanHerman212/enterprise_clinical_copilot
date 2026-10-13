# UX workshop — capture procedure

Purpose: capture the copilot's complete output for a fixed set of actions across the three
golden-path patients, so that after the run we can compare each output against what the
action was meant to produce. Where the two disagree, the gap is recorded as a known issue.

## Patients

Names as they appear in the console. The local database and production now hold the same
87 records.

| Patient | Admission |
|---|---|
| Bruce Hollingsworth | 90000023 |
| Susan Okafor | 90000019 |
| Cynthia Petrov | 90000015 |

## Actions per patient

Four slash commands, then two typed questions, in this order:

1. `/risk` — the 30-day readmission estimate
2. `/meds` — the discharge medication list
3. `/ehr` — a summary of the discharge notes
4. `/pubmed` — recent literature on this patient's condition
5. a typed clinical question, chosen at the time
6. a second typed question, chosen at the time

The screen is refreshed between patients, so no patient sees the previous patient's output.
The command pill is cleared between turns, so a command from an earlier turn cannot carry
into a typed question.

## What is recorded for every turn

Written to `turns/<admission>-<patient>.md`, one file per patient, one section per turn:

- **action** — the slash command, or the typed question verbatim;
- **intention** — what the action was meant to produce;
- **stages** — the status sequence, in order, as displayed;
- **answer** — the complete answer text, verbatim, not an excerpt;
- **source cards** — each card's label and its full section text, read after expanding;
- **screenshot** — the file name in `screenshots/`.

The screenshot is taken before anything is expanded, so it shows what a clinician sees.

## After the run

For each of the eighteen turns, compare the recorded output with its intention. Agreement
is recorded as agreement; disagreement is recorded in `orchestrator-known-issues.md` with
the turn it came from, what was expected, what was produced, and the likely cause.
