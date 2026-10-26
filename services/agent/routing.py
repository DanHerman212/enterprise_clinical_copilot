"""The routing table: what each command runs, and what it may not.

Phase 1 of the UX redesign. A command is deterministic — the orchestrator runs
the tool it names and formats the result. The model never chooses a tool for a
command, and the command's formatting turn is made with **no tools bound**, so
"strictly forbidden from running secondary searches" is enforced by the absence
of anything to call rather than by a sentence in an instruction.

This module is the single owner of the command vocabulary and its rules. It is
read by the deterministic executor, by the prompt that formats each result, and
by the tests that assert only the named tool ran. Two declarations of one concept
in one place is a drift hazard; two declarations in two repositories is the
hazard this table exists to remove.

Wire names
----------
The site has always sent `chip` values, and those values are what arrive here:
`risk`, `meds`, `summarize`, `literature`. The clinician-facing commands are
`/risk`, `/meds`, `/ehr`, `/pubmed`. The two vocabularies are deliberately kept
side by side here — the wire name is what the caller sends, the command name is
what a person types — so neither has to be renamed to be understood. Renaming
the wire field is Phase 2 work, when the composer is rebuilt.

Sections
--------
`summarize` names the four clinical sections the summary may draw on. They are
spelled the way the note parser spells them, because a section name is baked into
every chunk id in the search index: renaming one here without rebuilding the
index would silently retrieve nothing. The document's `hospital_course` and
`discharge_diagnoses` are `brief_hospital_course` and `discharge_diagnosis` in
storage.

Topics
------
`/pubmed` is the one command that needs the caller's own words. A literature
search has no defensible default, and the alternative — letting the model read
the patient's chart so it can invent search terms — is the implicit retrieval
step this phase removes. `topic_arg` names the tool argument those words fill. A
command without it refuses free text rather than ignoring it, because text that
arrives beside a command is either the command's own input or a second intention,
and the two are told apart here rather than by whichever layer saw it last.
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field

# The section vocabulary, not the chunker. The agent's image ships the vocabulary
# module and not the chunker, so importing this constant from `chunking` failed the
# container at startup and the deploy silently served the previous revision.
from services.agent.admission_context import PREFLIGHT_TOOLS
from services.agent.admission_phenotypes import bucket_for, phenotype_of_query
from services.mcp.retrieval.sections import SUMMARY_SECTIONS

#: What `/ehr` searches for when the sectioned retrieval returns nothing.
#:
#: `rag_search_sections` looks for four named discharge-summary headings. An
#: outpatient note has none of them, so the call legitimately returned zero
#: passages — and the answer then reported the RECORD as empty rather than those
#: sections as absent (90000051, "Summarize the note", 2026-10-21). One
#: record-wide search is what makes an absence statement a statement about the
#: record: if this returns nothing either, there really is nothing to summarize.
EHR_RECORD_QUERY = "patient history, clinical course, diagnosis, treatment and plan"


class UnknownCommand(ValueError):
    """An intent arrived that this table does not define."""

    def __init__(self, name: str):
        super().__init__(f"Unknown command: {name!r}")
        self.name = name


class MissingTopic(ValueError):
    """A command that needs the caller's words was given none.

    The message is the one a clinician reads, and it is composed here rather
    than at the refusal site so the example names the command it is about. It is
    user-facing text with no internal detail in it, which is what lets the BFF
    relay it instead of replacing it with a generic failure.
    """

    def __init__(self, command: "Command"):
        super().__init__(f"{command.slash} needs a topic")
        self.command = command
        self.message = (
            "Please include a clinical topic to search "
            f"(e.g., {command.slash} coronary artery disease)."
        )


@dataclass(frozen=True)
class ToolSpec:
    """One additional call a command makes, and the arguments it makes it with."""

    tool: str
    args: dict
    #: Whether this call is skipped when the turn has no admission.
    #:
    #: Off by default, and deliberately: a command whose supplementary call needs an
    #: admission needs the admission, and a turn that cannot run its declared calls is
    #: refused rather than quietly run short. `/pubmed` sets it True, and it is the only
    #: one — its primary tool searches the literature, which takes no patient, so the
    #: command has a real second shape and `search_literature` is the same call in both.
    #:
    #: What the flag must not become is a general "best effort" rule. A skipped call is a
    #: comparison this turn cannot make, and the surface is required to say so rather
    #: than draw a matrix with no baseline in it and leave the reader to notice.
    skipped_without_admission: bool = False


@dataclass(frozen=True)
class Command:
    """One deterministic command and the rules that bind it."""

    #: What the caller sends on the wire (the legacy `chip` value).
    wire: str
    #: What a clinician types.
    slash: str
    #: The one tool the orchestrator runs. Nothing else runs.
    tool: str
    #: The MCP tool arguments, as a template over `hadm_id`.
    args: dict
    #: The tool argument the caller's own words fill, when the command needs
    #: them. Set means the command requires free text — there is no defensible
    #: default for a literature search, and the alternative (letting the model
    #: read the chart to invent search terms) is the implicit retrieval step this
    #: phase exists to remove. Unset means the command refuses free text.
    topic_arg: str | None = None
    #: Tools that must not appear in this command's tool calls. Asserted by
    #: tests rather than trusted, because a regression here is invisible in the
    #: answer: a secondary search that succeeds looks like diligence.
    forbidden: frozenset[str] = field(default_factory=frozenset)
    #: Further calls this command makes, after `tool`.
    #:
    #: This is the single place the "a command names one tool" rule is relaxed, and
    #: it is relaxed for a reason rather than because two calls are convenient. The
    #: readmission score and the discharge narrative are two views of one admission,
    #: and a clinician made to ask twice to see them together is being asked to hold
    #: the first answer in their head while they wait for the second. One turn also
    #: keeps the figure and the prose that qualifies it inside one guardrail pass.
    #:
    #: A supplementary call's result reaches the formatting turn like any other — see
    #: `deterministic.run_command`, and the note there on why the withholding that once
    #: stood in that path was reversed. This docstring asserted the old behaviour for a
    #: day after it stopped being true, which is the failure a comment of this kind is
    #: meant to prevent.
    supplementary: tuple[ToolSpec, ...] = ()
    #: Sections this command may cite, when it is a section command.
    sections: tuple[str, ...] = ()
    #: What the formatting turn must produce. The model receives the tool result
    #: and this instruction, and has no tools, so it can only format.
    formatting: str = ""
    #: Whether this command's answer is prose at all. `/risk` sets this False: the
    #: widget prints the score, the band and the threshold, so a sentence beneath it
    #: said the same thing a second time. Composing that sentence was also the only
    #: LLM call the command made, and it was formatting arithmetic — work the widget
    #: does exactly and the model could not do at the same precision (it printed
    #: 33.0% beside a widget drawing 33.2%). With this False the command makes one
    #: tool call, composes a surface, and returns no prose at all.
    prose: bool = True

    #: Whether this command's prose may be announced as it is written. UX Phase 2
    #: granted this only where the text was pre-validated by construction — `/meds`'s
    #: narrative, written in code from verified rows — and buffered `/ehr` on the
    #: grounds that its `^[n]` markers are resolved after the turn has finished.
    #:
    #: UX Phase 4 grants it to the model-prose commands as well. What carries the risk
    #: is no longer withholding the text, it is the swap protocol: a caller may be shown
    #: prose the guardrails then correct, and that correction is reported as it happens
    #: instead of prevented (`http._prose_settled`, and the swap the client performs
    #: when the terminal frame says the settled answer differs). The Phase 2 wording is
    #: kept above rather than deleted because the risk it names has not gone away — it is
    #: paid differently now, and the cost is a clinician reading a number the system then
    #: withdraws.
    streams_prose: bool = False

    #: Whether the announced prose is the formatting turn's own output.
    #:
    #: This is about what there is to announce, not whether to announce. `/meds`'s model
    #: reply is the extraction JSON, so its prose is written in code and this is False.
    #: `/ehr` and `/pubmed` summarize retrieved material, where the model's reply IS the
    #: answer and this is True — so the formatting turn streams, and it is asked to stream
    #: only when this says the reply is what the caller wants to read.
    prose_is_model_output: bool = False

    @property
    def tools(self) -> tuple[str, ...]:
        """Every tool this command may run, primary first."""
        return (self.tool, *(spec.tool for spec in self.supplementary))


# The four clinical sections the summary draws on are imported from the
# retrieval vocabulary (`services.mcp.retrieval.chunking`) rather than declared
# here, so the serving tool and this table cannot disagree about which sections a
# summary covers.


RISK_FORMATTING = """Retired 2026-10-13 (UX ruling). The risk command returns the widget
and no prose, so this instruction has no turn to inhabit. Kept as a name rather than
deleted because the reasoning it carried is the record of why the prose went away:
an LLM was in this command's loop only to format a number the widget already prints
exactly, and it could not do it at the widget's precision.

Amended 2026-10-15 (UX remediation). The retirement above is about `/risk`, whose answer IS
the widget, and it stands untouched. A unified `/ehr` turn is the other case, and both are
governed by one rule:

  The widget owns the quantitative score and the driver bars. On a unified `/ehr` turn the
  narrative's fourth section synthesizes how those drivers relate to the clinical course
  without restating raw unformatted floats.

That is a division of labour, not a reversal of the 2026-10-13 ruling, and the difference is
what each sentence is for. The deleted sentence said the same thing the widget said, in the
widget's own terms — a second copy of one number. The fourth section says what no widget can:
why these drivers, for this patient, and what they mean for the follow-up plan."""

MEDS_FORMATTING = """Extract the discharge medications from the note sections you were given.
Return a JSON object with one key, "medications", holding an array of objects with exactly
these fields: "medication", "dosage", "frequency".

You are given TWO sections, and a medication may appear in either or both. The discharge
medications section is usually the list of drug names; the discharge instructions section
usually states the doses and frequencies, and sometimes names drugs the list omits.

Include EVERY medication that appears in either section. A drug named in the instructions but
absent from the list is still a discharge medication and must have a row: dropping a prescribed
drug is the one error this extraction must never make.

Order the rows as the discharge medications section lists them, then any drug that appears only
in the instructions, in the order it appears there.

Take each value verbatim from the text you were given. If neither section states a dosage or a
frequency for a medication, use an empty string. An empty field is the honest answer; a guessed
dose is a clinical error. Do not infer a dose from a similar drug, do not normalise or expand an
abbreviation, do not convert units or spellings, and do not add a medication, a dosage or a
frequency that does not appear in the text.

Return only the JSON object, with no prose before or after it."""

LABS_FORMATTING = """Describe the laboratory trajectories you were given, for a clinician who
asked what these measurements did during this admission.

Read the question at the top of this input. Where it names specific analytes — "her
hemoglobin and sodium", "the creatinine" — those analytes are the answer, and the prose is
about them alone. Describe the named analytes and say nothing about the rest of the panel:
a list of ten trajectories where two were asked for buries the answer, and the card already
shows every series. Where the question names no analyte, describe the panel you were given.

Write two to four sentences, in prose. Say what moved and in which direction, where each
analyte started, and how far it went. Give the values you need to make that concrete, at the
precision you were given them: the payload states each analyte's values to its own reporting
precision, and a value re-rounded in your sentence would disagree with the number in the card
beside it for no reason.

Name every abnormal reading. For each analyte you describe, state each timepoint the flag
calls abnormal, with its value and the flag's own term. Name the nadir (the lowest value) or
the peak (the highest) explicitly when the series has one that is abnormal, rather than
leaving the reader to find it in the series: "it fell to 7.1 g/dL on hospital day 3" is the
sentence that matters, and "it varied" is the sentence that loses it.

Every timepoint in the payload carries a `flag`: "normal", "low", "high", "critical_low" or
"critical_high". That flag is the reading's verdict against the reference interval, computed
before you saw it, and it alone decides whether a value is abnormal. Rely on the flag and do
not perform your own numerical comparison: do not compare a value against `reference_low` or
`reference_high`, and do not decide normality from the numbers. Where the flag says "low",
state that the value was low; where it says "high", state high; where it says "critical_low",
say critically low; where it says "critical_high", say critically high; where it says
"normal", say it was within its reference range. A value the flag calls low is never
described as normal, in range or unremarkable, however close to the interval it looks — and
where ANY timepoint in an analyte's series carries a non-"normal" flag, that analyte never
"remained within reference range", whatever its later values did.

The trajectories were generated from this admission's own recorded features. They were NOT read
from a laboratory record. Say so once, plainly, in the course of the answer — the card carries
the same statement, and a reader who assumes these are measured results has been misled by
omission. Do not overstate it, and do not repeat it in every sentence.

Describe only what the payload contains. Never invent a value, a trend, a cause or a
comparison against a normal population that is not in the reference interval you were given.
Where an analyte is inside its range, say that rather than implying a problem; where a
trajectory has no reference interval, say what it did without judging it.

Never write an admission id, a note id, or any other identifier into the prose: they are ours,
not clinical facts, and the reader cannot look them up. Write no heading — this is a short
answer about a course, not a report. Write nothing about medications, and append no disclaimer
about missing information."""

EHR_FORMATTING = """Summarise the note sections you were given, for a clinician.
Write flowing prose with these section labels, in this order:

- **Hospital course.** for an inpatient admission. Where the record is an outpatient
  encounter — the caveat in the tool result says when it is — there was no admission, so
  write **Clinical course.** instead. Never write a hospital course for a record that was
  not a stay.
- **Discharge diagnosis.** only if the passages you were given contain a discharge diagnosis.
- **Discharge instructions.** only if the passages you were given contain discharge instructions.
- **Readmission risk & trajectory.** only where a readmission-risk estimate was given to you.

A label the passages do not support is a label you omit. Omitting a section is the correct
answer; inventing one to satisfy the format is a clinical error, and a fabricated discharge plan
is the worst version of it. Never write a label and then fill it with a claim the passages do
not contain, and never write a placeholder such as "not documented" under a label of your own
choosing.

Open each section with its label in bold, followed by a period, at the START of its own
paragraph — exactly `**Hospital course.**`, or `**Clinical course.**` for an outpatient record,
and so on — then the prose of that section on the same line. The interface sets the bold run as
the section's heading, which is why it must be the first thing in the paragraph and must end
with the period: a label buried mid-sentence cannot be found, and markdown heading syntax
(`##`) is not used anywhere in this answer — it reaches the reader as literal characters while
the answer is being written.

Use bold for nothing else in this answer, and never wrap a whole section in a single bold run.

The note sections summarise the passages you were given, and a section is written only where the
passages contain what it names.

Cite each sentence with the number of the passage that supports it. Each passage in the
tool result carries a `cite` value: that is the number to write, and it is not the
position of the section you are writing. The two happen to match only by coincidence
otherwise, and a marker that resolves to another section is a citation the reader cannot
trust — a discharge diagnosis cited as the discharge instructions is the failure this
sentence exists to prevent.

Report what the record documents and stop there. Do not close with clinical advice, a vigilance
or monitoring recommendation, a follow-up plan of your own, or any statement about what the
clinician should do: those are orders this answer does not give, and a closing "clinical
vigilance ... remains appropriate" is exactly that (TF-8, 2026-10-23). A management step the
note itself states is reported as the note's, with its citation — "she was advised to follow up
with her surgeon on 09/21/2007" — so the only action in this answer is one a passage states
first.

The final section applies only where a readmission-risk estimate was also given, and it explains
that estimate for THIS patient: what
the named drivers mean clinically, how they connect to the course and the discharge
plan, and what they imply for follow-up. Name each driver exactly as the risk result names it,
so the sentence and the bar beside it describe one factor one way.

State the readmission risk using ONLY the `probability_display` and `threshold_display` strings
the risk result gives you, or describe the risk qualitatively. Never write an unrounded
floating-point number, and never write a figure of your own: the raw values are not in the
result you were given, so a decimal you produce was not read from anywhere. Do not round a
number, do not re-derive one, and do not compute a percentage, a count or a score of your own —
the widget prints the score to a precision this prose cannot reproduce, and a number here that
disagrees with the number beside it is worse than no number at all.

The drivers are given to you as names and directions, without magnitudes. Say which led and
what they mean for the follow-up plan, and describe magnitude qualitatively — a leading driver,
a modest contributor — from how the factors rank and which way they point. The bars beside
your sentence carry the quantities.

Cite with an inline marker of the form ^[1], ^[2], ^[3]: the FIRST passage you were given is
^[1], the second is ^[2], and so on, in the order the passages appear in the result. Put the
marker immediately after the first claim that draws on that passage. Cite each passage at
most once in the whole answer.

Never write a passage identifier into the prose. The result gives each passage an id such as
the note and section it came from; that id is a machine name for the passage, not a citation,
and it must not appear in an answer a clinician reads. The ^[n] marker is the only form of
citation, and it is what the interface turns into the numbered source the reader can open.

Write nothing about medications, and append no disclaimer about missing information."""

LITERATURE_FORMATTING = """Summarise the published literature you were given, for a clinician asking
the question above. The records were retrieved for that question, so the summary must answer it.

Write a concise synthesis in two to four sentences: what the retrieved evidence says about the
question, not what the question was. Do not open by naming or restating the topic — the reader
has the question in front of them, and a sentence that repeats it is the first thing they skip.

Ground every claim in the records. If the evidence is thin, mixed, or only partly answers the
question, say that plainly rather than generalising past it.

When a readmission-risk result for this patient is given to you as well, the turn is a
comparison and the comparison is what the answer is for. Say where this patient's estimate sits
against the published rates — above them, below them, or among them — and state the patient's own
estimate using ONLY the `probability_display` string the risk result gives you, exactly as it
writes it. Name the population each rate belongs to, because the record's cohort and this patient
are two different groups: a published rate is a reference point, never a claim about her, and a
sentence that lets the two blur is the failure this section must not make. A rate you quote must
be the figure the record states — do not average two rates into a new one, and do not compute a
difference the records do not state themselves.

Explain the comparison from this patient's own drivers, naming each as the risk result names it.
Where a record identifies a factor this patient also has, say so. Where none of them does, say
that the records do not address her drivers — an alignment the records do not make is one your
prose may not supply. If no usable risk result was given, write no comparison at all: a difference
needs two figures and you have one.

Cite with an inline marker of the form ^[1], ^[2], ^[3]: the FIRST record you were given is ^[1],
the second is ^[2], and so on, in the order the records appear. Put the marker immediately after
the claim that draws on that record. Cite each record at most once, and cite every record you
actually used.

The marker is the record's POSITION in the list, never its identifier: write ^[1], not
^[PMID: 35186616]. The interface resolves both, but only the position is checked against what
was retrieved, so an identifier printed by mistake may be removed from your answer.

Never write a PMID, a DOI, a journal name or an article title into the prose: the interface
draws the records from the structured result, with their links, beside this summary.

If no records were retrieved, or the search failed, write NOTHING that claims to summarise
evidence. State only that the search returned no records (or could not be completed) and do not
invent findings, authors, or a topic."""


COMMANDS: tuple[Command, ...] = (
    Command(
        wire="labs",
        slash="/labs",
        tool="get_lab_trends",
        args={"hadm_id": "{hadm_id}"},
        # Everything else is refused: this command draws one card from one call, and a
        # second retrieval on the same turn would be the model reading the chart to
        # describe the chart. `get_lab_trends` is absent from this set because it is the
        # tool this command runs, and the rule the set encodes is that no OTHER tool runs.
        forbidden=frozenset({"predict_readmission", "rag_search", "rag_search_sections",
                             "discharge_medications", "search_literature"})
                  | PREFLIGHT_TOOLS,
        formatting=LABS_FORMATTING,
        # The model's description of the trajectory IS the answer, so it is announced as
        # it is written — the same grant `/ehr` and `/pubmed` have, and for the same
        # reason. This command exists because asking for lab trajectories in free text
        # was the unreliable path (Phase 4B): the deterministic route has to be the one a
        # clinician reaches for, so it must be at least as good an experience as the free
        # text it replaces, streaming included.
        streams_prose=True,
        prose_is_model_output=True,
    ),
    # `/risk` was retired on 2026-10-26, after the charter item that opened with "the
    # `/risk` chip exists but produces no output when a patient is submitted with a blank
    # composer". It asked `predict_readmission` for the score and nothing else, and the
    # `/ehr` turn has carried the same score beside the narrative that explains it since
    # Phase 1 — so the command was a second control for a number the reader already has,
    # reachable only by typing it or by pressing Enter on an empty composer.
    #
    # The tool did not go with it. It is `/ehr`'s supplementary call, and a driver bar
    # clicked inside that chart still declares `intent="risk_driver"` to ask why a factor
    # counts. What is gone is the command, not the estimate.
    Command(
        wire="meds",
        slash="/meds",
        tool="discharge_medications",
        args={"hadm_id": "{hadm_id}"},
        forbidden=frozenset({"predict_readmission", "rag_search", "rag_search_sections",
                             "search_literature", "get_lab_trends"}) | PREFLIGHT_TOOLS,
        formatting=MEDS_FORMATTING,
        # Prose to announce, and the model's reply is not it: the narrative is written
        # in code, so `prose_is_model_output` stays False and the turn does not stream.
        streams_prose=True,
    ),
    Command(
        wire="summarize",
        slash="/ehr",
        tool="rag_search_sections",
        args={"hadm_id": "{hadm_id}"},
        # The one composite command (§3, 2026-10-15): the summary the clinician
        # asked for, and the readmission estimate that belongs beside it.
        supplementary=(ToolSpec("predict_readmission", {"hadm_id": "{hadm_id}"}),),
        # `predict_readmission` is absent from this set because this command runs
        # it. Every other command's tool is still refused here.
        forbidden=frozenset({"rag_search", "discharge_medications",
                             "search_literature", "get_lab_trends"}) | PREFLIGHT_TOOLS,
        sections=SUMMARY_SECTIONS,
        formatting=EHR_FORMATTING,
        # The model's cited summary is the answer, so it is announced as it is written.
        # Its markers are resolved after the turn, so what a caller watches arrive is
        # provisional until the citation layer has finished with it.
        streams_prose=True,
        prose_is_model_output=True,
    ),
    Command(
        wire="literature",
        slash="/pubmed",
        tool="search_literature",
        args={},
        # The one command that needs the caller's own words. `search_literature`
        # takes a query and no admission, so the query has to come from the
        # clinician: there is nothing for the agent to fill it with that would
        # not be a search of the patient's chart in order to invent one.
        topic_arg="query",
        # ...and the comparison it draws needs the patient (Phase 2, 2026-10-16).
        # The comparative matrix sets the patient's own estimate against published
        # cohort rates, so the turn fetches both. The privacy boundary is unchanged
        # and is the point of the split: `search_literature` is still handed the
        # clinician's words and nothing else — the admission goes to a second call
        # inside this process, never to the third party.
        #
        # The second call is skipped when no patient is in context rather than making
        # the command require one. This is the only command whose primary tool takes no
        # admission, and that shape is a documented capability — a clinician reading
        # about a condition with no chart open — so it stays: with a patient the turn is
        # a comparison, without one it is the literature search it has always been, and
        # the surface is required to say which. See `ToolSpec.skipped_without_admission`.
        supplementary=(
            ToolSpec("predict_readmission", {"hadm_id": "{hadm_id}"},
                     skipped_without_admission=True),
        ),
        # `predict_readmission` is absent from this set because this command runs it.
        # Every other command's tool is still refused here.
        forbidden=frozenset({"rag_search", "rag_search_sections",
                             "discharge_medications", "get_lab_trends"}) | PREFLIGHT_TOOLS,
        formatting=LITERATURE_FORMATTING,
        # The synthesis is the answer, and it cites PMIDs the citation layer resolves
        # after the turn: same swap protocol as `/ehr`.
        streams_prose=True,
        prose_is_model_output=True,
    ),
)

BY_WIRE: dict[str, Command] = {c.wire: c for c in COMMANDS}

# The tools a FREE-TEXT question may use (UX Phase 1B).
#
# Free text is the one path where the model chooses the tool, so it is the one path
# where the choice has to be bounded. A question that needs a risk score or a
# discharge medication list is not answered here: both have commands, and the
# commands exist so those answers are produced deterministically and completely
# rather than invented under a plausible heading. What is left is retrieval —
# semantic search, the summary's sections, and the literature.
#
# Written as an allow-list rather than as a list of banned tools, because the
# direction of the failure matters: a tool added to the server tomorrow is
# unreachable from free text until someone decides to admit it, instead of being
# reachable until someone remembers to ban it.
#
# Every tool the server registers is either the tool of a command or a member of
# this set; a test asserts that, because a tool that is neither is a tool nothing
# can reach.
FREETEXT_TOOLS: frozenset[str] = frozenset({
    "rag_search",
    "rag_search_sections",
    # NOT `search_literature`. It was admitted here unconditionally on 2026-10-16 and that
    # cost the chart turns: every free-text follow-up could reach PubMed, so a question
    # about the discharge note came back with an empty `LiteratureList` — "no matching
    # literature was found" — over an answer drawn from the notes, and the note citations
    # were discarded in the bargain. Published evidence is a different question from a
    # chart question and has its own cue test: `asks_for_literature`.
    #
    # Phase 01 (2026-10-22). A free-text question about medications used to be refused
    # and redirected to /meds, which answers with the extraction list and ignores the
    # conversation — so "are any of these anticoagulants?" was answered with the count
    # again. The loop can read the same structured extraction the command reads, and
    # `guardrail` verifies doses and frequencies against the note text either way.
    "discharge_medications",
    # Phase 3 (2026-10-16). A clicked risk driver arrives here as a question about a lab
    # value, and a question about a lab value is answered from a trajectory rather than
    # from a note passage. Admitting it is what lets the drill-down work at all: without
    # it the click would be answered in prose and the card would never be composed.
    "get_lab_trends",
})

#: Tools that read published evidence rather than the patient's record.
LITERATURE_TOOLS: frozenset[str] = frozenset({"search_literature"})

# The cue set for "is this a question about published evidence?".
#
# Restored from `d23684a^` (Phase 02.5, 2026-10-22) after `search_literature` was found
# bound on every free-text turn. Multi-word cues are matched as adjacent words, so
# "systematic review" and "meta-analysis" (normalised to two words) are one cue each
# rather than two unrelated ones.
_LITERATURE_CUES: tuple[tuple[str, ...], ...] = (
    ("literature",), ("evidence",),
    ("study",), ("studies",),
    ("trial",), ("trials",),
    ("meta", "analysis"), ("metaanalysis",),
    ("systematic", "review"),
    ("pubmed",), ("guideline",), ("guidelines",),
    ("efficacy",), ("effectiveness",), ("benchmark",), ("benchmarks",),
)


def asks_for_literature(question: str) -> bool:
    """Whether the question asks for published evidence rather than a record value.

    The gate on `search_literature` for a free-text turn. Deliberately a cue test rather
    than a model decision: whether PubMed is consulted is a fact about the request, and
    a turn that has to spend a model call to discover it has already spent the call.

    Empty when nothing matches, and empty then means the loop does not search the
    literature at all — a chart question answered from the chart never carries a
    published-records surface it did not ask for.
    """
    text = " " + re.sub(r"[^a-z0-9]+", " ", str(question or "").lower()) + " "
    for cue in _LITERATURE_CUES:
        if " " + " ".join(cue) + " " in text:
            return True
    return False


def command_for(wire: str) -> Command:
    """The command for a wire intent, or `UnknownCommand`.

    Raises rather than returning None: an intent that reaches this table has
    already passed the caller's allowlist, so an unknown one is a disagreement
    between two layers about the vocabulary, and that is a bug worth a 4xx rather
    than a silent fall back to the reasoning loop.
    """
    try:
        return BY_WIRE[wire]
    except KeyError:
        raise UnknownCommand(wire) from None


def command_for_slash(slash: str) -> Command | None:
    """The command a typed `/name` refers to, or None when nothing answers to it.

    Exists so a mistyped command can be refused at the seam instead of being answered
    as a question. `/ehrr` used to run an 18-step reasoning turn, spend the whole tool
    budget and come back with a `Reconciled` badge — a correction notice shown for a
    typo (2026-10-17).
    """
    token = (slash or "").strip().lower()
    for command in COMMANDS:
        if command.slash == token:
            return command
    return None


def is_command(wire: str | None) -> bool:
    """Whether this intent is one of the deterministic commands."""
    return wire is not None and wire in BY_WIRE


#: Question templates a canvas control composes for itself.
#:
#: A clicked risk driver is not a typed question. It arrives through the same door as
#: one — `askFollowUp` fills the composer and submits the form, so the wire carries
#: free text — but the reader chose a control that already means "explain this driver in
#: this chart".
#:
#: The free-text ROUTER this once fed has been removed (Phase 01, 2026-10-22): every
#: question that is not a command now reaches the reasoning loop, which is where an
#: affordance was always heading. What remains is the second caller, and it is still
#: needed: `asks_for_administrative_detail` must not read "as a readmission risk
#: driver" as a request for a feature field, because the answer lives in the model's
#: attribution and in the notes.
_AFFORDANCE_TEMPLATES: tuple[tuple[str, ...], ...] = (
    # `driverQuestion(label)` in the renderer's RiskInfluencers widget.
    ("as", "a", "readmission", "risk", "driver"),
)



#: Punctuation becomes a space before tokenising, so `hemoglobin,` and `(RDW)` are the
#: words they look like. Everything the matchers below see is `[a-z0-9]+` runs.
_PUNCTUATION = str.maketrans({ch: " " for ch in ".,;:!?()[]{}\"'/\\-–—"})


def _contains_affordance_template(tokens: list[str]) -> bool:
    """Whether these tokens came from a control rather than a person."""
    for template in _AFFORDANCE_TEMPLATES:
        span = len(template)
        if any(tokens[i:i + span] == list(template)
               for i in range(len(tokens) - span + 1)):
            return True
    return False


#: Vocabulary of questions the pre-flight feature manifest answers on its own.
#:
#: Not a router — nothing here decides what the answer IS. The model still writes it.
#: This decides one thing only: whether the graph may skip the retrieval it otherwise
#: forces on an answer that called no tool.
#:
#: That force exists to stop the model answering a question about the chart from
#: memory. But the manifest is evidence, and it is the only evidence for these fields:
#: a discharge note never states a length of stay, an admission type or an insurer. So
#: on 2026-10-17 the force fired on "what is the patient length of stay?", sent the
#: model looking through notes that cannot contain the number, and replaced the correct
#: answer (17.29 days, from the row the model was scored on) with "the discharge note
#: does not document this" — because only the LAST message becomes the answer.
#:
#: Phrases, matched as whole tokens like the free-text router, because "age" is inside
#: "dosage" and "los" is inside "closely".
_ADMINISTRATIVE_PHRASES: tuple[str, ...] = (
    "length of stay", "los", "days in hospital", "how long was the stay",
    "admission type", "type of admission", "admitted as", "emergency or elective",
    "discharge location", "discharged to", "discharge disposition",
    "insurance", "insurer", "payer",
    "age", "how old", "anchor age",
    "sex", "gender", "male or female",
    "prior admission", "prior admissions", "previous admissions", "admission count",
    "prior inpatient days", "inpatient days before",
    "ed visits", "emergency department visits",
    "race", "ethnicity",
)


#: Words that mean the caller wants note-derived content — a plan, a course, an outcome,
#: a reason — and so cannot be answered by a manifest field however the question starts.
#:
#: Added 2026-10-18. `asks_for_administrative_detail` matched on ANY administrative
#: phrase, so the Turn 2 follow-up "What was the discharge disposition and follow-up
#: plan?" qualified on "discharge disposition" — a real feature field — and the graph
#: skipped retrieval for the half of the question that only the note holds. The model
#: then answered with no tool call and invented a disposition ("discharged to ABCD"),
#: two of the ten pilot failures. A question that asks for the manifest AND the note
#: needs the note.
_NOTE_REQUIRED_TOKENS: tuple[str, ...] = (
    "plan", "plans", "planned", "follow", "followup", "instructions",
    "course", "outcome", "outcomes", "summarize", "summarise", "summary",
    "why", "explain", "narrative", "reason", "reasons",
)


def asks_for_administrative_detail(question: str) -> bool:
    """Whether this question is answered by the feature manifest rather than the notes.

    Used to withhold the forced retrieval, never to answer: a false positive means
    the model is allowed to reply from the manifest, and the manifest is verified
    ground truth for exactly this vocabulary. A false negative costs the searches it
    cost before this existed.
    """
    tokens = (question or "").lower().translate(_PUNCTUATION).split()
    if not tokens:
        return False
    if _contains_affordance_template(tokens):
        return False
    if any(token in _NOTE_REQUIRED_TOKENS for token in tokens):
        return False
    for phrase in _ADMINISTRATIVE_PHRASES:
        words = phrase.split()
        span = len(words)
        if span == 1:
            if words[0] in tokens:
                return True
        elif any(tokens[i:i + span] == words
                 for i in range(len(tokens) - span + 1)):
            return True
    return False


#: A question that asks WHY a named factor matters to the risk model, rather than what
#: the risk is. On 2026-10-21 "What in this patient's chart explains creatinine as a
#: readmission risk driver?" was answered with the creatinine series alone: the agent
#: could reach the values but not the model's attribution, so it described the numbers
#: and stopped (a6e4ca01, responsiveness 1, everything else 3).
#:
#: A false positive binds one extra tool to one turn, which costs a call; a false
#: negative returns the answer that failed. Both words are required — "risk" alone is a
#: score question, which /risk owns and this must not answer.
_RISK_DRIVER_WORDS = frozenset({"driver", "drivers", "drives", "drive", "driving"})
_WHY_WORDS = frozenset({"why", "explain", "explains", "explained", "explaining",
                        "reason", "because", "contributes", "contribute",
                        "contributed", "contributing", "contribution",
                        "contributions"})


def asks_why_risk_driver(question: str) -> bool:
    """Whether this question asks for the model's reasoning about a risk driver."""
    tokens = (question or "").lower().translate(_PUNCTUATION).split()
    if "risk" not in tokens and "readmission" not in tokens:
        return False
    return (any(token in _RISK_DRIVER_WORDS for token in tokens)
            and any(token in _WHY_WORDS for token in tokens))


def wants_risk_driver(question: str, intent: str | None = None) -> bool:
    """Whether this turn should bind `predict_readmission` in the reasoning loop.

    Two ways in, and the second is the one that scales. Wording catches a question
    asked in prose; a declared affordance catches a clicked control, whose text is a
    feature LABEL and therefore says nothing structural about why it was clicked —
    "what is the oncology history for this patient?" is a driver drill-down when a bar
    sent it and an ordinary history question when a person typed it.

    A false positive costs one bound tool on one turn; a false negative returns an
    answer about the chart when the question was about the model's reasoning.
    """
    return intent == "risk_driver" or asks_why_risk_driver(question)


#: How a question names one of the analytes the lab series carries.
#:
#: Two vocabularies meet here, and this list is the only place they are reconciled. A
#: chip is worded from the SERIES label — `dependencies/labs/synthetic.Analyte.label`:
#: "Red blood cells", "Monocytes", "eGFR", "NT-proBNP" — while this predicate began as
#: the model's feature language, which is why it carried five singular stems.
#:
#: On 2026-10-25 a `/labs` follow-up read "How does the abnormal discharge Red blood
#: cells (3.47 x10^6/uL) compare to its reference range and trend?" and this returned
#: False: `\b` after "cell" has no boundary before the plural "s". Eight of the ten
#: labels the series can produce could not satisfy it. A test now holds this list to
#: `synthetic.PANEL`, so an analyte cannot be added without a name here.
_LAB_ANALYTE_WORDS = (
    # The five the model scores on, whose driver drill-downs are the other caller.
    "hemoglobin", "haemoglobin", "hgb", "hb",
    "sodium",
    "red blood cell", "rbc", "erythrocyte",
    "rdw", "red cell distribution width",
    "monocyte",
    # The clinical panel: carried by the series, never model inputs.
    "creatinine",
    "egfr", "gfr",
    "potassium",
    "white blood cell", "wbc", "leukocyte",
    "nt-probnp", "bnp",
)

#: Every word above, and its plural. The series labels are written in the plural that
#: clinicians use ("Red blood cells"), so a singular-only stem is the bug this prevents.
_LAB_ANALYTE = re.compile(
    r"\b(?:" + "|".join(re.escape(word) + "s?" for word in _LAB_ANALYTE_WORDS) + r")\b",
    re.IGNORECASE,
)


def asks_about_a_lab_analyte(question: str) -> bool:
    """Whether this question is about a laboratory measurement rather than a count.

    The distinction decides one tool on one kind of turn. A risk-driver question is
    answered from the model's attribution plus whatever documents it; when the driver IS
    a lab value ("why is the RDW a driver?") that documentation is the series, and
    `get_lab_trends` is how it is read. When the driver is not ("why do prior inpatient
    days drive this?"), the series documents nothing the question asked — and calling it
    anyway put a lab panel in `state["tool_calls"]`, which then offered an unrelated lab
    follow-up chip under a utilization answer (measured in the production UI,
    `agent-00176-ngw`, 2026-10-24).

    Negative for anything that names no analyte, which is the safe direction: the tool
    is withheld, and a note search still answers what the record says.
    """
    return bool(_LAB_ANALYTE.search(question or ""))


def _templated_arguments(args: dict, hadm_id: int | None,
                         command: Command) -> dict:
    """`args` over `hadm_id`, refusing when an admission is needed and absent.

    Shared with `supplementary_arguments` so a second call cannot be built by a
    second set of rules.
    """
    if "{hadm_id}" in str(args.values()) and hadm_id is None:
        raise ValueError(f"{command.slash} needs an admission; none was given")
    return {k: (hadm_id if v == "{hadm_id}" else v) for k, v in args.items()}


#: Records in the demo cohort that are outpatient notes, not inpatient stays.
#:
#: A 30-day readmission risk and a discharge medication list are clinically
#: non-applicable to a speech-therapy or rehab progress note: there was no admission to
#: be readmitted after. The model will still score them, because the feature row exists,
#: and `/ehr` used to draw "Readmission risk & trajectory" with a 5.7% widget for a
#: patient whose entire chart is three voice-therapy sessions (2026-10-17).
#:
#: Declared here, in the one module that owns the command vocabulary, so the surface and
#: the eval harness can both read it rather than each carrying a copy.
OUTPATIENT_HADM_IDS = frozenset({90000028, 90000040, 90000051, 90000054, 90000073})

#: The instruction that tells a prompt the record is an outpatient encounter.
#:
#: One text, both paths. `deterministic._NO_READMISSION` appends it to a command's
#: formatting input and `graph._context_block` appends it to the free-text system prompt,
#: because the defect it prevents is a property of the RECORD and not of the route: 90000073
#: wrote "this admission" about a speech-therapy visit from the `/ehr` path, and 90000073
#: "Summarize the note" wrote it from the free-text path, where nothing had ever told the
#: loop the record was not a stay (2026-10-21).
#:
#: It is a constant here rather than in either caller because the vocabulary that decides
#: which admissions are outpatient lives here, and a second copy of this instruction would
#: drift from `OUTPATIENT_HADM_IDS` the first time one of them was edited.
OUTPATIENT_ENCOUNTER_NOTICE = (
    "OUTPATIENT ENCOUNTER: this record is an outpatient note — a visit, not an inpatient "
    "stay. There was no admission, no hospital course, no discharge and no readmission risk "
    "to estimate, so none of those words applies to it. Do not write the words "
    '"admission", "hospital course", "discharge diagnosis" or "discharge instructions", do '
    "not write a 'Readmission risk & trajectory' section, do not state a risk percentage "
    "and do not describe risk drivers. Write \"this encounter\" or \"this visit\" instead of "
    '"this admission", and write "Clinical course" instead of "Hospital course". A heading '
    "the passages do not support is omitted, never filled with a claim.")


#: What `/risk` says when the record is not an admission.
#:
#: A refusal rather than an estimate, because there is no readmission to estimate: the
#: record is an outpatient visit, so a probability computed for it would be a number
#: about a stay that never happened. It is deterministic prose, not a model turn — the
#: `/risk` command makes no model call at all, and asking one here would let the score
#: come back in words.
OUTPATIENT_RISK_REFUSAL = (
    "This record is an outpatient encounter, not a hospital admission, so there is no "
    "readmission risk to estimate. There was no stay to be readmitted from — ask about "
    "this visit instead, or run the command on an inpatient admission."
)

#: Served by `/ehr` on an outpatient record whose note yields nothing to summarise.
#:
#: Added 2026-10-23 (TF-6). The formatter was handed an empty record and abstained —
#: "I don't have an answer for that" — which named neither the encounter nor its
#: consequence, and left the turn's trace row with no output because the state it came
#: from carried no answer text. The notice says what the record is, what it therefore
#: does not hold, and what remains available; it invents no stay, no length of stay and
#: no discharge list.
OUTPATIENT_EHR_NOTICE = (
    "This record is an outpatient encounter, not a hospital admission, so there is no "
    "discharge summary to review: no hospital course, discharge diagnosis or discharge "
    "medication list exists for this visit, and nothing here should be read as an "
    "inpatient stay. Ask about this visit instead, or run the command on an inpatient "
    "admission."
)


def comparison_applies(command: Command, hadm_id: int | None,
                       topic: str | None) -> bool:
    """Whether the turn has anything to compare this patient against.

    The condition-lock has two halves and this is the half the tools answer. `a2ui`'s
    `_draws_matrix` already refuses to DRAW the cohort table when a `/pubmed` query is
    about a condition this patient does not have, but the estimate was still requested
    and still reached the formatter — so the model wrote the comparison paragraph the
    surface had just declined to illustrate. Observed on 90000015 (an orthopedic
    admission): `/pubmed congestive heart failure` produced an unprompted readmission-risk
    paragraph beside a heart-failure literature list.

    The rule is the surface's rule, applied before the call rather than after it: on a
    literature command the estimate is requested only when the query's subject resolves to
    the patient's own phenotype bucket. A query that names no condition, or an admission
    the reviewed map does not carry, resolves to a mismatch and the estimate is not
    requested — with no second figure there is nothing honest to compare, and
    `LITERATURE_FORMATTING` already says to write no comparison then.

    A non-literature command is untouched: `/ehr`'s estimate is the briefing itself.
    """
    if command.wire != "literature" or hadm_id is None:
        return True
    query = (topic or "").strip()
    if not query:
        # No topic never reaches a literature command's tools (`tool_arguments` refuses
        # it), so this is only the unit-test shape. Keep the declared call there: a rule
        # that silently drops calls whenever it cannot evaluate them is not a rule.
        return True
    bucket = bucket_for(hadm_id)
    return bool(bucket) and phenotype_of_query(query) == bucket.key


def supplementary_arguments(
        command: Command, hadm_id: int | None,
        topic: str | None = None) -> list[tuple[str, dict]]:
    """`(tool, arguments)` for each of this command's supplementary calls.

    A call flagged `skipped_without_admission` is left out when there is no admission to
    fill its arguments, which is what lets `/pubmed` answer a patient-less literature
    search with the one call it has always made. Tested in both directions: the skip must
    not become the silent default for a command that needs the patient.

    The readmission estimate is also left out for an OUTPATIENT record: scoring an
    outpatient note answers a question nobody asked, and the number then has to be drawn
    somewhere. See `OUTPATIENT_HADM_IDS`.

    `topic` is the caller's own words for a command that takes them, and it is what
    `comparison_applies` needs to decide whether a literature turn has a comparison at
    all. It is None for every command that does not take free text.
    """
    comparisons = comparison_applies(command, hadm_id, topic)
    return [
        (spec.tool, _templated_arguments(spec.args, hadm_id, command))
        for spec in command.supplementary
        if not (spec.skipped_without_admission and hadm_id is None)
        and not (spec.tool == "predict_readmission" and hadm_id in OUTPATIENT_HADM_IDS)
        and not (spec.tool == "predict_readmission" and not comparisons)
    ]


def tool_arguments(command: Command, hadm_id: int | None,
                   topic: str | None = None) -> dict:
    """The arguments to call the command's primary tool with.

    The admission is the only argument most of these tools take, and a command
    that needs one cannot run without it — refusing here is better than calling a
    tool with `None` and reporting its complaint as the answer.

    A command with `topic_arg` set takes a second argument: the caller's own
    words, passed through unaltered to the tool's argument of that name. They are
    passed through rather than composed on the way, because a search string the
    agent rewrote is a search string nobody can see.
    """
    arguments = _templated_arguments(command.args, hadm_id, command)
    if command.topic_arg is None:
        if topic is not None:
            raise ValueError(f"{command.slash} takes no topic")
        return arguments
    if not (topic or "").strip():
        raise MissingTopic(command)
    arguments[command.topic_arg] = topic.strip()
    return arguments
