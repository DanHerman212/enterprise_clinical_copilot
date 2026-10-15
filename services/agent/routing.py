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

from dataclasses import dataclass, field

# The section vocabulary, not the chunker. The agent's image ships the vocabulary
# module and not the chunker, so importing this constant from `chunking` failed the
# container at startup and the deploy silently served the previous revision.
from services.mcp.retrieval.sections import SUMMARY_SECTIONS


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
    #: A supplementary call's result is recorded for the composer and withheld from
    #: the formatting turn: the canvas prints the score exactly, and a model handed
    #: the payload would restate it in the narrative — the duplication the risk
    #: sentence was deleted for.
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

EHR_FORMATTING = """Summarise the note sections you were given, for a clinician.
Write flowing prose under four headings, in this order and with these names: Hospital course,
Discharge diagnosis, Discharge instructions, Readmission risk & trajectory.

Write each heading on its own line as a bold line of its own, with a blank line before it, so
the interface can set it as a block above the prose it names instead of running it into the end
of the previous sentence.

The first three sections summarise the note passages you were given.

The fourth explains the readmission-risk estimate you were also given, for THIS patient: what
the named drivers mean clinically, how they connect to the hospital course and the discharge
plan, and what they imply for follow-up. Name each driver exactly as the risk result names it,
so the sentence and the bar beside it describe one factor one way.

Quote a figure only in the form the risk result states it. Do not round it, re-derive it, or
compute a percentage, a count or a score of your own: the widget prints the score to a
precision this prose cannot reproduce, and a number here that disagrees with the number beside
it is worse than no number at all. Describe magnitude qualitatively — a leading driver, a
modest contributor — unless the stated string is what makes the sentence true.

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
        wire="risk",
        slash="/risk",
        tool="predict_readmission",
        args={"hadm_id": "{hadm_id}"},
        forbidden=frozenset({"rag_search", "rag_search_sections", "discharge_medications",
                             "search_literature"}),
        # The answer is the surface, and there is no prose at all: see `prose`.
        prose=False,
    ),
    Command(
        wire="meds",
        slash="/meds",
        tool="discharge_medications",
        args={"hadm_id": "{hadm_id}"},
        forbidden=frozenset({"predict_readmission", "rag_search", "rag_search_sections",
                             "search_literature"}),
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
                             "search_literature"}),
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
        forbidden=frozenset({"predict_readmission", "rag_search", "rag_search_sections",
                             "discharge_medications"}),
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
    "search_literature",
})


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


def is_command(wire: str | None) -> bool:
    """Whether this intent is one of the deterministic commands."""
    return wire is not None and wire in BY_WIRE


def _templated_arguments(args: dict, hadm_id: int | None,
                         command: Command) -> dict:
    """`args` over `hadm_id`, refusing when an admission is needed and absent.

    Shared with `supplementary_arguments` so a second call cannot be built by a
    second set of rules.
    """
    if "{hadm_id}" in str(args.values()) and hadm_id is None:
        raise ValueError(f"{command.slash} needs an admission; none was given")
    return {k: (hadm_id if v == "{hadm_id}" else v) for k, v in args.items()}


def supplementary_arguments(
        command: Command, hadm_id: int | None) -> list[tuple[str, dict]]:
    """`(tool, arguments)` for each of this command's supplementary calls."""
    return [(spec.tool, _templated_arguments(spec.args, hadm_id, command))
            for spec in command.supplementary]


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
