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

from services.mcp.retrieval.chunking import SUMMARY_SECTIONS


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
    #: Sections this command may cite, when it is a section command.
    sections: tuple[str, ...] = ()
    #: What the formatting turn must produce. The model receives the tool result
    #: and this instruction, and has no tools, so it can only format.
    formatting: str = ""


# The four clinical sections the summary draws on are imported from the
# retrieval vocabulary (`services.mcp.retrieval.chunking`) rather than declared
# here, so the serving tool and this table cannot disagree about which sections a
# summary covers.


RISK_FORMATTING = """Write exactly two sentences for a clinician.
The first states the risk score and whether it is above or below the operating threshold.
The second names the factors that drove the score, in the order you were given, using the
clinical label you were given for each one and saying which way each pushed the score.
Use the labels exactly as they are written: do not use a feature key, and do not write a
second name for a factor the labels have already named.
Do not write the contribution numbers and do not use parentheses. The interface draws each
factor's contribution from the structured result, and a number quoted here is not one the
system can vouch for — it is removed, which leaves the sentence broken.
Do not add a citation, a disclaimer, or any clinical detail that is not in the tool result."""

MEDS_FORMATTING = """Extract the discharge medications from the note text you were given.
Return a JSON object with one key, "medications", holding an array of objects with exactly
these fields: "medication", "dosage", "frequency".
Take each value verbatim from the note text. If a field is not stated for a medication, use
an empty string. Do not infer, normalise, expand an abbreviation, or add a medication that
does not appear in the text.
Return only the JSON object, with no prose before or after it."""

EHR_FORMATTING = """Summarise the note sections you were given, for a clinician.
Write flowing prose. Group it under bold inline headers naming the section, in the order the
sections were given.

Cite with an inline marker of the form ^[1], ^[2], ^[3]: the FIRST passage you were given is
^[1], the second is ^[2], and so on, in the order the passages appear in the result. Put the
marker immediately after the first claim that draws on that passage. Cite each passage at
most once in the whole answer.

Never write a passage identifier into the prose. The result gives each passage an id such as
the note and section it came from; that id is a machine name for the passage, not a citation,
and it must not appear in an answer a clinician reads. The ^[n] marker is the only form of
citation, and it is what the interface turns into the numbered source the reader can open.

Write nothing about medications, and append no disclaimer about missing information."""

LITERATURE_FORMATTING = """Write one sentence naming the clinical topic that was searched.
Do not list the records, their titles, their abstracts, or their identifiers — the interface
renders them from the structured result. Do not summarise or interpret the findings."""


COMMANDS: tuple[Command, ...] = (
    Command(
        wire="risk",
        slash="/risk",
        tool="predict_readmission",
        args={"hadm_id": "{hadm_id}"},
        forbidden=frozenset({"rag_search", "rag_search_sections", "discharge_medications",
                             "search_literature"}),
        formatting=RISK_FORMATTING,
    ),
    Command(
        wire="meds",
        slash="/meds",
        tool="discharge_medications",
        args={"hadm_id": "{hadm_id}"},
        forbidden=frozenset({"predict_readmission", "rag_search", "rag_search_sections",
                             "search_literature"}),
        formatting=MEDS_FORMATTING,
    ),
    Command(
        wire="summarize",
        slash="/ehr",
        tool="rag_search_sections",
        args={"hadm_id": "{hadm_id}"},
        forbidden=frozenset({"predict_readmission", "rag_search", "discharge_medications",
                             "search_literature"}),
        sections=SUMMARY_SECTIONS,
        formatting=EHR_FORMATTING,
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


def tool_arguments(command: Command, hadm_id: int | None,
                   topic: str | None = None) -> dict:
    """The arguments to call the command's tool with.

    The admission is the only argument most of these tools take, and a command
    that needs one cannot run without it — refusing here is better than calling a
    tool with `None` and reporting its complaint as the answer.

    A command with `topic_arg` set takes a second argument: the caller's own
    words, passed through unaltered to the tool's argument of that name. They are
    passed through rather than composed on the way, because a search string the
    agent rewrote is a search string nobody can see.
    """
    if "{hadm_id}" in str(command.args.values()) and hadm_id is None:
        raise ValueError(f"{command.slash} needs an admission; none was given")
    arguments = {k: (hadm_id if v == "{hadm_id}" else v)
                 for k, v in command.args.items()}
    if command.topic_arg is None:
        if topic is not None:
            raise ValueError(f"{command.slash} takes no topic")
        return arguments
    if not (topic or "").strip():
        raise MissingTopic(command)
    arguments[command.topic_arg] = topic.strip()
    return arguments
