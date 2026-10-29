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
and the two are told apart here rather than by whichever layer saw it last."""
from __future__ import annotations

import re

from dataclasses import dataclass, field

# The section vocabulary, not the chunker. The agent's image ships the vocabulary
# module and not the chunker, so importing this constant from `chunking` failed the
# container at startup and the deploy silently served the previous revision.
from services.agent.routing.admission_context import PREFLIGHT_TOOLS
from services.agent.routing import encounter
# `admission_phenotypes` is no longer imported here. It was used by the pre-flight
# condition lock, which existed to stop `/pubmed` requesting an estimate the surface would
# then decline to compare — and that command no longer requests one at all (Phase F.1,
# 2026-10-26). What decides the comparison now is `a2ui`/`risk_canvas`, from the chart the
# turn read; this module answers only "which tools run".
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
        # And nothing else runs on this command (Phase F.1, 2026-10-26).
        #
        # It used to run `predict_readmission` beside the search, so the turn could pin
        # this patient's estimate above the records. The comparison was defensible and the
        # contamination was not: having the estimate in the turn's tool calls made the
        # `/pubmed` formatter append a risk paragraph and a driver inventory to what the
        # clinician asked to be a literature answer ("23.0%, driven by prior inpatient
        # days"), and it made `followups` offer the chart's own risk-driver chip under a
        # question about published evidence. Two domains, one turn, and the second one
        # was never asked for.
        #
        # `/ehr` owns the chart and the estimate; `/pubmed` owns the literature. A reader
        # who wants the comparison asks for the briefing.
        # `forbidden` is therefore the whole rest of the tool surface, which is also what
        # makes the turn's tool calls exactly one thing.
        forbidden=frozenset({"rag_search", "rag_search_sections",
                             "discharge_medications", "get_lab_trends",
                             "predict_readmission"}) | PREFLIGHT_TOOLS,
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


#: The words a question uses when it wants the ESTIMATE, or where the estimate sits
#: against its threshold. Kept separate from `_RISK_DRIVER_WORDS` because the two answer
#: different questions: a driver question wants the attribution, and a score question
#: wants the number and the comparison.
_RISK_SUBJECTS = frozenset({"risk", "readmission", "probability"})
_RISK_SCORE_WORDS = frozenset({"score", "scores", "probability", "percent",
                               "percentage", "threshold"})
_RISK_SCORE_COMPARISONS = frozenset({"above", "below", "exceeds", "exceed"})


def asks_for_risk_score(question: str) -> bool:
    """Whether this question asks for the readmission estimate, or for its verdict.

    This is the second condition under which `predict_readmission` crosses into free
    text (`graph._freetext_allowed`), and it exists because of a measured deflection.

    Asked for the estimate on a thread that had never run `/ehr`, the tool was not
    bound, so the answer named the command that owns the score and stopped: "… are
    provided by the /ehr command." The reader was told to go and type something the
    assistant could have run itself. A clinician asking for a number should get the
    number.

    A driver question is excluded even though it names the score: `asks_why_risk_driver`
    is the other crossing, and it has its own reason for binding the same tool. Two
    predicates admitting the same call for different reasons would make the binding
    unauditable.

    A false positive costs one bound tool on a turn that turns out to want prose; a
    false negative is the deflection above.
    """
    tokens = (question or "").lower().translate(_PUNCTUATION).split()
    if not tokens:
        return False
    if asks_why_risk_driver(question):
        return False
    if not any(token in _RISK_SUBJECTS for token in tokens):
        return False
    if any(token in _RISK_SCORE_WORDS for token in tokens):
        return True
    return ("risk" in tokens
            and any(token in _RISK_SCORE_COMPARISONS for token in tokens))


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


#: Records in the demo cohort that are outpatient notes, not inpatient stays are
#: identified by `routing.encounter`, which reads `encounter_type` off the record's own
#: feature row. There is no id set here any more: a 30-day readmission risk and a discharge
#: medication list are non-applicable to a speech-therapy or rehab progress note, and that
#: is a fact about the encounter, not something to be recognised from an admission number.
#: `docs/agent_evaluation/00b1-outpatient-signal_measurement_271026.md` records why no
#: other field could stand in for it.

#: What `/risk` says when the record is not an admission.
#:
#: A refusal rather than an estimate, because there is no readmission to estimate: the
#: record is an outpatient visit, so a probability computed for it would be a number
#: about a stay that never happened. It is deterministic prose, not a model turn — the
def supplementary_arguments(
        command: Command, hadm_id: int | None,
        topic: str | None = None) -> list[tuple[str, dict]]:
    """`(tool, arguments)` for each of this command's supplementary calls.

    Only `/ehr` declares one: the readmission estimate that belongs beside the briefing.

    A call flagged `skipped_without_admission` is left out when there is no admission to
    fill its arguments, which is what lets a command whose primary tool takes no
    admission answer without one. `/pubmed` used to be that case and no longer declares a
    supplementary call at all (Phase F.1, 2026-10-26) — see the command's own comment for
    why a literature turn stopped requesting the estimate.

    The readmission estimate is also left out for an OUTPATIENT record: scoring an
    outpatient note answers a question nobody asked, and the number then has to be drawn
    somewhere. See `routing.encounter`.

    `topic` is the caller's own words for a command that takes them. It is accepted and
    unused, because the only rule that ever read it — the pre-flight condition lock — was
    deleted with the call it guarded, and removing the parameter would churn a signature
    two callers share for nothing.
    """
    del topic
    return [
        (spec.tool, _templated_arguments(spec.args, hadm_id, command))
        for spec in command.supplementary
        if not (spec.skipped_without_admission and hadm_id is None)
        # The estimate describes a stay. On a record that was never one, the call is not
        # made at all — the encounter type is read from the row the pre-flight already
        # fetched, so nothing here has to know which admissions those are.
        and not (spec.tool == "predict_readmission"
                 and not encounter.states_inpatient_fields())
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


# ===========================================================================
# Consolidated from `questions.py` (Phase 7).
#
# The question wording: the second half of the prompt.
#
# The model receives two texts — the system prompt in `prompts.py`, and the
# question. The question's wording used to be written in the *website* repository
# (`demo/fixtures.py`) and composed there before the call, which meant the chain
# artifact lived in two repositories: a change to half the prompt could ship
# without touching the chain, its revision, or its review.
#
# It lives here now. The website sends intent — a chip name and a patient, or free
# text and a patient — and the agent owns the words.
#
# Every string below is the one the website used to build, so the model sees
# exactly what it saw before. The change is where the wording lives, not what it
# says; anything else would be a prompt change smuggled in as a refactor, and
# prompt changes move answers.
# ===========================================================================

import re

# The starter chips the console offers, and the question each one asks.
#
# `risk` was here, and it was also `DEFAULT_TEMPLATE` below: pressing Enter on an empty
# composer with a patient selected ran the readmission model. Both left on 2026-10-26
# with the command — `/ehr` carries the estimate beside the narrative that explains it,
# and a blank composer is not a question.
CHIP_QUESTIONS = {
    "meds": "What medications was this patient discharged on?",
    "summarize": "Summarize the recent discharge notes for this patient.",
    # The literature chip. The wording says "this condition" and the admission is
    # appended below, so the model knows which patient to reduce to search terms
    # while the tool call itself carries none — `search_literature` takes no hadm_id,
    # which is what stops note text reaching a third party. See
    # docs/literature-tool-plan.md.
    "literature": "Search PubMed for recent literature on this condition.",
    # The lab-trajectory chip (Phase 4B). It exists because the free-text route was the
    # unreliable one: a question about what a measurement did could reach the reasoning
    # loop and settle empty, and a deterministic route has to be the better answer, not
    # the fallback. The wording asks about the course rather than naming an analyte,
    # because the command draws the panel — a clinician who wants one analyte can click
    # the driver bar for it, which is the drill-down that produced the request.
    "labs": "Describe the laboratory trajectories for this patient's admission.",
}

# Appended when a patient is in scope, so the model never has to ask which
# admission is meant.
ADMISSION_TEMPLATE = " For admission {hadm_id}."

# A command that takes the caller's own words as a tool argument needs a wording
# that names what was searched. `CHIP_QUESTIONS["literature"]` says "on this
# condition", which describes a retrieval step the command does not take: the
# topic is the clinician's, and the agent does not read the chart to invent one.
TOPIC_QUESTIONS = {
    "literature": "Search PubMed for recent literature on {topic}.",
}


class UnknownChip(ValueError):
    """The caller asked for a chip this chain does not define."""

    def __init__(self, chip: str):
        super().__init__(f"Unknown chip: {chip!r}")
        self.chip = chip


def own_words(*, chip=None, question=None, hadm_id=None, topic=None) -> str:
    """The question in the caller's own words, without the admission appended.

    Separate from `compose_question` because two different readers want two
    different things from the same intent. The model is asked the composed
    question, admission and all; the follow-up selector has to recognise the
    question the reader just asked, and it recognises it by the words the reader
    used. Comparing a chip against the composed text never matches — the chip
    carries no `For admission 90000015.` — so a clicked chip was offered back
    (measured in the production UI, `agent-00175-ttr`, 2026-10-24).

    Raises exactly what `compose_question` raises, for the same reasons.
    """
    if chip is not None:
        wording = CHIP_QUESTIONS.get(chip)
        if wording is None:
            raise UnknownChip(chip)
        template = TOPIC_QUESTIONS.get(chip)
        if template is not None:
            if not (topic or "").strip():
                raise ValueError(f"{chip} was asked with no topic")
            wording = template.format(topic=topic.strip())
        return wording
    if isinstance(question, str) and question.strip():
        return question.strip()
    raise ValueError("nothing to ask: no chip and no question")


def compose_question(*, chip=None, question=None, hadm_id=None, topic=None) -> str:
    """Build the question the model is asked, from the caller's intent.

    Mirrors exactly what the website used to compose, including the default that
    applies when only a patient is given. Raises `UnknownChip` for a chip that is
    not in the table, and `ValueError` when there is nothing to ask.

    `topic` is the caller's own words for a command that takes them, and it is
    separate from `question` because the two are not interchangeable: a question
    is what the model is asked, a topic is what the tool is called with, and a
    command may have the second without the first.
    """
    wording = own_words(chip=chip, question=question, hadm_id=hadm_id,
                        topic=topic)
    if hadm_id is not None:
        wording += ADMISSION_TEMPLATE.format(hadm_id=hadm_id)
    return wording


#: The admission sentence `compose_question` appends, at the end of a question. It
#: always follows a finished sentence — the composed question ends in a full stop before
#: it — so the period is part of the anchor. Without it, the default question's own
#: "for admission 90000015." was eaten and a whole sentence became a fragment.
_ADMISSION_SUFFIX = re.compile(r"(?<=[.?!])\s+For admission \d+\.\s*$", re.IGNORECASE)


def words_of(question: str) -> str:
    """A question reduced back to the words the reader used, whoever wrote it down.

    The site stores the COMPOSED question on the turn — that is what the agent was
    asked, and it is what a follow-up has to be replayed against — so every replayed
    turn carries the admission sentence the current turn's `own_words` does not.
    Fingerprinting the two against a chip therefore matched on the turn the chip was
    clicked and matched nothing afterwards, which is how a chip rehydrated on Turn 3
    (measured in the production UI, `agent-00176-ngw`, 2026-10-24): Turn 1 excluded
    it, Turn 2's stored question carried `For admission 90000015.` and did not.

    Text that never carried the suffix comes back unchanged, so this is safe to apply
    to every replayed question rather than only the ones this deployment composed.
    """
    text = " ".join(str(question or "").split())
    stripped = _ADMISSION_SUFFIX.sub("", text).strip()
    return stripped or text


# ==== Tool registries (moved from telemetry/stages.py 2026-10-29) ====
#
# Law 4: tool wire names, capability groups and the human label for each are declared
# once, here. `engine/`, `postprocess/` and `telemetry/` read this table; none of them
# keeps a copy, so a new tool is added in one place.
#
#: Tools whose result is note text. One number per passage they return, and the
#: passages are what the dose, age, disposition and lab guards verify claims against.
PASSAGE_TOOLS: frozenset[str] = frozenset(
    {"rag_search", "rag_search_sections", "discharge_medications"}
)

# One label per tool the MCP server advertises (`services/mcp/server.py`).
# Each is phrased as the action the call performs, so it stays true even when
# the call fails. A test walks the live tool list and fails if a tool is added
# without a label here, which is what stops this table drifting into fiction.
TOOL_LABELS = {
    "predict_readmission": "Reading the Risk Model",
    "rag_search": "Searching the Discharge Notes",
    "rag_search_sections": "Reading the Note Sections",
    "discharge_medications": "Reading the Discharge Medications",
    "search_literature": "Searching the Literature",
    # A lab drill-down is a chart read like any other, and it was unlabelled until
    # 2026-10-22: every `/labs` step and every clicked driver announced itself as
    # "Consulting a Tool", which is the fallback speaking for a tool this table had not
    # been taught. The specific name is the whole point of the row.
    "get_lab_trends": "Reading the Lab Trajectories",
    # The pre-flight manifest is normally hydrated before the model runs, so its
    # label is only ever drawn when the model asks for it itself.
    "get_admission_profile": "Reading the Admission Profile",
}


# Used when the MCP server advertises a tool this table has not been taught.
# Deliberately vague AND still true: a tool is being consulted. The caller
# never sees a name it cannot interpret, and the omission is logged (see
# `graph._emit`) so it is fixed rather than silently rendered.
LABEL_UNKNOWN_TOOL = "Consulting a Tool"


# Result labels. Each reports the call's own output; the failure wording exists
# so a tool that did not answer is never rendered as a count of zero, which
# would say "nothing was found" when the truth is "nothing was asked".
LABEL_RESULT_FAILED = "The Tool Did Not Respond"


LABEL_RESULT_RISK = "Read the Risk Score"


LABEL_RESULT_PASSAGES = "Read the Note Passages"


LABEL_RESULT_SECTIONS = "Read the Note Sections"


LABEL_RESULT_MEDICATIONS = "Read the Discharge Medications"


LABEL_RESULT_LITERATURE = "Read the Articles"


LABEL_RESULT_LABS = "Read the Lab Trajectories"


RESULT_LABELS = {
    "predict_readmission": LABEL_RESULT_RISK,
    "rag_search": LABEL_RESULT_PASSAGES,
    "rag_search_sections": LABEL_RESULT_SECTIONS,
    "discharge_medications": LABEL_RESULT_MEDICATIONS,
    "search_literature": LABEL_RESULT_LITERATURE,
    "get_lab_trends": LABEL_RESULT_LABS,
}


# What a returned count counts. The noun travels with the tool because one number
# means different things to different tools, and a count is only informational if
# the noun is right: three passages of a discharge note, three published records.
# "Found 3 Note Passages" was served for a PubMed search until 2026-10-12 — the
# count was correct and the noun described a different system, which is the same
# defect as a label that names a tool the caller cannot see.
RESULT_NOUNS = {
    "rag_search": "Note Passage",
    "rag_search_sections": "Note Section",
    "search_literature": "Literature Record",
    # A trajectory is one analyte's course, not a passage and not a record: the count
    # under a lab drill-down says how many series were drawn.
    "get_lab_trends": "Lab Trajectory",
}


# For a tool this table has not been taught. Vague and true, like LABEL_UNKNOWN_TOOL.
RESULT_NOUN_FALLBACK = "Result"
