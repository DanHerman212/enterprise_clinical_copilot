"""Validated request contract for the private agent HTTP service."""

import re
from typing import Any, NotRequired, TypedDict

from services.agent import routing
from services.agent.questions import UnknownChip, compose_question, own_words


class AgentRequest(TypedDict):
    """What a validated request resolves to.

    `kind` is the command the caller sent, or None for free text, and it selects
    the path: a command runs deterministically, free text goes to the reasoning
    loop. `hadm_id` travels beside it because a command passes the admission to
    its tool as an argument, and re-reading it back out of the composed question
    would turn a validated value into parsed text.

    `topic` is the same argument-fed value for the one command that needs the
    caller's own words: it is the search string the literature tool is called
    with, and it travels here for the same reason the admission does.

    `stream_prose` is the caller asking to be shown a command's prose as it is
    written, rather than only in the settled answer. It is a capability, not a
    setting: it defaults off, only a command whose prose is pre-validated honours
    it, and a caller that does not ask receives the frames it always received.

    `follow_ups` is the caller asking for the next questions to be proposed beneath
    the answer. Off by default for the same reason, and for a sharper one: it costs a
    model call on every turn, and the evaluation loop asks a thousand questions that no
    reader will ever see a follow-up row under.
    """

    question: str
    #: The question in the caller's own words, before the admission was appended.
    #: `question` is what the model is asked; this is what the reader asked, and it
    #: is what the follow-up selector compares a chip against. Two different readers,
    #: two different strings, both of them honest.
    own_words: str
    kind: str | None
    hadm_id: int | None
    topic: str | None
    turns: list[dict[str, Any]]
    stream_prose: bool
    follow_ups: bool


class AgentToolCall(TypedDict):
    name: str
    args: dict[str, Any]
    response: dict[str, Any]
    derivable: bool


class RecordedToolCall(TypedDict):
    name: str
    args: dict[str, Any]
    response: dict[str, Any]


class ResolvedSource(TypedDict):
    cite: int
    section: str
    text: str
    query: str


class LiteratureArticle(TypedDict):
    """One PubMed record as it arrives from the MCP tool.

    Declared here rather than imported from `services.mcp.contracts`: the agent and
    the MCP server are separate images, and this is the agent's view of a payload
    that crossed a network boundary — the same reason `AgentToolCall` is declared
    here rather than shared.
    """

    pmid: str
    title: str
    journal: str
    pub_date: str
    abstract: str
    url: str


class LiteratureResult(TypedDict):
    query: str
    returned: int
    articles: list[LiteratureArticle]
    # Present when a search legitimately matched nothing. Absent when it matched
    # something, which is why the whole result is carried rather than only the
    # articles: an empty list cannot say whether a search happened.
    note: NotRequired[str]
    # True when the search could not be made at all. The agent is told to report this
    # as unavailability rather than as absence, because claiming a thing does not
    # exist when nobody looked for it invents a fact.
    degraded: NotRequired[bool]


# Whether a tool's result can be produced again by asking the same tool the same
# question, and therefore whether a caller that stores a turn needs to keep it.
#
# The site stores a turn so a later turn can be replayed, and what it keeps of a
# tool call is the name, the arguments, and the result only when the result
# cannot be obtained again: retrieval returns discharge-note text, and a store of
# clinical prose in the web application's database is exactly what the memory
# policy refuses. A prediction score is kept, because re-deriving it re-bills the
# endpoint and the score behind a published answer should be the score that was
# shown.
#
# The two sets are exhaustive over the tools the MCP server registers, and a test
# holds them to that: a tool that is added without being classified makes the
# suite fail rather than quietly deciding by default.
# `search_literature` is derivable in the ordinary sense the comment above means:
# asking PubMed the same question again produces the same articles, so a stored turn
# gains nothing by holding on to them. It is also the one tool with no admission
# attached, so there is no note text among its results for the store to be careful
# with in the first place.
RE_DERIVABLE_TOOLS = frozenset(
    # `get_lab_trends` joins them (Phase 3, 2026-10-16) and is the clearest case of the
    # rule: its series are generated from the admission's own feature row by a seeded
    # function, so asking again returns the same numbers by construction rather than by
    # luck. It also carries no note prose — values, intervals and flags — so there is
    # nothing in its result for the store to be careful with.
    {"rag_search", "rag_search_sections", "discharge_medications",
     "search_literature", "get_lab_trends"}
)
RESULT_KEPT_TOOLS = frozenset({"predict_readmission"})


class AgentSuccess(TypedDict):
    question: str
    answer: str
    guardrail_flags: list[str]
    tool_calls: list[AgentToolCall]
    a2ui: dict[str, Any] | None
    sources: list[ResolvedSource]
    literature: LiteratureResult | None
    model: str
    code_revision: str
    mcp_transport: str


class AgentRequestError(ValueError):
    """A request failed the agent service's public input contract."""

    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class AgentResponseError(ValueError):
    """The agent produced a payload outside its HTTP success contract."""


# The complete set of fields this request accepts. Closed deliberately: an
# unknown field used to be ignored, so a caller that sent `history` received a
# confident answer that had silently dropped it — a capability gap wearing the
# costume of a working feature. Refusing is the honest answer, and it is what
# keeps the shape of a request a decision rather than an accident.
#
# `turns` is the one accepted way to carry earlier turns, and it carries them as
# structure rather than as text: the chain replays each one as its own messages
# instead of pasting a transcript into the prompt. A caller that reaches for
# `history`, `messages`, `session_id` or any other name is refused by
# `CONVERSATION_FIELDS` below, which says why rather than leaving the caller to
# infer it from an answer that quietly lost its context.
ACCEPTED_REQUEST_FIELDS = frozenset(
    {"question", "hadm_id", "chip", "turns", "stream_prose", "follow_ups",
     "run_tag", "intent"}
)

#: The affordances a surface can declare about its own request.
#:
#: A clicked risk-driver bar is not a typed question: the reader chose a control that
#: already means "explain this driver in this chart". Until Phase 01 the service could
#: only infer that from the wording, and the wording is a feature label — "what is the
#: oncology history for this patient?" says nothing structural. Declaring it removes the
#: guess, and removes the temptation to match a specific label.
#:
#: A closed vocabulary, deliberately: an intent the service does not know is refused
#: rather than ignored, because a surface that thinks it declared something and did not
#: would be silently back to the wording game.
REQUEST_INTENTS = frozenset({"risk_driver"})

# Names a caller reaches for when it expects the agent to remember the
# conversation by itself. The agent holds nothing between requests, so a turn is
# only in the context window if the caller sent it in `turns`.
CONVERSATION_FIELDS = frozenset(
    {"history", "messages", "session_id", "conversation_id", "context"}
)

# What one replayed turn carries, and how much of the conversation may be
# replayed. Bounded because every replayed turn enters the context window: an
# unbounded transcript is a cost, and a long enough one pushes the current
# question out of the window it is being asked about.
#
# `tool_calls` is what makes a replay structurally faithful. A tool result has a
# place in the message sequence, and re-inserting one requires knowing the call
# that produced it. A call carries its name and arguments; its payload is carried
# only when it cannot be re-derived (a prediction score cannot), and is left out
# when it can (a retrieval result is resolved again from the corpus, which is
# what keeps clinical text out of the store).
REPLAYED_TURN_FIELDS = frozenset({"question", "answer", "tool_calls"})
REPLAYED_TOOL_CALL_FIELDS = frozenset({"name", "args", "payload"})
MAX_REPLAYED_TURNS = 6
MAX_REPLAYED_TOOL_CALLS = 5
MAX_TURN_ANSWER_CHARS = 8000
MAX_TOOL_NAME_CHARS = 64


def _parse_tool_calls(value: Any, turn_index: int) -> list[dict[str, Any]]:
    """Validate the tool calls a replayed turn says it made.

    Strict for the same reason the rest of the contract is: a field the chain
    cannot place in the message sequence is refused rather than carried, because
    a carried field is one the model never sees while the caller believes it was
    sent. Whether a name is a tool the server actually has is not decided here —
    that is answered by the call itself, and by a refusal at replay time.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise AgentRequestError(
            "invalid_request", f"'turns[{turn_index}].tool_calls' must be a list", 400
        )
    if len(value) > MAX_REPLAYED_TOOL_CALLS:
        raise AgentRequestError(
            "invalid_request",
            f"'turns[{turn_index}].tool_calls' holds {len(value)} calls; at most "
            f"{MAX_REPLAYED_TOOL_CALLS} are replayed.",
            413,
        )

    calls: list[dict[str, Any]] = []
    for call_index, call in enumerate(value):
        where = f"'turns[{turn_index}].tool_calls[{call_index}]'"
        if not isinstance(call, dict):
            raise AgentRequestError(
                "invalid_request", f"{where} must be an object", 400
            )
        unknown = sorted(set(call) - REPLAYED_TOOL_CALL_FIELDS)
        if unknown:
            named = ", ".join(f"'{name}'" for name in unknown)
            raise AgentRequestError(
                "unsupported_field",
                f"Unsupported field(s) in {where}: {named}. A replayed tool call "
                "carries 'name', 'args' and optionally 'payload'.",
                400,
            )
        name = call.get("name")
        if not isinstance(name, str) or not name.strip():
            raise AgentRequestError(
                "invalid_request", f"{where}.name must be a non-empty string", 400
            )
        if len(name) > MAX_TOOL_NAME_CHARS:
            raise AgentRequestError(
                "invalid_request", f"{where}.name is over "
                f"{MAX_TOOL_NAME_CHARS} characters",
                400,
            )
        args = call.get("args") or {}
        if not isinstance(args, dict):
            raise AgentRequestError(
                "invalid_request", f"{where}.args must be an object", 400
            )
        payload = call.get("payload")
        if payload is not None and not isinstance(payload, (dict, str)):
            raise AgentRequestError(
                "invalid_request",
                f"{where}.payload must be an object or a string when present",
                400,
            )
        calls.append({"name": name, "args": args, "payload": payload})
    return calls


def _parse_turns(value: Any, max_question_chars: int) -> list[dict[str, str]]:
    """Validate the earlier turns a caller wants this question considered with.

    Strict on purpose, and strict in the same way the rest of the contract is:
    a turn is a question and an answer and nothing else, each a bounded string,
    and the list is capped. A field the chain does not know how to replay is
    refused rather than carried, because a carried field is one the model never
    sees while the caller believes it was sent.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise AgentRequestError("invalid_request", "'turns' must be a list", 400)
    if len(value) > MAX_REPLAYED_TURNS:
        raise AgentRequestError(
            "too_many_turns",
            f"At most {MAX_REPLAYED_TURNS} earlier turns may be replayed.",
            413,
        )

    turns: list[dict[str, str]] = []
    for index, turn in enumerate(value):
        if not isinstance(turn, dict):
            raise AgentRequestError(
                "invalid_request", f"'turns[{index}]' must be an object", 400
            )
        unknown = sorted(set(turn) - REPLAYED_TURN_FIELDS)
        if unknown:
            named = ", ".join(f"'{name}'" for name in unknown)
            raise AgentRequestError(
                "unsupported_field",
                f"Unsupported field(s) in 'turns[{index}]': {named}. A replayed "
                "turn carries 'question' and 'answer'.",
                400,
            )
        question = turn.get("question")
        answer = turn.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise AgentRequestError(
                "invalid_request",
                f"'turns[{index}].question' must be a non-empty string",
                400,
            )
        if len(question) > max_question_chars:
            raise AgentRequestError(
                "question_too_long",
                f"'turns[{index}].question' is over the "
                f"{max_question_chars} character limit.",
                413,
            )
        # A turn whose command answered with a widget and no prose carries an empty
        # `answer` legiti­mately: `/risk` renders the score and the driver bars and
        # states no sentence, by design. Requiring non-empty text made every such turn
        # unreplayable, so a follow-up about a risk score was refused with 400 before
        # the model ever saw it — a conversation that cannot continue past the command
        # a user most often wants to ask about. The turn's evidence travels in
        # `tool_calls`, where a prediction payload is kept precisely so the score
        # behind the answer can be replayed rather than re-billed.
        if not isinstance(answer, str):
            raise AgentRequestError(
                "invalid_request",
                f"'turns[{index}].answer' must be a string",
                400,
            )
        if len(answer) > MAX_TURN_ANSWER_CHARS:
            raise AgentRequestError(
                "answer_too_long",
                f"'turns[{index}].answer' is over the {MAX_TURN_ANSWER_CHARS} "
                "character limit.",
                413,
            )
        turns.append({
            "question": question.strip(),
            "answer": answer.strip(),
            "tool_calls": _parse_tool_calls(turn.get("tool_calls"), index),
        })
    return turns


def _unsupported_field_error(unknown: list[str]) -> AgentRequestError:
    """Refuse fields outside the contract, naming them.

    A conversation field gets its own wording because the refusal is the design
    decision made visible at the boundary: the product is single-turn, and
    conversation state belongs to the memory layer.
    """
    named = ", ".join(f"'{name}'" for name in unknown)
    if CONVERSATION_FIELDS.intersection(unknown):
        return AgentRequestError(
            "unsupported_field",
            f"Unsupported field(s): {named}. The agent is single-turn, so a "
            "conversation field is refused rather than silently ignored; "
            "conversation state belongs to the memory layer.",
            400,
        )
    return AgentRequestError(
        "unsupported_field",
        f"Unsupported field(s): {named}. Send 'question', or a 'chip' with a "
        "'hadm_id'.",
        400,
    )


def _leading_command(question: str) -> tuple[str, str] | None:
    """(chip, remainder) when a question opens with a known slash command.

    Matched against the command table's own slash forms, so the vocabulary this
    accepts cannot drift from the one the composer's menu renders. An unknown slash
    token is left alone: a slash is also a ratio, a path and a date separator, and
    claiming those would be a worse failure than the one this prevents.
    """
    match = re.match(r"^\s*/([A-Za-z]+)\s*(.*)$", question, re.S)
    if not match:
        return None
    name = match.group(1).lower()
    for command in routing.COMMANDS:
        if command.slash.lstrip("/").lower() == name:
            return command.wire, match.group(2).strip()
    return None


def parse_agent_request(body: Any, *, max_question_chars: int) -> AgentRequest:
    """Validate JSON-decoded input and return the agent request contract.

    Four shapes, each of which the caller used to resolve itself before
    composing the question on its side:

        {"question": "..."}                  free text
        {"question": "...", "hadm_id": 123}  free text about one patient
        {"chip": "risk", "hadm_id": 123}     a starter chip
        {"chip": "literature", "hadm_id": 123,
         "question": "dual antiplatelet therapy"}
                                             a command that needs a topic

    The wording is composed here (`questions.compose_question`) rather than by the
    caller, so half the prompt no longer lives in another repository. The result
    still carries exactly one field — the question the model is asked — so
    everything downstream, including the response contract, is unchanged.

    The field set is closed: anything outside those three is refused with
    `unsupported_field` rather than ignored, and a field implying conversation
    state says why. There is no conversation state to send, and the contract is
    where that stops being implicit.

    The result carries more fields than it used to: `kind`, the command the
    question came from, or None for free text; `hadm_id`, the validated
    admission; and `topic`, the caller's own words for a command that takes them.
    Both of the latter are tool arguments rather than question text, which is why
    they travel as values instead of being read back out of the composed string.
    The question the model is asked is still a single string, so no downstream
    consumer changed.

    Text beside a command is the command's own input when the command takes one,
    and a refusal when it does not. The contract decides that, not the route and
    not the browser, because the vocabulary is the agent's (`routing.py`) and two
    layers deciding it is how the two ends up disagreeing.

    `turns` carries the earlier turns of the conversation the caller is having,
    when there is one: the caller owns that state, this service holds none, and
    the chain replays what it is given as messages rather than as text. An empty
    list and an absent field mean the same thing, which is that this question
    stands alone.
    """
    if not isinstance(body, dict):
        raise AgentRequestError(
            "invalid_request",
            "Send a 'question', or a 'chip' with a 'hadm_id'",
            400,
        )

    unknown = sorted(set(body) - ACCEPTED_REQUEST_FIELDS)
    if unknown:
        raise _unsupported_field_error(unknown)

    hadm_id = body.get("hadm_id")
    if hadm_id is not None:
        # bool is an int in Python: without the explicit check, hadm_id=True
        # would compose a question about admission 1.
        if not isinstance(hadm_id, int) or isinstance(hadm_id, bool) or hadm_id <= 0:
            raise AgentRequestError(
                "invalid_request", "'hadm_id' must be a positive integer", 400
            )

    chip = body.get("chip")
    if chip is not None and not isinstance(chip, str):
        raise AgentRequestError("invalid_request", "'chip' must be a string", 400)

    question = body.get("question")
    if question is not None and not isinstance(question, str):
        raise AgentRequestError("invalid_request", "'question' must be a string", 400)

    intent = body.get("intent")
    if intent is not None:
        if not isinstance(intent, str) or intent not in REQUEST_INTENTS:
            raise AgentRequestError(
                "invalid_request",
                f"'intent' must be one of {sorted(REQUEST_INTENTS)}",
                400,
            )

    # A slash command typed into the question is the same intent as a chip, and it is
    # resolved HERE, before anything downstream can route it probabilistically. The
    # command vocabulary is closed and lives in one table, so this is a lookup rather
    # than an inference: `/pubmed X` is the literature command with the topic X, not a
    # sentence that happens to contain the word "pubmed".
    #
    # The client sends the chip for a command picked from the menu; this covers the
    # command typed by hand, which is the path that reached the tool-choosing model and
    # had the word "risk" decide the route (observed live 2026-10-13, where the request
    # called no tool at all and answered with a routing sentence about /risk).
    if chip is None and isinstance(question, str):
        typed_command = _leading_command(question)
        if typed_command is not None:
            chip, question = typed_command

    # Checked as a bool rather than for truthiness, for the same reason `hadm_id`
    # rejects True: the string "false" is truthy, and a caller that sent it would
    # stream prose while its own payload reads as off (UX Phase 2).
    stream_prose = body.get("stream_prose", False)
    if not isinstance(stream_prose, bool):
        raise AgentRequestError(
            "invalid_request", "'stream_prose' must be a boolean", 400
        )

    # What produced this turn, when the caller is a batch run rather than a person.
    # The evaluation harness sends its run label here and the agent puts it on the
    # trace, so an evaluation's answers can be found and scored by tag instead of by
    # guessing at a time window — which is what had to happen before this existed, and
    # it meant "delete this run's traces" could not be answered without risking a
    # production turn. Constrained because it becomes a telemetry tag: no spaces, no
    # separators that a tag query would split on, and bounded so one request cannot
    # write an unbounded label.
    run_tag = body.get("run_tag")
    if run_tag is not None:
        if not isinstance(run_tag, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", run_tag):
            raise AgentRequestError(
                "invalid_request",
                "'run_tag' must be 1-64 characters of letters, digits, dot, dash or "
                "underscore",
                400,
            )

    # Asked for rather than assumed, and type-checked the same way. The caller that
    # wants this is the one drawing a row of clickable questions under the answer; the
    # evaluation loop is not, and should not pay a model call per question for a row it
    # never renders.
    follow_ups = body.get("follow_ups", False)
    if not isinstance(follow_ups, bool):
        raise AgentRequestError(
            "invalid_request", "'follow_ups' must be a boolean", 400
        )

    # A command that takes the caller's own words is the one case where text
    # arriving beside a command is the command's own input rather than a second
    # intention. For every other command it is two intentions, and the request can
    # only be answered as one of them. This used to be resolved the quiet way —
    # the command won and the question was dropped — which served a repeat of the
    # previous answer to a clinician who had asked something new. Refusing is the
    # honest answer, and it keeps the deterministic boundary a boundary rather
    # than a preference (UX Phase 1).
    topic: str | None = None
    if chip is not None and routing.is_command(chip):
        command = routing.command_for(chip)
        typed = question.strip() if isinstance(question, str) else ""
        if command.topic_arg is not None:
            # No defensible default exists, and inventing one would mean reading
            # the patient's chart to fill it in — the implicit retrieval step
            # this phase removes. The refusal carries the prompt the clinician
            # needs, composed beside the command it names.
            if not typed:
                raise AgentRequestError(
                    "topic_required", routing.MissingTopic(command).message, 400
                )
            topic = typed
        elif typed:
            raise AgentRequestError(
                "ambiguous_intent",
                "A command and a question arrived together. Send the command on its own, "
                "or ask your question without the command.",
                400,
            )

    turns = _parse_turns(body.get("turns"), max_question_chars)

    try:
        composed = compose_question(
            chip=chip, question=question, hadm_id=hadm_id, topic=topic
        )
        asked_in_own_words = own_words(
            chip=chip, question=question, hadm_id=hadm_id, topic=topic
        )
    except UnknownChip as exc:
        # Raised before UnknownChip's parent ValueError: it is the more specific
        # answer and the caller can act on it.
        raise AgentRequestError("unknown_chip", str(exc), 400) from exc
    except ValueError as exc:
        raise AgentRequestError(
            "invalid_request",
            "Send a 'question', or a 'chip' with a 'hadm_id'",
            400,
        ) from exc

    if len(composed) > max_question_chars:
        raise AgentRequestError(
            "question_too_long",
            f"Limit is {max_question_chars} characters.",
            413,
        )
    parsed = {
        "question": composed,
        "own_words": asked_in_own_words,
        "kind": chip,
        "hadm_id": hadm_id,
        "topic": topic,
        "turns": turns,
        "stream_prose": stream_prose,
        "follow_ups": follow_ups,
    }
    # Only when one was declared, for the same reason as `run_tag`: a field that is
    # always present but usually None makes "did a control send this?" a question about
    # a value rather than about a key. Absent means the request is ordinary.
    if intent is not None:
        parsed["intent"] = intent
    # Only when one was sent. A person's turn carries no run label, and a field that is
    # always present but usually None makes "is this an evaluation turn?" a question
    # about a value instead of about a key.
    if run_tag is not None:
        parsed["run_tag"] = run_tag
    return parsed


def _draws_the_answer(payload: dict) -> bool:
    """Whether the payload carries a surface capable of drawing the answer alone.

    Combined by every caller with "and there is no prose", because either half on its
    own says the opposite of what it looks like: a surface WITH prose is an ordinary
    answer, and no surface with no prose is a failed turn that must still be refused.
    """
    surface = payload.get("a2ui")
    return isinstance(surface, dict) and bool(surface.get("messages"))


def validate_agent_success(payload: Any) -> AgentSuccess:
    """Validate the complete payload emitted by the agent HTTP service."""
    if not isinstance(payload, dict):
        raise AgentResponseError("Agent produced a malformed response.")

    required_strings = ("question", "answer", "model", "mcp_transport")
    if any(not isinstance(payload.get(field), str) for field in required_strings):
        raise AgentResponseError("Agent produced an invalid response field.")
    # An empty answer is legitimate for exactly one shape: a payload whose answer is
    # drawn by its own surface. `/risk` returns the widget and no prose, because the
    # widget prints the score, the band and the threshold (UX ruling 2026-10-13), and
    # the command's formatting turn was the only LLM call in its path. The rule stays
    # narrow on purpose: with no surface, an empty answer means nothing carries the
    # answer at all, which is still a bug.
    surface_only = _draws_the_answer(payload)
    if not payload["answer"].strip() and not surface_only:
        raise AgentResponseError("Agent produced an empty answer.")

    flags = payload.get("guardrail_flags")
    if not isinstance(flags, list) or not all(
        isinstance(flag, str) for flag in flags
    ):
        raise AgentResponseError("Agent produced malformed guardrail flags.")

    calls = payload.get("tool_calls")
    if not isinstance(calls, list):
        raise AgentResponseError("Agent produced malformed tool calls.")
    for call in calls:
        if not isinstance(call, dict):
            raise AgentResponseError("Agent produced malformed tool calls.")
        if not isinstance(call.get("name"), str):
            raise AgentResponseError("Agent produced malformed tool calls.")
        if not isinstance(call.get("response"), dict):
            raise AgentResponseError("Agent produced malformed tool calls.")
        if not isinstance(call.get("args", {}), dict):
            raise AgentResponseError("Agent produced malformed tool calls.")
        if not isinstance(call.get("derivable"), bool):
            raise AgentResponseError("Agent produced malformed tool calls.")

    # Published literature travels beside the note citations and is validated for
    # the same reason they are: the browser renders it directly, so a malformed
    # record is a broken card rather than a bad number somewhere downstream.
    literature = payload.get("literature")
    if literature is not None:
        if not isinstance(literature, dict):
            raise AgentResponseError("Agent produced malformed literature.")
        articles = literature.get("articles")
        if not isinstance(articles, list) or not isinstance(
            literature.get("returned"), int
        ):
            raise AgentResponseError("Agent produced malformed literature.")
        if literature["returned"] != len(articles):
            raise AgentResponseError("Agent produced inconsistent literature.")
        for article in articles:
            if not isinstance(article, dict) or not isinstance(
                article.get("pmid"), str
            ):
                raise AgentResponseError("Agent produced malformed literature.")
        if "degraded" in literature and not isinstance(literature["degraded"], bool):
            raise AgentResponseError("Agent produced malformed literature.")

    # The revision is what makes a stored turn explainable after a deploy. An
    # empty string is legitimate (a local run with no revision), a missing field
    # is not: it would leave the store unable to say which code answered.
    if not isinstance(payload.get("code_revision"), str):
        raise AgentResponseError("Agent produced no code revision.")

    # The same rule as the revision, with the same legitimacy for an empty
    # string: tracing is a sink, so a run with Langfuse unconfigured has no trace
    # to point at and says so by sending an empty id. What is not legitimate is
    # the field being absent or of another type — a caller stores this value, and
    # a caller that has to guess the type of a pointer guesses wrong.
    if not isinstance(payload.get("langfuse_trace_id"), str):
        raise AgentResponseError("Agent produced no langfuse trace id.")

    if payload.get("a2ui") is not None and not isinstance(payload["a2ui"], dict):
        raise AgentResponseError("Agent produced malformed A2UI data.")

    # The strict payload (UX Phase 1). Required rather than optional, unlike the
    # site's copy of this contract: the agent is the only producer of these
    # fields, so an agent revision that omitted one is a bug rather than a
    # deployment-order question. The site can be lenient about them precisely
    # because they are additive on its side.
    # A blank narrative is legitimate for exactly one shape: a payload whose answer is
    # drawn by its own surface. `/risk` returns the widget and no prose, because the
    # widget prints the score, the band and the threshold, and a sentence beneath it
    # said the same thing a second time. The rule stays narrow on purpose — a blank
    # narrative with no surface is still a bug, because then nothing carries the
    # answer at all.
    narrative = payload.get("narrative_text")
    drawn_by_surface = _draws_the_answer(payload)
    if not isinstance(narrative, str):
        raise AgentResponseError("Agent produced no narrative text.")
    if not narrative.strip() and not drawn_by_surface:
        raise AgentResponseError("Agent produced no narrative text.")

    citations = payload.get("citations")
    if not isinstance(citations, list) or not all(
        isinstance(citation, dict) for citation in citations
    ):
        raise AgentResponseError("Agent produced malformed citations.")

    components = payload.get("ui_components")
    if not isinstance(components, list):
        raise AgentResponseError("Agent produced malformed UI components.")
    for component in components:
        if not isinstance(component, dict):
            raise AgentResponseError("Agent produced malformed UI components.")
        if not isinstance(component.get("component"), str):
            raise AgentResponseError("Agent produced malformed UI components.")
        # `props` is the component's whole input, so it has to be an object even
        # when it is empty: a list would be read as positional arguments by a
        # caller that spreads it, which is a failure mode with no upside.
        if not isinstance(component.get("props", {}), dict):
            raise AgentResponseError("Agent produced malformed UI components.")

    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise AgentResponseError("Agent produced malformed sources.")
    for source in sources:
        if not isinstance(source, dict):
            raise AgentResponseError("Agent produced malformed sources.")
        if not isinstance(source.get("cite"), int) or isinstance(source.get("cite"), bool):
            raise AgentResponseError("Agent produced malformed sources.")
        if any(not isinstance(source.get(k), str) for k in ("section", "text", "query")):
            raise AgentResponseError("Agent produced malformed sources.")

    return payload


def validate_recorded_tool_call(payload: Any) -> RecordedToolCall:
    """Validate the internal trace record before it enters agent state."""
    if not isinstance(payload, dict):
        raise AgentResponseError("Recorded tool call is malformed.")
    if not isinstance(payload.get("name"), str):
        raise AgentResponseError("Recorded tool call name is invalid.")
    if not isinstance(payload.get("args"), dict):
        raise AgentResponseError("Recorded tool call arguments are invalid.")
    if not isinstance(payload.get("response"), dict):
        raise AgentResponseError("Recorded tool call response is invalid.")
    return payload