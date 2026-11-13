"""The one vocabulary the whole service uses to describe how a model turn ended.

Every provider reports the end of a response in its own words. Gemini says
`finish_reason: "SAFETY"`; Anthropic says `stop_reason: "refusal"`; an
OpenAI-compatible endpoint says `status: "incomplete"` with a reason beside it. Three
vocabularies, one question: did the turn stop, run out of room, get withheld, or
something we cannot name?

This module answers that question in four words — `stop`, `length`, `filtered`,
`unknown` — plus two facts that travel with them: whether the provider blocked the
request, and which filter categories it named.

Why a leaf module in `shared/`
------------------------------
Two layers need the answer and neither may import the other. `engine/` reads it to
decide what to do about the turn; `postprocess/` reads it to decide what sentence to
serve the reader, and `postprocess/` is forbidden to import `engine/` (Charter 13).
`shared/` is the only place both may reach, which is the same reason `labels.py` is
there.

The translation table lives here rather than in the adapter because it has exactly two
readers and they must never disagree: a table copied into `postprocess/` would be the
second copy of one decision, which is how the citation layer and the composer drifted
apart once already. The adapter imports this table rather than restating it.

What this module deliberately does not do: it does not know a provider's name, choose a
provider, or change behaviour per vendor. It maps tokens to words.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

#: The four outcomes. Every reader outside this module switches on these and nothing else.
STOP = "stop"
LENGTH = "length"
FILTERED = "filtered"
UNKNOWN = "unknown"

#: In the order a reader should think about them: an answer, a partial answer, no answer
#: by policy, no answer for a reason nobody reported.
OUTCOMES: tuple[str, ...] = (STOP, LENGTH, FILTERED, UNKNOWN)

# The four keys a normalised turn carries in `response_metadata`. Prefixed so nothing
# can collide with a provider's own key of the same name.
OUTCOME_KEY = "model_outcome"
BLOCKED_KEY = "model_blocked"
CATEGORIES_KEY = "model_blocked_categories"
NATIVE_KEY = "model_native_reason"

# ---------------------------------------------------------------------------
# The translation table.
#
# One token per line, lower-cased on the way in. A token that appears in none of the
# three sets maps to `unknown` — which is a statement about our knowledge, not about the
# provider, and is why `unknown` is a first-class outcome rather than an error.
# ---------------------------------------------------------------------------

#: The model decided it was finished. `stop_sequence` is included because a turn that
#: ended on a stop sequence ended normally; the sequence is not a failure.
STOP_TOKENS = frozenset({
    "stop", "end_turn", "completed", "complete", "finished", "eos",
    "stop_sequence", "stop_now", "end_turn_token",
})

#: The allowance ran out — thinking and answer share it on a reasoning model, so this is
#: about our own ceiling as much as about the model.
LENGTH_TOKENS = frozenset({
    "max_tokens", "length", "incomplete", "max_output_tokens",
    "max_completion_tokens", "token_limit", "truncated",
})

#: The provider withheld the reply. This set is the one `postprocess/answer.py` reads to
#: decide whether a reader is told "a filter withheld this" or "the agent did not
#: answer", so every token that has ever meant the former belongs here.
FILTER_TOKENS = frozenset({
    "safety", "recitation", "spii", "prohibited_content", "blocklist", "image_safety",
    "refusal", "refused", "content_filter", "content_filtered", "blocked",
    "harmful_content", "policy_violation",
})

#: The narrower question `postprocess/answer.py` actually asks: did a filter take a reply
#: away, as opposed to the model declining to produce one?
#:
#: These are not the same statement and the difference is user-visible. A reader told "the
#: safety filter withheld its summary" is told which mechanism acted; a RECITATION or
#: PROHIBITED_CONTENT stop names a different mechanism, and borrowing the filter's name for
#: it would misreport the turn. So a reason can be a refusal without being a withholding,
#: and `FILTER_TOKENS` is the wider of the two sets.
WITHHELD_TOKENS = frozenset({"safety", "content_filter", "content_filtered"})

#: `prompt_feedback.block_reason` values — and their equivalents — that mean the prompt
#: was NOT blocked. A provider reports the field on ordinary responses too, so treating
#: any value as a block would make every turn look withheld.
PROMPT_NOT_BLOCKED = frozenset({
    "", "block_reason_unspecified", "unspecified", "blocked_reason_unspecified",
    "none", "none_specified", "not_blocked", "0",
})

#: Where a provider states the reason it stopped, most specific first.
_REASON_KEYS = ("finish_reason", "stop_reason", "finish_details", "incomplete_reason",
                "status")

#: Where a provider states that the prompt itself never reached the model.
_PROMPT_FEEDBACK_KEYS = ("prompt_feedback", "prompt_filter_results")


def _metadata(message: Any) -> Mapping[str, Any]:
    """A message's response metadata as a mapping, or an empty one.

    Takes the message rather than its metadata because every caller has a message and
    three of them were reaching into `response_metadata` by hand.
    """
    if message is None:
        return {}
    metadata = getattr(message, "response_metadata", None)
    return metadata if isinstance(metadata, Mapping) else {}


def _token(value: Any) -> str:
    """A provider's reason as a lookup key: lower-case, no spaces, no enum wrapper."""
    if value is None:
        return ""
    # An enum member names itself; `str()` on one can give the fully-qualified form.
    name = getattr(value, "name", None)
    text = str(name if name is not None else value).strip().lower()
    return text.replace(" ", "_").replace("-", "_")


def map_token(value: Any) -> str:
    """One provider token, as one of the four outcomes."""
    token = _token(value)
    if token in STOP_TOKENS:
        return STOP
    if token in LENGTH_TOKENS:
        return LENGTH
    if token in FILTER_TOKENS:
        return FILTERED
    return UNKNOWN


def native_reason(message: Any) -> str | None:
    """The provider's own reason for stopping, upper-cased as it reported it.

    Upper-cased because that is the form the execution record has always carried and a
    reader greps, and because the emitter of the fact is a constant, not user text.
    """
    metadata = _metadata(message)
    for key in _REASON_KEYS:
        value = metadata.get(key)
        if value in (None, ""):
            continue
        text = getattr(value, "name", value)
        return str(text).upper()
    return None


def prompt_block_reason(message: Any) -> str | None:
    """Why the prompt was refused before any candidate was produced, or None.

    Read from every shape a provider uses for it. The value is returned as reported;
    `is_blocked` is the function that decides what it means.
    """
    metadata = _metadata(message)
    for key in _PROMPT_FEEDBACK_KEYS:
        block = metadata.get(key)
        if isinstance(block, list):
            block = block[0] if block else None
        if not isinstance(block, Mapping):
            continue
        for inner in ("block_reason", "blocked_reason", "reason"):
            value = block.get(inner)
            if _token(value) in PROMPT_NOT_BLOCKED:
                continue
            if value not in (None, ""):
                return str(getattr(value, "name", value))
    return None


def _ratings_in(node: Any) -> Iterable[str]:
    """The categories a ratings list flagged as blocked, in the order it names them."""
    if not isinstance(node, list):
        return ()
    found: list[str] = []
    for rating in node:
        if not isinstance(rating, Mapping):
            continue
        if not rating.get("blocked"):
            continue
        category = str(
            getattr(rating.get("category"), "name", rating.get("category"))
            or "unspecified"
        )
        if category not in found:
            found.append(category)
    return tuple(found)


def safety_ratings(message: Any) -> list[dict]:
    """The per-category scores the provider returned, when it returned any."""
    metadata = _metadata(message)
    ratings = metadata.get("safety_ratings")
    return list(ratings) if isinstance(ratings, list) else []


def blocked_categories(message: Any) -> tuple[str, ...]:
    """Which filter categories this response was flagged for, if any.

    Read from the per-category ratings rather than inferred from the reason, because a
    rating says *which* harm was involved. A prompt that never reached the model carries
    its ratings beside the block reason instead, so both places are read — and so are an
    OpenAI-compatible endpoint's `content_filter_results`, which states the same fact in
    a third shape.
    """
    metadata = _metadata(message)
    found: list[str] = []
    for rating_list in (metadata.get("safety_ratings"),):
        for category in _ratings_in(rating_list):
            if category not in found:
                found.append(category)
    for key in _PROMPT_FEEDBACK_KEYS:
        block = metadata.get(key)
        if isinstance(block, list):
            block = block[0] if block else None
        if isinstance(block, Mapping):
            for category in _ratings_in(block.get("safety_ratings")):
                if category not in found:
                    found.append(category)
            for category in _ratings_in(block.get("content_filter_results")):
                if category not in found:
                    found.append(category)
    for category in _ratings_in(metadata.get("content_filter_results")):
        if category not in found:
            found.append(category)
    return tuple(found)


def is_blocked(message: Any) -> bool:
    """Whether the provider refused the request rather than answering it."""
    return prompt_block_reason(message) is not None


def is_filtered_reason(reason: Any) -> bool:
    """Whether one reason token means the provider withheld the reply.

    The string form of the rule, for callers that hold a reason rather than a message —
    `postprocess.answer.reply_withheld` is one, and it is handed the token across a layer
    boundary that already exists.
    """
    return map_token(reason) == FILTERED


def is_withheld_reason(reason: Any) -> bool:
    """Whether one reason token means a filter took a reply away, specifically.

    Narrower than `is_filtered_reason`: every withholding is a refusal and not every
    refusal is a withholding. `postprocess/answer.py` asks this one, because the sentence it
    chooses names the filter as the mechanism.
    """
    return _token(reason) in WITHHELD_TOKENS


@dataclass(frozen=True)
class TurnOutcome:
    """How one model turn ended, in the neutral vocabulary.

    `native_reason` is kept beside the neutral word on purpose: the neutral word is what
    the code branches on, and the native token is what a person reads in a trace when the
    neutral word is `unknown`.
    """

    outcome: str
    blocked: bool
    blocked_categories: tuple[str, ...]
    native_reason: str | None

    @property
    def stop(self) -> bool:
        return self.outcome == STOP

    @property
    def truncated(self) -> bool:
        return self.outcome == LENGTH

    @property
    def refused(self) -> bool:
        return self.outcome == FILTERED or self.blocked


def derive(message: Any) -> TurnOutcome:
    """Read a turn's outcome from whatever the provider put on the message.

    A blocked prompt outranks a finish reason: if the request never reached the model,
    nothing the model said about finishing is relevant. That ordering is the one rule
    this function adds to the table.
    """
    blocked_reason = prompt_block_reason(message)
    categories = blocked_categories(message)
    reason = native_reason(message)
    if blocked_reason is not None:
        return TurnOutcome(FILTERED, True, categories, blocked_reason)
    return TurnOutcome(map_token(reason), False, categories, reason)


def read(message: Any) -> TurnOutcome:
    """The turn's outcome: the stamped one when present, otherwise derived.

    Stamping is what the adapter does at the boundary. Deriving is what makes this work
    on a message that no adapter saw — a test double, a replayed turn from a record
    written before this module existed, a chunk the streaming path accumulated. Both
    produce the same words, so a reader never has to know which happened.
    """
    metadata = _metadata(message)
    stamped = metadata.get(OUTCOME_KEY)
    if isinstance(stamped, str) and stamped in OUTCOMES:
        categories = metadata.get(CATEGORIES_KEY) or ()
        if isinstance(categories, (list, tuple)):
            categories = tuple(str(item) for item in categories)
        else:
            categories = ()
        return TurnOutcome(
            outcome=stamped,
            blocked=bool(metadata.get(BLOCKED_KEY)),
            blocked_categories=categories,
            native_reason=(
                str(metadata[NATIVE_KEY]) if metadata.get(NATIVE_KEY) else None
            ),
        )
    return derive(message)


def stamp(message: Any) -> Any:
    """Write the neutral outcome onto `message` and return it.

    Mutates the metadata mapping in place, which is safe because a message's metadata is
    per-response and never shared — and because the alternative, constructing a new
    message, would break identity for every caller that is holding the one it got back.
    A message with no metadata mapping is returned unchanged rather than being given one:
    a fake with a frozen object is not this module's to edit.
    """
    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, dict):
        return message
    derived = derive(message)
    metadata[OUTCOME_KEY] = derived.outcome
    metadata[BLOCKED_KEY] = derived.blocked
    metadata[CATEGORIES_KEY] = list(derived.blocked_categories)
    if derived.native_reason is not None:
        metadata[NATIVE_KEY] = derived.native_reason
    return message
