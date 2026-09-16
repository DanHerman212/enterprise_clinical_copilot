"""How a model turn ended, read from the response instead of guessed at.

Google's requirement is that a failure is reported for what it is. The response
carries everything needed to do that, and until this module existed the chain read
none of it: a truncated answer, a refusal by the model's content filters and a
prompt that never reached the model all arrived at the caller as the same "please
retry" — which is wrong advice for two of the three, because at temperature 0 a
request that was refused reaches the same decision again.

Three signals, one place:

  - `finish_reason` on the message metadata — `STOP`, `MAX_TOKENS`, `SAFETY`,
    `RECITATION`, `SPII`, `PROHIBITED_CONTENT` and so on.
  - `safety_ratings`, per harm category, when the model scored the response.
  - `prompt_feedback.block_reason`, when the prompt itself was blocked and no
    candidate was ever produced. Google's own example of this is a prompt flagged
    for prohibited content.

The mapping to an outcome is deliberately small: "the allowance ran out" and "the
model made a decision" are the two failures a caller can act on differently, and
everything else stays unknown rather than being guessed into a category.
"""

from typing import Any

# What happened to the turn, in the terms a caller cares about.
OK = "ok"
TRUNCATED = "truncated"
REFUSED = "refused"
UNKNOWN = "unknown"

# A decision rather than a failure: these cannot be fixed by trying again.
# From Google's finish-reason table for the Gemini API.
_REFUSALS = frozenset({
    "SAFETY",
    "RECITATION",
    "SPII",
    "PROHIBITED_CONTENT",
    "BLOCKLIST",
    "IMAGE_SAFETY",
})

# The allowance ran out — thinking and answer share it on a reasoning model.
_TRUNCATIONS = frozenset({"MAX_TOKENS", "UNKNOWN_0"})

# A prompt that was not blocked reports no block reason. 0 is
# BLOCK_REASON_UNSPECIFIED, which means "not blocked" rather than "blocked for an
# unstated reason", so it must not be treated as a refusal.
_NO_BLOCK = (0, None, "", "BLOCK_REASON_UNSPECIFIED")

# The sentence a caller sees when nothing can be said about the turn.
UNKNOWN_SENTENCE = "The agent did not produce an answer. Please retry."
TRUNCATED_SENTENCE = (
    "The answer was cut off before it was complete. Please retry."
)


def metadata(message: Any) -> dict:
    """The response metadata of a message, or an empty mapping."""
    return dict(getattr(message, "response_metadata", None) or {})


def finish_reason(message: Any) -> str | None:
    """The model's own reason for stopping, upper-cased, or None if unreported."""
    reason = metadata(message).get("finish_reason")
    return str(reason).upper() if reason else None


def blocked_reason(message: Any) -> str | None:
    """Why the prompt was blocked, or None when the prompt reached the model."""
    feedback = metadata(message).get("prompt_feedback") or {}
    reason = feedback.get("block_reason")
    if reason in _NO_BLOCK:
        return None
    return str(reason)


def safety_ratings(message: Any) -> list[dict]:
    """The per-category scores the model returned, when it returned any."""
    ratings = metadata(message).get("safety_ratings")
    return list(ratings) if isinstance(ratings, list) else []


def outcome(message: Any) -> str:
    """What happened to this turn: ok, truncated, refused or unknown."""
    if blocked_reason(message) is not None:
        return REFUSED
    reason = finish_reason(message)
    if reason in _REFUSALS:
        return REFUSED
    if reason in _TRUNCATIONS:
        return TRUNCATED
    if reason in (None, "STOP"):
        return OK
    return UNKNOWN


def refusal_sentence(message: Any) -> str:
    """What to tell a caller whose request was refused.

    It deliberately does not suggest retrying. At temperature 0 the same question
    reaches the same decision, so "please retry" would be advice that cannot work —
    and in a clinical product, advice that cannot work is worse than none.
    """
    reason = blocked_reason(message) or finish_reason(message) or "unspecified"
    return (
        f"The model declined to answer this request ({reason}). "
        "Asking it again in the same way will not change the outcome."
    )


def classify(message: Any) -> tuple[str, str, str | None]:
    """Describe a turn that produced no answer.

    Returns the error code, the sentence to show the caller, and the finish reason
    to record — the last of which is None when the response did not report one.
    Called only when there is no text to serve, so a turn with no signal at all is
    reported as unavailable rather than as an answer.
    """
    reason = finish_reason(message)
    kind = outcome(message)
    if kind == REFUSED:
        return "answer_refused", refusal_sentence(message), reason
    if kind == TRUNCATED:
        return "answer_truncated", TRUNCATED_SENTENCE, reason
    return "answer_unavailable", UNKNOWN_SENTENCE, reason
