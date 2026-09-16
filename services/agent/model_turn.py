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


def filtered_categories(message: Any) -> list[str] | None:
    """Which content-filter categories this response was flagged for.

    Read from the ratings rather than inferred from the finish reason, because a
    rating carries `blocked` per category and says *which* harm was involved. A
    prompt that never reached the model carries its ratings under `prompt_feedback`
    instead, so both places are read.

    Returns None when nothing was flagged, which is a different statement from an
    empty list: one means the response was scored and passed, the other that no
    scoring happened at all.
    """
    flagged: list[str] = []
    meta = metadata(message)
    feedback = meta.get("prompt_feedback") or {}
    for ratings in (meta.get("safety_ratings"), feedback.get("safety_ratings")):
        if not isinstance(ratings, list):
            continue
        for rating in ratings:
            if not isinstance(rating, dict) or not rating.get("blocked"):
                continue
            category = str(rating.get("category") or "unspecified")
            if category not in flagged:
                flagged.append(category)
    return flagged or None


def served_model(message: Any) -> str | None:
    """The model version that answered, as opposed to the one we asked for.

    The record's `model` field is the pin, which is what we intended; this is what
    served the request. They are the same string today and would not be during a
    migration, which is when the distinction matters for attribution.
    """
    name = metadata(message).get("model_name")
    return str(name) if name else None


def token_usage(state: dict) -> dict | None:
    """The tokens this execution was billed for, summed over its model turns.

    Summed rather than taken from the last turn, because each turn resends the
    conversation and is billed for it — so the sum is the cost of the question,
    not a count of unique text. Answering turns, tool-calling turns and thinking
    all count.

    `cached` is a subset of `input`, billed at a discount, so it is reported
    alongside rather than subtracted: what the totals would be after the discount
    is the billing system's answer, not ours.

    Returns None when no turn reported usage at all, rather than a dict of zeros.
    A zero is a measurement somebody made; absence is not.
    """
    totals = {"input": 0, "output": 0, "total": 0, "thinking": 0, "cached": 0}
    reported = False
    for message in state.get("messages") or []:
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            continue
        reported = True
        turn_in = usage.get("input_tokens") or 0
        turn_out = usage.get("output_tokens") or 0
        totals["input"] += turn_in
        totals["output"] += turn_out
        totals["total"] += usage.get("total_tokens") or 0
        totals["thinking"] += (
            (usage.get("output_token_details") or {}).get("reasoning") or 0
        )
        totals["cached"] += (
            (usage.get("input_token_details") or {}).get("cache_read") or 0
        )
    return totals if reported else None


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
