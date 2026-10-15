"""Three follow-up questions for the turn that was just answered.

Why this is a model call and not a table. The useful follow-up is about THIS patient and
THIS answer — the next thing a clinician would want after reading the one they just read
— and a table of generic prompts ("What is the risk?", "Show the medications") is exactly
the primitive chip row this replaces. So the questions are proposed by the model, and
nothing here decides what they should say.

What this module does own is the shape and the failure. They arrive as one JSON array of
strings, are cleaned and capped here, and *any* failure — a timeout, a refusal,
unparseable JSON, an empty list — becomes an empty list rather than an error. A turn that
cannot propose follow-ups is a turn with no follow-up row. It is not a failed answer, and
it must never become one: the answer is already written by the time this runs, and losing
it to a cosmetic call would be the worst trade in the chain.

The scope is passed in rather than inferred, because it is already decided
deterministically from the tool calls (`a2ui.turn_scope`) and it changes what a useful
next question is: a chart turn's follow-ups should ask about the record, a literature
turn's about the evidence.
"""

from __future__ import annotations

import asyncio
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

# One place configures the model for this service (`graph._build_llm` sets the model id
# and the per-call bound that the chain's timeout budget depends on). Building a second
# client here would be a second place for those settings to drift, so the private helper
# is imported deliberately — it is the same service, and the alternative is a copy.
from services.agent.graph import _build_llm
from services.agent.message_text import text_of

logger = logging.getLogger(__name__)

#: How many questions a turn offers. Three is the documented contract; a model that
#: returns more is cut here rather than trusted to have counted.
FOLLOW_UP_COUNT = 3

#: This call is a flourish on an answer that has already been written, so it is given a
#: small budget and no retries. It is also the only model call in the chain whose failure
#: is invisible, which is why the timeout is short: a reader waiting on the settled answer
#: should not be able to tell whether this ran.
FOLLOW_UP_TIMEOUT_SECONDS = 8.0

_INSTRUCTION = """\
You propose the next questions a clinician would ask after reading an answer about a
patient's discharge notes.

Return ONLY a JSON array of exactly {count} strings, and nothing else — no prose before
it, no explanation after it, no markdown fence. Each string is a complete question in the
clinician's own voice, standing alone with no reference to "the answer above".

{scope_line}

Ask about what the answer did NOT cover and about what a careful reader would want
checked next. Do not restate the question that was just answered, do not ask the reader
to clarify anything, and do not propose anything the record cannot answer.\
"""

_PATIENT_SCOPE = (
    "This answer rests on the patient's own chart. Ask about this patient: their notes, "
    "their medications, their results, what the record does and does not say."
)

_GENERAL_SCOPE = (
    "This answer rests on published literature rather than the patient's chart. Ask about "
    "the evidence itself — what else is published, how strong it is, where it disagrees."
)

#: A follow-up is one line in a row of clickable rows, so the cap is a length and not a
#: sentence count: past roughly this point a row stops reading as a control and starts
#: reading as a paragraph. Over the cap the question is dropped rather than truncated,
#: because a truncated question is one the reader did not ask.
_MAX_QUESTION_CHARS = 200


def _questions(text: str, limit: int) -> list[str]:
    """The questions in a model reply, or [] when it did not write a usable list.

    Forgiving about the wrapper and strict about the contents: a model that fences its
    JSON or prefixes it with a sentence still gets its questions used, because the
    alternative is throwing away three good questions over punctuation. A reply with no
    array at all yields nothing and is not an error — this is a flourish.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        parsed = json.loads(text[start:end + 1])
    except ValueError:
        return []
    if not isinstance(parsed, list):
        return []

    questions: list[str] = []
    for item in parsed:
        if not isinstance(item, str):
            continue
        question = " ".join(item.split())
        # See `_MAX_QUESTION_CHARS`: length is the rule, and dropping beats truncating.
        if not question or len(question) > _MAX_QUESTION_CHARS:
            continue
        questions.append(question)
        if len(questions) >= limit:
            break
    return questions


async def suggest(
    question: str,
    answer: str,
    *,
    model: str,
    scope: str | None = None,
) -> list[str]:
    """Up to `FOLLOW_UP_COUNT` questions to offer beneath `answer`.

    Never raises. Every failure mode — including the model being unreachable — returns an
    empty list, because the caller has an answer in hand and this is not allowed to
    endanger it.
    """
    if not answer.strip():
        return []

    instruction = _INSTRUCTION.format(
        count=FOLLOW_UP_COUNT,
        scope_line=_PATIENT_SCOPE if scope == "patient_chart" else _GENERAL_SCOPE,
    )
    prompt = f"Question asked:\n{question}\n\nAnswer given:\n{answer}"

    try:
        reply = await asyncio.wait_for(
            _build_llm(model).ainvoke(
                [SystemMessage(content=instruction), HumanMessage(content=prompt)]
            ),
            timeout=FOLLOW_UP_TIMEOUT_SECONDS,
        )
    except Exception as error:  # noqa: BLE001 - see the docstring: this never fails a turn
        # Logged at info, not warning: a turn with no follow-ups is a turn that shows no
        # follow-up row, and a warning on a routine absence is how a log becomes noise.
        logger.info("follow-ups were not proposed trace=- error=%s", type(error).__name__)
        return []

    # Through `text_of`, not `str(...)`: this model answers with content blocks, so a
    # naive stringify hands `_questions` the repr of a list — `[{'type': 'text', …}]` —
    # which has brackets and therefore *looks* parseable and never is. That is the
    # failure `message_text` exists to prevent, and it silently cost every turn its
    # follow-ups until it was caught live.
    return _questions(text_of(reply.content), FOLLOW_UP_COUNT)
