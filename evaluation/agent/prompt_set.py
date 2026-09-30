"""The evaluation case set: the questions, and the follow-up that follows each.

A case is one admission and one chip. Three chips, 89 admissions, 267 cases; each
case is two turns, so a run is 534 requests.

**The first question is the chip's own wording**, composed by the chain that owns
those words, `services/agent/questions.py`. It is imported rather than restated
because a spelling of a question is part of a case set's identity: two copies
drift, and the moment they do the evaluation is measuring a question the product
does not ask. Before this, the harness kept its own three spellings, which is what
the import removes.

**The follow-up is authored here.** Each one refers back to the first answer with a
demonstrative — *that*, *those* — which has no referent unless the first answer is
in the conversation. That is the property under test: the agent holds nothing
between requests, so a second turn is answerable only if the caller sends the
first turn back with it, and an answer that reads correctly therefore evidences
that the earlier turn was both sent and used.

None of the three asks for a decision, so none puts the safety dimension of the
rubric at risk by construction. Each is a question a physician asks next.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The chain's own wording lives in the agent package, which this harness sits
# beside rather than inside. Imported so a change to the product's chips is a
# change to the case set, without a second place to make it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.agent.questions import compose_question  # noqa: E402

# The three chips the console offers. The names are the contract's: they are what
# a request carries in `chip`.
PROMPT_NAMES = ("risk", "meds", "summarize")

# The second question of each case, by chip.
FOLLOW_UPS = {
    "risk": "What is driving that?",
    "meds": "Why were they on those?",
    "summarize": "What was the outcome?",
}


def question_for(prompt: str, hadm_id: int) -> str:
    """The first question: the chip as the product words it, for one admission."""
    return compose_question(chip=prompt, hadm_id=hadm_id)


def follow_up_for(prompt: str) -> str:
    """The second question of the case, which is not addressed to an admission.

    Deliberately without the admission suffix the first question carries: the
    follow-up refers to the conversation, and naming the admission again would let
    it be answered from the admission alone, which is the thing being tested.
    """
    return FOLLOW_UPS[prompt]
