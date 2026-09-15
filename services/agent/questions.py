"""The question wording: the second half of the prompt.

The model receives two texts — the system prompt in `prompts.py`, and the
question. The question's wording used to be written in the *website* repository
(`demo/fixtures.py`) and composed there before the call, which meant the chain
artifact lived in two repositories: a change to half the prompt could ship
without touching the chain, its revision, or its review.

It lives here now. The website sends intent — a chip name and a patient, or free
text and a patient — and the agent owns the words.

Every string below is the one the website used to build, so the model sees
exactly what it saw before. The change is where the wording lives, not what it
says; anything else would be a prompt change smuggled in as a refactor, and
prompt changes move answers.
"""

# The starter chips the console offers, and the question each one asks.
CHIP_QUESTIONS = {
    "risk": "Assess the 30-day readmission risk for this patient.",
    "meds": "What medications was this patient discharged on?",
    "summarize": "Summarize the recent discharge notes for this patient.",
    "compare": "Compare this assessment to the previous one for this patient.",
}

# Appended when a patient is in scope, so the model never has to ask which
# admission is meant.
ADMISSION_TEMPLATE = " For admission {hadm_id}."

# Asked when a patient is selected but no chip was chosen and no text typed.
# Deliberately NOT the risk chip's wording plus an admission: the two phrases
# differ, and both are preserved as they were so moving the strings cannot
# change what the model is asked.
DEFAULT_TEMPLATE = "Assess the 30-day readmission risk for admission {hadm_id}."


class UnknownChip(ValueError):
    """The caller asked for a chip this chain does not define."""

    def __init__(self, chip: str):
        super().__init__(f"Unknown chip: {chip!r}")
        self.chip = chip


def compose_question(*, chip=None, question=None, hadm_id=None) -> str:
    """Build the question the model is asked, from the caller's intent.

    Mirrors exactly what the website used to compose, including the default that
    applies when only a patient is given. Raises `UnknownChip` for a chip that is
    not in the table, and `ValueError` when there is nothing to ask.
    """
    if chip is not None:
        wording = CHIP_QUESTIONS.get(chip)
        if wording is None:
            raise UnknownChip(chip)
    elif isinstance(question, str) and question.strip():
        wording = question.strip()
    elif hadm_id is not None:
        return DEFAULT_TEMPLATE.format(hadm_id=hadm_id)
    else:
        raise ValueError("nothing to ask: no chip, no question, no admission")

    if hadm_id is not None:
        wording += ADMISSION_TEMPLATE.format(hadm_id=hadm_id)
    return wording
