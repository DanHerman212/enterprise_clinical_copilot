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

# Asked when a patient is selected but no chip was chosen and no text typed.
# Deliberately NOT the risk chip's wording plus an admission: the two phrases
# differ, and both are preserved as they were so moving the strings cannot
# change what the model is asked.
DEFAULT_TEMPLATE = "Assess the 30-day readmission risk for admission {hadm_id}."

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
    if hadm_id is not None:
        return DEFAULT_TEMPLATE.format(hadm_id=hadm_id)
    raise ValueError("nothing to ask: no chip, no question, no admission")


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
    # The default is a whole sentence naming the admission, so appending the
    # admission to it would say it twice.
    if hadm_id is not None and wording != DEFAULT_TEMPLATE.format(hadm_id=hadm_id):
        wording += ADMISSION_TEMPLATE.format(hadm_id=hadm_id)
    return wording
