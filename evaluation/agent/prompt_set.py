"""The clinical prompt set, in one place because two collectors send it.

The wording is not incidental: it is part of the case set's identity, and two
different spellings of the same question are two different questions as far as
the model is concerned. `collect.py` (local transport) and `collect_http.py`
(deployed transport) must ask identical questions or the runs they produce are
not comparable with each other, or with the archive either of them wrote last
week.

This module exists because that equality used to be maintained by two copies of
a dictionary in two files, which is a maintenance claim rather than a mechanism.
"""

from __future__ import annotations

# name -> the question, given an admission id.
PROMPTS = {
    "risk": "What is the 30-day readmission risk for admission {hadm_id}?",
    "meds": "What medications were they discharged on? For admission {hadm_id}.",
    "summarize": "Summarize the recent discharge notes. For admission {hadm_id}.",
}

PROMPT_NAMES = ("risk", "meds", "summarize")


def question_for(prompt: str, hadm_id: int) -> str:
    return PROMPTS[prompt].format(hadm_id=hadm_id)
