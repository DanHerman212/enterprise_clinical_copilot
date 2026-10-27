"""Is this record a hospital admission, or a visit that was never a stay?

Why this is a field and not a rule
-----------------------------------
The distinction decides three things: whether the pre-flight metadata may state a length of
stay, an admission type and a discharge disposition; whether `predict_readmission` runs as a
command's supplementary call; and whether the canvas may draw `RiskInfluencers` and
`ComparativeRiskMatrix`. A readmission rate describes a stay, so a visit that had none has to
be excluded from all three or the product states four facts about an admission that never
happened — which is what the first canary found, a 7.9-day "length of stay" reported for an
outpatient encounter.

It used to be a frozenset of five admission ids, in two places. That is a runtime branch on
an identifier: the record was classified by recognising its number rather than by anything
the record said. The measurement in
`docs/agent_evaluation/00b1-outpatient-signal_measurement_271026.md` established that no
other field separated the two groups — not the ten decoded features, not the cohort row, and
not the note's section structure (which misclassified six of eighty-seven) — so the fix was
to record the fact on the row, not to infer it.

The rule this module exists to keep
-----------------------------------
One predicate, one source. `encounter_type` comes from the feature row, arrives with the
pre-flight profile, and is read by every call site through `current_is_outpatient()`. Three
independent readings of the same fact is how the three of them came to disagree.

Unknown is not inpatient
------------------------
A missing or unrecognised column value returns None, and None is **not** treated as
inpatient by the callers that make a claim: `states_inpatient_fields` refuses to assert an
admission either way, and the estimate and the widgets keep the behaviour they had before the
column existed. A record added without the column is then visible in the logs rather than
silently mistyped as a stay.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

OUTPATIENT = "outpatient"
INPATIENT = "inpatient"

#: The encounter type of the turn being served.
#:
#: Request-scoped, set once by the pre-flight in `admission_context.hydrate` before the model
#: runs, and read by the graph, the command gate, the canvas and the response composer. A
#: context variable rather than a parameter for the reason `observability.set_run_tag`
#: records: `ask` is the seam every route and every test double stands on, and a keyword
#: added to it is sixteen fakes edited to accept a field they do not care about. Every turn
#: sets it, including to None, so it cannot carry a previous turn's answer.
_CURRENT: ContextVar[str | None] = ContextVar("encounter_type", default=None)


def classify(profile: Any) -> str | None:
    """The encounter type a decoded profile states, or None when it states neither.

    Reads the human sentence `decode_profile` produces ("outpatient encounter (no
    inpatient stay)") by its leading word, because the wording is for a reader and the
    classification is for this module. An unknown string is None, not a guess.
    """
    if not isinstance(profile, dict):
        return None
    text = str(profile.get("encounter_type") or "").strip().lower()
    if text.startswith(OUTPATIENT):
        return OUTPATIENT
    if text.startswith(INPATIENT):
        return INPATIENT
    return None


def is_outpatient(value: str | None) -> bool | None:
    """Whether an encounter type is an outpatient visit. None means it was not stated."""
    if value == OUTPATIENT:
        return True
    if value == INPATIENT:
        return False
    return None


def set_current(value: str | None) -> None:
    """Record the encounter type for the rest of this turn."""
    _CURRENT.set(value)


def current() -> str | None:
    """The encounter type of the turn being served, or None when it was not stated."""
    return _CURRENT.get()


def current_is_outpatient() -> bool | None:
    """`is_outpatient` for the turn being served."""
    return is_outpatient(_CURRENT.get())


def states_inpatient_fields() -> bool:
    """Whether this turn may state stay-shaped fields at all.

    True unless the encounter is known to be an outpatient visit. Unknown keeps the fields,
    which is the behaviour every record had before the column existed: withdrawing evidence
    a clinician is already reading, on the strength of a missing value, is the worse error.
    """
    return current_is_outpatient() is not True
