"""The admission's structured features, stated before the model runs.

The blind spot this closes
--------------------------
`predict_readmission` scores a tabular row and returns the features that drove
the score — `admission_type` among them. The retrieval tools search discharge-note
text, and a discharge summary writes the clinical narrative without ever declaring
"Admission type: URGENT". So the agent could correctly report that admission type
was a driver and then, held to its grounding rules, answer that the note does not
document it. Both statements were true and the pair was useless.

The fix is pre-flight hydration rather than a new capability for the model. The
profile is fetched once, by the caller, against the admission the turn is already
about, and folded into the system prompt as verified ground truth. Three
consequences, each of them the reason for this shape:

* **No tool budget.** The model never spends one of its five calls on a fact that
  is already in front of it, and a turn cannot run out of budget trying.
* **No schema reasoning.** "What is the admission type?" is answered from a row
  that was read to compute the score; there is no SQL to generate, nothing to
  hallucinate and no warehouse surface exposed to a prompt.
* **No second source of truth.** The profile is decoded from the same row the
  booster scored (`services/mcp/dependencies/features/profile.py`), so the words
  and the number cannot disagree.

Failure is silence, not a broken turn: a profile that cannot be read leaves the
prompt exactly as it was. An answer that says the note does not document something
is worse than one that knows — but far better than a turn that fails.
"""

from __future__ import annotations

import logging
from typing import Any

from services.agent.routing import encounter

_LOG = logging.getLogger(__name__)

#: The MCP tool this calls. It is registered on the server so the pre-flight call
#: can reach it, and hidden from the model's function declarations so the model
#: cannot call it a second time (see `mcp_client.gemini_tool`).
PROFILE_TOOL = "get_admission_profile"

#: Every tool the caller invokes on the model's behalf before the model runs.
#: Owned here, by the module that makes the call, because `routing` and
#: `mcp_client` both have to agree with it and `contracts` cannot be the owner:
#: `contracts` imports `routing`, so a constant defined there and needed by
#: `routing` is a circular import at module load.
PREFLIGHT_TOOLS = frozenset({PROFILE_TOOL})

#: Field order in the block. The keys come from the server's decoder, which owns
#: the wording; this module only decides what a reader sees first.
_FIELDS: tuple[tuple[str, str], ...] = (
    ("age", "age"),
    ("gender", "gender"),
    ("admission_type", "admission_type"),
    ("discharge_location", "discharge_location"),
    ("insurance", "insurance"),
    ("race", "race"),
    ("prior_admission_count", "prior_admissions"),
    ("prior_inpatient_days", "prior_inpatient_days"),
    ("recent_ed_visits", "ed_visits_prior_year"),
    ("index_los_days", "length_of_stay_days"),
)


def _display(value: Any) -> str:
    """A number exactly as the row carries it, and everything else as it stands.

    The feature store returns `prior_inpatient_days` as "24.6667" because that is the
    encoded row. This function used to round it to one decimal — 16.0417 became "16.0" —
    on the argument that four decimals claim a precision neither the record nor the
    model has, after an answer was seen saying "recorded as 16.0417 days"
    (`agent-00177-qr5`, 2026-10-24).

    That was the wrong layer to fix it. The manifest is verified ground truth that the
    judge holds the answer against, so a manifest that states "24.7" for a row holding
    24.6667 is a manifest that disagrees with the dataset — an answer repeating it is
    then scored as a fidelity failure against the expected full figure.
    Whispering the record is not the same as writing it well: the
    prompt asks the model to reproduce these values character-for-character and to use
    them descriptively, and it cannot do the first if this layer has already shortened
    the number.

    So the value travels exactly. Prose style is the prompt's business; the digits are
    the record's.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    # The profile arrives from BigQuery with every column as a string, so the value that
    # needs care is usually `"24.6667"` rather than 24.6667. Anything that is not a
    # number — "Emergency (EMER)", "Home" — is returned untouched.
    return str(value).strip()


def _clean(value: Any) -> str:
    """One line, no markup. The values are ours, and this keeps them ours.

    They reach a prompt, so a newline or an angle bracket in one would let a
    data value change the shape of the instruction around it. Nothing upstream
    produces such a value today; the check costs nothing and removes the class.
    """
    text = " ".join(_display(value).split())
    return text.replace("<", "(").replace(">", ")")


#: Fields that only exist because there was a stay.
#:
#: Dropped from the block when the record is an outpatient visit. The distinction is not
#: cosmetic: `index_los_days` is a real number on an outpatient row (the encoder built the
#: row as an admission), and stating it is what produced the canary's "length of stay was
#: 7.9 days" for an encounter that had none. A fact the record does not have is not made
#: true by the column being populated.
STAY_ONLY_FIELDS: frozenset[str] = frozenset({
    "admission_type", "discharge_location", "index_los_days",
})

#: Said in place of the dropped fields, rather than leaving their absence to be inferred.
#: A model handed a block with no length of stay will sometimes supply one from the note;
#: a model told there is no stay does not.
OUTPATIENT_LINE = (
    "encounter_type: outpatient encounter — this record is a visit, not an inpatient "
    "stay, so it has no admission type, no discharge destination and no length of stay"
)


def render(hadm_id: int, profile: dict[str, Any]) -> str:
    """The `<admission_metadata>` block for one admission, or "" if unusable."""
    if not isinstance(profile, dict) or not profile:
        return ""
    outpatient = encounter.is_outpatient(encounter.classify(profile)) is True
    lines = [
        f'<admission_metadata hadm_id="{int(hadm_id)}" source="feature store row '
        f'scored by the readmission model">'
    ]
    if outpatient:
        lines.append(f"- {OUTPATIENT_LINE}")
    for key, label in _FIELDS:
        if outpatient and key in STAY_ONLY_FIELDS:
            continue
        value = profile.get(key)
        if value in (None, ""):
            continue
        lines.append(f"- {label}: {_clean(value)}")
    lines.append("</admission_metadata>")
    # A block with no fields in it is noise that also implies a fact was found.
    return "\n".join(lines) if len(lines) > 2 else ""


async def hydrate(box: Any, hadm_id: int | None) -> str:
    """Fetch and render the profile, and record the turn's encounter type.

    The encounter type is set here, unconditionally, because this is the one place the
    record's own row is read before anything else runs: the graph, the command gate, the
    canvas and the response composer all read it back through `encounter` rather than
    recognising an admission id. Setting None as well as a value is what stops a turn with
    no profile from inheriting the previous turn's answer.
    """
    encounter.set_current(None)
    if not hadm_id or box is None:
        return ""
    try:
        payload = await box.call(PROFILE_TOOL, {"hadm_id": int(hadm_id)})
    except Exception:  # noqa: BLE001 — a missing profile may not fail the turn
        _LOG.warning("admission profile pre-flight failed for hadm %s", hadm_id,
                     exc_info=True)
        return ""
    if not isinstance(payload, dict) or payload.get("error"):
        # Expected rather than alarming: an admission outside the cohort has no
        # row, and the turn proceeds without the block.
        _LOG.info(
            "admission profile unavailable for hadm %s: %s",
            hadm_id, payload.get("error") if isinstance(payload, dict) else "no payload",
        )
        return ""
    profile = payload.get("profile") or {}
    encounter.set_current(encounter.classify(profile))
    return render(hadm_id, profile)
