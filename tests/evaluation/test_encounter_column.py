"""The encounter kind is a column on the row, not an admission id in the code.

Why this file exists
--------------------
The product used to answer "is this an outpatient visit?" by testing `hadm_id` against a
frozenset of five ids. That is a classification by identifier: the record was recognised
rather than read. `docs/agent_evaluation/00b1-outpatient-signal_measurement_271026.md`
measured every other candidate signal on the 87 served records and found none that
separated the groups, because the distinction was curation metadata and lived nowhere but
that set. So it was recorded where it belongs — `encounter_type` on
`readmission.hybrid_features` and `readmission.demo_cohort` — and the runtime reads it.

What could go wrong, and which assertions catch it
--------------------------------------------------
*The column is absent or mis-spelled.* `decode_profile` reports "Not recorded", the
predicate returns None, and the callers keep their previous behaviour rather than
inventing a type. Asserted hermetically.

*The backfill is wrong or partial.* The served records must classify 87/87 the way the
cohort was built. That comparison needs an independent statement of the curation
decision, which is `evaluation.outpatient_oracle` — a copy of the product's own rule
would only ever agree with itself. Asserted live, because it is a fact about BigQuery.

*The value reaches the prompt as a stay.* The five outpatient rows carry a real
`index_los_days` and an emergency `admission_type`, because the row exists to be scored.
An outpatient block must state the absence instead of printing them. Asserted on the
rendered block.
"""

from __future__ import annotations

import asyncio

import pytest

from evaluation.outpatient_oracle import INPATIENT, OUTPATIENT, OUTPATIENT_HADM_IDS
from services.agent.ingress.contracts import SERVED_HADM_IDS
from services.agent.routing import admission_context, encounter
from services.mcp.dependencies.features.profile import decode_profile

#: An id that is in no cohort, no boundary and no set of any kind. If classification
#: depends on recognising an id, this record has no way to be typed.
SYNTHETIC = 99999999

#: One decoded row, shaped the way `decode_profile` produces it. The stay-shaped fields
#: are populated because the encoder populated them for every record.
ROW = {
    "encounter_type": "inpatient",
    "age": "67",
    "gender": "1",
    "admission_type_ew_emer": "1",
    "discharge_location_home": "1",
    "insurance_medicare": "1",
    "race_white": "1",
    "prior_admission_count": "2",
    "prior_inpatient_days": "13.4167",
    "recent_ed_visits": "0",
    "index_los_days": "7.91667",
}


class _Box:
    """A pre-flight toolbox returning one profile payload."""

    def __init__(self, profile: dict | None = None, error: str | None = None) -> None:
        self.profile = profile
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    async def call(self, name, args):
        self.calls.append((name, args))
        if self.error:
            return {"error": self.error, "hadm_id": args["hadm_id"]}
        return {"hadm_id": args["hadm_id"], "profile": self.profile}


# --- the column is the signal ------------------------------------------------

def test_a_record_with_no_id_in_any_set_classifies_from_the_row_alone():
    """The id is deliberately one no cohort contains — it must not matter."""
    profile = decode_profile({**ROW, "encounter_type": OUTPATIENT})

    assert encounter.classify(profile) == OUTPATIENT
    assert encounter.is_outpatient(encounter.classify(profile)) is True


def test_an_inpatient_row_classifies_as_an_admission():
    profile = decode_profile({**ROW, "encounter_type": INPATIENT})

    assert encounter.classify(profile) == INPATIENT
    assert encounter.is_outpatient(encounter.classify(profile)) is False


def test_the_column_is_declared_before_anything_it_conditions():
    assert next(iter(decode_profile(ROW))) == "encounter_type"


@pytest.mark.parametrize("raw", [None, "", "   ", "unknown", "day_clinic"])
def test_an_unusable_value_is_unknown_and_never_a_guess(raw):
    profile = decode_profile({**ROW, "encounter_type": raw})

    assert profile["encounter_type"] == "Not recorded"
    assert encounter.classify(profile) is None
    assert encounter.is_outpatient(None) is None


def test_an_absent_column_is_unknown_not_inpatient():
    """Unknown must keep the old behaviour: withdrawing evidence a clinician is reading,
    on the strength of a missing value, is the worse error."""
    row = {k: v for k, v in ROW.items() if k != "encounter_type"}
    profile = decode_profile(row)

    assert encounter.classify(profile) is None
    encounter.set_current(encounter.classify(profile))
    assert encounter.states_inpatient_fields() is True


# --- what the hydrated block may say ----------------------------------------

def test_an_outpatient_block_states_the_absence_instead_of_the_stay():
    block = admission_context.render(SYNTHETIC, decode_profile(
        {**ROW, "encounter_type": OUTPATIENT}))

    assert "outpatient encounter" in block
    assert "no admission type" in block
    # The two fields that only exist for a stay, and the number that leaked into prose
    # as "the length of stay was 7.9 days" on the fifth canary.
    assert "length_of_stay_days" not in block
    assert "admission_type:" not in block
    assert "discharge_location" not in block
    assert "7.9" not in block


def test_an_inpatient_block_keeps_every_field_it_had():
    block = admission_context.render(90000015, decode_profile(
        {**ROW, "encounter_type": INPATIENT}))

    assert "length_of_stay_days: 7.9" in block
    assert "admission_type: Emergency (EW EMER.)" in block
    assert "discharge_location: Home" in block
    assert "outpatient" not in block


def test_an_unknown_record_keeps_the_fields_it_had():
    block = admission_context.render(90000015, decode_profile(ROW))

    assert "length_of_stay_days: 7.9" in block
    assert "outpatient" not in block


# --- the pre-flight records it for the turn ---------------------------------

def _turn_encounter(box: _Box, hadm_id: int | None) -> bool | None:
    """Run the pre-flight and read the answer back in the same task.

    `asyncio.run` wraps its coroutine in a task, and a task runs in a *copy* of the
    calling context: a write inside it is invisible outside. Production awaits
    `hydrate` and then `ask` in one task (`turn.py`), so the read has to happen in one
    too, or the test would be measuring the harness rather than the behaviour.
    """

    async def run() -> bool | None:
        await admission_context.hydrate(box, hadm_id)
        return encounter.current_is_outpatient()

    return asyncio.run(run())

def test_hydrate_records_the_encounter_type_for_the_rest_of_the_turn():
    box = _Box(profile=decode_profile({**ROW, "encounter_type": OUTPATIENT}))

    assert _turn_encounter(box, SYNTHETIC) is True


def test_hydrate_clears_the_previous_turns_answer():
    """A turn with no profile must not inherit the last turn's encounter."""
    encounter.set_current(OUTPATIENT)
    box = _Box(error="unknown_patient")

    assert _turn_encounter(box, SYNTHETIC) is None


def test_hydrate_without_an_admission_clears_it_too():
    encounter.set_current(OUTPATIENT)

    assert _turn_encounter(_Box(), None) is None


# --- the backfill, against the authority ------------------------------------

@pytest.mark.live
def test_every_served_record_classifies_from_the_column_the_cohort_was_built_with():
    """87/87. The column is the fact; this asserts it says what the builder meant.

    `get_admission_profile` is the pre-flight call itself, so this measures the payload
    the agent receives rather than the table behind it.
    """
    from services.mcp.tools.admission_profile import get_admission_profile

    served = sorted(SERVED_HADM_IDS)
    gate = asyncio.Semaphore(8)

    async def read_one(hadm_id: int):
        async with gate:
            return hadm_id, await get_admission_profile(hadm_id)

    async def read():
        return dict(await asyncio.gather(*(read_one(h) for h in served)))

    payloads = asyncio.run(read())

    classified: dict[int, str | None] = {}
    for hadm_id, payload in payloads.items():
        assert not payload.get("error"), (hadm_id, payload.get("error"))
        classified[hadm_id] = encounter.classify(payload["profile"])

    assert len(classified) == 87, sorted(classified)
    assert None not in classified.values(), {
        h: v for h, v in classified.items() if v is None
    }

    outpatient = {h for h, value in classified.items() if value == OUTPATIENT}
    assert outpatient == set(OUTPATIENT_HADM_IDS), sorted(outpatient)
    assert {h for h, value in classified.items() if value == INPATIENT} == (
        set(served) - set(OUTPATIENT_HADM_IDS)
    )
