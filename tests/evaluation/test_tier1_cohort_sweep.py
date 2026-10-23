"""Phase 03, Section 2 — the deterministic cohort sweep, Tier 1.

Zero LLM. Every assertion here is either a pure function of the cohort tables or a
single read of the feature store; nothing is sent to a model, and no judge is involved.

The cohort is DISCOVERED, not hardcoded. `served_admissions.SERVED_HADM_IDS` is the
authorization boundary the service refuses against, `routing.OUTPATIENT_HADM_IDS` is the
declared outpatient set, and `admission_phenotypes.NON_ADMISSION` / `UNRESOLVED` are the
phenotype map's view of the same facts. A test that carried its own copy of 87 ids would
agree with whatever the code did, which is the failure mode this file exists to avoid —
so the partitions are derived and then cross-checked against each other.

The manifest sweep is the one part that touches the network: `<admission_metadata>` is
rendered from the feature-store row that scored the patient, and there is no offline copy
of those rows (`cases_82.json` carries outcomes, not profiles). It reads ONE row per
admission, concurrently, and it never calls a model.
"""

from __future__ import annotations

import asyncio

import pytest

from services.agent import admission_context, admission_phenotypes, routing
from services.agent import served_admissions
from services.agent.a2ui import compose_presentation
from services.mcp.tools.admission_profile import get_admission_profile

# --- discovery ---------------------------------------------------------------

SERVED = frozenset(served_admissions.SERVED_HADM_IDS)
OUTPATIENT = frozenset(routing.OUTPATIENT_HADM_IDS)
INPATIENT = SERVED - OUTPATIENT
#: In the authorization boundary with neither a note nor a feature row. Two of them,
#: and nothing can be served for either.
UNSERVED = frozenset(admission_phenotypes.UNRESOLVED)
#: An id that is not a record at all — the shape a malformed caller sends.
SYNTHETIC_UNKNOWN = 99999999

#: The structural facts a hydrating manifest must carry, by the wire names `render`
#: writes. Age and sex are always present on a scored row; the discharge location is
#: the field an administrative question is answered from, which is why the manifest is
#: rendered at all.
REQUIRED_MANIFEST_FIELDS = ("age", "gender", "discharge_location")


def _hydrate_all(hadm_ids: tuple[int, ...]) -> dict[int, str]:
    """Render `<admission_metadata>` for every admission, concurrently.

    Concurrent because the read is one row per admission and sequential would put 82
    round trips in a row; bounded so the feature store is not asked for 82 at once.
    """

    async def one(hadm_id: int, gate: asyncio.Semaphore) -> tuple[int, str]:
        async with gate:
            payload = await get_admission_profile(hadm_id)
        profile = payload.get("profile") if isinstance(payload, dict) else None
        if not isinstance(profile, dict):
            return hadm_id, ""
        return hadm_id, admission_context.render(hadm_id, profile)

    async def gather() -> dict[int, str]:
        gate = asyncio.Semaphore(8)
        pairs = await asyncio.gather(*(one(h, gate) for h in hadm_ids))
        return dict(pairs)

    return asyncio.run(gather())


@pytest.fixture(scope="module")
def manifests() -> dict[int, str]:
    return _hydrate_all(tuple(sorted(INPATIENT)))


# --- the partition -----------------------------------------------------------

def test_the_served_cohort_is_87_records():
    """82 inpatient + 5 outpatient. The number the whole phase is sized on."""
    assert len(SERVED) == 87, sorted(SERVED)
    assert len(INPATIENT) == 82, sorted(INPATIENT)
    assert len(OUTPATIENT) == 5, sorted(OUTPATIENT)


def test_the_two_sets_are_disjoint_and_cover_the_cohort():
    assert not (INPATIENT & OUTPATIENT)
    assert INPATIENT | OUTPATIENT == SERVED


def test_the_unserved_orphans_are_outside_the_boundary():
    """They are in the boundary list only as a documented gap."""
    assert UNSERVED == frozenset({90000039, 90000107})
    assert not (UNSERVED & SERVED), sorted(UNSERVED & SERVED)


def test_the_outpatient_set_is_declared_in_both_places():
    """Two modules, one fact. A drift between them would refuse the wrong records."""
    assert OUTPATIENT == frozenset(admission_phenotypes.NON_ADMISSION)


# --- inpatient manifests -----------------------------------------------------

def test_every_inpatient_admission_has_a_manifest_that_hydrates(manifests):
    """One failure names the admission and the field, rather than 'somewhere in 82'."""
    missing = sorted(h for h in INPATIENT if not manifests.get(h))
    assert not missing, f"no manifest rendered for {missing}"

    thin: dict[int, list[str]] = {}
    for hadm_id in sorted(INPATIENT):
        text = manifests[hadm_id]
        absent = [field for field in REQUIRED_MANIFEST_FIELDS if field not in text]
        if absent:
            thin[hadm_id] = absent
    assert not thin, f"manifest missing required fields: {thin}"


def test_the_manifest_names_the_admission_it_belongs_to(manifests):
    wrong = [h for h in sorted(INPATIENT)
             if f'hadm_id="{h}"' not in manifests[h]]
    assert not wrong, f"manifest rendered under the wrong id: {wrong}"


# --- the outpatient boundary -------------------------------------------------

def test_an_outpatient_record_is_never_paired_with_a_readmission_estimate():
    """Scoring an outpatient note answers a question nobody asked.

    `routing.supplementary_arguments` drops `predict_readmission` for these ids, so a
    `/ehr` or `/pubmed` turn on an outpatient record carries no estimate — and therefore
    no inpatient figure to leak into a briefing about a visit. Asserted against a
    command whose estimate IS a supplementary call; on `/risk` the estimate is the
    primary tool and the outpatient policy is expressed by the encounter notice below.
    """
    summarize = routing.command_for("summarize")
    for hadm_id in sorted(OUTPATIENT):
        secondary = [tool for tool, _ in
                     routing.supplementary_arguments(summarize, hadm_id)]
        assert "predict_readmission" not in secondary, hadm_id

    # The control: the same command on an inpatient record does carry the estimate.
    secondary = [tool for tool, _ in
                 routing.supplementary_arguments(summarize, 90000015)]
    assert "predict_readmission" in secondary


def test_an_outpatient_turn_carries_the_encounter_notice():
    """The record says what it is, in the text the formatter is handed.

    The defect this prevents is textual: 90000073 had "this admission" written about a
    speech-therapy visit from both routes. One notice, both paths.
    """
    from services.agent import deterministic

    with_notice = deterministic._formatter_input([], True, "Summarise the note.")
    without = deterministic._formatter_input([], False, "Summarise the note.")

    assert routing.OUTPATIENT_ENCOUNTER_NOTICE in with_notice
    assert routing.OUTPATIENT_ENCOUNTER_NOTICE not in without


def test_an_outpatient_record_is_not_comparable_against_a_cohort():
    """The comparative table needs a stay; a visit has nothing to compare."""
    for hadm_id in sorted(OUTPATIENT):
        assert admission_phenotypes.is_matrix_eligible(hadm_id) is False, hadm_id
    # The control: an inpatient admission in a curated bucket is eligible.
    assert admission_phenotypes.is_matrix_eligible(90000023) is True


# --- the unserved boundary ---------------------------------------------------

@pytest.mark.parametrize("hadm_id", sorted(UNSERVED))
def test_an_unserved_orphan_is_refused_by_the_authorization_boundary(hadm_id):
    assert served_admissions.is_served(hadm_id) is False


def test_a_synthetic_unknown_id_is_refused_like_an_unserved_orphan():
    """The refusal must not depend on knowing the id."""
    assert served_admissions.is_served(SYNTHETIC_UNKNOWN) is False
    assert served_admissions.is_served(0) is False
    # A string is a type error, not a record: same refusal.
    assert served_admissions.is_served("90000015") is False


@pytest.mark.parametrize("hadm_id", [*sorted(UNSERVED), SYNTHETIC_UNKNOWN])
def test_the_unserved_refusal_is_an_answer_rather_than_an_exception(hadm_id):
    """A 200 with an abstention envelope, which is what the console can draw."""
    from services.agent import http

    payload = asyncio.run(http._unserved_admission(
        {"hadm_id": hadm_id, "question": "What is her risk?"}, "",
    ))

    assert payload is not None, hadm_id
    assert payload["routed_to"] == "abstain"
    assert "admission_not_found" in payload["guardrail_flags"]
    # Nothing from a chart leaks into a refusal.
    assert payload["sources"] == []
    assert payload["verified"] is False


def test_a_served_admission_is_not_refused():
    """The control: the boundary does not over-refuse."""
    assert served_admissions.is_served(90000015) is True
