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
from services.agent import deterministic
from services.agent.a2ui import compose_presentation, compose_risk_canvas, predict_payload
from services.mcp.retrieval.sections import SUMMARY_SECTIONS
from services.mcp.tools import (
    discharge_medications,
    get_lab_trends,
    predict_readmission,
    rag_search_sections,
)
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


#: One admission is one sweep slot. Its reads run in sequence inside the slot and the
#: slots run concurrently.
#:
#: 6 is a credential limit, not a data one. Every tool call builds its own gRPC channel
#: and asks the metadata server for a token, and past roughly eight simultaneous
#: refreshes `oauth2.googleapis.com` starts answering `SSL: UNEXPECTED_EOF_WHILE_READING`
#: — the sweep then fails wholesale with a `ServiceUnavailable` that looks like a code
#: defect and is not. Measured: 4 -> 16.6s per 10 admissions, 8 -> 13.0s, 24 -> unstable.
SWEEP_CONCURRENCY = 6
#: How many times a read is retried on a transient transport failure. A token refresh
#: that fails is not evidence about the record, so it is retried; a tool that answers
#: with its own error payload is NOT retried, because that answer is the data.
READ_ATTEMPTS = 3
READ_BACKOFF_SECONDS = 2.0


async def _read(call, hadm_id: int, *args):
    """One tool read, retried while the failure is the transport rather than the data."""
    last: Exception | None = None
    for attempt in range(READ_ATTEMPTS):
        try:
            return await call(hadm_id, *args)
        except Exception as exc:  # noqa: BLE001 — retried, or re-raised below
            transient = type(exc).__name__ in (
                "ServiceUnavailable", "_InactiveRpcError", "RetryError",
                "ConnectionError", "SSLError", "DeadlineExceeded",
            )
            last = exc
            if not transient:
                raise
            await asyncio.sleep(READ_BACKOFF_SECONDS * (attempt + 1))
    raise last  # type: ignore[misc]


def _sweep(hadm_ids: tuple[int, ...]) -> dict[int, dict]:
    """Every patient-scoped read for every admission, bounded and concurrent."""

    async def one(hadm_id: int, gate: asyncio.Semaphore) -> tuple[int, dict]:
        async with gate:
            profile = await _read(get_admission_profile, hadm_id)
            risk = await _read(predict_readmission, hadm_id)
            labs = await _read(get_lab_trends, hadm_id)
            meds = await _read(discharge_medications, hadm_id)
            notes = await _read(rag_search_sections, hadm_id)
        return hadm_id, {"profile": profile, "risk": risk, "labs": labs,
                         "meds": meds, "notes": notes}

    async def gather() -> dict[int, dict]:
        gate = asyncio.Semaphore(SWEEP_CONCURRENCY)
        pairs = await asyncio.gather(*(one(h, gate) for h in hadm_ids))
        return dict(pairs)

    return asyncio.run(gather())


@pytest.fixture(scope="module")
def sweep() -> dict[int, dict]:
    return _sweep(tuple(sorted(SERVED)))


@pytest.fixture(scope="module")
def manifests(sweep) -> dict[int, str]:
    """`<admission_metadata>` for every inpatient, rendered from the same sweep."""
    rendered: dict[int, str] = {}
    for hadm_id in sorted(INPATIENT):
        payload = sweep.get(hadm_id) or {}
        profile = payload.get("profile")
        profile = profile.get("profile") if isinstance(profile, dict) else None
        rendered[hadm_id] = (admission_context.render(hadm_id, profile)
                             if isinstance(profile, dict) else "")
    return rendered


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


def test_an_outpatient_record_is_never_scored():
    """TF-1, fixed 2026-10-23 for `/risk` and kept when it was retired (2026-10-26).

    Before the gate, the outpatient rule only dropped `predict_readmission` where it was
    a supplementary call, so the command that ran it as its primary tool scored the
    visit: 90000051 came back at 0.2916, above threshold, for a speech-therapy note.

    `predict_readmission` is supplementary-only now — `/ehr` is its one caller — so the
    guarantee is the one `supplementary_arguments` makes, and it is swept over the whole
    outpatient cohort rather than on one record.
    """
    ehr = routing.command_for("summarize")
    broken: dict[int, str] = {}
    for hadm_id in sorted(OUTPATIENT):
        requested = {tool for tool, _args in
                     routing.supplementary_arguments(ehr, hadm_id)}
        if "predict_readmission" in requested:
            broken[hadm_id] = "requested the readmission estimate"
    assert not broken, broken


def test_the_estimate_is_still_requested_on_an_inpatient_record():
    """The control: the rule is about outpatient records and nothing else."""
    ehr = routing.command_for("summarize")
    for hadm_id in (90000015, 90000023):
        requested = {tool for tool, _args in
                     routing.supplementary_arguments(ehr, hadm_id)}
        assert "predict_readmission" in requested, hadm_id


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


# =============================================================================
# Section 3 — the tool and parser sweep, all 87 records
# =============================================================================
#
# Four patient-scoped tools, each with its deterministic payload builder. One admission
# is one sweep slot: its four reads run in sequence inside that slot, and the slots run
# concurrently, because the stores dislike 350 concurrent readers and none of them is
# slow enough to justify it.
#
# Nothing here calls a model. `rag_search` is deliberately absent: it embeds the query,
# and an embedding call is a model call. The sectioned reader is the deterministic path
# `/ehr` uses, and it is the one swept.

def _payload_for(payloads: dict[int, dict], hadm_id: int, tool: str) -> dict:
    payload = (payloads.get(hadm_id) or {}).get(tool)
    assert isinstance(payload, dict), f"{tool} returned no payload for {hadm_id}"
    return payload


# --- 1. the risk tool and its canvas -----------------------------------------

def test_predict_readmission_returns_a_scorable_payload_for_every_inpatient(
        sweep):
    broken: dict[int, str] = {}
    for hadm_id in sorted(INPATIENT):
        payload = _payload_for(sweep, hadm_id, "risk")
        if payload.get("error"):
            broken[hadm_id] = f"error: {payload['error']}"
            continue
        probability = payload.get("probability")
        threshold = payload.get("threshold")
        if not isinstance(probability, (int, float)) or not 0.0 <= probability <= 1.0:
            broken[hadm_id] = f"probability {probability!r}"
        elif not isinstance(threshold, (int, float)) or not 0.0 < threshold < 1.0:
            broken[hadm_id] = f"threshold {threshold!r}"
        elif payload.get("decision") not in (0, 1):
            broken[hadm_id] = f"decision {payload.get('decision')!r}"
        elif not payload.get("top_factors"):
            broken[hadm_id] = "no feature drivers"
    assert not broken, broken


def test_every_feature_driver_has_a_name_a_size_and_a_direction(sweep):
    broken: dict[int, list] = {}
    for hadm_id in sorted(INPATIENT):
        payload = _payload_for(sweep, hadm_id, "risk")
        bad = [
            factor for factor in payload.get("top_factors") or []
            if not isinstance(factor.get("feature"), str) or not factor["feature"]
            or not isinstance(factor.get("contribution"), (int, float))
            or factor.get("direction") not in ("increases", "decreases")
        ]
        if bad:
            broken[hadm_id] = bad
    assert not broken, broken


def test_the_risk_canvas_builds_for_every_inpatient(sweep):
    """`/risk` draws a widget and no prose, so the surface IS the answer."""
    broken: dict[int, str] = {}
    for hadm_id in sorted(INPATIENT):
        payload = _payload_for(sweep, hadm_id, "risk")
        call = {"name": "predict_readmission", "args": {"hadm_id": hadm_id},
                "response": payload}
        try:
            canvas = compose_risk_canvas(predict_payload([call]), None)
        except Exception as exc:  # noqa: BLE001 — reported per record, not raised
            broken[hadm_id] = f"{type(exc).__name__}: {exc}"
            continue
        components = canvas["messages"][1]["updateComponents"]["components"]
        types = {component["component"] for component in components}
        if "RiskInfluencers" not in types:
            broken[hadm_id] = f"no risk widget in {sorted(types)}"
    assert not broken, broken


# --- 2. the medication tool and its parser -----------------------------------

def _recorded_meds(hadm_id: int, payload: dict) -> list[dict]:
    return [{"name": "discharge_medications", "args": {"hadm_id": hadm_id},
             "response": payload}]


def test_discharge_medications_returns_a_structured_payload_for_every_inpatient(
        sweep):
    """An empty list is a valid answer; an error payload is not."""
    errors: dict[int, str] = {}
    for hadm_id in sorted(INPATIENT):
        payload = _payload_for(sweep, hadm_id, "meds")
        if payload.get("error"):
            errors[hadm_id] = str(payload["error"])
        elif not isinstance(payload.get("passages"), list):
            errors[hadm_id] = "no passages list"
    assert not errors, errors


def test_every_medication_passage_carries_the_text_the_parser_reads(sweep):
    broken: dict[int, list] = {}
    for hadm_id in sorted(INPATIENT):
        payload = _payload_for(sweep, hadm_id, "meds")
        bad = [passage for passage in payload.get("passages") or []
               if not isinstance(passage.get("text"), str) or not passage["text"].strip()]
        if bad:
            broken[hadm_id] = bad
    assert not broken, broken


def test_the_consolidation_invariant_holds_for_every_inpatient(sweep):
    """Phase 02.5: `/meds` cites with one marker, so its passages are one passage.

    The section key survives because citation resolution matches on it — the first
    version of this check used the human label and resolution silently served the "no
    information" sentence instead of the passage.
    """
    broken: dict[int, str] = {}
    for hadm_id in sorted(INPATIENT):
        payload = _payload_for(sweep, hadm_id, "meds")
        recorded = _recorded_meds(hadm_id, dict(payload))
        before = list(payload.get("passages") or [])
        deterministic._consolidate_medication_passages(recorded)
        after = recorded[0]["response"]["passages"]
        if len(after) > 1:
            broken[hadm_id] = f"{len(after)} passages after consolidation"
        elif after and before and after[0].get("section") != before[0].get("section"):
            broken[hadm_id] = "section key was relabelled"
        elif after and len(before) > 1:
            joined = after[0]["text"]
            lost = [passage["text"] for passage in before
                    if passage["text"] not in joined]
            if lost:
                broken[hadm_id] = f"{len(lost)} chunk(s) dropped"
    assert not broken, broken


def test_the_medication_narrative_handles_both_outcomes(sweep):
    """Rows, or the sentence that says there are none. Neither may raise."""
    for hadm_id in sorted(INPATIENT):
        payload = _payload_for(sweep, hadm_id, "meds")
        returned = len(payload.get("passages") or [])
        empty = deterministic.medication_narrative([], returned)
        assert isinstance(empty, str) and empty.strip(), hadm_id
        with_rows = deterministic.medication_narrative(
            [{"medication": "warfarin", "dose": "4 mg"}], returned)
        assert "warfarin" in with_rows, hadm_id


def test_the_meds_payload_builds_and_its_marker_resolves(sweep):
    """`^[1]` must address the consolidated text, on every record that has one."""
    broken: dict[int, str] = {}
    rows = [{"medication": "warfarin", "dose": "4 mg"}]
    for hadm_id in sorted(INPATIENT):
        payload = _payload_for(sweep, hadm_id, "meds")
        if not (payload.get("passages") or []):
            continue
        recorded = _recorded_meds(hadm_id, dict(payload))
        deterministic._consolidate_medication_passages(recorded)
        answer = deterministic.medication_narrative(rows, 1)
        try:
            out = compose_presentation(
                "What medications is she going home on?", answer, recorded,
                {"medications": rows, "dropped": [], "returned_passages": 1},
                "meds",
            )
        except Exception as exc:  # noqa: BLE001
            broken[hadm_id] = f"{type(exc).__name__}: {exc}"
            continue
        if not out["sources"]:
            broken[hadm_id] = "the ^[1] marker resolved to nothing"
        elif out["sources"][0]["text"] != recorded[0]["response"]["passages"][0]["text"]:
            broken[hadm_id] = "the cited passage is not the consolidated one"
    assert not broken, broken


# --- 3. the laboratory tool and its card -------------------------------------

def test_get_lab_trends_returns_a_valid_schema_for_every_inpatient(sweep):
    broken: dict[int, str] = {}
    for hadm_id in sorted(INPATIENT):
        payload = _payload_for(sweep, hadm_id, "labs")
        if payload.get("error"):
            broken[hadm_id] = f"error: {payload['error']}"
            continue
        trends = payload.get("trends")
        if not isinstance(trends, list) or not trends:
            broken[hadm_id] = "no trends"
            continue
        for trend in trends:
            if not isinstance(trend.get("label"), str) or not trend["label"]:
                broken[hadm_id] = f"trend without a label: {trend.get('key')!r}"
                break
            if not isinstance(trend.get("decimals"), int):
                broken[hadm_id] = f"decimals {trend.get('decimals')!r}"
                break
            series = trend.get("series")
            if not isinstance(series, list) or not series:
                broken[hadm_id] = f"empty series for {trend.get('key')!r}"
                break
            point = series[0]
            if not isinstance(point.get("value"), (int, float)) \
                    or not isinstance(point.get("flag"), str):
                broken[hadm_id] = f"malformed point in {trend.get('key')!r}"
                break
    assert not broken, broken


def test_every_lab_trajectory_has_usable_reference_bounds(sweep):
    broken: dict[int, list] = {}
    for hadm_id in sorted(INPATIENT):
        payload = _payload_for(sweep, hadm_id, "labs")
        bad = [
            {"key": trend.get("key"),
             "low": trend.get("reference_low"), "high": trend.get("reference_high")}
            for trend in payload.get("trends") or []
            if not all(isinstance(trend.get(bound), (int, float))
                       for bound in ("reference_low", "reference_high"))
        ]
        if bad:
            broken[hadm_id] = bad
    assert not broken, broken


# --- 4. the clinical note reader ---------------------------------------------

def test_every_returned_note_passage_carries_its_citation_metadata(sweep):
    """A passage that cannot be cited cannot be shown as evidence."""
    broken: dict[int, str] = {}
    for hadm_id in sorted(SERVED):
        payload = _payload_for(sweep, hadm_id, "notes")
        if payload.get("error"):
            broken[hadm_id] = f"error: {payload['error']}"
            continue
        for passage in payload.get("passages") or []:
            if not isinstance(passage.get("id"), str) or not passage["id"]:
                broken[hadm_id] = "passage without a note id"
                break
            if passage.get("section") not in SUMMARY_SECTIONS:
                broken[hadm_id] = f"unknown section {passage.get('section')!r}"
                break
            if not isinstance(passage.get("text"), str) or not passage["text"].strip():
                broken[hadm_id] = "empty passage text"
                break
    assert not broken, broken


def test_every_note_read_returns_sections_or_an_explained_empty(sweep):
    """All 87 records. A record with no summary headings is not a parse failure.

    Five records carry a note that contains none of the four discharge-summary headings
    — three inpatients (90000026, 90000033, 90000087) and two outpatients (90000051,
    90000073). The tool answers those with a structured empty and a sentence saying so,
    which is the correct answer: the note exists and simply is not a discharge summary.
    What must not happen is a bare empty with no explanation, or an exception.
    """
    unexplained: dict[int, str] = {}
    errored: dict[int, str] = {}
    for hadm_id in sorted(SERVED):
        payload = _payload_for(sweep, hadm_id, "notes")
        if payload.get("error"):
            errored[hadm_id] = str(payload["error"])
            continue
        if payload.get("passages"):
            continue
        note = payload.get("note")
        if not isinstance(note, str) or not note.strip():
            unexplained[hadm_id] = "no passages and no explanation"
    assert not errored, errored
    assert not unexplained, unexplained

    # The two shapes that exist, stated as counts so a change in either is visible.
    with_sections = [h for h in sorted(SERVED)
                     if (_payload_for(sweep, h, "notes").get("passages") or [])]
    assert len(with_sections) + 5 == len(SERVED), sorted(set(SERVED) - set(with_sections))
