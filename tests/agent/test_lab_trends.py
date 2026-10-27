"""The lab-trajectory generator, and the one promise it exists to keep.

Phase 3 (2026-10-16). The generator's reason for existing is the anchor guarantee: a value
the model was scored on is reproduced by the trajectory built to explain it. Everything here
is either that promise, a way it was broken once, or the determinism that lets a reader see
the same series twice.

Offline. The feature row is supplied rather than fetched, so no BigQuery and no credentials
are involved — which is also why the generator takes `features`: a callable that could only
be tested against a live table would be tested by nobody.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.mcp.contracts import (  # noqa: E402
    ToolContractError,
    validate_lab_trend_result,
)
from services.mcp.dependencies.labs import synthetic as syn  # noqa: E402

#: Admission 90000015's own feature values, as served — read from the scored cohort the
#: demo seed is built from. Inline rather than read from that file: it is a build artifact
#: of the demo pipeline and a test that needs it would fail on a checkout that has none.
ROW = {
    "hemoglobin_min": 12.1,
    "monocytes_min": 0.7,
    "rbc_last": 4.03,
    "rbc_min": 3.73,
    "rdw_max": 13.5,
    "sodium_last": 139.0,
    "sodium_max": 141.0,
    "sodium_min": 137.0,
    # From the cohort's own summary line for this admission: "61F · elective admission ·
    # 1 prior admission · 1 procedure · 8-day stay · discharged to rehab".
    "index_los_days": 8.0,
    "has_procedure": True,
}


def _by_key(trends):
    return {series["key"]: series for series in trends}


def _values(series):
    return [point["value"] for point in series["series"]]


def test_every_anchored_value_is_reproduced_exactly():
    """The promise. A driver bar and the card it opens show one number, not two.

    Asserted as equality on the model's own values rather than as proximity, because a
    trajectory that lands NEAR the driver's value is the failure mode this generator was
    built to make impossible — and it is the failure a fitting-based generator produces.
    """
    trends = _by_key(syn.lab_trends(90000015, features=ROW))

    assert max(_values(trends["rdw"])) == ROW["rdw_max"]
    assert min(_values(trends["rbc"])) == ROW["rbc_min"]
    assert _values(trends["rbc"])[-1] == ROW["rbc_last"]
    assert min(_values(trends["hemoglobin"])) == ROW["hemoglobin_min"]
    assert min(_values(trends["sodium"])) == ROW["sodium_min"]
    assert max(_values(trends["sodium"])) == ROW["sodium_max"]
    assert _values(trends["sodium"])[-1] == ROW["sodium_last"]
    assert min(_values(trends["monocytes"])) == ROW["monocytes_min"]


def test_a_lone_maximum_anchor_is_never_exceeded():
    """The bug that made the window rule what it is.

    `rdw_max` alone was read as both bounds, so the window became [13.5, 14.2] and every
    value in the series sat ABOVE the maximum it was supposed to respect. A one-sided anchor
    now takes the textbook interval's far bound as its other end.
    """
    trends = _by_key(syn.lab_trends(90000015, features=ROW, analytes=["rdw"]))

    assert max(_values(trends["rdw"])) == ROW["rdw_max"]
    assert all(value <= ROW["rdw_max"] for value in _values(trends["rdw"]))


def test_a_final_value_does_not_erase_the_peak():
    """The second bug: two anchors claiming one index.

    Sodium carries a minimum, a maximum and a last value. Writing them in order let the
    final value overwrite the peak whenever the peak fell on the last timepoint, so a
    sodium whose maximum was 141 appeared to top out at 140 — a card that disagreed with
    the model about a number it had itself written.
    """
    trends = _by_key(syn.lab_trends(90000015, features=ROW, analytes=["sodium"]))
    values = _values(trends["sodium"])

    assert max(values) == ROW["sodium_max"] == 141.0
    assert values[-1] == ROW["sodium_last"] == 139.0


def test_the_same_admission_produces_the_same_series():
    """Determinism, which is what lets a stored turn be re-rendered.

    A hash of the admission and the analyte rather than a seeded `random`: the series must
    survive a new process, and `hash()` over a string does not survive even that, being
    salted per process.
    """
    first = syn.lab_trends(90000015, features=ROW)
    second = syn.lab_trends(90000015, features=ROW)

    assert first == second
    # And a different admission is a different patient, not a re-ordered one.
    assert syn.lab_trends(90000023, features=ROW) != first


def test_a_conditional_analyte_is_absent_without_its_condition():
    """A BNP on a knee replacement is a test nobody ordered.

    NT-proBNP is drawn only when note text names a cardiac condition. An absent note is not
    evidence of a healthy heart, so the row is omitted rather than drawn normal — and the
    payload is simply short one series, which the count field reports honestly.
    """
    orthopedic = _by_key(syn.lab_trends(90000015, features=ROW))
    cardiac = _by_key(syn.lab_trends(
        90000015, features=ROW, note_text="acute decompensated heart failure"))

    assert "nt_probnp" not in orthopedic
    assert "nt_probnp" in cardiac


def test_a_flag_reads_the_reference_interval_not_the_drawn_window():
    """A value below the textbook range is flagged, not quietly normalised.

    The window for an anchored analyte is built from the patient's own extremes, which may
    sit outside the population range. The flag has to follow the reference interval, or a
    card would report a low haemoglobin as normal because it is the value the model scored.
    """
    trends = _by_key(syn.lab_trends(90000015, features=ROW))
    rbc = trends["rbc"]

    assert rbc["series"][1]["value"] == ROW["rbc_min"] == 3.73
    assert rbc["series"][1]["flag"] == "low", "3.73 is below the 4.20-5.40 interval"


def test_every_series_is_a_course_and_not_scatter():
    """Six timepoints, in order, spread across the stay.

    The first version drew each point independently, which produced a plausible set of
    numbers in an implausible sequence. A clinician reads a trajectory, so the shape is
    control points and the hash only perturbs them.
    """
    trends = syn.lab_trends(90000015, features=ROW, analytes=["hemoglobin"])
    days = [point["day"] for point in trends[0]["series"]]

    assert len(days) == syn.MAX_POINTS
    assert days == sorted(days)
    assert days[0] == 0.0
    assert days[-1] == ROW["index_los_days"]


def test_the_payload_says_it_is_generated():
    """`synthetic` is in the contract so a surface cannot pass a drawing off as a record."""
    payload = {
        "hadm_id": 90000015,
        "returned": 1,
        "synthetic": True,
        "trends": [syn.lab_trends(90000015, features=ROW, analytes=["hemoglobin"])[0]],
    }

    assert validate_lab_trend_result(payload) is payload

    with pytest.raises(ToolContractError):
        validate_lab_trend_result({**payload, "synthetic": False})
    with pytest.raises(ToolContractError):
        validate_lab_trend_result({**payload, "returned": 2})


def test_an_analyte_the_contract_defines_is_the_only_flag_vocabulary():
    """A flag outside the contract would render as an unrecognised state in the card, which
    is why the validator refuses one rather than passing it through."""
    payload = {
        "hadm_id": 1,
        "returned": 1,
        "synthetic": True,
        "trends": [{
            "key": "rdw", "label": "RDW", "unit": "%",
            "reference_low": 11.5, "reference_high": 14.5, "decimals": 1,
            "anchored": [],
            "series": [{"day": 0.0, "value": 13.5, "flag": "borderline"}],
        }],
    }

    with pytest.raises(ToolContractError):
        validate_lab_trend_result(payload)


# --- composition -------------------------------------------------------------

def _lab_payload():
    trends = syn.lab_trends(90000015, features=ROW, analytes=["hemoglobin"])
    return {
        "hadm_id": 90000015,
        "returned": len(trends),
        "trends": trends,
        "synthetic": True,
        "note": "Generated from this admission's features.",
    }


def _components(out):
    return out["a2ui"]["messages"][1]["updateComponents"]["components"]


def test_a_lab_turn_draws_the_card_and_a_caption_that_describes_it():
    """The card is the answer, so it is second in the column and the caption names it.

    The caption matters more than it looks: without a branch for it, a drill-down turn was
    served with "No 30-day readmission risk estimate was requested for this question" —
    true, and useless to a reader who asked what a value did.
    """
    from services.agent.presentation.a2ui import compose_presentation

    out = compose_presentation(
        "What did her hemoglobin do during this admission?",
        "It fell to 12.1 g/dL and recovered before discharge.",
        [{"name": "get_lab_trends", "args": {"hadm_id": 90000015},
          "response": _lab_payload()}],
    )

    types = [c["component"] for c in _components(out)]
    # Directly under the narrative, which is the claim the composer makes: the sentence
    # describes the course and the card is the course, so nothing may sit between them.
    assert types[:3] == ["Column", "CitedNarrative", "LabTrendCard"]
    assert out["a2ui"]["fallback_text"].startswith("1 laboratory trajectory")


def test_a_lab_card_is_evidence_enough_to_keep_the_scope():
    """A drill-down reads the patient's chart, so the badge is right — but the rule that
    withholds a chart scope with no NOTE citation must not fire, or every lab turn drops its
    pill and raises a safety swap over an answer nothing was removed from."""
    from services.agent.presentation.a2ui import compose_presentation

    out = compose_presentation(
        "What did her hemoglobin do during this admission?",
        "It fell to 12.1 g/dL and recovered before discharge.",
        [{"name": "get_lab_trends", "args": {"hadm_id": 90000015},
          "response": _lab_payload()}],
    )

    assert out["scope"] == "patient_chart"
    assert out["scope_suppressed"] is False


def test_a_failed_lab_call_draws_no_card():
    """A course with no values in it is not a shorter answer, it is a wrong one. The turn
    still says what happened; the card is simply not drawn."""
    from services.agent.presentation.a2ui import compose_presentation

    out = compose_presentation(
        "What did her hemoglobin do during this admission?",
        "The lab trajectories could not be generated for this admission.",
        [{"name": "get_lab_trends", "args": {"hadm_id": 90000015},
          "response": {"error": "lab_trends_failed",
                       "message": "The lab trajectories could not be generated."}}],
    )

    assert "LabTrendCard" not in [c["component"] for c in _components(out)]
