"""The lab series ranks itself, so the model never has to compare digits.

Gate 2, `dev_t3_03`: "which lab was worst at discharge?" was answered with haemoglobin,
"the laboratory value furthest outside its reference range". Haemoglobin was `low`, and so
were the red cells and the RDW — the per-timepoint `flag` cannot rank within a category, so
the answer picked one. The judge normalised each analyte against its reference width and
found the red cells further out.

The prompt forbids the model from doing that arithmetic — a digit the model holds is a
digit it can misread — so the ranking is computed in the projection instead and arrives as
`excursion`. These tests pin the arithmetic and the fact that the recorded payload is
untouched.
"""

from __future__ import annotations

import pytest

from services.agent.engine.mcp_client import _excursion, _presented_for_model

#: The three abnormal values the judge scored, with their intervals.
TRENDS = {
    "trends": [
        {"key": "rdw", "label": "RDW", "unit": "%", "reference_low": 11.5,
         "reference_high": 14.5, "decimals": 1,
         "series": [{"day": 0.0, "value": 12.0, "flag": "normal"},
                    {"day": 9.1, "value": 17.5, "flag": "high"}]},
        {"key": "rbc", "label": "Red blood cells", "unit": "x10^6/uL",
         "reference_low": 4.2, "reference_high": 5.4, "decimals": 2,
         "series": [{"day": 0.0, "value": 3.42, "flag": "low"},
                    {"day": 9.1, "value": 2.69, "flag": "low"}]},
        {"key": "hgb", "label": "Hemoglobin", "unit": "g/dL", "reference_low": 12.0,
         "reference_high": 15.5, "decimals": 1,
         "series": [{"day": 0.0, "value": 9.1, "flag": "low"},
                    {"day": 9.1, "value": 10.0, "flag": "low"}]},
    ]
}


def _discharge(trends: dict) -> dict[str, float]:
    return {trend["key"]: trend["series"][-1]["excursion"]
            for trend in trends["trends"]}


def test_the_visit_is_annotated_with_an_excursion_per_timepoint():
    presented = _presented_for_model("get_lab_trends", TRENDS)

    for trend in presented["trends"]:
        for point in trend["series"]:
            assert "excursion" in point


def test_the_ranking_is_the_one_the_judge_computed():
    """RBC 1.258, RDW 1.0, haemoglobin 0.571 — the order the judge derived by hand."""
    discharge = _discharge(_presented_for_model("get_lab_trends", TRENDS))

    assert discharge["rbc"] > discharge["rdw"] > discharge["hgb"]
    assert discharge["hgb"] == pytest.approx(0.571, abs=1e-3)


def test_a_value_inside_its_interval_has_zero_excursion():
    assert _excursion(4.2, 5.4, 5.0) == 0.0


def test_the_interval_width_is_what_normalises_the_ranking():
    """A plain distance ranks the wider interval above the narrower one however abnormal
    the reading is; 1.0 unit out of 1.2 beats 3.0 units out of 3.0."""
    assert _excursion(4.2, 5.4, 2.69) > _excursion(11.5, 14.5, 17.5)


def test_a_missing_interval_is_not_ranked():
    assert _excursion(None, None, 4.0) is None


def test_the_recorded_payload_is_untouched():
    """The caller's payload and the execution record say exactly what the tool said."""
    before = {trend["key"]: [dict(point) for point in trend["series"]]
              for trend in TRENDS["trends"]}

    _presented_for_model("get_lab_trends", TRENDS)

    after = {trend["key"]: [dict(point) for point in trend["series"]]
             for trend in TRENDS["trends"]}
    assert after == before
    assert all("excursion" not in point
               for points in before.values() for point in points)


def test_another_tool_passes_through_untouched():
    payload = {"trends": [{"key": "x"}]}
    assert _presented_for_model("discharge_medications", payload) == payload
