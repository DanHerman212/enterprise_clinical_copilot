"""The lab series ranks itself, and publishes an ORDER rather than a quantity.

Gate 2, `dev_t3_03`: "which lab was worst at discharge?" was answered with haemoglobin,
"the laboratory value furthest outside its reference range". Haemoglobin was `low`, and so
were the red cells and the RDW — the per-timepoint `flag` cannot rank within a category, so
the answer picked one. The judge normalised each analyte against its reference width and
found the red cells further out.

The prompt forbids the model from doing that arithmetic — a digit the model holds is a
digit it can misread — so the ordering is computed in the projection instead.

Earlier that ordering travelled as a float (`excursion`, `excursion_max`). On the `00c8`
retest the model quoted it: "an excursion of 1.258". The judge sees only what the tool
returned, could not match that number to anything, and scored it as fabrication. It now
travels as an ordinal rank, which cannot be mistaken for a measurement.

These tests pin the arithmetic, the published order, and the fact that the record is
untouched.
"""

from __future__ import annotations

from services.agent.engine.mcp_client import (
    ABNORMALITY_RANK,
    ABNORMALITY_RANK_AT_LATEST,
    _excursion,
    _presented_for_model,
)

#: The three abnormal values the judge scored, with their intervals. Deviations at the
#: last timepoint: red cells 1.258, RDW 1.0, haemoglobin 0.571.
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
        {"key": "na", "label": "Sodium", "unit": "mmol/L", "reference_low": 135,
         "reference_high": 145, "decimals": 0,
         "series": [{"day": 0.0, "value": 138, "flag": "normal"},
                    {"day": 9.1, "value": 140, "flag": "normal"}]},
    ]
}


def _ranks(presented: dict, key: str) -> dict[str, int]:
    return {trend["label"]: trend[key] for trend in presented["trends"]
            if key in trend}


def test_the_abnormal_analytes_are_ranked_furthest_out_first():
    """Red cells 1, RDW 2, haemoglobin 3 — the order the judge derived by hand."""
    ranks = _ranks(_presented_for_model("get_lab_trends", TRENDS), ABNORMALITY_RANK)

    assert ranks == {"Red blood cells": 1, "RDW": 2, "Hemoglobin": 3}


def test_the_rank_at_the_latest_timepoint_is_the_one_discharge_asks_about():
    ranks = _ranks(
        _presented_for_model("get_lab_trends", TRENDS), ABNORMALITY_RANK_AT_LATEST)

    assert ranks == {"Red blood cells": 1, "RDW": 2, "Hemoglobin": 3}


def test_the_two_orders_are_independent():
    """An analyte worst early and normal by discharge ranks first on one and not the other.

    This is why there are two keys: "which lab was worst" and "which lab was worst at
    discharge" are different questions, and the prompt already forbids answering the
    second with a mid-stay nadir.
    """
    panel = {"trends": [
        {"label": "Creatinine", "reference_low": 0.5, "reference_high": 1.2,
         "series": [{"day": 0.0, "value": 4.0, "flag": "high"},
                    {"day": 9.0, "value": 1.0, "flag": "normal"}]},
        {"label": "Potassium", "reference_low": 3.5, "reference_high": 5.0,
         "series": [{"day": 0.0, "value": 4.0, "flag": "normal"},
                    {"day": 9.0, "value": 6.5, "flag": "critical_high"}]},
    ]}

    presented = _presented_for_model("get_lab_trends", panel)

    assert _ranks(presented, ABNORMALITY_RANK)["Creatinine"] == 1
    assert ABNORMALITY_RANK_AT_LATEST not in presented["trends"][0]
    assert _ranks(presented, ABNORMALITY_RANK_AT_LATEST) == {"Potassium": 1}


def test_an_analyte_inside_its_interval_carries_no_rank():
    presented = _presented_for_model("get_lab_trends", TRENDS)
    sodium = next(trend for trend in presented["trends"] if trend["label"] == "Sodium")

    assert ABNORMALITY_RANK not in sodium
    assert ABNORMALITY_RANK_AT_LATEST not in sodium


def test_no_float_from_the_arithmetic_reaches_the_model():
    """The number that orders the panel is internal; only the order is published.

    `00c8` quoted `excursion: 1.258` in an answer and the judge scored it as invented:
    it is computed here, after the tool returned, so the tool's own output has no such
    field to verify against.
    """
    presented = _presented_for_model("get_lab_trends", TRENDS)

    for trend in presented["trends"]:
        assert "excursion" not in trend
        assert "excursion_max" not in trend
        for point in trend["series"]:
            assert "excursion" not in point

    ranks = [value for key in (ABNORMALITY_RANK, ABNORMALITY_RANK_AT_LATEST)
             for value in _ranks(presented, key).values()]
    assert ranks
    assert all(isinstance(rank, int) and not isinstance(rank, bool) for rank in ranks)


def test_a_tie_breaks_on_the_label_so_the_order_is_reproducible():
    panel = {"trends": [
        {"label": "Zeta", "reference_low": 1.0, "reference_high": 2.0,
         "series": [{"day": 1, "value": 3.0}]},
        {"label": "Alpha", "reference_low": 1.0, "reference_high": 2.0,
         "series": [{"day": 1, "value": 3.0}]},
    ]}

    ranks = _ranks(_presented_for_model("get_lab_trends", panel), ABNORMALITY_RANK)

    assert ranks == {"Alpha": 1, "Zeta": 2}


def test_a_value_inside_its_interval_has_zero_deviation():
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
    before_trends = [dict(trend) for trend in TRENDS["trends"]]

    _presented_for_model("get_lab_trends", TRENDS)

    after = {trend["key"]: [dict(point) for point in trend["series"]]
             for trend in TRENDS["trends"]}
    assert after == before
    assert [dict(trend) for trend in TRENDS["trends"]] == before_trends


def test_another_tool_passes_through_untouched():
    payload = {"trends": [{"key": "x"}]}
    assert _presented_for_model("discharge_medications", payload) == payload
