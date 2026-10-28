"""The lab panel reaches the model exactly as the tool returned it.

The projection used to rewrite this payload: every analyte carried an ordinal
`_abnormality_rank` and `_abnormality_rank_at_latest`, computed from how far it sat
outside its reference interval (and, before that, a float `excursion` that one answer
quoted as a clinical measurement). The rank, the excursion arithmetic and the projection
itself have been removed, so these tests are now about absence: nothing is added to a lab
payload, and a lab payload is treated like every other tool result.
"""

from __future__ import annotations

from services.agent.engine import mcp_client

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
        {"key": "na", "label": "Sodium", "unit": "mmol/L", "reference_low": 135,
         "reference_high": 145, "decimals": 0,
         "series": [{"day": 0.0, "value": 138, "flag": "normal"},
                    {"day": 9.1, "value": 140, "flag": "normal"}]},
    ]
}


def test_a_lab_panel_passes_through_untouched():
    presented = mcp_client._presented_for_model("get_lab_trends", TRENDS)

    assert presented == TRENDS
    for trend in presented["trends"]:
        assert "_abnormality_rank" not in trend
        assert "_abnormality_rank_at_latest" not in trend


def test_the_ranking_machinery_no_longer_exists():
    """The removal, asserted rather than implied: no key, no helper, no projection."""
    for name in ("ABNORMALITY_RANK", "ABNORMALITY_RANK_AT_LATEST",
                 "_presented_labs", "_excursion"):
        assert not hasattr(mcp_client, name), name


def test_the_recorded_payload_is_untouched():
    """The caller's payload and the execution record say exactly what the tool said."""
    before = {trend["key"]: [dict(point) for point in trend["series"]]
              for trend in TRENDS["trends"]}
    before_trends = [dict(trend) for trend in TRENDS["trends"]]

    mcp_client._presented_for_model("get_lab_trends", TRENDS)

    after = {trend["key"]: [dict(point) for point in trend["series"]]
             for trend in TRENDS["trends"]}
    assert after == before
    assert [dict(trend) for trend in TRENDS["trends"]] == before_trends


def test_another_tool_passes_through_untouched():
    payload = {"passages": [{"section": "discharge_medications", "text": "text"}]}
    assert mcp_client._presented_for_model("discharge_medications", payload) == payload
