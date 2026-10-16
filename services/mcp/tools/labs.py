"""The lab-trajectory tool: one admission's serial labs, for the drill-down card.

Phase 3 (2026-10-16). A clicked risk driver asks what a measurement did during the
admission, and the answer is a course rather than a number — so this returns a series per
analyte, each carrying its reference interval and a flag per point.

The values are GENERATED. Nothing here reads a laboratory table: the cohort is served under
a DUA that permits no redistribution of raw events, and the trajectories are synthesised
deterministically from the admission's own feature row — see
`services/mcp/dependencies/labs/synthetic.py`, which is where the anchoring guarantee and
its reasoning live.

The feature row is read through the same seam the predictor uses, which is the point: the
card cannot disagree with the driver bar that led to it, because one read answers both.
"""

import asyncio

from services.mcp.contracts import (
    LabTrendResult,
    ToolError,
    tool_error,
    validate_lab_trend_result,
)
from services.mcp.dependencies.features import FEATURE_SOURCE, get_feature_source
from services.mcp.dependencies.labs.synthetic import lab_trends

#: Analytes this tool will draw, named for the caller. Derived from the panel rather than
#: listed again, so a row added to the generator is reachable the moment it exists.
def known_analytes() -> list[str]:
    from services.mcp.dependencies.labs.synthetic import PANEL

    return [analyte.key for analyte in PANEL]


def _log_and_error(code: str, message: str, hadm_id: int) -> ToolError:
    return tool_error(code, message, hadm_id=hadm_id, feature_source=FEATURE_SOURCE)


async def get_lab_trends(hadm_id: int, analyte: str = "") -> LabTrendResult | ToolError:
    """Serial synthetic lab values for one hospital admission.

    Returns, for each analyte asked for, an admission-to-discharge trajectory with its
    reference interval and a flag per timepoint. The values are generated deterministically
    from the admission's own features — the same call gives the same series — and every
    value the model was shown for an analyte is reproduced exactly at that analyte's
    reporting precision.

    Not every analyte is a model input. The eight the booster scored are anchored on the
    feature row and named in each series' `anchored` field; the clinical panel (creatinine,
    eGFR, potassium, white cells) is generated from reference intervals because the model
    was never trained on them. `anchored` is how a reader tells the two apart.

    Args:
        hadm_id: MIMIC-IV hospital admission id.
        analyte: One analyte key to draw. Empty draws the full panel, which is what a
            drill-down that has no specific analyte in mind should ask for.
    """
    source = get_feature_source()
    if not await asyncio.to_thread(source.exists, hadm_id):
        return _log_and_error(
            "unknown_patient",
            f"No admission {hadm_id} in the feature source ({FEATURE_SOURCE}).",
            hadm_id,
        )

    wanted = [analyte] if analyte else None
    if wanted is not None and analyte not in known_analytes():
        # A refused name rather than an empty list: an empty panel and a misspelled
        # analyte are different answers, and only one of them means "nothing to show".
        return _log_and_error(
            "unknown_analyte",
            f"{analyte!r} is not an analyte this tool draws "
            f"({', '.join(known_analytes())}).",
            hadm_id,
        )

    try:
        # BigQuery is synchronous. Under the HTTP transport a blocking read stalls the
        # event loop for every concurrent caller, so the work goes to a worker thread —
        # the same reason `predict_readmission` does.
        trends = await asyncio.to_thread(
            lab_trends, hadm_id, wanted, None, "",
        )
    except KeyError:
        return _log_and_error(
            "unknown_patient",
            f"No admission {hadm_id} in the feature source ({FEATURE_SOURCE}).",
            hadm_id,
        )
    except Exception:
        return _log_and_error(
            "lab_trends_failed",
            "The lab trajectories could not be generated for this admission.",
            hadm_id,
        )

    payload = {
        "hadm_id": hadm_id,
        "returned": len(trends),
        "trends": trends,
        "synthetic": True,
        "note": (
            "Generated from this admission's features, not read from a laboratory "
            "record. Values the model scored are reproduced exactly."
        ),
    }
    return validate_lab_trend_result(payload)
