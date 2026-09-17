"""The predict_readmission tool.

Returns plain JSON, never A2UI. A tool that returns UI is welded to one
presentation layer and stops being usable from Claude Desktop or CI; the agent
composes A2UI from this payload in §16 of the build guide.
"""

import asyncio
import logging
from functools import lru_cache
from typing import Any

from ..dependencies.model_endpoint import predict_one
from ..contracts import (
    PredictionResult,
    ToolContractError,
    ToolError,
    tool_error,
    validate_prediction_result,
)
from ..dependencies.features import (
    FEATURE_SOURCE,
    FeatureSource,
    get_feature_source,
    to_instance,
)
from ..dependencies.features.manifest import feature_order
from ._validation import valid_hadm_id

# The risk card renders five; returning all 23 parent groups would just be
# tokens the model has to skim past.
MAX_FACTORS = 5

_LOG = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _source() -> FeatureSource:
    """Cached so the BigQuery client is built once."""
    return get_feature_source()


def _error(
    hadm_id: int, code: str, message: str, *, detail: str | None = None
) -> dict[str, Any]:
    """A failure the caller can act on, with the detail left in the log.

    The message IS the tool result: it reaches the model, the caller and the
    browser, so it carries a sentence and nothing else. Column names and feature
    source internals are diagnostics; they go to the log.
    """
    if detail:
        _LOG.warning("%s refused for hadm %s: %s", code, hadm_id, detail)
    return tool_error(hadm_id, code, message, feature_source=FEATURE_SOURCE)


def _model_identity(pred: dict[str, Any], deployed_model_id: str) -> str:
    """What produced this number, as reported by the endpoint that ran it.

    The deployment stamps the provenance record's name into the container
    (``deploy_cpr.py``), so the response can name the model in the terms the
    registry uses. If an older deployment carries no stamp, the deployed model's
    own resource id is still the truth about what answered. What is *not*
    acceptable is the registry lookup this replaced: it named the newest
    registered record, which is a different thing from the one that served.
    """
    named = pred.get("model_version")
    if isinstance(named, str) and named:
        return named
    return deployed_model_id or "unknown"


def _predict(hadm_id: int) -> dict[str, Any]:
    """Blocking implementation. Wrapped in a thread by the tool below."""
    if not valid_hadm_id(hadm_id):
        return _error(hadm_id, "bad_request", "hadm_id must be a positive integer")
    order = feature_order()

    try:
        row = _source().fetch(hadm_id)
    except KeyError:
        return _error(
            hadm_id, "unknown_patient",
            f"No admission {hadm_id} in the feature source ({FEATURE_SOURCE}).",
        )
    except Exception as exc:  # network, auth, unsynced online store
        # Detail stays server-side (ECC-21): exception text can embed table
        # names, project ids and URLs, and the error message reaches the model.
        _LOG.error("predict: feature fetch failed for hadm %s", hadm_id, exc_info=exc)
        return _error(
            hadm_id, "feature_fetch_failed",
            "The feature source could not be read. Try again shortly.",
        )

    # A missing *value* is legitimate — the model reads null as NaN by design.
    # A missing *column* is not: `to_instance` fills absent keys with None, so a
    # short row would silently shift every feature after the gap.
    missing = [col for col in order if col not in row]
    if missing:
        return _error(
            hadm_id, "incomplete_features",
            "The feature source returned an incomplete row for this admission.",
            detail=(
                f"{FEATURE_SOURCE} returned {len(order) - len(missing)}/{len(order)} "
                f"columns; missing {missing}"
            ),
        )

    try:
        pred, deployed_model_id = predict_one(to_instance(row, order))
    except Exception as exc:
        _LOG.error("predict: prediction failed for hadm %s", hadm_id, exc_info=exc)
        return _error(
            hadm_id, "prediction_failed",
            "The prediction service could not be reached. Try again shortly.",
        )

    # Which model answered is worth a line: it is the end of the chain from a
    # surprising number back to the artifact that produced it, and it costs one
    # log statement to have it in the record.
    _LOG.info(
        "predict: hadm %s served by %s (deployed model %s)",
        hadm_id, _model_identity(pred, deployed_model_id), deployed_model_id or "unknown",
    )

    factors = [
        {
            "feature": f["feature"],
            "contribution": round(float(f["attribution"]), 4),
            "direction": "increases" if float(f["attribution"]) > 0 else "decreases",
        }
        for f in pred.get("top_factors", [])[:MAX_FACTORS]
    ]

    return {
        "hadm_id": hadm_id,
        "probability": round(float(pred["probability"]), 6),
        "threshold": float(pred["threshold"]),
        "decision": int(pred["prediction"]),
        "base_value": round(float(pred["base_value"]), 6),
        "top_factors": factors,
        "model_version": _model_identity(pred, deployed_model_id),
        "feature_source": FEATURE_SOURCE,
    }


async def predict_readmission(hadm_id: int) -> PredictionResult | ToolError:
    """Predict 30-day unplanned readmission risk for one hospital admission.

    Returns the calibrated probability, the threshold decision, and the feature
    attributions that drove it (TreeSHAP, aggregated to clinical parent
    features, in logit space). A positive contribution increases risk.

    Args:
        hadm_id: MIMIC-IV hospital admission id.
    """
    # BigQuery and Vertex calls are synchronous. Under the HTTP transport a
    # blocking tool stalls the event loop for every concurrent caller, so the
    # work goes to a worker thread.
    payload = await asyncio.to_thread(_predict, hadm_id)
    try:
        return validate_prediction_result(payload)
    except ToolContractError:
        _LOG.error("predict: produced payload outside its tool contract")
        return _error(hadm_id, "invalid_tool_response", "Prediction response was malformed.")
