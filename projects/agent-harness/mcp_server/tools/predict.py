"""The predict_readmission tool.

Returns plain JSON, never A2UI. A tool that returns UI is welded to one
presentation layer and stops being usable from Claude Desktop or CI; the agent
composes A2UI from this payload in §16 of the build guide.
"""

import asyncio
from functools import lru_cache
from typing import Any

from ..config import FEATURE_SOURCE
from ..endpoint import predict_one
from ..features import FeatureSource, get_feature_source, to_vector
from ..features.manifest import feature_order, model_version

# The risk card renders five; returning all 23 parent groups would just be
# tokens the model has to skim past.
MAX_FACTORS = 5


@lru_cache(maxsize=1)
def _source() -> FeatureSource:
    """Cached so the BigQuery / Feature Store client is built once."""
    return get_feature_source()


def _error(hadm_id: int, code: str, message: str) -> dict[str, Any]:
    """A failure the agent can read and explain, rather than a stack trace."""
    return {
        "hadm_id": hadm_id,
        "error": code,
        "message": message,
        "feature_source": FEATURE_SOURCE,
    }


def _predict(hadm_id: int) -> dict[str, Any]:
    """Blocking implementation. Wrapped in a thread by the tool below."""
    order = feature_order()

    try:
        row = _source().fetch(hadm_id)
    except KeyError:
        return _error(
            hadm_id, "unknown_patient",
            f"No admission {hadm_id} in the feature source ({FEATURE_SOURCE}).",
        )
    except Exception as exc:  # network, auth, unsynced online store
        return _error(hadm_id, "feature_fetch_failed", f"{type(exc).__name__}: {exc}")

    # A missing *value* is legitimate — the model reads null as NaN by design.
    # A missing *column* is not: to_vector fills absent keys with None, so a
    # short row would silently shift every feature after the gap.
    missing = [col for col in order if col not in row]
    if missing:
        return _error(
            hadm_id, "incomplete_features",
            f"Feature source returned {len(order) - len(missing)}/{len(order)} columns; "
            f"missing {missing[:5]}{'…' if len(missing) > 5 else ''}.",
        )

    try:
        pred = predict_one(to_vector(row, order))
    except Exception as exc:
        return _error(hadm_id, "prediction_failed", f"{type(exc).__name__}: {exc}")

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
        "model_version": model_version(),
        "feature_source": FEATURE_SOURCE,
    }


async def predict_readmission(hadm_id: int) -> dict[str, Any]:
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
    return await asyncio.to_thread(_predict, hadm_id)
