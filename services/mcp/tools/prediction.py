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
from ..config import FEATURE_SOURCE
from ..dependencies.features import (
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
    return tool_error(code, message, hadm_id=hadm_id, feature_source=FEATURE_SOURCE)


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

    factors = [_scored_factor(f, row) for f in pred.get("top_factors", [])[:MAX_FACTORS]]

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


def _row_value(row: dict, feature: str) -> float | None:
    """The value this admission's row carries for one feature, or None.

    The row arrives from BigQuery with every column as a string, so the value that needs
    care is usually `"15.2"` rather than 15.2. A column that is absent, null or not a
    number returns None — never 0, which is a value the model read.

    Only a RAW column resolves. Some attributions name an aggregated parent
    (`discharge_location`, which is thirteen one-hot columns), and those have no single
    value to state; the manifest block says them in words and this says nothing rather
    than picking one column out of the encoding.
    """
    raw = row.get(feature)
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


#: How many decimals a served feature value is published with.
#:
#: A row's value is an ENCODED feature and carries binary noise: an index length of stay
#: arrives as `42.458333333333336`. The prompt tells the model to state a served value
#: exactly as it is given, so the precision a reader sees has to be decided before the
#: model sees it — otherwise the prose copies fourteen digits, or the guardrail scrapes
#: them off afterwards with a regex that only fires on three or more decimals.
#:
#: Two, not one: a Gate 3 item was scored a fidelity failure for printing `24.7` where the
#: row holds `24.6667`, and `24.67` keeps the record recognisable. The binary tail is what
#: goes, not the measurement.
_FEATURE_VALUE_DECIMALS = 2


def _clinical_value(value: float) -> float | int:
    """One feature value at the precision the payload publishes.

    An integer-valued float becomes an integer, because a count of `0.0` prior admissions
    is a formatted number rather than a measurement, and the prompt's exact-quoting rule
    would print the `0.0` at a reader.
    """
    rounded = round(float(value), _FEATURE_VALUE_DECIMALS)
    return int(rounded) if rounded == int(rounded) else rounded


def _scored_factor(factor: dict, row: dict) -> dict:
    """One attribution, with the patient's own value for the feature beside it.

    The attribution says how much a feature moved this admission's estimate; it says
    nothing about what this admission's value OF that feature was, and until this existed
    nothing else did either for a laboratory feature. Asked "what in this patient's chart
    explains red cell distribution width as a readmission risk driver", the agent could
    reach the attribution and the note text and neither held the number, so it answered
    that the chart did not document the laboratory value — which was true of everything it
    had been given (reported 2026-11-02).

    The value is the row's own, not a summary of it: `rdw_max` is stated as the row holds
    it, because that is the number the booster scored. It is PUBLISHED at clinical
    precision (`_clinical_value`) rather than raw: the model is instructed to quote a
    served value exactly, and a length of stay of `42.458333333333336` days in a clinical
    sentence is a formatting fault the reader has to read past. The rounding happens here,
    once, so the number the model quotes, the number the guardrail verifies against and
    the number a later turn replays are the same number.
    """
    entry: dict[str, Any] = {
        "feature": factor["feature"],
        "contribution": round(float(factor["attribution"]), 4),
        "direction": "increases" if float(factor["attribution"]) > 0 else "decreases",
    }
    feature = str(factor["feature"])
    if feature not in row:
        # An aggregated parent (`discharge_location` is thirteen one-hot columns) has no
        # single value to state. The key is omitted rather than nulled: a null would name
        # a column whose absence it describes, and there is none.
        return entry
    value = _row_value(row, feature)
    if value is None:
        # The column exists and this admission's row holds NULL. The booster scored a
        # missing value, and that is a fact about the evidence rather than about the
        # patient — so it is stated explicitly as null instead of being dropped, which
        # would make it indistinguishable from the aggregated parent above.
        #
        # It is NEVER filled from `get_lab_trends`. The series is a different measurement
        # (a last reading is not a maximum), so a number taken from it would be presented
        # as the input the model read when the model read nothing.
        entry["value"] = None
        return entry
    entry["value"] = _clinical_value(value)
    return entry


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
