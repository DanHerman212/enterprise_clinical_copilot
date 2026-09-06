"""Runtime contracts for MCP tool results."""

from math import isfinite
from typing import Any, NotRequired, TypedDict


class ToolError(TypedDict, total=False):
    hadm_id: int
    error: str
    message: str
    feature_source: str


class PredictionFactor(TypedDict):
    feature: str
    contribution: float
    direction: str


class PredictionResult(TypedDict):
    hadm_id: int
    probability: float
    threshold: float
    decision: int
    base_value: float
    top_factors: list[PredictionFactor]
    model_version: str
    feature_source: str


class RetrievalPassage(TypedDict):
    id: str
    section: str
    text: str
    score: NotRequired[float]
    retrieval: NotRequired[str]


class RetrievalResult(TypedDict):
    hadm_id: int
    query: str
    returned: int
    passages: list[RetrievalPassage]


class ToolContractError(ValueError):
    """An MCP tool returned a payload outside its declared contract."""


def tool_error(
    hadm_id: int,
    code: str,
    message: str,
    *,
    feature_source: str | None = None,
) -> ToolError:
    """Create the common structured error payload used by MCP tools."""
    payload: ToolError = {"hadm_id": hadm_id, "error": code, "message": message}
    if feature_source is not None:
        payload["feature_source"] = feature_source
    return payload


def _validate_error(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload.get("error"), str) or not isinstance(
        payload.get("message"), str
    ):
        raise ToolContractError("Tool error payload is malformed.")
    return payload


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def validate_prediction_result(payload: Any) -> PredictionResult | ToolError:
    if not isinstance(payload, dict):
        raise ToolContractError("Prediction payload is not an object.")
    if "error" in payload:
        return _validate_error(payload)
    required = (
        "hadm_id", "probability", "threshold", "decision", "base_value",
        "top_factors", "model_version", "feature_source",
    )
    if any(field not in payload for field in required):
        raise ToolContractError("Prediction payload is missing a required field.")
    if not isinstance(payload["hadm_id"], int) or isinstance(payload["hadm_id"], bool):
        raise ToolContractError("Prediction hadm_id is invalid.")
    if any(not _number(payload[field]) for field in ("probability", "threshold", "base_value")):
        raise ToolContractError("Prediction numeric fields are invalid.")
    if not isinstance(payload["decision"], int) or isinstance(payload["decision"], bool):
        raise ToolContractError("Prediction decision is invalid.")
    if not isinstance(payload["model_version"], str) or not isinstance(
        payload["feature_source"], str
    ):
        raise ToolContractError("Prediction provenance is invalid.")
    factors = payload["top_factors"]
    if not isinstance(factors, list):
        raise ToolContractError("Prediction factors are invalid.")
    for factor in factors:
        if not isinstance(factor, dict) or not isinstance(factor.get("feature"), str):
            raise ToolContractError("Prediction factor is malformed.")
        if not _number(factor.get("contribution")) or not isinstance(
            factor.get("direction"), str
        ):
            raise ToolContractError("Prediction factor is malformed.")
    return payload


def validate_retrieval_result(payload: Any) -> RetrievalResult | ToolError:
    if not isinstance(payload, dict):
        raise ToolContractError("Retrieval payload is not an object.")
    if "error" in payload:
        return _validate_error(payload)
    required = ("hadm_id", "query", "returned", "passages")
    if any(field not in payload for field in required):
        raise ToolContractError("Retrieval payload is missing a required field.")
    if not isinstance(payload["hadm_id"], int) or isinstance(payload["hadm_id"], bool):
        raise ToolContractError("Retrieval hadm_id is invalid.")
    if not isinstance(payload["query"], str) or not isinstance(payload["returned"], int):
        raise ToolContractError("Retrieval metadata is invalid.")
    passages = payload["passages"]
    if not isinstance(passages, list) or payload["returned"] != len(passages):
        raise ToolContractError("Retrieval passages are inconsistent.")
    for passage in passages:
        if not isinstance(passage, dict) or any(
            not isinstance(passage.get(field), str)
            for field in ("id", "section", "text")
        ):
            raise ToolContractError("Retrieval passage is malformed.")
        if passage.get("retrieval") == "deterministic":
            continue
        if not _number(passage.get("score")):
            raise ToolContractError("Retrieval passage is malformed.")
    return payload