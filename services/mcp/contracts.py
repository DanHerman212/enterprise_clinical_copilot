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
    # Set when the serving chunker could not reproduce the matched chunk and the whole
    # note is returned instead. Declared here because a key the schema does not declare is
    # dropped from the structured payload the client receives.
    granularity: NotRequired[str]


class RetrievalResult(TypedDict):
    hadm_id: int
    query: str
    returned: int
    passages: list[RetrievalPassage]
    # Present when an empty result needs explaining — no discharge note, or none of the
    # summary sections. Declared for the same reason as `granularity`.
    note: NotRequired[str]


class LiteratureArticle(TypedDict):
    """One PubMed record, as the console renders it and the model cites it."""

    pmid: str
    title: str
    journal: str
    # As the record states it — "2025 Mar 12", or "2025 Jan-Feb" when the issue has
    # no month. Deliberately not normalised into a date type: the reader is judging
    # whether the paper is current, and the record's own precision is the honest one.
    pub_date: str
    abstract: str
    url: str


class LiteratureResult(TypedDict):
    """A literature search. Not scoped to an admission, unlike every other result here.

    There is no `hadm_id`, and its absence is load-bearing rather than an omission:
    this tool is the only one that sends anything to a third party, so what it can
    send is bounded by what it is given. See `services/mcp/tools/literature.py`.
    """

    query: str
    returned: int
    articles: list[LiteratureArticle]
    # Present when a search legitimately matched nothing, so an empty result can say
    # why instead of looking like a failure. Declared for the same reason as `note`
    # above: a key the schema does not declare is dropped from the payload the client
    # receives.
    note: NotRequired[str]
    # True when the search could not be performed at all — a rate limit, a timeout, a
    # network failure — as opposed to being performed and matching nothing. The
    # distinction is the whole point: an empty list alone cannot say whether anything
    # was ever looked for, and reporting an unexamined thing as absent invents a fact.
    degraded: NotRequired[bool]


class ToolContractError(ValueError):
    """An MCP tool returned a payload outside its declared contract."""


def tool_error(
    code: str,
    message: str,
    *,
    hadm_id: int | None = None,
    feature_source: str | None = None,
) -> ToolError:
    """Create the common structured error payload used by MCP tools.

    `hadm_id` is keyword-only and optional because not every tool is scoped to an
    admission. This signature used to lead with it, which wrote "every tool belongs
    to a patient" into the shared contract — true while the only tools were
    prediction and retrieval, and false now that one searches the literature. A tool
    with no admission omits the field rather than inventing one to satisfy a shape.

    The absence is also the privacy boundary. A tool that was never handed an
    admission cannot look the patient up, so the only thing it is able to put on the
    wire is the query it received. See docs/literature-tool-plan.md.
    """
    payload: ToolError = {"error": code, "message": message}
    if hadm_id is not None:
        payload["hadm_id"] = hadm_id
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
        if "granularity" in passage and not isinstance(passage["granularity"], str):
            raise ToolContractError("Retrieval passage granularity is invalid.")
        if passage.get("retrieval") == "deterministic":
            continue
        if not _number(passage.get("score")):
            raise ToolContractError("Retrieval passage is malformed.")
    return payload


def validate_literature_result(payload: Any) -> LiteratureResult | ToolError:
    if not isinstance(payload, dict):
        raise ToolContractError("Literature payload is not an object.")
    if "error" in payload:
        return _validate_error(payload)
    required = ("query", "returned", "articles")
    if any(field not in payload for field in required):
        raise ToolContractError("Literature payload is missing a required field.")
    if not isinstance(payload["query"], str) or not isinstance(payload["returned"], int):
        raise ToolContractError("Literature metadata is invalid.")
    articles = payload["articles"]
    if not isinstance(articles, list) or payload["returned"] != len(articles):
        raise ToolContractError("Literature articles are inconsistent.")
    for article in articles:
        if not isinstance(article, dict) or any(
            not isinstance(article.get(field), str)
            for field in ("pmid", "title", "journal", "pub_date", "abstract", "url")
        ):
            raise ToolContractError("Literature article is malformed.")
        # A record with no identifier cannot be cited or linked, which is the whole
        # use of it, so it is rejected here rather than rendered as a blank badge.
        if not article["pmid"]:
            raise ToolContractError("Literature article has no pmid.")
    if "degraded" in payload and not isinstance(payload["degraded"], bool):
        raise ToolContractError("Literature degraded flag is invalid.")
    return payload


# Which validator applies to which tool. Keyed by the name the server advertises, because
# that is all a client has when a result arrives — the payload does not say what it is meant
# to be. Both sides of the boundary import this, so a tool and its contract cannot ship apart.
TOOL_CONTRACTS = {
    "predict_readmission": validate_prediction_result,
    "rag_search": validate_retrieval_result,
    "rag_search_sections": validate_retrieval_result,
    "search_literature": validate_literature_result,
}


def validate_tool_result(name: str, payload: Any) -> dict[str, Any]:
    """Validate one tool's payload against that tool's contract.

    Raises `ToolContractError` for a payload outside its contract. A tool with no contract
    registered here is returned unchanged: the only way that happens is a partial deploy,
    and failing every call to that tool would turn a missing declaration into an outage.
    """
    validator = TOOL_CONTRACTS.get(name)
    if validator is None:
        return payload
    return validator(payload)