import json

import pytest

from services.mcp.contracts import (
    ToolContractError,
    tool_error,
    validate_prediction_result,
    validate_retrieval_result,
)


def test_every_tool_declares_its_output_schema():
    """Gap 1: a typed schema on the way out, not only on the way in.

    The SDK derives the advertised output schema from the return annotation, so a tool
    annotated `-> dict[str, Any]` advertises an object with no properties while a tool
    annotated with its contract advertises the shape. Both shapes are asserted here, because
    a contract that omits a key the tool emits is worse than no contract: the SDK drops that
    key from the structured payload.
    """
    import asyncio

    from services.mcp.server import server

    tools = asyncio.run(server.list_tools())
    assert tools, "the server advertises no tools at all"

    for tool in tools:
        properties = (tool.output_schema or {}).get("properties") or {}
        assert properties, f"{tool.name} advertises no output shape at all"
        # A union return type is declared under a single `result` property.
        assert "result" in properties, f"{tool.name} does not declare a result"

    # The keys the tools actually emit, including the two the contracts were missing.
    declared = json.dumps([tool.output_schema for tool in tools])
    for key in ("granularity", "note", "top_factors", "passages", "model_version"):
        assert f'"{key}"' in declared, f"{key} is emitted but not declared anywhere"


def test_tool_error_has_common_shape():
    assert tool_error(90000009, "unknown_patient", "Not found") == {
        "hadm_id": 90000009,
        "error": "unknown_patient",
        "message": "Not found",
    }


def test_prediction_contract_accepts_result():
    payload = {
        "hadm_id": 90000009,
        "probability": 0.2,
        "threshold": 0.11,
        "decision": 1,
        "base_value": 0.05,
        "top_factors": [{
            "feature": "prior_inpatient_days",
            "contribution": 0.2,
            "direction": "increases",
        }],
        "model_version": "model-x",
        "feature_source": "synthetic",
    }

    assert validate_prediction_result(payload) is payload


def test_prediction_contract_accepts_structured_error():
    payload = tool_error(90000009, "unknown_patient", "Not found")

    assert validate_prediction_result(payload) is payload


def test_prediction_contract_rejects_missing_factor_fields():
    payload = {
        "hadm_id": 90000009,
        "probability": 0.2,
        "threshold": 0.11,
        "decision": 1,
        "base_value": 0.05,
        "top_factors": [{"feature": "age"}],
        "model_version": "model-x",
        "feature_source": "synthetic",
    }

    with pytest.raises(ToolContractError):
        validate_prediction_result(payload)


def test_retrieval_contract_checks_returned_count():
    payload = {
        "hadm_id": 90000009,
        "query": "medications",
        "returned": 1,
        "passages": [{
            "id": "note-1",
            "section": "discharge_medications",
            "text": "Aspirin",
            "score": 0.2,
        }],
    }

    assert validate_retrieval_result(payload) is payload


def test_retrieval_contract_rejects_inconsistent_count():
    payload = {
        "hadm_id": 90000009,
        "query": "medications",
        "returned": 2,
        "passages": [],
    }

    with pytest.raises(ToolContractError):
        validate_retrieval_result(payload)