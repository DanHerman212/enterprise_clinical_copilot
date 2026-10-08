import json

import pytest

from services.mcp.contracts import (
    ToolContractError,
    tool_error,
    validate_literature_result,
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
    for key in (
        "granularity", "note", "top_factors", "passages", "model_version",
        # Literature, added 2026-10-08. Listed for the same reason as the rest: the
        # SDK drops an undeclared key, so a contract missing one of these ships a
        # payload with the field silently absent and nothing failing anywhere.
        "articles", "pmid", "title", "journal", "pub_date", "abstract", "url",
    ):
        assert f'"{key}"' in declared, f"{key} is emitted but not declared anywhere"


def test_tool_error_has_common_shape():
    assert tool_error("unknown_patient", "Not found", hadm_id=90000009) == {
        "hadm_id": 90000009,
        "error": "unknown_patient",
        "message": "Not found",
    }


def test_tool_error_omits_an_absent_admission():
    """A tool with no patient must not be handed one to satisfy a signature.

    The literature tool is the case this exists for. Asserted because `hadm_id` led
    the signature positionally until 2026-10-08, and the cheapest way back to a green
    suite would have been to pass a placeholder admission rather than remove it.
    """
    assert tool_error("literature_failed", "PubMed could not be reached.") == {
        "error": "literature_failed",
        "message": "PubMed could not be reached.",
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
    payload = tool_error("unknown_patient", "Not found", hadm_id=90000009)

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


def test_the_advertised_schema_declares_the_one_bound():
    """Gap 2: the model is told the range instead of discovering it by failing.

    Asserted against the cleaned schema, because that is what the client hands the model —
    a bound that does not survive `_clean_schema` does not reach anyone. The default does not
    survive it (`default` is stripped, as Gemini rejects the key), so the range is the only
    thing that can tell the model what a sensible call looks like.
    """
    import asyncio

    from services.agent.mcp_client import _clean_schema
    from services.mcp.server import server
    from services.mcp.tools.retrieval import TOP_K_MAX, TOP_K_MIN

    tools = asyncio.run(server.list_tools())
    declared = {
        tool.name: _clean_schema(tool.input_schema)["properties"]["top_k"]
        for tool in tools
        if "top_k" in (_clean_schema(tool.input_schema).get("properties") or {})
    }

    assert declared, "no advertised tool declares top_k at all"
    for name, prop in declared.items():
        assert prop.get("minimum") == TOP_K_MIN, f"{name} does not declare the lower bound"
        assert prop.get("maximum") == TOP_K_MAX, f"{name} does not declare the upper bound"


def test_retrieval_contract_rejects_inconsistent_count():
    payload = {
        "hadm_id": 90000009,
        "query": "medications",
        "returned": 2,
        "passages": [],
    }

    with pytest.raises(ToolContractError):
        validate_retrieval_result(payload)


def _literature_article(**overrides):
    article = {
        "pmid": "31234567",
        "title": "Ceftriaxone for lower extremity edema",
        "journal": "N Engl J Med",
        "pub_date": "2025 Mar 12",
        "abstract": "BACKGROUND: Cellulitis is common.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/31234567/",
    }
    article.update(overrides)
    return article


def test_literature_contract_accepts_result():
    payload = {
        "query": "ceftriaxone lower extremity edema",
        "returned": 1,
        "articles": [_literature_article()],
    }

    assert validate_literature_result(payload) is payload


def test_literature_contract_accepts_an_empty_result():
    """Empty is an answer, not an error.

    The site reads ANY `error` key in a tool response as an outage: it refunds the
    request and tells the clinician the service is unavailable. So a search that
    matched nothing has to arrive as a result with a reason attached, or a correctly
    answered question is reported as a failure.
    """
    payload = {
        "query": "a question with no matches",
        "returned": 0,
        "articles": [],
        "note": "PubMed found no articles for this search.",
    }

    assert validate_literature_result(payload) is payload


def test_literature_contract_rejects_an_inconsistent_count():
    payload = {
        "query": "ceftriaxone",
        "returned": 2,
        "articles": [_literature_article()],
    }

    with pytest.raises(ToolContractError):
        validate_literature_result(payload)


def test_literature_contract_rejects_an_article_with_no_pmid():
    """No identifier means no citation and no link, which is the entire use of it."""
    payload = {
        "query": "ceftriaxone",
        "returned": 1,
        "articles": [_literature_article(pmid="")],
    }

    with pytest.raises(ToolContractError):
        validate_literature_result(payload)