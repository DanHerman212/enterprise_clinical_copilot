import asyncio
from types import SimpleNamespace

from services.agent.mcp_client import MCPToolbox


class FakeSession:
    def __init__(self, payload, *, is_error=False):
        self.payload = payload
        self.is_error = is_error

    async def list_tools(self):
        return SimpleNamespace(tools=[
            SimpleNamespace(
                name="predict_readmission",
                description="Predict readmission risk",
                input_schema={"type": "object"},
            ),
            SimpleNamespace(
                name="rag_search",
                description="Search discharge notes",
                input_schema={"type": "object"},
            ),
            SimpleNamespace(
                name="rag_search_sections",
                description="Retrieve note sections",
                input_schema={"type": "object"},
            ),
        ])

    async def call_tool(self, name, arguments, read_timeout_seconds):
        return SimpleNamespace(
            structured_content=self.payload,
            content=[],
            is_error=self.is_error,
        )


def test_mcp_client_loads_the_three_registered_tool_names():
    toolbox = MCPToolbox(session=FakeSession({}))

    asyncio.run(toolbox.load())

    assert toolbox.names == [
        "predict_readmission",
        "rag_search",
        "rag_search_sections",
    ]


def test_mcp_client_preserves_structured_tool_errors_as_data():
    toolbox = MCPToolbox(session=FakeSession({
        "hadm_id": 90000009,
        "error": "unknown_patient",
        "message": "No admission found.",
    }))
    asyncio.run(toolbox.load())

    result = asyncio.run(toolbox.call("predict_readmission", {"hadm_id": 90000009}))

    assert result == {
        "hadm_id": 90000009,
        "error": "unknown_patient",
        "message": "No admission found.",
    }


def test_mcp_client_converts_transport_errors_to_structured_data():
    class BrokenSession(FakeSession):
        async def call_tool(self, name, arguments, read_timeout_seconds):
            raise RuntimeError("private transport detail")

    toolbox = MCPToolbox(session=BrokenSession({}))
    asyncio.run(toolbox.load())

    result = asyncio.run(toolbox.call("rag_search", {"hadm_id": 90000009}))

    assert result["error"] == "tool_call_failed"
    assert "RuntimeError" in result["message"]


# --- what the boundary does with the result (gap 1) --------------------------

VALID_PREDICTION = {
    "hadm_id": 90000009,
    "probability": 0.2,
    "threshold": 0.11,
    "decision": 1,
    "base_value": 0.05,
    "top_factors": [
        {"feature": "prior_inpatient_days", "contribution": 0.2, "direction": "increases"}
    ],
    "model_version": "model-x",
    "feature_source": "synthetic",
}


def _call(name, payload, *, tool="predict_readmission"):
    toolbox = MCPToolbox(session=FakeSession(payload))
    asyncio.run(toolbox.load())
    return asyncio.run(toolbox.call(tool, {"hadm_id": 90000009}))


def test_a_valid_payload_passes_through_unchanged():
    # The check is not allowed to rewrite good data on its way past.
    assert _call("predict_readmission", VALID_PREDICTION) == VALID_PREDICTION


def test_the_union_envelope_is_unwrapped_before_the_payload_is_used():
    # A union return type is advertised — and sent — as `{"result": …}`.
    assert _call("predict_readmission", {"result": VALID_PREDICTION}) == VALID_PREDICTION


def test_a_payload_outside_its_contract_does_not_reach_the_model():
    broken = {"result": {"hadm_id": 90000009, "probability": "very likely"}}

    result = _call("predict_readmission", broken)

    assert result["error"] == "invalid_tool_response"
    assert "probability" not in result


def test_a_text_only_result_is_not_mistaken_for_an_envelope():
    # The fallback shape is `{"result": "<text>"}`; unwrapping that would hand the caller a
    # string where a payload is expected.
    class TextSession(FakeSession):
        async def call_tool(self, name, arguments, read_timeout_seconds):
            return SimpleNamespace(
                structured_content=None,
                content=[SimpleNamespace(text="not json at all")],
                is_error=False,
            )

    toolbox = MCPToolbox(session=TextSession({}))
    asyncio.run(toolbox.load())

    result = asyncio.run(toolbox.call("rag_search", {"hadm_id": 90000009}))

    assert isinstance(result, dict)
    assert result["error"] == "invalid_tool_response"