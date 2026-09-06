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