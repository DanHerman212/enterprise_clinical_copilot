"""Unit tests for deterministic `/ehr` Readmission risk & trajectory rendering,
single-pass citation settlement, and MCP `/health` code_revision parity."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from langchain_core.messages import AIMessage
from starlette.testclient import TestClient

from services.agent.engine import deterministic
from services.agent.engine import graph
from services.agent.engine import prompts
from services.agent.postprocess import answer as answer_text
from services.agent.postprocess import guardrail
from services.agent.presentation.a2ui import compose_presentation


def _prediction(prob: float = 0.09, thresh: float = 0.11, *, with_null_driver: bool = False):
    factors = [
        {"feature": "admission_type", "direction": "increases", "value": "Emergency, direct admission"},
        {"feature": "rdw_max", "direction": "increases", "value": 15.6},
        {"feature": "monocytes_min", "direction": "decreases", "value": 0.2},
        {"feature": "prior_inpatient_days", "direction": "decreases", "value": 4.75},
        {"feature": "recent_ed_visits", "direction": "increases", "value": 0},
        {"feature": "oncology_flag", "direction": "increases", "value": 1},
    ]
    if with_null_driver:
        factors.append({"feature": "sodium_min", "direction": "decreases", "value": None})
    return {
        "hadm_id": 90000015,
        "probability": prob,
        "threshold": thresh,
        "probability_display": f"{prob * 100:.1f}%",
        "threshold_display": f"{thresh * 100:.1f}%",
        "top_factors": factors,
    }


def test_render_readmission_section_formats_drivers_verbatim_without_editorializing():
    raw = _prediction(0.09, 0.11, with_null_driver=True)
    from services.agent.engine.mcp_client import _presented_prediction

    projected = _presented_prediction("predict_readmission", raw, {})
    rendered = answer_text.render_readmission_section(
        projected,
        raw_probability=raw["probability"],
        raw_threshold=raw["threshold"],
        raw_factors=raw["top_factors"],
    )

    assert rendered.startswith("**Readmission risk & trajectory.**")
    assert "9.0%, which falls below the decision threshold of 11.0%." in rendered
    assert "4.75" in rendered
    assert "recent ED visits (0)" in rendered
    assert "oncology history" in rendered
    assert "oncology history (no recorded value on the scored row)" not in rendered
    assert "lowest sodium (no recorded value on the scored row)" in rendered
    for banned in ("favorable", "protective", "paradoxical", "reassuring", "contribution"):
        assert banned not in rendered.lower()


def test_render_readmission_section_formats_single_two_arm_and_range_comparators():
    raw = _prediction(0.09, 0.11)
    from services.agent.engine.mcp_client import _presented_prediction

    projected = _presented_prediction("predict_readmission", raw, {})

    single = answer_text.render_readmission_section(
        projected,
        [{"pmid": "39165878", "reported_rate": "1.1%", "rate_kind": "point",
          "cohort_summary": "patients undergoing total knee arthroplasty"}],
        raw_probability=0.09, raw_threshold=0.11,
    )
    assert "higher than the 1.1% ^[PMID: 39165878] reported for patients undergoing total knee arthroplasty" in single
    assert "population benchmarks (rather than individual predictions)" in single

    two_arm = answer_text.render_readmission_section(
        projected,
        [{"pmid": "40000001", "reported_rate": "2.6% vs. 12.0%", "rate_kind": "comparison",
          "cohort_summary": "intervention vs. control"}],
        raw_probability=0.09, raw_threshold=0.11,
    )
    assert "between the two published rates (2.6% vs. 12.0%) ^[PMID: 40000001]" in two_arm

    rng = answer_text.render_readmission_section(
        projected,
        [{"pmid": "40000002", "reported_rate": "1.0% to 1.4%", "rate_kind": "range",
          "cohort_summary": "elective joint cohort"}],
        raw_probability=0.09, raw_threshold=0.11,
    )
    assert "above the published range of 1.0% to 1.4% ^[PMID: 40000002]" in rng


def test_render_readmission_section_fallback_literature_prepends_notice():
    raw = _prediction(0.176, 0.11)
    from services.agent.engine.mcp_client import _presented_prediction

    projected = _presented_prediction("predict_readmission", raw, {})
    rendered = answer_text.render_readmission_section(
        projected,
        comparators=[],
        raw_probability=0.176,
        raw_threshold=0.11,
        fallback_literature_prose="Pancreatic malignancy predicts higher post-discharge utilization ^[PMID: 12345678].",
    )
    assert "17.6%, which falls above the decision threshold of 11.0%." in rendered
    assert "No directly comparable 30-day cohort benchmark was verified for this admission." in rendered
    assert "^[PMID: 12345678]" in rendered


def test_ehr_run_command_attaches_deterministic_risk_section_and_announces_stream():
    class _FakeToolbox:
        async def call(self, name, args):
            if name == "rag_search_sections":
                return {
                    "hadm_id": 90000015,
                    "returned": 2,
                    "passages": [
                        {"id": "n1", "section": "history_of_present_illness",
                         "text": "61-year-old female admitted after right total knee replacement."},
                        {"id": "n2", "section": "discharge_diagnosis",
                         "text": "Status post right total knee replacement."},
                    ],
                }
            if name == "predict_readmission":
                return _prediction(0.09, 0.11)
            return {}

    from langchain_core.messages import AIMessageChunk

    class _FakeLLM:
        async def ainvoke(self, messages, **kwargs):
            return AIMessage(
                content=(
                    "**Presentation & diagnosis.** Admitted after right total knee replacement ^[1]. "
                    "Discharged with status post right total knee replacement ^[2]."
                )
            )

        async def astream(self, messages, **kwargs):
            yield AIMessageChunk(
                content=(
                    "**Presentation & diagnosis.** Admitted after right total knee replacement ^[1]. "
                    "Discharged with status post right total knee replacement ^[2]."
                )
            )

    async def _no_comparators(*args, **kwargs):
        return (
            [{"pmid": "39165878", "journal": "World J Orthop", "year": "2024",
              "reported_rate": "1.1%", "rate_kind": "point",
              "cohort_summary": "patients undergoing total knee arthroplasty",
              "evidence_tier": "cohort study"}],
            None,
            [{"name": "search_literature", "args": {"query": "TKA"}, "response": {
                "articles": [{"pmid": "39165878", "title": "TKA study",
                              "journal": "World J Orthop", "pub_date": "2024",
                              "authors": "Jethi T", "abstract": "1.1%"}]
            }}],
            [],
            "verified",
            [],
        )

    events: list[dict] = []
    from services.agent.engine import model as model_gateway
    from services.agent.routing import routing

    cmd = routing.command_for("summarize")
    with patch.object(deterministic, "_readmission_comparators", _no_comparators), \
         patch.object(model_gateway, "build_llm", lambda *a, **kw: _FakeLLM()):
        state = asyncio.run(deterministic.run_command(
            _FakeToolbox(), cmd, 90000015,
            question="/ehr", model="test",
            on_event=events.append, prose_stream=True,
        ))

    text = graph.final_text(state)
    assert "**Readmission risk & trajectory.**" in text
    assert "9.0%, which falls below the decision threshold of 11.0%." in text
    assert "^[PMID: 39165878]" in text
    assert any("**Readmission risk & trajectory.**" in ev.get("text", "") for ev in events)

    graph._settle(state, prose=text)
    assert state.get("citations_settled") is True
    guarded = guardrail.guard_answer(
        graph.final_text(state), state["tool_calls"], state=state,
    )
    assert "9.0%" in guarded["answer"]
    assert not [f for f in guarded["flags"] if "citation" in f]

    pres = compose_presentation(
        "/ehr", guarded["answer"], state["tool_calls"],
        command="summarize",
        evidence=state["tool_calls"],
        evidence_state=state.get("evidence_state"),
        citation_map=state.get("citation_display_map") or {},
        comparators=state.get("comparators") or [],
    )
    assert any(ref.get("pmid") == "39165878" for ref in pres["references"])


def test_guard_answer_skips_duplicate_citation_enforcement_when_already_settled():
    """Live turns already settled in `graph._settle` do not re-run `enforce_citations`."""
    prose = "Claim with an unresolved marker ^[9]."
    standalone = guardrail.guard_answer(prose, [], state={})
    assert "^[9]" not in standalone["answer"]
    assert any("citation" in flag for flag in standalone["flags"])

    already_settled = guardrail.guard_answer(prose, [], state={"citations_settled": True})
    assert "^[9]" in already_settled["answer"]
    assert not any("citation_out_of_range" in flag for flag in already_settled["flags"])


def test_mcp_health_includes_code_revision():
    from services.mcp import server as mcp_server

    client = TestClient(mcp_server.server.streamable_http_app(stateless_http=True))
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "code_revision" in body and body["code_revision"]


def test_ehr_prompt_diet_and_expanded_sections():
    text = prompts.EHR_FORMATTING
    assert len(text.split()) < 800
    for sec in ("past_medical_history", "past_surgical_history", "allergies",
                "physical_exam", "pertinent_results"):
        assert sec in text
