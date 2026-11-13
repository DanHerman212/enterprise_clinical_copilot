"""Sprint 04b — the pin is `gemini-3.8-flash` and `gemini-3.1-flash-lite` is retired.

Five behaviours, each of which was a live failure or a decision before this sprint:

* the default serves the baseline, not the model it replaced;
* a retired id stops the deployment instead of serving traffic;
* binding a client does not change which model's facts are applied to it — the Sprint 04a
  defect, which sent `THINKING_LEVEL_MINIMAL` to a model that rejects it;
* the extraction request is stated semantically and lands on the level this model accepts;
* the output ceiling is the client's, everywhere, and it is 16,384.
"""

from __future__ import annotations

import pytest
from langchain_core.tools import tool

from services.agent.engine import deterministic
from services.agent.engine import gateway
from services.agent.engine import model as model_turn
from services.mcp import config

STRUCTURED = {"thinking_level": "minimal", "response_mime_type": "application/json"}


def test_the_default_profile_is_the_baseline_and_not_the_retired_pin():
    assert config.GEMINI_MODEL == "gemini-3.8-flash"
    assert config.AGENT_MODEL == "gemini-3.8-flash"
    assert gateway.DEFAULT_PROFILE_NAME == "gemini-3.8-flash"
    assert gateway.profile_for(object()).name == "gemini-3.8-flash"


def test_the_retired_pin_is_gone_from_the_registry():
    assert "gemini-3.1-flash-lite" not in gateway.PROFILES
    assert set(gateway.PROFILES) == {
        "gemini-3.8-flash", "gemini-3.1-pro", "claude-sonnet-5-5", "muse-spark-1.3",
    }


def test_asking_for_the_retired_pin_raises_before_any_request_is_made():
    with pytest.raises(gateway.RetiredModel) as raised:
        gateway.chat_model("gemini-3.1-flash-lite")
    assert "gemini-3.1-flash-lite" in str(raised.value)
    assert "retired on 2026-11-13" in str(raised.value)


@tool
def noop(query: str) -> str:
    """A tool with one argument, so a binding has something to bind."""
    return query


def test_a_binding_keeps_the_profile_of_the_client_it_came_from():
    """The Sprint 04a defect: a bound runnable resolved to the default profile."""
    client = gateway.chat_model("gemini-3.8-flash")
    for derived in (client.bind(max_output_tokens=16), client.bind_tools([noop])):
        resolved = gateway.profile_for(derived)
        assert resolved.name == "gemini-3.8-flash"
        assert gateway.wire_bindings(resolved, STRUCTURED) == {
            "thinking_level": "low",
            "response_mime_type": "application/json",
        }


def test_the_extraction_request_is_semantic_and_lands_on_the_accepted_level():
    assert "thinking_level" not in deterministic._STRUCTURED_CALL_PARAMS
    assert deterministic._STRUCTURED_CALL_PARAMS == {
        "effort": gateway.EFFORT_EXTRACTION, "json_mode": True,
    }
    for name in ("gemini-3.8-flash", "gemini-3.1-pro"):
        assert gateway.wire_bindings(gateway.profile(name),
                                     deterministic._STRUCTURED_CALL_PARAMS) == {
            "thinking_level": "low",
            "response_mime_type": "application/json",
        }


def test_a_level_the_table_cannot_name_is_refused_rather_than_sent():
    with pytest.raises(gateway.UnsupportedLevel):
        gateway.wire_bindings(gateway.profile("gemini-3.8-flash"),
                              {"thinking_level": "off"})


def test_the_output_ceiling_is_the_clients_and_it_is_16384():
    assert config.GEMINI_MAX_OUTPUT_TOKENS == 16384
    assert deterministic.COMPARATOR_REPLY_TOKENS == config.GEMINI_MAX_OUTPUT_TOKENS
    client = gateway.chat_model("gemini-3.8-flash")
    assert client.max_output_tokens == 16384


def test_the_record_names_the_pin_and_the_model_it_replaced():
    record = config.MODEL_CHOICE
    assert record["model"] == "gemini-3.8-flash"
    assert record["previous"] == "gemini-3.1-flash-lite"
    assert record["previous_retired"] == "2027-05-07"
    assert record["retires"] == "2027-09-02"
    assert record["decided"] == "2026-11-13"
