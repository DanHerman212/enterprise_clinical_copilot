"""Tests for the A2UI adapter.

These matter more than their size suggests. Every mistake this module can make
fails *silently in a browser*: a v0.8 component shape, a wrong catalog id, or a
dangling child reference all produce a console warning and an empty box, not an
exception. Catching those here turns an invisible rendering bug into a red test.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.a2ui import (  # noqa: E402
    AUDIENCE,
    BASIC_CATALOG_ID,
    SURFACE_ID,
    build_risk_card,
    risk_card_from_tool_calls,
)

# The pinned fixture from §14 — the same admission used everywhere else in the
# suite, so a change in its numbers shows up as one obvious diff.
GOOD_PAYLOAD = {
    "hadm_id": 20924467,
    "probability": 0.131398,
    "threshold": 0.12,
    "decision": 1,
    "base_value": -1.33862,
    "top_factors": [
        {"feature": "n_prior_admissions", "contribution": 0.4213, "direction": "increases"},
        {"feature": "age", "contribution": -0.1021, "direction": "decreases"},
    ],
    "model_version": "readmission-final-20260723172647",
    "feature_source": "bigquery",
}


def _by_id(card):
    components = card["messages"][1]["updateComponents"]["components"]
    return {c["id"]: c for c in components}


def _data(card):
    return card["messages"][2]["updateDataModel"]["value"]


class TestMessageEnvelope:
    def test_message_order_is_create_update_data(self):
        """The renderer applies messages in order; data before components is a no-op."""
        card = build_risk_card(GOOD_PAYLOAD)
        keys = [next(k for k in m if k != "version") for m in card["messages"]]
        assert keys == ["createSurface", "updateComponents", "updateDataModel"]

    def test_every_message_declares_v0_9(self):
        card = build_risk_card(GOOD_PAYLOAD)
        assert all(m["version"] == "v0.9" for m in card["messages"])

    def test_catalog_id_matches_the_shipped_bundle(self):
        """A mismatch here resolves no components and raises nothing."""
        create = build_risk_card(GOOD_PAYLOAD)["messages"][0]["createSurface"]
        assert create["catalogId"] == BASIC_CATALOG_ID
        assert create["surfaceId"] == SURFACE_ID

    def test_create_surface_has_no_root_property(self):
        """v0.9 dropped `root`; the component with id 'root' is the root."""
        create = build_risk_card(GOOD_PAYLOAD)["messages"][0]["createSurface"]
        assert "root" not in create
        assert "root" in _by_id(build_risk_card(GOOD_PAYLOAD))


class TestComponentShape:
    def test_component_is_a_string_not_a_nested_object(self):
        """The v0.8 shape `{component: {Text: {...}}}` renders an empty box."""
        for component in _by_id(build_risk_card(GOOD_PAYLOAD)).values():
            assert isinstance(component["component"], str), component

    def test_every_child_reference_resolves(self):
        """A dangling id renders nothing, silently."""
        components = _by_id(build_risk_card(GOOD_PAYLOAD))
        for component in components.values():
            for ref in [component.get("child")] + list(component.get("children") or []):
                if ref is not None:
                    assert ref in components, f"dangling reference {ref!r}"

    def test_every_bound_path_exists_in_the_data_model(self):
        """A path with no value renders blank rather than erroring."""
        card = build_risk_card(GOOD_PAYLOAD)
        data = _data(card)
        for component in _by_id(card).values():
            text = component.get("text")
            if isinstance(text, dict):
                key = text["path"].lstrip("/")
                assert key in data, f"unbound path {text['path']!r}"

    def test_no_clinical_values_are_baked_into_components(self):
        """Values live in the data model, so the component tree stays a template."""
        blob = repr(_by_id(build_risk_card(GOOD_PAYLOAD)))
        assert "0.131" not in blob
        assert "20924467" not in blob


class TestContent:
    def test_probability_is_shown_as_percent_and_raw(self):
        data = _data(build_risk_card(GOOD_PAYLOAD))
        assert "13.1%" in data["probability"]
        assert "0.1314" in data["probability"]

    def test_flagged_decision_states_the_threshold(self):
        data = _data(build_risk_card(GOOD_PAYLOAD))
        assert "Flagged" in data["decision"]
        assert "12%" in data["decision"]

    def test_unflagged_decision_says_not_flagged(self):
        payload = {**GOOD_PAYLOAD, "decision": 0, "probability": 0.0447}
        data = _data(build_risk_card(payload))
        assert "Not flagged" in data["decision"]

    def test_factor_direction_is_words_not_a_signed_number_alone(self):
        data = _data(build_risk_card(GOOD_PAYLOAD))
        assert "increases risk" in data["factors"]
        assert "decreases risk" in data["factors"]

    def test_factors_are_one_markdown_list_not_one_per_component(self):
        """Each Text is parsed as its own Markdown document.

        A component per factor yields several single-item lists instead of one
        list, which reads correctly on screen and wrongly to a screen reader.
        """
        card = build_risk_card(GOOD_PAYLOAD)
        rows = [c for c in _by_id(card) if c.startswith("factor") and c != "factors-title"]
        assert rows == ["factors"]
        assert _data(card)["factors"].count("\n- ") == len(GOOD_PAYLOAD["top_factors"]) - 1

    def test_underscores_in_feature_names_are_escaped(self):
        """Text is Markdown; an unescaped name relies on a CommonMark quirk."""
        data = _data(build_risk_card(GOOD_PAYLOAD))
        assert "n\\_prior\\_admissions" in data["factors"]

    def test_provenance_declares_the_name_is_synthetic(self):
        data = _data(build_risk_card(GOOD_PAYLOAD))
        assert "synthetic" in data["provenance"]

    def test_payload_without_factors_still_renders(self):
        payload = {**GOOD_PAYLOAD, "top_factors": []}
        card = build_risk_card(payload)
        components = _by_id(card)
        assert "factors-title" not in components
        assert "probability" in components


class TestFallbackAndAudience:
    def test_fallback_text_is_always_present(self):
        """R8 — if the renderer never boots the user still gets the answer."""
        assert build_risk_card(GOOD_PAYLOAD)["fallback_text"].strip()

    def test_fallback_carries_the_number_and_the_decision(self):
        fallback = build_risk_card(GOOD_PAYLOAD)["fallback_text"]
        assert "13.1%" in fallback
        assert "flagged" in fallback.lower()

    def test_audience_hides_the_payload_from_the_model(self):
        """R7 — stops the model re-reading its own UI as evidence."""
        assert build_risk_card(GOOD_PAYLOAD)["audience"] == AUDIENCE == ["user"]


class TestErrorPaths:
    def test_tool_error_renders_an_error_card_not_a_number(self):
        payload = {
            "hadm_id": -1,
            "error": "unknown_patient",
            "message": "No admission -1 in the feature source (bigquery).",
            "feature_source": "bigquery",
        }
        card = build_risk_card(payload)
        data = _data(card)
        assert "unavailable" in data["title"].lower()
        assert "unknown_patient" in data["detail"]
        assert "unknown_patient" in card["fallback_text"]

    def test_missing_required_keys_do_not_render_a_confident_card(self):
        """A partial payload must not produce something that looks like a result."""
        card = build_risk_card({"hadm_id": 1, "probability": 0.4})
        assert "unavailable" in _data(card)["title"].lower()

    def test_non_dict_payload_is_handled(self):
        assert "unavailable" in _data(build_risk_card(None)).get("title", "").lower()

    @pytest.mark.parametrize("payload", [{}, {"error": ""}])
    def test_empty_payloads_never_raise(self, payload):
        assert build_risk_card(payload)["fallback_text"]


class TestToolCallExtraction:
    def test_picks_the_prediction_out_of_a_run(self):
        calls = [{"name": "predict_readmission", "args": {}, "response": GOOD_PAYLOAD}]
        assert "13.1%" in risk_card_from_tool_calls(calls)["fallback_text"]

    def test_uses_the_last_prediction_when_several_ran(self):
        other = {**GOOD_PAYLOAD, "hadm_id": 111, "probability": 0.9, "decision": 1}
        calls = [
            {"name": "predict_readmission", "response": GOOD_PAYLOAD},
            {"name": "predict_readmission", "response": other},
        ]
        assert "111" in risk_card_from_tool_calls(calls)["fallback_text"]

    def test_returns_none_when_no_prediction_ran(self):
        """Answering without a tool call is legitimate; an empty card is not."""
        assert risk_card_from_tool_calls([]) is None
        assert risk_card_from_tool_calls([{"name": "other", "response": {}}]) is None
