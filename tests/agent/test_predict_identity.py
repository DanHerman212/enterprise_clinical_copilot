"""Gap 2 — the payload names the model that actually answered.

The tool used to fill ``model_version`` from the newest ``readmission-final-*``
registry record while the endpoint served whatever was deployed last: two
independent lookups and nothing comparing them. These tests pin the replacement
— the endpoint's own report of what ran — and the reason the vector is sent by
name rather than position, which is what turns a feature-order change into a
refusal instead of a wrong score.
"""

from unittest.mock import patch

import services.mcp.tools.prediction as pr
from services.mcp.contracts import validate_prediction_result

_DEPLOYED_ID = "projects/778397675435/locations/us-east1/models/42"
_FINAL_RECORD = "readmission-final-20260902014308"


class _Source:
    def fetch(self, hadm_id):
        return {"f1": 1.0}


def _reply(**overrides):
    payload = {
        "probability": 0.2,
        "threshold": 0.11,
        "prediction": 0,
        "base_value": 0.0,
        "top_factors": [],
    }
    payload.update(overrides)
    return payload


def test_the_payload_names_the_model_the_endpoint_reported():
    seen = {}

    def fake_predict_one(instance):
        seen["instance"] = instance
        return _reply(model_version=_FINAL_RECORD), _DEPLOYED_ID

    with patch.object(pr, "feature_order", lambda: ["f1"]), \
         patch.object(pr, "_source", lambda: _Source()), \
         patch.object(pr, "predict_one", fake_predict_one):
        result = pr._predict(90000009)

    assert result["model_version"] == _FINAL_RECORD
    # The endpoint is asked in names, not positionally: it validates a name it
    # does not know and refuses; an equal-width vector would just be scored.
    assert seen["instance"] == {"f1": 1.0}
    # And the payload still satisfies the tool contract the model is given.
    validate_prediction_result(result)


def test_an_unstamped_endpoint_still_identifies_itself():
    """No stamp from the deployment: the deployed model's own id is the truth."""

    def fake_predict_one(instance):
        return _reply(), _DEPLOYED_ID

    with patch.object(pr, "feature_order", lambda: ["f1"]), \
         patch.object(pr, "_source", lambda: _Source()), \
         patch.object(pr, "predict_one", fake_predict_one):
        result = pr._predict(90000009)

    assert result["model_version"] == _DEPLOYED_ID
    validate_prediction_result(result)


def test_nothing_resolves_identity_from_the_registry_any_more():
    """The old lookup is gone, not merely unused.

    ``model_version`` came from ``...features.manifest.model_version()``, which
    named the newest registered record. If a change reintroduces that, this
    fails rather than letting a plausible-but-unrelated name back in.
    """
    import services.mcp.dependencies.features.manifest as manifest_module

    assert not hasattr(manifest_module, "model_version")
