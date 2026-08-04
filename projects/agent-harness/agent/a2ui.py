"""Compose A2UI v0.9 messages from a `predict_readmission` payload.

This is the single adapter module §16e calls for. Every A2UI shape in the
project is built here, so the v1.0 migration (`theme` -> `surfaceProperties`,
and whatever else moves) is a one-file change rather than a search across the
agent, the server, and a template.

Two rules this module exists to enforce:

1. **Tools return JSON, the agent composes UI.** `predict.py` deliberately
   never emits A2UI — a tool that returns UI is welded to one presentation
   layer and stops working from Claude Desktop or CI. The translation happens
   here, on the agent side, where a presentation choice belongs.

2. **Never render without a text fallback (R8).** Every return carries
   `fallback_text`. If the CDN is down, the renderer throws, or the spec
   version drifts, the caller still has something true to show. A demo that
   renders nothing is worse than one that renders plainly.

The payload shape is v0.9, which is *not* backward compatible with the v0.8
examples that dominate search results. In v0.9 `component` is a string and
properties sit inline. Feeding v0.8 shapes to a v0.9 renderer raises nothing —
it logs "Component implementation not found for type: [object Object]" and
draws an empty box. The tests in tests/test_a2ui.py pin the v0.9 shape so that
regression is caught here rather than in a browser.
"""

from typing import Any

# Read off the shipped bundle rather than guessed: @a2ui/lit@0.10.2 constructs
# basicCatalog with this literal id. The renderer matches the surface's
# catalogId against it, and a mismatch means no components resolve.
BASIC_CATALOG_ID = "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"

A2UI_VERSION = "v0.9"
SURFACE_ID = "risk-card"

# R7: rendered payloads are for the user, not for the model to re-read on a
# later turn. Hiding them cuts tokens and, more importantly, stops the model
# treating its own UI output as evidence — a faithfulness risk under Tier 2.
AUDIENCE = ["user"]

_MD_SPECIALS = "\\`*_[]#"


def _md(text: str) -> str:
    """Escape Markdown control characters.

    A2UI text properties are Markdown — `variant: 'h2'` is implemented by
    prepending "## ", not by a heading element. That means feature names go
    through a Markdown parser on the way to the screen, and names like
    `n_prior_admissions` or `dx_*` contain characters the parser reacts to.
    CommonMark happens to leave intraword underscores alone, but that is a
    quirk we would be relying on rather than a guarantee.
    """
    out = []
    for ch in text:
        if ch in _MD_SPECIALS:
            out.append("\\")
        out.append(ch)
    return "".join(out)


def _text(component_id: str, path: str) -> dict[str, Any]:
    """A Text component bound to the data model rather than carrying a literal.

    Keeping clinical values out of `updateComponents` means the component tree
    is a fixed template and the payload-specific part is confined to
    `updateDataModel`.
    """
    return {"id": component_id, "component": "Text", "text": {"path": path}}


def _messages(components: list[dict[str, Any]], data: dict[str, str]) -> list[dict[str, Any]]:
    """Order matters: createSurface -> updateComponents -> updateDataModel.

    `createSurface` takes no `root` property in v0.9 — the component whose id
    is literally "root" is the tree root.
    """
    return [
        {
            "version": A2UI_VERSION,
            "createSurface": {"surfaceId": SURFACE_ID, "catalogId": BASIC_CATALOG_ID},
        },
        {
            "version": A2UI_VERSION,
            "updateComponents": {"surfaceId": SURFACE_ID, "components": components},
        },
        {
            "version": A2UI_VERSION,
            "updateDataModel": {"surfaceId": SURFACE_ID, "value": data},
        },
    ]


def _envelope(
    components: list[dict[str, Any]], data: dict[str, str], fallback: str
) -> dict[str, Any]:
    return {
        "surface_id": SURFACE_ID,
        "audience": AUDIENCE,
        "messages": _messages(components, data),
        "fallback_text": fallback,
    }


def _error_card(payload: dict[str, Any]) -> dict[str, Any]:
    """Render the tool's own error contract instead of inventing a number.

    `predict.py` returns structured errors precisely so the failure can be
    explained rather than guessed at. Showing "unknown patient" is honest;
    showing a blank card is not.
    """
    code = str(payload.get("error", "unknown_error"))
    message = str(payload.get("message", "The prediction could not be completed."))

    components = [
        {"id": "root", "component": "Card", "child": "card-body"},
        {"id": "card-body", "component": "Column", "children": ["title", "detail"]},
        _text("title", "/title"),
        _text("detail", "/detail"),
    ]
    data = {
        "title": "## Prediction unavailable",
        "detail": f"{_md(message)}\n\n`{code}`",
    }
    fallback = f"Prediction unavailable ({code}): {message}"
    return _envelope(components, data, fallback)


def build_risk_card(payload: dict[str, Any]) -> dict[str, Any]:
    """Turn one `predict_readmission` response into a renderable envelope.

    Returns a dict with `messages` (the A2UI v0.9 message list), `fallback_text`
    (R8), `audience` (R7) and `surface_id`. Pure — no I/O, no clock, no
    randomness — so it is fully testable without a browser or a live endpoint.
    """
    if not isinstance(payload, dict):
        return _error_card({"error": "invalid_payload", "message": "Tool returned no object."})

    if payload.get("error"):
        return _error_card(payload)

    # A payload missing these is not a risk card, whatever else it contains.
    required = ("hadm_id", "probability", "threshold", "decision")
    missing = [key for key in required if key not in payload]
    if missing:
        return _error_card(
            {
                "error": "malformed_payload",
                "message": f"Prediction payload is missing {', '.join(missing)}.",
            }
        )

    hadm_id = payload["hadm_id"]
    probability = float(payload["probability"])
    threshold = float(payload["threshold"])
    flagged = int(payload["decision"]) == 1
    factors = payload.get("top_factors") or []

    components: list[dict[str, Any]] = [
        {"id": "root", "component": "Card", "child": "card-body"},
        _text("title", "/title"),
        _text("probability", "/probability"),
        _text("decision", "/decision"),
    ]
    data: dict[str, str] = {
        "title": f"## Readmission risk — admission {hadm_id}",
        # Percent is what a clinician reads; the raw probability follows so the
        # number on screen can be traced back to the model output.
        "probability": f"**{probability:.1%}** 30-day readmission probability "
        f"({probability:.4f})",
        "decision": (
            f"{'**Flagged**' if flagged else 'Not flagged'} at a "
            f"{threshold:.0%} decision threshold"
        ),
    }

    children = ["title", "probability", "decision"]

    if factors:
        components.append(_text("factors-title", "/factors_title"))
        data["factors_title"] = "### What drove this"
        children.append("factors-title")

        # One Text component holding the whole Markdown list, not one per
        # factor. Each Text is parsed as an independent Markdown document, so
        # a component per factor produces three single-item <ul>s rather than
        # one three-item list — visually near-identical, and wrong to a screen
        # reader. Caught by the accessibility tree in the render check, not by
        # a unit test.
        lines = []
        for factor in factors:
            name = _md(str(factor.get("feature", "unknown")))
            direction = str(factor.get("direction", "")).strip()
            contribution = factor.get("contribution")

            # Direction is stated in words because a signed number in logit
            # space is not something a clinician should have to interpret.
            arrow = "increases risk" if direction == "increases" else "decreases risk"
            magnitude = f" ({float(contribution):+.4f})" if contribution is not None else ""
            lines.append(f"- {name} — {arrow}{magnitude}")

        components.append(_text("factors", "/factors"))
        data["factors"] = "\n".join(lines)
        children.append("factors")

    components.append(_text("provenance", "/provenance"))
    data["provenance"] = _md(
        f"Model {payload.get('model_version', 'unknown')} · "
        f"features from {payload.get('feature_source', 'unknown')} · "
        "synthetic name, real MIMIC-IV admission"
    )
    children.append("provenance")

    components.insert(1, {"id": "card-body", "component": "Column", "children": children})

    fallback_factors = (
        "".join(
            f"\n  - {f.get('feature')} {f.get('direction')} risk"
            for f in factors
        )
        if factors
        else ""
    )
    fallback = (
        f"Admission {hadm_id}: {probability:.1%} 30-day readmission probability "
        f"({probability:.4f}), "
        f"{'flagged' if flagged else 'not flagged'} at a {threshold:.0%} threshold."
        f"{fallback_factors}"
    )

    return _envelope(components, data, fallback)


def risk_card_from_tool_calls(tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find the last predict_readmission result in a run and render it.

    Returns None when the agent answered without predicting — a definitional
    question, say. The caller shows prose in that case rather than an empty
    card.
    """
    for call in reversed(tool_calls or []):
        if call.get("name") == "predict_readmission":
            return build_risk_card(call.get("response") or {})
    return None
