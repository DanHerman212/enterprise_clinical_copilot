"""Compose the A2UI v0.9 risk canvas and citation metadata for an answer.

This module is the single presentation adapter for the agent's HTTP contract.
The canvas and the citation metadata are composed HERE — in the agent, where
the answer, the guardrails, and the retrieved evidence meet — never downstream
in the BFF or the browser. The caller (Django) passes the envelope through
unchanged; the browser renders it.

Three rules this module exists to enforce:

1. **Tools return JSON, the agent composes UI.** `prediction.py` deliberately
   never emits A2UI — a tool that returns UI is welded to one presentation
   layer and stops working from Claude Desktop or CI. The translation happens
   here, on the agent side, where a presentation choice belongs.

2. **Never render without a text fallback (R8).** Every return carries
   `fallback_text`. If the CDN is down, the renderer throws, or the spec
   version drifts, the caller still has something true to show. A demo that
   renders nothing is worse than one that renders plainly.

3. **The model's citation numbers are advisory.** They are renumbered
   deterministically (first appearance order) and the SourceCard resolves its
   passage by section intent, not by the model's numbering — see
   `services/agent/citations.py`.

The payload shape is v0.9, which is *not* backward compatible with the v0.8
examples that dominate search results. In v0.9 `component` is a string and
properties sit inline. Feeding v0.8 shapes to a v0.9 renderer raises nothing —
it logs "Component implementation not found for type: [object Object]" and
draws an empty box. The tests in tests/agent/test_a2ui.py pin the v0.9 shape
so that regression is caught here rather than in a browser.

The canvas uses the custom components registered in the site's vendored
renderer (static/vendor/a2ui/a2ui_risk_components.js): RiskBar, FactorBars,
SourceCard. The surface's `catalogId` must equal the combined catalog id the
front-end registers, so one surface can use basic AND custom components
together.
"""

from services.agent.citations import (
    citation_remap,
    cited_numbers,
    extract_section,
    first_citation,
    intent_sections,
    renumber_citations,
)
from services.agent.feature_labels import humanize

# Read off the site's registered catalog: the vendored renderer resolves the
# custom components against this exact id, and a mismatch means nothing draws.
CATALOG_ID = "https://example.com/catalogs/readmission-risk-v1.json"

SURFACE_ID = "risk-canvas"
A2UI_VERSION = "v0.9"

# R7: rendered payloads are for the user, not for the model to re-read on a
# later turn. Hiding them cuts tokens and, more importantly, stops the model
# treating its own UI output as evidence — a faithfulness risk under Tier 2.
AUDIENCE = ["user"]


def _band(probability: float, threshold: float) -> str:
    """low = below threshold · borderline = threshold to threshold + 0.08."""
    if probability < threshold:
        return "low"
    if probability < threshold + 0.08:
        return "borderline"
    return "high"


def _usable_estimate(predict) -> dict | None:
    """Return predict only when it carries a real estimate (numeric probability
    + threshold). A predict tool that failed (endpoint down, bad payload)
    returns a dict *without* those keys — treat it as "no estimate", never as a
    risk score. This is what keeps the canvas composer from 500ing on a failed
    tool call.
    """
    if not isinstance(predict, dict):
        return None
    try:
        float(predict["probability"])
        float(predict["threshold"])
    except (KeyError, TypeError, ValueError):
        return None
    return predict


def _unavailable_text(section: str) -> str:
    """The deterministic message for a question whose target section the note
    does not have. Never mine the content from unrelated narrative."""
    if section == "discharge_medications":
        return "No discharge medication information is available for this patient."
    return f"No {section.replace('_', ' ')} information is available for this patient."


def resolve_source(passages: list[dict], passage_n: int,
                   sections: tuple[str, ...] = ()) -> dict:
    """Resolve one citation to the passage and section body it supports.

    `passage_n` is the model's ORIGINAL 1-based passage number (before
    renumbering). `sections` is the question's section intent; when given, the
    section is located by label or extracted from any whole-note chunk, and
    the model's number is ignored — the model mis-numbers citations (a meds
    answer cites ^[1] while its supporting passage sits elsewhere).

    Returns {"section", "text"}. Callers own the badge number and the query.
    """
    cited = None
    intent_body = None
    matched_section = None
    for sec in sections:
        cited = next((p for p in passages if p.get("section") == sec), None)
        if cited is not None:
            matched_section = sec
            break
        # The index stores whole-note chunks: a passage labeled with a
        # different section still CONTAINS the target section, and the
        # labeled chunk can miss the top-k (near-tied whole-note
        # embeddings). Pull the section body out of the passage text.
        for p in passages:
            body = extract_section(p.get("text", ""), sec)
            if body:
                cited = p
                intent_body = body
                matched_section = sec
                break
        if cited is not None:
            break

    if cited is None and sections:
        # The note has NONE of the targeted sections. The deterministic answer
        # is "not available", not a passage mined from unrelated narrative.
        return {"section": "not available",
                "text": _unavailable_text(sections[0])}

    if cited is None:
        # Clamp to a real passage so an out-of-range number never raises.
        idx = min(max(passage_n, 1), len(passages)) - 1
        cited = passages[idx]
        shown_section = cited.get("section", "discharge note")
        section_text = extract_section(cited.get("text", ""), shown_section)
    elif intent_body is not None:
        shown_section = matched_section
        section_text = intent_body
    else:
        shown_section = matched_section or cited.get("section", "discharge note")
        section_text = extract_section(cited.get("text", ""), shown_section)

    return {"section": shown_section,
            "text": section_text or cited.get("text", "")}


def resolve_sources(answer: str, citation_map: dict[str, int],
                    rag: dict | None, sections: tuple[str, ...]) -> list[dict]:
    """One resolved source per citation number in the (renumbered) answer.

    This is the list the browser renders on a footnote click: it looks up the
    clicked number and displays the entry — no vocabulary, no heuristics.
    Section intent is applied only to a single-citation answer; a
    multi-citation answer cites its passages in array order and maps by
    number.
    """
    passages = (rag or {}).get("passages") or []
    if not passages:
        return []
    query = (rag or {}).get("query") or "discharge note"
    cites = cited_numbers(answer)
    if not cites:
        return []
    intent = sections if len(cites) == 1 else ()
    return [
        {"cite": n,
         **resolve_source(passages, citation_map.get(str(n), n), intent),
         "query": query}
        for n in cites
    ]


def compose_risk_canvas(predict: dict | None, rag: dict | None,
                        cite: int = 1, sections: tuple[str, ...] = (),
                        source: dict | None = None) -> dict:
    """Turn one predict (+ rag) payload into the A2UI risk-canvas envelope.

    predict is None when the question did not request a readmission estimate
    (e.g. a medication or summarize question), or an unusable dict when the
    predict tool errored — the canvas then answers with a plain honest note
    plus the cited source instead of a risk score, so the agent never 500s on
    a non-risk or failed-risk turn.

    `source` is an already-resolved entry from `resolve_sources`; when given
    it is the SourceCard verbatim, so the canvas and the click-through list
    can never disagree. Without it the card is resolved here from `cite` and
    `sections`.
    """
    components: list[dict] = [
        {"id": "root", "component": "Card", "child": "body"},
        {"id": "body", "component": "Column", "children": []},
    ]
    children = components[1]["children"]

    estimate = _usable_estimate(predict)
    if predict is not None and estimate is None:
        # The predict tool ran but returned no usable estimate (e.g. the
        # serving endpoint was unreachable). Honest note — never a 500, never
        # a made-up number.
        children += ["note"]
        components.append({"id": "note", "component": "Text",
                           "text": "The readmission-risk service did not return "
                                   "a usable estimate for this question.",
                           "variant": "h2"})
        fallback = ("The readmission-risk service did not return a usable "
                    "estimate for this question.")
    elif estimate is None:
        # Non-risk question (meds / summarize / free text): no estimate, and no
        # big "no risk estimate" heading either — that space belongs to the
        # citation source. The small provenance caption below already says
        # plainly that no estimate was requested.
        fallback = "No 30-day readmission risk estimate was requested for this question."
    else:
        probability = float(estimate["probability"])
        threshold = float(estimate["threshold"])
        band = _band(probability, threshold)
        factors = estimate.get("top_factors") or []
        # Post-process the SHAP names for visual consumption: each factor gets
        # a human label (feature_labels.humanize) so the bars never show a raw
        # model key (camelCase or snake_case).
        factors = [
            {**f, "label": humanize(f["feature"])}
            for f in factors
        ]

        # Each widget is a self-contained custom component whose chrome (card,
        # widget title, band pill, SHAP bars, cited source) mirrors the custom
        # demo.
        children += ["risk", "factors"]
        components.append({"id": "risk", "component": "RiskBar",
                           "probability": probability, "threshold": threshold,
                           "band": band})
        components.append({"id": "factors", "component": "FactorBars",
                           "factors": factors})

        fallback = (
            f"Admission {estimate.get('hadm_id')}: {probability:.1%} 30-day readmission "
            f"probability ({probability:.4f}), {band} at a {threshold:.2f} threshold."
        )

    # Provenance caption. For a risk estimate it names the model + feature
    # source (the two facts a reviewer checks). For a non-risk question there
    # is no estimate, so we say so plainly instead of rendering bare dashes
    # that read as a rendering bug.
    if estimate:
        model = estimate.get('model_version', 'unknown')
        feature_source = estimate.get('feature_source', 'unknown')
        prov_text = f"Model {model} · features from {feature_source} · A2UI canvas"
    else:
        prov_text = "No model — no readmission estimate was requested for this question."
    children.append("prov")
    components.append({"id": "prov", "component": "Text",
                       "text": prov_text,
                       "variant": "caption"})

    # Cited source — the passage the answer cites first (mirrors the custom
    # demo). Kept as the LAST child so it can pin to the bottom of the canvas
    # (sticky) and the discharge notes stay in view however long the session
    # gets.
    passages = (rag or {}).get("passages") or []
    query = (rag or {}).get("query") or "discharge note"
    children.append("source")
    if source is not None:
        components.append({"id": "source", "component": "SourceCard",
                           "cite": source["cite"],
                           "section": source["section"],
                           "text": source["text"],
                           "query": source.get("query", query)})
    elif passages:
        resolved = resolve_source(passages, cite, sections)
        components.append({"id": "source", "component": "SourceCard",
                           "cite": max(cite, 1),
                           "section": resolved["section"],
                           "text": resolved["text"],
                           "query": query})
    else:
        components.append({"id": "source", "component": "SourceCard",
                           "cite": 1, "section": "not found",
                           "text": "No supporting note passage was found for this "
                                   "question. An empty result is a real answer — "
                                   "the agent does not fabricate passages.",
                           "query": query})

    return {
        "surface_id": SURFACE_ID,
        "audience": AUDIENCE,
        "messages": [
            {"version": A2UI_VERSION,
             "createSurface": {"surfaceId": SURFACE_ID, "catalogId": CATALOG_ID}},
            {"version": A2UI_VERSION,
             "updateComponents": {"surfaceId": SURFACE_ID, "components": components}},
        ],
        "fallback_text": fallback,
    }


def predict_payload(tool_calls: list[dict]) -> dict | None:
    """The last `predict_readmission` response in a run, or None.

    The last call wins when several ran — it reflects the final state of the
    conversation. None means the agent answered without predicting (a meds or
    summarize question), which composes the no-estimate canvas.
    """
    for call in reversed(tool_calls or []):
        if call.get("name") == "predict_readmission":
            return call.get("response") or None
    return None


def rag_payload(tool_calls: list[dict]) -> dict | None:
    """The retrieval response — the free-text `rag_search` or the
    deterministic `rag_search_sections` used for summaries — so the canvas
    source card is drawn for both paths."""
    for name in ("rag_search", "rag_search_sections"):
        for call in tool_calls or []:
            if call.get("name") == name:
                return call.get("response") or None
    return None


def compose_presentation(question: str, answer: str,
                         tool_calls: list[dict]) -> dict:
    """The full presentation contract for an answered question.

    One call produces every field the browser consumes, so the answer's
    evidence semantics are decided in exactly one place (here) and the BFF is
    a pass-through:

      - answer: the guarded answer with citations renumbered to
        first-appearance order
      - sources: one resolved {cite, section, text, query} per citation in
        the answer — the browser looks a clicked footnote up by number
      - a2ui: the composed risk-canvas envelope; its SourceCard is sources[0]

    citation_map and intent_sections are inputs to that resolution and are
    returned for tests and diagnostics; the browser does not need them.

    Pure — no I/O, no clock, no randomness — so it is fully testable without
    a browser or a live endpoint.
    """
    renumbered = renumber_citations(answer)
    remap = citation_remap(answer)
    intent = intent_sections(question)
    rag = rag_payload(tool_calls)
    sources = resolve_sources(renumbered, remap, rag, intent)
    return {
        "answer": renumbered,
        "citation_map": remap,
        "intent_sections": list(intent),
        "sources": sources,
        "a2ui": compose_risk_canvas(
            predict_payload(tool_calls),
            rag,
            cite=first_citation(renumbered),
            sections=intent,
            source=sources[0] if sources else None,
        ),
    }
