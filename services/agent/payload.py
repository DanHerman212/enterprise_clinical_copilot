"""The strict payload: narrative, citations, and the components to render.

UX Phase 1. A command's answer is delivered as three named things rather than as
prose the caller has to interpret:

  - ``narrative_text`` — the guarded narrative, after citation renumbering. This
    is the same string the legacy ``answer`` field carries; the name is the
    contract's, and it exists so a Phase 2 component binds to a field that means
    what it says rather than to one that grew its meaning over three layers.
  - ``citations`` — the numbered note citations the narrative's markers resolve
    against, as the same records the legacy ``sources`` field carries. Built
    once, published twice: one list is the source of truth, and the second name
    is the interface Phase 2 was specified against.
  - ``ui_components`` — an ordered list of ``{"component", "props"}``, one entry
    per thing the answer should be drawn as. The visuals are Phase 2's; this
    module decides only *which* component and *what data* it is handed.

Everything here is additive. The console keeps rendering ``answer``, ``a2ui``,
``sources`` and ``literature`` exactly as before, so this session produces a
payload both the old renderer and a new one can read.

Why a separate module from ``a2ui.py``
--------------------------------------
``a2ui.py`` composes the legacy canvas — the risk-and-answer envelope the console
draws today, and the code Phase 2 replaces. This module composes the payload that
replaces it. Keeping them apart means the legacy envelope can be deleted in one
file, and means nothing here silently inherits the legacy canvas's decisions
about what an answer is.

The evidence selectors (``predict_payload``, ``rag_payload``,
``literature_payload``) are imported from ``a2ui``: they answer "which call's
result is this turn's evidence", which is a question about the transcript rather
than about the canvas, and they will outlive the envelope they currently sit
beside.

Unavailable is a state, not an absence
--------------------------------------
A component whose tool did not produce usable data is still listed, with
``props`` naming why. Omitting it would render as "nothing to show", which reads
as "we found nothing" — a different and much worse statement than "the tool
failed".
"""

from __future__ import annotations

from typing import Any, Callable

from services.agent.a2ui import literature_payload, predict_payload, rag_payload
from services.agent.feature_labels import display_probability

# The component each command is drawn as. The names on the right are the
# contract's ("Risk Visualizer", "Medication Table", "Markdown Narrative +
# Source Cards", "Literature Cards"), lower-cased as wire identifiers because a
# component name is a string a renderer switches on.
#
# The keys are the WIRE names — the values the request carries and the value
# `state["command"]` holds — not the clinician-facing ones. That distinction is
# not cosmetic: the two vocabularies overlap for two of the four commands, so a
# table keyed by the wrong one works for half its entries and silently falls
# through to the derived path for the rest. A test walks `routing.COMMANDS`
# against this table, because reading it did not catch it.
COMMAND_COMPONENTS: dict[str, str] = {
    "risk": "risk_visualizer",
    "meds": "medication_table",
    "summarize": "narrative_with_sources",
    "literature": "literature_cards",
}

# The same mapping for the free-text path, where the tool rather than the command
# decides what was produced. A free-text question gets the components its
# evidence calls for, in call order, which is what makes this field meaningful
# outside the four commands without a second contract for it.
TOOL_COMPONENTS: dict[str, str] = {
    "predict_readmission": "risk_visualizer",
    "discharge_medications": "medication_table",
    "rag_search": "source_cards",
    "rag_search_sections": "source_cards",
    "search_literature": "literature_cards",
}


def _component(name: str, props: dict[str, Any]) -> dict[str, Any]:
    return {"component": name, "props": props}


def _unavailable(payload: dict | None, tool: str) -> dict[str, str]:
    """Why a component has no data: the tool failed, or it never ran.

    The two are kept distinct for the same reason a failed search is kept
    distinct from an empty one: a caller that renders them identically tells a
    clinician the notes say nothing when the truth is that nothing was read.
    """
    if payload is None:
        return {"unavailable": f"{tool} did not run for this answer"}
    return {"unavailable": str(payload.get("message") or payload.get("error") or
                               f"{tool} returned no usable result")}


def _risk(tool_calls: list[dict], _command_payload: dict | None) -> dict[str, Any]:
    response = predict_payload(tool_calls)
    if not response or "error" in response:
        return _component("risk_visualizer",
                          _unavailable(response, "the prediction tool"))
    return _component("risk_visualizer", {
        # The score at the precision the narrative describes it at, so the widget
        # and the prose cannot show one number two ways. The threshold is already
        # two decimals in the contract, and the decision it drives is the served
        # value's, not this one's.
        "probability": display_probability(response.get("probability")),
        "threshold": response.get("threshold"),
        "decision": response.get("decision"),
        "model_version": response.get("model_version"),
        # The attributions behind the score, in the order the model ranked them.
        # The widget draws these; the narrative's two sentences summarize them.
        "top_factors": response.get("top_factors") or [],
    })


def _medications(tool_calls: list[dict],
                 command_payload: dict | None) -> dict[str, Any]:
    """The verified medication rows, with the refused ones beside them.

    `dropped` travels with the list rather than being logged and forgotten: a row
    the note does not support is the failure a clinician cannot see, and the
    count of them is what says the extraction was partial.
    """
    if not command_payload:
        return _component("medication_table",
                          _unavailable(None, "the medication extraction"))
    return _component("medication_table", {
        "medications": command_payload.get("medications") or [],
        "dropped": command_payload.get("dropped") or [],
        "returned_passages": command_payload.get("returned_passages", 0),
    })


def _narrative_with_sources(tool_calls: list[dict],
                            _command_payload: dict | None) -> dict[str, Any]:
    """The summary narrative's component.

    The prose itself is `narrative_text` and the popovers are `citations`, both
    top-level, so this props block carries only what is neither: which sections
    the summary is made of. Carrying the citation texts here as well would put
    the same note passages on the wire twice.
    """
    response = rag_payload(tool_calls)
    if not response or "error" in response:
        return _component("narrative_with_sources",
                          _unavailable(response, "the summary retrieval"))
    sections = [
        passage.get("section")
        for passage in response.get("passages") or []
        if isinstance(passage, dict)
    ]
    if not sections:
        return _component("narrative_with_sources", _unavailable(
            {"message": response.get("note") or "no summary section was retrieved"},
            "the summary retrieval"))
    return _component("narrative_with_sources", {
        "sections": sections,
        "returned": response.get("returned", len(sections)),
    })


def _literature(tool_calls: list[dict],
                _command_payload: dict | None) -> dict[str, Any]:
    """The published records, projected to the card's fields.

    The same articles the legacy `literature` field carries, projected to
    (pmid, title, abstract) rather than passed through whole. The duplication is
    deliberate and temporary: Phase 2 removes the legacy field, at which point
    this stops being a second copy and becomes the only one.
    """
    response = literature_payload(tool_calls)
    if not response or "error" in response:
        return _component("literature_cards",
                          _unavailable(response, "the literature search"))
    articles = response.get("articles") or []
    if not articles:
        # A search that ran and found nothing is not a failure, and must not be
        # rendered as one — it is an empty card list.
        return _component("literature_cards", {
            "cards": [], "returned": 0,
            "degraded": bool(response.get("degraded", False)),
        })
    return _component("literature_cards", {
        "cards": [
            {
                "pmid": article.get("pmid"),
                "title": article.get("title"),
                "abstract": article.get("abstract"),
            }
            for article in articles
            if isinstance(article, dict)
        ],
        "returned": response.get("returned", len(articles)),
        "degraded": bool(response.get("degraded", False)),
    })


def _source_cards(tool_calls: list[dict],
                  _command_payload: dict | None) -> dict[str, Any]:
    """The note passages a free-text answer cites.

    Which passages those are is `citations`, top-level, where the marker
    numbering already lives; carrying the passage text here as well would be the
    same clinical text twice on the wire. What the card needs from this side is
    which sections were read and how many passages came back.
    """
    response = rag_payload(tool_calls)
    if not response or "error" in response:
        return _component("source_cards", _unavailable(response, "the retrieval"))
    passages = [
        passage for passage in response.get("passages") or []
        if isinstance(passage, dict)
    ]
    if not passages:
        return _component("source_cards", _unavailable(
            {"message": response.get("note") or "no passage was retrieved"},
            "the retrieval"))
    return _component("source_cards", {
        "sections": [passage.get("section") for passage in passages],
        "returned": response.get("returned", len(passages)),
    })


_BUILDERS: dict[str, Callable[[list[dict], dict | None], dict[str, Any]]] = {
    "risk_visualizer": _risk,
    "medication_table": _medications,
    "narrative_with_sources": _narrative_with_sources,
    "source_cards": _source_cards,
    "literature_cards": _literature,
}


def _derived(tool_calls: list[dict]) -> list[dict[str, Any]]:
    """The components a free-text answer's own evidence calls for, in call order.

    Deduplicated by component: a run that made three `rag_search` calls produced
    one set of source cards, not three.
    """
    components: list[dict[str, Any]] = []
    seen: set[str] = set()
    for call in tool_calls:
        name = TOOL_COMPONENTS.get(call.get("name", ""))
        if name is None or name in seen:
            continue
        seen.add(name)
        components.append(_BUILDERS[name](tool_calls, None))
    return components


def build_ui_components(
    command: str | None,
    tool_calls: list[dict],
    command_payload: dict | None = None,
) -> list[dict[str, Any]]:
    """The components this answer is drawn as.

    A command produces exactly one, named by its own table entry — that is the
    point of the command, and a second component would mean the command had
    reached for evidence it does not own. Free text is derived from the calls the
    model actually made.
    """
    if command:
        name = COMMAND_COMPONENTS.get(command)
        if name is not None:
            return [_BUILDERS[name](tool_calls, command_payload)]
    return _derived(tool_calls)
