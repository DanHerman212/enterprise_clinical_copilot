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

The canvas uses the custom components registered in the site's clinical bundle
(static/js/a2ui_clinical_components.js): RiskInfluencers, LiteratureList,
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
from services.agent.feature_labels import display_probability, humanize

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


# The fields a literature record is put on the wire with, in the order the card shows
# them. The abstract is not among them: nothing renders it — the reader's next step is
# the record at PubMed, not a second copy of its text here (UX ruling 2026-10-13).
_LITERATURE_FIELDS = ("pmid", "title", "journal", "pub_date", "url")


def _usable_articles(literature) -> list[dict]:
    """The articles from a literature payload, each shaped for the canvas.

    A record with no pmid is dropped: it cannot be cited or linked, which is the
    whole use of it. The remaining fields are coerced to strings rather than
    passed through, because the console's component schema is strict — one
    missing key drops the entire card, and a card that fails to draw is worse
    than a card with a blank journal.

    An empty list is the answer for three different situations — no search was
    run, the search failed, or it matched nothing — which the canvas treats the
    same and the answer text and provenance caption distinguish between.
    """
    if not isinstance(literature, dict):
        return []
    articles = literature.get("articles")
    if not isinstance(articles, list):
        return []
    usable = []
    for article in articles:
        if not isinstance(article, dict) or not article.get("pmid"):
            continue
        usable.append({
            field: article.get(field) if isinstance(article.get(field), str) else ""
            for field in _LITERATURE_FIELDS
        })
    return usable


# Shown when a search ran, matched nothing, and the tool offered no wording of its own.
_LITERATURE_EMPTY_NOTE = "No matching literature was found for this query."

# Shown when the search could not be performed at all. Deliberately a different
# sentence from the one above: that one says the evidence does not exist, this one says
# nobody looked, and a clinician deciding whether to trust an answer needs to know which
# of those they are being told.
_LITERATURE_UNAVAILABLE_NOTE = "PubMed literature search is temporarily unavailable."


def _literature_view(literature) -> tuple[list[dict], str, bool] | None:
    """The articles, the note, and whether the search was degraded — or None.

    None means no search happened and the canvas keeps its usual behaviour. A triple
    with an EMPTY article list means a search ran and produced nothing to draw, and the
    note says which of two very different things that was: matched nothing, or could
    not be made at all.

    Three states, and the canvas needs all three. Collapsing any pair of them is what
    would leave the clinician unable to tell "no results" from "nothing happened" —
    the ambiguity the wrapper object on the wire exists to prevent, and throwing it
    away one layer further down would be the same defect somewhere new.

    A tool error can still arrive for one case: a blank query, which is a defect in
    our own pipeline rather than anyone else's service. It is treated as degraded
    rather than as no-search, because a payload did cross the boundary and the canvas
    has to account for it.
    """
    if not isinstance(literature, dict):
        return None
    if "error" in literature:
        return [], _LITERATURE_UNAVAILABLE_NOTE, True
    degraded = bool(literature.get("degraded"))
    articles = _usable_articles(literature)
    if articles:
        return articles, "", degraded
    note = literature.get("note")
    if isinstance(note, str) and note.strip():
        return [], note.strip(), degraded
    # The fallback has to respect the flag. Defaulting a degraded search to the
    # "found nothing" sentence would state that something was searched for and absent
    # when in fact nothing was ever examined — the precise claim this whole path
    # exists to avoid making. The tool always sends a note today; this is what keeps
    # that guarantee from being the only thing holding the sentence up.
    fallback = _LITERATURE_UNAVAILABLE_NOTE if degraded else _LITERATURE_EMPTY_NOTE
    return [], fallback, degraded


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


# The section names as a clinician reads them. A separate vocabulary from
# `feature_labels`, which names MODEL features: these name parts of a note, and the
# two happen to share the word "discharge" without meaning the same thing by it.
_SECTION_LABELS = {
    "history_of_present_illness": "History of present illness",
    "brief_hospital_course": "Brief hospital course",
    "discharge_diagnosis": "Discharge diagnosis",
    "discharge_medications": "Discharge medications",
    "medications_on_transfer": "Medications on transfer",
    "discharge_instructions": "Discharge instructions",
    "discharge_condition": "Discharge condition",
    "discharge_disposition": "Discharge disposition",
    "followup_instructions": "Follow-up instructions",
}


def _section_label(section: object) -> str:
    name = str(section or "")
    return _SECTION_LABELS.get(name, name.replace("_", " ") or "Note passage")


def narrative_surface(answer: str, citations: list[dict]) -> dict:
    """The prose, and the evidence its markers point at, as one component.

    The markers and the sentences they annotate travel together because a marker is a
    control the reader clicks while reading the sentence it sits in. The answer used to
    be drawn as plain text in a separate element with nothing to click, because the
    basic Text component renders Markdown and Markdown cannot host a button.

    `citations` are built by the caller from the evidence this turn actually rests on;
    see `note_citations` and `literature_citations`. Each entry's `index` is the
    number the marker shows, so the two cannot drift.
    """
    return {"text": answer, "citations": citations, "streaming": False}


def note_citations(sources: list[dict]) -> list[dict]:
    """One citation per resolved note passage, keyed by its renumbered marker.

    The citation layer renumbers `^[n]` to first-appearance order; reading the
    renumbered `cite` here keeps the marker and the popover in step without a second
    mapping to drift.
    """
    return [
        {
            "index": int(source["cite"]),
            # The channel the evidence came from. Note passages all share one channel
            # and differ by section, so the section becomes the title.
            "source": "Discharge note",
            "title": _section_label(source.get("section")),
            "excerpt": str(source.get("text") or ""),
        }
        for source in sources
    ]


def literature_citations(articles: list[dict]) -> list[dict]:
    """One citation per retrieved article, numbered as the summary cites them.

    `index` is the 1-based position in the record order the model was given, which is
    exactly what the formatting instruction tells it to cite — so the marker in the
    prose and the record beneath it are the same record by construction, and a marker
    with no record behind it can only mean the model cited a number it never got.

    `pmid` is what the marker DISPLAYS. A literature question is answered against
    published records and the reader's next step is to open one, so the marker carries
    the identifier itself and the console links it to PubMed. The number stays as the
    wire index because the prose the model wrote is numbered; what that number looks
    like on screen is the console's decision, not the model's.

    No excerpt. That was the abstract, kept for a popover that no longer exists on this
    path, and it travelled as a second copy of text the record already carries.
    """
    citations = []
    for position, article in enumerate(articles, start=1):
        citation = {
            "index": position,
            "source": "PubMed",
            "title": str(article.get("title") or "Untitled record"),
        }
        if article.get("pmid"):
            citation["pmid"] = str(article["pmid"])
        if article.get("url"):
            citation["url"] = str(article["url"])
        citations.append(citation)
    return citations


def _narrative(answer: str, sources: list[dict], literature: dict | None) -> dict | None:
    """The narrative surface for this answer, or None when there is no prose.

    Citations come from whichever evidence the turn rests on. A literature turn cites its
    articles and nothing else: those records ARE the evidence, and resolving the markers
    against note passages would leave every marker in a summary unbacked — which the
    component draws as a disabled control, on a summary that was properly grounded.
    """
    if not answer.strip():
        return None
    if literature is not None:
        return narrative_surface(answer, literature_citations(_usable_articles(literature)))
    return narrative_surface(answer, note_citations(sources))


def compose_risk_canvas(predict: dict | None, rag: dict | None,
                        cite: int = 1, sections: tuple[str, ...] = (),
                        source: dict | None = None,
                        literature: dict | None = None,
                        medications: dict | None = None,
                        narrative: dict | None = None) -> dict:
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

    `literature` is the `search_literature` response. When a search ran, its records
    take the canvas in place of the risk widgets — including when it matched nothing or
    when it could not be made, in which case the record list is empty, the note says
    which happened, and `degraded` marks the second for the browser. `None` means no
    search ran and the risk path is untouched.

    `medications` is the `/meds` command's own payload: rows verified against the note
    text before they got here. The grid draws them and nothing else — a list of drugs
    in prose is what the grid replaces, and an empty grid would read as "this patient
    takes nothing", which is a clinical claim this system has no basis for. With no
    rows the narrative carries the absence alone, exactly as the missing SourceCard
    says nothing rather than explaining itself.
    """
    components: list[dict] = [
        {"id": "root", "component": "Column", "children": []},
    ]
    children = components[0]["children"]

    # The answer leads and its evidence follows, for every command. The narrative is
    # the one child that is prose, so it goes first: a reader who has to scroll past a
    # grid to find the sentence that explains it has been given the parts in the wrong
    # order. For `/meds` and a literature search the sentence is a caption — "9
    # discharge medications" — and a caption above its table is where it belongs.
    if narrative is not None:
        children.append("narrative")
        components.append({
            "id": "narrative",
            "component": "CitedNarrative",
            "text": narrative["text"],
            "citations": narrative["citations"],
            "streaming": bool(narrative.get("streaming")),
        })

    estimate = _usable_estimate(predict)
    literature_view = _literature_view(literature)
    if literature_view is not None:
        articles, note, degraded = literature_view
    else:
        articles, note, degraded = [], "", False

    if literature_view is not None:
        # Literature takes the canvas. This is the swap the feature exists for: the
        # question was about published evidence, so the risk widgets are not merely
        # absent — they are the wrong thing to be looking at. The records are shown
        # as the tool returned them, unedited, which is what lets the clinician
        # check the summary instead of trusting it.
        #
        # An empty result keeps the swap and carries a note instead. Leaving the
        # previous view standing would be worse than showing nothing: the clinician
        # clicked a chip, and a canvas that did not change reads as a click that did
        # not work.
        children += ["literature"]
        components.append({"id": "literature", "component": "LiteratureList",
                           "articles": articles,
                           "note": note,
                           "degraded": degraded})
        query_text = (literature or {}).get("query") or ""
        if articles:
            fallback = (
                f"{len(articles)} PubMed record"
                f"{'s' if len(articles) != 1 else ''} retrieved"
                + (f" for \"{query_text}\"" if query_text else "")
                + "."
            )
        else:
            fallback = note
    elif predict is not None and estimate is None:
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
        # The band is a decision, so it reads the served value; the number is a
        # display, so it is rounded. Rounding before the comparison would let the
        # display precision move a threshold call.
        band = _band(probability, threshold)
        shown = display_probability(probability)
        factors = estimate.get("top_factors") or []

        # One widget, not a score bar plus a separate bar list: the component
        # restates the score and the threshold inside itself, so the band and the
        # number it belongs to cannot be pulled apart by layout.
        #
        # `probability` goes over UNROUNDED. The widget formats both numbers to one
        # decimal place, so pre-rounding to the prose's two decimals would discard
        # exactly the digit that distinguishes a score from its threshold — the
        # reason the widget stopped rounding to whole percent. `fallback_text` below
        # keeps the rounded form, because that sentence is written, not computed.
        #
        # `decision` is passed rather than left to the widget's own `>=`
        # comparison: the model already made the threshold call, and the two must
        # not be able to disagree. A payload without one falls back to the
        # comparison, which is why 0 is a safe default rather than a claim.
        #
        # A factor carries no `direction` key at all when the tool did not state
        # one. The component's schema is strict, so an explicit null would be
        # rejected and would take the whole widget down with it.
        children += ["risk"]
        components.append({
            "id": "risk",
            "component": "RiskInfluencers",
            "decision": int(estimate.get("decision") or 0),
            "probability": probability,
            "threshold": threshold,
            "factors": [
                {
                    "label": humanize(f["feature"]),
                    "contribution": float(f["contribution"]),
                    **({"direction": f["direction"]}
                       if isinstance(f.get("direction"), str) else {}),
                }
                for f in factors
            ],
        })

        # Both forms come from the rounded value, so the caption and the bar show
        # one number at one precision. Whole percent is the same precision as two
        # decimals of a probability — 0.09 is 9%, not 9.0%.
        # No admission id: this is prose a clinician can read, and `fallback_text`
        # is what a client without the renderer shows. The identifier ban covers it
        # for the same reason it covers the narrative (UX Phase 1B) — the id is
        # ours, not a clinical fact, and the reader cannot look it up.
        fallback = (
            f"{shown:.0%} 30-day readmission "
            f"probability ({shown:.2f}), {band} at a {threshold:.2f} threshold."
        )

    # `/meds` brings its own rows. They are verified against the note text before
    # they reach here (`deterministic.parse_medications`), so the grid draws what was
    # read and nothing that was inferred. Inserted before the caption and the citation
    # so the structured answer comes first and the evidence it rests on follows.
    #
    # No rows means no grid at all, not an empty one: a grid with no rows reads as
    # "this patient takes nothing", which is a clinical claim this system has no basis
    # for. The narrative already says the section was retrieved and could not be read,
    # and one absence stated once is the same rule the missing SourceCard follows.
    if medications is not None and (medications.get("medications") or []):
        children.append("medications")
        components.append({
            "id": "medications",
            "component": "MedicationGrid",
            "medications": [
                {
                    "medication": str(row.get("medication") or ""),
                    "dosage": str(row.get("dosage") or ""),
                    "frequency": str(row.get("frequency") or ""),
                }
                for row in medications["medications"]
            ],
            # The rows that failed verification are never drawn — they are the ones
            # the system could not stand behind. The count is carried so the grid can
            # say how many are missing from it, which is the difference between a
            # complete list and a list with a hole in it.
            "dropped_count": len(medications.get("dropped") or []),
            "returned_passages": int(medications.get("returned_passages") or 0),
        })

    # Provenance caption: what the answer came from. Two forms, and a third case
    # with no caption at all. A risk estimate names the model and the feature
    # source, the two facts a reviewer checks. A literature turn names the query
    # and the record count. A question that is neither has no provenance to state,
    # and the sentence that used to stand in for one — "no readmission estimate was
    # requested" — was a note to us rendered to a clinician, who had not asked for
    # an estimate and could only read its absence as a fault (removed 2026-10-12).
    if literature_view is not None:
        query_text = (literature or {}).get("query") or ""
        if degraded:
            prov_text = "PubMed · unavailable · A2UI canvas"
        else:
            prov_text = f"PubMed · {len(articles)} record(s)"
            if query_text:
                prov_text += f" for \"{query_text[:80]}\""
            prov_text += " · A2UI canvas"
    elif estimate:
        model = estimate.get('model_version', 'unknown')
        feature_source = estimate.get('feature_source', 'unknown')
        prov_text = f"Model {model} · features from {feature_source} · A2UI canvas"
    else:
        prov_text = None
    if prov_text is not None:
        children.append("prov")
        components.append({"id": "prov", "component": "Text",
                           "text": prov_text,
                           "variant": "caption"})

    # Retired 2026-10-13 (UX ruling): the cited narrative carries the passage text per
    # sentence, as an interactive marker the reader opens. The card showed the same
    # evidence a second time, in a second place, disconnected from the sentence it
    # supported — redundant weight next to a better affordance. The comment below is
    # kept because it records why the card appeared only sometimes:
    #
    # Cited source — the passage the answer cites first (mirrors the custom
    # demo). Kept as the LAST child so it can pin to the bottom of the canvas
    # (sticky) and the discharge notes stay in view however long the session
    # gets. No card when there is no passage to cite: an empty citation slot is
    # not evidence of a fault and does not need to say anything.

    return {
        "surface_id": SURFACE_ID,
        "audience": AUDIENCE,
        # Tells the console the prose is drawn by this surface, so it must not also
        # write the answer into its own text element. One answer, one place: the
        # duplicate is what a reader notices first and it is the failure the risk
        # sentence was deleted for.
        "renders_prose": narrative is not None,
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


def literature_payload(tool_calls: list[dict]) -> dict | None:
    """The `search_literature` response for this run, or None.

    First match rather than last, unlike `predict_payload`: a run normally makes one
    search, and if a model made two the answer was written against the first.

    A failed search is returned as it stands. Its `error` key is what tells the
    composer and the caption that the tool did not answer, which is a different
    statement from "nothing was found" — and dropping it here would collapse the
    two into the same empty canvas.
    """
    for call in tool_calls or []:
        if call.get("name") == "search_literature":
            return call.get("response") or None
    return None


def compose_presentation(question: str, answer: str,
                         tool_calls: list[dict],
                         command_payload: dict | None = None) -> dict:
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

    `command_payload` is a command's own structured result when it has one — `/meds`
    extracts rows the canvas draws as a grid. It is passed through rather than
    recomposed, because the rows were verified against the note text in the command
    and re-deriving them here would be a second implementation of that check.

    Pure — no I/O, no clock, no randomness — so it is fully testable without
    a browser or a live endpoint.
    """
    renumbered = renumber_citations(answer)
    remap = citation_remap(answer)
    intent = intent_sections(question)
    rag = rag_payload(tool_calls)
    literature = literature_payload(tool_calls)
    sources = resolve_sources(renumbered, remap, rag, intent)
    return {
        "answer": renumbered,
        "citation_map": remap,
        "intent_sections": list(intent),
        "sources": sources,
        # The published records the answer rests on, passed through whole. Kept
        # separate from `sources` on purpose: that list is the discharge-note
        # citation channel — numbered, renumbered, and resolved against retrieved
        # passages — and an article has no passage to resolve against.
        "literature": literature,
        "a2ui": compose_risk_canvas(
            predict_payload(tool_calls),
            rag,
            cite=first_citation(renumbered),
            sections=intent,
            source=sources[0] if sources else None,
            literature=literature,
            medications=command_payload,
            # The narrative chooses its citations from this turn's evidence: the
            # articles for a literature turn, the resolved note passages otherwise.
            narrative=_narrative(renumbered, sources, literature),
        ),
    }
