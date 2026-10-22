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
from services.agent.admission_phenotypes import (
    bucket_for,
    is_matrix_eligible,
    NON_ADMISSION,
    phenotype_of_query,
)
from services.agent.literature_benchmarks import benchmarks_for_phenotype

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
            # What the marker SAYS, resolved here rather than in the browser. The marker
            # reads `1 · Hospital Course` and not a bare `1`, because a number tells the
            # reader nothing about what they are about to open — and resolving it here
            # keeps the section vocabulary in the one module that already owns it
            # (`_SECTION_LABELS`), rather than a second copy in the client that would
            # drift the first time a section is renamed.
            "label": f"{int(source['cite'])} · {_section_label(source.get('section'))}",
            "excerpt": str(source.get("text") or ""),
        }
        for source in sources
    ]


#: How long a literature marker's source token may get before it is cut back.
#:
#: A journal name is unbounded — "Journal of the American College of Cardiology" is
#: forty-seven characters — and an inline marker is a pill set inside a sentence, so an
#: unbounded label is a pill whose own box wraps to a second line and reads as a broken
#: one. Asked for by a reader looking at exactly that, 2026-10-16.
#:
#: Twenty-four characters keeps "Home healthcare now" and "Patient education and…"
#: recognisable. The full title is a hover away on the marker's popover and a line below in
#: the record list, which is where a reader goes to weigh a source anyway.
_CITE_SOURCE_CHARS = 24


def _short_source(name: str) -> str:
    """A source name cut back to the marker's budget, at a word boundary.

    At a boundary rather than mid-word, and with an ellipsis rather than silently: a cut
    that lands inside a word reads as a typo, and one with nothing to mark it reads as the
    whole name.
    """
    text = " ".join(str(name or "").split())
    if len(text) <= _CITE_SOURCE_CHARS:
        return text
    cut = text[:_CITE_SOURCE_CHARS]
    space = cut.rfind(" ")
    if space > 8:
        cut = cut[:space]
    return cut.rstrip(" ,.;:") + "…"


def literature_citations(articles: list[dict]) -> list[dict]:
    """One citation per retrieved article, numbered as the summary cites them.

    `index` is the 1-based position in the record order the model was given, which is
    exactly what the formatting instruction tells it to cite — so the marker in the
    prose and the record beneath it are the same record by construction, and a marker
    with no record behind it can only mean the model cited a number it never got.

    `label` is what the marker DISPLAYS and `pmid` is what it links to. The two used to be
    one string: the marker printed the identifier, on the reasoning that a literature
    question is answered against published records and the reader's next step is to open
    one. The 2026-10-15 remediation moved the display to the journal, because on a
    literature turn the reader is weighing sources and the journal is what says what kind
    of claim a record makes. The identifier is still on screen a hover away and is still
    the link target, so nothing became unactionable — but this is a deliberate trade of a
    terse identifier for an informative name, and it is the thing to revisit first if the
    inline pills start reading long.

    The number stays as the wire index because the prose the model wrote is numbered;
    what that number looks like on screen is the console's decision, not the model's.

    The trade made on 2026-10-15 was revisited on 2026-10-16, which is what the paragraph
    above asked for: the pills WERE reading long. The journal is kept and cut to
    `_CITE_SOURCE_CHARS`, so the marker still says what kind of source a record is without
    its box wrapping inside a sentence. The identifier is untouched as the link target.

    No excerpt. That was the abstract, kept for a popover that no longer exists on this
    path, and it travelled as a second copy of text the record already carries.
    """
    citations = []
    for position, article in enumerate(articles, start=1):
        journal = str(article.get("journal") or "").strip()
        citation = {
            "index": position,
            "source": "PubMed",
            "title": str(article.get("title") or "Untitled record"),
            # The journal when the record carries one, the identifier otherwise. Chosen
            # from what the record actually has rather than assumed: a record with no
            # journal must not render as an empty pill. Cut to a marker's length — see
            # `_short_source` for why, and for why the cut is marked.
            "label": _short_source(journal) or (
                f"PMID {article['pmid']}" if article.get("pmid") else "PubMed"),
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

    The turn is only a literature turn when it produced RECORDS. An empty search result
    is a `dict`, so a bare `is not None` test made every chart answer that happened to
    carry a fruitless PubMed call cite zero articles: the note passages had already been
    resolved, and this function threw them away in favour of an empty list. The reader
    then had a `^[1]` that resolved to nothing beside a card saying no literature was
    found (2026-10-22). Records or note sources — never an empty list over a full one.
    """
    if not answer.strip():
        return None
    articles = _usable_articles(literature or {})
    if articles and literature is not None:
        return narrative_surface(answer, literature_citations(articles))
    return narrative_surface(answer, note_citations(sources))


#: What the pinned row calls the estimate's source.
#:
#: Deliberately not the algorithm's name. The directive that asked for this row wrote
#: "Enterprise Clinical Copilot (XGBoost)", and nothing in the payload says XGBoost: the
#: tool reports a served model version and a feature source, and no learner. Naming a
#: learner here would be this layer asserting an implementation detail it cannot
#: evidence, which is the one kind of claim it must not make. The version IS in the
#: payload, so the version carries the provenance and the source stays what it is.
_BASELINE_SOURCE = "Enterprise Clinical Copilot"


def _draws_matrix(command: str | None, has_literature: bool, phenotype: str,
                  query: str) -> bool:
    """Whether this turn may carry the comparative cohort table at all.

    Reserved to the command whose question it answers. The table exists to qualify a
    readmission estimate against published cohorts, and only two turns ask that: `/ehr`,
    which is the briefing the estimate belongs to, and a `/pubmed` turn that is ABOUT the
    patient's own condition.

    The gate is the command rather than the evidence, and that distinction is the bug this
    replaces. It used to ask whether a search had run and whether labs had been drawn, and
    the phenotype resolved from any tool call carrying an admission — so `/meds`, a lab
    drill-down and every follow-up question drew the table, because all of them read the
    patient's chart. Observed live: `/meds` and a follow-up about blood transfusions both
    arrived with a heart-failure cohort table under them, captioned "No readmission
    estimate was requested for this turn" — the component explaining its own irrelevance,
    which is the surface telling us it should not have been drawn.

    A command that is not `/ehr` or `/pubmed` therefore draws nothing, including `/risk`
    (its widget is the whole answer), `/labs` (the trajectories are), `/meds` (the list is)
    and a free-text turn, which has no command at all. Follow-ups are the case that
    matters most and are free-text turns: they inherit the patient from the thread they
    continue, so every one of them carries an admission and none of them asks about
    cohorts.
    """
    wire = str(command or "").strip().lower()
    if wire == "summarize":
        return True
    if wire == "literature":
        # The condition-lock's other half. Asking about another condition draws the
        # records and no comparison; asking about this patient's draws both.
        return bool(has_literature) and phenotype_of_query(query) == phenotype
    return False


def _comparative_matrix(estimate: dict | None, phenotype: str | None) -> dict | None:
    """This patient's estimate set against the published cohorts, or None.

    Rows come from `literature_benchmarks`, selected by the admission's resolved
    phenotype (`admission_phenotypes`). None when the admission resolved to no phenotype,
    or to one the curated corpus has no cohort for. The PMIDs a search returned have
    nothing to do with whether the table draws, and that is the Phase 4B change: under the
    rule this replaces, rows were matched against the records a search returned, no live
    search ever returned a curated one, and the matrix therefore did not render at all.

    A missing estimate does NOT suppress the table. The cohorts are worth reading on
    their own, and the component is told which of the two things it is drawing: with a
    baseline row when there is an estimate, and with a sentence saying so when there is
    not. What it must never do is draw the baseline row with an empty figure in it.
    """
    benchmarks = benchmarks_for_phenotype(phenotype)
    if not benchmarks:
        return None
    # `matched_features` is dropped here rather than sent and ignored. The component's
    # schema is `.strict()`, so a field it no longer renders is not a harmless extra —
    # it fails the parse and the table does not draw at all.
    rows = [
        {key: value for key, value in row.items() if key != "matched_features"}
        for row in benchmarks
    ]

    # The rows carry no "matched risk factors" column any more (Phase 4C §1.2). The
    # driver bars directly above the table already show which factors move this
    # patient's estimate and in which direction, so repeating them as flat pills in a
    # fourth column was the same evidence twice — and two renderings of one source read
    # as two sources, which is the misreading the citation contract exists to prevent,
    # one surface down. The benchmark still records `matched_features`; nothing in this
    # envelope sends it, and the component's schema no longer accepts it.

    component: dict = {
        "id": "comparison",
        "component": "ComparativeRiskMatrix",
        "rows": rows,
        "baseline_note": "",
    }
    if estimate is None:
        component["baseline_note"] = (
            "No readmission estimate was requested for this turn, so the table shows "
            "the published cohorts alone."
        )
        return component

    component["baseline"] = {
        "source": _BASELINE_SOURCE,
        # The figures go over RAW and the component formats them, exactly as the widget
        # does — same field, same formatter, same rounding — so the row and the bar
        # beside it cannot print one estimate two ways.
        "probability": float(estimate["probability"]),
        "threshold": float(estimate["threshold"]),
        "decision": int(estimate.get("decision") or 0),
        "cohort": "Index admission, this patient",
        "provenance": f"Model {estimate.get('model_version') or 'unknown'}",
    }
    return component


def compose_risk_canvas(predict: dict | None, rag: dict | None,
                        cite: int = 1, sections: tuple[str, ...] = (),
                        source: dict | None = None,
                        literature: dict | None = None,
                        labs: dict | None = None,
                        medications: dict | None = None,
                        narrative: dict | None = None,
                        phenotype: str = "",
                        query: str = "",
                        command: str | None = None,
                        eligible: bool = True) -> dict:
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

    `phenotype` is the admission's resolved condition, and it is what selects the
    comparative matrix's cohort rows (`admission_phenotypes`, Phase 4B). Empty means the
    admission did not resolve, or resolved to a bucket with no comparison policy, and the
    matrix then draws nothing at all: an unknown condition is the one thing a
    condition-matched table must not guess at.

    `query` is the clinician's own words on a turn that had them, and it is the second
    half of the condition-lock: on a literature turn the comparison is drawn only when
    the query is about the patient's own condition. Asking about heart failure while
    looking at an orthopedic admission draws the records and nothing else — not heart
    failure cohorts pinned under her estimate, and not orthopedic cohorts answering a
    question about heart failure either.
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

    # The lab trajectories, when this turn asked for them. Second in the column and directly
    # under the narrative, because on a drill-down turn the card IS the answer: the prose
    # above is the sentence describing a course, and a reader who has to scroll past anything
    # else to reach the values has been given the conclusion before the evidence.
    #
    # `synthetic` travels with the card rather than being decided by the surface. Every value
    # in it is generated — see `dependencies/labs/synthetic.py` — and the payload states so
    # in its own contract, so a component cannot present a drawing as a measurement without
    # contradicting what it was handed.
    lab_caption = ""
    lab_trends = (labs or {}).get("trends") if isinstance(labs, dict) else None
    if isinstance(lab_trends, list) and lab_trends:
        count = len(lab_trends)
        lab_caption = (
            f"{count} laboratory trajector{'y' if count == 1 else 'ies'} across "
            "this admission."
        )
        children.append("labs")
        components.append({
            "id": "labs",
            "component": "LabTrendCard",
            "trends": lab_trends,
            "synthetic": bool((labs or {}).get("synthetic")),
            "note": str((labs or {}).get("note") or ""),
        })

    estimate = _usable_estimate(predict)
    literature_view = _literature_view(literature)
    if literature_view is not None:
        articles, note, degraded = literature_view
    else:
        articles, note, degraded = [], "", False

    # Selected once, before either surface claims the column, because both draw it: the
    # literature turn above the record list and the risk turn beneath the widget. One
    # selection means the two placements cannot disagree about which cohorts this patient
    # is being compared against.
    #
    # Whether the turn may carry it at all is `_draws_matrix`'s question, and it is asked
    # here rather than at either placement so no surface can draw the table on a turn that
    # did not ask for it.
    comparison = (
        _comparative_matrix(estimate, phenotype)
        if _draws_matrix(command, literature_view is not None, phenotype, query) else None
    )

    if literature_view is not None:
        # The comparison, when this patient's phenotype has curated cohorts. It goes above
        # the records: the patient's position among the published cohorts is the answer to
        # "how does she compare", and the records beneath it are the sources a reader
        # checks that answer against.
        if comparison is not None:
            children.append("comparison")
            components.append(comparison)

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
        #
        # Unless a lab card was drawn, in which case the caption says what the card holds.
        # "No readmission estimate was requested" is true of a lab drill-down and useless to
        # a reader who asked about a value, so the sentence follows the surface.
        fallback = lab_caption or (
            "No 30-day readmission risk estimate was requested for this question.")
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
        #
        # Unless there is a narrative, and as of 2026-10-15 there can be one beside
        # an estimate: `/ehr` draws the readmission chip next to the discharge
        # summary it qualifies, in one turn. For that turn the fallback has to be the
        # summary. The sentence below is the whole answer only when the surface IS
        # the answer — `/risk`, whose widget prints the score and prints it exactly.
        # Handing a reader a bare risk figure in place of the summary they asked for
        # is worse than handing them nothing, because it reads as an answer.
        if narrative is not None:
            fallback = narrative["text"]
        else:
            fallback = (
                f"{shown:.0%} 30-day readmission "
                f"probability ({shown:.2f}), {band} at a {threshold:.2f} threshold."
            )

    # The comparison on a turn with no literature list, which is `/ehr`: the same component
    # under the risk widget instead of above a record list, because the two turns ask
    # different questions. `/pubmed` answers "how does she compare, and what is the
    # evidence", so the cohorts lead and the records follow; `/ehr` answers "what is her
    # risk", so the widget leads and the cohorts qualify it immediately beneath — which is
    # the placement the directive asks for and the reason the matrix had to stop being a
    # literature-turn-only surface.
    #
    # A lab turn draws no comparison at all. A drill-down asks what a measurement did, and
    # a table of published readmission cohorts is an answer to a question nobody asked —
    # the card is the answer and the prose above it describes the course. Without this the
    # trajectories arrived under an empty-baseline matrix on every drill-down, which is
    # what the first composed lab turn did.
    if (literature_view is None and comparison is not None
            and not (isinstance(lab_trends, list) and lab_trends)):
        children.append("comparison")
        components.append(comparison)

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

    # Retired 2026-10-13 (UX ruling): the cited narrative carries the passage text per    # sentence, as an interactive marker the reader opens. The card showed the same
    # evidence a second time, in a second place, disconnected from the sentence it
    # supported — redundant weight next to a better affordance. The comment below is
    # kept because it records why the card appeared only sometimes:
    #
    # Cited source — the passage the answer cites first (mirrors the custom
    # demo). Kept as the LAST child so it can pin to the bottom of the canvas
    # (sticky) and the discharge notes stay in view however long the session
    # gets. No card when there is no passage to cite: an empty citation slot is
    # not evidence of a fault and does not need to say anything.

    # Outpatient suppression (2026-10-18). `/ehr` never drew this widget on an
    # outpatient note because it never ran a prediction there; the explicit `/risk`
    # command did, and rendered a full 30-day inpatient estimate — probability, band and
    # driver bars — for a record with no admission. Suppression is applied here, after
    # composition, because every branch above can add the widget and a second guard in
    # each of them would be five places to keep in step.
    if not eligible:
        components = [c for c in components
                      if c.get("component") != "RiskInfluencers"]
        children = [cid for cid in children if cid != "risk"]
        outpatient_note = (
            "Readmission risk does not apply to this record. It is an outpatient "
            "note, not an inpatient admission, and the 30-day model is defined over "
            "inpatient stays."
        )
        components.append({"id": "risk_outpatient", "component": "Text",
                           "text": outpatient_note, "variant": "h2"})
        children.append("risk_outpatient")
        fallback = outpatient_note

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


def admission_eligible(tool_calls: list[dict]) -> bool:
    """Whether the record this turn is about is an admission at all.

    Narrow on purpose: this is *not* `is_matrix_eligible`, which additionally requires a
    curated phenotype bucket before the comparative table may be drawn. Using that wider
    rule suppressed the risk widget for every admission whose condition had not been
    curated — most of the real corpus — which is a different question from whether an
    inpatient readmission estimate applies to the record.

    Defaults to True when no admission is named: a literature or free-text turn without a
    `hadm_id` is not making an inpatient claim, so there is nothing to suppress.
    """
    for call in tool_calls or []:
        hadm = (call.get("args") or {}).get("hadm_id")
        if isinstance(hadm, int) and not isinstance(hadm, bool):
            return hadm not in NON_ADMISSION
    return True


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
    """The retrieval evidence for this run, as ONE passage array.

    The prompt numbers citations GLOBALLY across calls — "the first call's passages are
    ^[1], ^[2] …; the second call's passages continue after them" — so the array a
    marker indexes has to be that same concatenation. This returned only the FIRST
    retrieval call's payload, which meant a marker pointing into the second call
    resolved against an unrelated passage or clamped to the end of the first
    (Phase 01 audit, 2026-10-22; `/ehr` is the case, with its sectioned call plus the
    record-wide fallback).

    A single call is returned untouched, which is every command turn and most free-text
    ones — the merge only exists for the multi-call turn it was always wrong for.
    """
    rag_calls = [call for call in tool_calls or []
                 if call.get("name") in ("rag_search", "rag_search_sections",
                                         "discharge_medications")
                 and isinstance(call.get("response"), dict)]
    if not rag_calls:
        return None
    if len(rag_calls) == 1:
        return rag_calls[0]["response"]

    merged = dict(rag_calls[0]["response"])
    passages: list = []
    for call in rag_calls:
        passages.extend(call["response"].get("passages") or [])
    merged["passages"] = passages
    merged["returned"] = len(passages)
    return merged


def lab_trend_payload(tool_calls: list[dict]) -> dict | None:
    """The `get_lab_trends` response for this run, or None.

    None for a failed call as well as for no call: the card draws a course, and a course
    with no values in it is not a shorter answer, it is a wrong one. The turn still says
    what happened — the tool's error reaches the model like any other result.
    """
    for call in tool_calls or []:
        if call.get("name") != "get_lab_trends":
            continue
        response = call.get("response") or {}
        if isinstance(response, dict) and not response.get("error"):
            return response
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


#: What a turn's answer rested on. The console labels the turn with this, so a reader
#: never mistakes a claim about the patient for a claim about the literature.
SCOPE_PATIENT = "patient_chart"
SCOPE_GENERAL = "general_evidence"


def turn_scope(tool_calls: list[dict]) -> str | None:
    """What this turn's evidence was, or None when it rested on nothing.

    A tool call that carried a `hadm_id` read the patient's record; a call without one
    did not. That is the whole test, and it is read off the arguments the tools were
    actually given rather than off a list of tool names — so a tool added later is
    classified by what it did rather than by being remembered here.

    None when no tool ran. A redirect ("that is /risk's answer to give") is a complete
    answer that rests on no evidence, and labelling it either way would claim a basis
    it does not have.
    """
    if not tool_calls:
        return None
    for call in tool_calls:
        args = call.get("args") or {}
        if args.get("hadm_id") is not None:
            return SCOPE_PATIENT
    return SCOPE_GENERAL


def turn_phenotype(tool_calls: list[dict]) -> str:
    """The phenotypic bucket this turn's admission resolved to, or "".

    Read off the `hadm_id` the tools were actually given, the same way `turn_scope` reads
    the scope, so a tool added later is classified by what it did. The admission id
    travels as an argument rather than being passed alongside the calls because this
    layer's whole job is to derive the surface from the evidence the turn produced.

    Three ways to get nothing back, and the caller cannot tell them apart — which is
    correct, because the consequence is the same and it is the safe one:

      * no tool carried an admission (a condition-only search, a free-text question);
      * the admission is not in the reviewed map, so its condition is unknown;
      * the admission resolved to a bucket whose policy is no comparison, or is not a
        hospital admission at all.

    The condition-lock is enforced here rather than at the matrix, so a mismatched
    comparison cannot be drawn by any caller that forgets a check: a `/pubmed` turn whose
    query is about a condition this patient does not have still carries `hadm_id` for the
    estimate, and the matrix for it is selected by the PATIENT's phenotype. A query about
    heart failure on an orthopedic admission therefore draws the orthopedic rows beside
    her estimate or, when the query is the point, the records alone — never heart failure
    cohorts pinned under an orthopedic patient.
    """
    for call in tool_calls or []:
        hadm_id = (call.get("args") or {}).get("hadm_id")
        if not isinstance(hadm_id, int) or isinstance(hadm_id, bool):
            continue
        # The first admission the turn used is the one it is about: every call in a turn
        # carries the same one, so there is nothing here to reconcile between calls.
        if not is_matrix_eligible(hadm_id):
            return ""
        bucket = bucket_for(hadm_id)
        return bucket.key if bucket else ""
    return ""


def compose_presentation(question: str, answer: str,
                         tool_calls: list[dict],
                         command_payload: dict | None = None,
                         command: str | None = None,
                         context_grounded: bool = False) -> dict:
    """The full presentation contract for an answered question.

    One call produces every field the browser consumes, so the answer's
    evidence semantics are decided in exactly one place (here) and the BFF is
    a pass-through:

      - answer: the guarded answer with citations renumbered to
        first-appearance order
      - sources: one resolved {cite, section, text, query} per citation in
        the answer — the browser looks a clicked footnote up by number
      - scope: what the answer rested on, or None when it rested on nothing —
        which includes a chart read that resolved no note citation, reported as
        `scope_suppressed` rather than asserted over prose it cannot evidence
      - a2ui: the composed risk-canvas envelope; its SourceCard is sources[0]

    citation_map and intent_sections are inputs to that resolution and are
    returned for tests and diagnostics; the browser does not need them.

    `command_payload` is a command's own structured result when it has one — `/meds`
    extracts rows the canvas draws as a grid. It is passed through rather than
    recomposed, because the rows were verified against the note text in the command
    and re-deriving them here would be a second implementation of that check.

    `context_grounded` is the caller saying this turn also rests on the patient record
    through a channel this layer cannot see: the pre-flight manifest that answered an
    administrative question, or the replayed turns' chart calls that a follow-up is
    quoting. It feeds `verified` and nothing else — evidence the composer never saw
    cannot change what it draws, but it does change whether the answer was checked
    against the record.

    Pure — no I/O, no clock, no randomness — so it is fully testable without
    a browser or a live endpoint.
    """
    renumbered = renumber_citations(answer)
    remap = citation_remap(answer)
    intent = intent_sections(question)
    rag = rag_payload(tool_calls)
    literature = literature_payload(tool_calls)
    labs = lab_trend_payload(tool_calls)
    sources = resolve_sources(renumbered, remap, rag, intent)
    narrative = _narrative(renumbered, sources, literature)

    scope = turn_scope(tool_calls)

    # A turn that retrieved records answers from them, so it is general medical evidence
    # even when a supplementary call also fetched this patient's own estimate (Phase 2,
    # 2026-10-16). The badge exists so a clinician never mistakes published literature for
    # something the chart supports; badging a comparator turn "Patient Chart Synthesis"
    # would claim the notes support a sentence drawn from JAMA. The one patient figure such
    # a turn states is attributed to the model in the table beside it, not to a passage.
    #
    # Without this, `/pubmed` read as a chart read that resolved no note citation — the
    # suppression rule immediately below — so every comparative turn dropped its scope pill
    # AND raised a visible safety swap over an answer nothing had been removed from. A
    # false correction is worse than none: it teaches a reader to skip the badge.
    if scope == SCOPE_PATIENT and _usable_articles(literature):
        scope = SCOPE_GENERAL

    # A chart read that resolved no note citation has nothing to substantiate the
    # claim. The scope says what the answer rests on, not that a chart tool ran, so
    # it is withheld here rather than asserted over prose it cannot evidence — the
    # same rule that gives a redirect no scope at all. Reported alongside it so the
    # BFF can carry the suppression into `guardrail_flags` instead of it ending
    # silently at this boundary.
    #
    # A lab card is the third channel, and it substantiates too (Phase 3, 2026-10-16). A
    # drill-down turn reads the patient's chart — `get_lab_trends` takes an admission, so
    # the scope is `patient_chart` and that is the right badge — but its evidence is a
    # trajectory rather than a passage, so `sources` is empty and the rule above would have
    # withheld the scope AND raised a visible safety swap over an answer nothing had been
    # removed from. The same false correction the comparator turn produced, from the same
    # cause: a channel added after the rule was written.
    lab_evidenced = bool((labs or {}).get("trends"))
    # A command's own structured result is a fourth channel (2026-10-18). `/risk` answers
    # from the model estimate and `/meds` from an extraction verified against the note,
    # so `sources` is empty on both by construction — not because a citation failed to
    # resolve — and the rule above withheld the scope and raised a visible safety swap
    # over an answer nothing had been removed from. Identical false positive to the
    # comparator and lab turns, from the identical cause: a channel added after the rule
    # was written.
    #
    # Deliberately limited to the commands whose payload IS their evidence. `/ehr` also
    # carries a payload (the estimate it compares against) but its prose is a note
    # synthesis, so an uncited `/ehr` answering is still the thing this flag exists to
    # report.
    payload_evidenced = bool(command_payload) and command in ("risk", "meds")
    # A widget-only turn has no prose to evidence. Flagging "synthesis without
    # citations" against an empty sentence describes an omission that did not happen:
    # the canvas is the answer, and the reader was never shown a claim needing
    # attribution.
    no_prose = not (renumbered or "").strip()
    scope_suppressed = (scope == SCOPE_PATIENT and not sources
                        and not lab_evidenced and not payload_evidenced
                        and not no_prose)
    if scope_suppressed:
        scope = None

    # Whether this turn is DRAWN as a literature turn.
    #
    # Two ways it is: the search returned records, or it returned nothing and there is
    # no patient-record evidence to show instead. The second case is a real one —
    # `/pubmed` about a condition this patient does not have still owes the reader the
    # records and the sentence saying none matched.
    #
    # What it must never be is "a dict arrived". An empty result beside a chart answer
    # swapped the canvas to an empty `LiteratureList` and dropped the note citations, so
    # a correct chart answer arrived under a card saying no literature was found about a
    # question nobody asked of the literature (2026-10-22).
    articles = _usable_articles(literature or {})
    draws_literature = bool(articles) or (
        literature is not None and not sources and not lab_evidenced
        and not payload_evidenced
    )
    drawn_literature = literature if draws_literature else None

    # Whether the answer was verified against the patient record.
    #
    # The rule is grounding, not citation. A `/labs` answer rests on a trajectory, a
    # `/risk` answer on the model's own estimate, `/meds` on an extraction checked
    # against the note — none of them carries a `^[n]`, and requiring one would withhold
    # the badge from the turns that are most tightly bound to the chart. A pure PubMed
    # turn is the case it must NOT cover: those records are somebody else's patients.
    #
    # The guardrail's own verdict is ANDed by the caller (`http`), because only that
    # layer knows whether the answer survived it or was replaced by an abstention.
    # The estimate is a fifth channel, and it is the one `/risk` has: a widget-only turn
    # carries no prose and no passage, and its evidence is the model's own scoring of
    # this admission. A driver question reaches the same place from free text.
    estimate_evidenced = _usable_estimate(predict_payload(tool_calls)) is not None
    grounded_in_record = (
        bool(sources) or lab_evidenced or payload_evidenced or estimate_evidenced
        or bool(context_grounded)
    )
    # A literature turn is about other people's patients, whatever rides beside it — the
    # comparator turn draws this patient's estimate as a baseline row and is still an
    # answer about published cohorts. Note sources are the exception: a turn that cites
    # the record is grounded in it, whichever surface it drew.
    literature_only = draws_literature and not sources
    # A widget IS an answer. `/risk` returns no prose by design, so "substantive" cannot
    # mean "has prose" — it means there is something the reader was given that can be
    # vouched for, and a drawn estimate is that.
    substantive = (not no_prose) or payload_evidenced or estimate_evidenced
    verified = bool(
        grounded_in_record and substantive and not scope_suppressed
        and not literature_only
    )

    return {
        "answer": renumbered,
        "citation_map": remap,
        "intent_sections": list(intent),
        "sources": sources,
        # The published records the answer rests on, passed through whole. Kept
        # separate from `sources` on purpose: that list is the discharge-note
        # citation channel — numbered, renumbered, and resolved against retrieved
        # passages — and an article has no passage to resolve against.
        "literature": drawn_literature,
        # Whether this answer was checked against, and grounded in, the patient's own
        # record. Stated once, here, because the turns that qualify are decided by
        # clinical rules and a browser re-deriving them would be a second opinion.
        "verified": verified,
        # What the turn rested on, and the rows the console's references drawer lists.
        # Derived here rather than in the browser because this is the layer that saw the
        # tool calls and resolved the citations; a second derivation downstream would be
        # a second answer to the same question, and the two would drift.
        #
        # `references` is the narrative's own citation list, so the marker a reader
        # clicks and the row they find at the foot of the turn are the same record by
        # construction rather than by two lists agreeing today.
        "scope": scope,
        "scope_suppressed": scope_suppressed,
        "references": list((narrative or {}).get("citations") or []),
        "a2ui": compose_risk_canvas(
            predict_payload(tool_calls),
            rag,
            cite=first_citation(renumbered),
            sections=intent,
            source=sources[0] if sources else None,
            literature=drawn_literature,
            labs=labs,
            medications=command_payload,
            # The narrative chooses its citations from this turn's evidence: the
            # articles for a literature turn, the resolved note passages otherwise.
            narrative=narrative,
            # The patient's own condition selects the cohort rows, whatever the
            # question was about (see `turn_phenotype`), and `command` decides whether
            # this turn is one that carries the table at all (see `_draws_matrix`).
            phenotype=turn_phenotype(tool_calls),
            eligible=admission_eligible(tool_calls),
            # The clinician's own words, when the turn had any: the other half of the
            # condition-lock, so a question about a condition this patient does not have
            # is answered with records and no comparison.
            query=str((literature or {}).get("query") or "") if isinstance(literature, dict) else "",
            command=command,
        ),
    }
