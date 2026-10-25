"""The risk canvas: what a turn's evidence is drawn as.

The A2UI v0.9 envelope for an answer — the estimate's band, the comparative cohort
table on a literature turn, the medication grid, the lab trajectories, the note
references, and the ordering and unavailable-states that decide which of them a
given turn actually gets.

Split out of `a2ui.py`, which is the composer. This module answers "what does this
turn look like"; `a2ui` answers "which calls were this turn's evidence, and what
goes on the wire". Everything here is derived from payloads that were already
extracted, so the module takes no toolbox and no state: `compose_risk_canvas`
receives the estimate, the retrieval payload and the command, and draws.

`_usable_articles` is imported from `note_presentation` rather than living here,
because what may be cited is a rule about the evidence, and this module is a
consumer of it.
"""

from services.agent.a2ui_envelope import (
    A2UI_VERSION,
    AUDIENCE,
    CATALOG_ID,
    SURFACE_ID,
)
from services.agent.admission_phenotypes import phenotype_of_query
from services.agent.feature_labels import display_probability, humanize
from services.agent.literature_benchmarks import benchmarks_for_phenotype
from services.agent.note_presentation import _usable_articles

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
