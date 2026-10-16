"""The canvas swap: a literature answer draws articles, not a risk score.

This is the feature's whole visible behaviour, and it is pure — no browser, no
endpoint, no clock — so it is asserted here rather than eyeballed in a page. The
regression half matters as much as the new half: the risk canvas must look exactly
as it did before when no search ran.
"""

from services.agent.a2ui import (
    _narrative,
    _usable_articles,
    compose_presentation,
    compose_risk_canvas,
    literature_citations,
    literature_payload,
)

ARTICLE = {
    "pmid": "31234567",
    "title": "Ceftriaxone for lower extremity edema",
    "journal": "N Engl J Med",
    "pub_date": "2025 Mar 12",
    "abstract": "BACKGROUND: Cellulitis is common.",
    "url": "https://pubmed.ncbi.nlm.nih.gov/31234567/",
}

PREDICT = {
    "hadm_id": 90000009,
    "probability": 0.2,
    "threshold": 0.11,
    "decision": 1,
    "base_value": 0.05,
    "top_factors": [{
        "feature": "prior_inpatient_days",
        "contribution": 0.2,
        "direction": "increases",
    }],
    "model_version": "model-x",
    "feature_source": "synthetic",
}


def _search(**overrides):
    payload = {"query": "ceftriaxone cellulitis", "returned": 1, "articles": [ARTICLE]}
    payload.update(overrides)
    return payload


def _call(payload):
    return [{"name": "search_literature", "args": {"query": "x"}, "response": payload}]


def _components(envelope):
    return envelope["messages"][1]["updateComponents"]["components"]


def _types(envelope):
    return [component["component"] for component in _components(envelope)]


def _texts(envelope):
    return [c.get("text", "") for c in _components(envelope)]


def _note(envelope):
    rendered = next(c for c in _components(envelope) if c["component"] == "LiteratureList")
    return rendered["note"]


# --- the comparative matrix (Phase 4B, 2026-10-16) ---------------------------

# Selected by the admission's resolved phenotype, not by matching the PMIDs a search
# returned. The fixtures that used to drive these tests with retrieved records are gone
# on purpose: under the old rule the records decided whether the table drew, which is the
# defect the change removed — no live search ever returned a curated PMID, so the matrix
# never rendered online and its one verified appearance came from a fixture capture.
#
# A phenotype name is therefore the input, and the absence of one is a case with its own
# test: an unresolved admission must draw no comparison rather than a guessed one.


def _matrix(envelope):
    return next(c for c in _components(envelope)
                if c["component"] == "ComparativeRiskMatrix")


def test_a_resolved_phenotype_draws_the_comparison_above_the_records():
    """Order asserted, not merely presence.

    The value is relational — a table of cohort rates below a list of citations reads as
    an appendix to them — so "it appears" would pass while the feature missed. The first
    three, not the whole list: the provenance caption trails both and is not the subject.
    """
    envelope = compose_risk_canvas(
        PREDICT, None, literature=_search(),
        phenotype="heart_failure_cardiorenal",
        query="congestive heart failure 30-day readmission risk",
        command="literature")

    assert _types(envelope)[:3] == ["Column", "ComparativeRiskMatrix", "LiteratureList"]


def test_the_condition_lock_refuses_a_query_about_another_condition():
    """The directive's clinical-integrity rule, as behaviour.

    A clinician reading about heart failure while looking at an orthopedic admission asked
    about heart failure. Pinning that patient's estimate beside orthopedic cohorts answers a
    question they did not ask; pinning it beside heart failure cohorts is worse. So neither
    is drawn, and the records are — which is what the turn is actually about.

    Before the lock, this exact turn drew the patient's own bucket: an orthopedic cohort
    table under a question about heart failure.
    """
    envelope = compose_risk_canvas(
        PREDICT, None, literature=_search(),
        phenotype="orthopedic_arthroplasty",
        query="congestive heart failure 30-day readmission risk")

    assert "ComparativeRiskMatrix" not in _types(envelope)
    assert "LiteratureList" in _types(envelope)


def test_a_query_naming_no_condition_draws_no_comparison():
    """An unrecognised subject is not a licence to assume agreement.

    The gate opens on a match, so a question the vocabulary cannot place draws the
    records alone. The alternative — treating "no recognisable condition" as "same
    condition" — would make every unclassifiable query an unguarded one.
    """
    envelope = compose_risk_canvas(
        PREDICT, None, literature=_search(),
        phenotype="orthopedic_arthroplasty",
        query="what should we watch for after discharge?")

    assert "ComparativeRiskMatrix" not in _types(envelope)


def test_a_lab_turn_draws_no_cohort_table():
    """A drill-down asks what a measurement did, so the card is the whole answer.

    The first composed lab turn drew the trajectories under an empty-baseline matrix — the
    phenotype resolves from the admission, so a lab turn looked like a risk turn to the
    selector. A table of published readmission rates beside a haemoglobin course is an
    answer to a question nobody asked.
    """
    labs = {
        "hadm_id": 90000015, "synthetic": True, "returned": 1,
        "note": "Generated from this admission's features.",
        "trends": [{
            "key": "rdw", "label": "RDW", "unit": "%",
            "reference_low": 11.5, "reference_high": 14.5,
            "anchored": ["rdw_max"],
            "series": [{"day": 0.0, "value": 12.0, "flag": "normal"},
                       {"day": 4.0, "value": 15.6, "flag": "high"}],
        }],
    }
    envelope = compose_risk_canvas(
        None, None, labs=labs, phenotype="orthopedic_arthroplasty")

    assert "ComparativeRiskMatrix" not in _types(envelope)
    assert "LabTrendCard" in _types(envelope)


def test_the_table_draws_with_no_search_in_the_turn_at_all():
    """The change, stated as behaviour: `/ehr` gets the matrix without a literature query.

    This is the case that did not work before. The numerator used to require a retrieved
    record with a curated counterpart, so the unified briefing — narrative, risk widget,
    cohorts, in one turn — could not draw the third part of itself.
    """
    envelope = compose_risk_canvas(PREDICT, None, phenotype="heart_failure_cardiorenal",
                                   command="summarize")

    assert "ComparativeRiskMatrix" in _types(envelope)
    assert "LiteratureList" not in _types(envelope)


def test_the_pinned_row_carries_the_patients_figures_and_its_drivers():
    matrix = _matrix(compose_risk_canvas(
        PREDICT, None, phenotype="heart_failure_cardiorenal", command="summarize"))

    baseline = matrix["baseline"]
    # Raw, so the component formats them with the same formatter the widget uses.
    # Pre-rendering them here would be a second implementation of the display
    # precision, and the two would part on the first change to either.
    assert baseline["probability"] == 0.2
    assert baseline["threshold"] == 0.11
    assert baseline["drivers"] == ["prior inpatient days"]
    assert baseline["provenance"] == "Model model-x"
    assert "XGBoost" not in str(baseline), "the learner is not in the payload"

    row = matrix["rows"][0]
    assert row["pmid"] == "37495280"
    assert row["reported_rate"] == "13.2%"
    assert row["horizon_days"] == 30
    assert row["source"] == "Journal of the American College of Cardiology · 2023"
    # The excerpt IS the sentence the rate was read from, so the figure and its
    # provenance are the same text and neither can drift from the other.
    assert row["excerpt"].startswith("Of the 1.5 million individuals at risk")
    assert "13.2%" in row["excerpt"]


def test_the_orbopedic_bucket_now_has_a_row():
    """Cynthia Petrov's bucket, which had none.

    The directive's headline case: an orthopedic patient's estimate had no cohort to sit
    beside, which is why the matrix either drew nothing or drew heart failure literature.
    """
    matrix = _matrix(compose_risk_canvas(
        PREDICT, None, phenotype="orthopedic_arthroplasty", command="summarize"))

    row = matrix["rows"][0]
    assert row["pmid"] == "34793857"
    assert row["reported_rate"] == "3.4%"
    assert row["evidence_tier"] == "Retrospective Cohort"
    assert row["source"] == "The Journal of Arthroplasty · 2022"
    # The cohort's own driver is prior non-elective utilisation, which this patient's
    # row measures as ED visits and admissions — so the pills are populated, and they
    # carry the same names the driver bars carry.
    assert row["matched_features"] == ["recent ED visits", "prior admissions"]


def test_a_matched_driver_is_named_the_way_the_bars_name_it():
    """One factor, one name — the rule the label map exists for, applied to a pill.

    The cohort says "number of hospitalizations in the previous year"; the pill says what
    the driver bar beside the patient says, because both go through `humanize`.
    """
    matrix = _matrix(compose_risk_canvas(PREDICT, None, phenotype="copd_respiratory",
                                         command="summarize"))

    assert matrix["rows"][0]["matched_features"] == ["prior inpatient days"]


def test_rows_are_ordered_strongest_evidence_first():
    """COPD has three rows; the pooled estimate leads and the narrative review trails.

    Order is asserted because the table is a comparison: a single-centre series above a
    meta-analysis reads as the stronger evidence, and the reader has no way to tell that
    the order was incidental.
    """
    matrix = _matrix(compose_risk_canvas(PREDICT, None, phenotype="copd_respiratory",
                                         command="summarize"))

    assert [row["evidence_tier"] for row in matrix["rows"]] == [
        "Meta-Analysis", "Narrative Review"]


def test_a_phenotype_with_no_curated_cohort_draws_no_comparison():
    """Psychiatric is a bucket with no rows, and the table must simply not appear.

    The alternative — reaching for a bucket that does have rows — is the whole failure
    this phase exists to remove, so it is asserted rather than left to the selector's
    implementation.
    """
    envelope = compose_risk_canvas(
        PREDICT, None, literature=_search(), phenotype="psychiatric")

    assert "ComparativeRiskMatrix" not in _types(envelope)
    assert "LiteratureList" in _types(envelope)


def test_an_unresolved_admission_draws_no_comparison():
    """No phenotype is not a licence to guess one."""
    envelope = compose_risk_canvas(PREDICT, None, literature=_search())

    assert "ComparativeRiskMatrix" not in _types(envelope)


def test_the_table_draws_without_an_estimate_and_says_which_it_is():
    """A patient-less turn still has cohorts worth reading.

    What it must not do is draw the pinned row with a blank figure in it, so the row is
    absent and a sentence stands where it would have been.
    """
    matrix = _matrix(compose_risk_canvas(
        None, None, phenotype="heart_failure_cardiorenal", command="summarize"))

    assert "baseline" not in matrix
    assert matrix["baseline_note"].strip()
    assert "No readmission estimate" in matrix["baseline_note"]


def test_a_long_journal_name_is_cut_to_the_marker_budget():
    """A pill sits inside a sentence, so its label has a length budget (2026-10-16).

    Reported from the live turn: a button wraps its own text by default, so "Journal of the
    American College of Cardiology" broke the pill's box across two lines and the rounded
    border read as a pill cut in half. The journal was kept and bounded rather than
    replaced by the identifier again — the full title is on the marker's popover and in the
    record list beneath the answer.
    """
    citations = literature_citations([
        {**ARTICLE, "journal": "Journal of the American College of Cardiology"},
        {**ARTICLE, "journal": "Home healthcare now"},
    ])

    long_label = citations[0]["label"]
    assert len(long_label) <= 25, long_label          # 24 characters, plus the ellipsis
    # Marked, not silently cut: an unmarked cut reads as the whole name.
    assert long_label.endswith("…")
    assert "Journal of the American College of Cardiology".startswith(
        long_label[:-1].rstrip())
    # A name inside the budget is passed through whole and unmarked.
    assert citations[1]["label"] == "Home healthcare now"


# --- selecting the payload -------------------------------------------------

def test_literature_payload_finds_the_search():
    assert literature_payload(_call(_search()))["query"] == "ceftriaxone cellulitis"


def test_literature_payload_is_none_when_no_search_ran():
    assert literature_payload([{"name": "rag_search", "response": {}}]) is None
    assert literature_payload([]) is None
    assert literature_payload(None) is None


def test_literature_payload_keeps_a_failed_search():
    """A failure must not be flattened into the same shape as an empty result.

    "the tool did not answer" and "the tool answered that nothing exists" are
    different statements to a clinician, and a selector that returned None for both
    would leave the canvas unable to tell them apart.
    """
    failed = {"error": "literature_failed", "message": "PubMed could not be reached."}

    assert literature_payload(_call(failed)) == failed


# --- shaping the records ---------------------------------------------------

def test_articles_drop_a_record_with_no_pmid():
    payload = _search(articles=[ARTICLE, {**ARTICLE, "pmid": ""}])

    assert [a["pmid"] for a in _usable_articles(payload)] == ["31234567"]


def test_articles_coerce_missing_fields_rather_than_omitting_them():
    """The console's component schema is strict, so a missing key drops the card."""
    payload = _search(articles=[{"pmid": "1", "title": "Only a title"}])

    article = _usable_articles(payload)[0]
    assert set(article) == {"pmid", "title", "journal", "pub_date", "url"}
    assert article["journal"] == ""


def test_articles_is_empty_for_a_payload_that_is_not_one():
    for payload in (None, {}, {"articles": "not a list"}, {"articles": [None, 7]}):
        assert _usable_articles(payload) == []


# --- the swap --------------------------------------------------------------

def test_a_literature_answer_draws_articles_instead_of_the_risk_view():
    envelope = compose_risk_canvas(None, None, literature=_search())

    assert "LiteratureList" in _types(envelope)
    assert "RiskInfluencers" not in _types(envelope)


def test_the_records_reach_the_canvas_as_the_tool_returned_them():
    """The citation details are the record's own, so the row a reader checks and the
    summary they are checking it against are the same publication."""
    envelope = compose_risk_canvas(None, None, literature=_search())

    rendered = next(c for c in _components(envelope) if c["component"] == "LiteratureList")
    record = rendered["articles"][0]
    assert record["pmid"] == ARTICLE["pmid"]
    assert record["title"] == ARTICLE["title"]
    assert record["journal"] == ARTICLE["journal"]
    assert record["pub_date"] == ARTICLE["pub_date"]
    assert record["url"].endswith("31234567/")


def test_the_abstract_is_not_put_on_the_wire_at_all():
    """Nothing renders it since the marker became a link to the source, and a field
    no component draws is a promise the schema is no longer keeping — the records it
    was duplicated into carried the same text a second time."""
    envelope = compose_risk_canvas(None, None, literature=_search())

    rendered = next(c for c in _components(envelope) if c["component"] == "LiteratureList")
    assert "abstract" not in rendered["articles"][0]
    assert "excerpt" not in rendered["articles"][0]


def test_the_risk_view_is_unchanged_without_literature():
    envelope = compose_risk_canvas(PREDICT, None)

    assert "RiskInfluencers" in _types(envelope)
    assert "LiteratureList" not in _types(envelope)


def test_literature_takes_the_canvas_over_an_estimate_in_the_same_turn():
    """The swap is the requested behaviour, so it wins a tie rather than merging."""
    envelope = compose_risk_canvas(PREDICT, None, literature=_search())

    assert "LiteratureList" in _types(envelope)
    assert "RiskInfluencers" not in _types(envelope)


def test_a_literature_turn_does_not_claim_no_note_passage_was_found():
    """There is no note citation on this turn; saying so would read as a failure."""
    envelope = compose_risk_canvas(None, None, literature=_search())

    assert "SourceCard" not in _types(envelope)


def test_provenance_names_pubmed_and_the_query():
    envelope = compose_risk_canvas(None, None, literature=_search())

    provenance = " ".join(_texts(envelope))
    assert "PubMed" in provenance
    assert "ceftriaxone cellulitis" in provenance
    assert "No model" not in provenance


def test_an_empty_search_still_takes_the_canvas_and_says_so():
    """A search that ran and matched nothing must not leave the old view standing.

    The clinician clicked a chip. A canvas that did not change reads as a click that
    did nothing, and the note is the only thing that distinguishes "no results" from
    "no search". This test previously asserted the opposite — that an empty search
    left the risk path alone — which was the ambiguity the wrapper object exists to
    remove, reintroduced one layer down.
    """
    envelope = compose_risk_canvas(
        None, None,
        literature=_search(returned=0, articles=[], note="No articles matched."),
    )

    assert "LiteratureList" in _types(envelope)
    rendered = next(c for c in _components(envelope) if c["component"] == "LiteratureList")
    assert rendered["articles"] == []
    assert rendered["note"] == "No articles matched."


def test_an_empty_search_falls_back_to_a_sentence_of_its_own():
    """The tool does not always supply a note; the canvas still has to say something."""
    envelope = compose_risk_canvas(None, None, literature=_search(returned=0, articles=[]))

    rendered = next(c for c in _components(envelope) if c["component"] == "LiteratureList")
    assert rendered["note"] == "No matching literature was found for this query."


def test_no_search_at_all_leaves_the_risk_path_alone():
    """None is not the same as empty: nothing was asked, so nothing swaps.

    This is the other half of the distinction — an empty result takes the canvas,
    an absent one does not.
    """
    envelope = compose_risk_canvas(None, None)

    assert "LiteratureList" not in _types(envelope)


# --- the prose, and the records its markers point at --------------------------

def _second():
    return {**ARTICLE, "pmid": "2", "title": "Second record"}


def test_citations_are_numbered_in_the_order_the_model_was_given_the_records():
    """The marker and the card beneath it are the same record by construction, so a
    marker with nothing behind it can only mean the model cited a number it was
    never given."""
    citations = literature_citations([ARTICLE, _second()])

    assert [c["index"] for c in citations] == [1, 2]
    assert [c["title"] for c in citations] == [ARTICLE["title"], "Second record"]
    assert citations[0]["source"] == "PubMed"
    assert citations[0]["url"] == ARTICLE["url"]


def test_a_citation_carries_the_pmid_its_marker_displays():
    """The marker shows the identifier, so the citation has to carry it: an index
    that the console had to resolve back to a record would be one mapping more than
    the thing needs."""
    citations = literature_citations([ARTICLE, _second()])

    assert [c["pmid"] for c in citations] == [ARTICLE["pmid"], "2"]
    assert citations[1]["pmid"] == _second()["pmid"]


def test_a_citation_carries_no_abstract():
    """It was the popover's excerpt. The popover no longer opens on this path, and the
    text travelled as a second copy of what the record already carried."""
    citation = literature_citations([ARTICLE])[0]

    assert "excerpt" not in citation


def test_a_citation_with_no_pmid_still_carries_its_title():
    """The key is left out rather than sent empty, so the component's marker falls
    back to the number instead of linking to a record called ""."""
    citation = literature_citations([{"pmid": "", "title": "A record"}])[0]

    assert "pmid" not in citation
    assert citation["title"] == "A record"
    assert citation["index"] == 1


def test_a_record_with_no_url_carries_no_url_key():
    """The component's schema is strict: an empty string would be a link to nowhere
    and an explicit null would take the narrative down with it."""
    citation = literature_citations([{"pmid": "1", "title": "Only a title"}])[0]

    assert "url" not in citation
    assert citation["pmid"] == "1"
    assert citation["title"] == "Only a title"


def test_a_record_with_no_title_is_named_rather_than_left_blank():
    assert literature_citations([{"pmid": "1"}])[0]["title"] == "Untitled record"


def test_a_record_with_no_pmid_is_not_cited():
    """The canvas drops it, so citing it would open a popover for a card that is
    not on the page."""
    citations = literature_citations(_usable_articles(_search(
        articles=[ARTICLE, {**ARTICLE, "pmid": ""}],
    )))

    assert [c["index"] for c in citations] == [1]


def test_a_literature_summary_is_cited_to_its_articles_not_to_the_note():
    """The turn read no note passage. Resolving its markers against note passages
    left every marker in a properly grounded summary unbacked — which the component
    draws as a disabled control, on the summary that was the one thing working
    (live defect, 2026-10-13)."""
    sources = [{"cite": 1, "section": "hospital_course", "text": "a note passage"}]

    narrative = _narrative("Day-case surgery had lower odds ^[1].", sources, _search())

    assert [c["source"] for c in narrative["citations"]] == ["PubMed"]
    assert narrative["citations"][0]["title"] == ARTICLE["title"]
    assert narrative["citations"][0]["pmid"] == ARTICLE["pmid"]


def test_a_note_turn_still_cites_the_note():
    narrative = _narrative(
        "He recovered ^[1].",
        [{"cite": 1, "section": "hospital_course", "text": "Recovered well"}],
        None,
    )

    assert [c["source"] for c in narrative["citations"]] == ["Discharge note"]


def test_no_prose_is_no_narrative_at_all():
    """An empty narrative draws an empty prose block under the records."""
    assert _narrative("   \n ", [{"cite": 1, "section": "x", "text": "y"}],
                      _search()) is None
    assert _narrative("", [], None) is None


def test_an_empty_search_still_draws_a_narrative_with_nothing_cited():
    """The sentence says the search found nothing. There is no evidence to open, and
    a citation invented to keep the shape symmetrical is the one thing this feature
    must not do."""
    narrative = _narrative("The search returned no records.", [],
                           _search(returned=0, articles=[]))

    assert narrative["text"] == "The search returned no records."
    assert narrative["citations"] == []
    assert narrative["streaming"] is False


def test_the_canvas_root_is_a_column_and_the_prose_leads_it():
    """Ruled flat: the prose sits under the root beside the records rather than in a
    card of its own, and above them — a reader who scrolls past the records to find
    the sentence that explains them has been given the parts in the wrong order."""
    narrative = _narrative("Day-case surgery had lower odds ^[1].", [], _search())
    envelope = compose_risk_canvas(None, None, literature=_search(),
                                   narrative=narrative)

    root = _components(envelope)[0]

    assert root["id"] == "root"
    assert root["component"] == "Column"
    assert root["children"][0] == "narrative"
    assert "literature" in root["children"]
    assert envelope["renders_prose"] is True


def test_the_narrative_is_still_drawn_when_there_is_no_records_list_to_lead():
    """`/meds` and `/ehr` compose a narrative with no search behind them, and the
    flag is what stops the console writing the same answer into its own element a
    second time."""
    narrative = _narrative("Nine discharge medications.", [], None)
    envelope = compose_risk_canvas(None, None, narrative=narrative)

    assert _components(envelope)[0]["children"][0] == "narrative"
    assert envelope["renders_prose"] is True
    assert compose_risk_canvas(None, None)["renders_prose"] is False


# --- the whole presentation ------------------------------------------------

def test_compose_presentation_carries_literature_beside_the_note_sources():
    presentation = compose_presentation(
        "search pubmed for recent literature", "One article.", _call(_search())
    )

    assert presentation["literature"]["query"] == "ceftriaxone cellulitis"
    assert presentation["sources"] == []
    assert "LiteratureList" in _types(presentation["a2ui"])


def test_compose_presentation_reports_no_literature_when_none_ran():
    presentation = compose_presentation("what is the risk?", "0.2.", [])

    assert presentation["literature"] is None
    assert "LiteratureList" not in _types(presentation["a2ui"])


# --- degradation -----------------------------------------------------------


def test_unavailable_and_not_found_are_different_sentences():
    """Only one of these is true, and the clinician is deciding what to believe.

    "The search found nothing" and "the search did not happen" are different claims
    about the world. Rendering them identically is the same defect as an empty list
    that cannot distinguish "no search" — just further along.
    """
    unavailable = compose_risk_canvas(
        None, None,
        literature={"query": "q", "returned": 0, "articles": [], "degraded": True},
    )
    not_found = compose_risk_canvas(
        None, None,
        literature={"query": "q", "returned": 0, "articles": []},
    )

    assert _note(unavailable) == "PubMed literature search is temporarily unavailable."
    assert _note(not_found) == "No matching literature was found for this query."


def test_a_degraded_search_still_takes_the_canvas():
    """The clinician clicked a chip, so something has to change on screen."""
    envelope = compose_risk_canvas(
        None, None,
        literature={"query": "q", "returned": 0, "articles": [], "degraded": True},
    )

    assert "LiteratureList" in _types(envelope)
    rendered = next(c for c in _components(envelope) if c["component"] == "LiteratureList")
    assert rendered["degraded"] is True
    assert rendered["articles"] == []


def test_a_normal_search_is_not_marked_degraded():
    envelope = compose_risk_canvas(None, None, literature=_search())

    rendered = next(c for c in _components(envelope) if c["component"] == "LiteratureList")
    assert rendered["degraded"] is False


def test_an_empty_search_is_not_marked_degraded():
    """A search that ran is not a search that failed."""
    envelope = compose_risk_canvas(
        None, None, literature={"query": "q", "returned": 0, "articles": []}
    )

    rendered = next(c for c in _components(envelope) if c["component"] == "LiteratureList")
    assert rendered["degraded"] is False


def test_a_tool_error_payload_degrades_the_canvas_rather_than_vanishing():
    """An error can still reach here — a blank query, which is our own defect.

    A payload crossed the boundary, so the canvas accounts for it rather than
    silently falling through to a view that does not mention the search at all.
    """
    envelope = compose_risk_canvas(
        None, None, literature={"error": "bad_request", "message": "Blank query."}
    )

    assert "LiteratureList" in _types(envelope)
    assert _note(envelope) == "PubMed literature search is temporarily unavailable."


def test_provenance_says_unavailable_rather_than_counting_records():
    envelope = compose_risk_canvas(
        None, None,
        literature={"query": "q", "returned": 0, "articles": [], "degraded": True},
    )

    assert any("unavailable" in text for text in _texts(envelope))
    assert not any("0 record" in text for text in _texts(envelope))
