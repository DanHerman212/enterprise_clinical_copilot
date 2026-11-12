"""The canvas swap: a literature answer draws articles, not a risk score.

This is the feature's whole visible behaviour, and it is pure — no browser, no
endpoint, no clock — so it is asserted here rather than eyeballed in a page. The
regression half matters as much as the new half: the risk canvas must look exactly
as it did before when no search ran.
"""

from services.agent.presentation.a2ui import _narrative, compose_presentation, compose_risk_canvas, literature_citations, literature_payload, note_citations
from services.agent.presentation.domain import _usable_articles

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


# --- the comparative matrix --------------------------------------------------
#
# The rows are the comparators the ENGINE verified on the turn: retrieved for this
# admission's own condition, each figure checked against the sentence it was read from
# (`deterministic.verify_comparator`). Until Stage Gate 2 they were selected here, by a
# per-admission phenotype map, from a curated corpus — which is why "no live search ever
# returned a curated PMID" and the table never rendered online. The fixtures below are
# therefore comparators in the shape verification emits, and the canvas's job is now the
# narrow one: draw what it was given, or say what is missing.

HEART_FAILURE_COMPARATOR = {
    "pmid": "37495280",
    "title": "Global Comparison of Readmission Rates.",
    "authors": "Smith JA, Doe B.",
    "journal": "Journal of the American College of Cardiology",
    "pub_date": "2023 Aug 01",
    "reported_rate": "13.2%",
    "rate_kind": "point",
    "cohort_summary": "Pooled international cohorts of hospitalised heart failure.",
    "evidence_tier": "Meta-Analysis",
    "rate_sentence": "Of the 1.5 million individuals at risk, 13.2% were readmitted "
                     "within 30 days.",
    "url": "https://pubmed.ncbi.nlm.nih.gov/37495280/",
}

ORTHO_COMPARATOR = {
    "pmid": "34793857",
    "title": "Pre-operative healthcare utilization and readmission after "
             "total joint arthroplasty.",
    "authors": "Creager AE, Kleven AD, et al.",
    "journal": "The Journal of Arthroplasty",
    "pub_date": "2022 Mar 12",
    "reported_rate": "3.4%",
    "rate_kind": "point",
    "cohort_summary": "One institution, 2015-2019, primary total joint arthroplasty.",
    "evidence_tier": "Retrospective Cohort",
    "rate_sentence": "The 30-day readmission rate was 3.4%.",
    "url": "https://pubmed.ncbi.nlm.nih.gov/34793857/",
}

REVIEW_COMPARATOR = {
    **ORTHO_COMPARATOR,
    "pmid": "31234567",
    "title": "Readmission after joint replacement: a narrative review.",
    "journal": "JBJS Reviews",
    "pub_date": "2021 Jan 05",
    "reported_rate": "5.1%",
    "evidence_tier": "Narrative Review",
    "rate_sentence": "Readmission within 30 days is reported at 5.1%.",
    "url": "https://pubmed.ncbi.nlm.nih.gov/31234567/",
}


def _matrix(envelope):
    return next(c for c in _components(envelope)
                if c["component"] == "ComparativeRiskMatrix")


def test_a_verified_comparator_draws_the_comparison_above_the_records():
    """Order asserted, not merely presence.

    The records are no longer a child of this column (Phase F, 2026-10-26): they are
    citations in the prose and rows in the turn's references list, so what the canvas
    draws beneath the prose on a literature turn is the comparison and nothing that
    duplicates it.
    """
    envelope = compose_risk_canvas(
        PREDICT, None, literature=_search(),
        comparators=[HEART_FAILURE_COMPARATOR], command="summarize")

    assert _types(envelope)[:2] == ["Column", "ComparativeRiskMatrix"]
    assert "LiteratureList" not in _types(envelope)


def test_a_literature_turn_draws_no_cohort_table():
    """The question a `/pubmed` turn asks is what the literature says.

    Stage Gate 2 narrowed the table to `/ehr`. Before it, a comparative literature turn
    drew the patient's own bucket above the records — an answer to a different question
    printed beside the one that was asked.
    """
    envelope = compose_risk_canvas(
        PREDICT, None, literature=_search(),
        comparators=[HEART_FAILURE_COMPARATOR], command="literature")

    assert "ComparativeRiskMatrix" not in _types(envelope)
    assert "LiteratureList" not in _types(envelope)


def test_a_lab_turn_draws_no_cohort_table():
    """A drill-down asks what a measurement did, so the card is the whole answer.

    The first composed lab turn drew the trajectories under an empty-baseline matrix: the
    admission resolved to a phenotype, so a lab turn looked like a risk turn to the
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
        None, None, labs=labs, comparators=[ORTHO_COMPARATOR], command="summarize")

    assert "ComparativeRiskMatrix" not in _types(envelope)
    assert "LabTrendCard" in _types(envelope)


def test_the_table_draws_with_no_search_in_the_turn_at_all():
    """The change, stated as behaviour: `/ehr` gets the matrix without a literature query.

    This is the case that did not work before. The numerator used to require a retrieved
    record with a curated counterpart, so the unified briefing — narrative, risk widget,
    cohorts, in one turn — could not draw the third part of itself. Now the cohorts arrive
    with the estimate.
    """
    envelope = compose_risk_canvas(PREDICT, None,
                                   comparators=[HEART_FAILURE_COMPARATOR],
                                   command="summarize")

    assert "ComparativeRiskMatrix" in _types(envelope)
    assert "LiteratureList" not in _types(envelope)


def test_an_outpatient_turn_draws_no_cohort_table():
    """A 30-day readmission cohort is defined over inpatient stays.

    The gate is the encounter, passed in by the orchestrator, not a list of admission ids
    compiled into this layer — which is what the eligibility rule used to be.
    """
    envelope = compose_risk_canvas(
        PREDICT, None, comparators=[HEART_FAILURE_COMPARATOR],
        command="summarize", eligible=False)

    assert "ComparativeRiskMatrix" not in _types(envelope)


def test_the_pinned_row_carries_the_patients_figures_and_its_provenance():
    matrix = _matrix(compose_risk_canvas(
        PREDICT, None, comparators=[HEART_FAILURE_COMPARATOR], command="summarize"))

    baseline = matrix["baseline"]
    # Raw, so the component formats them with the same formatter the widget uses.
    # Pre-rendering them here would be a second implementation of the display
    # precision, and the two would part on the first change to either.
    assert baseline["probability"] == 0.2
    assert baseline["threshold"] == 0.11
    # The shared-factors column is gone (Phase 4C §1.2), and the pinned row lost it
    # with the others: the bars above the table are where drivers are read.
    assert "drivers" not in baseline
    assert baseline["provenance"] == "Model model-x"
    assert "XGBoost" not in str(baseline), "the learner is not in the payload"

    row = matrix["rows"][0]
    assert row["pmid"] == "37495280"
    assert row["reported_rate"] == "13.2%"
    assert row["source"] == "Journal of the American College of Cardiology · 2023"
    # The excerpt IS the sentence the rate was read from, so the figure and its
    # provenance are the same text and neither can drift from the other.
    assert row["excerpt"].startswith("Of the 1.5 million individuals at risk")
    assert "13.2%" in row["excerpt"]


def test_the_orthopedic_comparator_reaches_the_row_intact():
    """Cynthia Petrov's bucket, which had no cohort at all when the corpus selected it.

    The directive's headline case: an orthopedic patient's estimate had nothing to sit
    beside, which is why the matrix either drew nothing or drew heart failure literature.
    """
    matrix = _matrix(compose_risk_canvas(
        PREDICT, None, comparators=[ORTHO_COMPARATOR], command="summarize"))

    row = matrix["rows"][0]
    assert row["pmid"] == "34793857"
    assert row["reported_rate"] == "3.4%"
    assert row["evidence_tier"] == "Retrospective Cohort"
    assert row["source"] == "The Journal of Arthroplasty · 2022"
    # No shared-factors column: the driver bars above the table carry this, and a
    # second rendering of the same evidence read as a second source.
    assert "matched_features" not in row


def test_no_row_carries_a_second_copy_of_the_drivers():
    """The three-column contract, asserted across a turn with several rows.

    Phase 4C §1.2 removed the `Shared risk factors` column entirely, including from the
    pinned baseline row — the matrix is `Source | Rate | Cohort` and nothing else. The
    reason is a contract rather than a preference: the SHAP bars directly above the
    table already name each driver and its direction, so a flat pill list repeating
    them is the same evidence twice, and two renderings of one source read as two
    sources.
    """
    matrix = _matrix(compose_risk_canvas(
        PREDICT, None, comparators=[ORTHO_COMPARATOR, REVIEW_COMPARATOR],
        command="summarize"))

    assert len(matrix["rows"]) == 2
    for row in matrix["rows"]:
        assert "matched_features" not in row
    assert "drivers" not in matrix["baseline"]


def test_the_canvas_preserves_the_engines_row_order():
    """Order is the engine's, and this layer must not hold a second opinion.

    The engine hands the verified rows over in the order the extraction turn produced
    them. A re-sort here would be a second ordering computed from less than the engine
    holds, and the two would disagree on the first table where they were given different
    evidence.
    """
    matrix = _matrix(compose_risk_canvas(
        PREDICT, None, comparators=[REVIEW_COMPARATOR, ORTHO_COMPARATOR],
        command="summarize"))

    assert [row["evidence_tier"] for row in matrix["rows"]] == [
        "Narrative Review", "Retrospective Cohort"]


def test_a_turn_that_verified_no_cohort_draws_the_estimate_alone():
    """The honest empty: the table still draws, and it says what is missing.

    This is now a real outcome rather than a fallback for an uncurated condition —
    retrieval can retrieve and verify nothing. Drawing no table at all would be
    indistinguishable on screen from "this patient cannot be compared".
    """
    matrix = _matrix(compose_risk_canvas(PREDICT, None, comparators=[],
                                         command="summarize"))

    assert matrix["rows"] == []
    assert "baseline" in matrix
    assert "No cohort has been substituted" in matrix["baseline_note"]


def test_a_turn_with_no_command_draws_no_comparison():
    """The command gate: only `/ehr` asks the question the table answers.

    A free-text follow-up carries comparators like any other turn's state does; what it
    does not carry is the question. Drawing them under an answer to something else is the
    fourth part of a briefing arriving unasked.
    """
    envelope = compose_risk_canvas(
        PREDICT, None, literature=_search(),
        comparators=[HEART_FAILURE_COMPARATOR])

    assert "ComparativeRiskMatrix" not in _types(envelope)
    assert "LiteratureList" not in _types(envelope)


def test_the_table_draws_without_an_estimate_and_says_which_it_is():
    """A patient-less turn still has cohorts worth reading.

    What it must not do is draw the pinned row with a blank figure in it, so the row is
    absent and a sentence stands where it would have been.
    """
    matrix = _matrix(compose_risk_canvas(
        None, None, comparators=[HEART_FAILURE_COMPARATOR], command="summarize"))

    assert "baseline" not in matrix
    assert matrix["baseline_note"].strip()
    assert "No readmission estimate" in matrix["baseline_note"]


def test_a_long_journal_name_is_cut_to_the_marker_budget():
    """A pill sits inside a sentence, so its journal part has a length budget.

    Reported from the live turn (2026-10-16): a button wraps its own text by default, so
    "Journal of the American College of Cardiology" broke the pill's box across two lines
    and the rounded border read as a pill cut in half. Phase F (2026-10-26) gave the pill
    a leading number like every other citation in the product, so the budget is asserted
    on the journal part and the whole label is checked for the number it opens with.

    What this asserts is the REFERENCES LIST's line, which is still a numbered, dated
    bibliography entry. The inline pill is pinned separately, below.
    """
    citations = literature_citations([
        {**ARTICLE, "journal": "Journal of the American College of Cardiology"},
        {**ARTICLE, "journal": "Home healthcare now"},
    ])

    long_label = citations[0]["label"]
    assert long_label.startswith("1 · ")
    journal_part = long_label.split(" · ")[1]
    assert len(journal_part) <= 25, journal_part    # 24 characters, plus the ellipsis
    # Marked, not silently cut: an unmarked cut reads as the whole name.
    assert journal_part.endswith("…")
    assert "Journal of the American College of Cardiology".startswith(
        journal_part[:-1].rstrip())
    # A name inside the budget is passed through whole and unmarked, and the year closes
    # the list line so a reader can tell a 2009 record from a 2024 one without opening it.
    assert citations[1]["label"] == "2 · Home healthcare now · 2025"


def test_the_inline_pill_is_the_compact_journal_and_nothing_else():
    """No record number and no year — the pill is a marker, not a citation line.

    A reader reported the pill reading `World Neurosurgery · 2024` inside a sentence, and
    the same string stretching the comparison table's source column (2026-11-02). The
    number belongs to the bibliography a line below; the year is what turned a marker into
    a second citation line; and the compact form is the record's own abbreviation, because
    that is how a clinician cites the paper.
    """
    citations = literature_citations([
        {**ARTICLE, "journal": "JAMA network open", "journal_abbrev": "JAMA Netw Open",
         "pub_date": "2024 Mar"},
        {**ARTICLE, "journal": "Cureus", "journal_abbrev": "Cureus",
         "pub_date": "2023"},
        {**ARTICLE, "journal": "Journal of the American College of Cardiology",
         "journal_abbrev": "J Am Coll Cardiol", "pub_date": "2022 Jan"},
    ])

    assert [citation["pill_label"] for citation in citations] == [
        "JAMA Netw Open", "Cureus", "J Am Coll Cardiol"]
    for citation in citations:
        assert citation["kind"] == "literature"
        assert citation["pill_label"] == citation["pill_label"].strip()


def test_a_journal_title_is_title_cased_when_it_stands_in_for_the_abbreviation():
    citations = literature_citations([
        {**ARTICLE, "journal": "JAMA network open", "journal_abbrev": ""},
    ])

    assert citations[0]["pill_label"] == "JAMA Network Open"


def test_a_record_that_names_no_journal_is_marked_pubmed():
    citations = literature_citations([{**ARTICLE, "journal": "", "journal_abbrev": ""}])

    assert citations[0]["pill_label"] == "PubMed"


def test_a_note_pill_is_the_section_heading_in_title_case():
    """The badge carries the section's own name, not a ward shorthand.

    The pill was abbreviated for a while — `HPI`, `Discharge dx` — and a reader reported
    the opposite of what the abbreviation was for: `dx` is a chart shorthand and neither
    form is what the section is called. The words on the badge and the words in the note
    agree now, and the badge is compacted by its styling rather than by its vocabulary.
    """
    citations = note_citations([
        {"cite": 1, "section": "brief_hospital_course", "text": "Admitted."},
        {"cite": 2, "section": "history_of_present_illness", "text": "Pain."},
        {"cite": 3, "section": "laboratory_trends", "text": "Creatinine 1.4."},
        {"cite": 4, "section": "some_new_section", "text": "New."},
    ])

    assert [citation["pill_label"] for citation in citations] == [
        "Brief Hospital Course", "History of Present Illness", "Laboratory Trends",
        "Some New Section",
    ]
    # The colour channel, decided here: a note passage is this patient's own record.
    assert {citation["kind"] for citation in citations} == {"note"}
    # The references list keeps the number and the note's own casing.
    assert citations[0]["label"] == "1 · Brief hospital course"


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
    assert set(article) == {"pmid", "title", "journal", "pub_date", "authors",
                            "abstract", "url"}
    assert article["journal"] == ""
    # Phase F: the authors and the abstract are what the reference row and the hover card
    # are made of, so a record that states neither is coerced to empty rather than absent.
    assert article["authors"] == ""
    assert article["abstract"] == ""


def test_articles_is_empty_for_a_payload_that_is_not_one():
    for payload in (None, {}, {"articles": "not a list"}, {"articles": [None, 7]}):
        assert _usable_articles(payload) == []


# --- the swap --------------------------------------------------------------

def test_a_literature_answer_draws_no_risk_view():
    """The swap the feature exists for, and the reason it is still a swap. """
    envelope = compose_risk_canvas(None, None, literature=_search())

    assert "RiskInfluencers" not in _types(envelope)
    # And no second copy of the records: they are citations on this turn, poured into the
    # same reference list a note citation uses (Phase F, 2026-10-26).
    assert "LiteratureList" not in _types(envelope)


def test_the_records_reach_the_citation_channel_as_the_tool_returned_them():
    """The citation details are the record's own, so the row a reader checks and the
    summary they are checking it against are the same publication.

    Asserted on `sources` rather than on a canvas card, because that is where a record
    lives once it is a citation: the hover card and the drawer row are both built from
    this entry.
    """
    presentation = compose_presentation(
        "/pubmed ceftriaxone cellulitis", "One article.^[PMID: 31234567]",
        _call(_search()))

    (article,) = [source for source in presentation["sources"]
                  if source.get("kind") == "Literature"]
    assert article["pmid"] == ARTICLE["pmid"]
    assert article["title"] == ARTICLE["title"]
    assert article["section"] == "N Engl J Med · 2025"
    assert article["url"].endswith("31234567/")


def test_the_abstract_is_the_passage_the_reference_opens():
    """It was removed on 2026-10-13 because the only surface was a card showing a title
    and a journal. A reference that opens onto a title cannot be weighed, so the abstract
    is the excerpt now — carried once, in the citation, and not duplicated into a list.
    """
    presentation = compose_presentation(
        "/pubmed ceftriaxone cellulitis", "One article.^[PMID: 31234567]",
        _call(_search()))

    (row,) = [entry for entry in presentation["references"]
              if entry.get("pmid") == ARTICLE["pmid"]]
    assert row["excerpt"] == ARTICLE["abstract"]
    assert "LiteratureList" not in _types(presentation["a2ui"])


def test_the_risk_view_is_unchanged_without_literature():
    envelope = compose_risk_canvas(PREDICT, None)

    assert "RiskInfluencers" in _types(envelope)
    assert "LiteratureList" not in _types(envelope)


def test_literature_takes_the_canvas_over_an_estimate_in_the_same_turn():
    """The swap is the requested behaviour, so it wins a tie rather than merging."""
    envelope = compose_risk_canvas(PREDICT, None, literature=_search())

    assert "RiskInfluencers" not in _types(envelope)
    assert "RiskInfluencersWidget" not in _types(envelope)


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
    # The eyebrow under the title is the citation line, because the title is the article
    # now — there is no section to name instead.
    assert citations[0]["source"] == (
        "N Engl J Med. 2025. · PMID 31234567")
    assert citations[0]["url"] == ARTICLE["url"]


def test_a_citation_carries_the_pmid_its_marker_displays():
    """The marker shows the identifier, so the citation has to carry it: an index
    that the console had to resolve back to a record would be one mapping more than
    the thing needs."""
    citations = literature_citations([ARTICLE, _second()])

    assert [c["pmid"] for c in citations] == [ARTICLE["pmid"], "2"]
    assert citations[1]["pmid"] == _second()["pmid"]


def test_a_citation_carries_the_abstract_as_its_excerpt():
    """It was the popover's excerpt all along; Phase F makes the popover the surface
    again, so the text comes back — bounded, because the card floats over the answer
    being read."""
    citation = literature_citations([ARTICLE])[0]

    assert citation["excerpt"] == ARTICLE["abstract"]


def test_a_long_abstract_is_an_opening_rather_than_the_whole_record():
    """A structured abstract is served whole up to a bound that reaches its RESULTS.

    The bound was 400 characters, which is the opening of a structured abstract and stops
    before the section that carries the figures — a 1,866-character abstract was served as
    its first 394 characters, with the two arms' rates missing and no marker saying a cut
    had happened (2026-11-12). 2,400 characters covers an abstract PubMed serves in full.
    """
    long_abstract = "BACKGROUND: Cellulitis is common. " * 200
    citation = literature_citations([{**ARTICLE, "abstract": long_abstract}])[0]

    assert len(citation["excerpt"]) <= 2401
    assert citation["excerpt"].startswith("BACKGROUND: Cellulitis is common.")
    # Marked as an opening: either it lands on a sentence end inside the budget, or the
    # cut is shown with an ellipsis. What it must never be is an unmarked cut, which reads
    # as the whole abstract.
    assert (citation["excerpt"].endswith(".")
            or citation["excerpt"].endswith("…"))


def test_a_citation_with_no_pmid_still_carries_its_title():
    """The key is left out rather than sent empty, so the component's marker falls
    back to the number instead of linking to a record called ""."""
    citation = literature_citations([{"pmid": "", "title": "A record"}])[0]

    assert "pmid" not in citation
    assert citation["title"] == "A record"
    assert citation["index"] == 1


def test_a_record_with_no_url_key_still_links_to_its_record():
    """The url is derived from the identifier rather than trusted, so a record that
    omitted it cannot produce a link to nowhere."""
    citation = literature_citations([{"pmid": "1", "title": "Only a title"}])[0]

    assert citation["url"] == "https://pubmed.ncbi.nlm.nih.gov/1/"
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


def test_a_literature_marker_is_cited_to_its_article_even_beside_the_note():
    """Each marker opens its own record when a turn rests on both kinds of evidence.

    The live defect of 2026-10-13 was a literature summary whose markers resolved
    against note passages, leaving every one of them unbacked — a disabled control on
    the summary that was the one thing working. The fix then cited the articles
    INSTEAD of the note, which broke the reverse case on 2026-10-30: a chart marker
    beside an article went grey and the drawer lost the records. Both channels are
    cited now, each under its own number, and the record numbers continue after the
    note's.
    """
    sources = [{"cite": 1, "section": "hospital_course", "text": "a note passage"}]

    narrative = _narrative(
        "Day-case surgery had lower odds ^[PMID: 31234567] and he recovered ^[1].",
        sources, _search(), literature_start=2)

    # The eyebrow is the citation line now; the article title is the heading the card
    # shows, and the identifier is still what the marker resolves against.
    article = next(c for c in narrative["citations"] if c.get("pmid"))
    assert article["subtitle"] == "N Engl J Med. 2025. · PMID 31234567"
    assert article["title"] == ARTICLE["title"]
    # And the note's own marker is still a note citation, not the article in disguise.
    assert [c["source"] for c in narrative["citations"] if not c.get("pmid")] == [
        "Discharge note"]


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
    # No `streaming` key: see `test_a2ui.py` — the component's inert-marker state had
    # no writer and is gone.
    assert set(narrative) == {"text", "citations"}


def test_the_canvas_root_is_a_column_and_the_prose_leads_it():
    """Ruled flat: the prose sits under the root beside nothing else on a literature
    turn — the records are its citations, and their list is the turn's reference drawer,
    which the page draws rather than the canvas."""
    narrative = _narrative("Day-case surgery had lower odds ^[1].", [], _search())
    envelope = compose_risk_canvas(None, None, literature=_search(),
                                   narrative=narrative)

    root = _components(envelope)[0]

    assert root["id"] == "root"
    assert root["component"] == "Column"
    assert root["children"][0] == "narrative"
    assert "literature" not in root["children"]
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

def test_compose_presentation_publishes_the_records_as_citations():
    """Phase F (2026-10-26): a literature turn used to carry an empty `sources` and a
    boxed card. It now carries the articles in the citation channel, numbered, so the
    marker, the hover card and the reference row name one record."""
    presentation = compose_presentation(
        "search pubmed for recent literature", "One article.^[PMID: 31234567]",
        _call(_search())
    )

    assert presentation["literature"]["query"] == "ceftriaxone cellulitis"
    assert [source["pmid"] for source in presentation["sources"]] == ["31234567"]
    assert presentation["sources"][0]["cite"] == 1
    assert "LiteratureList" not in _types(presentation["a2ui"])


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
    """A search that returned records carries no degraded flag — and, on this surface,
    no records list either: the flag is what the empty case reads."""
    presentation = compose_presentation(
        "/pubmed ceftriaxone cellulitis", "One article.^[PMID: 31234567]",
        _call(_search()))

    assert not presentation["literature"].get("degraded")
    assert "LiteratureList" not in _types(presentation["a2ui"])


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
