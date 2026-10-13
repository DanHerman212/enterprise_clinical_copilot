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
