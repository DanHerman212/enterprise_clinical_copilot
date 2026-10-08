"""The canvas swap: a literature answer draws articles, not a risk score.

This is the feature's whole visible behaviour, and it is pure — no browser, no
endpoint, no clock — so it is asserted here rather than eyeballed in a page. The
regression half matters as much as the new half: the risk canvas must look exactly
as it did before when no search ran.
"""

from services.agent.a2ui import (
    _usable_articles,
    compose_presentation,
    compose_risk_canvas,
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
    assert set(article) == {"pmid", "title", "journal", "pub_date", "abstract", "url"}
    assert article["journal"] == ""


def test_articles_is_empty_for_a_payload_that_is_not_one():
    for payload in (None, {}, {"articles": "not a list"}, {"articles": [None, 7]}):
        assert _usable_articles(payload) == []


# --- the swap --------------------------------------------------------------

def test_a_literature_answer_draws_articles_instead_of_the_risk_view():
    envelope = compose_risk_canvas(None, None, literature=_search())

    assert "LiteratureList" in _types(envelope)
    assert "RiskBar" not in _types(envelope)
    assert "FactorBars" not in _types(envelope)


def test_the_articles_reach_the_canvas_whole():
    """Unedited, because the point is that the clinician can check the summary."""
    envelope = compose_risk_canvas(None, None, literature=_search())

    rendered = next(c for c in _components(envelope) if c["component"] == "LiteratureList")
    assert rendered["articles"][0]["abstract"] == ARTICLE["abstract"]
    assert rendered["articles"][0]["url"].endswith("31234567/")


def test_the_risk_view_is_unchanged_without_literature():
    envelope = compose_risk_canvas(PREDICT, None)

    assert "RiskBar" in _types(envelope)
    assert "FactorBars" in _types(envelope)
    assert "LiteratureList" not in _types(envelope)


def test_literature_takes_the_canvas_over_an_estimate_in_the_same_turn():
    """The swap is the requested behaviour, so it wins a tie rather than merging."""
    envelope = compose_risk_canvas(PREDICT, None, literature=_search())

    assert "LiteratureList" in _types(envelope)
    assert "RiskBar" not in _types(envelope)


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
