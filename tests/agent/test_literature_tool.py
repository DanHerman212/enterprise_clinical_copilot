"""The literature tool's parsing, against the shapes PubMed actually returns.

These call the module's helpers directly rather than going out over HTTP. The
network path is two calls whose failure modes are already reported as structured
errors; what is worth pinning here is the extraction, because that is where a
misread record becomes a citation that points at the wrong paper — a failure that
looks like success to everything downstream.

The XML below is trimmed from real efetch output, keeping the parts that vary:
structured against single-block abstracts, `<MedlineDate>` against `Year`/`Month`,
and inline markup inside a title.
"""

import asyncio
import xml.etree.ElementTree as ElementTree

import httpx
import pytest

import services.mcp.tools.literature as lit
from services.mcp.tools.literature import _abstract, _articles, _pmids, _pub_date, _text

ARTICLE_SET = """
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>31234567</PMID>
      <Article>
        <Journal>
          <Title>New England Journal of Medicine</Title>
          <ISOAbbreviation>N Engl J Med</ISOAbbreviation>
          <JournalIssue>
            <PubDate><Year>2025</Year><Month>Mar</Month><Day>12</Day></PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Ceftriaxone for <i>lower extremity</i> edema</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Cellulitis is common.</AbstractText>
          <AbstractText Label="METHODS">We randomised 400 patients.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">31234567</ArticleId>
        <ArticleId IdType="doi">10.1056/NEJMoa1234567</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


def _root(xml: str) -> ElementTree.Element:
    return ElementTree.fromstring(xml)


def test_a_record_is_extracted_field_by_field():
    articles = _articles(_root(ARTICLE_SET))

    assert len(articles) == 1
    article = articles[0]
    assert article["pmid"] == "31234567"
    assert article["journal"] == "New England Journal of Medicine"
    assert article["pub_date"] == "2025 Mar 12"
    assert article["url"] == "https://pubmed.ncbi.nlm.nih.gov/31234567/"


def test_inline_markup_inside_a_title_is_not_truncated():
    """`.text` alone would stop at the `<i>` and return half a title.

    This is the reason `_text` uses `itertext`: PubMed wraps spans in markup
    routinely, and a truncated title is still a plausible-looking title.
    """
    article = _articles(_root(ARTICLE_SET))[0]

    assert article["title"] == "Ceftriaxone for lower extremity edema"


def test_abstract_labels_are_kept_and_sections_joined():
    article = _articles(_root(ARTICLE_SET))[0]

    assert article["abstract"] == (
        "BACKGROUND: Cellulitis is common. METHODS: We randomised 400 patients."
    )


def test_a_medline_date_is_returned_verbatim():
    """Not normalised: the record does not claim a month, so neither do we."""
    article = _root("""
    <PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID>
      <Article><Journal><JournalIssue>
        <PubDate><MedlineDate>2025 Jan-Feb</MedlineDate></PubDate>
      </JournalIssue></Journal>
      <ArticleTitle>T</ArticleTitle></Article>
    </MedlineCitation></PubmedArticle></PubmedArticleSet>
    """).find(".//PubmedArticle")

    assert _pub_date(article) == "2025 Jan-Feb"


def test_a_journal_falls_back_to_its_abbreviation():
    """Some records carry no full title, and a blank journal reads as a mistake."""
    article = _articles(_root("""
    <PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID>
      <Article><Journal><ISOAbbreviation>Lancet</ISOAbbreviation></Journal>
      <ArticleTitle>T</ArticleTitle></Article>
    </MedlineCitation></PubmedArticle></PubmedArticleSet>
    """))[0]

    assert article["journal"] == "Lancet"


def test_a_record_with_no_pmid_is_dropped():
    """It cannot be cited or linked, so it must not reach the canvas at all."""
    articles = _articles(_root("""
    <PubmedArticleSet><PubmedArticle><MedlineCitation>
      <Article><ArticleTitle>No identifier</ArticleTitle></Article>
    </MedlineCitation></PubmedArticle>
    <PubmedArticle><MedlineCitation><PMID>7</PMID>
      <Article><ArticleTitle>Has one</ArticleTitle></Article>
    </MedlineCitation></PubmedArticle></PubmedArticleSet>
    """))

    assert [a["pmid"] for a in articles] == ["7"]


def test_non_article_elements_are_ignored():
    """`DeleteCitation` and book records share the response; neither is an article."""
    assert _articles(_root("""
    <PubmedArticleSet>
      <DeleteCitation><PMID>1</PMID></DeleteCitation>
      <PubmedBookArticle><BookDocument><PMID>2</PMID></BookDocument></PubmedBookArticle>
    </PubmedArticleSet>
    """)) == []


def test_whitespace_is_collapsed():
    """The XML is indented, and the indentation must not reach the canvas."""
    assert _text(_root("<a>\n  one\n  <b>two</b>\n  three\n</a>")) == "one two three"


def test_pmids_are_coerced_to_strings():
    """A numeric id from another retmode would break `",".join` with a TypeError."""
    assert _pmids({"esearchresult": {"idlist": [31234567, "31234568"]}}) == [
        "31234567",
        "31234568",
    ]


def test_a_malformed_search_response_yields_no_pmids():
    """Every shape is checked, so an unexpected body becomes an empty search."""
    for payload in (None, [], {}, {"esearchresult": None}, {"esearchresult": {}}, {"esearchresult": {"idlist": "31234567"}}):
        assert _pmids(payload) == []


def test_an_empty_abstract_is_an_empty_string():
    """Not a crash, and not the word None on the canvas."""
    assert _abstract(_root("<PubmedArticle><Article></Article></PubmedArticle>")) == ""


# --- degradation ------------------------------------------------------------
#
# This is the only tool that reaches outside the project, and no clinical question
# depends on it. The site treats any `error` in a tool result as an outage — it
# refunds the attempt and shows the unavailable page — so an external failure must
# not be reported as one: that would be the wrong answer to a problem that is not
# ours, at the moment a clinician asked about a patient.


@pytest.fixture(autouse=True)
def _configured(monkeypatch):
    """A contact address and no retry pauses.

    The address so these tests exercise the search rather than the unconfigured
    branch, and the pauses so a rate-limit test does not sleep through a backoff.
    """
    monkeypatch.setattr(lit, "NCBI_EMAIL", "test@example.com")
    monkeypatch.setattr(lit, "_RATE_LIMIT_RETRIES", 0)


def test_a_timeout_degrades_rather_than_failing(monkeypatch):
    async def timeout(client, url, params):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(lit, "_get", timeout)

    result = asyncio.run(lit.search_literature("ceftriaxone cellulitis"))

    assert "error" not in result, "an external timeout must not become a tool error"
    assert result["degraded"] is True
    assert result["returned"] == 0
    assert result["articles"] == []


def test_a_rate_limit_degrades_rather_than_failing(monkeypatch):
    request = httpx.Request(
        "GET", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    )

    class Limited:
        status_code = 429

        def raise_for_status(self):
            raise httpx.HTTPStatusError("429", request=request, response=self)

    async def limited(client, url, params):
        return Limited()

    monkeypatch.setattr(lit, "_get", limited)

    result = asyncio.run(lit.search_literature("ceftriaxone cellulitis"))

    assert "error" not in result
    assert result["degraded"] is True


def test_an_unconfigured_address_degrades_rather_than_failing(monkeypatch):
    """A clinician should not lose their answer to our misconfiguration."""
    monkeypatch.setattr(lit, "NCBI_EMAIL", "")

    result = asyncio.run(lit.search_literature("ceftriaxone cellulitis"))

    assert "error" not in result
    assert result["degraded"] is True


def test_a_normal_result_carries_no_degraded_flag(monkeypatch):
    """Only an unavailable search is marked; an empty one is an answer."""
    async def empty(client, url, params):
        request = httpx.Request("GET", url)
        return httpx.Response(200, json={"esearchresult": {"idlist": []}}, request=request)

    monkeypatch.setattr(lit, "_get", empty)

    result = asyncio.run(lit.search_literature("a query with no matches"))

    assert result["returned"] == 0
    assert "degraded" not in result, "an empty search is not a degraded one"


def test_a_blank_query_is_still_an_error():
    """The one failure this tool still raises, and deliberately so.

    A blank query is a defect in our own pipeline, not a failure of anyone else's
    service. Degrading it would say "temporarily unavailable" about a service that is
    working, and would hide the bug behind a reassuring sentence.
    """
    result = asyncio.run(lit.search_literature("   "))

    assert result["error"] == "bad_request"


def test_a_degraded_payload_satisfies_the_declared_contract():
    """The canvas renders it, so it has to validate like any other result."""
    from services.mcp.contracts import validate_literature_result

    payload = {
        "query": "ceftriaxone",
        "returned": 0,
        "articles": [],
        "note": "PubMed literature search is temporarily unavailable.",
        "degraded": True,
    }

    assert validate_literature_result(payload) is payload
