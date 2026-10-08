"""The search_literature tool.

Searches PubMed for recent literature on a clinical question and returns the
records themselves — title, journal, publication date, abstract, and the link to
the source — so the agent can summarise the evidence and the console can show the
articles the summary rests on.

Returns plain JSON, never A2UI (same rule as predict.py and retrieval.py). The
agent composes presentation from this payload.

This tool takes NO patient identifier, and that is deliberate. Every other tool is
scoped to an admission; a literature search is not — the same query returns the
same articles whoever the patient is. Keeping `hadm_id` out of the signature is
also the privacy boundary: with no admission to look anything up by, the only
thing this tool can put on the wire to a third party is the search string it was
handed. The orchestrator must de-identify the question before calling.
See docs/literature-tool-plan.md.

Retrieval is two NCBI E-utilities calls rather than LangChain's `PubMedAPIWrapper`,
which returns one concatenated text blob whose fields have to be recovered by
parsing. PubMed records vary too much for that — structured abstracts against
single blocks, inconsistent author lists, per-journal copyright lines — and an
abstract is not something to guess at the boundaries of.

Contract:
    search_literature(query: str, max_results: int = 3) -> dict
    {
      "query": "ceftriaxone lower extremity edema",
      "returned": 2,
      "articles": [
        {"pmid": "31234567",
         "title": "Ceftriaxone for ...",
         "journal": "N Engl J Med",
         "pub_date": "2025 Mar 12",
         "abstract": "BACKGROUND: ... METHODS: ...",
         "url": "https://pubmed.ncbi.nlm.nih.gov/31234567/"}
      ]
    }

Empty is a real answer, and is reported as one: {"articles": [], "returned": 0,
"note": "..."}. It is never an error payload. A caller that reads a missing result
as a failure refunds the request and tells the clinician the service is down, at
the moment it answered correctly. Never fabricate a citation either.
"""

import asyncio
import logging
import xml.etree.ElementTree as ElementTree
from typing import Annotated, Any

import httpx
from pydantic import Field

from ..config import NCBI_API_KEY, NCBI_EMAIL, NCBI_EUTILS_BASE, NCBI_TIMEOUT_SECONDS
from ..contracts import (
    LiteratureResult,
    ToolError,
    tool_error,
)

# How many articles one search may return, and the range the model is told it may
# ask for. The ceiling exists because every record is an abstract that lands in the
# model's context and on the canvas: ten is already a page of reading.
MAX_RESULTS_MIN = 1
MAX_RESULTS_MAX = 10
DEFAULT_MAX_RESULTS = 3

# Longest search string accepted, in characters. A search string, not a paragraph —
# the clinical question is reduced to terms before it arrives here, and the cap is
# what keeps a runaway model from posting a note's worth of prose to a third party.
MAX_QUERY_CHARS = 300

# Kept, deliberately, rather than assumed: PubMed will happily return a case report
# from a minor journal as the most recent hit for a common question. Restricting to
# higher evidence tiers is one line when it is wanted —
#   '"systematic review"[pt] OR "meta-analysis"[pt] OR "randomized controlled trial"[pt]'
# — and it is off by default because the same filter turns a narrow question into no
# result at all, which reads as the tool being broken.
PUBLICATION_TYPE_FILTER = None

# Ordering and the recency window, decided together (2026-10-08).
#
# Relevance, not date. Sorting by date alone surfaces whatever was indexed most
# recently against the keywords, which for a common question is often a case report
# in a minor journal while the landmark trial sits on page two. Relevance is what
# makes the top three worth reading.
#
# Recency comes from `reldate`, which NCBI resolves against the current date. The
# alternative — a date range written into the query string, `("2021"[Date -
# Publication] : "3000"[...])` — also works and goes quietly stale: the window widens
# by a year every year, and nobody re-reads a constant embedded in a query. `reldate`
# is a rolling window by construction, so "the last five years" means that in 2030 as
# well as today.
SORT_ORDER = "relevance"
RECENCY_YEARS = 5
# 366 rather than 365: a leap year inside the window must not shorten it.
RECENCY_DAYS = RECENCY_YEARS * 366

# NCBI asks callers to identify themselves on every request.
_TOOL_NAME = "ecc-clinical-copilot"

_LOG = logging.getLogger(__name__)


def _error(code: str, message: str, *, detail: str | None = None) -> ToolError:
    """A failure the caller can act on, with the detail left in the log.

    Same rule as predict.py: the message IS the tool result and reaches the model,
    the caller and the browser, so it carries a sentence and nothing else. URLs,
    status codes and response bodies are diagnostics and go to the log.

    There is no `hadm_id` to record, unlike predict.py's helper, because this tool
    is not scoped to an admission — that absence is the point of it.
    """
    if detail:
        _LOG.warning("literature %s: %s", code, detail)
    return tool_error(code, message)


def _empty(query: str, note: str) -> LiteratureResult:
    """A search that legitimately matched nothing."""
    return {"query": query, "returned": 0, "articles": [], "note": note}


# One sentence for the clinician, deliberately not distinguishing causes: a rate
# limit, a timeout and a misconfiguration all mean the same thing at the bedside,
# and naming the mechanism would tell them something they cannot act on.
_LITERATURE_UNAVAILABLE = "PubMed literature search is temporarily unavailable."


def _degraded(
    query: str, code: str, detail: str, *, misconfiguration: bool = False
) -> LiteratureResult:
    """A search that could not be performed, reported as nothing found WITH a flag.

    This tool is the only one that reaches outside the project, and no clinical
    question depends on it: a clinician asking about a patient still wants the risk
    score and the note summary while an outside agency rate-limits its own server.
    The site treats any `error` in a tool result as an outage — it refunds the
    attempt and shows the unavailable page — which would be the wrong answer to a
    problem that is not ours.

    `degraded` is what keeps that from becoming a lie. It is the difference between
    "the search found nothing" and "the search did not happen", and the agent is told
    to report the second as such, because claiming something was absent when nobody
    looked is inventing a fact.

    Cause and detail go to the log rather than into the payload: the sentence reaches
    the model and the browser, and neither can act on a status code. A
    misconfiguration logs at ERROR because it will not fix itself; an external
    failure logs at WARNING because it might.
    """
    log = _LOG.error if misconfiguration else _LOG.warning
    log("literature %s (degraded): %s", code, detail)
    return {
        "query": query,
        "returned": 0,
        "articles": [],
        "note": _LITERATURE_UNAVAILABLE,
        "degraded": True,
    }


def _request_params() -> dict[str, str]:
    """Identity parameters NCBI expects on every E-utilities request."""
    params = {"tool": _TOOL_NAME}
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params


def _text(node: ElementTree.Element | None) -> str:
    """All the text under a node, with whitespace collapsed.

    `itertext` rather than `.text` because PubMed wraps spans in markup — `<i>`,
    `<sup>`, `<b>` — and reading only `.text` truncates a title at the first tag.
    Whitespace is collapsed because the XML is indented, so the raw text carries
    newlines and leading spaces into whatever renders it.
    """
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _pub_date(article: ElementTree.Element) -> str:
    """Publication date as the record states it, or empty.

    Two shapes occur: `<Year>`/`<Month>`/`<Day>` for a journal issue with a known
    date, and a single `<MedlineDate>` such as "2025 Jan-Feb" for one without. Both
    are returned as they appear rather than normalised — the reader is judging
    whether the paper is current, and tidying "2025 Jan-Feb" into a month would
    invent precision the record does not have.
    """
    published = article.find(".//Journal/JournalIssue/PubDate")
    if published is None:
        return ""
    medline = _text(published.find("MedlineDate"))
    if medline:
        return medline
    parts = (
        _text(published.find("Year")),
        _text(published.find("Month")),
        _text(published.find("Day")),
    )
    return " ".join(part for part in parts if part)


def _abstract(article: ElementTree.Element) -> str:
    """The abstract, with its structure labels kept.

    A structured abstract arrives as several `<AbstractText>` elements, each
    carrying its own `Label` — BACKGROUND, METHODS, RESULTS. The labels are kept
    because they are how a clinician reads it, and the pieces are joined into one
    string because that is the shape the canvas and the model both want.
    """
    parts: list[str] = []
    for node in article.findall(".//Abstract/AbstractText"):
        text = _text(node)
        if not text:
            continue
        label = (node.get("Label") or "").strip()
        parts.append(f"{label}: {text}" if label else text)
    return " ".join(parts)


def _pmids(payload: Any) -> list[str]:
    """PMIDs from an esearch response.

    Coerced to strings rather than assumed: the JSON carries them as strings, but a
    different `retmode` returns them as numbers, and `",".join` over a list of ints
    raises at the moment a clinician asked a question.
    """
    if not isinstance(payload, dict):
        return []
    result = payload.get("esearchresult")
    if not isinstance(result, dict):
        return []
    identifiers = result.get("idlist")
    if not isinstance(identifiers, list):
        return []
    return [str(identifier) for identifier in identifiers if str(identifier)]


def _articles(root: ElementTree.Element) -> list[dict[str, str]]:
    """Every `PubmedArticle` in an efetch response, in document order."""
    articles: list[dict[str, str]] = []
    for node in root.findall(".//PubmedArticle"):
        pmid = _text(node.find(".//MedlineCitation/PMID"))
        if not pmid:
            # The one field everything downstream depends on: without it there is
            # no citation to render and no link to resolve, so a record missing it
            # is dropped rather than shown with a blank badge.
            continue
        journal = _text(node.find(".//Journal/Title")) or _text(
            node.find(".//Journal/ISOAbbreviation")
        )
        articles.append({
            "pmid": pmid,
            "title": _text(node.find(".//Article/ArticleTitle")),
            "journal": journal,
            "pub_date": _pub_date(node),
            "abstract": _abstract(node),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return articles


# NCBI allows 3 requests per second from an address without an API key, and a search is
# two requests. Measured 2026-10-08 without a key: the first search answered and the
# second was refused. Two retries with a widening pause turn that into a slow answer
# instead of a failed one, which matters because a clinician who clicked a chip has no
# way to tell a rate limit from an outage. The API key is the real fix; this is what
# keeps a deployment without one usable.
_RATE_LIMIT_RETRIES = 2
_RATE_LIMIT_BACKOFF_SECONDS = 0.6


async def _get(
    client: httpx.AsyncClient, url: str, params: dict[str, str]
) -> httpx.Response:
    """GET, pausing and retrying a bounded number of times when NCBI says slow down.

    The final response is returned whatever its status: the caller raises on it and
    the error branch below reports the outcome, so a rate limit that outlasts the
    retries is still reported as a rate limit rather than as a generic failure.
    """
    for attempt in range(_RATE_LIMIT_RETRIES + 1):
        response = await client.get(url, params=params)
        if response.status_code != 429 or attempt == _RATE_LIMIT_RETRIES:
            return response
        await asyncio.sleep(_RATE_LIMIT_BACKOFF_SECONDS * (2 ** attempt))
    raise AssertionError("unreachable: the loop returns on its final attempt")


async def _fetch(query: str, limit: int) -> LiteratureResult:
    """esearch for PMIDs, then efetch for the records.

    Every failure here degrades. Nothing that an outside service does should turn
    into a request failure for a question about a patient.
    """
    common = {**_request_params(), "db": "pubmed"}
    term = f"({query}) AND ({PUBLICATION_TYPE_FILTER})" if PUBLICATION_TYPE_FILTER else query

    try:
        # One client for both calls: the second reuses the connection the first
        # opened, and the timeout applies to the pair rather than being reset.
        async with httpx.AsyncClient(timeout=NCBI_TIMEOUT_SECONDS) as client:
            found = await _get(
                client,
                f"{NCBI_EUTILS_BASE}/esearch.fcgi",
                {
                    **common,
                    "term": term,
                    "retmode": "json",
                    "retmax": str(limit),
                    "sort": SORT_ORDER,
                    # Publication date, not entry date: a 1998 paper indexed last
                    # week is not recent literature.
                    "datetype": "pdat",
                    "reldate": str(RECENCY_DAYS),
                },
            )
            found.raise_for_status()
            identifiers = _pmids(found.json())
            if not identifiers:
                return _empty(query, "PubMed found no articles for this search.")

            records = await _get(
                client,
                f"{NCBI_EUTILS_BASE}/efetch.fcgi",
                {**common, "id": ",".join(identifiers), "retmode": "xml"},
            )
            records.raise_for_status()
            body = records.text
    except httpx.TimeoutException as exc:
        return _degraded(query, "literature_timeout", str(exc))
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        code = "literature_rate_limited" if status == 429 else "literature_failed"
        return _degraded(query, code, f"{status} from {exc.request.url.path}")
    except (httpx.HTTPError, ValueError) as exc:
        return _degraded(query, "literature_failed", f"unreadable response: {exc}")

    try:
        articles = _articles(ElementTree.fromstring(body))
    except ElementTree.ParseError as exc:
        return _degraded(query, "literature_failed", f"unparsable xml: {exc}")

    if not articles:
        return _empty(query, "PubMed returned no readable records for this search.")

    _LOG.info("literature: %d article(s) for %r", len(articles), query[:80])
    return {"query": query, "returned": len(articles), "articles": articles}


async def search_literature(
    query: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_QUERY_CHARS,
            description=(
                "PubMed search terms, for example 'ceftriaxone lower extremity "
                "edema'. A de-identified clinical question in search syntax, "
                "never patient text and never an admission identifier."
            ),
        ),
    ],
    max_results: Annotated[
        int,
        Field(
            ge=MAX_RESULTS_MIN,
            le=MAX_RESULTS_MAX,
            description="how many articles to return (1-10, default 3)",
        ),
    ] = DEFAULT_MAX_RESULTS,
) -> LiteratureResult | ToolError:
    """Search PubMed for recent literature on a clinical question.

    Returns the matching articles, each with its title, journal, publication date,
    full abstract and a link to the record on PubMed, so the evidence behind a
    summary can be read rather than taken on trust.

    Call this when the question is what the literature says rather than what this
    patient's record says. The query must be a de-identified search string: this
    tool has no access to the patient and must not be given their text.

    An empty result is a real answer and comes back as `returned: 0` with a `note`
    explaining it, not as an error.

    Args:
        query: PubMed search terms, de-identified.
        max_results: how many articles to return, 1 to 10.
    """
    cleaned = " ".join(query.split())
    if not cleaned:
        # The one failure this tool still raises. A blank query is a defect in our own
        # pipeline, not a failure of anybody else's service: degrading it would report
        # "temporarily unavailable" about a service that is working, and would hide the
        # bug behind a reassuring sentence.
        return _error(
            "bad_request",
            "A literature search needs search terms, not blank space.",
        )

    if not NCBI_EMAIL:
        # Degraded rather than an error, and logged at ERROR so somebody is told: the
        # clinician should not lose their answer because a deployment forgot a
        # variable, but the omission must not be silent either.
        return _degraded(
            cleaned,
            "literature_unavailable",
            "NCBI_EMAIL is not set",
            misconfiguration=True,
        )

    # Clamped rather than trusted. The advertised schema tells the model the bounds,
    # and the bounds are enforced again here, because the schema is advice.
    limit = max(MAX_RESULTS_MIN, min(int(max_results), MAX_RESULTS_MAX))
    return await _fetch(cleaned[:MAX_QUERY_CHARS], limit)
