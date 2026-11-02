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
import re
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


def _authors(article: ElementTree.Element) -> str:
    """The first three authors, as a citation writes them, or empty.

    Phase F (2026-10-26): a reference line reads `Journal. Year. Authors. · PMID`, and
the authors have to come from the record rather than from a second lookup. PubMed
names an author as `<LastName>` plus either `<Initials>` or `<ForeName>`; the first
three are named and the rest are elided, which is the convention a bibliography uses
and the only form that fits beside a journal and a year.

    A record with no author list returns empty rather than a placeholder: a collective
or anonymous work states no author, and "et al." against nobody would be a claim.
    """
    names: list[str] = []
    for node in article.findall(".//AuthorList/Author"):
        last = _text(node.find("LastName"))
        if not last:
            # A collective name ("The SPRINT Research Group") is not a person and is
            # not abbreviated; it is used whole or not at all.
            collective = _text(node.find("CollectiveName"))
            if collective:
                names.append(collective)
            continue
        initials = _text(node.find("Initials"))
        names.append(f"{last} {initials}".strip())
    if not names:
        return ""
    head = ", ".join(names[:3])
    return f"{head}, et al." if len(names) > 3 else head


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
            # The compact form a citation uses and a pill prints — `JAMA Netw Open`,
            # `N Engl J Med`, `Spine`. Carried beside the title rather than derived from
            # it: an abbreviation is the publisher's, and the one place that can make it
            # is the record that states it.
            "journal_abbrev": _text(node.find(".//Journal/ISOAbbreviation")),
            "pub_date": _pub_date(node),
            # The citation line's third part. Empty when the record states no author,
            # which the reference then leaves out rather than inventing.
            "authors": _authors(node),
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


async def _search(query: str, *, retmax: int, reldate_days: int) -> LiteratureResult:
    """esearch for PMIDs, then efetch for the records.

    Every failure here degrades. Nothing that an outside service does should turn
    into a request failure for a question about a patient.

    `retmax` and `reldate_days` are arguments rather than module constants because the
    two callers want opposite things from them: `/pubmed` asks for three recent records
    and a benchmark sweep asks for a dozen across fifteen years (see
    `search_readmission_benchmarks`). Everything between the two calls — the identity
    parameters, the sort, the degrade contract — is shared, so the two cannot drift into
    behaving differently on the same failure.
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
                    "retmax": str(retmax),
                    "sort": SORT_ORDER,
                    # Publication date, not entry date: a 1998 paper indexed last
                    # week is not recent literature.
                    "datetype": "pdat",
                    "reldate": str(reldate_days),
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


async def _fetch(query: str, limit: int) -> LiteratureResult:
    """The `/pubmed` search: recent records, relevance-sorted, no filtering.

    Behaviour unchanged by the benchmark path below. A `/pubmed` question wants the
    best three records for a clinical question whatever they report; a comparator
    sweep wants the few records that state a rate, which is a different question
    asked of the same two endpoints.
    """
    return await _search(query, retmax=limit, reldate_days=RECENCY_DAYS)


# --- readmission benchmark retrieval ----------------------------------------
#
# `/pubmed` asks "what does the literature say". This asks a narrower question: "which
# published cohorts state a 30-day readmission rate this patient's estimate can be set
# against". Same two E-utilities calls, three different choices — the outcome is pinned
# in the query, the window is wide enough to reach a landmark cohort, and the candidate
# pool is filtered on the abstract before any model reads it, because a record that
# states no rate cannot become a comparator however well it matches the condition.
#
# There is no curated table behind this and none may be added: every row a caller acts
# on comes from the records returned here, in this request.

#: The outcome half of a benchmark query. Field terms rather than prose, because the
#: abstract's own wording is what has to match; `[Title/Abstract]` keeps a MeSH-only hit
#: from qualifying a record whose abstract never mentions readmission at all.
READMISSION_OUTCOME_CLAUSE = (
    '("30-day readmission"[Title/Abstract] OR "readmission rate"[Title/Abstract] '
    'OR "unplanned readmission"[Title/Abstract] OR "hospital readmission"[Title/Abstract])'
)

#: Fifteen years rather than `/pubmed`'s five. A landmark readmission cohort is not
#: recent, and the comparison is of rates rather than of current practice: the recency
#: window that makes a treatment question answerable would exclude most rows worth
#: comparing against.
READMISSION_RECENCY_YEARS = 15

#: The broader outcome half, used on a second pass. The four phrases above are exact
#: strings, so a record that says "readmission within 30 days" or "30-day
#: rehospitalisation" matches none of them — measured on 90000090, where six femoral-neck
#: records came back and every one of them reported a 90-day rate because the 30-day
#: cohort never entered the result set. This clause asks for the horizon and the outcome
#: separately, so PubMed finds the record whichever phrase it used.
READMISSION_WIDE_CLAUSE = (
    "(readmission[Title/Abstract] OR readmitted[Title/Abstract] "
    "OR rehospitalization[Title/Abstract] OR rehospitalisation[Title/Abstract]) AND "
    '("30-day"[Title/Abstract] OR "30 days"[Title/Abstract] '
    'OR "thirty-day"[Title/Abstract] OR "thirty days"[Title/Abstract])'
)

#: Twenty-five rather than twelve. Sorting by relevance puts landmark 90-day series above
#: the 30-day cohorts for a common operation, so a small pool filters to nothing even when
#: a qualifying record exists further down the ranking.
READMISSION_CANDIDATE_POOL = 25

#: A readmission mention and a percentage, matched case-insensitively on the abstract.
_READMISSION_TERM = re.compile(r"readmi(?:ssion|tted)|rehospitali[sz]ation", re.IGNORECASE)
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s*%")


def _clinical_clause(terms: str | list[str] | tuple[str, ...]) -> str:
    """The clinical half of a broadened query, as an OR of quoted phrases.

    Quoted, because "femoral neck fracture" as free terms also matches a paper about
    femoral shaft fractures; OR'd, because the synonyms are one clinical concept and any
    one of them naming a record is the same statement about the patient.

    Only the broadened pass uses this. The first pass asks one plain question and lets
    PubMed's relevance ranking answer it: measured on 2026-11-02, an OR of six synonyms
    put empyema and lobectomy abstracts above a pleural-effusion cohort, because a broad
    OR gives the ranking more ways to be loudly wrong.
    """
    phrases = [terms] if isinstance(terms, str) else list(terms or [])
    quoted = [
        f'"{" ".join(str(phrase).split())}"'
        for phrase in phrases if str(phrase or "").strip()
    ]
    return f"({' OR '.join(quoted)})" if quoted else ""


def readmission_query(
    terms: str | list[str] | tuple[str, ...], *, wide: bool = False,
) -> str:
    """The E-utilities term for one benchmark search: the core target, or the wide pass.

    `terms` is the patient's PRIMARY CONDITION on the first pass and the anchor synonyms
    on the second. The clinical half is always the caller's — derived from the chart,
    never from a table in this module — and the outcome half is pinned here so that every
    benchmark search asks PubMed the same question about a different condition.

    The first pass takes one phrase unquoted, which is how PubMed's own relevance ranking
    gets to work on the condition alone. The broadened pass ORs the quoted synonyms and
    swaps in the horizon-and-outcome clause, which is the whole point of a second attempt.
    """
    if wide:
        clinical = _clinical_clause(terms)
        return f"{clinical} AND {READMISSION_WIDE_CLAUSE}" if clinical else READMISSION_WIDE_CLAUSE
    core = terms[0] if isinstance(terms, (list, tuple)) and terms else terms
    core = " ".join(str(core or "").split())
    return f"({core}) AND {READMISSION_OUTCOME_CLAUSE}" if core else READMISSION_OUTCOME_CLAUSE


def quantitative_readmission_articles(articles: list[dict]) -> list[dict]:
    """The records whose abstract states both a readmission and a percentage.

    A number is the only thing a comparator needs from a record, and most records for a
    condition do not state one: they report an odds ratio, a five-year figure, or a
    qualitative finding. Filtering here rather than asking a model to look is what keeps
    the model's context to records that can produce a row at all — and what makes the
    empty result an honest answer rather than a model's opinion about relevance.
    """
    kept: list[dict] = []
    for article in articles or []:
        if not isinstance(article, dict):
            continue
        abstract = str(article.get("abstract") or "")
        if not abstract or not _READMISSION_TERM.search(abstract):
            continue
        if not _PERCENT.search(abstract):
            continue
        kept.append(article)
    return kept


async def search_readmission_benchmarks(
    terms: str | list[str] | tuple[str, ...],
    *,
    wide: bool = False,
    max_results: int = READMISSION_CANDIDATE_POOL,
) -> LiteratureResult:
    """Live PubMed records that could serve as readmission comparators, or none.

    `terms` is the primary condition on the first pass and the anchor synonyms on the
    broadened one. `wide` selects the second-pass outcome clause.

    This function runs ONE query and returns the whole filtered pool — `max_results`
    defaults to the pool rather than to what a model should read. Ranking the candidates
    against the patient's own condition, and truncating to what the extraction turn can
    read, is the caller's job: it is the only layer that knows which phrase is the
    patient's condition, and on 2026-11-02 truncating on PubMed's order dropped a
    qualifying cohort from a pool that contained it.

    Empty is a real answer and is reported as one (`{"articles": [], ...}`), never as an
    error. A degraded search propagates unchanged, because "the search did not happen" and
    "the search found nothing" are different statements and the caller has to be able to
    tell them apart.
    """
    term = readmission_query(terms, wide=wide)
    result = await _search(
        term, retmax=READMISSION_CANDIDATE_POOL,
        reldate_days=READMISSION_RECENCY_YEARS * 366,
    )
    if result.get("degraded") or result.get("error"):
        return result

    articles = quantitative_readmission_articles(result.get("articles") or [])
    articles = articles[:max_results]
    if not articles:
        return _empty(
            term, "PubMed returned no record stating a 30-day readmission rate.",
        )
    _LOG.info("literature: %d benchmark record(s) for %r", len(articles), term[:80])
    return {"query": term, "returned": len(articles), "articles": articles}


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
    scope: Annotated[
        str,
        Field(
            description=(
                "'general' searches recent literature for the query; "
                "'readmission' runs the 30-day readmission benchmark retrieval for a "
                "patient's primary condition, which is a wider window and returns the "
                "whole filtered candidate pool."
            ),
        ),
    ] = "general",
    synonyms: Annotated[
        list[str] | None,
        Field(
            description=(
                "readmission scope only: clinical synonyms for the same condition. "
                "Supplying them runs the broadened second pass (horizon-and-outcome "
                "clause, OR of quoted phrases) instead of the first."
            ),
        ),
    ] = None,
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

    if str(scope or "general").strip().lower() == "readmission":
        # The benchmark retrieval, reached through the tool that is already registered.
        # One tool rather than two because the trace contract is written against
        # `tool.search_literature`: a second name would mean a second branch in
        # `record_turn_tools`, in the Langfuse dashboards and in every reader of a trace.
        #
        # The pool is returned whole rather than capped at `max_results`: the caller ranks
        # it against the patient's own condition and decides what the extraction turn
        # reads, and a cap here would truncate on PubMed's order before that ranking.
        terms: str | list[str] = cleaned
        if synonyms:
            terms = [cleaned, *(str(synonym).strip() for synonym in synonyms
                                if str(synonym or "").strip())]
        return await search_readmission_benchmarks(
            terms,
            wide=bool(synonyms),
            max_results=READMISSION_CANDIDATE_POOL,
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
    # Returned unvalidated on purpose (Phase 4, 2026-10-25). Its siblings call their
    # own validators before returning; this one relies on `TOOL_CONTRACTS` at the
    # agent boundary, which is the hop that actually sees what the model reads. Adding
    # a second check of the same payload on the same hop would buy nothing and would
    # put two definitions of the literature contract in two processes — the exact
    # drift the registry exists to prevent. Do not "fix" this asymmetry without
    # removing the registry entry in the same change.
    return await _fetch(cleaned[:MAX_QUERY_CHARS], limit)
