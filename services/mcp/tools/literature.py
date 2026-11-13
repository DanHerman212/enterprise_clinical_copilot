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
import math
import re
import xml.etree.ElementTree as ElementTree
from typing import Annotated, Any, Sequence

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


#: What a record's title says when the record has one. Every entry is a string that
#: carries no information about the paper, so serving one tells the reader the journal
#: published a record and nothing else.
_TITLE_PLACEHOLDERS = frozenset({
    "", "not available", "not available]", "no title", "untitled", "untitled article",
    "n/a", "na", "none", "title not available", "no title available",
})

#: Served when a record states no usable title in either field. A clean label rather than a
#: placeholder that leaked out of the XML: the record is real, the article is linkable by
#: its PMID, and "Untitled article" says exactly what is missing without pretending the
#: journal printed the words "Not Available".
UNTITLED_ARTICLE = "Untitled article"


def _strip_title_brackets(text: str) -> str:
    """One title, with the brackets PubMed wraps a translated title in removed.

    PubMed states a non-English article's title in the original language inside square
    brackets: `[Cervical spondylotic myelopathy].` The brackets are markup — they say
    "this is a translation" — and the trailing full stop belongs to the title, not to the
    markup, so it survives.

    Matched PAIRS only. A lone leading `[` or a trailing `]` with no partner is left
    exactly as the record wrote it: the first version of this stripped the opening bracket
    unconditionally, and `[Not Available].` came out as `Not Available].` — a string that
    is neither the record's title nor a clean label, and that no reader can interpret.
    Trailing punctuation is peeled off before the pair is tested and restored afterwards
    for the same reason.
    """
    body = " ".join(str(text or "").split())
    if not body:
        return ""
    trailing = ""
    while body and body[-1] in ".!?,;:":
        trailing = body[-1] + trailing
        body = body[:-1].rstrip()
    if len(body) >= 2 and body.startswith("[") and body.endswith("]"):
        inner = body[1:-1].strip()
        if inner:
            return inner + trailing
    return body + trailing


def _is_placeholder_title(text: str) -> bool:
    """Whether a title carries no information about the paper.

    Checked on the de-bracketed, punctuation-stripped form, so `[Not Available].`,
    `Not Available` and `not available` are one case rather than three.
    """
    core = _strip_title_brackets(text).strip().rstrip(".!?,;: ").strip().lower()
    return core in _TITLE_PLACEHOLDERS


def _title(article: ElementTree.Element) -> str:
    """The article's title, from the record's preferred field and then its fallback.

    `<ArticleTitle>` is what the journal published. For a non-English article it is
    sometimes the literal string `[Not Available].` with the real title in
    `<VernacularTitle>` — PubMed's own placeholder, not a failure of the request. Serving
    the placeholder puts the words "Not Available" in a reference drawer above an abstract
    that is present, and on 2026-11-12 it also produced a citation pill labelled from it.

    So the fallback order is ArticleTitle, then VernacularTitle, and then a clean label.
    Order matters: a bracketed translated title in ArticleTitle is REAL (it is the paper's
    title in its own language) and outranks the English rendering in VernacularTitle, which
    is why the placeholder test runs before the fallback rather than the bracket strip
    doing the choosing.
    """
    for path in (".//Article/ArticleTitle", ".//Article/VernacularTitle"):
        candidate = _strip_title_brackets(_text(article.find(path)))
        if not _is_placeholder_title(candidate):
            return candidate
    return UNTITLED_ARTICLE


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
            # Never the record's placeholder: see `_title`. An article with no usable
            # title in either field is labelled, not left empty, so the contract check
            # below still sees a string.
            "title": _title(node),
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

#: Statuses a second request can fix: NCBI refusing because it is busy, and not because
#: the request is wrong. A 400 or a 404 fails identically on the retry, so those are
#: returned at once and reported as they are.
_TRANSIENT_STATUSES = frozenset({500, 502, 503, 504})

#: The same class as the retrieval path's dropped gRPC stream (`retrieval.py`): the
#: connection went away mid-request. Retried on the same terms, because a clinician who
#: clicked a literature chip cannot tell an outage from a search that found nothing, and
#: the answer must not report the first as the second.
_TRANSIENT_HTTP_ERRORS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
)


async def _get(
    client: httpx.AsyncClient, url: str, params: dict[str, str]
) -> httpx.Response:
    """GET, pausing and retrying a bounded number of times when the attempt can be fixed.

    Two failures are retried, for the same reason: the rate limit, because a clinician
    who clicked a chip has no way to tell a slow-down from an outage and the API key is
    not always present; and a transient refusal or a dropped connection, because the
    request is fine and the second attempt usually lands.

    The final response is returned whatever its status: the caller raises on it and the
    error branch below reports the outcome, so a rate limit that outlasts the retries is
    still reported as a rate limit rather than as a generic failure.
    """
    for attempt in range(_RATE_LIMIT_RETRIES + 1):
        final = attempt == _RATE_LIMIT_RETRIES
        try:
            response = await client.get(url, params=params)
        except _TRANSIENT_HTTP_ERRORS:
            if final:
                raise
            await asyncio.sleep(_RATE_LIMIT_BACKOFF_SECONDS * (2 ** attempt))
            continue
        if final:
            return response
        if response.status_code == 429 or response.status_code in _TRANSIENT_STATUSES:
            await asyncio.sleep(_RATE_LIMIT_BACKOFF_SECONDS * (2 ** attempt))
            continue
        return response
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

#: The horizon the comparison is made at, as the vocabulary an abstract uses.
_THIRTY_DAY = re.compile(
    r"\b30[\s-]?days?\b|\bthirty[\s-]?days?\b|\bwithin 30 days\b", re.IGNORECASE
)

#: A readmission count stated as ZERO in words. A real result, and a different claim from
#: "3.4%": it is a statement about a cohort small enough for zero to be its finding.
_VERBAL_ZERO = re.compile(
    r"no\s+(?:patient|cases?|patients)\s+(?:was|were)\s+(?:re)?admitted"
    r"|\bno\s+readmissions?\b|\bzero\s+readmissions?\b"
    r"|\bnone\s+(?:of\s+the\s+patients\s+)?(?:was|were)\s+(?:re)?admitted"
    r"|\bno\s+unplanned\s+readmissions?\b",
    re.IGNORECASE,
)

#: A cohort size as a paper states it: "1140 patients", "2,996 admissions", "n = 583",
#: "All 33 patients", "2996 hip fracture patients". The largest figure in the abstract is
#: taken, because a study that reports its whole registry and then a subgroup reports the
#: registry as its size.
#:
#: Up to three words may sit between the figure and the noun, because a cohort is named as
#: "2996 hip fracture patients" as often as "2996 patients" — without the gap `39420969`
#: stated no size at all, and the +17.4 its 2,996 patients had earned was silently missing
#: (2026-11-09).
#:
#: "hospitalisations" is on the list because a COPD cohort says "a total of 782
#: hospitalizations for AECOPD were enrolled" and not "782 patients" — measured 2026-11-09
#: on `37037836`, which stated its size and parsed as unknown.
#:
#: Two digits at minimum. The figure has to be a cohort and not an event count: "an
#: incisional hematoma was reported in 2 patients" states a complication, and reading it as
#: a sample size would put a two-patient denominator under a benchmark. Ten is the floor
#: below which a series is not a benchmark anyway.
_SAMPLE_SIZE = re.compile(
    r"(?:n\s*=\s*|\b)(\d[\d,]{1,})\s+(?:[A-Za-z][A-Za-z()-]*\s+){0,3}?"
    r"(?:patients|cases|admissions|hospitali[sz]ations|encounters|discharges|procedures|"
    r"adults|subjects|participants|individuals|people|recipients)\b",
    re.IGNORECASE,
)

#: A thousands separator written as a space: "75 000 people", "1 859 patients". PubMed
#: abstracts use it, and without collapsing it first the digits after the space read as a
#: cohort of their own — measured 2026-11-09, `40664407` ("Around 75 000 people suffer from
#: hip fractures yearly") parsed as N=0.
_SPACED_THOUSANDS = re.compile(r"(?<=\d)\s(?=\d{3}(?!\d))")

#: Below this many records the strict readmission query is re-asked without the horizon
#: clause and the two pools are merged.
#:
#: The same threshold the agent used, moved here with the retrieval it belongs to. The
#: strict clause requires the abstract to spell "30-day" or "thirty days"; a series that
#: reports "readmission at one month" matches the broadened clause and not the strict one.
MIN_STRICT_CANDIDATES = 8

#: `_PEDIATRIC_MARKERS`, mirrored. A cohort of children is a different patient, not a
#: narrower band of this one, and the filter belongs with retrieval rather than four
#: hundred lines away in the engine. The engine keeps its own copy because the two
#: deployables do not share a code path: the agent image carries `services/agent/` and the
#: MCP *config* module, not `services/mcp/tools/`, so an import from here would not resolve
#: in the container. Collapsing the two is Stage 9B's job.
_PEDIATRIC_MARKERS = re.compile(
    r"\b(?:pediatric|paediatric|children|child|infant|infants|neonatal|neonate|"
    r"adolescent|adolescents|juvenile|under 18|< ?18)\b",
    re.IGNORECASE,
)


def names_pediatric_population(*parts: Any) -> bool:
    """Whether any of these texts describes a cohort of children."""
    return any(_PEDIATRIC_MARKERS.search(f" {str(part or '')} ") for part in parts)


def _sentences(abstract: str) -> list[str]:
    """The abstract as sentences, without a regex that needs shell-hostile characters."""
    return [part.strip() for part in re.split(r"(?<=[.]) ", abstract or "") if part.strip()]


def readmission_rate_sentence(abstract: str) -> str:
    """The first sentence that states a READMISSION figure, or "".

    A sentence qualifies only when it names readmission AND carries a percentage. That
    pairing is what separates a benchmark from a paper that reports mortality, length of
    stay or complications: `39420969` spent a sentence on "males were more likely to be
    readmitted at 30 … (P < .001)" and a separate one on "30-day mortality … 6.6% vs
    4.5%", and neither states a readmission rate (measured 2026-11-09).
    """
    for sentence in _sentences(abstract):
        if _READMISSION_TERM.search(sentence) and _PERCENT.search(sentence):
            return sentence
    return ""


def sample_size(abstract: str) -> int | None:
    """The largest cohort the abstract states, or None when it states no size.

    Spaced thousands are tightened first so "75 000 people" is one figure rather than 75 and
    000, and a single digit is refused so an event count is not read as a denominator.
    """
    text = abstract or ""
    previous = None
    while previous != text:
        previous, text = text, _SPACED_THOUSANDS.sub("", text)
    sizes = [
        int(value.replace(",", "")) for value in _SAMPLE_SIZE.findall(text)
        if int(value.replace(",", "")) >= 10
    ]
    return max(sizes) if sizes else None


def score_benchmark_candidate(article: dict, *, condition: str = "",
                              anchors: Sequence[str] = ()) -> dict:
    """How well this record can serve as a 30-day benchmark, and why.

    Five facts, added up, all of them read off the abstract the reader could read:

        +40  a sentence states a readmission figure  (the only thing a row needs)
        +25  that sentence, or the abstract, names a 30-day horizon
        +0..30 clinical match: +18 procedure/anchor in the text (+6 if in the title),
                              +12 condition in the text (+4 if in the title)
        +0..20 cohort size, as `min(20, 5*log10(N))`
        -15  a verbal zero, or a cohort under 50

    The size term is the one this lacked. `_comparator_relevance` ranked three booleans and
    was blind to scale, so a 33-patient single-centre series that readmitted nobody sorted
    above a 1,140-patient NSQIP cohort reporting 3.4%, and the matrix showed the former
    (`90000069`, 2026-11-09). A log scale because the difference that matters is an order
    of magnitude, not a patient: N=1,140 is +15.3 and N=2,996 is +17.4, while N=33 is +7.6
    and the penalty takes it to -7.4.

    The verbal-zero penalty is not a judgement about the finding. "No patient was
    readmitted within 30 days" is a real result, and it is the result of a small cohort;
    when a larger numeric cohort exists for the same operation it is the better benchmark,
    and when none does this one still surfaces because nothing outranks it.

    Deterministic: every term is a substring or a regex over text the record publishes, and
    ties break on the caller's order, which is PubMed's.
    """
    abstract = str(article.get("abstract") or "")
    title = str(article.get("title") or "")
    text = f"{title} {abstract}".lower()

    rate_sentence = readmission_rate_sentence(abstract)
    states_rate = bool(rate_sentence)
    horizon = bool(_THIRTY_DAY.search(rate_sentence or abstract))

    score = 0.0
    if states_rate:
        score += 40.0
    if states_rate and horizon:
        score += 25.0

    condition_text = str(condition or "").strip().lower()
    anchor_texts = [str(anchor or "").strip().lower() for anchor in anchors if str(anchor or "").strip()]
    if condition_text and condition_text in text:
        score += 12.0 + (4.0 if condition_text in title.lower() else 0.0)
    for anchor in anchor_texts:
        if anchor and anchor != condition_text and anchor in text:
            score += 18.0 + (6.0 if anchor in title.lower() else 0.0)
            break

    size = sample_size(abstract)
    if size is not None:
        score += min(20.0, round(5.0 * math.log10(max(size, 1)), 2))

    verbal_zero = bool(_VERBAL_ZERO.search(abstract))
    if (not states_rate and verbal_zero) or (size is not None and size < 50):
        score -= 15.0

    return {
        "rank_score": round(score, 2),
        "sample_size_n": size,
        "has_readmission_rate_sentence": states_rate,
    }



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
    """The records that state a readmission result at all, as a number or as a zero.

    A number is what a comparator needs, and most records for a condition do not state one:
    they report an odds ratio, a five-year figure, or a qualitative finding. Filtering here
    rather than asking a model to look is what keeps the model's context to records that
    can produce a row at all — and what makes the empty result an honest answer rather than
    a model's opinion about relevance.

    A VERBAL ZERO counts as a number, which it did not until 2026-11-09. "No patient was
    readmitted within 30 days" is a real finding about a real cohort, and it was reaching
    the pool only when the same abstract happened to contain some other percentage — so a
    small series that readmitted nobody was dropped as having no figures, while the same
    series with an unrelated 72% discharge rate survived to be ranked first. The ranker
    now scores that kind of record down instead of the filter deleting it unseen.
    """
    kept: list[dict] = []
    for article in articles or []:
        if not isinstance(article, dict):
            continue
        abstract = str(article.get("abstract") or "")
        if not abstract or not _READMISSION_TERM.search(abstract):
            continue
        if not _PERCENT.search(abstract) and not _VERBAL_ZERO.search(abstract):
            continue
        kept.append(article)
    return kept


async def search_readmission_benchmarks(
    terms: str | list[str] | tuple[str, ...],
    *,
    wide: bool = False,
    max_results: int = READMISSION_CANDIDATE_POOL,
) -> LiteratureResult:
    """Live PubMed records ranked as readmission benchmarks, or none.

    This is the whole retrieval decision, in one place: which query to ask, whether to ask
    a second one, which cohorts belong to this patient's population, and in what order a
    model should read them. It used to be split — the tool chose the query and the agent
    re-asked, re-filtered and re-ranked — and the two halves drifted: the agent's ranker was
    three booleans with no notion of cohort size, and the tool's pool was filtered on a
    different rule than the agent's (`90000069`, `90000090`, 2026-11-09).

    `terms` is the condition followed by its anchor synonyms; `condition` and `anchors` are
    read off it, so the caller passes one list and the ranking names the same phrases the
    query was built from.

    `wide` is carried, not consulted: it is what the tool-call contract calls the
    benchmark scope, and both passes run here regardless. A flag that switched the passes
    off would be a way for a caller to silently lose the broadening that `90000047`
    needed.

    1. ASK THE STRICT QUESTION. The horizon clause pins "30-day" / "thirty days".
    2. ASK IT AGAIN, BROADER, IF THAT WAS THIN. Under `MIN_STRICT_CANDIDATES` the horizon
       clause is dropped and the two pools are merged by pmid. A series that reports
       "readmission at one month" matches the broadened clause and not the strict one, and
       on 2026-11-08 a salpingectomy admission returned four records and drew no matrix.
    3. DROP COHORTS OF CHILDREN. Before ranking, so a paediatric cohort cannot take a slot.
    4. SCORE AND SORT. `score_benchmark_candidate`, descending, so the model reads the
       records that can produce a row first.

    Every returned article carries `rank_score`, `sample_size_n` and
    `has_readmission_rate_sentence`, so a Langfuse trace shows the order and the reason
    without anyone reconstructing it — which is what the last audit had to do, and why the
    ranking was unobservable.

    Empty is a real answer and is reported as one (`{"articles": [], ...}`), never as an
    error. A degraded search propagates unchanged, because "the search did not happen" and
    "the search found nothing" are different statements and the caller has to be able to
    tell them apart.
    """
    condition = terms[0] if isinstance(terms, (list, tuple)) and terms else str(terms or "")
    anchors = [str(term).strip() for term in
               (terms[1:] if isinstance(terms, (list, tuple)) else []) if str(term).strip()]

    strict_term = readmission_query(terms, wide=False)
    result = await _search(
        strict_term, retmax=READMISSION_CANDIDATE_POOL,
        reldate_days=READMISSION_RECENCY_YEARS * 366,
    )
    if result.get("degraded") or result.get("error"):
        return result

    articles = quantitative_readmission_articles(result.get("articles") or [])
    term = strict_term
    if len(articles) < MIN_STRICT_CANDIDATES:
        broad_term = readmission_query(terms, wide=True)
        broader = await _search(
            broad_term, retmax=READMISSION_CANDIDATE_POOL,
            reldate_days=READMISSION_RECENCY_YEARS * 366,
        )
        if not (broader.get("degraded") or broader.get("error")):
            known = {str(article.get("pmid") or "") for article in articles}
            added = [article for article in quantitative_readmission_articles(
                broader.get("articles") or [])
                if str(article.get("pmid") or "") not in known]
            _LOG.info(
                "literature: strict query returned %d record(s), below the %d that make a "
                "pool; the broadened query added %d more",
                len(articles), MIN_STRICT_CANDIDATES, len(added),
            )
            articles += added
            term = broad_term if added else strict_term

    adults = [
        article for article in articles
        if not names_pediatric_population(
            article.get("title"), article.get("journal"), article.get("abstract"))
    ]
    for article in adults:
        article.update(score_benchmark_candidate(
            article, condition=str(condition), anchors=anchors))
    adults.sort(key=lambda article: -float(article.get("rank_score") or 0.0))
    adults = adults[:max_results]

    if not adults:
        return _empty(
            term, "PubMed returned no record stating a 30-day readmission rate.",
        )
    _LOG.info("literature: %d benchmark record(s) for %r", len(adults), term[:80])
    return {"query": term, "returned": len(adults), "articles": adults}


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
                "They are OR'd with the query so a record naming any of them is a "
                "statement about the same condition. They do not choose the query "
                "form: the retrieval asks the horizon-pinned question first and "
                "broadens it only if that pool is thin."
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
