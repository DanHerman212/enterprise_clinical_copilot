"""find_readmission_benchmarks — candidate 30-day readmission cohorts, for review.

Why this exists
---------------
`services/agent/literature_benchmarks.py` pairs a patient's predicted 30-day risk with a
published cohort rate, and every curated row carries the verbatim abstract sentence the
rate was read from so the pairing can be checked by eye. Finding those rows is the part
that must not be done from memory: a PMID, a rate, or a horizon that nobody published is
the single worst thing this feature could ship, because it would read exactly like the
ones that are real.

So candidate rows are retrieved here and reviewed before they are curated. This script
does NOT write `literature_benchmarks.py`; it prints what PubMed returns and what each
record actually says about its horizon, and a human decides.

Why it does not reuse `services/mcp/tools/literature.py`
--------------------------------------------------------
That path is built to answer a clinician's question about a patient: it searches recent
publication dates only, filters by publication type, and returns six fields. Curation
wants the opposite of the first two — a landmark cohort study is allowed to be ten years
old, and a meta-analysis is one of the things being looked for — and it needs the
publication types that path deliberately does not carry. Reusing it would silently narrow
the candidate set, which is a bad failure for a search whose whole purpose is completeness.

The horizon rule this enforces
------------------------------
A row may only be curated when the study reports a **30-day** readmission rate: the model
predicts 30 days, and a 90-day rate set beside it is a different quantity wearing the same
label. Records that state only 90-day or 1-year figures are therefore listed under
`HORIZON MISMATCH` rather than dropped, because "this bucket has no 30-day study" is a
finding worth knowing before a bucket is promised one.

    .venv/bin/python scripts/agent/find_readmission_benchmarks.py --phenotype orthopedic_arthroplasty
    .venv/bin/python scripts/agent/find_readmission_benchmarks.py --all --json-out /tmp/candidates.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from xml.etree import ElementTree

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.mcp.config import (  # noqa: E402
    NCBI_API_KEY,
    NCBI_EMAIL,
    NCBI_EUTILS_BASE,
    NCBI_TIMEOUT_SECONDS,
)

#: The curation queries, per phenotype.
#:
#: Every one of these is FIELD-TAGGED and scoped to title/abstract, and that is not
#: cosmetic. Plain relevance queries — "total knee arthroplasty 30-day readmission rate" —
#: return whatever PubMed ranks highest for the words, which measured as GLP-1 receptor
#: agonists, an Area Deprivation Index study, and a revision-implant comparison: papers
#: that mention 30 days and an arthroplasty without reporting a 30-day readmission rate
#: for a cohort. The tagged form asks the question the benchmark actually needs. On the
#: first pass the plain form missed BOTH usable elective-TJA cohort studies (36126887,
#: 34793857), which is the failure mode that matters: a bucket declared to have no 30-day
#: evidence when it has two.
#:
#: The outcome phrase is required in the title or abstract rather than anywhere in the
#: record, and both spellings of the outcome are accepted because the field is split
#: between them.
_OUTCOME = ('("30-day readmission"[Title/Abstract] '
            'OR "30-day unplanned readmission"[Title/Abstract] '
            'OR "30-day all-cause readmission"[Title/Abstract])')

QUERIES: dict[str, tuple[str, ...]] = {
    "orthopedic_arthroplasty": (
        f'(("total knee arthroplasty"[Title/Abstract] OR "total hip arthroplasty"[Title/Abstract]'
        f' OR "total joint arthroplasty"[Title/Abstract])) AND {_OUTCOME}',
        '("hip fracture"[Title/Abstract]) AND "30-day readmission"[Title/Abstract]',
    ),
    "heart_failure_cardiorenal": (
        f'("heart failure"[Title/Abstract]) AND {_OUTCOME}',
    ),
    "copd_respiratory": (
        f'("chronic obstructive pulmonary disease"[Title/Abstract] OR "COPD"[Title/Abstract]) '
        f'AND {_OUTCOME}',
    ),
    "spine_neurosurgery": (
        f'(("lumbar fusion"[Title/Abstract] OR "lumbar decompression"[Title/Abstract]'
        f' OR "spine surgery"[Title/Abstract] OR "lumbar stenosis"[Title/Abstract])) '
        f'AND {_OUTCOME}',
    ),
    "sepsis_infectious": (
        f'("sepsis"[Title/Abstract]) AND {_OUTCOME}',
        f'("pneumonia"[Title/Abstract]) AND {_OUTCOME}',
    ),
    "gi_hepatobiliary_surgery": (
        f'(("colorectal surgery"[Title/Abstract] OR "gastrointestinal bleeding"[Title/Abstract]'
        f' OR "hepatobiliary"[Title/Abstract])) AND {_OUTCOME}',
    ),
    "general_medicine_metabolic": (
        f'("general medicine"[Title/Abstract] OR "internal medicine"[Title/Abstract]) '
        f'AND {_OUTCOME}',
    ),
    "renal_urologic": (
        f'("acute kidney injury"[Title/Abstract]) AND {_OUTCOME}',
    ),
    "neurological_medical": (
        f'("ischemic stroke"[Title/Abstract] OR "acute stroke"[Title/Abstract]) AND {_OUTCOME}',
    ),
}

#: Publication types that make a record a usable benchmark, best evidence first. Not a
#: filter — records outside this list are still shown, marked, because the point of the
#: review is to see what is there.
TIER_ORDER = (
    "Meta-Analysis", "Systematic Review", "Randomized Controlled Trial",
    "Observational Study", "Multicenter Study", "Comparative Study",
    "Validation Study", "Journal Article", "Review",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PERCENT = re.compile(r"\b(\d{1,2}(?:\.\d{1,2})?)\s?%")
_THIRTY_DAY = re.compile(r"\b30[- ]day\b|\bthirty[- ]day\b", re.IGNORECASE)
_NINETY_DAY = re.compile(r"\b90[- ]day\b|\bninety[- ]day\b", re.IGNORECASE)


def _text(node) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _abstract(node) -> str:
    """The abstract, structured labels included: a rate is often under `RESULTS:`."""
    parts: list[str] = []
    for block in node.findall(".//Article/Abstract/AbstractText"):
        label = block.get("Label")
        body = _text(block)
        parts.append(f"{label}: {body}" if label and body else body)
    return "\n".join(part for part in parts if part)


def _year(node) -> str:
    for path in (".//JournalIssue/PubDate/Year", ".//Article/ArticleDate/Year",
                 ".//JournalIssue/PubDate/MedlineDate"):
        value = _text(node.find(path))
        if value:
            found = re.search(r"\d{4}", value)
            if found:
                return found.group(0)
    return ""


def _pub_types(node) -> list[str]:
    return [_text(n) for n in node.findall(".//PublicationTypeList/PublicationType")
            if _text(n)]


def _tier(types: list[str]) -> str:
    for tier in TIER_ORDER:
        if tier in types:
            return tier
    return "Unclassified"


def _rate_sentences(abstract: str) -> tuple[list[str], list[str]]:
    """Sentences stating a 30-day rate, and sentences stating a 90-day one.

    Both are returned because the second is the reason a record gets rejected, and a
    rejection nobody can see is indistinguishable from a search that found nothing.
    """
    thirty: list[str] = []
    ninety: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(" ".join(abstract.split())):
        if not _PERCENT.search(sentence):
            continue
        if _THIRTY_DAY.search(sentence):
            thirty.append(sentence.strip())
        elif _NINETY_DAY.search(sentence):
            ninety.append(sentence.strip())
    return thirty, ninety


def _request_params() -> dict[str, str]:
    params = {"tool": "clinical_copilot_curation", "email": NCBI_EMAIL or "unset@example.com",
              "retmode": "json"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params


def search(query: str, limit: int) -> list[dict]:
    """esearch then efetch. No recency window and no publication-type filter."""
    common = {**_request_params(), "db": "pubmed"}
    with httpx.Client(timeout=NCBI_TIMEOUT_SECONDS) as client:
        found = client.get(f"{NCBI_EUTILS_BASE}/esearch.fcgi",
                           params={**common, "term": query, "retmax": str(limit),
                                   # Without this PubMed answers newest-first, and the
                                   # first pass came back as eleven 2026 records, none of
                                   # them a cohort benchmark. A landmark 30-day
                                   # readmission study is allowed to be older than the
                                   # current year, so relevance is the right ordering and
                                   # recency is a deliberate non-filter here.
                                   "sort": "relevance"})
        found.raise_for_status()
        ids = found.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        time.sleep(0.4)  # NCBI allows 3 requests/second without a key; a search is two.
        records = client.get(f"{NCBI_EUTILS_BASE}/efetch.fcgi",
                             params={**common, "id": ",".join(ids), "retmode": "xml"})
        records.raise_for_status()

    root = ElementTree.fromstring(records.text)
    results: list[dict] = []
    for node in root.findall(".//PubmedArticle"):
        pmid = _text(node.find(".//MedlineCitation/PMID"))
        if not pmid:
            continue
        abstract = _abstract(node)
        thirty, ninety = _rate_sentences(abstract)
        types = _pub_types(node)
        results.append({
            "pmid": pmid,
            "title": _text(node.find(".//Article/ArticleTitle")),
            "journal": (_text(node.find(".//Journal/Title"))
                        or _text(node.find(".//Journal/ISOAbbreviation"))),
            "year": _year(node),
            "evidence_tier": _tier(types),
            "publication_types": types,
            "rate_sentences_30d": thirty,
            "rate_sentences_90d": ninety,
            "abstract": abstract,
        })
    return results


def _markdown(everything: dict[str, list[dict]]) -> str:
    """The review document.

    Generated from the retrieved records rather than written by hand, because the whole
    point of the review is that a human can check a rate against the sentence it came
    from — and a rate retyped into a document is exactly the step where that stops being
    true.
    """
    lines = [
        "# 04B candidate benchmarks — for review",
        "",
        "Generated by `scripts/agent/find_readmission_benchmarks.py`. Nothing here is",
        "curated yet: these are the records PubMed returns for each phenotype, with the",
        "sentences that state a rate, so each one can be checked before it is used.",
        "",
        "A row may only be curated when the study reports a **30-day** rate. Records that",
        "state a 90-day figure and no 30-day figure are listed at the end of each bucket",
        "rather than dropped, because \"this bucket has no 30-day study\" is a finding worth",
        "seeing before a bucket is promised one.",
        "",
    ]
    for phenotype, records in everything.items():
        usable = [r for r in records if r["rate_sentences_30d"]]
        mismatched = [r for r in records if not r["rate_sentences_30d"]
                      and r["rate_sentences_90d"]]
        lines += [f"## {phenotype}", "",
                  f"{len(usable)} record(s) state a 30-day rate.", ""]
        for r in usable:
            lines.append(f"### {r['pmid']} — {r['year']} — {r['evidence_tier']}")
            lines.append(f"*{r['title']}*")
            lines.append("")
            lines.append(f"- Journal: {r['journal']}")
            lines.append(f"- Publication types: {', '.join(r['publication_types']) or 'n/a'}")
            lines.append("- Sentences stating a 30-day rate:")
            for sentence in r["rate_sentences_30d"][:3]:
                lines.append(f"    - {sentence}")
            lines.append("")
        if mismatched:
            lines += [f"### Rejected on horizon ({len(mismatched)})", "",
                      "30-day figure absent; 90-day figure present. Not usable as a",
                      "30-day benchmark.", ""]
            for r in mismatched:
                lines.append(f"- {r['pmid']} ({r['year']}, {r['journal']}): "
                             f"{r['title'][:110]}")
                for sentence in r["rate_sentences_90d"][:1]:
                    lines.append(f"    - {sentence[:240]}")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phenotype", action="append", default=None,
                        help=f"Repeatable. One of: {', '.join(QUERIES)}")
    parser.add_argument("--all", action="store_true", help="Every phenotype in QUERIES.")
    parser.add_argument("--limit", type=int, default=12, help="Records per query.")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None,
                        help="Write the review document here.")
    args = parser.parse_args()

    wanted = list(QUERIES) if args.all else (args.phenotype or ["orthopedic_arthroplasty"])
    unknown = [p for p in wanted if p not in QUERIES]
    if unknown:
        parser.error(f"unknown phenotype(s): {', '.join(unknown)}")

    everything: dict[str, list[dict]] = {}
    for phenotype in wanted:
        seen: set[str] = set()
        records: list[dict] = []
        for query in QUERIES[phenotype]:
            print(f"\n=== {phenotype} :: {query!r}")
            try:
                found = search(query, args.limit)
            except (httpx.HTTPError, ValueError) as exc:
                print(f"    SEARCH FAILED: {type(exc).__name__}: {exc}")
                continue
            for record in found:
                if record["pmid"] in seen:
                    continue
                seen.add(record["pmid"])
                records.append(record)
        everything[phenotype] = records

        usable = [r for r in records if r["rate_sentences_30d"]]
        mismatched = [r for r in records if not r["rate_sentences_30d"]
                      and r["rate_sentences_90d"]]
        silent = [r for r in records if not r["rate_sentences_30d"]
                  and not r["rate_sentences_90d"]]

        def show(group: list[dict], heading: str) -> None:
            if not group:
                return
            print(f"\n  --- {heading}")
            for r in group:
                print(f"  {r['pmid']}  {r['year']}  {r['evidence_tier']:<18} {r['journal'][:44]}")
                print(f"      {r['title'][:150]}")
                for sentence in (r["rate_sentences_30d"] or r["rate_sentences_90d"])[:3]:
                    print(f"        * {sentence[:260]}")

        show(usable, f"30-DAY CANDIDATES ({len(usable)})")
        show(mismatched, f"HORIZON MISMATCH - 90-day only, NOT usable ({len(mismatched)})")
        print(f"\n  {len(silent)} further record(s) state no horizon rate at all.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(everything, indent=2) + "\n")
        print(f"\ncandidates written to {args.json_out}")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(_markdown(everything))
        print(f"review document written to {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
