"""Curated cohort benchmarks for the comparative readmission matrix.

`/pubmed` answers a question with the records PubMed returned. The matrix answers a
different question — how does THIS patient's 30-day readmission estimate compare with
published cohorts — and the two are not the same kind of thing. A retrieved record is
whatever the search returned; a benchmark is a cohort with a rate a reader can hold
against the model's, which most records in a result set do not have. Of the thirty
records captured for the three demo conditions, **four** state an absolute readmission
rate at a horizon the model's estimate can be compared with. The rest report a relative
measure (RR, OR, HR), or a rate at five years, or no rate at all.

So this is a curation and not a parse, and that is why it is written out rather than
derived. There is no extraction step in the repository to drift, no regex over an
abstract that can silently stop matching, and no value here a script produced. Each
entry names the PMID its figures come from and quotes, in the comment directly above
it, the sentence the figures were read off — so the number can be checked without
re-reading the paper, and a wrong one has somewhere to point.

Three rules the entries hold to, in the order they are likely to be tempted:

  * A row is included only when its horizon is the model's. A five-year composite beside
    a 30-day probability is not a comparison, it is two unrelated numbers in adjacent
    cells with a subtraction waiting to happen. `PARTNER 3` is the case that makes the
    rule concrete rather than theoretical: both its papers state a rate, and neither
    belongs in this table.
  * The rate is quoted as the record states it. "≈20%" rather than "20%", because the
    abstract says "close to 20%" and the qualifier is part of the figure — dropping it
    would turn a hedged sentence into a measured one.
  * A row is one record, not one cohort-mate. Where a study compares two groups (the
    durotomy readmissions below), the second group is named in `cohort_summary` rather
    than given a row of its own, so the table cannot be read as two independent findings.

Keyed by PMID rather than attached to the captured fixtures, and the difference is
load-bearing twice over. A fixture is meant to be a response the live tool could also
have returned, so a key added to one is a key the browser can tell the two paths apart
by — the one thing `demo/fixtures.py` exists to prevent. And a live search returns
different PMIDs for the same query, so what belongs to a record has to be looked up by
identifier rather than shipped beside it. The cost of that choice is stated plainly: a
live search matches none of these, the matrix renders no rows, and the literature
surface falls back to the cards it draws today.

`matched_features` is deliberately near-empty, and the emptiness is the honest answer.
It names THIS patient's drivers that the cohort itself identified as risk factors for
readmission. Where a record names none of them the tuple is empty, because a blank cell
says the literature is silent — which is a fact about the literature and the reader is
owed it. Filling the column from cohort inclusion criteria would put a claim in the
record's mouth that the record did not make.

Feature keys, not labels: the pill under a cohort and the bar beside the patient have to
name one factor one way, and the only way to guarantee that is to look the name up in
the same map the bars use (`feature_labels.humanize`).
"""

from dataclasses import dataclass

#: The horizon the model's estimate is made at, in days.
#:
#: The matrix compares only rows at this horizon, and it is declared here rather than
#: assumed so that a row at another horizon cannot join the comparison by being added
#: carefully. If the model is ever retrained at a different horizon, this is the one
#: number that has to move with it, and every row below stops qualifying until it does.
MODEL_HORIZON_DAYS = 30


@dataclass(frozen=True)
class CohortBenchmark:
    """One published cohort, as a row the matrix can compare a patient against."""

    pmid: str
    #: The rate, as the record states it. A string, because the qualifier is a digit
    #: or it is not: "≈20%" and "20%" are different claims and one type cannot hold both.
    reported_rate: str
    cohort_summary: str
    evidence_tier: str
    horizon_days: int
    #: This patient's drivers, as feature keys, that the cohort itself named.
    matched_features: tuple[str, ...] = ()


BENCHMARKS: dict[str, CohortBenchmark] = {
    # "Of the 1.5 million individuals at risk, 13.2% (95% CI: 10.5%-16.1%) were
    # readmitted within 30 days and 35.7% (95% CI: 27.1%-44.9%) within 1 year."
    # — "Global Comparison of Readmission Rates for Patients With Heart Failure",
    # Journal of the American College of Cardiology, 2023. The 30-day figure is the
    # one taken; the same paper reports the 1-year rate and it does not qualify.
    "37495280": CohortBenchmark(
        pmid="37495280",
        reported_rate="13.2%",
        cohort_summary=(
            "Hospitalised heart failure, pooled across 24 observational reports "
            "and ~1.5 million patients, with no country or care setting excluded"
        ),
        evidence_tier="Meta-Analysis",
        horizon_days=30,
        # The cohort reports the rate globally and identifies no feature-level driver,
        # so there is nothing here that names any of this patient's factors.
        matched_features=(),
    ),
    # "The pooled 30-, 60-, 90-, 180-, and 365-day readmission rates were 11%, 17%,
    # 17%, 30%, and 37%, respectively." — "Readmission rate for acute exacerbation of
    # chronic obstructive pulmonary disease: A systematic review and meta-analysis",
    # Respiratory Medicine, 2023. 46 studies; the 30-day pooled rate is the one taken.
    "36528962": CohortBenchmark(
        pmid="36528962",
        reported_rate="11%",
        cohort_summary=(
            "Acute exacerbation of COPD, pooled across 24 of 46 included studies "
            "that reported a 30-day rate"
        ),
        evidence_tier="Meta-Analysis",
        horizon_days=30,
        # "We also identified potential risk factors for COPD readmission, including
        # male sex, number of hospitalizations in the previous year, LOS, and
        # comorbidities such as heart failure, tumor or cancer, and diabetes."
        #
        # "Number of hospitalizations in the previous year" is the one phrase in the
        # whole capture that names a factor this patient also has, and it is prior
        # utilisation however it is counted — so the pill carries the patient's own
        # name for that factor rather than a second name for it.
        matched_features=("prior_inpatient_days",),
    ),
    # "Importantly, close to 20% of patients are readmitted within 30 days after
    # hospital discharge, with great cost to the person and society." — "Differential
    # Diagnosis of Suspected Chronic Obstructive Pulmonary Disease Exacerbations in
    # the Acute Care Setting: Best Practice", American Journal of Respiratory and
    # Critical Care Medicine, 2023.
    "36701677": CohortBenchmark(
        pmid="36701677",
        reported_rate="≈20%",
        cohort_summary=(
            "Hospital discharge after an exacerbation of COPD, as the prevalence "
            "quoted in a clinical practice review of the acute care setting"
        ),
        evidence_tier="Narrative Review",
        horizon_days=30,
        matched_features=(),
    ),
    # "Of the 52 patients discharged home same day, 2 were readmitted (3.8%); of the
    # 59 patients admitted postoperatively, 5 were readmitted (8.5%)." — "Readmission
    # rates after intraoperative durotomy and repair in outpatient lumbar spinal
    # surgery", North American Spine Society Journal, 2026. The same-day group is the
    # row; the admitted group is named in the summary rather than given a row.
    "42004460": CohortBenchmark(
        pmid="42004460",
        reported_rate="3.8%",
        cohort_summary=(
            "Outpatient lumbar spine surgery complicated by an intraoperative "
            "durotomy, repaired and discharged the same day (n=52; 8.5% in the "
            "postoperatively admitted comparator, n=59)"
        ),
        evidence_tier="Observational Cohort",
        horizon_days=30,
        matched_features=(),
    ),
}


def _year(pub_date: str) -> str:
    """The four-digit year a PubMed date states, or an empty string.

    PubMed dates are not one format — "2023", "2023 Aug 01", "2022 Sep-Oct 01" — so
    this reads the year off rather than parsing a date. Nothing here needs a day.
    """
    for token in str(pub_date or "").replace("-", " ").split():
        if len(token) == 4 and token.isdigit():
            return token
    return ""


def _excerpt(abstract: str, limit: int = 220) -> str:
    """The opening of an abstract, ended at a sentence where one is available.

    The matrix's column is a prompt to open the source, not a substitute for it, so
    this is deliberately short and stops at a boundary when it can. A hard cut
    mid-sentence would read as a damaged record rather than a shortened one.
    """
    text = " ".join(str(abstract or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if stop > limit // 3:
        return cut[: stop + 1]
    return cut.rstrip() + "…"


def benchmarks_for(articles: list[dict]) -> list[dict]:
    """The curated benchmark rows for these retrieved records, in the order given.

    Only the records that have one. A record with no benchmark is not a row: the matrix
    exists to compare a patient against cohorts that published a rate, and a record that
    states none of the table's columns has nothing to contribute to it. That is why a
    live search usually yields an empty list — see the module docstring.

    The journal and the date come from the RECORD, not from this table, so the row shows
    the citation the search actually returned rather than a second copy of it that could
    disagree.
    """
    rows: list[dict] = []
    for article in articles or []:
        if not isinstance(article, dict):
            continue
        benchmark = BENCHMARKS.get(str(article.get("pmid") or ""))
        if benchmark is None:
            continue
        year = _year(article.get("pub_date", ""))
        journal = str(article.get("journal") or "").strip()
        rows.append({
            "pmid": benchmark.pmid,
            "source": " · ".join(part for part in (journal, year) if part),
            "reported_rate": benchmark.reported_rate,
            "cohort_summary": benchmark.cohort_summary,
            "matched_features": list(benchmark.matched_features),
            "evidence_tier": benchmark.evidence_tier,
            "horizon_days": benchmark.horizon_days,
            "url": str(article.get("url") or ""),
            "excerpt": _excerpt(article.get("abstract", "")),
        })
    return rows
