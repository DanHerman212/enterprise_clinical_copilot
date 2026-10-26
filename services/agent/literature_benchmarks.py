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

Keyed by PHENOTYPE, since Phase 4B, and the change is the point of that phase. Rows used
to be looked up by matching the PMIDs a live search returned, which meant the matrix drew
its rows only when a search happened to return a record that had been curated — measured
at the time of the change: no live search ever did, so the matrix rendered nothing, and
the one verification where it appeared did so by luck. A patient's condition is not a
property of what a search returned, so the rows are now selected by the admission's own
resolved phenotype (`admission_phenotypes`), which is reviewed, committed, and the same
table every time. The cost of the old keying is therefore removed rather than documented:
the matrix renders for any admission whose phenotype has rows below.

What the rows are NOT keyed by is the retrieval, and that is deliberate in the other
direction too: `/pubmed` on a condition the patient does not have still shows the records
PubMed returned, as evidence cards badged general — it simply does not borrow the
patient's own estimate to compare them against, because the two are not the same thing.

Quoting the rate's own sentence
-------------------------------
Each row now carries `rate_sentence`: the sentence from the abstract that states the
figure, reproduced verbatim. It is what the matrix shows, so the number in the table and
the sentence it came from are the same sentence, and a reader checking the figure does not
have to take the curation's word for it. A summary written here could be checked only
against the paper; the paper's own sentence is checkable in the table.

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
    #: The phenotype this row may be shown for, spelled as `admission_phenotypes` spells
    #: it. A row is only ever rendered for a patient resolved to this bucket, which is
    #: what stops an orthopedic patient's estimate meeting a heart failure cohort.
    phenotype: str
    #: The rate, as the record states it. A string, because the qualifier is a digit
    #: or it is not: "≈20%" and "20%" are different claims and one type cannot hold both.
    reported_rate: str
    cohort_summary: str
    evidence_tier: str
    horizon_days: int
    #: The sentence the rate was read from, verbatim. Shown in the matrix, so the figure
    #: and its provenance are the same text.
    rate_sentence: str
    #: Citation parts. Carried here because there is no retrieved record to take them
    #: from now that selection is by phenotype — and a second copy that could disagree
    #: with the search is exactly what this replaces, so these are the only copy.
    journal: str = ""
    year: str = ""
    #: The article's own title, and its first three authors. Read off the record
    #: once (NCBI esummary) and carried here so the references drawer shows the
    #: paper rather than a journal name — the row a reader opens is the paper.
    title: str = ""
    authors: str = ""
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
        phenotype="heart_failure_cardiorenal",
        reported_rate="13.2%",
        journal="Journal of the American College of Cardiology",
        year="2023",
        title=(
            "Global Comparison of Readmission Rates for Patients With Heart "
            "Failure."
        ),
        authors="Foroutan F, Rayner DG, Ross HJ, et al.",
        rate_sentence=(
            "Of the 1.5 million individuals at risk, 13.2% (95% CI: 10.5%-16.1%) were "
            "readmitted within 30 days and 35.7% (95% CI: 27.1%-44.9%) within 1 year."
        ),
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
        phenotype="copd_respiratory",
        reported_rate="11%",
        journal="Respiratory Medicine",
        year="2023",
        title=(
            "Readmission rate for acute exacerbation of chronic obstructive "
            "pulmonary disease: A systematic review and meta-analysis."
        ),
        authors="Ruan H, Zhang H, Wang J, et al.",
        rate_sentence=(
            "The pooled 30-, 60-, 90-, 180-, and 365-day readmission rates were 11%, "
            "17%, 17%, 30%, and 37%, respectively."
        ),
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
        phenotype="copd_respiratory",
        reported_rate="≈20%",
        journal="American Journal of Respiratory and Critical Care Medicine",
        year="2023",
        title=(
            "Differential Diagnosis of Suspected Chronic Obstructive Pulmonary "
            "Disease Exacerbations in the Acute Care Setting: Best Practice."
        ),
        authors="Celli BR, Fabbri LM, Aaron SD, et al.",
        rate_sentence=(
            "Importantly, close to 20% of patients are readmitted within 30 days "
            "after hospital discharge, with great cost to the person and society."
        ),
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
        phenotype="spine_neurosurgery",
        reported_rate="3.8%",
        journal="North American Spine Society Journal",
        year="2026",
        title=(
            "Readmission rates after intraoperative durotomy and repair in "
            "outpatient lumbar spinal surgery."
        ),
        authors="Proal JD, Moon AS, LaHaise K, et al.",
        rate_sentence=(
            "Of the 52 patients discharged home same day, 2 were readmitted (3.8%); "
            "of the 59 patients admitted postoperatively, 5 were readmitted (8.5%)."
        ),
        cohort_summary=(
            "Outpatient lumbar spine surgery complicated by an intraoperative "
            "durotomy, repaired and discharged the same day (n=52; 8.5% in the "
            "postoperatively admitted comparator, n=59)"
        ),
        evidence_tier="Observational Cohort",
        horizon_days=30,
        matched_features=(),
    ),
    # "The 30-day readmission rate was 3.2% and 3.4% and the CMS-defined complication
    # rate was 3.8% and 2.9% for THA and TKA, respectively." — "The Impact of
    # Pre-Operative Healthcare Utilization on Complications, Readmissions, and
    # Post-Operative Healthcare Utilization Following Total Joint Arthroplasty",
    # The Journal of Arthroplasty, 2022.
    #
    # The knee figure is the row and the hip figure is named in the summary, for the
    # same reason the durotomy comparator is: one study is one row, so a reader cannot
    # total two numbers that a paper reported as alternatives.
    "34793857": CohortBenchmark(
        pmid="34793857",
        phenotype="orthopedic_arthroplasty",
        reported_rate="3.4%",
        journal="The Journal of Arthroplasty",
        year="2022",
        title=(
            "The Impact of Pre-Operative Healthcare Utilization on Complications, "
            "Readmissions, and Post-Operative Healthcare Utilization Following "
            "Total Joint Arthroplasty."
        ),
        authors="Creager AE, Kleven AD, Kesimoglu ZN, et al.",
        rate_sentence=(
            "The 30-day readmission rate was 3.2% and 3.4% and the CMS-defined "
            "complication rate was 3.8% and 2.9% for THA and TKA, respectively."
        ),
        cohort_summary=(
            "Elective primary total knee arthroplasty (n=2,159; 3.2% in the total hip "
            "comparator, n=1,785), single institution, 2015-2019"
        ),
        evidence_tier="Retrospective Cohort",
        horizon_days=30,
        # The cohort's own finding is about prior NON-ELECTIVE utilisation: "presence
        # of any pre-HU was associated with increased risk of 30-day readmission (OR
        # 2.85)" for THA and "≥2 pre-HU events" (OR 3.52) for TKA, where pre-HU is
        # emergency department, urgent care, observation or inpatient admission in the
        # 90 days before surgery. Two of this patient's features measure that concept,
        # so they are named here. The windows need not match for the pill to be honest —
        # it says the cohort identified prior utilisation as a driver, which it did —
        # but the pill is not a claim that the two are computed the same way.
        matched_features=("recent_ed_visits", "prior_admission_count"),
    ),
    # "The overall 30-day unplanned readmission rate of these 33,840 patients was
    # 5.5%." — "Prolonged Length of Stay and Risk of Unplanned 30-Day Readmission After
    # Elective Spine Surgery: Propensity Score-Matched Analysis of 33,840 Patients",
    # Spine, 2020.
    "32341301": CohortBenchmark(
        pmid="32341301",
        phenotype="spine_neurosurgery",
        reported_rate="5.5%",
        journal="Spine",
        year="2020",
        title=(
            "Prolonged Length of Stay and Risk of Unplanned 30-Day Readmission "
            "After Elective Spine Surgery: Propensity Score-Matched Analysis of "
            "33,840 Patients."
        ),
        authors="Kim RB, Wilkerson C, Karsy M, et al.",
        rate_sentence=(
            "The overall 30-day unplanned readmission rate of these 33,840 patients "
            "was 5.5%."
        ),
        cohort_summary=(
            "Elective spine surgery, ACS-NSQIP, 16,920 propensity score-matched pairs "
            "drawn from 99,575 patients"
        ),
        evidence_tier="Observational Cohort",
        horizon_days=30,
        # "Multivariate analysis of the propensity score-matched population ... showed
        # pLOS was associated with an increased risk of 30-day unplanned readmission
        # (odds ratio [OR] 1.423 ...)", where pLOS is a stay of 4 days or more. Length
        # of stay is a feature this patient has, so the pill names it.
        matched_features=("index_los_days",),
    ),
    # "UGIB had a 30-day all-cause readmission rate of 17.4% (95% confidence interval
    # [CI] 16.7-18.2%), stratification revealed a higher rate for variceal UGIB [19.6%
    # (95% CI 17.6-21.5%)] than non-variceal [16.8% (95% CI 16.0-17.5%)]." — "30-day
    # readmission rates among upper gastrointestinal bleeds: A systematic review and
    # meta-analysis", Journal of Gastroenterology and Hepatology, 2023.
    #
    # The overall pooled rate is the row; the variceal split is named in the summary
    # rather than given rows, because the strata are the same 70 studies.
    "36808766": CohortBenchmark(
        pmid="36808766",
        phenotype="gi_hepatobiliary_surgery",
        reported_rate="17.4%",
        journal="Journal of Gastroenterology and Hepatology",
        year="2023",
        title=(
            "30-day readmission rates among upper gastrointestinal bleeds: A "
            "systematic review and meta-analysis."
        ),
        authors="Dunne CL, Kaur S, Delacruz B, et al.",
        rate_sentence=(
            "UGIB had a 30-day all-cause readmission rate of 17.4% (95% confidence "
            "interval [CI] 16.7-18.2%), stratification revealed a higher rate for "
            "variceal UGIB [19.6% (95% CI 17.6-21.5%)] than non-variceal [16.8% "
            "(95% CI 16.0-17.5%)]."
        ),
        cohort_summary=(
            "Upper gastrointestinal bleeding, pooled across 70 studies and 4,292,714 "
            "patients; certainty of evidence low to very low for all outcomes"
        ),
        evidence_tier="Meta-Analysis",
        horizon_days=30,
        matched_features=(),
    ),
    # "The 30-day readmission rate was 18.4% (5645 patients), the 30-day death rate was
    # 3.7% (1146 patients), and the rate of either outcome was 19.7% (6066 patients)."
    # — "Variation in Acute Care Rehabilitation and 30-Day Hospital Readmission or
    # Mortality in Adult Patients With Pneumonia", JAMA Network Open, 2020.
    #
    # A cohort description inside a study whose question is about therapy, which is why
    # the tier says cohort and not trial: the rate is the cohort's own, and the
    # intervention arm it was collected for is not what this row claims.
    "32886119": CohortBenchmark(
        pmid="32886119",
        phenotype="sepsis_infectious",
        reported_rate="18.4%",
        journal="JAMA Network Open",
        year="2020",
        title=(
            "Variation in Acute Care Rehabilitation and 30-Day Hospital "
            "Readmission or Mortality in Adult Patients With Pneumonia."
        ),
        authors="Freburger JK, Chou A, Euloth T, et al.",
        rate_sentence=(
            "The 30-day readmission rate was 18.4% (5645 patients), the 30-day death "
            "rate was 3.7% (1146 patients), and the rate of either outcome was 19.7% "
            "(6066 patients)."
        ),
        cohort_summary=(
            "Adults discharged alive with a primary or secondary diagnosis of "
            "pneumonia or an influenza-related condition, 30,746 patients across 12 "
            "acute care hospitals"
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


#: Evidence tiers, strongest first, for the order the matrix draws them in. The tiers are
#: free text on each row rather than an enum, so a rank that is missing degrades to the
#: end of the table instead of dropping the row: an unfamiliar tier should still be shown,
#: just not promoted above a meta-analysis.
_TIER_RANK: dict[str, int] = {
    "Meta-Analysis": 0,
    "Systematic Review": 1,
    "Randomized Controlled Trial": 2,
    "Observational Cohort": 3,
    "Retrospective Cohort": 4,
    "Narrative Review": 5,
}


def benchmarks_for_phenotype(phenotype: str | None) -> list[dict]:
    """The curated rows for one resolved phenotype, strongest evidence first.

    Selected by the patient's condition rather than by what a search returned, which is
    the Phase 4B change and the reason the matrix now renders at all: a search's PMIDs
    are an artifact of a query, and the patient's condition is not. A phenotype with no
    rows returns an empty list, and the caller is expected to draw the estimate alone
    rather than reach for a bucket that does have rows — the one failure this whole phase
    exists to remove is a patient's figure being compared against the wrong cohort.

    Ordering is by evidence tier first, so a pooled estimate leads and a single-centre
    series follows it, with the PMID breaking ties. The order is stable because the table
    is a constant; a test holds it, since a comparison table that reorders itself between
    turns is one a reader cannot check against a previous reading.

    The citation is built here from the row's own journal and year, because selection by
    phenotype means there is no retrieved record to take them from any more. The URL is
    derived from the PMID rather than stored, so a row can never point at a record other
    than the one it cites.
    """
    if not phenotype:
        return []
    rows: list[dict] = []
    for benchmark in BENCHMARKS.values():
        if benchmark.phenotype != phenotype:
            continue
        if benchmark.horizon_days != MODEL_HORIZON_DAYS:
            # Unreachable with today's rows, and it stays because the rule is the point:
            # the horizon filter is applied at selection, not trusted to curation.
            continue
        rows.append({
            "pmid": benchmark.pmid,
            "source": " · ".join(
                part for part in (benchmark.journal, benchmark.year) if part
            ),
            # The paper itself, so the row and the card can name it. `source` is the
            # citation tram: a journal and a year locate the record, and the reference
            # that a reader opens is the article.
            "title": benchmark.title,
            "authors": benchmark.authors,
            # The parts `source` joins, kept as themselves: a citation line reads
            # `Journal. Year. Authors.` and a pre-joined string cannot be punctuated
            # back into that without guessing where the join was.
            "journal": benchmark.journal,
            "year": benchmark.year,
            "reported_rate": benchmark.reported_rate,
            "cohort_summary": benchmark.cohort_summary,
            "matched_features": list(benchmark.matched_features),
            "evidence_tier": benchmark.evidence_tier,
            "horizon_days": benchmark.horizon_days,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{benchmark.pmid}/",
            # The sentence the rate was read from, which is what the matrix shows: the
            # figure and its provenance in the same text, so a reader can check one
            # against the other without leaving the page.
            "excerpt": benchmark.rate_sentence,
        })
    rows.sort(key=lambda row: (_TIER_RANK.get(row["evidence_tier"], len(_TIER_RANK)),
                               row["pmid"]))
    return rows
