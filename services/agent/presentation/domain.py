"""Display vocabulary for the presentation layer — and nothing clinical.

What used to be here
--------------------
Until Stage Gate 2 this module also carried the comparative table's study data: a
`CohortBenchmark` dataclass, a nine-row `BENCHMARKS` dictionary keyed by PMID, a
phenotype bucket table, an `admission_id -> phenotype` map, and a keyword classifier that
resolved an admission's condition from its note. The table was curated by hand and each
row quoted the abstract sentence its figure came from, which is why it survived as long as
it did. It was still a table of published rates, compiled into the service, selected by a
keyword match, and drawn beside a patient it might not describe — and the surface could
not tell a correct cohort from a wrong one, because both are well-formed rows.

The comparative rows now come from `state["comparators"]`: retrieved live from PubMed
during the `/ehr` turn through the traced `search_literature` tool, verified against the
abstracts they were read from, and carried to this layer as data. Nothing in the
presentation layer chooses a study, and nothing here knows a PMID.

What is left is what a projection of settled state actually needs: the display precision
of a probability, the feature-label vocabulary (re-exported from `shared/labels`, where
`engine` and `postprocess` can reach it without importing this package), the shape a
literature record is drawn in, and one eligibility question.
"""

RISK_DECIMALS = 2


def display_probability(value):
    """The risk score at the precision it is displayed and described at.

    Returns None for anything that is not a number, so a caller holding no
    estimate renders "unavailable" rather than 0.0. A rounded absence would be
    the most confident wrong number in the payload.

    Callers round for DISPLAY only. A threshold decision reads the served value:
    rounding first would let presentation precision move a clinical call, and it
    would move it in the direction of looking safer at the boundary.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), RISK_DECIMALS)


# The feature label map and its two accessors live in `shared/labels.py` since
# 2026-10-29 (`00d7` Row 10): `engine/mcp_client.py` and `postprocess/followups.py`
# needed to name a feature, and importing them from this module made both packages
# depend on the presentation layer. Re-exported here so this module's own callers —
# `a2ui`, `canvas` and the tests — keep one import site.
from shared.labels import FEATURE_LABELS, humanize, label_for  # noqa: E402  (re-export)


def is_matrix_eligible(*, is_outpatient: bool | None = None) -> bool:
    """Whether a cohort comparison can describe this record at all.

    One question, and it is not a clinical one: a readmission rate is a statement about a
    hospital stay, and a visit that was never a stay has nothing for it to describe, so an
    outpatient encounter is refused. Everything else is eligible.

    It used to take a `hadm_id` and consult two frozensets — one of records that were not
    admissions, one of records in the authorization boundary with nothing servable. Both
    were corpus facts maintained by hand, and both are the kind of map that silently
    decides a clinical question. The encounter already knows whether this is a stay; the
    orchestrator passes it in, and this function asks nothing else.
    """
    return is_outpatient is not True


#: The fields every drawn literature record carries. Fixed rather than discovered: the
#: console's component schema is strict, so one missing key drops the whole card.
_LITERATURE_FIELDS = ("pmid", "title", "journal", "pub_date", "authors", "abstract", "url")


def _usable_articles(literature) -> list[dict]:
    """The articles from a literature payload, each shaped for the canvas.

    A record with no pmid is dropped: it cannot be cited or linked, which is the
    whole use of it. The remaining fields are coerced to strings rather than
    passed through, because the console's component schema is strict — one
    missing key drops the entire card, and a card that fails to draw is worse
    than a card with a blank journal.

    An empty list is the answer for three different situations — no search was
    run, the search failed, or it matched nothing — which the canvas treats the
    same and the answer text and provenance caption distinguish between.
    """
    if not isinstance(literature, dict):
        return []
    articles = literature.get("articles")
    if not isinstance(articles, list):
        return []
    usable = []
    for article in articles:
        if not isinstance(article, dict) or not article.get("pmid"):
            continue
        usable.append({
            field: article.get(field) if isinstance(article.get(field), str) else ""
            for field in _LITERATURE_FIELDS
        })
    return usable
