"""Concept tagging for discharge notes, with assertion handling.

Runs ONLY under .venv-nlp (medspaCy pulls numpy 2.x; the harness venv is pinned
to numpy 1.26 for the model stack). Nothing in mcp_server or agent imports this
module — tags are precomputed at ingestion time and stored, never computed at
query time.

What "assertion handling" buys us: a naive phrase search counts "denies IV drug
use" as substance use. medspaCy's ConText classifies each mention as negated,
hypothetical ("call your doctor if you miss a dose"), historical, or attributed
to someone else ("father had CHF"), so the probe can count only mentions that
are actually about this patient, now or in their past.

Positivity policy (documented because it is a judgment call):
    positive = not negated, not hypothetical, not about family.
    Historical mentions COUNT as positive - "history of IV drug use" is real
    readmission signal even when remote - but the flag is preserved on every
    mention so downstream code can revisit that choice without re-tagging.

The phrase lists below are seeds, not the quality bar. The quality bar is
tests/data/concept_sentences.json - a labeled set of sentences with expected
outcomes. The implementation behind it (these rules, or a model later) is
swappable; the labeled set is the durable asset.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import medspacy
from medspacy.context import ConTextRule
from medspacy.ner import TargetRule

# Phrases are matched case-insensitively as whole-token sequences.
CONCEPT_RULES: dict[str, tuple[str, ...]] = {
    "non_adherence": (
        "noncompliant", "non-compliant", "noncompliance", "non-compliance",
        "nonadherent", "non-adherent", "nonadherence", "non-adherence",
        "medication noncompliance", "not taking his medications",
        "not taking her medications", "not been taking",
        "ran out of medications", "ran out of medication",
        "stopped taking", "refuses medications", "refuses medication",
        "missed doses", "miss a dose", "self-discontinued",
    ),
    "missed_followup": (
        "missed several appointments", "missed appointments",
        "missed multiple appointments", "missed outpatient appointments",
        "lost to follow up", "lost to follow-up",
        "did not follow up", "no follow-up scheduled", "no follow up scheduled",
        "no pcp", "does not have a pcp", "no primary care",
        "no-show", "no show",
    ),
    "ama": (
        "against medical advice", "left ama", "eloped",
    ),
    "substance_use": (
        "alcohol abuse", "alcohol dependence", "alcohol use disorder",
        "alcoholism", "etoh abuse", "etoh dependence", "alcohol use",
        "heroin use", "iv drug use", "ivdu", "intravenous drug use",
        "cocaine use", "opioid use disorder", "opiate abuse",
        "substance abuse", "substance use disorder", "polysubstance",
        "drug abuse", "heavy drinking", "drinks per day",
    ),
    "functional_decline": (
        "lives alone", "homeless", "no caregiver", "without a caregiver",
        "needs assistance", "requires assistance", "unable to care for",
        "functional decline", "failure to thrive", "deconditioned",
        "frequent falls", "recurrent falls", "multiple falls",
        "unsteady gait", "poor mobility",
    ),
    "polypharmacy": (
        "polypharmacy", "complex medication regimen",
        "complicated medication regimen", "complex regimen",
    ),
    "goals_of_care": (
        "goals of care", "palliative", "hospice", "comfort measures",
        "comfort care", "do not resuscitate", "dnr/dni", "end of life",
        "end-of-life",
    ),
    "hedging": (
        "i am concerned", "concerned about", "concern about", "concern for",
        "guarded prognosis", "prognosis is guarded", "prognosis remains guarded",
        "tenuous", "high risk of readmission", "high risk for readmission",
        "poor prognosis",
    ),
}

# Phrases where the absence IS the concept: "does not have a PCP" is the risk
# signal, not a negated mention of having one. ConText correctly reads these as
# negated grammar, so tag() clears the flag for exact matches of these phrases.
ABSENCE_IS_THE_SIGNAL: frozenset[str] = frozenset({
    "no pcp", "does not have a pcp", "no primary care",
    "no follow-up scheduled", "no follow up scheduled",
    "did not follow up", "no-show", "no show",
    "no caregiver", "without a caregiver",
})

# Hedging is reported but never counted toward the gate's pass/fail (settled
# 2026-08-04), and only inside sections where a clinician is assessing the
# patient rather than instructing them. Enforced by the caller, which owns
# section context; recorded here so the policy has one home.
GATING_CONCEPTS: tuple[str, ...] = tuple(
    name for name in CONCEPT_RULES if name != "hedging"
)
HEDGING_SECTIONS: tuple[str, ...] = ("discharge_condition", "brief_hospital_course")


@dataclass(frozen=True)
class ConceptMention:
    """One concept found in text, with ConText's assertion flags."""

    concept: str
    matched_text: str
    sentence: str
    negated: bool
    hypothetical: bool
    historical: bool
    family: bool

    @property
    def is_positive(self) -> bool:
        """True if this mention should count as risk signal (see module doc)."""
        return not (self.negated or self.hypothetical or self.family)


@lru_cache(maxsize=1)
def get_nlp():
    """Build the medspaCy pipeline once; ~seconds, so cache it."""
    # PyRuSH logs every sentence split at DEBUG via loguru; at corpus scale
    # that is millions of lines of I/O.
    from loguru import logger
    logger.disable("PyRuSH")

    nlp = medspacy.load()
    rules = [
        TargetRule(phrase, concept)
        for concept, phrases in CONCEPT_RULES.items()
        for phrase in phrases
    ]
    nlp.get_pipe("medspacy_target_matcher").add(rules)
    nlp.get_pipe("medspacy_context").add([
        # "family meeting" is a care-planning event, not attribution to a
        # relative; the PSEUDO rule stops the bare "family" cue firing inside it.
        ConTextRule("family meeting", "PSEUDO_FAMILY", direction="PSEUDO"),
        # "prior discharge was AMA": ConText's defaults miss bare "prior".
        ConTextRule("prior", "HISTORICAL", direction="FORWARD"),
    ])
    return nlp


def tag(text: str) -> list[ConceptMention]:
    """All concept mentions in `text`, positive or not.

    Callers filter on `is_positive`; returning everything keeps the negated and
    hypothetical mentions visible for debugging and for the labeled tests.
    """
    if not text or not text.strip():
        return []
    doc = get_nlp()(text)
    return [
        ConceptMention(
            concept=ent.label_,
            matched_text=ent.text,
            sentence=ent.sent.text.strip(),
            negated=(ent._.is_negated
                     and ent.text.lower() not in ABSENCE_IS_THE_SIGNAL),
            hypothetical=ent._.is_hypothetical,
            historical=ent._.is_historical,
            family=ent._.is_family,
        )
        for ent in doc.ents
    ]


def positive_concepts(text: str) -> set[str]:
    """The distinct concepts positively present in `text`."""
    return {m.concept for m in tag(text) if m.is_positive}
