"""Tests for concept tagging, driven by the labeled sentence set.

These run only under .venv-nlp (medspaCy is deliberately absent from the
harness venv - numpy conflict, see rag/concepts.py). Under the harness venv the
whole module skips, so the main suite stays green:

    .venv-nlp/bin/python -m pytest tests/test_concepts.py -q
"""

import json
import sys
from pathlib import Path

import pytest

medspacy = pytest.importorskip("medspacy", reason="concept tagging runs in .venv-nlp only")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.concepts import (  # noqa: E402
    CONCEPT_RULES,
    GATING_CONCEPTS,
    positive_concepts,
    tag,
)

CASES = json.loads(
    (Path(__file__).parent / "data" / "concept_sentences.json").read_text()
)["cases"]


def _mentions_of(text: str, concept: str):
    return [m for m in tag(text) if m.concept == concept]


@pytest.mark.parametrize(
    "case", CASES, ids=[c["text"][:48] for c in CASES]
)
def test_labeled_sentence(case):
    text, concept, expect = case["text"], case["concept"], case["expect"]
    mentions = _mentions_of(text, concept)

    if expect == "none":
        assert mentions == [], f"unexpected {concept} mention: {mentions}"
        return

    assert mentions, f"no {concept} mention found"

    if expect in ("affirmed", "historical"):
        assert any(m.is_positive for m in mentions), (
            f"expected a positive mention, got {mentions}"
        )
        if expect == "historical":
            assert any(m.historical for m in mentions)
    else:  # negated / hypothetical / family: must NOT count as signal
        assert not any(m.is_positive for m in mentions), (
            f"{expect} mention wrongly counted as positive: {mentions}"
        )
        assert any(getattr(m, expect) for m in mentions)


def test_positive_concepts_summarises_distinct_concepts():
    text = ("Patient is homeless and has been noncompliant with medications. "
            "Denies alcohol use.")
    assert positive_concepts(text) == {"functional_decline", "non_adherence"}


def test_empty_text_yields_nothing():
    assert tag("") == []
    assert tag("   \n") == []
    assert positive_concepts("") == set()


def test_hedging_is_excluded_from_gating_concepts():
    assert "hedging" not in GATING_CONCEPTS
    assert set(GATING_CONCEPTS) == set(CONCEPT_RULES) - {"hedging"}


def test_labeled_set_covers_every_concept():
    """A concept without labeled cases has no quality bar at all."""
    covered = {c["concept"] for c in CASES}
    assert covered == set(CONCEPT_RULES)


def test_labeled_set_covers_every_trap_category():
    expectations = {c["expect"] for c in CASES}
    assert {"negated", "hypothetical", "historical", "family", "none"} <= expectations
