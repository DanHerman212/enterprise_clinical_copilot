"""The section registry and the words a reader sees are one vocabulary, twice.

`services/mcp/retrieval/sections.py` decides what a section is called internally: a
canonical key that becomes part of every passage id and every citation a turn resolves.
`services/agent/presentation/a2ui.py` decides what that key is called on screen — twice,
because a references list has room for "History of present illness" and an inline pill does
not.

Three maps, and the failure this file prevents is that they drift. A key the presentation
layer has never heard of does not raise: `_section_label` turns it into its own underscores,
so a citation reads "admission diagnosis" in lower case beside ones that are capitalised,
and the pill falls back to `str(section)` in Title Case. Both are legible enough that no
test notices and a reader wonders why one badge is spelled differently from the rest.

So the registry is the source of truth and the two presentation maps must cover it. The
converse is not asserted: the presentation layer also names things that are not note
sections at all — a laboratory trajectory, for instance — and holding it to the registry
would refuse those.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.agent.presentation.a2ui import (  # noqa: E402
    _PILL_SECTION_LABELS,
    _SECTION_LABELS,
)
from services.mcp.retrieval.sections import KNOWN_HEADINGS  # noqa: E402


def test_every_canonical_section_has_a_reference_list_label():
    missing = sorted(set(KNOWN_HEADINGS) - set(_SECTION_LABELS))

    assert not missing, (
        "sections with no entry in a2ui._SECTION_LABELS — a citation from one of these "
        f"would print its own key: {missing}"
    )


def test_every_canonical_section_has_a_pill_label():
    missing = sorted(set(KNOWN_HEADINGS) - set(_PILL_SECTION_LABELS))

    assert not missing, (
        "sections with no entry in a2ui._PILL_SECTION_LABELS — the inline pill would fall "
        f"back to the raw key: {missing}"
    )


def test_the_two_presentation_maps_describe_the_same_sections():
    """A section named in one and not the other is a section that reads differently in
    the prose and in the references list, which is the drift this file exists to stop."""
    only_list = sorted(set(_SECTION_LABELS) - set(_PILL_SECTION_LABELS))
    only_pill = sorted(set(_PILL_SECTION_LABELS) - set(_SECTION_LABELS))

    assert only_list == [] and only_pill == [], (
        f"reference-list only: {only_list}; pill only: {only_pill}"
    )


def test_a_pill_label_is_title_case_and_a_list_label_is_not():
    """The pill is a badge and the list is a citation line, so they are cased differently.

    Asserted over the whole map rather than on examples, because a single lower-case entry
    beside a column of Title Case ones is exactly the kind of detail that survives review.
    """
    lower_words = {"on", "of", "and", "the", "in", "for", "to", "a", "an", "with", "by"}

    for key, pill in _PILL_SECTION_LABELS.items():
        words = pill.split()
        for position, word in enumerate(words):
            if word.isupper() or any(c.isupper() for c in word[1:]):
                continue  # an acronym or a proper name: H&P, TURP, JAMA
            if 0 < position < len(words) - 1 and word.lower() in lower_words:
                continue
            assert word[:1].isupper(), f"{key}: pill label {pill!r} is not Title Case"
