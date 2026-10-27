"""The citation-marker check must accept the channel the product actually writes.

`dev_t1_06` failed Gate 2 on a correctly formatted `/pubmed` answer. The check searched
the prose for the literal `"PMID:"`; since Phase F the prose carries numbered pills
(`^[1]`, `^[3]`) and the PMID lives on the source the pill addresses — which is the thing
a reader clicks.

These tests pin both directions: a resolvable citation passes, and a marker that addresses
nothing still fails. The second half is the point of the check — a marker past the end of
the array is a citation the reader cannot open.
"""

from __future__ import annotations

import pytest

from evaluation.platform.evaluators import structural_check

TURNS = {"turns": ["/pubmed dual antiplatelet therapy after coronary stenting"]}


def _output(answer: str, sources: list[dict]) -> dict:
    return {
        "turn_1_response": answer,
        "turn_responses": [answer],
        "turn_sources": [sources],
    }


SOURCE_WITH_PMID = {"cite": 1, "section": "Literature", "text": "DAPT trial",
                    "kind": "Literature", "pmid": "35100001",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/35100001/"}


def _expectations(marker: str = "PMID") -> dict:
    return {"expectations": {"citation_marker": marker}}


def test_a_literal_pmid_in_the_prose_still_passes():
    ok, failures = structural_check(
        _expectations(), _output("See PMID: 35100001.", [SOURCE_WITH_PMID]))
    assert ok, failures


def test_a_resolvable_inline_marker_passes_without_the_literal():
    answer = "Short DAPT then P2Y12 monotherapy reduced bleeding ^[1]."
    ok, failures = structural_check(_expectations(), _output(answer, [SOURCE_WITH_PMID]))
    assert ok, failures


def test_a_marker_that_addresses_no_source_fails():
    answer = "Short DAPT then P2Y12 monotherapy reduced bleeding ^[2]."
    ok, failures = structural_check(_expectations(), _output(answer, [SOURCE_WITH_PMID]))
    assert not ok
    assert any("addresses no source" in failure for failure in failures), failures


def test_an_answer_with_no_marker_and_no_literal_fails():
    ok, failures = structural_check(
        _expectations(), _output("DAPT was discussed.", [SOURCE_WITH_PMID]))
    assert not ok
    assert any("no PMID citation resolvable" in failure for failure in failures), failures


def test_a_marker_whose_source_carries_no_identifier_fails():
    """A note passage is not a PMID citation, however well it resolves."""
    note = {"cite": 1, "section": "discharge_diagnosis", "text": "CAD",
            "query": "diagnosis"}
    ok, failures = structural_check(
        _expectations(), _output("As documented ^[1].", [note]))
    assert not ok
    assert any("no PMID citation resolvable" in failure for failure in failures), failures


def test_a_label_naming_the_marker_counts():
    """The directive allows the identifier to live in the label instead of a field."""
    labelled = {"cite": 1, "section": "Literature", "text": "Trial",
                "label": "1 · PMID 35100001 · JACC", "url": "https://example.invalid"}
    ok, failures = structural_check(
        _expectations(), _output("As published ^[1].", [labelled]))
    assert ok, failures


@pytest.mark.parametrize("marker", ["PMID", "pmid"])
def test_the_marker_name_is_matched_case_insensitively_on_the_label(marker):
    labelled = {"cite": 1, "section": "Literature", "text": "Trial",
                "label": "1 · PMID 35100001 · JACC", "url": "https://example.invalid"}
    ok, failures = structural_check(
        _expectations(marker), _output("As published ^[1].", [labelled]))
    assert ok, failures


def test_every_marker_must_resolve_not_only_one():
    answer = "Two articles support this ^[1], ^[2]."
    ok, failures = structural_check(_expectations(), _output(answer, [SOURCE_WITH_PMID]))
    assert not ok
    assert any("addresses no source" in failure for failure in failures), failures


def test_the_runner_carries_the_pmid_into_the_envelope():
    """The assertion can only read what the task captured.

    `_sources` narrows each entry for the judge, and the narrowing is applied before the
    envelope is built. Without the identifier the live check had nothing to resolve a
    literature marker against, however correct the answer was — its own unit test passed
    because that test builds the envelope by hand with `pmid` already present.
    """
    from evaluation.platform.task import _sources

    kept = _sources({"sources": [SOURCE_WITH_PMID,
                                 {"cite": 2, "section": "discharge_diagnosis",
                                  "text": "CAD", "query": "diagnosis"}]})

    assert kept[0]["pmid"] == "35100001"
    assert kept[0]["text"] == "DAPT trial"
    # A note passage carries no identifier, and none is invented for it.
    assert "pmid" not in kept[1]
