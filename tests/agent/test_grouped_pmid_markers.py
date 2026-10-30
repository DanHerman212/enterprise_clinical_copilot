"""A badge naming several records is one marker the browser cannot draw.

The model writes `^[PMID: 39559360, 37028984, 36774746]` when one sentence rests on
three papers. `_KNOWN_MARKER_RE` accepts that form, so the marker survived the unknown-
marker filter; `_PMID_BODY_RE` is a `fullmatch` on a single identifier, so the group
failed there, failed the numeric test too, and fell through to "leave the marker exactly
as written". It reached the reader as raw markup — `MARKER` in the console requires one
number and one closing bracket — while every machine check reported the turn as clean,
because a marker that is left alone raises no flag.

The split belongs here rather than in the console, for the reason the numeric branch
already splits `^[1, 9]` here: one marker per unit is the only shape the browser can turn
into a control, and only this layer knows which identifiers name records the turn may
cite. `test_the_grouped_form_is_untouched_by_renumbering` pins the other half — the badge
is not a note citation and the renumberer must keep its hands off it.
"""

from services.agent.postprocess.citations import enforce_citations, renumber_citations

# The three records from the live turn of 2026-10-30, in the order the model wrote them.
GROUPED = "^[PMID: 39559360, 37028984, 36774746]"
ALL_THREE = frozenset({"39559360", "37028984", "36774746"})


class TestGroupedBadgeExpansion:
    def test_every_record_that_resolves_gets_its_own_marker(self):
        """One badge becomes three, because three controls is what the reader can open.

        Left whole, the group is either raw text (no match) or, if the console ever
        learned the grouped shape, one control that opens a single record while naming
        three.
        """
        answer = f"Two trials agree.{GROUPED}"

        kept, flags = enforce_citations(answer, 0, ALL_THREE)

        assert kept == (
            "Two trials agree."
            "^[PMID: 39559360] ^[PMID: 37028984] ^[PMID: 36774746]"
        )
        assert flags == []

    def test_the_records_that_do_not_resolve_lose_their_marker_and_raise_the_flag(self):
        """Partial resolution, mirroring the numeric cluster's rule.

        A badge may name a record the turn was not shown — here the third — and that
        one goes while the two that did resolve stay. Silently keeping it would print
        an identifier nobody retrieved; dropping the whole marker would delete two
        correct citations to punish one wrong one.
        """
        answer = f"Two trials agree.{GROUPED}"

        kept, flags = enforce_citations(answer, 0, {"39559360", "37028984"})

        assert kept == "Two trials agree.^[PMID: 39559360] ^[PMID: 37028984]"
        assert flags == ["citation_out_of_range:^36774746"]

    def test_a_group_that_resolves_nothing_goes_whole(self):
        """The case the empty allowance produces, and why the two fixes ship together.

        Before the replay widening, a declared literature follow-up had an empty
        allowance. Splitting the group without widening it would have deleted all three
        markers and served the sentence with no citation at all — the raw markup was
        the *symptom* of an allowance that knew nothing, not a separate defect.
        """
        answer = f"Two trials agree.{GROUPED}"

        kept, flags = enforce_citations(answer, 0)

        assert kept == "Two trials agree."
        assert flags == [
            "citation_out_of_range:^39559360",
            "citation_out_of_range:^37028984",
            "citation_out_of_range:^36774746",
        ]

    def test_a_record_named_twice_gets_one_marker(self):
        """A repeated identifier is one record, so it is one control."""
        answer = "A trial agrees.^[PMID: 39559360, 39559360]"

        kept, flags = enforce_citations(answer, 0, {"39559360"})

        assert kept == "A trial agrees.^[PMID: 39559360]"
        assert flags == []

    def test_a_single_badge_is_left_byte_for_byte(self):
        """The ungrouped form is already correct, so nothing about it is rewritten.

        Including the spacing: `^[PMID:39559360]` has no space after the colon, and a
        normalising pass would change text that was never wrong. Checked because the
        expansion above is the kind of edit that starts rewriting passing answers.
        """
        answer = "A trial agrees.^[PMID:39559360]"

        kept, flags = enforce_citations(answer, 0, {"39559360"})

        assert kept == answer
        assert flags == []

    def test_a_grouped_badge_is_expanded_beside_a_note_citation(self):
        """Two schemes in one sentence: the note keeps its number, the badges split."""
        answer = "The note says this^[2] and three trials agree.^[PMID: 39559360, 37028984]"

        kept, flags = enforce_citations(answer, 3, {"39559360", "37028984"})

        assert kept == (
            "The note says this^[2] and three trials agree."
            "^[PMID: 39559360] ^[PMID: 37028984]"
        )
        assert flags == []


class TestTheGroupedFormIsStillOneMarkerToTheRenumberer:
    def test_the_grouped_form_is_untouched_by_renumbering(self):
        """Renumbering orders note passages; a badge names a record and is not one.

        The expansion in `enforce_citations` runs at settlement, after this pass, so the
        two must not both have an opinion about the grouped shape.
        """
        answer = f"Two trials agree.{GROUPED}"

        assert renumber_citations(answer) == answer
