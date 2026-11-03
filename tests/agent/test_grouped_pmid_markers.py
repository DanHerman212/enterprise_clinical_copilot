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

from services.agent.postprocess.citations import (
    collapse_repeated_citations,
    drop_unknown_markers,
    enforce_citations,
    renumber_citations,
)

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


class TestAGroupedBadgeSurvivesSettlement:
    """The lapses that motivated the normaliser, all seen or reasoned from 2026-11-03.

    Two cohorts in one sentence produced either no badge at all or a badge naming both.
    The first is silent data loss — `drop_unknown_markers` does not recognise
    `^[PMID: 111, PMID: 222]`, so it deleted it and the paragraph stated two published
    rates with nothing to open. The second reached the browser as raw markup, because one
    badge can only be one control.
    """

    def test_a_badge_repeating_the_label_is_split_into_one_marker_per_record(self):
        answer = "The rates were 4.1% and 3.7%.^[PMID: 35985540, PMID: 33534292]"

        kept, flags = enforce_citations(answer, 0, {"35985540", "33534292"})

        assert kept == (
            "The rates were 4.1% and 3.7%."
            "^[PMID: 35985540] ^[PMID: 33534292]"
        )
        assert flags == []

    def test_a_badge_with_one_label_and_two_ids_is_split_too(self):
        answer = "The rates were 4.1% and 3.7%.^[PMID: 35985540, 33534292]"

        kept, flags = enforce_citations(answer, 0, {"35985540", "33534292"})

        assert kept == (
            "The rates were 4.1% and 3.7%."
            "^[PMID: 35985540] ^[PMID: 33534292]"
        )
        assert flags == []

    def test_the_repeated_label_form_is_not_treated_as_an_unknown_marker(self):
        """`drop_unknown_markers` runs at settlement too, and used to delete this shape."""
        answer = "The rates were 4.1% and 3.7%.^[PMID: 35985540, PMID: 33534292]"

        assert drop_unknown_markers(answer) == answer

    def test_two_distinct_badges_in_one_sentence_both_survive(self):
        answer = (
            "The first cohort reported 4.1%^[PMID: 35985540] and the second 3.7%"
            "^[PMID: 33534292]."
        )

        kept, flags = enforce_citations(answer, 0, {"35985540", "33534292"})

        assert kept == answer
        assert flags == []

    def test_two_distinct_badges_are_not_collapsed_as_a_repeated_citation(self):
        """The run-collapse rule is about a repeated note passage, not about cohorts."""
        answer = (
            "The first cohort reported 4.1%^[PMID: 35985540] and the second 3.7%"
            "^[PMID: 33534292]."
        )

        assert collapse_repeated_citations(answer) == (answer, [])

    def test_the_split_badges_carry_no_note_number(self):
        """A publication is not a position: the split must not invent one."""
        kept, _ = enforce_citations(
            "Rates.^[PMID: 35985540, PMID: 33534292]", 0, {"35985540", "33534292"})

        assert "^[1]" not in kept
