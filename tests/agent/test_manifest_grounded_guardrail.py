"""The disposition guardrail, and the two structural rules it needed.

Two defects met in one function on 2026-11-05 (`dev_t2_01` turn 1, `dev_t5_01` turn 1):

1. **The evidence channel was the note text alone.** `discharge_location` is an
   administrative fact no discharge summary declares — which is why the manifest is
   hydrated before the model runs — so a check that reads only retrieved passages
   deletes the sentence that states it. The model wrote "discharged to home with home
   health" from the block on both turns; both sentences were removed as invented.
2. **The filter damaged structure it was not judging.** A section label written as
   `**Discharge plan & medications.** The patient was discharged …` is one split unit,
   because the character before the space is `*` rather than `.` — so deleting the
   sentence deleted the label with it, and the `/ehr` briefing was served three headings
   out of four while the model had written all four. The old joiner also collapsed every
   paragraph break in the answer whenever a single sentence was removed.

These tests pin the evidence channel and both structural rules.
"""

from __future__ import annotations

import pytest

from services.agent.postprocess import guardrail as g
from services.agent.presentation.a2ui import compose_presentation

MANIFEST = (
    '<admission_metadata hadm_id="1" source="feature store row scored by the '
    'readmission model">\n'
    "- age: 48\n"
    "- admission_type: Urgent (URGENT)\n"
    "- discharge_location: Home with home health\n"
    "</admission_metadata>"
)


def _note(text: str) -> list[dict]:
    return [{"name": "rag_search", "args": {"hadm_id": 1},
             "response": {"returned": 1, "passages": [
                 {"section": "brief_hospital_course", "text": text, "cite": 1}]}}]


#: A note that states nothing about where the patient went.
NO_DISPOSITION_NOTE = _note(
    "The patient remained medically stable throughout. The operation was uncomplicated "
    "and she was seen by physiotherapy on the second postoperative day."
)


# ---------------------------------------------------------------------------
# 1. The manifest is evidence
# ---------------------------------------------------------------------------


def test_a_disposition_the_manifest_states_is_kept():
    """The sentence is a quotation from a block the model was shown, not an invention."""
    answer = ("The patient was discharged to home with home health ^[1].")

    out = g.guard_answer(answer, NO_DISPOSITION_NOTE,
                         state={"admission_context": MANIFEST})

    assert out["answer"] == answer, out
    assert not any(f.startswith("disposition_") for f in out["flags"]), out["flags"]


def test_the_same_sentence_is_still_removed_without_the_manifest():
    """The guard has not been weakened: with no block, the claim is unsupported and goes.

    This is the pairing that makes the first test mean something. `home health` is not
    in the note, so a turn that had not been shown the block is making the claim up.
    """
    answer = "The patient was discharged to home with home health ^[1]."

    out = g.guard_answer(answer, NO_DISPOSITION_NOTE)

    assert "home health" not in out["answer"].lower(), out
    assert "disposition_unsupported:home" in out["flags"], out["flags"]


def test_a_manifest_that_states_another_destination_does_not_vouch_for_this_one():
    """Support is per category, so a block saying `home` cannot excuse `hospice`."""
    manifest = MANIFEST.replace("Home with home health", "Hospice — comfort care only")
    answer = "The patient was discharged to a hospice ^[1]."

    out = g.guard_answer(answer, NO_DISPOSITION_NOTE, state={"admission_context": manifest})

    assert "hospice" in out["answer"].lower(), out
    # And the same answer against a block that says home is unsupported again.
    out_home = g.guard_answer(answer, NO_DISPOSITION_NOTE,
                              state={"admission_context": MANIFEST})
    assert "hospice" not in out_home["answer"].lower(), out_home


# ---------------------------------------------------------------------------
# 2. A section label survives its opening sentence
# ---------------------------------------------------------------------------

HEADED = (
    "**Presentation & diagnosis.** She presented with hip pain.\n\n"
    "**Discharge plan & medications.** She was discharged to a skilled nursing "
    "facility. She was prescribed warfarin 2.5 mg daily."
)


def test_a_section_label_moves_to_the_next_sentence_when_its_own_is_removed():
    """The label belongs to the section, not to the sentence that was deleted."""
    out = g.verify_dispositions(
        "**Discharge plan & medications.** She was discharged to a skilled nursing "
        "facility. She was prescribed warfarin 2.5 mg daily.",
        "warfarin 2.5 mg daily",
    )

    kept, flags = out
    assert kept == ("**Discharge plan & medications.** She was prescribed warfarin "
                    "2.5 mg daily."), kept
    assert "disposition_unsupported:skilled_nursing" in flags, flags


def test_the_label_goes_when_nothing_survives_under_it():
    kept, flags = g.verify_dispositions(
        "**Discharge plan & medications.** She was discharged to a skilled nursing "
        "facility.",
        "",
    )

    assert kept == "", kept
    assert flags


def test_paragraphs_survive_a_removal_in_another_section():
    """One removed sentence used to turn every section of a briefing into one block."""
    kept, flags = g.verify_dispositions(HEADED, "")

    assert flags
    assert "presented with hip pain" in kept
    assert "\n\n" in kept, kept
    assert kept.split("\n\n")[0].startswith("**Presentation & diagnosis.**")


def test_a_clean_answer_is_returned_byte_for_byte():
    """A pass that fires nothing may not rewrite passing prose."""
    answer = HEADED.replace("skilled nursing facility", "home")

    assert g.verify_dispositions(answer, "home") [0] == answer


# ---------------------------------------------------------------------------
# 3. A manifest-answered turn is grounded without a citation
# ---------------------------------------------------------------------------


def test_context_grounding_suppresses_nothing():
    """`/labs` and `/meds` are exempt from the citation requirement by name; a
    manifest-answered turn needs the same exemption for the same reason — it rests on
    something that publishes no passage (`dev_t7_01` turn 2)."""
    calls = [{"name": "rag_search_sections", "args": {"hadm_id": 1},
              "response": {"returned": 1, "passages": [
                  {"section": "discharge_summary", "text": "A visit.", "cite": 1}]}}]
    out = compose_presentation(
        "What was the length of stay?", "This record is a visit, so it has none.",
        calls, None, None, context_grounded=True,
    )

    assert out["verified"] is True
    assert out["scope"] == "patient_chart"
    assert out["scope_suppressed"] is False


def test_without_context_grounding_the_same_answer_is_still_suppressed():
    """The exemption is the flag, not the prose shape."""
    calls = [{"name": "rag_search_sections", "args": {"hadm_id": 1},
              "response": {"returned": 1, "passages": [
                  {"section": "discharge_summary", "text": "A visit.", "cite": 1}]}}]
    out = compose_presentation(
        "What was the length of stay?", "This record is a visit, so it has none.",
        calls, None, None,
    )

    assert out["verified"] is False
    assert out["scope"] is None
    assert out["scope_suppressed"] is True
