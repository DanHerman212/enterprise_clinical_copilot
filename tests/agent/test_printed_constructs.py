"""The served prose may carry only the syntax the browser can draw.

Two chip follow-ups on 2026-11-01 served, inside the answer's prose, the construct this
package prints into a tool result for the model to read:

  * an `/ehr` answer ended with `[CITATION ^[1] | CLINICAL NOTE SECTION —
    discharge_instructions]`, a verbatim copy of the copyable token;
  * a risk-chip answer ended with two of them, labelled `ADMISSION METADATA` and
    `PREDICT_READMISSION`. No code path prints either label: the model had generalised
    the grammar it was shown.

Neither was caught, because the settlement vocabulary knew only `^[...]` and the wrapper
around a marker was prose to every pass. `citations.citation_header` no longer prints a
bracket, so a compliant copy is now drawable by construction; these tests are the other
half of the contract, the half that does not depend on the model complying.
"""

from services.agent.postprocess import answer as answer_text
from services.agent.postprocess import citations

DASH = "\u2014"

#: One sample per entry in `citations.PRINTED_FORMS`. The sets are asserted equal, so a
#: print form added without a normaliser fails here rather than in front of a reader.
SAMPLES = {
    "cite_header": (
        "The patient takes oxycodone SR 20 mg every 12 hours ^[1].\n\n"
        f"[CITATION ^[1] | CLINICAL NOTE SECTION {DASH} discharge_instructions]"
    ),
    "admission_metadata": (
        "Admission type is emergency.\n\n"
        '<admission_metadata hadm_id="90000015" source="feature store row">\n'
        "- admission_type: Emergency (EMER)\n"
        "</admission_metadata>"
    ),
    "zero_unit_note": "The section lists nothing.\n\n" + citations.ZERO_UNIT_NOTE,
}


def test_every_printed_form_has_a_normaliser():
    """The coupling: the printer's inventory and this fixture set are the same set."""
    assert set(SAMPLES) == set(citations.PRINTED_FORMS)
    for name, text in SAMPLES.items():
        prose, flags = citations.strip_printed_constructs(text)
        assert f"citation_printed_construct:{name}" in flags
        assert "CITATION" not in prose
        assert "<admission_metadata" not in prose
        assert citations.ZERO_UNIT_NOTE not in prose


def test_the_printed_token_is_the_marker_and_nothing_else():
    """What the model is asked to copy is exactly what the renderer can draw."""
    modality = citations.note_modality("brief_hospital_course")
    fields = citations.citation_fields(6, modality)

    assert citations.citation_header(6, modality) == "^[6]"
    assert fields == {
        "cite": 6,
        "cite_modality": modality,
        "cite_header": "^[6]",
    }


def test_a_copied_header_loses_its_bracket_and_keeps_its_marker():
    prose, flags = citations.strip_printed_constructs(SAMPLES["cite_header"])

    assert "[CITATION" not in prose
    assert prose.endswith("^[1].")
    assert flags == ["citation_printed_construct:cite_header"]


def test_an_invented_label_is_the_same_grammar():
    """The risk-chip answer: labels no code path prints, handled by no new rule."""
    prose, flags = citations.strip_printed_constructs(
        "The patient's admission type is an emergency, direct admission ^[1]. "
        "The readmission model identifies this admission type as a factor that "
        "increases the patient's readmission risk ^[2].\n\n"
        "[CITATION ^[1] | ADMISSION METADATA] [CITATION ^[2] | PREDICT_READMISSION]"
    )

    assert "CITATION" not in prose
    assert "ADMISSION METADATA" not in prose
    assert "PREDICT_READMISSION" not in prose
    assert prose.count("^[1]") == 1
    assert prose.count("^[2]") == 1
    assert flags == ["citation_printed_construct:cite_header"]


def test_a_bracket_around_a_marker_is_machine_syntax_whatever_the_label():
    """The grammar rule, exercised with a wrapper the code has never printed."""
    prose, flags = citations.strip_printed_constructs(
        "The patient improved ^[2].\n\n[NOTE SOURCE ^[2]]"
    )

    assert "[NOTE SOURCE" not in prose
    assert prose == "The patient improved ^[2]."
    assert flags == ["citation_printed_construct:cite_header"]


def test_a_header_citing_a_passage_the_prose_never_cited_keeps_the_marker():
    """The bracket is annotation; the number inside it is the citation, and it survives."""
    prose, _ = citations.strip_printed_constructs(
        "The patient was admitted emergently.\n\n"
        f"[CITATION ^[7] | CLINICAL NOTE SECTION {DASH} brief_hospital_course]"
    )

    assert prose == "The patient was admitted emergently. ^[7]"


def test_a_header_does_not_double_cite_a_passage():
    """The prompt allows one citation per passage; a copy of it is not a second one."""
    prose, _ = citations.strip_printed_constructs(
        "The dose was tapered ^[3].\n\n"
        f"[CITATION ^[3] | CLINICAL NOTE SECTION {DASH} discharge_instructions]"
    )

    assert prose == "The dose was tapered ^[3]."


def test_a_clean_answer_is_untouched():
    text = ("The patient was admitted emergently ^[1]. No further detail is recorded "
            "in the notes ^[2].")

    assert citations.strip_printed_constructs(text) == (text, [])


def test_settlement_serves_clean_prose_and_reports_the_construct():
    """The wiring: settlement runs the pass and the violation reaches `guardrail_flags`."""
    settled = answer_text.settle_answer(
        prose=("The dose was tapered ^[1].\n\n"
               f"[CITATION ^[1] | CLINICAL NOTE SECTION {DASH} discharge_instructions]"),
        deterministic=None,
        tool_calls=[{"name": "rag_search", "response": {
            "passages": [{"section": "discharge_instructions", "text": "taper"}]}}],
        replayed_tool_calls=[],
    )

    assert settled.prose == "The dose was tapered ^[1]."
    assert "citation_printed_construct:cite_header" in settled.flags
