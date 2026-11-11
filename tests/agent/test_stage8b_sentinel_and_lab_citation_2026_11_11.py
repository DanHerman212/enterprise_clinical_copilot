"""Stage 8B: the three defects the 2026-11-11 Dev gate exposed, held shut.

Two of them are the same failure in two places — a machine token served as if it were
prose — and the third is a served document losing its paragraph break.

`dev_t6_01` turn 3 answered a false premise correctly and then appended the decline object
as a second paragraph. `graph.read_evidence_reply` only recognises a reply that BEGINS with
the object, so it read the whole message as prose, the decline was never registered, and
the reader was shown `{"evidence_found": false}` under a correct clinical answer. The
sentinel is now removed at settlement, by the rule that does not depend on the model
complying, and the reader's reply is still parsed through a fence when the fence is the
whole message.

`dev_t4_01` turn 3 quoted the model's own input — `top_factors[0].value`, 0.47 — and cited
it to the `monocytes` trajectory drawn from that same feature row. The generator reproduces
an anchor at the precision the analyte is REPORTED at, and monocytes report one decimal, so
the series it drew says 0.5. Two numbers, both true of their own source, and the sentence
used one to cite the other. The marker is removed where the sentence quotes none of the
series it names.

`dev_t4_02` turn 1 was served `… ^[5].**Readmission risk & trajectory.**`: the discharge
paragraph is rebuilt from sentences, and the blank line between it and the next heading was
not in any sentence.
"""

from __future__ import annotations

import pytest

from services.agent.engine import graph, prompts
from services.agent.postprocess import answer as answer_text
from services.agent.postprocess import citations, discharge_section, risk_lab_citation
from services.agent.postprocess import evidence_state as session_evidence

HADM = 90000108
FENCE = "```"

#: `dev_t6_01` turn 3, verbatim as served, with the object it appended. The answer's own
#: two sentences are correct; the artefact is what made the item a structural failure.
T6_ANSWER = (
    "The record does not document that the patient underwent a re-exploration for an "
    "anastomotic leak. The clinical course describes an elective laparoscopic Roux-en-Y "
    "gastric bypass that was tolerated well, with the patient recovering in the PACU and "
    "subsequently advancing his diet and activity levels without concerning symptoms or "
    "complications ^[1], ^[2]."
)
T6_SENTINEL = '{"evidence_found": false}'

#: `dev_t4_01` turn 2, the monocytes trajectory as the tool returned it. One decimal, and
#: the anchor it reproduces is 0.47 — so the series says 0.5.
MONOCYTE_SERIES = [0.7, 0.5, 0.5, 0.5, 0.5, 0.6]

#: The model's own inputs for that admission, from `predict_readmission`.
TOP_FACTORS = [
    {"feature": "monocytes_min", "contribution": -0.5619, "direction": "decreases",
     "value": 0.47},
    {"feature": "rdw_max", "contribution": 0.1172, "direction": "increases",
     "value": 17.5},
    {"feature": "sodium_min", "contribution": 0.1125, "direction": "increases",
     "value": 133},
]


def _prediction() -> dict:
    return {"hadm_id": HADM, "probability": 0.175919, "threshold": 0.11,
            "decision": 1, "top_factors": list(TOP_FACTORS)}


def _trend(key: str, label: str, unit: str, decimals: int, low: float, high: float,
           values: list[float]) -> dict:
    return {"key": key, "label": label, "unit": unit, "decimals": decimals,
            "reference_low": low, "reference_high": high,
            "anchored": [], "series": [{"day": float(index), "value": value,
                                        "flag": "normal"}
                                       for index, value in enumerate(values)]}


def _lab_calls() -> list[dict]:
    """The three series the failing turn cited, in the order the panel gave them.

    Position 1 is the note passage the same thread retrieved, so RDW is 2, sodium is 3 and
    monocytes is 4 — the numbering the stored `dev_t4_01` answer carries.
    """
    return [
        {"name": "rag_search_sections", "args": {}, "response": {
            "hadm_id": HADM,
            "passages": [{"id": "p-1", "section": "history_of_present_illness",
                          "text": "Presented with jaundice and weight loss."}],
        }},
        {"name": "predict_readmission", "args": {}, "response": _prediction()},
        {"name": "get_lab_trends", "args": {}, "response": {
            "hadm_id": HADM, "synthetic": True,
            "trends": [
                _trend("rdw", "RDW", "%", 1, 11.5, 14.5,
                       [12.0, 13.3, 14.2, 15.5, 16.5, 17.5]),
                _trend("sodium", "Sodium", "mmol/L", 0, 135.0, 145.0,
                       [137.0, 134.0, 133.0, 134.0, 134.0, 136.0]),
                _trend("monocytes", "Monocytes", "x10^3/uL", 1, 0.2, 0.8,
                       MONOCYTE_SERIES),
            ],
        }},
    ]


# --------------------------------------------------------------------------------------
# 1. The sentinel never reaches the prose.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("appended", [
    "\n\n" + T6_SENTINEL,
    "\n\n" + FENCE + "json\n" + T6_SENTINEL + "\n" + FENCE,
    "\n\n" + FENCE + "\n" + T6_SENTINEL + "\n" + FENCE,
    "\n\n{\n  \"evidence_found\": false\n}",
    " " + T6_SENTINEL,
    "\n\n" + '{"evidence_found": true}',
])
def test_the_decline_object_is_removed_however_the_model_appends_it(appended):
    """Every shape measured or plausible, removed — and the prose kept word for word."""
    prose, flags = citations.strip_printed_constructs(T6_ANSWER + appended)

    assert "evidence_found" not in prose
    assert prose == T6_ANSWER
    assert flags == ["citation_printed_construct:evidence_found"]


def test_a_reply_that_is_only_the_object_is_still_a_decline():
    """The reader's half: the object alone, bare and fenced, is what the prompt asks for."""
    assert graph.read_evidence_reply(T6_SENTINEL).declined is True
    assert graph.read_evidence_reply(T6_SENTINEL).text == ""
    assert graph.read_evidence_reply(
        FENCE + "json\n" + T6_SENTINEL + "\n" + FENCE).declined is True
    assert graph.read_evidence_reply(
        FENCE + "\n" + T6_SENTINEL + "\n" + FENCE).declined is True


def test_a_fenced_envelope_carrying_an_answer_is_unwrapped_to_its_prose():
    """A model that fences the object it does have an answer in is read through the fence."""
    reply = graph.read_evidence_reply(
        FENCE + 'json\n{"evidence_found": true, "answer": "Metoprolol."}\n' + FENCE)

    assert reply.declined is False
    assert reply.text == "Metoprolol."


def test_prose_with_the_object_after_it_is_prose_for_the_reader_and_an_artefact_later():
    """The two halves are deliberate, and this is the case that separates them.

    `read_evidence_reply` must NOT read this as a decline — the turn has an answer, and
    withdrawing it would serve nothing to a clinician who asked a fair question. What it
    must do is leave the object where the settlement pass can find it.
    """
    reply = graph.read_evidence_reply(T6_ANSWER + "\n\n" + T6_SENTINEL)

    assert reply.declined is False
    assert reply.text == T6_ANSWER + "\n\n" + T6_SENTINEL

    prose, flags = citations.strip_printed_constructs(reply.text)

    assert prose == T6_ANSWER
    assert flags == ["citation_printed_construct:evidence_found"]


def test_an_answer_discussing_the_verdict_is_not_scrubbed():
    """The reason the rule matches the quoted key and a boolean, not the word."""
    prose = "The answer records whether evidence was found for that question."

    assert citations.strip_printed_constructs(prose) == (prose, [])


def test_the_whole_turn_settles_without_the_object_and_with_the_flag():
    """End to end: the served prose carries no sentinel, and the turn says it removed one."""
    settled = answer_text.settle_answer(
        prose=T6_ANSWER + "\n\n" + T6_SENTINEL,
        deterministic=None,
        tool_calls=[{"name": "rag_search", "args": {}, "response": {
            "hadm_id": HADM,
            "passages": [{"id": "p-1", "section": "brief_hospital_course",
                          "text": "Tolerated the procedure well."},
                         {"id": "p-2", "section": "major_procedure",
                          "text": "Laparoscopic Roux-en-Y gastric bypass."}],
        }}],
        replayed_tool_calls=None,
        command=None,
    )

    assert "evidence_found" not in (settled.prose or "")
    assert "citation_printed_construct:evidence_found" in settled.flags
    assert settled.prose.endswith("complications ^[1], ^[2].")


# --------------------------------------------------------------------------------------
# 2. A risk value is not a laboratory reading.
# --------------------------------------------------------------------------------------


def _record() -> session_evidence.SessionEvidenceState:
    return session_evidence.SessionEvidenceState.hydrate_from_calls(HADM, _lab_calls())


def test_the_array_the_failing_turn_was_numbered_against():
    """The premise of every test below: RDW 2, sodium 3, monocytes 4."""
    state = _record()

    sections = [state.items[position].section for position in (2, 3, 4)]

    assert sections == ["rdw", "sodium", "monocytes"]


def test_the_risk_value_loses_the_marker_and_the_readings_keep_theirs():
    """The measured sentences, all four of them, from `dev_t4_01` turn 3."""
    prose = (
        "His RDW, which peaked at 17.5% at discharge, is a risk driver that increases his "
        "readmission probability ^[2]. His sodium level, which reached a nadir of "
        "133 mmol/L during the stay, is also identified as a risk driver ^[3]. "
        "Conversely, the model attributes a decrease in risk to his lowest monocyte count "
        "of 0.47 x10^3/uL ^[4]. The notes do not account for that attribution."
    )

    out, flags = risk_lab_citation.filter_risk_value_lab_citations(
        prose, tool_calls=_lab_calls(), evidence_state=_record())

    assert flags == ("risk_value_lab_citation_stripped:^4",)
    assert "0.47 x10^3/uL." in out, "the marker goes; the value is the model's own input"
    assert "readmission probability ^[2]." in out
    assert "as a risk driver ^[3]." in out
    assert "risk_value_lab_citation_stripped" not in out


def test_a_sentence_that_quotes_the_series_keeps_its_marker():
    """The limit that keeps the rule off correct prose.

    A sentence quoting 0.5 is citing the trajectory it names, so the marker stays — even
    when the model's own 0.47 sits in the same paragraph.
    """
    prose = ("His monocyte count reached its lowest recorded value of "
             "0.5 x10^3/uL ^[4].")

    out, flags = risk_lab_citation.filter_risk_value_lab_citations(
        prose, tool_calls=_lab_calls(), evidence_state=_record())

    assert out == prose
    assert flags == ()


def test_a_risk_value_that_the_series_does_hold_keeps_its_marker():
    """Sodium's `133` is both the model's input and a reading, so nothing is stripped."""
    prose = "His sodium reached 133 mmol/L ^[3]."

    out, flags = risk_lab_citation.filter_risk_value_lab_citations(
        prose, tool_calls=_lab_calls(), evidence_state=_record())

    assert out == prose
    assert flags == ()


def test_a_note_passage_marker_is_never_touched():
    """The rule is about laboratory series only; a note citation is not in its scope."""
    prose = "He presented with jaundice and weight loss ^[1]."

    out, flags = risk_lab_citation.filter_risk_value_lab_citations(
        prose, tool_calls=_lab_calls(), evidence_state=_record())

    assert out == prose
    assert flags == ()


def _administrative_calls() -> list[dict]:
    """The same thread, with a panel whose first position is hemoglobin.

    `oncology_flag` and `prior_inpatient_days` are in `top_factors` because they are in
    the real payload: the failing admission's estimate is driven mostly by them, and they
    are the factors whose values are counts rather than measurements.
    """
    prediction = _prediction()
    prediction["top_factors"] = [
        {"feature": "monocytes_min", "contribution": -0.5619, "direction": "decreases",
         "value": 0.47},
        {"feature": "oncology_flag", "contribution": 0.3075, "direction": "increases",
         "value": 1},
        {"feature": "prior_inpatient_days", "contribution": -0.2527,
         "direction": "decreases", "value": 0},
    ]
    return [
        {"name": "predict_readmission", "args": {}, "response": prediction},
        {"name": "get_lab_trends", "args": {}, "response": {
            "hadm_id": HADM, "synthetic": True,
            "trends": [
                {"key": "hemoglobin", "label": "Hemoglobin", "unit": "g/dL",
                 "decimals": 1, "reference_low": 12.0, "reference_high": 15.5,
                 "anchored": ["hemoglobin_min"],
                 "series": [{"day": 0.0, "value": 11.2}, {"day": 4.0, "value": 6.1},
                            {"day": 9.0, "value": 10.0}]},
            ],
        }},
    ]


def test_a_qualitative_lab_sentence_keeps_its_marker():
    """No number, so no claim about a value — and no administrative value to match one.

    `oncology_flag` is 1 and `prior_inpatient_days` is 0 in this payload. Read as values,
    `1` and `0` are numbers any sentence may happen to contain, which is why they are no
    longer in the comparison set at all.
    """
    prose = "His hemoglobin remained low at discharge ^[1]."

    out, flags = risk_lab_citation.filter_risk_value_lab_citations(
        prose, tool_calls=_administrative_calls(), evidence_state=None)

    assert out == prose
    assert flags == ()


def test_an_administrative_count_is_not_a_risk_value():
    """The exclusion, where it bites: a sentence quoting a count keeps its marker.

    `1` here is `oncology_flag`'s value and `0` is `prior_inpatient_days`'s. Neither is a
    measurement, neither is in the hemoglobin series — and if either were read as the
    model's own input, this sentence would lose a marker it has every right to.
    """
    prose = ("He has 1 active oncology diagnosis and 0 prior inpatient days "
             "this admission ^[1].")

    out, flags = risk_lab_citation.filter_risk_value_lab_citations(
        prose, tool_calls=_administrative_calls(), evidence_state=None)

    assert out == prose
    assert flags == ()


def test_the_laboratory_factor_still_qualifies():
    """The restriction narrows the rule; it does not disable it."""
    prose = ("The model attributes a decrease in risk to his lowest monocyte count of "
             "0.47 x10^3/uL ^[1].")

    out, flags = risk_lab_citation.filter_risk_value_lab_citations(
        prose, tool_calls=_administrative_calls(), evidence_state=None)

    assert out == ("The model attributes a decrease in risk to his lowest monocyte "
                   "count of 0.47 x10^3/uL.")
    assert flags == ("risk_value_lab_citation_stripped:^1",)


def test_a_unit_exponent_is_not_a_quoted_value():
    """`x10^3` carries a `3`; read as a value it would match any series holding 3."""
    calls = _administrative_calls()
    calls[1]["response"]["trends"][0]["series"].append({"day": 5.0, "value": 3.0})

    prose = "His hemoglobin was 6.1 g/dL at its lowest ^[1]."

    out, flags = risk_lab_citation.filter_risk_value_lab_citations(
        prose, tool_calls=calls, evidence_state=None)

    assert out == prose
    assert flags == ()


def test_a_turn_with_no_estimate_is_left_alone():
    """No prediction, no risk value, nothing to confuse with a reading."""
    calls = [call for call in _lab_calls() if call["name"] != "predict_readmission"]
    prose = "His monocyte count reached its lowest recorded value of 0.47 x10^3/uL ^[3]."

    out, flags = risk_lab_citation.filter_risk_value_lab_citations(
        prose, tool_calls=calls, evidence_state=None)

    assert out == prose
    assert flags == ()


def test_the_remaining_markers_compact_to_one_through_k():
    """The point of running before the page is numbered: no gap is left behind."""
    prose = (
        "His RDW peaked at 17.5% ^[2]. His sodium reached 133 mmol/L ^[3]. "
        "The model attributes a decrease in risk to his monocyte count of "
        "0.47 x10^3/uL ^[4]."
    )

    settled = answer_text.settle_answer(
        prose=prose,
        deterministic=None,
        tool_calls=[],
        replayed_tool_calls=_lab_calls(),
        evidence_state=_record(),
        command=None,
    )

    assert "risk_value_lab_citation_stripped:^4" in settled.flags
    assert citations.cited_numbers(settled.prose) == [1, 2]
    assert settled.display == {"1": 2, "2": 3}


# --------------------------------------------------------------------------------------
# 3. The paragraph break after the discharge section.
# --------------------------------------------------------------------------------------


T4_SECTIONS = (
    "**Presentation & diagnosis.** A 60-year-old man presented with chest pain ^[1].\n\n"
    "**Hospital course & procedures.** He underwent cardiac catheterisation ^[2]. "
    "He was prescribed Plavix daily for one month ^[3].\n\n"
    "**Discharge plan & medications.** The patient was discharged home. He was advised "
    "to follow a low-fat diet ^[5]. Follow-up is planned with his primary care physician "
    "in one to two weeks ^[5].\n\n"
    "**Readmission risk & trajectory.** His estimated 31.7% readmission risk is above "
    "the 11.0% threshold. The primary drivers are oncology history ^[6]."
)

T4_PASSAGES = [
    {"section": "discharge_instructions",
     "text": "Follow a low-fat diet. Follow up with your primary care physician."},
    {"section": "brief_hospital_course",
     "text": "He was prescribed Plavix daily for one month."},
]


def _settled_t4() -> str:
    return discharge_section.filter_discharge_section(
        T4_SECTIONS,
        passages=T4_PASSAGES,
        units=[{"section": "history_of_present_illness"},
               {"section": "brief_hospital_course"},
               {"section": "brief_hospital_course"},
               {"section": "brief_hospital_course"},
               {"section": "discharge_instructions"},
               {"section": "discharge_instructions"}],
    )[0]


def test_the_next_heading_starts_its_own_paragraph():
    """`dev_t4_02` turn 1 as served was `^[5].**Readmission risk & trajectory.**`."""
    out = _settled_t4()

    assert "**Discharge plan & medications.**" in out
    assert "\n\n**Readmission risk & trajectory.**" in out
    # The defect as served: the full stop of the last kept sentence ran straight into
    # the opening `**` of the next heading, with no break between them.
    assert "5].**Readmission" not in out
    assert ".\n\n**Readmission" in out


def test_the_removal_still_happens_on_the_way():
    """The break is restored, and the Plavix sentence is still gone."""
    out = _settled_t4()

    assert "He was prescribed Plavix daily for one month" not in out.split(
        "**Discharge plan")[1].split("**Readmission")[0]
    assert out.count("**Hospital course & procedures.**") == 1


def test_the_other_break_the_answer_arrived_with_is_unchanged():
    """Nothing about the pass makes a blank line where the model did not write one."""
    out = _settled_t4()

    for heading in ("**Presentation & diagnosis.**", "**Hospital course & procedures.**",
                    "**Discharge plan & medications.**", "**Readmission risk & trajectory.**"):
        assert heading in out
    assert out.count("\n\n") == 3


def test_a_paragraph_with_nothing_following_it_gains_no_trailing_blank_line():
    """The restore is conditional on there being something after it."""
    prose = ("**Discharge plan & medications.** The patient was discharged home ^[1]. "
             "He was advised on diet ^[1].")

    out, flags = discharge_section.filter_discharge_section(
        prose,
        passages=T4_PASSAGES,
        units=[{"section": "discharge_instructions"}],
    )

    assert out == prose
    assert flags == ()


# --------------------------------------------------------------------------------------
# 4. The prompt rules that no test above can enforce.
# --------------------------------------------------------------------------------------


def test_the_hub_forbids_a_mechanism_the_sources_do_not_state():
    """`dev_t4_01` turn 3 explained why RDW and sodium are abnormal in cancer.

    Nothing in the passages said it. The rule that was there permitted it — "give the
    standard clinical context for those facts" — and that sentence is gone.
    """
    hub = " ".join(prompts.CORE_HUB_PROMPT.split())

    assert "NEVER SUPPLY A MECHANISM" in hub
    assert "differential etiology" in hub
    assert "not as general medicine" in hub
    assert "the notes do not account for the mechanism" in hub
    assert "standard clinical context" not in hub
    assert "General context is not a patient fact" not in hub


def test_the_risk_spoke_says_a_factor_value_takes_no_marker():
    text = " ".join(prompts.RISK_SPOKE.split())

    assert "take no `^[n]` marker" in text
    assert ("a series rounded to fewer decimals never held a higher-precision model "
            "input") in text
    # The rule is stated without a Dev-set record: the pair that produced it belongs to
    # `dev_t4_01` and shipped its own numbers into a prompt writing about another patient.
    assert "0.47" not in text
