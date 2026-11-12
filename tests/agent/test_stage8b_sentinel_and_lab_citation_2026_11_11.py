"""Stage 8B: the defects the 2026-11-11 Dev gate exposed that survived Stage 12C.

A machine token served as if it were prose, and a served document losing its paragraph
break. The third block this module was written for — a post-hoc pass that matched the
model's own input against a laboratory series and removed the marker — was deleted in
Stage 12C as a postprocess patch; the prompt rule that keeps an attribution out of a
cited sentence is pinned in `test_prompt_contract.py`.

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
from services.agent.postprocess import citations, discharge_section

FENCE = "```"

#: The admission the two case studies in this module belong to.
HADM = 90000108

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
    """A model `value` is not a reading, and a series never evidences one.

    The second assertion used to pin the rationale for the post-hoc pass that compared a
    risk value against a laboratory series (`a series rounded to fewer decimals never held
    a higher-precision model input`). Stage 12C deleted that pass as a postprocess patch;
    what the prompt must still say is the rule itself, which is that no laboratory marker
    goes on a sentence quoting the model's own number.
    """
    text = " ".join(prompts.RISK_SPOKE.split())

    assert "take no `^[n]` marker" in text
    assert "no laboratory marker belongs on a sentence quoting one" in text
    # The rule is stated without a Dev-set record: the pair that produced it belongs to
    # `dev_t4_01` and shipped its own numbers into a prompt writing about another patient.
    assert "0.47" not in text
