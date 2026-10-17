"""The deterministic free-text router (Phase 4C §1.4, directive 3).

A free-text question that is plainly asking for a command's answer is sent to that
command before the model is consulted. The redirect that used to do this job lived
in the system prompt, and on 2026-10-17 it lost an argument with another paragraph
in the same prompt: a free-text risk question began calling `rag_search` instead of
naming /risk, purely because a new scope rule shifted the balance. A rule that has
to win against the rest of the prompt is not a boundary.

These are unit tests, deliberately. The classifier is a pure function and the seam
is a dict rewrite, so neither needs a model, a tool, or a served endpoint — which
means the routing contract can be asserted on every run rather than only where a
live service happens to be up.
"""

import pytest

from services.agent import http, routing


@pytest.mark.parametrize("question,expected", [
    ("What did her hemoglobin and sodium do during this admission?", "labs"),
    ("Show me the discharge medications and dosing for this admission", "meds"),
    ("what labs were drawn", "labs"),
    ("Was her RDW ever abnormal?", "labs"),
    # Abstention, in both directions. A summary is not one of these commands, and a
    # risk question is left where it was: its redirect still holds through the
    # prompt (asserted by the live tests), so routing it here would be a behaviour
    # change with no defect behind it.
    ("Summarize the hospital course", None),
    ("What is the readmission risk for this admission?", None),
    ("", None),
    ("   ", None),
    # Substring traps. Each of these contains a lab word as a substring and must
    # abstain, which is why the matcher compares whole tokens.
    ("Please elaborate on the admission", None),
    ("Did the teams collaborate on discharge?", None),
    ("Count every syllable in the note", None),
    # A clicked risk-driver control composes this template and submits it as free
    # text, so the router sees the driver's label and nothing else. Answering it
    # with /labs because the label says RDW is the router overriding a control the
    # reader already chose.
    ("What in this patient's chart explains red cell distribution width (RDW) "
     "as a readmission risk driver?", None),
    ("What in this patient's chart explains admission type as a readmission "
     "risk driver?", None),
])
def test_classify_free_text(question, expected):
    assert routing.classify_free_text(question) == expected


def test_punctuation_does_not_hide_a_word():
    """`hemoglobin,` and `(sodium)` are the words they look like."""
    assert routing.classify_free_text("Her hemoglobin, and (sodium) values?") == "labs"
    assert routing.classify_free_text("Medications: which ones?") == "meds"


def test_a_multi_word_phrase_is_matched_as_a_phrase():
    assert routing.classify_free_text("What was her red cell distribution?)") == "labs"


def test_a_risk_driver_control_is_not_hijacked_by_its_own_label():
    """The label is the reason the reader clicked, not the question they asked."""
    for label in ("red cell distribution width (RDW)", "admission type",
                  "lowest monocyte count", "prior inpatient days"):
        question = (f"What in this patient's chart explains {label} "
                    "as a readmission risk driver?")
        assert routing.classify_free_text(question) is None


def test_the_affordance_rule_does_not_swallow_a_real_lab_question():
    """The abstention is about the template, not about the word inside it."""
    assert routing.classify_free_text("Was her RDW ever abnormal?") == "labs"
    assert routing.classify_free_text(
        "What did her hemoglobin and sodium do during this admission?") == "labs"


@pytest.mark.parametrize("question,expected", [
    # Answerable from the pre-flight manifest, so the graph must not force a note
    # search. This is the question that produced 17.29 days and then had it
    # replaced by "the discharge note does not document this" (2026-10-17).
    ("what is the patient length of stay?", True),
    ("What is the admission type for this patient?", True),
    ("Where was she discharged to?", True),
    ("What is her insurance?", True),
    ("How old is this patient?", True),
    ("How many prior admissions has she had?", True),
    ("How long was the stay?", True),
    # Not administrative: the notes are the evidence, so the demand still applies.
    ("What did the discharge note say about her anxiety?", False),
    ("Why did her knee pain increase during the stay?", False),
    # Token matching, not substrings: "dosage" is not "age", "closely" is not "los".
    ("What dosage of metoprolol was she on?", False),
    ("How closely did the teams work together?", False),
])
def test_asks_for_administrative_detail(question, expected):
    assert routing.asks_for_administrative_detail(question) is expected


def test_a_routed_question_takes_the_command_field():
    """`kind` is the whole mechanism: the chain already reads it as the command."""
    parsed = {"question": "Show me the discharge medications", "kind": None}

    routed = http._route_free_text(parsed)

    assert routed["kind"] == "meds"
    assert routed["routed_to"] == "meds"
    # The request dict is not edited in place. A router that mutates its argument
    # has an effect that is invisible in a diff of the caller.
    assert parsed["kind"] is None
    assert "routed_to" not in parsed


def test_an_explicit_chip_is_never_overridden():
    """The router is for questions nobody classified, not for second-guessing."""
    parsed = {"question": "Show me the discharge medications", "kind": "summarize"}

    routed = http._route_free_text(parsed)

    assert routed["kind"] == "summarize"
    assert "routed_to" not in routed


def test_an_unclassified_question_is_left_alone():
    """Abstaining leaves the question with the reasoning loop, which can still answer."""
    parsed = {"question": "Summarize the hospital course", "kind": None}

    routed = http._route_free_text(parsed)

    assert routed["kind"] is None
    assert "routed_to" not in routed
