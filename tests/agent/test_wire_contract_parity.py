"""The contract this repository offers is the one the site writes against.

Two repositories, one wire. Neither can import the other — CI builds each one on
its own runner — so the contract is kept in two places and has to be checked in
both. This file holds the checks that are true from this side:

  * the request fields this service declares are exactly the ones its parser
    honours, in both directions, which is the internal half and runs everywhere;
  * the constants that exist in both repositories agree — the question ceiling and
    the chip vocabulary — which needs the sibling checkout and is skipped where
    there is none.

The declared sets had already drifted when this file was written: the site's
`CONVERSATION_FIELDS` and this service's disagreed, and nothing anywhere
noticed. A field added to the site's forward and not to this schema produces a
400 in production with both suites green.
"""

import ast
from pathlib import Path
from unittest import skipUnless

import pytest

from services.agent import contracts, questions

# The sibling repository, when it is checked out beside this one.
SIBLING = Path(__file__).resolve().parents[3] / "danielmherman"
SITE_VIEWS = SIBLING / "demo" / "views.py"

HAS_SIBLING = SITE_VIEWS.exists()
NO_SIBLING = "the site repository is not checked out beside this one"

# One sendable body per declared field, written out rather than derived: the point
# is that the list of names is a decision somebody made, and a test that read the
# names from the schema could not tell a kept promise from a vendored one.
#
# Each body is the smallest request that carries the field, because the parser
# refuses two intentions at once: a chip beside a question is a second command,
# not the same one said twice.
BODIES = {
    "question": {"question": "Assess risk."},
    "hadm_id": {"question": "Assess risk.", "hadm_id": 90000009},
    "chip": {"chip": "risk", "hadm_id": 90000009},
    "turns": {"question": "Assess risk.",
              "turns": [{"question": "q", "answer": "a"}]},
    "stream_prose": {"question": "Assess risk.", "stream_prose": False},
    "follow_ups": {"question": "Assess risk.", "follow_ups": False},
    "run_tag": {"question": "Assess risk.", "run_tag": "eval-1"},
    "intent": {"question": "Assess risk.", "intent": "risk_driver"},
}


def _assignment(path, name):
    """The value node of a top-level `name = ...` in `path`."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            if any(getattr(target, "id", None) == name for target in node.targets):
                return node.value
    raise AssertionError(f"{name} not found in {path}")


def _declared_names(path, name):
    """The string members of a `name = frozenset({...})` or a dict literal.

    Read with `ast` rather than imported: `views.py` imports Django and this test
    run has no settings module. The literal is the whole contract, so reading the
    literal is the honest check.
    """
    value = _assignment(path, name)
    if isinstance(value, ast.Call):  # frozenset({...})
        value = value.args[0]
    if isinstance(value, ast.Dict):
        return {key.value for key in value.keys if isinstance(key, ast.Constant)}
    return {
        element.value
        for element in value.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    }


# --- the request schema ------------------------------------------------------

def test_the_declared_request_fields_are_the_fields_this_file_exercises():
    """The other half of the check below: a declared field nobody sends is a field
    whose parser branch nothing covers."""
    assert set(BODIES) == contracts.ACCEPTED_REQUEST_FIELDS


@pytest.mark.parametrize("field", sorted(BODIES))
def test_a_declared_request_field_is_accepted_rather_than_refused(field):
    """A field the site forwards and this parser does not admit is a 400 in
    production with both suites green."""
    contracts.parse_agent_request(BODIES[field], max_question_chars=2000)


def test_a_field_outside_the_declared_set_is_refused_rather_than_dropped():
    with pytest.raises(contracts.AgentRequestError) as caught:
        contracts.parse_agent_request({**BODIES["question"], "temperature": 0.2},
                                      max_question_chars=2000)

    assert "temperature" in str(caught.value)


def test_the_chip_vocabulary_is_the_five_commands_and_nothing_else():
    """Pinned by name here as well as in the site's test, because a chip is a
    promise about a prompt: the wording for it lives in this repository, and a
    sixth entry would be a command with no question written for it."""
    assert set(questions.CHIP_QUESTIONS) == {
        "risk", "meds", "summarize", "literature", "labs"}


def test_the_declared_intents_are_the_controls_a_surface_can_draw():
    assert contracts.REQUEST_INTENTS == {"risk_driver"}


# --- the constants both repositories hold ------------------------------------

@skipUnless(HAS_SIBLING, NO_SIBLING)
def test_the_site_question_ceiling_is_this_service_s_ceiling():
    from services.agent.http import MAX_QUESTION_CHARS

    assert ast.literal_eval(_assignment(SITE_VIEWS, "MAX_QUESTION_CHARS")) == (
        MAX_QUESTION_CHARS)


@skipUnless(HAS_SIBLING, NO_SIBLING)
def test_the_site_chip_names_are_this_service_s_chip_names():
    # `CHIP_NAMES` is `tuple(CHIPS)`, so the vocabulary is read from the table it
    # is built from rather than from the tuple wrapper.
    assert _declared_names(SIBLING / "demo" / "fixtures.py",
                           "CHIPS") == set(questions.CHIP_QUESTIONS)


@skipUnless(HAS_SIBLING, NO_SIBLING)
def test_every_field_the_site_forwards_is_a_field_this_service_admits():
    """The browser-only names are subtracted rather than expected to be absent:
    `conversation_id` and `asked` are resolved in the site and never forwarded,
    and this service refuses both by design."""
    approved = _declared_names(SITE_VIEWS, "APPROVED_REQUEST_FIELDS")

    assert approved - {"conversation_id", "asked"} <= (
        contracts.ACCEPTED_REQUEST_FIELDS)
