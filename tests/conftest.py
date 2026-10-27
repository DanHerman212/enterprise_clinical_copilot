"""Encounter state is request-scoped, so it must not survive a test.

`services.agent.routing.encounter` holds the turn's encounter type in a ContextVar —
set once by the pre-flight and read by the command gate, the graph, the canvas and the
response composer. In production every turn sets it, including to None, because the
pre-flight runs before anything else. A test that does not run the pre-flight would
otherwise inherit whatever the previous test set, and a passing assertion could be
evidence about the wrong turn.

This fixture makes the reset unconditional rather than leaving it to each test to
remember, and it is the reason the encounter tests can set the value inline.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_encounter_state():
    from services.agent.routing import encounter  # noqa: PLC0415

    encounter.set_current(None)
    yield
    encounter.set_current(None)
