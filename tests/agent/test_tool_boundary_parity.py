"""Every tool answers a malformed id the same way, and every tool has a contract.

Phase 4, 2026-10-25. Two gaps in one file, because they are the same gap seen from
two directions: a tool that skips the shared request check and a tool that skips the
shared response contract are both a tool that does not participate in the boundary
the others do.

The id half is written against `tools.__all__` and the tool functions themselves, so
it stays true when a tool is added. The contract half asserts the registry covers
every tool — read from the package, not listed again.
"""

import asyncio

import pytest

from services.mcp import contracts
from services.mcp.tools import __all__ as REGISTERED_TOOLS
from services.mcp.tools import (
    discharge_medications,
    get_admission_profile,
    get_lab_trends,
    predict_readmission,
    rag_search,
    rag_search_sections,
)

# The six tools scoped to one admission. `search_literature` is the seventh and is
# deliberately absent: it takes no `hadm_id`, which is its own kind of conformance.
PATIENT_SCOPED = {
    "predict_readmission": lambda hadm: predict_readmission(hadm),
    "rag_search": lambda hadm: rag_search(hadm, "medications"),
    "rag_search_sections": lambda hadm: rag_search_sections(hadm),
    "discharge_medications": lambda hadm: discharge_medications(hadm),
    "get_lab_trends": lambda hadm: get_lab_trends(hadm),
    "get_admission_profile": lambda hadm: get_admission_profile(hadm),
}

# `True` is the case `_validation.valid_hadm_id` exists for — a bool is an int
# subclass, so it would otherwise bind as admission 1 and answer `unknown_patient`,
# which is a wrong code for a malformed request. The string is the shape the site
# used to send, and the negative is neither.
MALFORMED_IDS = [True, "90000015", -1]


def test_the_patient_scoped_list_is_exactly_the_tools_that_take_an_admission():
    """Guards the two tables below against a tool being added under a new name."""
    assert set(PATIENT_SCOPED) == set(REGISTERED_TOOLS) - {"search_literature"}


@pytest.mark.parametrize("tool", sorted(PATIENT_SCOPED))
@pytest.mark.parametrize("hadm_id", MALFORMED_IDS)
def test_a_malformed_admission_id_is_refused_as_bad_request(tool, hadm_id):
    """The same code from every tool, before any dependency is touched.

    `get_lab_trends` was the exception until Phase 4: it went straight to the
    feature source, so `True` came back as `unknown_patient` and a string raised
    inside the BigQuery client, outside the tool's own handlers.
    """
    payload = asyncio.run(PATIENT_SCOPED[tool](hadm_id))

    assert payload["error"] == "bad_request", payload
    assert "hadm_id" in payload["message"].lower()


def test_every_tool_in_the_package_has_a_contract():
    """Read from `tools.__all__` rather than listed again.

    Two tools were missing from the registry: `discharge_medications`, whose
    payload therefore reached the model unchecked, and `get_admission_profile`,
    which had no contract at either boundary. A new tool now cannot be added
    without deciding its contract — the same exhaustive-by-construction rule as
    the derivable/kept classification.
    """
    assert REGISTERED_TOOLS, "no tools found — has the registry moved?"
    assert set(contracts.TOOL_CONTRACTS) == set(REGISTERED_TOOLS)


def test_a_tool_without_a_contract_is_returned_unchanged():
    """The lenient path stays, and stays narrow.

    It exists so a partial deploy cannot turn a missing declaration into an outage.
    With the registry now exhaustive it covers only a tool the server advertises
    before this build knows about it.
    """
    payload = {"anything": "goes"}

    assert contracts.validate_tool_result("a_tool_from_the_future", payload) is payload


def test_the_admission_profile_contract_rejects_a_non_string_value():
    """The profile is read aloud into a prompt, and the server renders an absent
    value as "Not recorded" rather than null. A numeric value here therefore means
    the payload was built somewhere else — and the reader's coercion would hide it.
    """
    good = {
        "hadm_id": 90000009,
        "profile": {"age": "63", "prior_admissions": "Not recorded"},
        "feature_source": "bigquery",
    }

    assert contracts.validate_admission_profile_result(good) is good

    with pytest.raises(contracts.ToolContractError):
        contracts.validate_admission_profile_result({**good, "profile": {"age": 63}})


def test_the_admission_profile_contract_accepts_an_error_payload():
    """A tool error is a legitimate return, and the registry must not turn one
    into a contract failure — that would convert a handled absence into an outage.
    """
    error = contracts.tool_error("unknown_patient", "No admission.", hadm_id=1)

    assert contracts.validate_admission_profile_result(error) is error
