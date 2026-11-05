"""The suite may not contradict the contract the agent is given.

Two defects from the 2026-11-04 runs, both of the same kind: an assertion that
disagreed with a rule the product enforces.

`dev_t3_01`, `dev_t4_03` and `dev_t5_01` all require `"Lovenox"` in `must_contain` on
admission 90000015. The chart documents Lovenox only as an arrival-list drug that was
discontinued on 08/30/2007 — `dev_t1_03` and `dev_t2_04` were reconciled to say exactly
that — and `MEDS_FORMATTING` tells the agent to exclude drugs the patient merely arrived
on. So the three scenarios demanded a name and, in `dev_t5_01`, the judge then failed the
answer for naming it in the only framing the discharge list allows. `must_contain` is too
weak a primitive to carry a framing rule: it asserts that a token appears, not how the
answer must present it. These pins add the framing to `facts`, `behaviour` and
`forbidden_claims` without deleting the token, so coverage is kept and the conflict ends.

`dev_t1_04` and `dev_t4_02` sit on admissions with no dedicated `discharge_medications`
section, so `discharge_medications` falls back to a narrative section. Their `behaviour`
already reflects that; this pins it, so a later edit cannot quietly re-expect a formal
list from a record that has none.

The sectioned-fetch inventory is inlined rather than read from `evaluation/results/`,
which is gitignored: a test that depends on a machine-local artifact cannot run in CI.
The payloads below are the `returned` sections from those two admissions during
`pilot-verify-20261104-230618`.
"""

from __future__ import annotations

import json
from pathlib import Path

DATASETS = Path(__file__).resolve().parents[2] / "evaluation" / "datasets"


def _rows(name: str) -> list[dict]:
    with (DATASETS / name).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


DEV = _rows("dev_suite.jsonl")
BY_ID = {row["id"]: row for row in DEV}

#: What `discharge_medications` actually returned for these two admissions. The list is
#: the `section` of every passage in the tool's response: a dedicated medication section
#: would appear here by name, and does not.
SECTIONED_FETCH = {
    90000106: {"tool": "discharge_medications", "sections": ["brief_hospital_course"]},
    90000023: {"tool": "discharge_medications", "sections": ["discharge_instructions"]},
}

QUALIFIED_SCENARIOS = ("dev_t3_01", "dev_t4_03", "dev_t5_01")

#: Every scenario written against admission 90000015. The suite holds nine, and the rule
#: below is about all of them rather than about the ones that mention Lovenox by name: a
#: discharge medication list that carries the drug is a claim about the record whether the
#: scenario's question asks about anticoagulants or not.
FIFTEEN = tuple(row["id"] for row in DEV if str(row.get("hadm_id")) == "90000015")

#: Phrases that show a behaviour is describing an absence rather than a list. Any one is
#: enough; the point is that the answer is not expected to produce a discharge drug list.
_ABSENCE_PHRASES = (
    "no discharge medication",
    "rather than a discharge",
    "report that absence",
    "does not contain a dedicated",
    "no dedicated discharge medications",
)


def _blob(value: object) -> str:
    return json.dumps(value, ensure_ascii=False).lower()


# ---------------------------------------------------------------- the red test


def test_90000015_lovenox_qualification_shape():
    """Every scenario that names Lovenox must say how Lovenox may be named.

    The old form of this test also required `"Lovenox"` in `must_contain` for all three
    scenarios. That assertion was the defect it was meant to guard: on `dev_t4_03` and
    `dev_t5_01` the turn asks what was prescribed AT DISCHARGE, so requiring the token
    while forbidding the only framing that satisfies it made the two directions of the
    same rule mutually exclusive. The token is kept where the question is about the two
    anticoagulants the record names (`dev_t3_01`), and the framing — facts, behaviour,
    forbidden claim — is what every one of the three must carry.
    """
    failures: list[str] = []
    for scenario in QUALIFIED_SCENARIOS:
        row = BY_ID[scenario]
        expected = row["expected_output"]
        facts = expected.get("facts") or {}
        behaviour = str(expected.get("behaviour") or "").lower()
        claims = [str(claim).lower() for claim in (expected.get("forbidden_claims") or [])]
        facts_blob = _blob(facts)

        if "08/30/2007" not in facts_blob:
            failures.append(f"{scenario}: facts do not record the 08/30/2007 discontinuation")
        if "transfer" not in facts_blob:
            failures.append(f"{scenario}: facts do not say Lovenox is on the transfer list")
        if "discharge" not in facts_blob:
            failures.append(f"{scenario}: facts do not say what continues at discharge")
        if not any(word in behaviour for word in ("transfer", "discontinued", "discontinu")):
            failures.append(f"{scenario}: behaviour does not state the required framing")
        if not any("prescribed at discharge" in claim for claim in claims):
            failures.append(
                f"{scenario}: forbidden_claims do not prohibit presenting Lovenox as "
                "prescribed at discharge")

    assert not failures, "\n".join(failures)


def test_no_90000015_scenario_claims_lovenox_is_a_discharge_medication():
    """One record, one story: Lovenox is on the transfer list and was discontinued.

    `dev_t2_03` and `dev_t2_04` both carried a `discharge_medications` list ending in
    `"Lovenox"` while their own `forbidden_claims` said it was not prescribed at
    discharge, and while `dev_t1_03` — same admission, same note — listed it under
    `arrival_only`. A suite that states both readings cannot be satisfied by any answer,
    and the judge is asked to grade against whichever one it reads first.

    Two decidable forms of the claim are checked. A discharge-regimen LIST that names the
    drug is the claim outright; an `anticoagulant_status` entry is the framing, and it is
    only honest if it carries the date the record discontinued it.
    """
    assert len(FIFTEEN) == 9, f"expected the nine 90000015 scenarios, found {FIFTEEN}"

    failures: list[str] = []
    for scenario in FIFTEEN:
        facts = BY_ID[scenario]["expected_output"].get("facts") or {}
        for key in ("discharge_medications", "discharge_regimen"):
            items = facts.get(key)
            if not isinstance(items, list):
                continue
            named = [item for item in items if "lovenox" in str(item).lower()]
            if named:
                failures.append(
                    f"{scenario}: facts.{key} lists Lovenox ({named}) as a discharge "
                    "medication")
        status = (facts.get("anticoagulant_status") or {}).get("Lovenox")
        if status is not None and "08/30/2007" not in str(status):
            failures.append(
                f"{scenario}: facts.anticoagulant_status.Lovenox does not carry the "
                "08/30/2007 discontinuation")

    assert not failures, "\n".join(failures)


# --------------------------------------------------------- the green regression pin


def test_missing_med_section_fallback_pins():
    """An admission with no medication section does not get a medication list.

    GREEN on `87743f2` by design — this one is a pin, not a red test. It exists so that a
    later edit cannot re-expect a formal discharge medication list from a record whose
    `discharge_medications` call fell back to a narrative section.
    """
    failures: list[str] = []
    for scenario, hadm in (("dev_t1_04", 90000106), ("dev_t4_02", 90000023)):
        fetch = SECTIONED_FETCH[hadm]
        if "discharge_medications" in fetch["sections"]:
            failures.append(
                f"{hadm}: the fixture says a dedicated section WAS returned; the pin and "
                "the premise disagree")
        behaviour = str(BY_ID[scenario]["expected_output"].get("behaviour") or "").lower()
        if not any(phrase in behaviour for phrase in _ABSENCE_PHRASES):
            failures.append(
                f"{scenario}: behaviour does not describe the absence of a dedicated "
                f"discharge medications section: {behaviour[:120]!r}")

    assert not failures, "\n".join(failures)


def test_the_fixture_covers_the_admissions_the_pins_name():
    """Guard against the fixtures drifting away from the scenarios they justify."""
    assert set(SECTIONED_FETCH) == {90000106, 90000023}
    for scenario in ("dev_t1_04", "dev_t4_02"):
        assert int(BY_ID[scenario]["hadm_id"]) in SECTIONED_FETCH
