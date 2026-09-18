"""Gap 7 — one cohort, derived, and every artifact describing the same patients.

Three sets of admissions existed under one name. The table the config called the
authorisation boundary held 24, of which 20 were served. The corpus the tools and
the index serve holds 89, of which 69 were outside the boundary. The fixtures
held 108, including 19 admissions the live system cannot serve at all and six
captured answers for admissions that never existed in it.

The invariant these tests hold is that the smaller artifacts are derived from the
authoritative cohort rather than chosen beside it. The authority itself lives in
one file, ``data/agent/demo_cohort.json``, which is the list the site is seeded
from; the fixtures must be a subset of it, the per-admission fixtures must name
admissions inside it, and the config must name the script that writes the table
rather than a script that was deleted years of docstrings ago.
"""

import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
COHORT = REPO / "data/agent/demo_cohort.json"
FIXTURES = REPO / "data/agent/demo_fixtures"


def _authoritative() -> set[str]:
    return {str(p["hadm_id"]) for p in json.loads(COHORT.read_text())["patients"]}


def test_the_cohort_is_a_real_set():
    """A guard on the guard: an empty cohort would make every test below pass."""
    assert len(_authoritative()) > 1


def test_the_fixture_cohort_is_a_subset_of_the_authoritative_one():
    """An offline fixture answering for a patient that does not exist live is
    the one thing fixtures must never do: it presents a simulation as the
    system."""
    authoritative = _authoritative()
    patients = json.loads((FIXTURES / "demo_cohort.json").read_text())["patients"]
    ids = {str(p["hadm_id"]) for p in patients}

    assert ids <= authoritative, f"fixtures invent {sorted(ids - authoritative)}"
    assert len(ids) == len(authoritative), (
        "the fixture cohort no longer covers the whole cohort: "
        f"{len(authoritative - ids)} admissions have no fixture"
    )


def test_the_risk_payloads_cover_the_same_patients():
    authoritative = _authoritative()
    ids = set(json.loads((FIXTURES / "cohort_risk.json").read_text()))

    assert ids == authoritative, f"differs by {sorted(ids ^ authoritative)}"


def test_every_per_admission_fixture_names_an_admission_the_live_system_serves():
    authoritative = _authoritative()
    orphans = []
    for path in sorted(FIXTURES.glob("*.json")):
        ids = re.findall(r"(\d{5,})", path.name)
        if ids and not any(i in authoritative for i in ids):
            orphans.append(path.name)

    assert not orphans, f"fixtures for unserved admissions: {orphans}"


def test_the_config_names_the_script_that_writes_the_cohort_table():
    """The comment named scripts/build_demo_cohort.py, which does not exist.

    A comment that names a deleted script is worse than no comment: it reads as
    provenance, so nobody checks whether the table still describes the data.
    """
    source = (REPO / "services/mcp/config.py").read_text()
    block = source[: source.index("COHORT_TABLE = ")]
    comment = block[block.rindex("# The authorisation boundary"):]
    named = re.findall(r"scripts/[\w./-]+\.py", comment)

    assert named, "the cohort comment names no script"
    for path in named:
        assert (REPO / path).exists(), f"the cohort comment names a missing {path}"
