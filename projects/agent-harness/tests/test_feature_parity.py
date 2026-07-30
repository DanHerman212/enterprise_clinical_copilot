"""Feature-source parity.

The failure mode this guards against is subtle: a Feature Store view that is
stale, or that coerces types (INT64 -> FLOAT64, NULL -> 0), returns features
that are *plausible but wrong*. The endpoint accepts them, the model returns a
confident probability, and nothing errors. A demo that quietly lies is worse
than one that crashes.

Skips cleanly when the online store has not been provisioned, so this suite
stays green in CI without billing for serving nodes.
"""

import json
from pathlib import Path

import pytest

from mcp_server.features import get_feature_source, to_vector
from mcp_server.features.manifest import feature_order

FIXTURE = Path(__file__).parent / "fixtures" / "expected.json"
TOLERANCE = 1e-6


def _fixture_patients() -> list[int]:
    if not FIXTURE.exists():
        pytest.skip(f"{FIXTURE.name} not written yet — run smoke_test.py --write-fixture")
    return [int(k) for k in json.loads(FIXTURE.read_text())["patients"]]


@pytest.fixture(scope="module")
def bigquery_source():
    return get_feature_source("bigquery")


@pytest.fixture(scope="module")
def feature_store_source():
    from google.api_core.exceptions import GoogleAPIError

    try:
        source = get_feature_source("feature_store")
        source.fetch(_fixture_patients()[0])
    except (GoogleAPIError, KeyError) as e:
        pytest.skip(f"Feature Store not available ({type(e).__name__}) — "
                    f"run scripts/setup_feature_store.py")
    return source


def test_bigquery_returns_every_manifest_feature(bigquery_source):
    """The vector must line up with feature_order, nulls included."""
    hadm_id = _fixture_patients()[0]
    row = bigquery_source.fetch(hadm_id)
    assert set(row) == set(feature_order()), "column set differs from the manifest"
    assert len(to_vector(row, feature_order())) == len(feature_order())


def test_bigquery_missing_patient_raises(bigquery_source):
    with pytest.raises(KeyError):
        bigquery_source.fetch(-1)


@pytest.mark.parametrize("hadm_id", _fixture_patients())
def test_sources_agree(bigquery_source, feature_store_source, hadm_id):
    """Both sources must return identical values for the same admission."""
    bq = bigquery_source.fetch(hadm_id)
    fs = feature_store_source.fetch(hadm_id)

    assert set(bq) == set(fs), "column sets differ between sources"

    mismatches = []
    for col in feature_order():
        a, b = bq[col], fs[col]
        if a is None or b is None:
            # None vs 0.0 is the classic silent corruption — treat it as a
            # mismatch, not an equality of "empty" values.
            if a is not b:
                mismatches.append((col, a, b))
        elif abs(a - b) > TOLERANCE:
            mismatches.append((col, a, b))

    assert not mismatches, "feature values differ:\n" + "\n".join(
        f"  {col}: bigquery={a!r} feature_store={b!r}" for col, a, b in mismatches
    )
