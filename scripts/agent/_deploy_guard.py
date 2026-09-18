"""Shared guard: refuse to deploy a real-corpus index to a public endpoint.

The Vector Search endpoint is public (PSC/VPC would be required to serve real
MIMIC-derived data — ECC-36), so the only thing standing between a misdeploy
and a DUA exposure is a scale check: the demonstration corpus — public MTSamples
transcriptions, and a few hundred vectors — is three orders of magnitude below
the real MIMIC-derived index at ~555k. Anything over the limit is refused
before the endpoint is created or touched.
"""

SYNTHETIC_VECTOR_LIMIT = 100_000


def assert_synthetic_scale(vectors: int, index_ref: str) -> None:
    if vectors > SYNTHETIC_VECTOR_LIMIT:
        raise SystemExit(
            f"refusing to deploy {index_ref} ({vectors} vectors) to a public "
            f"endpoint: >{SYNTHETIC_VECTOR_LIMIT} vectors means the real "
            "MIMIC-derived corpus rather than the demonstration cohort "
            "(ECC-36/53)."
        )
