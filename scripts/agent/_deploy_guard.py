"""Shared guard: refuse to deploy a real-corpus index to a public endpoint.

The Vector Search endpoint is public (PSC/VPC would be required to serve real
MIMIC-derived data — ECC-36), so the only thing standing between a misdeploy
and a DUA exposure is a scale check: the synthetic demo corpus is a few
hundred vectors; the real MIMIC-derived index is ~555k. Anything over the
limit is refused before the endpoint is created or touched.
"""

SYNTHETIC_VECTOR_LIMIT = 100_000


def assert_synthetic_scale(vectors: int, index_ref: str) -> None:
    if vectors > SYNTHETIC_VECTOR_LIMIT:
        raise SystemExit(
            f"refusing to deploy {index_ref} ({vectors} vectors) to a public "
            f"endpoint: >{SYNTHETIC_VECTOR_LIMIT} vectors means the real "
            "MIMIC-derived corpus, not the synthetic demo cohort (ECC-36/53)."
        )
