"""Hash-verified joblib artifact I/O (ECC-31).

The training/eval components pass the fitted XGBoost model between steps as a
joblib pickle (an sklearn XGBClassifier). Pickle deserialization is arbitrary
code execution, so each artifact carries a SHA-256 sidecar — written on dump,
checked on load — and a tampered or mismatched artifact fails loudly instead of
being deserialized.

Serving does NOT use this path: the deployed endpoint reads the native
`model.bst` (a non-executable format), verified via the bundle's checksums.json
(ECC-61).
"""

import hashlib
import os

import joblib


def dump(obj, path: str) -> None:
    """joblib.dump + write a `<path>.sha256` sidecar."""
    joblib.dump(obj, path)
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    with open(f"{path}.sha256", "w", encoding="utf-8") as fh:
        fh.write(digest + "\n")


def load(path: str):
    """Verify the `<path>.sha256` sidecar, then joblib.load."""
    sidecar = f"{path}.sha256"
    if not os.path.exists(sidecar):
        raise RuntimeError(
            f"artifact integrity: missing {sidecar} — refusing to deserialize "
            "an unverifiable pickle (ECC-31)."
        )
    with open(path, "rb") as fh:
        actual = hashlib.sha256(fh.read()).hexdigest()
    with open(sidecar, encoding="utf-8") as fh:
        expected = fh.read().strip()
    if actual != expected:
        raise RuntimeError(
            f"artifact integrity: checksum mismatch for {path} "
            f"(expected {expected}, got {actual})"
        )
    return joblib.load(path)
