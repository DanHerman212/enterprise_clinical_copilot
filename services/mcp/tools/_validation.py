"""Shared request validation for the MCP tools (ECC-28).

One definition of a valid admission id, applied by every tool entry point, so
a malformed hadm_id gets a clean ``bad_request`` everywhere instead of an
opaque BigQuery binding error in whichever tool forgot to check.
"""

from typing import Any


def valid_hadm_id(hadm_id: Any) -> bool:
    """True iff hadm_id is a positive int (bool is an int subclass — reject)."""
    return (
        isinstance(hadm_id, int)
        and not isinstance(hadm_id, bool)
        and hadm_id > 0
    )
