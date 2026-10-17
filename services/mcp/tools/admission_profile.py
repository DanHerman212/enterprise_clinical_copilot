"""The admission profile tool.

Pre-flight, not a reasoning step. The agent calls this itself once per turn,
before the model runs, and folds the result into the context as verified ground
truth. The model never sees it as a tool (it is excluded from the function
declarations in `services/agent/mcp_client.py`), so it cannot be spent from the
turn's tool budget and cannot be retried by a model that misreads it.

That distinction is the whole design. An administrative question — "what is the
admission type?" — is answerable from a row that is already being read to score
the patient, and a model that has to decide to go and fetch it will sometimes
decide not to, or spend three of its five calls doing it. Stating the row up
front makes the answer free.
"""

import asyncio
import logging
from functools import lru_cache
from typing import Any

from ..contracts import AdmissionProfileResult, ToolError, tool_error
from ..dependencies.features import FEATURE_SOURCE, FeatureSource, get_feature_source
from ._validation import valid_hadm_id

_LOG = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _source() -> FeatureSource:
    """Cached so the BigQuery client is built once."""
    return get_feature_source()


def _error(hadm_id: int, code: str, message: str) -> dict[str, Any]:
    return tool_error(code, message, hadm_id=hadm_id, feature_source=FEATURE_SOURCE)


def _profile(hadm_id: int) -> dict[str, Any]:
    """Blocking implementation. Wrapped in a thread by the tool below."""
    if not valid_hadm_id(hadm_id):
        return _error(hadm_id, "bad_request", "hadm_id must be a positive integer")
    try:
        fields = _source().fetch_profile(hadm_id)
    except KeyError:
        return _error(
            hadm_id, "unknown_patient",
            f"No admission {hadm_id} in the feature source ({FEATURE_SOURCE}).",
        )
    except Exception:  # network, auth, unsynced online store
        # Detail stays server-side (ECC-21): exception text carries table names,
        # project ids and URLs, and this payload reaches the model.
        _LOG.error("profile: read failed for hadm %s", hadm_id, exc_info=True)
        return _error(
            hadm_id, "feature_fetch_failed",
            "The feature source could not be read. Try again shortly.",
        )
    if not fields:
        return _error(
            hadm_id, "incomplete_features",
            "The feature source returned no profile for this admission.",
        )
    return {
        "hadm_id": hadm_id,
        "profile": fields,
        "feature_source": FEATURE_SOURCE,
    }


async def get_admission_profile(hadm_id: int) -> AdmissionProfileResult | ToolError:
    """The administrative profile of one hospital admission, as words.

    Returns the values the readmission model was scored on — age, sex, admission
    type, discharge location, insurance, race and the utilisation counts —
    decoded from the same encoded row the score came from. Use it to answer
    administrative questions that the discharge note never states.

    Args:
        hadm_id: MIMIC-IV hospital admission id.
    """
    # BigQuery is synchronous; under the HTTP transport a blocking read would
    # stall the event loop for every concurrent caller.
    return await asyncio.to_thread(_profile, hadm_id)
