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
from ..config import FEATURE_SOURCE
from ..dependencies.features import FeatureSource, get_feature_source
from ._validation import valid_hadm_id

_LOG = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _source() -> FeatureSource:
    """Cached so the BigQuery client is built once."""
    return get_feature_source()


#: Fields that only exist because there was a stay.
#:
#: `index_los_days` is a real number on an outpatient row — the encoder built the row as
#: though it were an admission — and `discharge_location` is that row's disposition.
#: Neither describes a visit: the encounter has no length of stay and nothing was
#: discharged. Dropping them here means no consumer can repeat the number, whatever it
#: chooses to do with the rest of the profile.
STAY_ONLY_PROFILE_FIELDS: tuple[str, ...] = ("index_los_days", "discharge_location")

#: The decoder's own word for a visit (`dependencies/features/profile.py`,
#: `ENCOUNTER_LABELS`). Matched as a prefix so a later relabelling that keeps the noun
#: still classifies, and because nothing else in that vocabulary starts with this word.
OUTPATIENT_PREFIX = "outpatient"


def _is_outpatient(profile: dict[str, Any]) -> bool:
    return str(profile.get("encounter_type") or "").strip().lower().startswith(
        OUTPATIENT_PREFIX)


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
    if _is_outpatient(fields):
        fields = {key: value for key, value in fields.items()
                  if key not in STAY_ONLY_PROFILE_FIELDS}
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
