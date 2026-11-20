"""The admission's structured features, stated before the model runs.

The blind spot this closes
--------------------------
`predict_readmission` scores a tabular row and returns the features that drove
the score — `admission_type` among them. The retrieval tools search discharge-note
text, and a discharge summary writes the clinical narrative without ever declaring
"Admission type: URGENT". So the agent could correctly report that admission type
was a driver and then, held to its grounding rules, answer that the note does not
document it. Both statements were true and the pair was useless.

The fix is pre-flight hydration rather than a new capability for the model. The
profile is fetched once, by the caller, against the admission the turn is already
about, and folded into the system prompt as verified ground truth. Three
consequences, each of them the reason for this shape:

* **No tool budget.** The model never spends one of its five calls on a fact that
  is already in front of it, and a turn cannot run out of budget trying.
* **No schema reasoning.** "What is the admission type?" is answered from a row
  that was read to compute the score; there is no SQL to generate, nothing to
  hallucinate and no warehouse surface exposed to a prompt.
* **No second source of truth.** The profile is decoded from the same row the
  booster scored (`services/mcp/dependencies/features/profile.py`), so the words
  and the number cannot disagree.

Failure is silence, not a broken turn: a profile that cannot be read leaves the
prompt exactly as it was. An answer that says the note does not document something
is worse than one that knows — but far better than a turn that fails.

The read cache
--------------
Two of the calls a turn makes before it asks the model anything are deterministic reads of
this admission: the profile above, and the passage re-resolution `graph.replay_messages`
performs for a stored turn whose payload was not kept. Both returned the same payload on
every turn of a conversation, and a six-turn thread therefore paid for the same reads six
times before the model saw a word. `cached_read` holds one answer for a few minutes, keyed
strictly by `(hadm_id, tool, arguments)`, so no answer can cross between patients and no
two different calls can share an entry.
"""

from __future__ import annotations

import collections
import copy
import json
import logging
import math
import time
from typing import Any, Mapping

from services.agent.routing import encounter
from shared.labels import humanize

_LOG = logging.getLogger(__name__)

#: The MCP tool this calls. It is registered on the server so the pre-flight call
#: can reach it, and hidden from the model's function declarations so the model
#: cannot call it a second time (see `mcp_client.gemini_tool`).
PROFILE_TOOL = "get_admission_profile"

#: Every tool the caller invokes on the model's behalf before the model runs.
#: Owned here, by the module that makes the call, because `routing` and
#: `mcp_client` both have to agree with it and `contracts` cannot be the owner:
#: `contracts` imports `routing`, so a constant defined there and needed by
#: `routing` is a circular import at module load.
PREFLIGHT_TOOLS = frozenset({PROFILE_TOOL})

#: Field order in the block. The keys come from the server's decoder, which owns
#: the wording; this module only decides what a reader sees first.
_FIELDS: tuple[tuple[str, str], ...] = (
    ("age", "age"),
    ("gender", "gender"),
    ("admission_type", "admission_type"),
    ("discharge_location", "discharge_location"),
    ("insurance", "insurance"),
    ("race", "race"),
    ("prior_admission_count", "prior_admissions"),
    ("prior_inpatient_days", "prior_inpatient_days"),
    ("recent_ed_visits", "ed_visits_prior_year"),
    ("index_los_days", "length_of_stay_days"),
)


def _display(value: Any, feature: str = "") -> str:
    """A number at the clinical precision the rest of the turn uses, else as it stands.

    The feature store returns `prior_inpatient_days` as "24.6667" because that is the
    encoded row. This function used to round it to one decimal — 16.0417 became "16.0" —
    on the argument that four decimals claim a precision neither the record nor the
    model has, after an answer was seen saying "recorded as 16.0417 days"
    (`agent-00177-qr5`, 2026-10-24).

    It then swung to passing four decimals through exactly, because the manifest is
    verified ground truth that the judge holds the answer against, and a manifest stating
    "24.7" for a row holding 24.6667 is a manifest that disagrees with the dataset.

    Both were half a rule. The risk result publishes the same row's value at TWO decimals
    (`prediction._clinical_value`), so a briefing that read the manifest said "length of
    stay was 42.4583 days" and then, three lines later, "the recorded value was 42.46" —
    one number, two precisions, in one answer (measured 2026-11-20, 90000069). The
    precision is now the published one, stated once: two decimals at most, trailing zeros
    and a whole-number fraction dropped, so the manifest, the risk projection, the
    guardrail and the prose all write the same digits. Anything that is not a number —
    "Emergency (EMER)", "Home" — is returned untouched.

    A feature named in `COUNT_FEATURES` is a whole number instead, whatever the row holds:
    see `clinical_value`.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return _clinical_number(float(value), feature)
    return _clinical_number_text(str(value).strip(), feature)


#: Features whose row value counts discrete things rather than measuring one.
#:
#: `medication_order_count` was served as 4.85 and `prior_inpatient_days` as 4.75
#: (measured 2026-11-20, 90000020 and 90000015): the columns hold an order rate and an
#: averaged day count under names that promise a tally. Two decimals on a count reads as a
#: formatting fault — "4.85 orders" is not a number of orders — and the fraction carries no
#: clinical meaning, so the last digit is not information the reader is owed.
#:
#: Length of stay is deliberately absent. It is a duration the record measures, `42.46`
#: days is the number the risk result publishes for it, and rounding it to 42 would
#: disagree with the payload beside it (see `_display`).
COUNT_FEATURES: frozenset[str] = frozenset({
    "procedure_count",
    "medication_count",
    "medication_order_count",
    "prior_admission_count",
    "prior_inpatient_days",
    "recent_ed_visits",
})


def clinical_value(feature: str, value: float) -> int | float:
    """A feature's value at the precision a clinician would write it.

    The one place the rule lives, because four layers state the same number and three of
    them used to round it their own way: this module renders `<admission_metadata>`, the
    risk projection fills `top_factors[].value`, the session record repeats it, and the
    guardrail rewrites what the model wrote. A count rounds half UP rather than to even —
    Python's `round(4.5)` is 4, and a tally of four and a half is five — and everything
    else keeps two decimals, which is what `prediction._clinical_value` publishes.
    """
    if feature in COUNT_FEATURES:
        return int(math.floor(float(value) + 0.5))
    rounded = round(float(value), 2)
    return int(rounded) if rounded == int(rounded) else rounded


def _clinical_number(value: float, feature: str = "") -> str:
    """A count whole, everything else two decimals at most."""
    number = clinical_value(feature, value)
    if isinstance(number, int):
        return str(number)
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _clinical_number_text(text: str, feature: str = "") -> str:
    """The same rule for a value that arrived as a string, as this row's columns do.

    A string that is not a finite number is not this function's business: a label, a
    facility name and a "Not recorded" all pass through as they arrived.
    """
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if number != number or number in (float("inf"), float("-inf")):
        return text
    return _clinical_number(number, feature)


def _clean(value: Any, feature: str = "") -> str:
    """One line, no markup. The values are ours, and this keeps them ours.

    They reach a prompt, so a newline or an angle bracket in one would let a
    data value change the shape of the instruction around it. Nothing upstream
    produces such a value today; the check costs nothing and removes the class.
    """
    text = " ".join(_display(value, feature).split())
    return text.replace("<", "(").replace(">", ")")


#: Fields that only exist because there was a stay.
#:
#: Dropped from the block when the record is an outpatient visit. The distinction is not
#: cosmetic: `index_los_days` is a real number on an outpatient row (the encoder built the
#: row as an admission), and stating it is what produced the canary's "length of stay was
#: 7.9 days" for an encounter that had none. A fact the record does not have is not made
#: true by the column being populated.
STAY_ONLY_FIELDS: frozenset[str] = frozenset({
    "admission_type", "discharge_location", "index_los_days",
})

#: The features a ranked risk driver can also be one of the block's fields.
#:
#: `gender` is here because the model's row carries it as 0/1 while the block carries the
#: word. The four one-hot parents are here because `predict_readmission` has no single value
#: to state for them (`prediction._scored_factor` omits the key) and the block already holds
#: the decoded level. Nothing else is: every other manifest field also has a numeric value
#: in the risk payload, and preferring the block's string for one of those would give one
#: feature two written forms (2026-11-20).
ADMINISTRATIVE_FEATURES: frozenset[str] = frozenset({
    "gender", "admission_type", "discharge_location", "insurance", "race",
})

#: The decoder's word for a value the row does not hold (`features.profile.ABSENT`). It is
#: not passed on as a factor's value: "Not recorded" states nothing, and the prose would
#: read "(patient value: Not recorded)". The factor keeps its direction and no value, which
#: is the honest form of the same fact.
_NOT_RECORDED = "Not recorded"


def administrative_values(block: str) -> dict[str, str]:
    """The administrative values a rendered block states, keyed by FEATURE name.

    The risk payload and the block name one feature two ways: the payload names the model's
    parent feature, the block states the decoded level under a label that is usually the
    same word. Two labels are not the feature name (`prior_admissions`,
    `length_of_stay_days`), so the mapping is read back out of `_FIELDS` rather than
    assumed.

    Read from the text rather than from the profile dict because the block is what the
    model was given: a value that reached the window through an escape is the value the
    answer will be held to.

    Keyed by the FEATURE name and by its humanized label. Both readers exist and neither
    can do the other's lookup: the risk projection holds the raw key (`admission_type`),
    and the session record holds the label it was projected into (`admission type`). Two
    keys for one value is cheaper than a second vocabulary in either reader.
    """
    by_label = {label: key for key, label in _FIELDS}
    values: dict[str, str] = {}
    for line in str(block or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("- ") or ": " not in stripped:
            continue
        label, _, stated = stripped[2:].partition(": ")
        key = by_label.get(label.strip())
        if key not in ADMINISTRATIVE_FEATURES:
            continue
        stated = stated.strip()
        if not stated or stated == _NOT_RECORDED:
            continue
        values[key] = stated
        values[humanize(key)] = stated
    return values

#: Said in place of the dropped fields, rather than leaving their absence to be inferred.
#: A model handed a block with no length of stay will sometimes supply one from the note;
#: a model told there is no stay does not.
OUTPATIENT_LINE = (
    "encounter_type: outpatient encounter — this record is a visit, not an inpatient "
    "stay, so it has no admission type, no discharge destination and no length of stay"
)


def render(hadm_id: int, profile: dict[str, Any]) -> str:
    """The `<admission_metadata>` block for one admission, or "" if unusable.

    The block names what the row IS, not which row it is: no admission id, because this
    text goes into the model's window and the identifier is the one value in it that must
    never be repeated back. The model was reading it out of the tag and writing it into
    tool arguments — `discharge_medications({"hadm_id": "90000069"})`, measured 2026-11-06
    — which the session then had to accept or refuse on a value it had already supplied;
    the number is the session's to know, and a turn that cannot see it cannot mis-state it.
    The argument is still accepted and still names the admission in the log line below,
    because a reader of this function's signature is entitled to know what it is about.
    """
    if not isinstance(profile, dict) or not profile:
        return ""
    outpatient = encounter.is_outpatient(encounter.classify(profile)) is True
    lines = [
        f'<admission_metadata source="feature store row '
        f'scored by the readmission model">'
    ]
    if outpatient:
        lines.append(f"- {OUTPATIENT_LINE}")
    for key, label in _FIELDS:
        if outpatient and key in STAY_ONLY_FIELDS:
            continue
        value = profile.get(key)
        if value in (None, ""):
            continue
        lines.append(f"- {label}: {_clean(value, key)}")
    lines.append("</admission_metadata>")
    # A block with no fields in it is noise that also implies a fact was found.
    return "\n".join(lines) if len(lines) > 2 else ""


async def hydrate(box: Any, hadm_id: int | None) -> str:
    """Fetch and render the profile, and record the turn's encounter type.

    The encounter type is set here, unconditionally, because this is the one place the
    record's own row is read before anything else runs: the graph, the command gate, the
    canvas and the response composer all read it back through `encounter` rather than
    recognising an admission id. Setting None as well as a value is what stops a turn with
    no profile from inheriting the previous turn's answer.
    """
    encounter.set_current(None)
    if not hadm_id or box is None:
        return ""
    try:
        payload = await cached_read(
            box, PROFILE_TOOL, {"hadm_id": int(hadm_id)}, hadm_id=hadm_id)
    except Exception:  # noqa: BLE001 — a missing profile may not fail the turn
        _LOG.warning("admission profile pre-flight failed for hadm %s", hadm_id,
                     exc_info=True)
        return ""
    if not isinstance(payload, dict) or payload.get("error"):
        # Expected rather than alarming: an admission outside the cohort has no
        # row, and the turn proceeds without the block.
        _LOG.info(
            "admission profile unavailable for hadm %s: %s",
            hadm_id, payload.get("error") if isinstance(payload, dict) else "no payload",
        )
        return ""
    profile = payload.get("profile") or {}
    encounter.set_current(encounter.classify(profile))
    return render(hadm_id, profile)


# ---------------------------------------------------------------------------
# The short-lived read cache.
# ---------------------------------------------------------------------------

#: How long one answer is reused. Minutes, not hours: a deployment's reads are
#: deterministic within a revision, and a revision that lands mid-conversation should not
#: be invisible to the next turn for longer than it takes somebody to notice.
READ_CACHE_TTL_SECONDS = 300.0

#: How many answers are held at once. A bound is a requirement rather than a courtesy: the
#: service is long-lived and the key carries an admission id, so an unbounded table keyed
#: on request-shaped data is a leak with a friendly name. 128 entries is a few
#: conversations' worth of pre-flight and replay reads on a warm instance.
READ_CACHE_MAX_ENTRIES = 128

#: The reads this cache may answer: deterministic reads of ONE admission's record, and
#: nothing else. `search_literature` is deliberately absent — it asks a third party, and a
#: five-minute-old cohort is a different answer rather than a cheaper one.
CACHEABLE_READS: frozenset[str] = frozenset({
    PROFILE_TOOL, "rag_search", "rag_search_sections", "discharge_medications",
    "get_lab_trends", "predict_readmission",
})

_CACHE_KEY = tuple  # (tool, hadm_id, canonical arguments)

_READ_CACHE: "collections.OrderedDict[tuple, tuple[float, Any]]" = collections.OrderedDict()

#: The clock, as a module attribute so a test can move it without sleeping through a TTL.
_clock = time.monotonic


def _canonical(arguments: Mapping[str, Any] | None) -> str:
    """One spelling per argument set, so two identical calls share one entry.

    `sort_keys` is the whole of it: `{"hadm_id": 1, "query": "x"}` and the same pair in
    the other order are one call. `default=str` because a stored turn may carry an id as
    an integer where a live call carries it as a string — both are the same call, and
    neither may raise here.
    """
    return json.dumps(dict(arguments or {}), sort_keys=True, default=str)


def clear_read_cache() -> None:
    """Forget every cached answer. Called between tests, and by nothing in the service."""
    _READ_CACHE.clear()


def read_cache_size() -> int:
    """How many answers are currently held — for a test, and for a log line."""
    return len(_READ_CACHE)


def _served(box: Any) -> bool:
    """Whether this toolbox talks to a served MCP endpoint rather than standing in for one.

    The cache must never make two tests share an answer, and a test's toolbox is a stub by
    construction: it either replaces the class entirely or carries a `call` of its own on
    the instance. A served toolbox holds a `ClientSession` and resolves `call` from its
    class, so those two facts are the test. An offline seam therefore gets every call
    straight through, with nothing written and nothing read.
    """
    if getattr(box, "session", None) is None:
        return False
    return "call" not in getattr(box, "__dict__", {})


async def cached_read(box: Any, name: str, arguments: Mapping[str, Any],
                      *, hadm_id: int | None) -> Any:
    """One deterministic read of one admission, answered from the cache when possible.

    The key is `(tool, hadm_id, canonical arguments)`, and the admission is an explicit part
    of it rather than something inferred from the arguments: a call whose arguments happen
    to omit `hadm_id` must still never share an entry with another patient's read.

    Three things invalidate: the TTL, the entry count, and a payload that reports an error.
    An error is not cached at all — a transport failure pinned for five minutes would
    outlive the outage that caused it — and an expired entry is dropped rather than served.

    A miss is invisible. The caller makes exactly the call it always made, the answer it
    gets back is the tool's own object, and the table keeps a snapshot: a caller that
    projects or prunes a payload (as the replay does) must not be able to change what the
    next read of that tool returns.
    """
    if name not in CACHEABLE_READS or not _served(box):
        return await box.call(name, dict(arguments or {}))
    key = (str(name), None if hadm_id is None else int(hadm_id), _canonical(arguments))
    now = _clock()
    held = _READ_CACHE.get(key)
    if held is not None:
        expiry, snapshot = held
        if now < expiry:
            _READ_CACHE.move_to_end(key)
            _LOG.debug("read cache hit: %s for hadm %s (%d held)",
                       name, hadm_id, len(_READ_CACHE))
            return copy.deepcopy(snapshot)
        _READ_CACHE.pop(key, None)
    payload = await box.call(name, dict(arguments or {}))
    if isinstance(payload, dict) and payload.get("error"):
        return payload
    _READ_CACHE[key] = (now + READ_CACHE_TTL_SECONDS, copy.deepcopy(payload))
    while len(_READ_CACHE) > READ_CACHE_MAX_ENTRIES:
        _READ_CACHE.popitem(last=False)
    return payload
