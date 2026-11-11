"""A risk model's own value, cited to a laboratory series that never held it.

The defect this closes is one sentence, served on turn 3 of `dev_t4_01` (2026-11-11):

    The patient's monocyte count also fell to a nadir of 0.47 x10^3/uL ^[4]

`^[4]` is the `monocytes` trajectory, and that trajectory's lowest reading is `0.5`. The
`0.47` is the model's input: `predict_readmission` returned `top_factors[0].value` as
`0.47`, and the generator that draws a trajectory reproduces an anchor at the precision
the analyte is REPORTED at — `monocytes` is `decimals=1` — so the series it drew says
`0.5`. Two numbers, both true of their own source, and the sentence used one to cite the
other.

Which is the same shape as the discharge label's Plavix sentence: a marker that resolves
correctly, a claim true of one source, and a heading or a reading that promises another.
Prompt rules had already been written for it — `RISK_SPOKE` says a series "is a different
measurement, not the model's input" — and a model turn is not a guarantee. So the bound is
enforced here, after the markers are resolved and before the page is numbered.

Three deliberate limits.

It acts only where the turn holds a prediction with a numeric `top_factors[].value`. A turn
with no estimate has no risk value to confuse with a reading, and a rule written for a
comparison that cannot occur would fire on prose it has no opinion about.

It acts only when the sentence quotes NO reading from the series it cites. A sentence that
quotes the series — "the monocyte count fell from 0.7 to 0.5 x10^3/uL ^[4]" — is citing the
series correctly, whatever else it mentions, and the model's `0.47` may sit beside it as the
model's input. What is removed is a marker on a sentence that never touches the passage it
names.

It removes the MARKER, not the sentence or the number. Deleting the sentence would delete
the model's input from the answer, which is a value the record does hold and the reader
asked about; deleting the number would edit clinical prose. What was wrong was the
attribution and the attribution is what goes. The turn records a flag, so the removal is
countable rather than silent.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from services.agent.postprocess import citations
from services.agent.postprocess.discharge_section import split_sentences

#: The tool whose `top_factors` carry the model's own inputs.
PREDICTION_TOOL = "predict_readmission"

#: The tool whose `trends` are a laboratory series.
LAB_TOOL = "get_lab_trends"

#: What the flag says, and the one spelling a counter reads.
FLAG = "risk_value_lab_citation_stripped"

#: A number as prose writes it. Not a citation pattern — Law 1's marker language is
#: `citations`' and this module never spells it — but the arithmetic comparison below
#: needs the numerals a sentence carries.
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

#: A unit that carries a digit of its own. `0.47 x10^3/uL` holds `10` and `3` as well as
#: the reading, and `mL/min/1.73m2` holds `1.73`; read as values, any of them could match an
#: administrative feature and make the guard act on a sentence quoting nothing clinical.
#: Removed before the scan, so only the numbers a clinician would read survive it.
_UNIT_TOKEN_RE = re.compile(
    r"x\s*10\s*\^\s*\d+"      # x10^3, x10^6
    r"|\b10\s*\^\s*[36]\b"    # 10^3, 10^6 written bare
    r"|\b1\.73\s*m2\b",       # mL/min/1.73m2, 1.73 m2
    re.IGNORECASE,
)

#: Features whose number is an administrative or binary fact, never a measurement. Named
#: because the analyte-prefix test below cannot see them: `prior_inpatient_days` holds no
#: analyte, so it is excluded by falling through, but `ed_visits` and `age` would too —
#: and the day a panel grows an analyte called `age` this list is what keeps the guard off
#: a feature that is not one.
_NOT_LABORATORY: frozenset[str] = frozenset({
    "age", "ed_visits", "recent_ed_visits", "prior_inpatient_days", "length_of_stay",
    "index_los_days", "oncology_flag", "has_procedure", "los_days",
})

#: How the feature row names a point on a measurement's series: `monocytes_min`,
#: `rdw_max`, `rbc_last`. Read from the payload's own naming rather than from a second
#: copy of the laboratory panel, which would be a vocabulary to keep in step.
_MEASUREMENT_SUFFIXES: tuple[str, ...] = ("min", "max", "last", "mean")


def _call_name(call: Mapping[str, Any]) -> str:
    return str(call.get("name") or "")


def _call_payload(call: Mapping[str, Any]) -> Any:
    payload = call.get("response")
    return payload if payload is not None else call.get("payload")


def _numeric(value: Any) -> float | None:
    """The number a payload field carries, or None. Booleans are not numbers here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _numbers_in(text: str) -> set[float]:
    """Every numeral the sentence writes as a clinical value.

    Markers and unit exponents go first: `^[1]` is an attribution and `x10^3/uL` is a
    scale, and neither is a reading a sentence can be said to quote. A qualitative
    sentence — "his hemoglobin remained low at discharge ^[1]" — then holds no numbers at
    all, which is the answer that keeps the guard off it.
    """
    scrubbed = _UNIT_TOKEN_RE.sub(" ", citations.strip_markers(text or ""))
    found: set[float] = set()
    for token in _NUMBER_RE.findall(scrubbed):
        try:
            found.add(float(token))
        except ValueError:
            continue
    return found


def _laboratory_analytes(calls: Iterable[Mapping[str, Any]] | None) -> tuple[set[str],
                                                                            set[str]]:
    """The analyte keys and the anchor names this thread's panels named.

    Two vocabularies, because `predict_readmission` names a factor from the feature row
    (`monocytes_min`, `rdw_max`) while a panel names the analyte (`monocytes`, `rdw`). A
    feature is a laboratory factor if it is an anchor a series was drawn to reproduce, or
    if it opens with an analyte the panel knows.
    """
    analytes: set[str] = set()
    anchors: set[str] = set()
    for call in calls or ():
        if not isinstance(call, Mapping) or _call_name(call) != LAB_TOOL:
            continue
        payload = _call_payload(call)
        if not isinstance(payload, Mapping):
            continue
        for trend in payload.get("trends") or ():
            if not isinstance(trend, Mapping):
                continue
            key = str(trend.get("key") or "").strip().lower()
            if key:
                analytes.add(key)
            for anchor in trend.get("anchored") or ():
                name = str(anchor or "").strip().lower()
                if name:
                    anchors.add(name)
    return analytes, anchors


def _is_laboratory_factor(feature: Any, analytes: set[str], anchors: set[str]) -> bool:
    """Whether a `top_factors` entry is a measurement rather than an administrative fact.

    The distinction is the whole rule: a laboratory factor's value is a number a series
    was drawn to reproduce, so a sentence quoting it while citing that series is the
    confusion this module exists to remove. An administrative factor's value is a count —
    `oncology_flag` 1, `prior_inpatient_days` 0 — and a sentence quoting one is making no
    claim about a laboratory at all.

    Three ways to be a measurement, and no fourth. The anchor name is the strongest: the
    generator draws a series to reproduce that value, so `monocytes_min` is a measurement
    by construction. An analyte the panel returned catches a factor named for its series
    but not anchored to it. The suffix convention catches the rest — `_min`, `_max`,
    `_last` and `_mean` are how the feature row names an extreme or an end of a series,
    and no administrative field is named that way.

    A count is not caught by any of the three, which is the point: `age`, `ed_visits` and
    `prior_inpatient_days` are counted, not measured.
    """
    name = str(feature or "").strip().lower()
    if not name or name in _NOT_LABORATORY or name.endswith("_flag"):
        return False
    if name in anchors:
        return True
    if any(name == key or name.startswith(f"{key}_") for key in analytes):
        return True
    return any(name.endswith(f"_{suffix}") for suffix in _MEASUREMENT_SUFFIXES)



def risk_values(calls: Iterable[Mapping[str, Any]] | None,
                evidence_state: Any = None) -> set[float]:
    """The model's LABORATORY inputs, from every prediction this turn can see.

    The record's `prediction` first, because it is the payload the model was shown, and
    then every prediction among the calls — a thread can hold two estimates and a value
    from either is still the model's input rather than a measurement.

    Laboratory factors only. An administrative or binary factor's value is a count, not a
    measurement: `oncology_flag` 1 and `prior_inpatient_days` 0 are the patient's, and no
    laboratory series was ever drawn to reproduce them, so a sentence quoting one is
    making no claim about a laboratory and must not lose its marker. Including them put
    `0` and `1` into the comparison set, where they match the first count a sentence
    happens to mention.
    """
    analytes, anchors = _laboratory_analytes(calls)
    values: set[float] = set()
    payloads: list[Any] = []
    record = getattr(evidence_state, "prediction", None)
    if isinstance(record, Mapping):
        payloads.append(record)
    for call in calls or ():
        if not isinstance(call, Mapping) or _call_name(call) != PREDICTION_TOOL:
            continue
        payloads.append(_call_payload(call))
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for factor in payload.get("top_factors") or ():
            if not isinstance(factor, Mapping):
                continue
            if not _is_laboratory_factor(factor.get("feature"), analytes, anchors):
                continue
            number = _numeric(factor.get("value"))
            if number is not None:
                values.add(number)
    return values


def _series_by_analyte(calls: Iterable[Mapping[str, Any]] | None
                       ) -> dict[str, list[float]]:
    """Every laboratory series the calls produced, keyed by analyte."""
    series: dict[str, list[float]] = {}
    for call in calls or ():
        if not isinstance(call, Mapping) or _call_name(call) != LAB_TOOL:
            continue
        payload = _call_payload(call)
        if not isinstance(payload, Mapping):
            continue
        for trend in payload.get("trends") or ():
            if not isinstance(trend, Mapping):
                continue
            key = str(trend.get("key") or "").strip()
            if not key:
                continue
            values = [
                number
                for number in (_numeric(point.get("value"))
                               for point in trend.get("series") or ()
                               if isinstance(point, Mapping))
                if number is not None
            ]
            if values:
                series.setdefault(key, []).extend(values)
    return series


def _lab_positions(replayed: Iterable[Mapping[str, Any]] | None,
                   this_turn: Iterable[Mapping[str, Any]] | None,
                   evidence_state: Any) -> dict[int, list[float]]:
    """Position in the evidence array -> that analyte's readings.

    Two routes, because the array has two descriptions and only one of them is the
    numbering a marker was written against. With the session record, the position is the
    record's and the analyte is the one the record filed there. Without it, the array is
    the concatenation `citations.citable_passages` makes — the replayed turns' units first,
    then this turn's — so walking the same calls in the same order reproduces the same
    integers.
    """
    by_analyte = _series_by_analyte([*(replayed or ()), *(this_turn or ())])
    positions: dict[int, list[float]] = {}
    items = getattr(evidence_state, "items", None)
    ordinals = getattr(evidence_state, "ordered_citable_units", None)
    if isinstance(items, Mapping) and callable(ordinals):
        for position, unit in enumerate(ordinals(), start=1):
            if not isinstance(unit, Mapping) \
                    or unit.get("section") != citations.LAB_SECTION:
                continue
            item = items.get(position)
            analyte = str(getattr(item, "section", "") or "").strip()
            if analyte and analyte in by_analyte:
                positions[position] = by_analyte[analyte]
        return positions
    cursor = 1
    for call in [*(replayed or ()), *(this_turn or ())]:
        if not isinstance(call, Mapping):
            continue
        payload = _call_payload(call)
        if not isinstance(payload, Mapping):
            continue
        units = citations.citable_passages(payload)
        trends = [trend for trend in (payload.get("trends") or ())
                  if isinstance(trend, Mapping)]
        for index, unit in enumerate(units):
            if isinstance(unit, Mapping) \
                    and unit.get("section") == citations.LAB_SECTION:
                # `citable_passages` maps one analyte to one unit in the order the panel
                # gave them, so the analyte is read from the same index.
                trend = trends[index] if index < len(trends) else {}
                values = by_analyte.get(str(trend.get("key") or "").strip()) or []
                if values:
                    positions[cursor] = values
            cursor += 1
    return positions


def filter_risk_value_lab_citations(
    prose: str,
    *,
    tool_calls: Iterable[Mapping[str, Any]] | None = None,
    replayed_tool_calls: Iterable[Mapping[str, Any]] | None = None,
    evidence_state: Any = None,
) -> tuple[str, tuple[str, ...]]:
    """The prose with risk-value markers taken off sentences that never read the series.

    A sentence is edited when all three hold: it carries a numeric marker pointing at a
    laboratory series, it quotes one of the model's own inputs, and it quotes no reading
    from the series it cites.

    Returns `(text, flags)`. The flag names the position, so a turn that lost a marker says
    which one — a citation that disappears without a record of it is a defect the harness
    cannot count.
    """
    if not prose or "^[" not in prose:
        return prose, ()

    values = risk_values([*(replayed_tool_calls or ()), *(tool_calls or ())],
                         evidence_state)
    if not values:
        return prose, ()
    positions = _lab_positions(replayed_tool_calls, tool_calls, evidence_state)
    if not positions:
        return prose, ()

    stripped: list[int] = []
    text = prose
    cursor = 0
    for sentence in split_sentences(prose):
        # `split_sentences` returns stripped pieces of the prose it was given, so each one
        # is a substring of it, and searching from a cursor keeps them in order. Spliced
        # rather than rebuilt: the paragraph breaks and the blank lines between headings
        # belong to the page and not to this pass.
        at = text.find(sentence, cursor)
        if at < 0:
            continue
        cited = citations.cited_numbers(sentence)
        if not cited:
            cursor = at + len(sentence)
            continue
        quoted = _numbers_in(sentence)
        offending = [
            position for position in cited
            if position in positions
            # The sentence quotes this series, so the marker is doing its job — even when
            # the model's own input sits in the same sentence beside it.
            and not quoted & set(positions[position])
            and quoted & values
        ]
        if offending:
            replacement = citations.remove_citations(sentence, offending)
            text = f"{text[:at]}{replacement}{text[at + len(sentence):]}"
            stripped.extend(offending)
            cursor = at + len(replacement)
        else:
            cursor = at + len(sentence)

    if not stripped:
        return prose, ()
    return text, tuple(f"{FLAG}:^{position}" for position in sorted(set(stripped)))
