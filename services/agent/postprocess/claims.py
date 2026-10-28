"""Deterministic corrections the answer already has the evidence to make.

Everything here is a function of (answer, evidence) and nothing else: no model call, no
clock, no I/O. These are the three defects the Gate 3 ladder proved a prompt rule cannot
hold, each one rewritten where the facts are known rather than requested where they are
not.

  1. `rewrite_flat_claims` — a series that peaks, dips and returns to its earlier value
     must not be described as having "remained at" that value. `00d1` failed the item,
     `00d2` passed it, `00d3` failed it again with the same series and the same forbidden
     verb. A rule lowers the rate; a rewrite that reads the series makes the sentence
     false-impossible.

  2. `correct_absent_drivers` — a driver the model weights upward for a patient whose
     record says zero is a population-baseline effect, not a history. `00d` and `00d3` both
     answered "the primary driver increasing this risk is recent ED visits" for a record
     reading `recent_ed_visits: 0`. The projection now annotates the factor, and this
     rewrites the sentence if the annotation is ignored anyway.

  3. `rebind_citations` — a marker that resolves but does not hold the claim it is attached
     to. `00d3` cited `^[3]` (Clindamycin) for a sentence about the prednisone taper, and
     `^[2]` (hospital course) for the readmission estimate. Renumbering decides ORDER and
     the range check decides EXISTENCE; neither asks whether the passage supports the
     sentence. This does, by asking which unit actually contains the sentence's words.

Why all three live at settlement rather than in the presentation layer: the trace, the
judge and the browser read the answer from three places, and only the state is common to
all three. A correction applied downstream of the trace is a correction the judge never
sees — which is what made the `00d3` citation scores unreadable.
"""

from __future__ import annotations

import re

from services.agent.postprocess.citations import (
    _CITATION_RE,
    citable_passages,
    sanitize_citation_range,
)
from services.agent.presentation.domain import humanize
from services.agent.routing.admission_context import absent_in_manifest, manifest_value

#: Verbs that assert a value did not move. The prompt forbids all of them for a series
#: that moved; this list is what the rewrite matches on.
_FLAT_CLAUSE = re.compile(
    r"\b(?:remained|stayed|held\s+steady|persisted)(?:\s+unchanged)?\s+at\b",
    re.IGNORECASE,
)

#: Where a sentence ends: punctuation followed by whitespace or the end of the text. A
#: plain `[.!?]` split cuts "7.2 g/dL" in half and separates a claim from the analyte name
#: it contains.
_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")


def _sentences(text: str) -> list[str]:
    """The text's sentences, in order, covering every character exactly once.

    Every rewrite here substitutes within a sentence and rejoins, so a splitter that
    dropped a fragment would delete prose. This one cannot: each sentence ends at the
    first punctuation run followed by whitespace, and the tail is emitted whole.
    """
    out: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        out.append(text[start:match.end()])
        start = match.end()
    if start < len(text):
        out.append(text[start:])
    return out

#: Words that assert a factor is acting on this patient. Deliberately broad: the cost of a
#: false match is a rewritten sentence that states the record's own value, and the cost of
#: a missed match is a fabricated clinical history.
_ASSERTS_DRIVER = re.compile(
    r"\b(?:primary\s+driver|drivers?|drives?|driving|contribut\w*|increas\w*|"
    r"rais\w*|elevat\w*|due\s+to|because\s+of|attribut\w*|responsible)\b",
    re.IGNORECASE,
)

#: Words that already say the factor is not present — the sentence is contrasting the
#: model's weighting with the record, which is the behaviour the prompt asks for.
_DISOWNED = re.compile(
    r"\b(?:no|not|zero|none|never|absent|without|does\s+not|did\s+not|is\s+not|"
    r"are\s+not|lacks?|rather\s+than|baseline|population)\b",
    re.IGNORECASE,
)

#: Content words a citation is checked against. Short tokens are noise ("a", "of", "the"),
#: and numerals are excluded because a value appears in several passages at once.
_TOKEN = re.compile(r"[a-z][a-z0-9]{2,}")
_STOPWORDS = frozenset({
    "the", "and", "for", "was", "were", "with", "that", "this", "from", "has", "had",
    "have", "are", "but", "not", "its", "his", "her", "who", "which", "what", "when",
    "where", "than", "then", "there", "their", "they", "them", "she", "him", "also",
    "been", "being", "into", "over", "under", "more", "most", "less", "least", "both",
    "each", "per", "day", "days", "value", "values", "level", "levels", "note", "notes",
    "record", "records", "documented", "documents", "patient", "admission", "discharge",
    "during", "while", "after", "before", "above", "below", "same", "only", "such",
})


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _fmt(value: float, decimals: object) -> str:
    places = decimals if isinstance(decimals, int) and 0 <= decimals <= 4 else 2
    return f"{value:.{places}f}".rstrip("0").rstrip(".") if places else f"{value:.0f}"


def trajectory_summary(trend: object) -> dict | None:
    """The bounce-and-return shape of one analyte, or None if it has no such shape.

    The shape this exists for: the final value equals an earlier extreme, and something
    between the two points is different. `4.42 -> 3.06 -> 3.47 -> 3.06` is that shape —
    the series ends where it dipped to, having risen in between, so any sentence of the
    form "remained at 3.06" is false about the course while every digit in it is right.

    A series whose last two points are equal, with nothing in between, is NOT this shape:
    `142 -> 137 -> 139 -> 139` genuinely did hold at 139 over its final interval.
    """
    if not isinstance(trend, dict):
        return None
    points = [p for p in trend.get("series") or [] if isinstance(p, dict)]
    if len(points) < 3:
        return None
    last = _number(points[-1].get("value"))
    if last is None:
        return None
    interior = points[:-1]
    repeated = [_number(p.get("value")) for p in interior]
    if last not in repeated:
        return None
    first_index = repeated.index(last)
    between = points[first_index + 1:-1]
    moved = [
        (point, value) for point in between
        if (value := _number(point.get("value"))) is not None
        and abs(value - last) > 1e-9
    ]
    if not moved:
        return None
    mid_point, mid_value = max(moved, key=lambda pair: abs(pair[1] - last))
    decimals = trend.get("decimals")
    unit = str(trend.get("unit") or "").strip()
    label = str(trend.get("label") or trend.get("key") or "").strip()
    verb = "rose to" if mid_value > last else "fell to"
    with_unit = f" {unit}" if unit else ""
    return {
        "label": label,
        "unit": unit,
        "value": _fmt(last, decimals),
        "mid": _fmt(mid_value, decimals),
        "mid_day": mid_point.get("day"),
        "last_day": points[-1].get("day"),
        "clause": (
            f"{verb} {_fmt(mid_value, decimals)}{with_unit} on day {mid_point.get('day')} "
            f"before returning to"
        ),
        "summary": (
            f"did NOT remain flat at {_fmt(last, decimals)}{with_unit}: it "
            f"{verb} {_fmt(mid_value, decimals)}{with_unit} on day {mid_point.get('day')} "
            f"before returning to {_fmt(last, decimals)}{with_unit} at discharge "
            f"(day {points[-1].get('day')})"
        ),
    }


def repeated_extremes(tool_calls: list[dict] | None) -> list[dict]:
    """Every analyte in the turn's evidence whose series returned to an earlier value."""
    shapes: list[dict] = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        response = call.get("response")
        if not isinstance(response, dict):
            continue
        for trend in response.get("trends") or []:
            shape = trajectory_summary(trend)
            if shape is not None:
                shapes.append(shape)
    return shapes


def _label_pattern(label: str) -> re.Pattern | None:
    if not label:
        return None
    # "Red blood cells" is written "red blood cell" and "red blood cells"; the trailing
    # plural on the last word is the only variation worth allowing.
    return re.compile(rf"\b{re.escape(label[:-1] if label.endswith('s') else label)}s?\b",
                      re.IGNORECASE)


def rewrite_flat_claims(answer: str, tool_calls: list[dict] | None) -> tuple[str, list[str]]:
    """Replace a flatness clause with the movement the series actually shows.

    The substitution is on the VERB PHRASE, not on the sentence: everything the answer
    said about the value — its digits, its unit, its day — is left exactly as written, and
    what changes is the claim that it did not move. "remained at 3.06 x10^6/uL at
    discharge" becomes "rose to 3.47 x10^6/uL on day 2.8 before returning to 3.06 x10^6/uL
    at discharge", which is the same sentence with a true verb.

    One substitution per analyte: a series corrected twice in one answer would state the
    movement twice, and the second copy would be noise.
    """
    if not answer or not tool_calls:
        return answer, []
    shapes = repeated_extremes(tool_calls)
    if not shapes:
        return answer, []
    flags: list[str] = []
    remaining = list(shapes)
    out: list[str] = []
    for sentence in _sentences(answer):
        replacement = sentence
        for shape in list(remaining):
            if not _FLAT_CLAUSE.search(replacement):
                break
            label = _label_pattern(shape["label"])
            mentions_label = bool(label and label.search(sentence))
            mentions_value = shape["value"] in sentence
            if not (mentions_label or mentions_value):
                continue
            rebuilt, count = _FLAT_CLAUSE.subn(shape["clause"], replacement, count=1)
            if count:
                replacement = rebuilt
                remaining.remove(shape)
                flags.append(f"trajectory_flattened:{shape['label']}")
        out.append(replacement)
    return "".join(out), flags


def absent_increasing_drivers(
    tool_calls: list[dict] | None, manifest: dict | None
) -> list[dict]:
    """Increasing factors whose recorded value shows the patient does not have them.

    Read from the SERVED payload and the manifest — not from the model's copy — because
    this is the cross-reference the prose is checked against, and a check that read the
    projected copy would agree with itself.
    """
    found: list[dict] = []
    if not manifest:
        return found
    for call in tool_calls or []:
        if not isinstance(call, dict) or call.get("name") != "predict_readmission":
            continue
        response = call.get("response")
        if not isinstance(response, dict):
            continue
        for factor in response.get("top_factors") or []:
            if not isinstance(factor, dict):
                continue
            if factor.get("direction") != "increases":
                continue
            feature = str(factor.get("feature") or "")
            if not feature or not absent_in_manifest(feature, manifest):
                continue
            value = manifest_value(feature, manifest)
            if value is None:
                continue
            found.append({
                "feature": feature,
                "label": humanize(feature),
                "value": value,
            })
    return found


def correct_absent_drivers(
    answer: str, tool_calls: list[dict] | None, manifest: dict | None
) -> tuple[str, list[str]]:
    """Rewrite a sentence that asserts an absent factor is acting on this patient.

    The replacement keeps both facts and drops the inference: the model DID weight the
    factor upward, and the record says the patient does not have it. What it removes is
    the merge of the two, which is the fabrication — a contribution measured against a
    population baseline is not a history this patient has.

    Only sentences that assert (no contrast word, no value) are touched. A sentence that
    already contrasts the two — "the model weights it upward but the record shows zero" —
    is the correct answer and is left alone.
    """
    if not answer:
        return answer, []
    drivers = absent_increasing_drivers(tool_calls, manifest)
    if not drivers:
        return answer, []
    flags: list[str] = []
    pending = list(drivers)
    out: list[str] = []
    for sentence in _sentences(answer):
        replacement = sentence
        lead = sentence[: len(sentence) - len(sentence.lstrip())]
        for driver in list(pending):
            label = _label_pattern(driver["label"])
            if label is None or not label.search(replacement):
                continue
            if not _ASSERTS_DRIVER.search(replacement) or _DISOWNED.search(replacement):
                continue
            value = driver["value"]
            replacement = lead + (
                f"The model weights {driver['label']} as increasing this readmission risk, "
                f"but this patient's record documents {value} for that factor, so it is a "
                f"population-baseline effect rather than a history this patient has."
            )
            pending.remove(driver)
            flags.append(f"absent_driver_corrected:{driver['feature']}")
        out.append(replacement)
    return "".join(out), flags


def citable_units(evidence: list[dict] | None) -> list[dict]:
    """The conversation's citable units in the order the model was shown them.

    `citations.citable_passages` is the single definition of what a unit is and in which
    order it is numbered, so the index here is the same index a marker addresses.
    """
    units: list[dict] = []
    for call in evidence or []:
        if not isinstance(call, dict):
            continue
        for passage in citable_passages(call.get("response")):
            units.append({
                "cite": len(units) + 1,
                "text": str(passage.get("text") or ""),
            })
    return units


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS}


def rebind_citations(
    answer: str, evidence: list[dict] | None
) -> tuple[str, list[str]]:
    """Point a marker at the unit that actually holds the sentence's words.

    The test is one-sided on purpose: a marker is moved only when its own unit shares
    NONE of the sentence's content words and some other unit does. That means a sentence
    already citing a passage with any lexical overlap is never touched — most claims name
    something their passage contains — and a claim whose words are spread across several
    passages keeps the model's choice, because "which of these supports it" is not a
    question a token count answers.

    What it catches is the failure `00d3` recorded three times: a sentence about the
    prednisone taper citing the passage that lists the other drugs, and a sentence about
    the estimate citing the hospital course. In both, the cited text has nothing to do
    with the claim and exactly one other unit plainly does.
    """
    if not answer or not evidence:
        return answer, []
    units = citable_units(evidence)
    if len(units) < 2:
        return answer, []
    unit_tokens = [(unit["cite"], _tokens(unit["text"])) for unit in units]

    flags: list[str] = []
    out: list[str] = []
    for sentence in _sentences(answer):
        markers = list(_CITATION_RE.finditer(sentence))
        if not markers:
            out.append(sentence)
            continue
        words = _tokens(_CITATION_RE.sub(" ", sentence))
        if not words:
            out.append(sentence)
            continue
        overlap = {cite: len(words & tokens) for cite, tokens in unit_tokens}
        rebuilt = sentence
        for marker in markers:
            numbers = [int(n) for n in re.findall(r"\d+", marker.group(1))]
            if len(numbers) != 1:
                continue
            cited = numbers[0]
            if overlap.get(cited, 0) > 0:
                continue
            best = max(overlap.items(), key=lambda pair: (pair[1], -pair[0]))
            if best[1] == 0:
                continue
            rebuilt = rebuilt.replace(f"^[{cited}]", f"^[{best[0]}]")
            flags.append(f"citation_rebound:^{cited}->^{best[0]}")
        out.append(rebuilt)
    return "".join(out), flags


def enforce(
    answer: str,
    evidence: list[dict] | None,
    manifest: dict | None,
    source_total: int,
) -> tuple[str, list[str]]:
    """Every deterministic correction, in the order they must run.

    Rebinding first, because it reads the model's own numbering; then the two prose
    rewrites, which do not care where the markers point; then the range check, so a marker
    left addressing nothing is removed after the rebinding had its chance to fix it.
    """
    flags: list[str] = []
    answer, f = rebind_citations(answer, evidence)
    flags += f
    answer, f = correct_absent_drivers(answer, evidence, manifest)
    flags += f
    answer, f = rewrite_flat_claims(answer, evidence)
    flags += f
    cleaned = sanitize_citation_range(answer, source_total)
    if cleaned != answer:
        flags.append("citation_out_of_range_removed")
    return cleaned, flags
