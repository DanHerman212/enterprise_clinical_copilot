"""Seeded synthetic lab trajectories for the drill-down card (Phase 3, 2026-10-16).

A clinician clicking a lab-based risk driver asks a question the chart should answer: what
did this measurement actually do during the admission. Serial values are needed to answer
it, and there are three ways to get them, only one of which is available.

Reading MIMIC `labevents` is not available. The demo cohort is served under a DUA that
permits no redistribution of raw laboratory events, so a trajectory assembled from the
source table is not a shortcut that was declined — it is one that cannot be taken.

Static fixtures were available and are rejected. Hand-written values do not survive a
cohort: they cannot answer for a patient nobody wrote a fixture for, and they drift the
moment a feature row is regenerated. Worse, they are inventions dressed as records — the
same numbers every run, with nothing tying them to the patient they claim to describe.

So the values are generated, and the generator is built around one constraint that makes it
honest rather than decorative:

    A trajectory reproduces, EXACTLY, every value the model was shown for that analyte.

Click the "lowest red blood cell count" driver and the card's minimum is that driver's
number, not a number near it. This is the coherence rule the cohort curation pass exists to
enforce (`docs/synthetic_cohort_curation.md`, item 4: "narrative labs and LOS must match the
feature row") applied to the one surface that could not honour it before. The guarantee is
structural rather than approximate: the walk's window is built FROM the anchors, so the
extreme it must hit is the bound it draws inside. There is no fitting step that could land
close.

Determinism is `hash(hadm_id, analyte)`, as the roadmap specifies, and it is a hash rather
than a seeded `random` for a reason: the same admission must produce the same trajectory in
another process, on another machine, on another day, with no stored seed. `hashlib.sha256`
gives that. `random` seeded from anything ambient, or `hash()` over a string — which is
salted per process — do not.

What the generator is NOT: a model input. The eight analytes the model scored are anchored;
the clinical panel the roadmap also asks for — creatinine, eGFR, potassium, WBC — is drawn
from textbook intervals because the booster was never trained on them. Those rows are
clinical colour, they are honest synthetic values, and the payload names which analytes are
anchored so a surface cannot present a drawing as a measurement.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

#: Timepoints per trajectory. Four is the roadmap's floor and six its ceiling; a stay between
#: them gets one per day, which is roughly how often a basic metabolic panel is drawn.
MIN_POINTS = 4
MAX_POINTS = 6

#: How much of the textbook interval a stable admission's values use. A surgical admission
#: is allowed wider, which is where a post-operative value lives.
_STABLE_SPREAD = 0.60
_SURGICAL_SPREAD = 0.85


@dataclass(frozen=True)
class Analyte:
    """One measure, the window it is read against, and what it must reproduce.

    `low`/`high` are textbook reference intervals for an adult and they decide the FLAG —
    not the range a value is drawn in. A patient's own `hemoglobin_min` may sit below the
    reference interval and is reproduced anyway: the number the model scored outranks the
    population window, and a card that quietly lifted a value into the normal range to look
    tidier would be the exact failure this module exists to prevent.

    `decimals` is the reporting precision, and it is what keeps an anchor exact. An anchor is
    reproduced at THIS precision — see `_reproduced` — because the promise is exactness at
    the precision a reader can check, not at the payload's.
    """

    key: str
    label: str
    unit: str
    low: float
    high: float
    decimals: int
    #: Feature-row keys this analyte must reproduce. `_min` and `_max` are promises about
    #: the series' extremes; `_last` is a promise about its final value.
    anchors: tuple[str, ...] = field(default_factory=tuple)
    critical_low: float | None = None
    critical_high: float | None = None
    #: Where the series' lowest point falls, as a fraction of the stay. A post-operative
    #: haemoglobin is at its lowest on day one or two, not at admission.
    nadir: float = 0.25
    #: The course the series takes. See `_COURSE`.
    shape: str = "bounded"
    #: Drawn only when the phenotype says so. See `_phenotype`.
    conditional: str | None = None


#: The panel, model-anchored analytes first: they are the ones a reader arrives here from,
#: having clicked a driver bar.
PANEL: tuple[Analyte, ...] = (
    # --- scored by the model, therefore anchored on the feature row ---
    Analyte("rdw", "RDW", "%", 11.5, 14.5, 1, anchors=("rdw_max",), shape="rise"),
    Analyte("rbc", "Red blood cells", "x10^6/uL", 4.20, 5.40, 2,
            anchors=("rbc_min", "rbc_last"), nadir=0.25, shape="dip"),
    Analyte("hemoglobin", "Hemoglobin", "g/dL", 12.0, 15.5, 1,
            anchors=("hemoglobin_min",), critical_low=7.0, nadir=0.25, shape="dip"),
    Analyte("sodium", "Sodium", "mmol/L", 135.0, 145.0, 0,
            anchors=("sodium_min", "sodium_max", "sodium_last"), nadir=0.4,
            critical_low=120.0, critical_high=160.0, shape="bounded"),
    Analyte("monocytes", "Monocytes", "x10^3/uL", 0.20, 0.80, 1,
            anchors=("monocytes_min",), nadir=0.4, shape="dip"),
    # --- the clinical panel: never model inputs, so nothing to anchor to ---
    Analyte("creatinine", "Creatinine", "mg/dL", 0.50, 1.10, 2, critical_high=4.0,
            nadir=0.75, shape="rise"),
    Analyte("egfr", "eGFR", "mL/min/1.73m2", 90.0, 120.0, 0, nadir=0.85,
            shape="fall"),
    Analyte("potassium", "Potassium", "mmol/L", 3.5, 5.0, 1,
            critical_low=2.5, critical_high=6.5, nadir=0.5, shape="bounded"),
    Analyte("wbc", "White blood cells", "x10^3/uL", 4.0, 11.0, 1, nadir=0.2,
            shape="postop_rise"),
    Analyte("nt_probnp", "NT-proBNP", "pg/mL", 0.0, 300.0, 0, critical_high=1000.0,
            nadir=0.85, shape="fall", conditional="cardiac"),
)

_CARDIAC_MARKERS = ("heart failure", "cardiac", "coronary", "myocardial", "chf",
                    "atrial", "valve", "angina", "cardiomyopathy", "pacemaker")


def seed_for(hadm_id: int, analyte: str) -> int:
    """The deterministic seed for one admission's one analyte."""
    digest = hashlib.sha256(f"{int(hadm_id)}:{analyte}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _draw(seed: int, index: int) -> float:
    """The `index`-th draw in [0, 1) from one seed.

    Counter-mode, not a stream: a value can be drawn out of order without the draws before
    it existing, which is what lets a shape place its extremes before filling the middle.
    """
    state = (seed + (index + 1) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    state = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    state = ((state ^ (state >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    state ^= state >> 31
    return state / (1 << 64)


def _window(analyte: Analyte, present: dict[str, float],
            phenotype: dict[str, Any]) -> tuple[float, float]:
    """The range the trajectory is drawn in, built FROM the anchors.

    This is the whole mechanism of the guarantee. A `_min` anchor becomes the lower bound
    and a `_max` anchor the upper one, so every draw is already inside them and the extremes
    cannot miss. No fitting, no correction pass, nothing that can land near a number instead
    of on it.

    A ONE-SIDED anchor takes the textbook interval's far bound as its other end rather than
    a fraction of the anchor itself. Deriving both ends from one value is what let a lone
    `rdw_max` of 13.5 produce a window of [13.5, 14.2] — a series drawn entirely ABOVE the
    maximum it was supposed to respect. The far bound is also the clinically sane one: a
    patient whose haemoglobin bottoms at 12.1 may start at 14, and the reference interval
    is what says 14 is a haemoglobin rather than a number.

    With no extreme anchor at all — a `_last` alone, or the clinical panel — the window is
    the textbook interval narrowed by the admission's shape.
    """
    lows = [v for k, v in present.items() if k.endswith("_min")]
    highs = [v for k, v in present.items() if k.endswith("_max")]
    if lows and highs:
        lo, hi = min(lows), max(highs)
        if hi <= lo:
            hi = lo + max(abs(lo) * 0.05, 0.1)
        return lo, hi
    if highs:
        return min(analyte.low, min(highs)), max(highs)
    if lows:
        return min(lows), max(analyte.high, max(lows))

    spread = _SURGICAL_SPREAD if phenotype["surgical"] else _STABLE_SPREAD
    margin = (analyte.high - analyte.low) * (1 - spread) / 2
    return analyte.low + margin, analyte.high - margin


#: Each shape as the three values its course passes through, in normalized units where 0 is
#: the window's floor and 1 its ceiling: where it starts, where it turns, where it ends.
#:
#: Three numbers on straight lines rather than a smooth formula, because the course a
#: clinician reads is a start, a turning point and a recovery — and a curve defined by
#: exactly those is one this module can state, and a test can check.
_COURSE: dict[str, tuple[float, float, float]] = {
    "rise": (0.05, 0.00, 1.00),
    "fall": (1.00, 0.00, 0.10),
    "dip": (0.60, 0.00, 0.40),
    "bounded": (0.65, 0.10, 0.45),
    "postop_rise": (0.30, 0.85, 0.40),
}


def _curve(points: int, shape: str, nadir: float, seed: int) -> list[float]:
    """The shape of the stay, normalized, before any anchor is written into it."""
    start, turn, end = _COURSE.get(shape, _COURSE["bounded"])
    if shape in ("rise", "fall"):
        # Monotone, so the turning point is an endpoint rather than a knee in the middle.
        step = (end - start) / (points - 1)
        course = [start + step * index for index in range(points)]
    else:
        knee = max(1, min(points - 2, int(round(nadir * (points - 1)))))
        course = []
        for index in range(points):
            if index <= knee:
                course.append(start + (turn - start) * index / knee)
            else:
                course.append(turn + (end - turn) * (index - knee) / (points - 1 - knee))
    # A course is not a formula: a little jitter, so two patients with one shape do not
    # produce one series. Clamped, because the window is a promise about every value.
    return [
        min(1.0, max(0.0, value + (_draw(seed, index) - 0.5) * 0.16))
        for index, value in enumerate(course)
    ]


def _extreme_index(curve: list[float], *, highest: bool, avoid: set[int]) -> int:
    """Where a series' extreme sits, honouring the indices already spoken for.

    The collision this exists for: a `_max` anchor and a `_last` anchor both want the final
    timepoint, and writing them in order let the last value quietly erase the peak — a
    sodium whose maximum was 141 appearing to top out at 140.
    """
    order = sorted(range(len(curve)), key=lambda index: curve[index], reverse=highest)
    for index in order:
        if index not in avoid:
            return index
    return order[0]


def _stays(hadm_id: int) -> dict[str, Any]:
    """The feature row this admission was SCORED on.

    The same source the predictor reads, which is what makes the anchor guarantee real
    rather than aspirational: the card cannot disagree with the driver bar, because one row
    answers both.
    """
    from services.mcp.dependencies.features import get_feature_source

    return get_feature_source().fetch(hadm_id)


def _phenotype(features: dict[str, Any], note_text: str) -> dict[str, Any]:
    """What the admission's own shape says about a trajectory.

    Signals that the patient already carries rather than anything this module invents: how
    long the stay ran, whether it was surgical, and — for the one conditional row — whether
    the note reads as cardiac. `note_text` is passed in by a caller that has already
    retrieved it, so no evaluation artifact is read at serving time and an absent note
    simply means the conditional row is not drawn, which is the honest direction to fail in.
    """
    try:
        los = float(features.get("index_los_days"))
    except (TypeError, ValueError):
        los = 4.0
    los = min(max(los, 1.0), 30.0)
    return {
        "los": los,
        "points": int(min(MAX_POINTS, max(MIN_POINTS, round(los)))),
        "surgical": bool(features.get("has_procedure")),
        "oncology": bool(features.get("oncology_flag")),
        "cardiac": any(marker in (note_text or "").lower()
                       for marker in _CARDIAC_MARKERS),
    }


def _reproduced(analyte: Analyte, value: float) -> float:
    """An anchor at the precision the analyte is reported at.

    Sodium is reported whole; a row carrying 137.4 is reproduced as 137, because 137 is the
    number a reader can check against the panel.
    """
    return round(float(value), analyte.decimals)


def _flag(analyte: Analyte, value: float) -> str:
    """How a value reads against its interval, or against a critical bound."""
    if analyte.critical_low is not None and value < analyte.critical_low:
        return "critical_low"
    if analyte.critical_high is not None and value > analyte.critical_high:
        return "critical_high"
    if value < analyte.low:
        return "low"
    if value > analyte.high:
        return "high"
    return "normal"


def _series(analyte: Analyte, present: dict[str, float], phenotype: dict[str, Any],
            hadm_id: int) -> list[float]:
    """The values, in order, with every anchor reproduced exactly."""
    points = phenotype["points"]
    lo, hi = _window(analyte, present, phenotype)
    seed = seed_for(hadm_id, analyte.key)
    curve = _curve(points, analyte.shape, analyte.nadir, seed)
    values = [lo + (hi - lo) * position for position in curve]

    # The extremes are written at the positions the CURVE takes them, so the shape decides
    # where the low point and the peak fall and the hash decides nothing about physiology.
    # `_last` is spoken for before either, because it is a promise about the end of the
    # series and the extremes are promises about its shape.
    avoid: set[int] = set()
    if any(key.endswith("_last") for key in present):
        avoid.add(points - 1)
    highs = [v for k, v in present.items() if k.endswith("_max")]
    lows = [v for k, v in present.items() if k.endswith("_min")]
    if highs:
        peak = _extreme_index(curve, highest=True, avoid=avoid)
        values[peak] = max(highs)
        avoid.add(peak)
    if lows:
        values[_extreme_index(curve, highest=False, avoid=avoid)] = min(lows)
    for key, value in present.items():
        if key.endswith("_last"):
            values[-1] = value

    return [_reproduced(analyte, value) for value in values]


def trajectory(analyte: Analyte, features: dict[str, Any], phenotype: dict[str, Any],
               hadm_id: int) -> dict[str, Any] | None:
    """One analyte's serial values, or None when this admission does not call for it."""
    if analyte.conditional and not phenotype[analyte.conditional]:
        return None

    present = {
        key: float(features[key]) for key in analyte.anchors
        if isinstance(features.get(key), (int, float))
        and not isinstance(features.get(key), bool)
    }
    values = _series(analyte, present, phenotype, hadm_id)

    steps = max(len(values) - 1, 1)
    series = [
        {
            "day": round(phenotype["los"] * index / steps, 1),
            "value": value,
            "flag": _flag(analyte, value),
        }
        for index, value in enumerate(values)
    ]
    return {
        "key": analyte.key,
        "label": analyte.label,
        "unit": analyte.unit,
        "reference_low": analyte.low,
        "reference_high": analyte.high,
        "decimals": analyte.decimals,
        # What the model actually scored. Named in the payload so a surface can distinguish
        # the patient's numbers from this module's, rather than presenting a drawing as a
        # measurement.
        "anchored": sorted(present),
        "series": series,
    }


def lab_trends(hadm_id: int, analytes: list[str] | None = None,
               features: dict[str, Any] | None = None,
               note_text: str = "") -> list[dict[str, Any]]:
    """The lab trajectories for one admission.

    `features` and `note_text` are injectable for two reasons: a caller that already holds
    the row the predictor used does not read it twice, and the generator can be tested
    against a fixture row with no BigQuery and no server.
    """
    row = features if features is not None else _stays(hadm_id)
    phenotype = _phenotype(row, note_text)
    wanted = set(analytes) if analytes else None
    out = []
    for analyte in PANEL:
        if wanted is not None and analyte.key not in wanted:
            continue
        drawn = trajectory(analyte, row, phenotype, hadm_id)
        if drawn is not None:
            out.append(drawn)
    return out
