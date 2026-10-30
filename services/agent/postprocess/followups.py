"""The follow-up chips offered beneath an answer: a deterministic, zero-LLM engine.

Why this is no longer a model call. It used to be, and the questions it produced were
unbounded by construction: three per turn chosen by a model from the answer text alone,
with no guarantee the record could answer any of them and no way to predict what a turn
would offer. A suite that exercises the chips then measures the model's imagination
rather than the product's, and a clinician who clicks one can land on a question the
chart cannot answer. A chip is a control, so what it asks has to be a property of the
record.

How this works instead, in three steps and no network calls:

1. **Signals.** Read the chart facts already in memory for this turn — the tool payloads
   in `state["tool_calls"]` and `state["replayed_tool_calls"]`, plus `command_payload`
   for a medication extraction. Nothing is re-fetched: the audit of 2026-10-23 measured
   that every command leaves enough here for its own templates, and a store lookup on
   this path would put a network call inside a flourish that must never fail a turn.
2. **Phenotypes.** Each template declares the signal it needs, in priority order (P1…P9
   below, plus the literature chip). Nine numbered families, eleven template strings:
   P2 has two wordings and P7/P8 are two numbers over one block. A template whose
   signal is absent does not fire.
3. **Slots.** The slot values — a drug name, a lab analyte and value, a driver label —
   come from the verified payload, never composed. A template whose slot cannot be
   filled from the record is dropped rather than emitted with a guess, because a chip
   naming a drug the patient was not on is the worst thing this module could produce.

Two invariants are contractual rather than stylistic:

* **Turn-2 routability.** Every chip is answered by the free-text loop on the next turn,
  which binds `search_literature` on any literature cue (`routing.asks_for_literature`).
  A chart chip carrying the word "evidence" or "studies" would swap a chart answer for a
  literature list, so every patient-chart chip is vetoed by that same predicate before
  it is returned. No chip needs a cue test to unlock its tool: the chip is generated from
  a tool result that is in the transcript by the time it is clicked, the console declares
  `intent="follow_up"`, and a declared follow-up binds neither `predict_readmission` nor
  `search_literature` (`graph._freetext_allowed`) because the replay already holds what
  they would return.
* **Answerability.** A chip is offered only when the data behind it is in the record.
  Medication chips are suppressed on a record with no discharge medications rather than
  asking about drugs that are not there.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from services.agent.postprocess.citations import literature_calls
from services.agent.routing import routing
from shared.labels import humanize

logger = logging.getLogger(__name__)

#: How many questions a turn offers. Three is the documented contract; the selector
#: stops at the third distinct template rather than ranking all eleven.
FOLLOW_UP_COUNT = 3

#: A follow-up is one line in a row of clickable rows, so the cap is a length rather
#: than a sentence count: past roughly this point a row stops reading as a control and
#: starts reading as a paragraph. A chip over the cap is dropped, never truncated.
_MAX_QUESTION_CHARS = 200

#: Drug classes, brand names included. Promoted from
#: `evaluation/inventory/dev_fact_sheet.py`, where the generic-only version reported
#: anticoagulant absent on a note that says "Coumadin" and "Lovenox" — the false
#: negative this table exists to prevent.
DRUG_CLASSES: dict[str, re.Pattern[str]] = {
    "anticoagulant": re.compile(
        r"warfarin|coumadin|jantoven|apixaban|eliquis|rivaroxaban|xarelto|dabigatran"
        r"|pradaxa|edoxaban|savaysa|enoxaparin|lovenox|dalteparin|fragmin|fondaparinux"
        r"|arixtra|heparin", re.I),
    "beta-blocker": re.compile(
        r"metoprolol|toprol|carvedilol|coreg|bisoprolol|zebeta|atenolol|tenormin"
        r"|labetalol|propranolol|inderal|nebivolol|bystolic", re.I),
    "diuretic": re.compile(
        r"furosemide|lasix|bumetanide|bumex|torsemide|demadex|hydrochlorothiazide"
        r"|\bhctz\b|microzide|chlorthalidone|spironolactone|aldactone|triamterene"
        r"|dyazide|metolazone|zaroxolyn", re.I),
    "taper": re.compile(r"taper|weaning|wean|then decrease|decrease to|reduce to", re.I),
}

#: The drug a taper belongs to. In this cohort's notes the cue is followed by the drug
#: ("then decrease to oxycodone SR 10 mg"), so the capture is what follows the cue, and
#: the extended-release suffixes are part of the name a clinician would write.
_TAPER_TARGET = re.compile(
    r"(?:then\s+decrease\s+to|taper(?:ing)?(?:\s+to)?|wean(?:ing)?(?:\s+to)?"
    r"|reduce\s+to|decrease\s+to)\s+"
    r"([A-Za-z][A-Za-z0-9\-]{2,}(?:\s+(?:SR|ER|XL|CR|XR|HCT))?)",
    re.I)

#: Restrictions, warning signs and return precautions. Presence is the whole test: the
#: chip asks about whatever the record documents, so no slot is filled from it.
_PRECAUTION_CUES = re.compile(
    r"return\s+(?:for|to|if)|repeat\s+\w+|stent|fever|groin|wound|drain|shower"
    r"|lifting|clear\s+liquids|restriction|call\s+(?:in|the|us|if)|warning\s+signs"
    r"|no\s+driving|do\s+not\s+drive|signs\s+of\s+infection",
    re.I)

#: Follow-up arrangements: a provider, a clinic, an appointment or an interval.
_FOLLOW_UP_CUES = re.compile(
    r"follow[\s-]?up|clinic|appointment|urology|primary\s+care"
    r"|\bdr\.?\s+[A-Za-z]|in\s+(?:one|two|three|four|\d+)\s+(?:to\s+\w+\s+)?weeks?",
    re.I)

#: The sections that describe the DISCHARGE plan. A medication or an appointment named
#: anywhere else describes the stay, and a chip that asks "at discharge" about one of
#: those is a question the record cannot answer. Measured on the dev cohort
#: (2026-10-23): `90000105`'s hospital course documents DVT prophylaxis with "subcu
#: heparin", which produced a heparin dosing chip on a record discharged on no
#: medications; `90000106`'s course documents "Dr. X saw him in consultation", which
#: produced an outpatient follow-up chip.
_PLAN_SECTIONS = frozenset({"discharge_medications", "discharge_instructions",
                            "discharge_disposition", "medications_on_transfer"})

#: Anything that makes "monitoring" a documented fact rather than a guess.
_MONITORING_CUES = re.compile(
    r"\binr\b|monday\s+and\s+thursday|sliding\s+scale|monitor|every\s+\w+day",
    re.I)


# --------------------------------------------------------------------------- signals


def _responses(state: dict[str, Any], *names: str) -> list[dict[str, Any]]:
    """Every response a named tool returned this turn, including replayed ones.

    `replayed_tool_calls` is included deliberately: on a follow-up turn the chart calls
    that matter may belong to the turn before, and a chip about "these medications" is
    answerable because that call is in the conversation.
    """
    wanted = set(names)
    out: list[dict[str, Any]] = []
    for key in ("tool_calls", "replayed_tool_calls"):
        for call in state.get(key) or []:
            if not isinstance(call, dict) or call.get("name") not in wanted:
                continue
            response = call.get("response")
            if isinstance(response, dict):
                out.append(response)
    return out


def _passages(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Every passage any tool returned this turn, in call order."""
    out: list[dict[str, Any]] = []
    for key in ("tool_calls", "replayed_tool_calls"):
        for call in state.get(key) or []:
            if not isinstance(call, dict):
                continue
            response = call.get("response")
            if not isinstance(response, dict):
                continue
            for passage in response.get("passages") or []:
                if isinstance(passage, dict):
                    out.append(passage)
    return out


def _chart_text(state: dict[str, Any]) -> str:
    return "\n\n".join(str(p.get("text") or "") for p in _passages(state))


def _plan_text(state: dict[str, Any]) -> str:
    """The passages that describe the discharge plan, and only those.

    Two ways a passage qualifies: its section is one of `_PLAN_SECTIONS`, or a
    `discharge_medications` call returned it — that tool reads the discharge medication
    section whatever the section name on its chunks happens to be.
    """
    plan: list[str] = []
    for passage in _passages(state):
        if str(passage.get("section") or "") in _PLAN_SECTIONS:
            plan.append(str(passage.get("text") or ""))
    for response in _responses(state, "discharge_medications"):
        for passage in response.get("passages") or []:
            if isinstance(passage, dict):
                text = str(passage.get("text") or "")
                if text and text not in plan:
                    plan.append(text)
    return "\n\n".join(plan)


def _sections(state: dict[str, Any]) -> dict[str, str]:
    return {str(p.get("section") or ""): str(p.get("text") or "")
            for p in _passages(state)}


def _medication_rows(state: dict[str, Any]) -> list[str]:
    """The verified drug names an extraction produced, in the order it produced them."""
    payload = state.get("command_payload")
    if not isinstance(payload, dict):
        return []
    names: list[str] = []
    for row in payload.get("medications") or []:
        if isinstance(row, dict):
            name = str(row.get("medication") or "").strip()
            if name and name not in names:
                names.append(name)
    return names


def _medications_absent(state: dict[str, Any]) -> bool:
    """Whether the medication section came back empty on a `/meds` turn.

    Read off `command_payload["returned_passages"]`, which the command sets from the
    tool's own count, so this is the extraction's answer and not a re-parse of prose.
    """
    payload = state.get("command_payload")
    if not isinstance(payload, dict):
        return False
    return payload.get("returned_passages") == 0


def _labs(state: dict[str, Any]) -> list[dict[str, Any]]:
    trends: list[dict[str, Any]] = []
    for response in _responses(state, "get_lab_trends"):
        trends.extend(t for t in response.get("trends") or [] if isinstance(t, dict))
    return trends


def _discharge_point(trend: dict[str, Any]) -> dict[str, Any]:
    series = trend.get("series") or []
    return series[-1] if series else {}


def _excursion(trend: dict[str, Any]) -> float:
    """How far outside its reference range a discharge value sits, in range-widths.

    The ranking rule for "which lab was worst": a plain distance would rank a wide
    reference interval above a narrow one however abnormal the reading is.
    """
    point = _discharge_point(trend)
    value = point.get("value")
    low, high = trend.get("reference_low"), trend.get("reference_high")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0.0
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return 0.0
    width = max(float(high) - float(low), 1e-9)
    if value > high:
        return (float(value) - float(high)) / width
    if value < low:
        return (float(low) - float(value)) / width
    return 0.0


def _formatted_value(trend: dict[str, Any]) -> str:
    """The discharge value at the precision the tool reports, never re-derived."""
    point = _discharge_point(trend)
    value = point.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return ""
    decimals = trend.get("decimals")
    if isinstance(decimals, int) and not isinstance(decimals, bool) and decimals >= 0:
        return f"{float(value):.{decimals}f}"
    return str(value)


def _risk(state: dict[str, Any]) -> dict[str, Any] | None:
    responses = _responses(state, "predict_readmission")
    return responses[-1] if responses else None


def _increasing_driver(risk: dict[str, Any]) -> str:
    """The highest-ranked factor the model says pushes risk up, as a human label.

    Read straight off the served payload: the first `increases` entry, with no
    cross-reference to the patient's own row. The chip offers a question about what the
    model WEIGHED, which is what the estimate says and what the payload holds.
    """
    for factor in risk.get("top_factors") or []:
        if not isinstance(factor, dict):
            continue
        if str(factor.get("direction") or "").lower().startswith("increas"):
            feature = str(factor.get("feature") or "").strip()
            if not feature:
                continue
            return humanize(feature)
    return ""


def _decreasing_driver(risk: dict[str, Any]) -> str:
    """The highest-ranked factor the model says pushes risk DOWN, as a human label.

    The other half of `_increasing_driver`, read the same way.
    """
    for factor in risk.get("top_factors") or []:
        if not isinstance(factor, dict):
            continue
        if str(factor.get("direction") or "").lower().startswith("decreas"):
            feature = str(factor.get("feature") or "").strip()
            if not feature:
                continue
            return humanize(feature)
    return ""


# --------------------------------------------------------------------------- templates

_MED_TAPER = "What is the dosing and taper schedule for {drug} at discharge?"
_MED_ANTICOAG_MONITORED = "What is the discharge dosing and monitoring plan for {drug}?"
_MED_ANTICOAG = "What is the discharge dosing plan for {drug}?"
_CARE_RED_FLAGS = ("What post-discharge activity restrictions, diet instructions, "
                   "or warning signs are documented?")
_CARE_FOLLOW_UP = ("What outpatient follow-up appointments or clinic visits are "
                   "scheduled after discharge?")
_MED_REGIMEN = "What are the discharge dosing instructions for {first} and {second}?"
_RISK_DRIVER = "Why does {driver} increase this patient's readmission risk?"

#: The same question about a PROTECTIVE factor, offered when the payload names no
#: upward driver at all. It asks about the same model output from the direction that is
#: true of this record.
_RISK_PROTECTIVE = "Why does {driver} lower this patient's readmission risk?"

#: The last resort: questions anchored to what the record demonstrably contains, offered
#: only when every template above produced nothing.
#:
#: `/ehr` can end up with no chip at all: its note overview is the command's own answer,
#: so the note template is skipped for it by design, and the record's medication plan may
#: not be among the sections it returned. When nothing is
#: left, `follow_ups=True` returns an empty list and the item is lost as
#: unscored. An affordance row is a promise that something is worth asking next; a record
#: with a hospital course or a medication list always has that.
_COURSE_TREATMENT = ("What does the hospital course record about how this admission was "
                     "treated?")
_MED_SINGLE_DOSING = "What are the discharge dosing instructions for {drug}?"
_LAB_PANEL_OVERVIEW = ("Which laboratory values were abnormal at discharge, and which "
                       "were within their reference ranges?")
_LAB_ABNORMAL = ("How does the abnormal discharge {label} ({value} {unit}) compare to "
                 "its reference range and trend?")
_LAB_NORMAL = ("What were the discharge values for {first} and {second} across the "
               "normal laboratory panels?")
_NOTE_OVERVIEW = ("What were the primary diagnosis and hospital course documented in "
                  "the clinical note?")
_LITERATURE = "What else does the published literature report on {topic}?"

#: The prefixes the lab templates are recognised by; used by the outpatient filter,
#: which keeps lab chips and nothing else on a record with no note text.
_LAB_PREFIXES = ("How does the abnormal discharge ", "What were the discharge values ")


def _filled(template: str, **slots: str) -> str:
    """The template with its slots, or "" when any slot is empty.

    The empty return is the point: an unfilled slot means the record does not carry the
    fact the question would ask about, and a chip asking about a fact the record does
    not hold is the failure this module exists to remove.
    """
    for name, value in slots.items():
        if not str(value or "").strip():
            logger.info("follow-up template dropped: slot %r is empty", name)
            return ""
    return " ".join(template.format(**slots).split())


def _taper_drug(text: str, rows: list[str]) -> str:
    """The drug a taper belongs to, or "".

    Two ways in, both from the record: the cue immediately followed by a drug name, or
    a verified row whose name sits near a taper cue. Nothing is inferred from the drug
    class table alone, because a taper is a fact about this note's wording.
    """
    match = _TAPER_TARGET.search(text)
    if match:
        return match.group(1).strip()
    for name in rows:
        index = text.lower().find(name.lower())
        if index < 0:
            continue
        window = text[max(0, index - 160): index + 160]
        if DRUG_CLASSES["taper"].search(window):
            return name
    return ""


def _first_class_drug(text: str, drug_class: str) -> str:
    match = DRUG_CLASSES[drug_class].search(text)
    return match.group(0) if match else ""


def _second_drug(rows: list[str], text: str) -> tuple[str, str]:
    """Two verified drugs to name in a regimen question, or ("", "").

    Verified rows first — they survived the source-text check — then distinct drug
    classes, which at least proves the note names two drugs.
    """
    if len(rows) >= 2:
        return rows[0], rows[1]
    named: list[str] = []
    for drug_class in ("anticoagulant", "beta-blocker", "diuretic"):
        drug = _first_class_drug(text, drug_class)
        if drug and drug.lower() not in {n.lower() for n in named}:
            named.append(drug)
    if len(named) >= 2:
        return named[0], named[1]
    return "", ""


def _rank_abnormal(trends: list[dict[str, Any]]) -> list[dict[str, Any]]:
    abnormal = [t for t in trends if _excursion(t) > 0]
    return sorted(abnormal, key=_excursion, reverse=True)


# ----------------------------------------------------------------------------- engine


#: How the BFF expands a typed `/pubmed <topic>` into a sentence before it reaches the
#: service. The wording is the caller's, not the clinician's, so it must never appear in a
#: question the clinician is offered back.
_LITERATURE_WRAPPER = re.compile(
    r"^\s*(?:/pubmed|pubmed|search\s+pubmed|search\s+the\s+literature)\b"
    r"(?:\s+for\s+(?:recent\s+|peer-reviewed\s+)?(?:literature|evidence|articles)?)?"
    r"(?:\s+on)?\s*[:,-]?\s*",
    re.IGNORECASE,
)

#: The admission the request is scoped to, which is a contract field rather than part of
#: the subject. `graph._asked_about` strips the same suffix from the question it quotes
#: back to the model; this is the chip's half of that rule.
_ADMISSION_SUFFIX = re.compile(r"\s*\bFor\s+admission\s+\d+\s*\.?\s*$", re.IGNORECASE)


def _literature_topic(state: dict[str, Any], question: str) -> str:
    """What the literature search was ABOUT, as the clinician would say it.

    The `search_literature` call is the authority when it is in the turn: its `query`
    argument is what was actually searched, before any of this layer's own bookkeeping.
    That is the whole point — a chip is a question the reader can click, and the tool
    already holds the words for it.

    The question is the fallback, and it arrives WRAPPED. The console expands a typed
    `/pubmed hyperglycemia` into "Search PubMed for recent literature on hyperglycemia.
    For admission 90000067.", so slotting the question into the chip produced "What else
    does the published literature report on Search PubMed for recent literature on
    hyperglycemia. For admission 90000067?" (reported from the live turn, 2026-10-26).
    Both wrappers are stripped: the command's own sentence, and the admission the request
    is scoped to.

    A declared follow-up holds no search of its own — it answers from the one already in
    its transcript — so the call is read through `citations.literature_calls`, which looks
    at this turn and then at the replay. Without that, the topic came from the whole
    question and the chip repeated it verbatim: "What else does the published literature
    report on What else does the published literature report on bladder cancer?" (found
    on 2026-10-30 while repairing the same turn).

    The LAST search's query, not the first: a model that refines its query three times
    searched three different things, and the question worth asking next follows from the
    records the answer rests on (2026-10-30, again — the first and last queries of that
    turn described different searches).
    """
    for call in reversed(literature_calls(
            state.get("tool_calls"), state.get("replayed_tool_calls"))):
        query = str((call.get("args") or {}).get("query") or "").strip()
        if query:
            return query.strip(" ?.")[:120]
    cleaned = _ADMISSION_SUFFIX.sub("", str(question or "")).strip()
    cleaned = _LITERATURE_WRAPPER.sub("", cleaned).strip()
    # One wrapper can sit inside another ("Search PubMed for recent literature on ...");
    # strip until nothing changes rather than guessing at the nesting.
    while True:
        again = _LITERATURE_WRAPPER.sub("", cleaned).strip()
        if again == cleaned:
            break
        cleaned = again
    return cleaned.strip(" ?.")[:120]


def _candidates(*, state: dict[str, Any], scope: str | None, question: str,
                outpatient: bool, literature: bool = False) -> list[str]:
    """Every chip this turn can offer, in priority order, before the cap is applied."""
    text = _chart_text(state)
    plan = _plan_text(state)
    rows = _medication_rows(state)
    sections = _sections(state)
    meds_absent = _medications_absent(state)
    labs = _labs(state)
    risk = _risk(state)
    chips: list[str] = []

    if literature or scope == "literature":
        # A literature turn has no chart signals by construction; the only honest next
        # question is about the published evidence itself, and only when the search
        # actually returned something. `None` means "no articles", which is a turn with
        # nothing to follow up on — never a reason to fall back to the chart.
        topic = _literature_topic(state, question)
        chip = _filled(_LITERATURE, topic=topic)
        return [chip] if chip else []

    medications_available = bool(plan) and not meds_absent

    # P1 — a taper is the most specific medication fact a note carries.
    taper_present = bool(medications_available
                         and DRUG_CLASSES["taper"].search(plan))
    if taper_present:
        chip = _filled(_MED_TAPER, drug=_taper_drug(plan, rows))
        if chip:
            chips.append(chip)

    # P2 — anticoagulation, with the monitoring half only where it is documented.
    anticoag = _first_class_drug(plan, "anticoagulant") if medications_available else ""
    if anticoag:
        template = (_MED_ANTICOAG_MONITORED if _MONITORING_CUES.search(plan)
                    else _MED_ANTICOAG)
        chip = _filled(template, drug=anticoag)
        if chip:
            chips.append(chip)

    # P3 — what to watch for, and what the patient may not do. From the plan, like the
    # rules above: "stent" appears in `90000108`'s history as a stent PLACED during the
    # admission, and a chip asking about post-discharge restrictions on the strength of
    # that would ask about instructions the record does not contain.
    if plan and _PRECAUTION_CUES.search(plan):
        chips.append(_CARE_RED_FLAGS)

    # P4 — who is seeing this patient next. From the plan only: a consultant who saw the
    # patient during the stay is not an outpatient appointment.
    if plan and _FOLLOW_UP_CUES.search(plan):
        chips.append(_CARE_FOLLOW_UP)

    # P5 — dosing for two named drugs, unless the taper and anticoagulant chips already
    # ask it; a third question repeating the first two would waste the row.
    if medications_available and not (anticoag and taper_present):
        first, second = _second_drug(rows, plan)
        if first and second:
            chip = _filled(_MED_REGIMEN, first=first, second=second)
            if chip:
                chips.append(chip)

    # P6 — why the model's own ranking puts this patient where it does.
    if risk:
        chip = _filled(_RISK_DRIVER, driver=_increasing_driver(risk))
        if chip:
            chips.append(chip)
        else:
            # No upward driver in the payload. The protective directions are still true of
            # the record, so the same surface asks about one of those rather than falling
            # silent.
            chip = _filled(_RISK_PROTECTIVE, driver=_decreasing_driver(risk))
            if chip:
                chips.append(chip)

    # P7 / P8 — the abnormal discharge value, or the normal panel it belongs to.
    abnormal = _rank_abnormal(labs)
    if abnormal:
        worst = abnormal[0]
        chip = _filled(_LAB_ABNORMAL, label=str(worst.get("label") or ""),
                       value=_formatted_value(worst),
                       unit=str(worst.get("unit") or "").strip())
        if chip:
            chips.append(chip)
    elif len(labs) >= 2:
        labels = [str(t.get("label") or "") for t in labs if t.get("label")]
        if len(labels) >= 2:
            chip = _filled(_LAB_NORMAL, first=labels[0], second=labels[1])
            if chip:
                chips.append(chip)

    # P9 — the note, but only when the turn was not already the note overview.
    if state.get("command") != "summarize" and any(
            name in sections for name in
            ("discharge_diagnosis", "brief_hospital_course",
             "history_of_present_illness")):
        chips.append(_NOTE_OVERVIEW)

    # An outpatient record whose note came back empty has no chart to drill into: the
    # notice already said so, and a medication chip would be the fabrication it warns
    # against. Lab chips survive because those results really are there.
    if outpatient and not text:
        chips = [chip for chip in chips if chip.startswith(_LAB_PREFIXES)]
    elif not chips:
        # Last resort, and the reason the row is never empty on a record that has
        # something to ask about: every template above is conditional on a specific
        # signal, and `/ehr` skips the note one because the note was its own answer.
        chips.extend(_record_anchors(state, text=text, rows=rows, labs=labs))
    return chips


def _record_anchors(state: dict[str, Any], *, text: str,
                    rows: Sequence[dict[str, Any]],
                    labs: Sequence[dict[str, Any]]) -> list[str]:
    """Questions the record can answer, for a turn that produced no specific chip.

    Deliberately few, and each tied to evidence this turn already retrieved: a course
    question only where a hospital course or diagnosis is in the transcript, a dosing
    question only for a drug the note names, a panel question only where a panel exists.
    An anchor offered without its evidence is an affordance the record cannot honour, one
    layer down from a chip whose question has no answer.
    """
    anchors: list[str] = []
    sections = _sections(state)
    if any(name in sections for name in ("brief_hospital_course", "discharge_diagnosis",
                                         "history_of_present_illness")):
        anchors.append(_COURSE_TREATMENT)
    if rows:
        drug = str(rows[0] or "").strip()
        if drug:
            chip = _filled(_MED_SINGLE_DOSING, drug=drug)
            if chip:
                anchors.append(chip)
    if labs:
        anchors.append(_LAB_PANEL_OVERVIEW)
    return anchors


def _routable(chip: str, scope: str | None, literature: bool = False) -> bool:
    """Whether the free-text loop can answer this chip on the next turn.

    The literature predicate is the load-bearing one: it is the same test the loop uses
    to decide whether to bind `search_literature`, so a chart chip that trips it would
    be answered from PubMed instead of the record. A literature turn's own chip is exempt
    — that is the surface it came from.
    """
    if literature or scope == "literature":
        return True
    return not routing.asks_for_literature(chip)


def _fingerprint(text: str) -> str:
    """A question reduced to what it asks, for comparison with what was asked.

    Case, punctuation and whitespace carry no meaning here: a chip the user clicked and
    the question the client replays are the same question even when one has a trailing
    question mark and the other does not.
    """
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).split())


def _view(state: dict[str, Any], *, replayed: bool) -> dict[str, Any]:
    """This turn's state, optionally with the earlier turns' tool calls.

    The two are read separately because they answer different questions. This turn's
    calls say what the answer just established; the replayed ones say what the
    conversation already covered, and a chip is only worth offering about the latter
    once the former is exhausted.
    """
    return {
        "tool_calls": state.get("tool_calls") or [],
        "replayed_tool_calls": (state.get("replayed_tool_calls") or []) if replayed else [],
        "command": state.get("command"),
        "command_payload": state.get("command_payload"),
    }


def _literature_articles(state: dict[str, Any]) -> int:
    """How many articles this turn's own literature search returned."""
    total = 0
    for response in _responses(state, "search_literature"):
        articles = response.get("articles")
        if isinstance(articles, list):
            total += len(articles)
    return total


def _fit(chips: list[str], *, asked: set[str], scope: str | None,
         taken: list[str], literature: bool = False) -> list[str]:
    """The chips that can be offered, on top of the ones already taken.

    Four filters, in the order that makes a rejection explainable: an identical chip
    once, a question the thread has already asked never, the row cap, and the
    literature veto. `asked` is fingerprinted so a click and its replayed text match.
    """
    kept = list(taken)
    seen = {_fingerprint(chip) for chip in taken}
    for chip in _distinct(chips):
        if len(kept) >= FOLLOW_UP_COUNT:
            break
        fingerprint = _fingerprint(chip)
        if fingerprint in seen:
            continue
        if fingerprint in asked:
            logger.info("follow-up chip dropped: already asked in this thread")
            continue
        if len(chip) > _MAX_QUESTION_CHARS:
            logger.info("follow-up chip dropped: %s characters", len(chip))
            continue
        if not _routable(chip, scope, literature):
            logger.info("follow-up chip dropped: it would route to the literature")
            continue
        seen.add(fingerprint)
        kept.append(chip)
    return kept


def _distinct(chips: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for chip in chips:
        key = chip.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(chip)
    return out


def suggest(
    question: str,
    answer: str,
    *,
    model: str | None = None,
    scope: str | None = None,
    state: dict[str, Any] | None = None,
    outpatient: bool = False,
    asked: Sequence[str] = (),
    literature: bool = False,
) -> list[str]:
    """Up to `FOLLOW_UP_COUNT` chips for this turn, chosen from the record.

    Synchronous and deterministic: no model, no I/O, no clock. `model` is accepted and
    unused so the call site keeps the signature it had when this was a model call, and
    so a later reader can see the removal was deliberate rather than an oversight.

    An empty answer yields no chips. That is what keeps `/risk` — which answers in a
    widget and carries no prose — from offering follow-ups on a turn whose answer text
    does not exist.

    `asked` is every question already put to the service in this thread, this turn's
    included. A chip that matches one of them is not offered: clicking the taper chip
    used to re-render the taper chip, so the row repeated itself instead of advancing
    (measured in the production UI, `agent-00173-9pv`, 2026-10-23).

    A literature turn is answered from this turn's own search or not at all. It is never
    filled from the chart: a PubMed turn that found nothing is a turn with no follow-up,
    and reusing the replayed `/ehr` calls made it offer the chart chips it had just
    replaced.

    `literature` is that turn, declared by the caller from the search the turn actually
    ran (Phase F.1, 2026-10-26). It used to be inferred from `scope == "literature"`,
    which nothing ever passed: the payload's scope values are `patient_chart` and
    `general_evidence`, so the literature branch never ran, and a `/pubmed` turn whose
    tool calls included the readmission estimate offered the chart's own risk-driver chip
    under an answer about published evidence.
    """
    if not str(answer or "").strip():
        return []

    state = state or {}
    asks = {_fingerprint(item) for item in (*asked, question) if str(item or "").strip()}
    on_literature = literature or scope == "literature"

    if on_literature:
        if not _literature_articles(state):
            # Nothing was found, so there is nothing to ask next about the evidence.
            return []
        # The replayed calls are in this view only so the chip can name the topic the
        # search was actually about. Nothing else reads them: the literature branch
        # returns before any chart signal is consulted, so a replayed `/ehr` call cannot
        # put a chart chip under an answer about published evidence.
        return _fit(_candidates(state=_view(state, replayed=True), scope=scope,
                                question=question, outpatient=outpatient,
                                literature=on_literature),
                    asked=asks, scope=scope, taken=[], literature=on_literature)

    # This turn first: what the answer just established is the most useful thing to ask
    # about next. The earlier turns are the fallback, not the starting point.
    chips = _fit(_candidates(state=_view(state, replayed=False), scope=scope,
                             question=question, outpatient=outpatient),
                 asked=asks, scope=scope, taken=[])
    if len(chips) < FOLLOW_UP_COUNT:
        chips = _fit(_candidates(state=_view(state, replayed=True), scope=scope,
                                 question=question, outpatient=outpatient),
                     asked=asks, scope=scope, taken=chips)
    return chips
