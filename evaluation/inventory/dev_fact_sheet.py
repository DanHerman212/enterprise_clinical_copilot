"""Phase 04, Section 2 — the Dev cohort fact sheet.

Read-only, zero LLM. Everything in `evaluation/dev_cohort_fact_sheet.md` is either read
from `evaluation/cohort_provenance_inventory.json` or fetched again from the same tools
that produced it, plus the model's own feature row for the driver values the serving
tools do not expose.

    .venv/bin/python evaluation/inventory/dev_fact_sheet.py

Why the driver values need the row: `get_admission_profile` decodes a fixed set of
administrative columns, so `rdw_max`, `sodium_min`, `monocytes_min` and `oncology_flag`
have no value in any tool payload — the risk response names them and nothing more. They
are in the row the model was scored on, so they are read from there rather than guessed
from a lab table with a different definition (a discharge value is not a maximum).

Nothing here decides what is clinically interesting. Text is QUOTED, never paraphrased:
a fact sheet that summarised a note would be the first place a fabricated clinical claim
could enter the suite.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from google.cloud import bigquery  # noqa: E402

from services.mcp.config import ENTITY_ID_COLUMN, PROJECT, TABLE  # noqa: E402
from services.mcp.retrieval.sections import KNOWN_HEADINGS  # noqa: E402
from services.mcp.tools import (  # noqa: E402
    discharge_medications,
    get_lab_trends,
    predict_readmission,
    rag_search_sections,
)
from services.agent.engine.mcp_client import _presented_for_model  # noqa: E402
from shared.labels import driver_question, humanize  # noqa: E402

from evaluation.platform import dev_suite  # noqa: E402

INVENTORY = REPO / "evaluation" / "cohort_provenance_inventory.json"
OUT = REPO / "evaluation" / "dev_cohort_fact_sheet.md"

#: The dev cohort, read from `dev_suite` rather than copied here.
#:
#: It was a literal tuple of seven until 2026-11-22, and the sheet is what answer-key facts
#: are copied from character-for-character — a second list that could drift from the one
#: the suite runs against is the worst place in the repository to keep a copy.
DEV = tuple(int(hadm) for hadm in dev_suite.DEV_HADM_IDS)

#: Drug classes the suite's shorthand and false-premise trajectories will ask about, so
#: the fact sheet states presence or absence rather than leaving it to be inferred.
#:
#: Brand names matter as much as generic ones: this cohort's note says "Coumadin" and
#: "Lovenox", not warfarin and enoxaparin. The first version of this table listed only
#: generics and reported anticoagulant as ABSENT on 90000015 — a false negative on the
#: exact fact a false-premise trap would be built from.
#:
#: The taper pattern carries dose-reduction wording as well as the word "taper", because
#: a record can document a taper without ever using it: one of them says only that a
#: dose "was decreased". It deliberately does not match a bare "decreased" — that same
#: record reports decreasing urine output, which is not a medication schedule.
DRUG_CLASSES = {
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
    "taper": re.compile(
        r"taper|wean|then decrease|decrease to|reduce to|titrat"
        r"|dose was (?:decreased|reduced)|dose of [a-z]+ (?:was )?(?:decreased|reduced)",
        re.I),
}


def _evidence(pattern: re.Pattern[str], text: str) -> str:
    """The phrase the scan matched, so a reader can check the claim rather than trust it."""
    hit = pattern.search(text or "")
    return f" (`{hit.group(0)}`)" if hit else ""


def _disagreements(flagged: dict, text: str) -> list[str]:
    """Where the selection flag and the returned text say different things.

    One record was chosen for a taper on the strength of the inventory flag while none of
    the medication passages this tool returns names one; another's taper is a dose
    reduction stated in prose. Both are facts the next directive needs before it writes a
    taper item, and neither is this script's to settle.
    """
    out: list[str] = []
    taper = DRUG_CLASSES["taper"]
    matched = bool(taper.search(text or ""))
    if flagged.get("mentions_taper") and not matched:
        out.append("**Taper: the inventory records one; the passages above do not name "
                   "one.** Treat a taper item on this record as unverified until the "
                   "phrase is located.")
    if not flagged.get("mentions_taper") and matched:
        out.append("**Taper: the passages above name a dose reduction the inventory did "
                   "not flag.**")
    return out


def _rows(hadm_ids: tuple[int, ...]) -> dict[int, dict[str, object]]:
    """The exact feature row the model scored, read straight from the table."""
    client = bigquery.Client(project=PROJECT)
    query = f"SELECT * FROM {TABLE} WHERE {ENTITY_ID_COLUMN} IN UNNEST(@ids)"
    job = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("ids", "INT64", list(hadm_ids))])
    return {int(row[ENTITY_ID_COLUMN]): dict(row.items())
            for row in client.query(query, job_config=job).result()}


def _discharge_point(trend: dict) -> dict:
    series = trend.get("series") or []
    return series[-1] if series else {}


def _excursion(trend: dict) -> float:
    """How far outside its reference range the discharge value sits, in range-widths."""
    point = _discharge_point(trend)
    value = point.get("value")
    low, high = trend.get("reference_low"), trend.get("reference_high")
    if not isinstance(value, (int, float)) or not isinstance(low, (int, float)) \
            or not isinstance(high, (int, float)):
        return 0.0
    width = max(high - low, 1e-9)
    if value > high:
        return (value - high) / width
    if value < low:
        return (low - value) / width
    return 0.0


def _first_statement(text: str) -> str:
    """The first sentence that says something, with a list marker stripped.

    These notes open a section with a numbered list — `1.` on its own line — so taking
    the first sentence literally quoted `"1."` and told a reader nothing. A sentence is
    skipped while it is too short to carry a fact, and leading numbering is removed
    first because it is markup, not diagnosis.
    """
    body = re.sub(r"^\s*\d+[.)]\s*", "", (text or "").strip())
    for candidate in re.split(r"(?<=[.!?])\s+", body):
        cleaned = candidate.strip()
        if len(cleaned) >= 20:
            return cleaned
    return body.strip()[:300]


def _bullets(sections: dict[str, str]) -> list[str]:
    """Quoted sentence-openings from the note, never a paraphrase."""
    out: list[str] = []
    for key in ("discharge_diagnosis", "brief_hospital_course",
                "history_of_present_illness"):
        text = (sections.get(key) or "").strip()
        if not text:
            continue
        out.append(f"**{key}** (quoted): \"{_first_statement(text)[:300]}\"")
        if len(out) == 3:
            break
    return out


def _section_map(payload: dict) -> dict[str, str]:
    return {str(p.get("section")): str(p.get("text") or "")
            for p in payload.get("passages") or []}


#: The manifest fields a prompt or an answer key may quote, with the string the product
#: publishes for each. `admission_type` arrives as the warehouse code (`Emergency
#: (EW EMER.)`) and is published as the bare word; a length of stay is published as a
#: whole number of days. Both rules live in the presentation layer, and the columns here
#: are what a dataset row copies rather than re-deriving.
def _plain_code(raw: Any) -> str:
    return str(raw or "").split(" (")[0].strip() or "—"


def _whole_days(raw: Any) -> str:
    try:
        return str(int(round(float(raw))))
    except (TypeError, ValueError):
        return "—"


def _as_text(raw: Any) -> str:
    return "—" if raw in (None, "") else str(raw)


_MANIFEST_FIELDS = (
    ("encounter", "encounter_type", _as_text),
    ("age", "age", _as_text),
    ("gender", "gender", _as_text),
    ("admission_type", "admission_type", _plain_code),
    ("discharge_location", "discharge_location", _as_text),
    ("length_of_stay_days", "index_los_days", _whole_days),
    ("prior_admissions", "prior_admission_count", _as_text),
    ("prior_inpatient_days", "prior_inpatient_days", _as_text),
    ("ed_visits_prior_year", "recent_ed_visits", _as_text),
    ("insurance", "insurance", _as_text),
    ("race", "race", _as_text),
)


def _point(points: list[dict], index: int) -> dict:
    return points[index] if -len(points) <= index < len(points) else {}


def _extreme(points: list[dict], reference_low: Any, reference_high: Any) -> dict:
    """The reading furthest outside its reference interval, or the first point."""
    best = points[0] if points else {}
    worst = -1.0
    for point in points:
        value = point.get("value")
        if not isinstance(value, (int, float)):
            continue
        excursion = 0.0
        if isinstance(reference_low, (int, float)) and value < reference_low:
            excursion = (reference_low - value) / max(abs(reference_low), 1e-9)
        elif isinstance(reference_high, (int, float)) and value > reference_high:
            excursion = (value - reference_high) / max(abs(reference_high), 1e-9)
        if excursion > worst:
            worst, best = excursion, point
    return best


def _cell(point: dict) -> str:
    if not point:
        return "—"
    day = point.get("day")
    day_text = "—" if day is None else f"d{day:g}"
    return f"{day_text} · {point.get('value')} · `{point.get('flag')}`"


#: The `/ehr` headings, in the order `EHR_FORMATTING` asks for them, with the sections
#: that support each. The prompt's rule is "a label the passages do not support is a label
#: you omit", so the expected set is a function of the sections the tool returned —
#: computed here rather than left for whoever authors the next item to guess at.
EHR_HEADINGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Presentation & diagnosis",
     ("history_of_present_illness", "chief_complaint", "admission_diagnosis",
      "discharge_diagnosis")),
    ("Hospital course & procedures",
     ("brief_hospital_course", "major_procedure")),
    ("Discharge plan & medications",
     ("discharge_medications", "discharge_instructions")),
    ("Readmission risk & trajectory", ("<risk estimate>",)),
)


def heading_sets(sections: dict[str, str]) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Which `/ehr` headings these passages support, and which they do not.

    `Readmission risk & trajectory` is supported by the estimate rather than by a
    passage: every inpatient dev record has one, so it is always in the first list here.
    The support list is returned with each heading because a heading that is supported by
    the hospital course alone reads differently from one supported by the presenting
    history, and the difference decides whether an answer key may say "the note does not
    describe the procedure".
    """
    available = set(sections) | {"<risk estimate>"}
    present: list[tuple[str, list[str]]] = []
    absent: list[str] = []
    for heading, support in EHR_HEADINGS:
        hits = [key for key in support if key in available]
        if hits:
            present.append((heading, hits))
        else:
            absent.append(heading)
    return present, absent


async def _async_facts() -> dict[int, dict]:
    facts: dict[int, dict] = {}
    for hadm_id in DEV:
        meds = await discharge_medications(hadm_id)
        labs = await get_lab_trends(hadm_id)
        notes = await rag_search_sections(hadm_id)
        # The risk result as the MODEL reads it: `probability_display` and
        # `threshold_display` in place of the floats, and each driver's value stated
        # clinically with its unit. Projected through the same function the agent uses,
        # so a fact copied out of this sheet is the string the model was given rather than
        # a re-derivation of it.
        risk = await predict_readmission(hadm_id)
        presented = _presented_for_model("predict_readmission", risk)
        facts[hadm_id] = {"meds": meds, "labs": labs, "notes": notes,
                          "risk": risk, "risk_presented": presented}
    return facts


def render(inventory: dict, rows: dict[int, dict], facts: dict[int, dict]) -> str:
    lines: list[str] = [
        "# Dev Cohort Fact Sheet — Phase 04",
        "",
        "Generated by `evaluation/inventory/dev_fact_sheet.py` from "
        "`evaluation/cohort_provenance_inventory.json`, the four patient-scoped tools, "
        "and the feature row each patient was scored on. **Zero LLM calls.** Every "
        "figure is copied; every note sentence is quoted.",
        "",
        "Threshold is 0.11 on every record below; the decision is the model's own.",
        "",
    ]

    for hadm_id in DEV:
        # `dev` holds the seven records the inventory was built around; `all_records`
        # holds every served record. The cohort outgrew the first on 2026-11-22, so the
        # richer slice is preferred where it has an entry and the full inventory answers
        # for the rest — the fields are the same shape either way.
        record = inventory["dev"].get(str(hadm_id)) or inventory["all_records"][
            str(hadm_id)]
        row = rows.get(hadm_id, {})
        risk = record["risk"]
        profile = record["profile"]
        f = facts[hadm_id]
        lines += [f"## {hadm_id}", ""]
        if record["outpatient"]:
            lines += ["**OUTPATIENT** — no admission, no readmission risk.", ""]

        # 1 — manifest and risk
        presented = f.get("risk_presented") or {}
        lines += ["### Manifest and risk", "",
                  "The raw field and the string the product publishes for it. A sentence "
                  "may quote either; the right-hand column is the form the widget prints.",
                  "",
                  "| field | raw | published |", "|---|---|---|"]
        for label, key, display in _MANIFEST_FIELDS:
            raw = profile.get(key, "—")
            lines.append(f"| {label} | `{raw}` | {display(raw)} |")
        oncology = row.get("oncology_flag")
        lines.append(f"| oncology_flag (feature row) | `{oncology}` | — |")
        lines += [
            "",
            f"**Readmission probability {risk['probability']:.6f}** "
            f"(published **{presented.get('probability_display', '—')}**) against a "
            f"threshold of {risk['threshold']} "
            f"(published **{presented.get('threshold_display', '—')}**) → decision "
            f"`{'above' if risk['decision'] else 'below'}` "
            f"(model `{risk.get('model_version')}`).",
            "",
            "Drivers, in the order the model receives them. `label` is `humanize()`; "
            "`value` is the patient's own value in clinical form, or empty when the "
            "factor is absent; `chip` is the question a driver bar asks. `contribution` "
            "is read from the raw response and is **not** published to the model or the "
            "widget — it is here so a reviewer can rank the drivers, and no answer key "
            "may quote it.",
            "",
            "| # | feature | label | direction | value | contribution | chip |",
            "|---|---|---|---|---|---|---|",
        ]
        projected = presented.get("top_factors") or risk["drivers"]
        raw_contribution = {str(d.get("feature") or d.get("name") or ""): d.get("contribution")
                            for d in risk["drivers"]}
        for index, driver in enumerate(projected, start=1):
            feature = str(driver.get("feature") or driver.get("name") or "")
            label = humanize(feature)
            value = driver.get("value", row.get(feature))
            contribution = raw_contribution.get(feature)
            lines.append(
                f"| {index} | `{feature}` | {label} | {driver.get('direction', '—')} "
                f"| {value if value not in (None, '') else '—'} "
                f"| {'—' if contribution is None else f'{contribution:+.4f}'} "
                f"| {driver_question(label)} |")
        lines.append("")

        # 2 — medications
        passages = f["meds"].get("passages") or []
        text = "\n\n".join(str(p.get("text") or "") for p in passages).strip()
        lines += ["### Discharge medications", ""]
        # Two signals, published side by side because they disagree on two records and a
        # reader must not be given only the one that happens to be convenient. The
        # inventory flags are what the cohort was SELECTED on; the text scan is what the
        # passages this tool returns actually say. Where they differ, the sheet names the
        # difference rather than resolving it.
        flagged = record.get("medications") or {}
        flags = (f"inventory flags — taper: "
                 f"{'yes' if flagged.get('mentions_taper') else 'no'}, "
                 f"anticoagulant: "
                 f"{'yes' if flagged.get('mentions_anticoagulant') else 'no'}, "
                 f"dedicated medication section: "
                 f"{'yes' if flagged.get('dedicated_section_present') else 'no'}.")
        if not text:
            note = f["meds"].get("note") or "no medication text returned"
            lines += [f"**Empty state** — {len(passages)} passage(s), {note}.",
                      flags, ""]
        else:
            classes = ", ".join(
                f"{name}: {'PRESENT' if pattern.search(text) else 'absent'}"
                f"{_evidence(pattern, text)}"
                for name, pattern in DRUG_CLASSES.items())
            lines += [
                f"{len(passages)} passage(s), {len(text)} characters.",
                "",
                flags,
                f"text scan of the passages below — {classes}.",
            ]
            lines += _disagreements(flagged, text)
            lines += [
                "",
                "```",
                text[:4000],
                "```",
                "",
            ]

        # 3 — the laboratory panel, per analyte and per timepoint
        trends = f["labs"].get("trends") or []
        lines += [
            "### Laboratory panel",
            "",
            "One row per analyte, in the order the tool returned them. `baseline` is the "
            "first timepoint, `extreme` is the reading furthest outside the reference "
            "interval (the nadir or the peak), and `discharge` is the last. `anchored` "
            "names the values the readmission model was scored on: those are the "
            "patient's own numbers and may be stated as such; anything else was "
            "generated for a measure the model never saw.",
            "",
            "| analyte | unit | reference | decimals | baseline | extreme | discharge "
            "| anchored |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for trend in trends:
            points = trend.get("series") or []
            reference = f"{trend.get('reference_low')}–{trend.get('reference_high')}"
            anchored = ", ".join(trend.get("anchored") or []) or "—"
            extreme = _extreme(points, trend.get("reference_low"),
                               trend.get("reference_high"))
            lines.append(
                f"| {trend.get('label')} | {trend.get('unit')} | {reference} "
                f"| {trend.get('decimals')} | {_cell(_point(points, 0))} "
                f"| {_cell(extreme)} "
                f"| {_cell(_point(points, -1))} | {anchored} |")
        critical = [t for t in trends
                    if any(str(point.get("flag", "")).startswith("critical")
                           for point in t.get("series") or [])]
        abnormal = [t for t in trends if _discharge_point(t).get("flag") != "normal"]
        worst = max(trends, key=_excursion) if trends else None
        lines += [
            "",
            f"{len(trends)} panels; {len(abnormal)} abnormal at discharge"
            + (f" ({', '.join(t.get('label') for t in abnormal)})." if abnormal else "."),
            (f"Critical flags anywhere in the admission: "
             f"**{', '.join(str(t.get('label')) for t in critical)}**."
             if critical else "No analyte carries a critical flag on this record."),
            (f"Most abnormal at discharge: **{worst.get('label')}** "
             f"{_discharge_point(worst).get('value')} {worst.get('unit')} "
             f"(flag `{_discharge_point(worst).get('flag')}`, reference "
             f"{worst.get('reference_low')}–{worst.get('reference_high')})."
             if worst and _excursion(worst) > 0 else
             "No value sits outside its reference range at discharge."),
            "",
        ]

        # 4 — note sections, headings and the verbatim text
        sections = _section_map(f["notes"])
        lines += ["### Clinical note", ""]
        if sections:
            headings, missing = heading_sets(sections)
            absent = sorted(set(KNOWN_HEADINGS) - set(sections))
            absent_text = ", ".join(f"`{key}`" for key in absent)
            supported = ", ".join(
                f"**{name}.** ("
                + ", ".join("the risk estimate" if key == "<risk estimate>"
                            else f"`{key}`" for key in hits)
                + ")"
                for name, hits in headings)
            lines += [
                "Sections present: "
                + ", ".join(f"`{key}`" for key in sections) + ".",
                "",
                f"Sections absent: {len(absent)} of the "
                f"{len(KNOWN_HEADINGS)} headings the retriever knows"
                + (f" — {absent_text}" if absent else "") + ".",
                "",
                "`/ehr` headings these sections support, and the passage that earns each: "
                + (supported or "none") + ".",
                "`/ehr` headings to OMIT: "
                + (", ".join(f"**{name}.**" for name in missing) or "none") + ".",
                "",
            ]
            lines += [f"- {bullet}" for bullet in _bullets(sections)]
            lines += ["", "#### Verbatim passages", ""]
            for key, text in sections.items():
                lines += [f"**{key}**", "", "```", text.strip(), "```", ""]
        else:
            lines.append(
                f"No discharge-summary sections — {f['notes'].get('note') or 'none'}. "
                "`/ehr` can only answer this record through its record-wide fallback.")
            lines.append("")

    lines += [
        "---",
        "",
        "Outpatient records carry a model score in the store but are refused by `/risk` "
        "(`TF-1`, fixed in Phase 03 §5): the number is shown here as provenance, not as "
        "a clinical claim.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    rows = _rows(DEV)
    facts = asyncio.run(_async_facts())
    OUT.write_text(render(inventory, rows, facts), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(DEV)} dev records, 0 LLM calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
