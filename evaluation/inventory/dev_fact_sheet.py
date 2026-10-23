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

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from google.cloud import bigquery  # noqa: E402

from services.mcp.config import ENTITY_ID_COLUMN, PROJECT, TABLE  # noqa: E402
from services.mcp.tools import (  # noqa: E402
    discharge_medications,
    get_lab_trends,
    rag_search_sections,
)

INVENTORY = REPO / "evaluation" / "cohort_provenance_inventory.json"
OUT = REPO / "evaluation" / "dev_cohort_fact_sheet.md"

DEV = (90000015, 90000108, 90000106, 90000105, 90000023, 90000051, 90000073)

#: Drug classes the suite's shorthand and false-premise trajectories will ask about, so
#: the fact sheet states presence or absence rather than leaving it to be inferred.
#:
#: Brand names matter as much as generic ones: this cohort's note says "Coumadin" and
#: "Lovenox", not warfarin and enoxaparin. The first version of this table listed only
#: generics and reported anticoagulant as ABSENT on 90000015 — a false negative on the
#: exact fact a false-premise trap would be built from.
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
    "taper": re.compile(r"taper|wean|then decrease|decrease to|reduce to", re.I),
}


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


async def _async_facts() -> dict[int, dict]:
    facts: dict[int, dict] = {}
    for hadm_id in DEV:
        meds = await discharge_medications(hadm_id)
        labs = await get_lab_trends(hadm_id)
        notes = await rag_search_sections(hadm_id)
        facts[hadm_id] = {"meds": meds, "labs": labs, "notes": notes}
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
        record = inventory["dev"][str(hadm_id)]
        row = rows.get(hadm_id, {})
        risk = record["risk"]
        profile = record["profile"]
        f = facts[hadm_id]
        lines += [f"## {hadm_id}", ""]
        if record["outpatient"]:
            lines += ["**OUTPATIENT** — no admission, no readmission risk.", ""]

        # 1 — manifest and risk
        lines += ["### Manifest and risk", "", "| field | value |", "|---|---|"]
        for label, key in (("age", "age"), ("gender", "gender"),
                           ("discharge_location", "discharge_location"),
                           ("length_of_stay_days", "index_los_days"),
                           ("prior_inpatient_days", "prior_inpatient_days"),
                           ("prior_admission_count", "prior_admission_count"),
                           ("recent_ed_visits", "recent_ed_visits"),
                           ("admission_type", "admission_type"),
                           ("insurance", "insurance"), ("race", "race")):
            lines.append(f"| {label} | {profile.get(key, '—')} |")
        oncology = row.get("oncology_flag")
        lines.append(f"| oncology_flag (feature row) | {oncology} |")
        lines += [
            "",
            f"**Readmission probability {risk['probability']:.6f}** against a "
            f"threshold of {risk['threshold']} → decision "
            f"`{'above' if risk['decision'] else 'below'}` "
            f"(model `{risk.get('model_version')}`).",
            "",
            "| # | driver | raw value | contribution | direction |",
            "|---|---|---|---|---|",
        ]
        for index, driver in enumerate(risk["drivers"], start=1):
            name = driver["name"]
            value = row.get(name, driver.get("value"))
            if isinstance(value, float):
                value = f"{value:g}"
            lines.append(f"| {index} | `{name}` | {value if value is not None else '—'} "
                         f"| {driver['contribution']:+.4f} | {driver['direction']} |")
        lines.append("")

        # 2 — medications
        passages = f["meds"].get("passages") or []
        text = "\n\n".join(str(p.get("text") or "") for p in passages).strip()
        lines += ["### Discharge medications", ""]
        if not text:
            note = f["meds"].get("note") or "no medication text returned"
            lines += [f"**Empty state** — {len(passages)} passage(s), {note}.", ""]
        else:
            classes = ", ".join(
                f"{name}: {'PRESENT' if pattern.search(text) else 'absent'}"
                for name, pattern in DRUG_CLASSES.items())
            lines += [
                f"{len(passages)} passage(s), {len(text)} characters. {classes}.",
                "",
                "```",
                text[:4000],
                "```",
                "",
            ]

        # 3 — labs at discharge
        trends = f["labs"].get("trends") or []
        lines += ["### Laboratory values at discharge", "",
                  "| analyte | discharge value | unit | reference | decimals | flag |",
                  "|---|---|---|---|---|---|"]
        for trend in trends:
            point = _discharge_point(trend)
            ref = f"{trend.get('reference_low')}–{trend.get('reference_high')}"
            lines.append(
                f"| {trend.get('label')} | {point.get('value')} | {trend.get('unit')} "
                f"| {ref} | {trend.get('decimals')} | {point.get('flag')} |")
        abnormal = [t for t in trends if _discharge_point(t).get("flag") != "normal"]
        worst = max(trends, key=_excursion) if trends else None
        lines += [
            "",
            f"{len(trends)} panels; {len(abnormal)} abnormal at discharge"
            + (f" ({', '.join(t.get('label') for t in abnormal)})." if abnormal else "."),
            (f"Most abnormal at discharge: **{worst.get('label')}** "
             f"{_discharge_point(worst).get('value')} {worst.get('unit')} "
             f"(flag `{_discharge_point(worst).get('flag')}`, reference "
             f"{worst.get('reference_low')}–{worst.get('reference_high')})."
             if worst and _excursion(worst) > 0 else
             "No value sits outside its reference range at discharge."),
            "",
        ]

        # 4 — note sections
        sections = _section_map(f["notes"])
        lines += ["### Clinical note", ""]
        if sections:
            lines.append("Sections present: "
                         + ", ".join(f"`{key}`" for key in sections) + ".")
            lines.append("")
            lines += [f"- {bullet}" for bullet in _bullets(sections)]
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
