"""A4: does the note text know something the model does not?

This is the evidence gate's central measurement. For every test-split admission
with a discharge note, it tags the narrative sections for risk concepts and
joins against the model's predictions (readmission.test_predictions, built by
scripts/score_test_split.py).

The question is NOT "do concepts correlate with readmission" - a substance-use
mention is largely redundant with the substance-use ICD code the model already
has, so lift against the label would mostly measure redundancy. Instead the
probe restricts to admissions the model scored BELOW threshold and compares
concept rates between:

    FN  - model said low risk, patient was readmitted   (the model's blind spot)
    TN  - model said low risk, patient was not readmitted

Concept rate higher in FN than TN = the notes carry signal invisible to the
structured features. That is the thesis of the RAG build, stated falsifiably.

Two numbers per concept:
    lift      rate_fn / rate_tn - is the signal real?
    coverage  fraction of admissions with >=1 positive mention - lift can be
              real but rare; coverage tells us whether the demo will have
              anything to show.

Hedging is reported in its own rows and never counts toward the gate
(settled 2026-08-04); it is tagged only in the sections where a clinician is
assessing the patient (see HEDGING_SECTIONS).

Per-note tags are also written to the local cache so cohort selection (guide
sec 10) can reuse them without re-tagging the corpus.

Runs under .venv-nlp:
    .venv-nlp/bin/python scripts/probe_note_signal.py [--limit N]

Writes:  docs/probes/note_signal_probe.json      (aggregates only, committable)
         ~/.cache/.../note_concepts.jsonl.gz     (per-note tags, PHI-adjacent)
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.concepts import GATING_CONCEPTS, HEDGING_SECTIONS, tag  # noqa: E402
from rag.notes import CACHE_DIR, iter_notes  # noqa: E402
from rag.sections import (  # noqa: E402
    BRIEF_HOSPITAL_COURSE,
    DISCHARGE_CONDITION,
    HISTORY_OF_PRESENT_ILLNESS,
    PAST_MEDICAL_HISTORY,
    SOCIAL_HISTORY,
    parse_note,
)

PROJECT = "trim-icon-498815-a0"
PREDICTIONS_TABLE = f"{PROJECT}.readmission.test_predictions"
ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "docs" / "probes" / "note_signal_probe.json"
TAGS_PATH = CACHE_DIR / "note_concepts.jsonl.gz"

# Narrative sections only. Discharge Instructions are deliberately excluded:
# patient-directed boilerplate ("call your doctor if...") that adds noise, not
# assessment. Hedging is further restricted to HEDGING_SECTIONS.
TAGGED_SECTIONS = (
    HISTORY_OF_PRESENT_ILLNESS,
    PAST_MEDICAL_HISTORY,
    SOCIAL_HISTORY,
    BRIEF_HOSPITAL_COURSE,
    DISCHARGE_CONDITION,
)

QUADRANTS = ("fn", "tn", "fp", "tp")


def load_predictions(client: bigquery.Client) -> dict[int, dict]:
    rows = client.query(
        f"SELECT hadm_id, probability, decision, readmission_30d "
        f"FROM `{PREDICTIONS_TABLE}`"
    ).result()
    return {row["hadm_id"]: dict(row) for row in rows}


def quadrant(decision: int, label: int) -> str:
    if decision == 0:
        return "fn" if label == 1 else "tn"
    return "tp" if label == 1 else "fp"


def note_concepts(text: str) -> dict[str, list[str]]:
    """Positive concepts in a note, with the sections each was found in."""
    parsed = parse_note(text)
    found: dict[str, set[str]] = {}
    for section_name in TAGGED_SECTIONS:
        for section in parsed.get_all(section_name):
            for mention in tag(section.body):
                if not mention.is_positive:
                    continue
                if mention.concept == "hedging" and section_name not in HEDGING_SECTIONS:
                    continue
                found.setdefault(mention.concept, set()).add(section_name)
    return {concept: sorted(sections) for concept, sections in found.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="tag only the first N notes (throughput check)")
    args = parser.parse_args()

    client = bigquery.Client(project=PROJECT)
    predictions = load_predictions(client)
    print(f"Predictions loaded: {len(predictions)}")

    concept_names = (*GATING_CONCEPTS, "hedging")
    per_quadrant_total: Counter[str] = Counter()
    per_quadrant_concept: dict[str, Counter[str]] = {q: Counter() for q in QUADRANTS}
    notes_missing_prediction = 0
    tagged = 0
    started = time.monotonic()

    tags_out = gzip.open(TAGS_PATH.with_suffix(".tmp"), "wt", encoding="utf-8")
    with tags_out:
        for note in iter_notes():
            if args.limit is not None and tagged >= args.limit:
                break
            pred = predictions.get(note["hadm_id"])
            if pred is None:
                notes_missing_prediction += 1
                continue

            concepts = note_concepts(note["text"])
            q = quadrant(pred["decision"], pred["readmission_30d"])
            per_quadrant_total[q] += 1
            for concept in concepts:
                per_quadrant_concept[q][concept] += 1

            tags_out.write(json.dumps({
                "hadm_id": note["hadm_id"],
                "quadrant": q,
                "probability": pred["probability"],
                "concepts": concepts,
            }) + "\n")

            tagged += 1
            if tagged % 1000 == 0:
                rate = tagged / (time.monotonic() - started)
                remaining = (len(predictions) - tagged) / rate if rate else 0
                print(f"  …{tagged} notes  ({rate:.0f}/s, ~{remaining/60:.0f} min left)")

    if notes_missing_prediction:
        print(f"FAILED: {notes_missing_prediction} cached notes have no "
              f"prediction row; A2 and the cache disagree about the test split.")
        return 1
    TAGS_PATH.with_suffix(".tmp").rename(TAGS_PATH)

    elapsed = time.monotonic() - started
    n_fn, n_tn = per_quadrant_total["fn"], per_quadrant_total["tn"]

    concepts_report = {}
    for concept in concept_names:
        c_fn = per_quadrant_concept["fn"][concept]
        c_tn = per_quadrant_concept["tn"][concept]
        rate_fn = c_fn / n_fn if n_fn else 0.0
        rate_tn = c_tn / n_tn if n_tn else 0.0
        concepts_report[concept] = {
            "gates": concept in GATING_CONCEPTS,
            "fn_with_concept": c_fn,
            "tn_with_concept": c_tn,
            "rate_fn": round(rate_fn, 4),
            "rate_tn": round(rate_tn, 4),
            "lift": round(rate_fn / rate_tn, 2) if rate_tn else None,
            "rate_tp": round(
                per_quadrant_concept["tp"][concept] / per_quadrant_total["tp"], 4
            ) if per_quadrant_total["tp"] else None,
        }

    # Coverage: does the demo have something to show, whatever the lift says?
    def any_gating(counter_totals: str) -> int:
        # per-note "any gating concept" needs a second pass over the tags file
        count = 0
        with gzip.open(TAGS_PATH, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if counter_totals in (record["quadrant"], "all") and any(
                    c in GATING_CONCEPTS for c in record["concepts"]
                ):
                    count += 1
        return count

    covered_all = any_gating("all")
    covered_fn = any_gating("fn")

    artifact = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "notes_tagged": tagged,
        "limit": args.limit,
        "elapsed_seconds": round(elapsed, 1),
        "tagged_sections": list(TAGGED_SECTIONS),
        "quadrant_sizes": dict(per_quadrant_total),
        "coverage": {
            "any_gating_concept_all": covered_all,
            "any_gating_concept_all_pct": round(100 * covered_all / tagged, 1) if tagged else None,
            "any_gating_concept_fn": covered_fn,
            "any_gating_concept_fn_pct": round(100 * covered_fn / n_fn, 1) if n_fn else None,
        },
        "concepts": concepts_report,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")

    print(f"\nWrote {ARTIFACT_PATH}")
    print(f"Tagged {tagged} notes in {elapsed/60:.1f} min "
          f"({tagged/elapsed:.0f}/s)")
    print(f"Quadrants: {dict(per_quadrant_total)}")
    print(f"Coverage (any gating concept): {artifact['coverage']['any_gating_concept_all_pct']}% overall, "
          f"{artifact['coverage']['any_gating_concept_fn_pct']}% of FN")
    print(f"\n{'concept':<22}{'FN rate':>9}{'TN rate':>9}{'lift':>7}{'gates':>7}")
    for concept, row in concepts_report.items():
        lift = row["lift"] if row["lift"] is not None else "-"
        print(f"{concept:<22}{row['rate_fn']:>9}{row['rate_tn']:>9}"
              f"{lift:>7}{'yes' if row['gates'] else 'no':>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
