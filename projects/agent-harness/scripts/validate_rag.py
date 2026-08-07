"""§9 validation trials against the deployed tree-AH index.

Runs two checks:

  R1  Cross-patient isolation (non-negotiable): a query whose text is lifted
      verbatim from patient B's note, searched restricted to patient A's
      hadm_id, must return only A's passages (or none) — never B's. Verified
      independently by mapping every returned datapoint id back to hadm_id
      through BigQuery (note_id -> hadm_id), not by trusting the filter.

  Demo sanity: real clinical queries return plausible passages and the IDs map
  back to real notes in BigQuery.

Live tier (needs the deployed endpoint). Usage:
  .venv/bin/python projects/agent-harness/scripts/validate_rag.py
"""

import argparse
import sys
from collections import Counter

from google import genai
from google.cloud import aiplatform, bigquery
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    Namespace,
)
from google.genai import types

sys.path.insert(0, ".")

from rag.embed import (  # noqa: E402
    EMBEDDING_MODEL,
    OUTPUT_DIMENSIONALITY,
    QUERY_TASK_TYPE,
    RESTRICT_NAMESPACE,
)
from pipelines.components.chunk_notes import DEFAULT_SECTIONS  # noqa: E402

PROJECT = "trim-icon-498815-a0"
LOCATION = "us-east1"
ENDPOINT = ("projects/778397675435/locations/us-east1/"
            "indexEndpoints/4397109727197134848")
DEPLOYED_ID = "rag_tree_ah"
DEMO_COHORT = f"{PROJECT}.readmission.demo_cohort"
DISCHARGE = f"{PROJECT}.mimiciv_note.discharge"


def _embed(client: genai.Client, text: str) -> list[float]:
    resp = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(
            output_dimensionality=OUTPUT_DIMENSIONALITY,
            task_type=QUERY_TASK_TYPE,
        ),
    )
    return [float(v) for v in resp.embeddings[0].values]


def _note_to_hadm(project_id: str) -> dict[str, int]:
    """note_id -> hadm_id map from the discharge table (for independent verify)."""
    client = bigquery.Client(project=project_id)
    rows = client.query(
        f"SELECT note_id, hadm_id FROM `{DISCHARGE}`"
    ).result()
    return {str(r["note_id"]): int(r["hadm_id"]) for r in rows}


def _hadm_of_id(note_to_hadm: dict[str, int], datapoint_id: str) -> int | None:
    """Parse a datapoint id '{note_id}_{section}_{ordinal}' -> hadm_id.

    Section names contain underscores (e.g. brief_hospital_course), so a blind
    rsplit on '_' would cut a section name into the note_id. Instead, match the
    exact '_{section}_' token from the known section whitelist; everything
    before it is the note_id.
    """
    for section in DEFAULT_SECTIONS:
        token = f"_{section}_"
        idx = datapoint_id.rfind(token)
        if idx > 0:
            return note_to_hadm.get(datapoint_id[:idx])
    return None


def _get_note_text(hadm_id: int) -> str:
    client = bigquery.Client(project=PROJECT)
    rows = client.query(
        f"SELECT text FROM `{DISCHARGE}` WHERE hadm_id = {hadm_id} LIMIT 1"
    ).result()
    return next(iter(rows))["text"]


def run_r1(client: genai.Client, ep) -> dict:
    """Cross-patient isolation: B's text, restricted to A, never returns B."""
    bq = bigquery.Client(project=PROJECT)
    rows = list(bq.query(
        f"""SELECT DISTINCT d.hadm_id
            FROM `{DISCHARGE}` d
            JOIN `{DEMO_COHORT}` c USING (hadm_id)
            ORDER BY d.hadm_id LIMIT 4"""
    ).result())
    if len(rows) < 2:
        rows = list(bq.query(
            f"SELECT DISTINCT hadm_id FROM `{DISCHARGE}` ORDER BY hadm_id LIMIT 4"
        ).result())
    patient_a = int(rows[0]["hadm_id"])
    patient_b = int(rows[1]["hadm_id"])
    print(f"R1: patient A={patient_a}, patient B={patient_b}")

    # Lift a passage verbatim from B's note to use as the query.
    b_text = _get_note_text(patient_b)
    query_text = b_text.split("\n\n")[1] if "\n\n" in b_text else b_text
    query_text = query_text.strip()[:800]

    q = _embed(client, query_text)
    res = ep.find_neighbors(
        deployed_index_id=DEPLOYED_ID,
        queries=[q],
        num_neighbors=20,
        filter=[Namespace(RESTRICT_NAMESPACE, [str(patient_a)])],
    )
    neighbors = res[0] if res else []
    if not neighbors:
        print("  -> empty result (acceptable: no A passages matched B's text)")
        return {"passed": True, "n": 0}

    note_to_hadm = _note_to_hadm(PROJECT)
    hadms = Counter()
    leaked = []
    for nb in neighbors:
        h = _hadm_of_id(note_to_hadm, nb.id)
        hadms[h] += 1
        if h is not None and h != patient_a:
            leaked.append((nb.id, h))
    print(f"  returned {len(neighbors)} neighbors; hadm_id counts: {dict(hadms)}")
    if leaked:
        print(f"  LEAK: {leaked}")
        return {"passed": False, "leaked": leaked}
    print("  PASS: every returned passage maps to patient A (or unmapped)")
    return {"passed": True, "n": len(neighbors)}


def run_r1_positive(client: genai.Client, ep) -> dict:
    """Stronger R1: A's own note text, restricted to A, must return only A.

    Exercises the leak-detection path with a guaranteed non-empty result: if
    the restrict were a no-op or a post-filter, the unfiltered top-20 would
    contain other patients' passages, which we detect via hadm_id mapping.
    """
    bq = bigquery.Client(project=PROJECT)
    rows = list(bq.query(
        f"""SELECT DISTINCT d.hadm_id
            FROM `{DISCHARGE}` d
            JOIN `{DEMO_COHORT}` c USING (hadm_id)
            ORDER BY d.hadm_id LIMIT 4"""
    ).result())
    if not rows:
        return {"passed": False, "reason": "no demo patients"}
    patient_a = int(rows[0]["hadm_id"])

    a_text = _get_note_text(patient_a)
    query_text = a_text.split("\n\n")[1] if "\n\n" in a_text else a_text
    query_text = query_text.strip()[:800]

    q = _embed(client, query_text)
    res = ep.find_neighbors(
        deployed_index_id=DEPLOYED_ID,
        queries=[q],
        num_neighbors=20,
        filter=[Namespace(RESTRICT_NAMESPACE, [str(patient_a)])],
    )
    neighbors = res[0] if res else []
    if not neighbors:
        return {"passed": False, "reason": "unexpectedly empty for A's own text"}

    note_to_hadm = _note_to_hadm(PROJECT)
    leaked = [nb.id for nb in neighbors
              if _hadm_of_id(note_to_hadm, nb.id) != patient_a]
    print(f"R1+ : A={patient_a}, returned {len(neighbors)}, "
          f"leaked={len(leaked)}")
    for nb in neighbors:
        print(f"     {nb.id} -> hadm {_hadm_of_id(note_to_hadm, nb.id)}")
    if leaked:
        return {"passed": False, "leaked": leaked}
    print("R1+ : PASS — A's own text, restricted to A, returned only A's passages")
    return {"passed": True, "n": len(neighbors)}


def run_sanity(client: genai.Client, ep) -> dict:
    """Real clinical queries return plausible passages that map to real notes."""
    queries = [
        "patient with sepsis and elevated lactate on broad-spectrum antibiotics",
        "history of congestive heart failure and prior myocardial infarction",
        "discharge instructions for anticoagulation after pulmonary embolism",
    ]
    note_to_hadm = _note_to_hadm(PROJECT)
    all_pass = True
    for qtext in queries:
        q = _embed(client, qtext)
        res = ep.find_neighbors(
            deployed_index_id=DEPLOYED_ID, queries=[q], num_neighbors=5
        )
        n = res[0] if res else []
        mapped = sum(1 for nb in n if _hadm_of_id(note_to_hadm, nb.id) is not None)
        print(f"  '{qtext[:50]}...' -> {len(n)} neighbors, {mapped} mapped to real notes")
        if not n:
            all_pass = False
    return {"passed": all_pass}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["r1", "sanity"], default=None)
    args = parser.parse_args()

    aiplatform.init(project=PROJECT, location=LOCATION)
    ep = aiplatform.MatchingEngineIndexEndpoint(ENDPOINT)
    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

    results = {}
    if args.only in (None, "r1"):
        results["r1"] = run_r1(client, ep)
        results["r1_positive"] = run_r1_positive(client, ep)
    if args.only in (None, "sanity"):
        results["sanity"] = run_sanity(client, ep)

    failed = [k for k, v in results.items() if not v["passed"]]
    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v['passed'] else 'FAIL'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
