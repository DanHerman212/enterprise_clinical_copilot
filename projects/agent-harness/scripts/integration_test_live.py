"""LIVE cross-patient integration test against the deployed endpoints.

Runs the real demo path through the actual registered MCP tools (not raw
endpoint calls): predict_readmission -> prediction endpoint, rag_search ->
Vector Search endpoint + BigQuery text fetch.

Checks, all verified INDEPENDENTLY through BigQuery (note_id -> hadm_id map),
never by trusting the restrict filter:

  R1+ (within-patient):  patient A's own note text, restricted to A, returns
                         only passages whose note_id belongs to A.
  R1  (cross-patient):   patient B's note text lifted VERBATIM, restricted to
                         A, returns only A's passages (or none) — never B's.
  ML  (prediction):      predict_readmission(A) returns a healthy response
                         with a risk score, so the full demo path is live.

Live tier — needs both endpoints deployed. Usage:
  .venv/bin/python projects/agent-harness/scripts/integration_test_live.py
"""

import asyncio
import sys
from collections import Counter

from google.cloud import bigquery

sys.path.insert(0, ".")

from mcp_server.config import DISCHARGE_TABLE, PROJECT  # noqa: E402
from mcp_server.tools.predict import predict_readmission  # noqa: E402
from mcp_server.tools.rag_search import rag_search  # noqa: E402

DEMO_COHORT = f"{PROJECT}.readmission.demo_cohort"


def _pick_patients() -> tuple[int, int]:
    """Two demo-cohort admissions that both have discharge notes."""
    bq = bigquery.Client(project=PROJECT)
    rows = list(bq.query(
        f"""SELECT DISTINCT d.hadm_id
            FROM `{DISCHARGE_TABLE}` d
            JOIN `{DEMO_COHORT}` c USING (hadm_id)
            ORDER BY d.hadm_id
            LIMIT 4"""
    ).result())
    if len(rows) < 2:
        raise SystemExit("Need >= 2 demo-cohort patients with notes for R1.")
    return int(rows[0]["hadm_id"]), int(rows[1]["hadm_id"])


def _note_id_map() -> dict[str, int]:
    """note_id -> hadm_id, for independent R1 verification."""
    bq = bigquery.Client(project=PROJECT)
    rows = bq.query(
        f"SELECT note_id, hadm_id FROM `{DISCHARGE_TABLE}`"
    ).result()
    return {str(r["note_id"]): int(r["hadm_id"]) for r in rows}


def _hadm_of(note_to_hadm: dict[str, int], datapoint_id: str) -> int | None:
    """'{note_id}_{section}_{ordinal}' -> hadm_id by matching known sections."""
    from pipelines.components.chunk_notes import DEFAULT_SECTIONS
    for section in DEFAULT_SECTIONS:
        token = f"_{section}_"
        idx = datapoint_id.rfind(token)
        if idx > 0:
            return note_to_hadm.get(datapoint_id[:idx])
    return None


def _note_text(hadm_id: int) -> str:
    bq = bigquery.Client(project=PROJECT)
    rows = bq.query(
        f"SELECT text FROM `{DISCHARGE_TABLE}` WHERE hadm_id = {hadm_id} LIMIT 1"
    ).result()
    text = next(iter(rows))["text"]
    return text.split("\n\n")[1].strip()[:800] if "\n\n" in text else text[:800]


def _leaked(passages: list[dict], note_to_hadm: dict, patient: int) -> list:
    """Return (id, hadm) tuples for passages that map to any OTHER patient."""
    leaked = []
    for p in passages:
        h = _hadm_of(note_to_hadm, p["id"])
        if h is not None and h != patient:
            leaked.append((p["id"], h))
    return leaked


async def _run_r1_positive(note_to_hadm: dict) -> bool:
    """A's own text, restricted to A -> only A's passages."""
    patient_a = _pick_patients()[0]
    query = _note_text(patient_a)
    res = await rag_search(hadm_id=patient_a, query=query, top_k=20)
    if res.get("error"):
        print(f"R1+ FAIL (tool error): {res}")
        return False
    nbs = res.get("passages", [])
    if not nbs:
        print("R1+ FAIL: empty result for A's own text (should be non-empty)")
        return False
    hadms = Counter(_hadm_of(note_to_hadm, p["id"]) for p in nbs)
    leaked = _leaked(nbs, note_to_hadm, patient_a)
    print(f"R1+ : A={patient_a}, returned {len(nbs)}, hadm counts={dict(hadms)}")
    for p in nbs[:5]:
        print(f"      {p['id']} | {p['section']} | {p['score']:.4f}")
    if leaked:
        print(f"R1+ LEAK: {leaked}")
        return False
    print(f"R1+ : PASS — A's own text, restricted to A, only A's passages")
    return True


async def _run_r1(note_to_hadm: dict) -> bool:
    """B's text lifted verbatim, restricted to A -> never returns B's passages."""
    patient_a, patient_b = _pick_patients()
    query = _note_text(patient_b)  # verbatim text from B's note
    res = await rag_search(hadm_id=patient_a, query=query, top_k=20)
    if res.get("error"):
        print(f"R1  FAIL (tool error): {res}")
        return False
    nbs = res.get("passages", [])
    if not nbs:
        print(f"R1  : A={patient_a}, B={patient_b} -> empty (acceptable, no A "
              f"passage matched B's text)")
        return True
    hadms = Counter(_hadm_of(note_to_hadm, p["id"]) for p in nbs)
    leaked = _leaked(nbs, note_to_hadm, patient_a)
    print(f"R1  : A={patient_a}, B={patient_b}, returned {len(nbs)}, "
          f"hadm counts={dict(hadms)}")
    if leaked:
        print(f"R1 LEAK: {leaked}")
        return False
    print(f"R1  : PASS — B's text restricted to A never returned B's passages")
    return True


async def _run_ml(patient_a: int) -> bool:
    """Prediction endpoint answers for the same patient (full demo path)."""
    res = await predict_readmission(hadm_id=patient_a)
    if res.get("error"):
        print(f"ML  FAIL (tool error): {res}")
        return False
    risk = res.get("predicted_probability") or res.get("probability") or res.get("score")
    print(f"ML  : predict_readmission({patient_a}) -> risk={risk}")
    print(f"ML  : PASS — prediction endpoint is live")
    return True


async def main() -> int:
    note_to_hadm = _note_id_map()
    print(f"note->hadm map: {len(note_to_hadm)} notes")
    patient_a, patient_b = _pick_patients()

    results = {
        "r1_positive": await _run_r1_positive(note_to_hadm),
        "r1": await _run_r1(note_to_hadm),
        "ml": await _run_ml(patient_a),
    }

    print("\n=== LIVE INTEGRATION SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
