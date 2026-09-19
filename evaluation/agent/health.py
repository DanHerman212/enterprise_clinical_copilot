"""Retrieval health, classified the same way for both transports.

The classification lives here rather than in either collector because the two
runners must agree on what "this run is corrupted" means. It was a single copy
inside `run_eval.py` while only the local transport existed; a second collector
with its own idea of a failed trace is how a gated pipeline quietly stops being
a gate — the two would drift and only one of them would be measuring what the
threshold was set for.

The failure the gate exists for (2026-08-23): a serving config pointed the
discharge table at one corpus while the deployed index was built from another,
and two-thirds of a 324-question run produced zero-passage traces before anyone
noticed. Both symptoms below are retrieval-layer facts, visible in a single
response — which is why a six-question preflight can stand in for the run.
"""

from __future__ import annotations

RETRIEVAL_TOOLS = ("rag_search", "rag_search_sections")


def classify_trace(rec: dict) -> dict:
    """The problem flags for one trace record, from its tool calls alone."""
    out = {"missing_text": False, "zero_passage": False, "unknown_patient": False}
    for tc in rec.get("tool_calls") or []:
        resp = tc.get("response") or {}
        if not isinstance(resp, dict):
            continue
        if resp.get("error") == "missing_text":
            out["missing_text"] = True
        if resp.get("error") == "unknown_patient":
            out["unknown_patient"] = True
        if tc.get("name") in RETRIEVAL_TOOLS and (
            resp.get("returned", 0) == 0 and not resp.get("error")
        ):
            out["zero_passage"] = True
    return out


def retrieval_failed(flags: dict) -> bool:
    """The retrieval-specific failures the gate counts.

    `unknown_patient` is deliberately not one of them: it is the predict path
    rejecting an admission outside the served cohort, which is a correct answer
    to a question about a patient this system does not serve, and gating on it
    would fail a run for behaving properly.
    """
    return flags["missing_text"] or flags["zero_passage"]


def summarize(recs: list[dict]) -> dict:
    n = len(recs)
    if n == 0:
        return {"n": 0, "retrieval_fail": 0, "missing_text": 0, "zero_passage": 0,
                "unknown_patient": 0, "retrieval_fail_rate": 0.0}
    mt = zp = up = rf = 0
    for r in recs:
        f = classify_trace(r)
        mt += int(f["missing_text"])
        zp += int(f["zero_passage"])
        up += int(f["unknown_patient"])
        rf += int(retrieval_failed(f))
    return {"n": n, "retrieval_fail": rf, "missing_text": mt, "zero_passage": zp,
            "unknown_patient": up, "retrieval_fail_rate": rf / n}


def format_summary(s: dict, label: str = "summary") -> str:
    return (f"[{label}] n={s['n']} retrieval_fail={s['retrieval_fail']} "
            f"({s['retrieval_fail_rate']:.0%}) missing_text={s['missing_text']} "
            f"zero_passage={s['zero_passage']} "
            f"unknown_patient={s['unknown_patient']}")
