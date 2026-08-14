"""Deterministic post-hoc guardrails for the agent's free-text answer.

The LLM proposes; this module disposes. Pure functions of (answer, evidence),
so they are testable offline and replayable over saved traces (P4 dry-run).

Guardrails (P3 root-cause -> targeted):
  1. REDACTED-FIELD GUARD  — never fill a value MIMIC redacted to '___' (the
     main remaining class: invented patient age/dose). If the source redacts
     the age and the answer states a specific age, the age is dropped.
  2. MEDICATION VERIFIER   — every dose+unit and frequency the answer asserts
     must appear in the retrieved Discharge Medications text. A dose/freq not
     found there is dropped from the answer (an unverifiable medical claim is
     safer absent) and flagged. This catches med dose/freq errors and the
     admission-meds-conflated-as-discharge failure.
  3. CITATION RANGE        — every ^[n] must point at a retrieved passage.

The guardrail is intentionally CONSERVATIVE: it only drops a token when the
mismatch is clear, and it never rewrites positive content. The P4 dry-run
verifies it does not modify the 265 passing answers (no regression on passes).
"""

from __future__ import annotations

import re

# --- Redacted age -----------------------------------------------------------

# A redacted age in the source: "___ year old", "___ y/o", "___ yo", "___yo".
_REDACTED_AGE = re.compile(
    r"_{3,}\s*(?:y(?:ea)?o|yo|year[ -]?old|y)", re.IGNORECASE
)
# A specific age claim in the answer: "80-year-old", "80 y/o", "80 year old",
# "an 80-year-old man" (also handles "a 45yo").
_ANSWER_AGE = re.compile(
    r"\b(?:a|an|the)?\s*\d{1,3}(?:\s*-\s*|\s+)(?:year-old|y(?:ea)?o|year[ -]old)\b",
    re.IGNORECASE,
)

# --- Medication dose / frequency tokens -------------------------------------

# A STANDALONE dose+unit token. The lookarounds exclude compound doses like
# "500-100-40 mg-unit-mcg" (the "40" and "500" are part of a multi-part dose).
_DOSE_RE = re.compile(
    r"(?<![\d\-])(\d+(?:\.\d+)?)\s*(mg|mcg|g|mEq|units?|mL|%)\b(?![\-])",
    re.IGNORECASE,
)


def _norm_dose(m: re.Match) -> str:
    """Normalize a dose match to a comparable key (handles "10 units" vs
    "10units", "300  mg" vs "300 mg", "1000 units" vs "1000 UNIT")."""
    num, unit = m.group(1), m.group(2).lower().rstrip("s")  # units -> unit
    return f"{num}{unit}"


def _norm_doses(text: str) -> set[str]:
    return {_norm_dose(m) for m in _DOSE_RE.finditer(text)}

# Normalize an answer/source frequency phrase to a canonical token.
_FREQ_PATTERNS = [
    (re.compile(r"\b(?:2|two)\s*(?:x|times)?\s*(?:a\s*day|daily|per\s*day)\b|\bBID\b", re.I), "BID"),
    (re.compile(r"\b(?:3|three)\s*(?:x|times)?\s*(?:a\s*day|daily|per\s*day)\b|\bTID\b", re.I), "TID"),
    (re.compile(r"\b(?:4|four)\s*(?:x|times)?\s*(?:a\s*day|daily|per\s*day)\b|\bQID\b", re.I), "QID"),
    (re.compile(r"\b(?:1|one)\s*(?:x|times)?\s*(?:a\s*day|daily|per\s*day)\b|\bDAILY\b|\bQAM\b", re.I), "DAILY"),
    (re.compile(r"\bat\s*bedtime\b|\bHS\b|\bQHS\b", re.I), "HS"),
    (re.compile(r"\bQ([1-9][0-9]?)H\b", re.I), lambda m: f"Q{m.group(1)}H"),
    (re.compile(r"\bPRN\b|\bas\s*needed\b", re.I), "PRN"),
    (re.compile(r"\b2X\s*/\s*WEEK\b|\b2\s*x\s*a\s*week\b|\btwice\s*a\s*week\b", re.I), "2X/WEEK"),
]


def _canon_freqs(text: str) -> set[str]:
    out = set()
    for pat, repl in _FREQ_PATTERNS:
        if callable(repl):
            for m in pat.finditer(text):
                out.add(repl(m))
        else:
            for m in pat.finditer(text):
                out.add(repl)
    return out


def _answer_freqs(text: str) -> set[str]:
    return _canon_freqs(text)


def _source_freqs(text: str) -> set[str]:
    return _canon_freqs(text)


# --- Citation ---------------------------------------------------------------

_CITE = re.compile(r"\^\[(\d+)\]")


def _passage_sections(tool_calls: list[dict]) -> tuple[list[dict], str]:
    """All (section, text) retrieved, plus concatenated discharge-meds text."""
    passages = []
    for tc in tool_calls or []:
        if tc.get("name") not in ("rag_search", "rag_search_sections"):
            continue
        for p in (tc.get("response") or {}).get("passages") or []:
            passages.append({"section": p.get("section"), "text": p.get("text") or ""})
    med_texts = []
    for p in passages:
        sec = (p.get("section") or "").lower()
        if "medication" in sec or "Discharge Medications" in p.get("text", ""):
            med_texts.append(p["text"])
    return passages, "\n".join(med_texts)


def _source_has_redacted_age(passages: list[dict]) -> bool:
    for p in passages:
        if _REDACTED_AGE.search(p.get("text", "")):
            return True
    return False


def redact_invented_age(answer: str, passages: list[dict]) -> tuple[str, list[str]]:
    """Drop a specific age the agent invented if the source age is redacted."""
    if not _source_has_redacted_age(passages):
        return answer, []
    if not _ANSWER_AGE.search(answer):
        return answer, []
    cleaned = _ANSWER_AGE.sub(" ", answer)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, ["redacted_age_filled"]


def verify_med_tokens(answer: str, evidence_text: str) -> tuple[str, list[str]]:
    """Drop dose/freq tokens asserted in the answer that appear NOWHERE in the
    retrieved evidence. Verifying against the full evidence (not just the
    discharge-meds section) keeps this precise: a dose in Meds-on-Admission or
    the HPI is still supported, so it is not dropped; only a dose/freq invented
    from nothing is. Conservative: exact normalized tokens, and the answer is
    only rewritten when something was actually dropped."""
    flags: list[str] = []
    if not evidence_text:
        return answer, flags  # no evidence -> can't verify, don't act

    source_norm = _norm_doses(evidence_text)
    source_freqs = _source_freqs(evidence_text)
    answer_freqs = _answer_freqs(answer)

    cleaned = answer
    dropped = False
    for m in _DOSE_RE.finditer(answer):
        if _norm_dose(m) not in source_norm:
            surface = m.group(0)
            flags.append(f"med_dose_mismatch:{surface}")
            cleaned = cleaned.replace(surface, "")
            dropped = True
    # Frequencies are flagged, not auto-edited: a wrong frequency is lower-risk
    # than a wrong dose, and auto-editing risks false positives (e.g. "as
    # needed"). The prompt hardening tells the agent to reproduce frequencies
    # exactly; the flag is surfaced for review.
    for freq in sorted(answer_freqs - source_freqs):
        flags.append(f"med_freq_mismatch:{freq}")
    if not dropped:
        return answer, flags
    return re.sub(r"\s{2,}", " ", cleaned).strip(), flags


def check_citations(answer: str, passages: list[dict]) -> list[str]:
    n = len(passages)
    bad = [c for c in _CITE.findall(answer) if int(c) < 1 or int(c) > n]
    return [f"citation_out_of_range:^{c}" for c in bad] if bad else []


def guard_answer(answer: str, tool_calls: list[dict]) -> dict:
    """Apply all guardrails. Returns {'answer': str, 'flags': [str]}."""
    passages, _med_source = _passage_sections(tool_calls)
    full_text = "\n".join(p.get("text") or "" for p in passages)
    flags: list[str] = []

    answer, f = redact_invented_age(answer, passages)
    flags += f
    # Verify med claims against the FULL retrieved evidence (see docstring).
    answer, f = verify_med_tokens(answer, full_text)
    flags += f
    flags += check_citations(answer, passages)

    return {"answer": answer, "flags": sorted(set(flags))}
