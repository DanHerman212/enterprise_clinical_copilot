"""Deterministic post-hoc guardrails for the agent's free-text answer.

The LLM proposes; this module disposes. Pure functions of (answer, evidence),
so they are testable offline and replayable over saved traces (P4 dry-run).

Guardrails (P3 root-cause -> targeted):
  1. REDACTED-FIELD GUARD  — never fill a value MIMIC redacted to '___' (the
     main remaining class: invented patient age/dose). If the source redacts
     the age and the answer states a specific age, the age is dropped.
     Invented calendar dates are flagged (ECC-35).
  2. MEDICATION VERIFIER   — every dose+unit and frequency the answer asserts
     must appear in the retrieved Discharge Medications text. A dose/freq not
     found there is dropped from the answer (an unverifiable medical claim is
     safer absent) and flagged. This catches med dose/freq errors and the
     admission-meds-conflated-as-discharge failure.
  3. CITATION RANGE        — every ^[n] must point at a retrieved passage;
     out-of-range markers are stripped, not just flagged (ECC-14).
  4. RISK-NUMBER GUARD     — every risk-shaped number must match a
     `predict_readmission` response; a fabricated "risk is 0.14" (or any risk
     number with no predict call at all) is stripped + flagged (ECC-04).

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
# The number allows a thousands separator ("2,000 mcg") so it normalizes to
# the same key as the answer's "2000 mcg" — a comma used to break the match and
# made the verifier drop a CORRECT dose (regression on 24592634/summarize).
_DOSE_RE = re.compile(
    r"(?<![\d\-])(\d+(?:,\d{3})*(?:\.\d+)?)\s*(mg|mcg|g|mEq|units?|mL|%)\b(?![-])",
    re.IGNORECASE,
)


def _norm_dose(m: re.Match) -> str:
    """Normalize a dose match to a comparable key (handles "10 units" vs
    "10units", "300  mg" vs "300 mg", "1000 units" vs "1000 UNIT", and
    "2,000 mcg" vs "2000 mcg")."""
    num, unit = m.group(1), m.group(2).lower().rstrip("s")  # units -> unit
    return f"{num.replace(',', '')}{unit}"


def _norm_doses(text: str) -> set[str]:
    return {_norm_dose(m) for m in _DOSE_RE.finditer(text)}

# Normalize an answer/source frequency phrase to a canonical token.
#
# P3.2 (2026-08-18): the swaps the judge flagged in the golden re-run were
# written as "once daily"/"twice daily" (metoprolol "once daily" vs section
# "PO BID"; Bupropion "twice daily" vs "PO QAM"), but the canonicalizer only
# matched "1/2 times a day"/"BID" forms, so the guardrail could not see them.
# Added the prose forms once/twice a day|daily, every day, and QD.
_FREQ_PATTERNS = [
    (re.compile(r"\b(?:2|two)\s*(?:x|times)?\s*(?:a\s*day|daily|per\s*day)\b"
                r"|\btwice\s*(?:a\s*day|daily)\b|\bBID\b", re.I), "BID"),
    (re.compile(r"\b(?:3|three)\s*(?:x|times)?\s*(?:a\s*day|daily|per\s*day)\b|\bTID\b", re.I), "TID"),
    (re.compile(r"\b(?:4|four)\s*(?:x|times)?\s*(?:a\s*day|daily|per\s*day)\b|\bQID\b", re.I), "QID"),
    (re.compile(r"\b(?:1|one)\s*(?:x|times)?\s*(?:a\s*day|daily|per\s*day)\b"
                r"|\bonce\s*(?:a\s*day|daily)\b|\bevery\s*day\b|\bDAILY\b|\bQD\b|\bQAM\b", re.I), "DAILY"),
    (re.compile(r"\bat\s*bedtime\b|\bHS\b|\bQHS\b", re.I), "HS"),
    (re.compile(r"\bQ([1-9][0-9]?)H\b", re.I), lambda m: f"Q{m.group(1)}H"),
    (re.compile(r"\bPRN\b|\bas\s*needed\b", re.I), "PRN"),
    (re.compile(r"\b2X\s*/\s*WEEK\b|\b2\s*x\s*a\s*week\b|\btwice\s*a\s*week\b", re.I), "2X/WEEK"),
]


def _canon_freqs(text: str) -> set[str]:
    out: set[str] = set()
    work = text
    for pat, repl in _FREQ_PATTERNS:
        ms = list(pat.finditer(work))
        if not ms:
            continue
        for m in ms:
            out.add(repl(m) if callable(repl) else repl)
        # Blank each consumed span so a lower-priority pattern cannot re-match
        # inside a compound already canonicalized. Without this, "twice daily"
        # yields BOTH BID (via the BID pattern) and DAILY (the bare \bDAILY\b
        # branch re-matching the "daily" inside it) — a spurious {BID, DAILY}
        # that made the per-med verifier drop a correct "daily" and flag
        # med_freq_permed_mismatch:*:DAILY on passing answers (regression
        # caught by the P4 dry-run, 2026-08-18).
        chars = list(work)
        for m in ms:
            for i in range(m.start(), m.end()):
                chars[i] = " "
        work = "".join(chars)
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
        # Per-med verification needs ONLY the discharge_medications section.
        # Including medications_on_admission (or any passage whose body merely
        # mentions "Discharge Medications") concatenates whole notes and makes
        # _section_med_entries parse HPI/PMH numbered lists into junk entries
        # (observed: 97 entries named 'nausea', 'htn', … in a 41k-char blob).
        if sec == "discharge_medications":
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

    spans: list[tuple[int, int]] = []
    for m in _DOSE_RE.finditer(answer):
        if _norm_dose(m) not in source_norm:
            flags.append(f"med_dose_mismatch:{m.group(0)}")
            spans.append((m.start(), m.end()))
    # Frequencies are flagged, not auto-edited: a wrong frequency is lower-risk
    # than a wrong dose, and auto-editing risks false positives (e.g. "as
    # needed"). The prompt hardening tells the agent to reproduce frequencies
    # exactly; the flag is surfaced for review.
    for freq in sorted(answer_freqs - source_freqs):
        flags.append(f"med_freq_mismatch:{freq}")
    if not spans:
        return answer, flags
    # Remove by match span, never `str.replace`: a plain substring replace
    # ignores the regex lookarounds, so dropping an unsupported "5 mg" also
    # destroyed a supported "2.5 mg" (-> "2.") — the guardrail corrupting a
    # correct dose (ECC-11).
    cleaned = _remove_spans(answer, spans)
    # Collapse only intra-line space runs — never across newlines. The earlier
    # `re.sub(r"\s{2,}", ...)` also collapsed `\n\n` paragraph breaks and
    # markdown structure, which modified a passing summarize answer after a
    # dose drop (regression observed on 24592634/summarize).
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip(), flags


def check_citations(answer: str, passages: list[dict]) -> tuple[str, list[str]]:
    """Strip + flag ^[n] markers that point at no retrieved passage (ECC-14).

    Leaving them in rendered as citations to nothing — with zero passages an
    answer containing ^[1] was served intact.
    """
    n = len(passages)
    flags: list[str] = []
    spans: list[tuple[int, int]] = []
    for m in _CITE.finditer(answer):
        c = int(m.group(1))
        if c < 1 or c > n:
            flags.append(f"citation_out_of_range:^{c}")
            spans.append((m.start(), m.end()))
    if not spans:
        return answer, flags
    cleaned = _remove_spans(answer, spans)
    return re.sub(r" +([,.;:])", r"\1", cleaned).strip(), flags


# --- Risk-number guard (ECC-04) ----------------------------------------------
#
# The failure this closes: Gemini can produce a confident "the risk is 0.14"
# with zero tool calls, and every other guardrail is inert on empty evidence —
# a fabricated number served verbatim with HTTP 200. Deterministic backstop:
# every risk-shaped number in the answer must match (to its own stated
# precision) a probability or threshold actually returned by a successful
# `predict_readmission` call, or it is stripped and flagged.

_RISK_DECIMAL = re.compile(r"\b0\.\d{2,6}\b")
_RISK_PERCENT = re.compile(r"\b(\d{1,2}(?:\.\d{1,2})?)\s*%")
# `%` gets no trailing \b (a word boundary needs a word char after it, and
# "%" is followed by a space) — a decimal followed by % is the percent
# branch's job, not a fabricated probability.
_UNIT_TAIL = re.compile(r"\s*(?:%|(?:mg|mcg|g|mEq|units?|mL)\b)", re.IGNORECASE)


def _predict_values(tool_calls: list[dict]) -> list[float]:
    values: list[float] = []
    for tc in tool_calls or []:
        if tc.get("name") != "predict_readmission":
            continue
        resp = tc.get("response") or {}
        if not isinstance(resp, dict) or resp.get("error"):
            continue
        for key in ("probability", "threshold"):
            v = resp.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values.append(float(v))
    return values


def verify_risk_numbers(
    answer: str, tool_calls: list[dict], evidence_text: str = ""
) -> tuple[str, list[str]]:
    """Strip risk-shaped numbers no `predict_readmission` response supports.

    Tolerance is half a unit in the number's own last stated decimal place, so
    "0.13" honestly truncates 0.131398 but a fabricated "0.14" fails. Skipped:
    numbers followed by a dose unit, and numbers quoted verbatim from the
    retrieved passages (lab values like "creatinine 0.9") — those are med/note
    facts, owned by the other guardrails.
    """
    flags: list[str] = []
    values = _predict_values(tool_calls)
    spans: list[tuple[int, int]] = []

    def supported(x: float, tol: float) -> bool:
        return any(abs(x - v) <= tol for v in values)

    for m in _RISK_DECIMAL.finditer(answer):
        token = m.group(0)
        if _UNIT_TAIL.match(answer, m.end()):
            continue
        if evidence_text and token in evidence_text:
            continue
        decimals = len(token.split(".")[1])
        if not supported(float(token), 0.5 * 10 ** -decimals):
            flags.append(f"risk_number_unsupported:{token}")
            spans.append((m.start(), m.end()))
    for m in _RISK_PERCENT.finditer(answer):
        num = m.group(1)
        # Quoted verbatim from the notes — a med concentration ("0.05% cream")
        # or an O2 sat — is a note fact, not a risk claim.
        if evidence_text and (f"{num}%" in evidence_text or f"{num} %" in evidence_text):
            continue
        decimals = len(num.split(".")[1]) if "." in num else 0
        if not supported(float(num) / 100.0, 0.5 * 10 ** -decimals / 100.0):
            flags.append(f"risk_number_unsupported:{m.group(0)}")
            spans.append((m.start(), m.end()))
    if not spans:
        return answer, flags
    cleaned = _remove_spans(answer, spans)
    return re.sub(r" +([,.;:])", r"\1", cleaned).strip(), flags


# --- Invented calendar dates (ECC-35) ----------------------------------------

_ANSWER_DATE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
)


def flag_invented_dates(answer: str, passages: list[dict]) -> list[str]:
    """Flag concrete calendar dates the source cannot have supplied (ECC-35).

    MIMIC redacts dates to `___`; when the retrieved passages carry redactions
    and the answer states a date that appears in no passage, it was invented.
    Flag-only (no strip): dates are lower-risk than doses or risk numbers, and
    synthetic notes may legitimately contain dates — the verbatim check keeps
    those unflagged.
    """
    if not any("___" in (p.get("text") or "") for p in passages):
        return []
    source = "\n".join(p.get("text") or "" for p in passages)
    return [
        f"redacted_date_filled:{m.group(0)}"
        for m in _ANSWER_DATE.finditer(answer)
        if m.group(0) not in source
    ]


# --- Per-med frequency verification (P3.2, 2026-08-18) ----------------------
#
# The token-GLOBAL freq check cannot catch a per-med swap: answer says
# "metoprolol tartrate … once daily" but the med's own discharge entry says
# "PO BID", and "once daily"/DAILY exists elsewhere in the note, so the global
# check sees DAILY as "supported". This guardrail verifies each med's freq
# against THAT MED's entry in the discharge_medications section, matching by
# name AND dose (a med can appear at multiple doses, e.g. Levetiracetam 500 mg
# QAM + 1000 mg HS — dose disambiguates). A freq that contradicts the med's own
# entry is DROPPED (a missing freq is safer than a wrong one) and flagged.

_ENTRY_START = re.compile(r"(?m)^\s*\d+[\.\)]\s+")


def _med_name(chunk: str) -> str:
    """Leading med-name tokens up to the first digit ('metoprolol tartrate 25
    mg …' -> 'metoprolol tartrate'). Lowercased, punctuation stripped."""
    m = re.match(r"^[A-Za-z][A-Za-z\-']*(?:\s+[A-Za-z][A-Za-z\-']*)*", chunk)
    if not m:
        return ""
    return re.sub(r"[^a-z]+", " ", m.group(0).lower()).strip()


def _section_med_entries(med_text: str) -> list[dict]:
    """Per-med entries from the discharge_medications section: {name, doses, freqs}.

    A new entry starts only at a numbered line ("1. Drug …"). Continuation lines
    (RX detail, multi-line sigs, "as needed for …", following sub-headers) are
    APPENDED to the current entry — splitting them out produced junk entries whose
    freqs (e.g. Q24H/PRN from the med's own sig) then matched an answer chunk for a
    DIFFERENT med and dropped a correct freq. Trailing sections (Disposition /
    Diagnosis) bleed into the last entry's freqs, which only makes the check more
    permissive (safe direction)."""
    idx = med_text.find("Discharge Medications")
    body = med_text[idx:] if idx != -1 else med_text
    entries: list[dict] = []
    cur: list[str] = []
    for line in body.splitlines():
        if _ENTRY_START.match(line):
            if cur:
                entries.append(_make_entry(cur))
            cur = [line.strip()]
        elif line.strip():
            cur.append(line.strip())
    if cur:
        entries.append(_make_entry(cur))
    return [e for e in entries if e is not None]


def _make_entry(lines: list[str]) -> dict | None:
    """Build an entry dict from the lines of one numbered med entry."""
    text = " ".join(lines)
    first = lines[0] if lines else ""
    # The numbered prefix ("1. metoprolol tartrate …") must be stripped before
    # name extraction — _med_name requires the first char to be a letter.
    first = _ENTRY_START.sub("", first).strip()
    name = _med_name(first)
    if not name:
        return None
    return {
        "name": name,
        "doses": _norm_doses(text),
        "freqs": _canon_freqs(text),
    }


def _match_entry(chunk: str, doses: set[str], entries: list[dict]) -> dict | None:
    """Best section entry for an answer med-chunk.

    Match by finding which section entry's med name appears in the chunk
    (the answer embeds prose like 'the patient was prescribed metoprolol
    tartrate 25 mg…', so we look the clean section name up inside the chunk
    rather than trying to parse a name out of the prose). Among name matches,
    prefer the entry whose dose intersects the chunk's dose — a med can appear
    at multiple doses (Levetiracetam 500 mg QAM + 1000 mg HS).

    CONSERVATIVE: returns None when the chunk contains MORE THAN ONE distinct
    med name (e.g. 'Diltiazem 60 mg PO TID and amLODIPine 5 mg PO DAILY' —
    the answer glued two meds with 'and' and no separator). Verifying a
    multi-med chunk would misattribute one med's freq to the other and could
    DROP A CORRECT FREQ, so we skip it entirely. Also requires a real name
    (>= 4 chars) to avoid junk matches."""
    cl = chunk.lower()
    name_matches = [
        e for e in entries
        if e["name"] and len(e["name"]) >= 4 and e["name"] in cl
    ]
    if not name_matches:
        return None
    # Multi-med chunk -> not safe to act on (see docstring).
    distinct = {e["name"] for e in name_matches}
    if len(distinct) > 1:
        return None
    if doses:
        for e in name_matches:
            if e["doses"] & doses:
                return e
    return next((e for e in name_matches if e["freqs"]), None)


def _freq_token(m: re.Match, repl) -> str:
    return repl(m) if callable(repl) else repl


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Remove [start,end) slices (sorted, non-overlapping assumed)."""
    out = []
    prev = 0
    for s, e in sorted(spans):
        if s >= prev:
            out.append(text[prev:s])
            prev = e
    out.append(text[prev:])
    return "".join(out)


def verify_med_freqs_per_med(answer: str, med_text: str) -> tuple[str, list[str]]:
    """Drop a freq that contradicts the SAME med's discharge entry (per-med,
    name+dose matched). Flags f"med_freq_permed_mismatch:{name}:{token}"."""
    flags: list[str] = []
    if not med_text or not answer:
        return answer, flags
    entries = _section_med_entries(med_text)
    if not entries:
        return answer, flags

    # Answer's Discharge Medications region (or the whole answer if none).
    idx = answer.lower().find("discharge medication")
    region = answer[idx:] if idx != -1 else answer
    for hdr in ("discharge instructions", "brief hospital course",
                "discharge diagnosis", "discharge disposition"):
        j = region.lower().find(hdr)
        if j > 0:
            region = region[:j]
            break

    spans_to_remove: list[tuple[int, int]] = []
    base = idx if idx != -1 else 0
    # Chunk on newlines, commas AND semicolons: the meds prompt lists meds one
    # per line ("docusate sodium 100 mg … PO BID.\nsenna …"), while summarize
    # embeds them as comma- or semicolon-separated prose. Splitting on commas
    # alone turns a newline or semicolon list into one giant chunk and
    # misattributes every med's freq to the first name.
    for m in re.finditer(r"[^\n,;]+", region):
        raw = m.group(0)
        chunk = raw.strip()
        if not chunk or not any(c.isdigit() for c in chunk):
            continue
        chunk_doses = _norm_doses(chunk)
        # CONSERVATIVE: only act on a clean single-med chunk.
        #  - exactly ONE dose token: a chunk naming several meds ("… amLODIPine
        #    5 mg PO DAILY" glued on) or a held med list carries several doses
        #    and must not have one med's freq attributed to another.
        #  - no held/stopped/discontinued marker: held meds (Alyacen,
        #    GlipiZIDE, HYDROcodone) are NOT in the discharge section, so their
        #    chunk matches a section med by accident and their freq gets dropped.
        if len(chunk_doses) != 1:
            continue
        if re.search(r"\b(?:held|stop(?:ped)?|discontin|not\s*restart)\b", chunk, re.I):
            continue
        chunk_freqs = _canon_freqs(chunk)
        if not chunk_freqs:
            continue
        entry = _match_entry(chunk, chunk_doses, entries)
        if entry is None or not entry["freqs"]:
            continue
        bad = chunk_freqs - entry["freqs"]
        if not bad:
            continue
        for tok in sorted(bad):
            flags.append(f"med_freq_permed_mismatch:{entry['name']}:{tok}")
        # Remove the wrong freq phrases inside this chunk. Spans must be
        # relative to the RAW (unstripped) match so offsets line up.
        for pat, repl in _FREQ_PATTERNS:
            for fm in pat.finditer(raw):
                if _freq_token(fm, repl) in bad:
                    spans_to_remove.append((base + m.start() + fm.start(),
                                            base + m.start() + fm.end()))
    if not spans_to_remove:
        return answer, flags
    cleaned = _remove_spans(answer, spans_to_remove)
    # Fix only the "word ," artifacts left by a removal. Deliberately do NOT
    # collapse runs of spaces globally — that destroys markdown bullet
    # indentation ("*   Acetaminophen") and is a regression on passing answers.
    cleaned = re.sub(r" +([,.;:])", r"\1", cleaned)
    return cleaned.strip(), flags


def guard_answer(answer: str, tool_calls: list[dict]) -> dict:
    """Apply all guardrails. Returns {'answer': str, 'flags': [str]}."""
    passages, med_source = _passage_sections(tool_calls)
    full_text = "\n".join(p.get("text") or "" for p in passages)
    flags: list[str] = []

    # Risk numbers first — the one guard that must act even with zero evidence.
    answer, f = verify_risk_numbers(answer, tool_calls, full_text)
    flags += f
    answer, f = redact_invented_age(answer, passages)
    flags += f
    # Verify med claims against the FULL retrieved evidence (see docstring).
    answer, f = verify_med_tokens(answer, full_text)
    flags += f
    # Per-med freq verification against the med's OWN discharge entry (P3.2).
    answer, f = verify_med_freqs_per_med(answer, med_source)
    flags += f
    flags += flag_invented_dates(answer, passages)
    answer, f = check_citations(answer, passages)
    flags += f

    return {"answer": answer, "flags": sorted(set(flags))}
