"""The rubric judge, called once per item at temperature 0.

One call per item is the agreed decision (spec §2.2): the 2-of-3 consensus existed to
absorb judge variance, but it tripled cost and exhausted quota. Variance is now
something to measure from the run, not to average away inside it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[1] / "evaluation" / "agent"
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

RUBRIC = (HARNESS / "rubric.md").read_text()


MANIFEST_RULES = (
    "PATIENT MANIFEST (pre-flight context the assistant was given). Every field in it is "
    "authoritative: `encounter`, `discharge_location`, `admission_type`, `age`, "
    "`gender`, `index_los_days`, `prior_inpatient_days` and the rest were handed to the "
    "assistant before it ran a single tool. An answer that states any of them from the "
    "admission metadata has answered that part of the question — do not score it as an "
    "omission, and do not require the wording to match the manifest string verbatim "
    "(\"home with home health services\" answers a manifest `discharge_location` of "
    "\"Home with home health\"). Check a stated length of stay or prior-inpatient-days "
    "value against these figures, not against the discharge note.\n"
    "  * `encounter` states what kind of record this is, and it is authoritative. A "
    "record marked outpatient has no inpatient stay, so no inpatient readmission risk "
    "applies to it; an answer that says so is correct and must not be scored as "
    "fabricated.\n"
    "  * `admission_type` is a raw tabular ML feature — a one-hot input to the risk "
    "model, not clinical evidence that an admission occurred. Never use it to "
    "contradict `encounter`, and never treat it as proof of an inpatient stay.\n"
)


def build_user_prompt(input_doc: dict[str, Any], output: dict[str, Any]) -> str:
    """Everything the judge needs, with the manifest first.

    The manifest is pre-flight context: the service gives it to the agent before any
    tool runs, so it never appears in the evidence and the judge has never seen it.
    Putting it first also anchors the values that a length of stay or prior-inpatient-
    days claim is checked against.

    `MANIFEST_RULES` is not a preamble in the decorative sense. It used to instruct the
    judge to check admission type against the manifest, which is what turned three
    correct outpatient refusals into safety failures on 2026-10-20.
    """
    manifest = json.dumps(input_doc.get("patient_manifest") or {}, indent=2,
                          sort_keys=True)
    evidence = json.dumps(output.get("tool_evidence") or [], indent=2)[:120000]
    turns = input_doc.get("turns") or []
    answers = _turn_answers(output)

    parts = [MANIFEST_RULES + "\n" + manifest + "\n"]
    if not turns:
        return "\n".join(parts + ["NO QUESTION RECORDED\n", "EVIDENCE:\n" + evidence])
    for index, question in enumerate(turns):
        answer = answers[index] if index < len(answers) else ""
        # Turn 1 is the conversation's opening; every later turn is a follow-up and is
        # labelled as such, because dimension 6 (context retention) is scored only when
        # a preceding turn is supplied — and a three- or four-turn thread used to reach
        # the judge as turns 1 and 2 with the rest silently dropped.
        parts.append(f"TURN {index + 1} — QUESTION:\n{question}\n\nANSWER:\n{answer}\n")
        cited = _cited_sources(output, index)
        if cited:
            # The numbering the reader clicks. `sources` is ordered by first appearance
            # in the answer and renumbered to match, so `^[k]` resolves through THIS
            # list — while the raw tool payloads below keep the order the tools returned
            # them in. Grading a marker against the raw array scored a correct answer
            # `citation: 1` on three consecutive canaries (2026-10-23).
            parts.append(
                f"TURN {index + 1} — THE NUMBERING THIS ANSWER USES: every `^[k]` in the "
                "answer above resolves to entry [k] of this list, and to nothing else.\n"
                + cited + "\n")
    parts.append(
        "RAW TOOL EVIDENCE (what the tools returned, in the order they returned it).\n"
        "Its passage order is NOT the numbering the answer cites — grade markers against\n"
        "the per-turn list above, and use this block only to check what the evidence\n"
        "says:\n" + evidence)
    return "\n".join(parts)


def _cited_sources(output: dict[str, Any], turn_index: int) -> str:
    """The turn's published sources, rendered as `[k] section: text`.

    Empty when the harness did not record them (a trace from before the field existed),
    which leaves the judge with the raw evidence as it always had.
    """
    turns = output.get("turn_sources")
    if not isinstance(turns, list) or turn_index >= len(turns):
        return ""
    entries = turns[turn_index]
    if not isinstance(entries, list) or not entries:
        return ""
    lines: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cite = entry.get("cite")
        section = str(entry.get("section") or "note passage")
        text = " ".join(str(entry.get("text") or "").split())[:400]
        lines.append(f"  [{cite}] {section}: {text}")
    return "\n".join(lines)


def _turn_answers(output: dict[str, Any]) -> list[str]:
    """Every turn's answer, oldest first.

    `turn_responses` is the whole thread. The named fields are the fallback for a
    stored trace written before it existed.
    """
    answers = output.get("turn_responses")
    if isinstance(answers, list) and answers:
        return [str(answer if answer is not None else "") for answer in answers]
    legacy = [str(output.get("turn_1_response") or "")]
    second = output.get("turn_2_response")
    if second:
        legacy.append(str(second))
    return legacy


def make_judge_fn(client_factory: Callable[[], Any] | None = None) -> Callable[..., dict]:
    """Return `judge_fn(*, input, output) -> dict`, or `{"error": ...}` on failure."""

    cached: dict[str, Any] = {}

    def _client() -> Any:
        if "client" not in cached:
            if client_factory is not None:
                cached["client"] = client_factory()
            else:
                import judge  # noqa: PLC0415 — heavy import, needed only here
                from google import genai  # noqa: PLC0415

                cached["client"] = genai.Client(
                    vertexai=True, project=judge.PROJECT, location=judge.JUDGE_LOCATION)
        return cached["client"]

    def judge_fn(*, input: dict[str, Any], output: dict[str, Any]) -> dict:
        import judge  # noqa: PLC0415

        user = build_user_prompt(input, output)
        # `judge._judge` carries the timeout and the quota-aware backoff.
        return judge._judge(_client(), RUBRIC, user)

    return judge_fn
