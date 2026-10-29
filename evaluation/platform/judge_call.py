"""The rubric judge, called once per item at temperature 0.

One call per item is the agreed decision (spec §2.2): the 2-of-3 consensus existed to
absorb judge variance, but it tripled cost and exhausted quota. Variance is now
something to measure from the run, not to average away inside it.

Self-contained since 2026-10-29 (`00d8` Rows 5 and 6). It used to insert
`evaluation/agent` on `sys.path` and `import judge` inside a function, which made the
live harness depend on the retired collector, hid the dependency from every static scan,
and made `judge` impossible to import twice. The rubric moved to `evaluation/rubric.md`
— it is the contract both judges read, not a file owned by the collector — and the Vertex
call moved to `judge_client.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from evaluation.platform import judge_client

#: The canonical clinical grading rubric. Top-level, shared, and read once.
RUBRIC = (Path(__file__).resolve().parents[1] / "rubric.md").read_text()


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
            # The numbering the reader clicks. Sources are listed by their own number, so
            # `^[k]` resolves to the entry LABELLED [k] — not to entry k by position, and
            # not to anything in the raw evidence below. Numbers are no longer rewritten
            # to first-appearance order (`citations._renumber_citations`, 2026-10-28), so
            # the label and the number are now the same fact.
            parts.append(
                f"TURN {index + 1} — THE NUMBERING THIS ANSWER USES: every `^[k]` in the "
                "answer above resolves to the entry labelled [k] below, and to nothing "
                "else. A number with no entry here addresses no unit this turn published.\n"
                + cited + "\n")
    parts.append(
        "RAW TOOL EVIDENCE (what the tools returned, in the order they returned it).\n"
        "Each passage and each analyte carries the marker the answer must write for it:\n"
        "`\"cite\": \"^[10]\"` is the unit `^[10]` addresses. A number can be checked\n"
        "against this block directly, and a unit with no `cite` (an estimate, a medication\n"
        "list, a search result set) consumes none.\n" + evidence)
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
                from google import genai  # noqa: PLC0415 — heavy import, needed only here

                cached["client"] = genai.Client(
                    vertexai=True, project=judge_client.PROJECT,
                    location=judge_client.JUDGE_LOCATION)
        return cached["client"]

    def judge_fn(*, input: dict[str, Any], output: dict[str, Any]) -> dict:
        user = build_user_prompt(input, output)
        # `judge_client.judge` carries the ceiling and the quota-aware backoff, and
        # applies the item pass rule from `schemas.PASS_DIMENSIONS`.
        return judge_client.judge(_client(), RUBRIC, user)

    return judge_fn
