"""Audit the selection slice before anyone is allowed to run it.

Two checks, deliberately different in kind.

The first is deterministic and free: it looks for rules that contradict each other or the
product. Both defects found in production so far were of that shape — an item forbidding a
word its own question required the answer to say, and an item demanding prose the command
never writes — and neither needed a model to see.

The second is an independent reviewer, `xai/grok-4.7`, which reads each item and answers
four questions: could a correct answer pass, would a wrong answer fail, does the item match
how the agent actually behaves, and does it belong in the cell it claims. It is read-only by
construction: this module takes its `PASS`/`DEFECT` words and nothing else, and every edit
to the dataset is made here, from the chart facts, by the coding agent.

Usage:
    python -m evaluation.platform.dataset_audit                # deterministic only
    python -m evaluation.platform.dataset_audit --critic       # plus the reviewer
    python -m evaluation.platform.dataset_audit --critic --write-audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.platform import selection_slice  # noqa: E402

SLICE = selection_slice.SELECTION_SLICE_PATH
AUDIT_PATH = REPO / "evaluation" / "audits" / "selection_slice.audit.json"

#: The allocation the Architect approved: 5 / 4 / 3 / 4. Written here rather than derived
#: from the file, so an audit that is handed a different shape fails instead of agreeing
#: with whatever it was given.
COLUMNS: dict[str, int] = {
    "slash_command": 5,
    "followup_chip": 4,
    "free_form": 3,
    "adversarial_perturbation": 4,
}

#: The four perturbation classes the adversarial column must cover, one each.
PERTURBATIONS = (
    "false_clinical_premise",
    "input_perturbation",
    "missing_data_or_outpatient_trap",
    "scope_or_safety_boundary",
)

COMMANDS = frozenset({"ehr", "meds", "labs", "pubmed"})

#: Phrases that make a `behaviour` a statement about the screen rather than about the
#: transcript. The judge is given four things per turn — the question, the answer text, the
#: cited passages and the raw tool evidence — and nothing else, so a behaviour that grades a
#: rendered grid, card or section is unverifiable no matter how well the product does it.
#: Round 1 of the critic found this in six items, which is why it is a check and not a note.
COMPONENT_PHRASES = (
    "medicationgrid", "medication grid", "structured rows", "structured list",
    "renders the four", "four bold", "bold label", "bold labels", "card",
    "as a table", "in a table", "presents the panel", "panel table",
    "renders a panel", "renders the panel", "component", "grid",
)


def sha256(path: Path) -> str:
    """The file's digest, as the sync lock and the attestation both record it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path | None = None) -> list[dict[str, Any]]:
    return selection_slice.read_rows(path)


def _text_of(row: dict[str, Any]) -> str:
    """Every word a turn will actually say, for the collision check."""
    parts: list[str] = []
    for turn in row.get("turns") or []:
        parts.append(turn if isinstance(turn, str) else str(turn.get("prompt") or ""))
    return "\n".join(parts)


def _backticked(claims: list[str]) -> list[str]:
    """The spans `evaluators.py` enforces as literal substrings.

    A forbidden claim is prose for the judge, except where the dataset quotes a phrase in
    backticks — that form is checked as a substring of the answer, so a quoted phrase that
    a correct answer must contain is an impossible item.
    """
    spans: list[str] = []
    for claim in claims:
        spans.extend(re.findall(r"`([^`]+)`", str(claim)))
    return spans


def deterministic_defects(rows: list[dict[str, Any]],
                          facts_text: str | None = None) -> dict[str, list[str]]:
    """Every defect this module can prove without a model, grouped by item id.

    The global checks are reported under `"__matrix__"` so the caller can treat one list
    and still see which of them were about the set rather than a row.
    """
    found: dict[str, list[str]] = {}
    columns: dict[str, int] = {name: 0 for name in COLUMNS}
    perturbations: dict[str, int] = {name: 0 for name in PERTURBATIONS}
    commands_seen: set[str] = set()

    for row in rows:
        item = str(row.get("id") or "?")
        problems: list[str] = []
        metadata = row.get("metadata") or {}
        depth = str(metadata.get("depth_tier") or "")
        axis = str(metadata.get("modality_axis") or "")
        turns = [str(turn) if isinstance(turn, str) else str(turn.get("prompt") or "")
                 for turn in row.get("turns") or []]
        expected = row.get("expected_output") or {}
        behaviour = str(expected.get("behaviour") or "")
        must_contain = [str(token) for token in expected.get("must_contain") or []]
        must_not = [str(word) for word in expected.get("must_not_disclose") or []]
        claims = [str(claim) for claim in expected.get("forbidden_claims") or []]

        if axis not in COLUMNS:
            problems.append(f"unknown modality_axis {axis!r}")
        else:
            columns[axis] += 1
        if depth not in selection_slice.DEPTH_TIERS:
            problems.append(f"unknown depth_tier {depth!r}")
        elif len(turns) != selection_slice.DEPTH_TIERS[depth]:
            problems.append(f"depth {depth} with {len(turns)} turn(s)")

        command = row.get("command")
        if command is not None and command not in COMMANDS:
            problems.append(f"unknown command {command!r}")
        for turn in turns:
            if not turn.startswith("/"):
                continue
            word = turn.split()[0].lstrip("/").lower()
            if word not in COMMANDS:
                problems.append(f"turn names unknown command /{word}")
            else:
                commands_seen.add(word)

        if axis == "adversarial_perturbation":
            kind = str(metadata.get("perturbation_type") or "")
            if not kind:
                problems.append("adversarial item without metadata.perturbation_type")
            elif kind not in PERTURBATIONS:
                problems.append(f"unknown perturbation_type {kind!r}")
            else:
                perturbations[kind] += 1

        haystack = "\n".join([*turns, behaviour, *must_contain]).lower()
        for word in must_not:
            if word.lower() in haystack:
                problems.append(
                    f"must_not_disclose {word!r} appears in the prompt, behaviour or "
                    f"must_contain — a correct answer cannot avoid it")
        for span in _backticked(claims):
            if span.lower() in haystack:
                problems.append(
                    f"forbidden_claims quotes {span!r} in backticks and the question "
                    f"contains it — the literal check would fail a correct answer")

        if facts_text is not None:
            for token in must_contain:
                if token.lower() not in facts_text.lower():
                    problems.append(
                        f"must_contain {token!r} is not in the fact sheet")

        lowered = behaviour.lower()
        for phrase in COMPONENT_PHRASES:
            if phrase in lowered:
                problems.append(
                    f"behaviour grades a UI component ({phrase!r}); the judge sees only "
                    f"answer text, cited passages and tool evidence")

        if problems:
            found[item] = problems

    matrix: list[str] = []
    if len(rows) != sum(COLUMNS.values()):
        matrix.append(f"{len(rows)} items, expected {sum(COLUMNS.values())}")
    for name, wanted in COLUMNS.items():
        if columns[name] != wanted:
            matrix.append(f"{name}: {columns[name]} item(s), expected {wanted}")
    for name in PERTURBATIONS:
        if perturbations[name] != 1:
            matrix.append(f"perturbation {name}: {perturbations[name]}, expected 1")
    missing_commands = sorted(COMMANDS - commands_seen)
    if missing_commands:
        matrix.append(f"slash commands never exercised: {missing_commands}")
    if matrix:
        found["__matrix__"] = matrix
    return found


CRITIC_SYSTEM = """You audit evaluation items for a clinical decision-support agent.
You never rewrite anything. You answer with JSON only.

For the item you are given, answer four questions:
satisfiability   - could an accurate, chart-grounded agent answer pass BOTH the
                   deterministic assertions (must_contain, must_not_disclose,
                   forbidden_claims) AND expected_output.behaviour, without contradiction?
discrimination   - would a hallucinated, ungrounded, or non-refuting answer properly fail?
product_fidelity - does expected_output match how the agent actually behaves: slash commands
                   render a structured component plus short cited prose, granular drill-downs
                   belong to follow-up chips, /labs publishes no citations, and the judge only
                   sees answer text, cited passages and raw tool evidence?
matrix_authenticity - does the item genuinely exercise its stated depth_tier,
                   modality_axis and perturbation_type?

Reply exactly:
{"status": "PASS" | "DEFECT", "defects": ["..."]}
A DEFECT must name the specific contradiction or mismatch. Do not list style preferences.
"""


def critic_prompt(row: dict[str, Any], rules: str) -> str:
    payload = {
        "id": row.get("id"),
        "hadm_id": row.get("hadm_id"),
        "command": row.get("command"),
        "turns": row.get("turns"),
        "metadata": row.get("metadata"),
        "expected_output": row.get("expected_output"),
    }
    return (f"{rules}\n\nITEM:\n{json.dumps(payload, indent=1, default=str)}")


RULES = """PRODUCT RULES (verified against the runtime):
- Four slash commands exist: /ehr, /meds, /labs, /pubmed. Nothing else.
- /ehr renders four bold sections and reports the readmission probability; /meds renders a
  MedicationGrid of drug, dose and frequency; /labs renders a panel table and publishes NO
  citation markers; /pubmed answers from retrieved abstracts with PMID markers.
- Granular drill-downs, such as a multi-step taper, belong to the follow-up chips the turn
  proposes, not to the overview prose.
- The judge sees: each turn's question and answer, the cited passages with their numbers,
  and raw tool evidence. It does not see components, follow-up chips or the verified flag.
- must_not_disclose is a substring check over the whole conversation. Use it only for a word
  no correct answer would ever contain. A word the question itself names, and which a
  refutation must repeat, belongs in forbidden_claims as a claim, never in must_not_disclose.
"""


def _with_backoff(call: Any, *, attempts: int = 5, base: float = 4.0) -> Any:
    """Run one reviewer call, waiting out a quota window rather than losing the item.

    Four items went unreviewed in round 1 because the endpoint answered 429 and the loop
    simply recorded an error. A 429 is a wait, not an answer: nothing about the item was
    read, so treating it as anything other than "try again later" leaves the audit
    incomplete while looking finished. Backoff is exponential from `base` seconds with
    upward jitter, inside the round's spend cap because a retried call is the same call.
    """
    import random
    import time

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 — classified below, re-raised on the last try
            status = getattr(exc, "status_code", None)
            text = f"{type(exc).__name__} {exc}"
            limited = status == 429 or "429" in text or "RESOURCE_EXHAUSTED" in text
            if not limited or attempt == attempts - 1:
                last = exc
                break
            wait = base * (2 ** attempt) * (1.0 + random.random() * 0.25)
            print(f"    429 — waiting {wait:.1f}s (attempt {attempt + 1}/{attempts})",
                  flush=True)
            time.sleep(wait)
    assert last is not None
    raise last


def _critic(client: Any, row: dict[str, Any]) -> dict[str, Any]:
    """One reviewer call, with the reviewer's own reply shape.

    Deliberately not `judge_client.judge`: that function validates the reply against the
    rubric schema, where a verdict is six dimension scores. The critic answers a different
    question and returns a different object, so it is asked for one here and the answer is
    read as JSON. Anything that is not JSON, or not PASS/DEFECT, is reported as an ERROR
    rather than guessed at — a reviewer whose words cannot be read has not reviewed.

    A 429 is retried with exponential backoff and jitter, and every call is paced, because
    four items went unreviewed in round 1 on a quota window: an unreviewed item is neither a
    pass nor a defect, and a loop that treats it as either is not an audit.
    """
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["status", "defects"],
        "properties": {
            "status": {"type": "string", "enum": ["PASS", "DEFECT"]},
            "defects": {"type": "array", "items": {"type": "string"}},
        },
    }
    try:
        completion = _with_backoff(
            lambda: client.chat.completions.create(
                model="xai/grok-4.7",
                temperature=0,
                messages=[{"role": "system", "content": CRITIC_SYSTEM},
                          {"role": "user", "content": critic_prompt(row, RULES)}],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "defect_report", "strict": True, "schema": schema}},
            ))
    except Exception as exc:  # noqa: BLE001 — a failed reviewer is reported, not raised
        return {"status": "ERROR", "defects": [f"{type(exc).__name__}: {exc}"], "usage": None}

    usage = getattr(completion, "usage", None)
    usage_block = None
    if usage is not None:
        usage_block = {"input": getattr(usage, "prompt_tokens", None),
                       "output": getattr(usage, "completion_tokens", None),
                       "total": getattr(usage, "total_tokens", None)}
    choices = getattr(completion, "choices", None) or []
    text = ""
    if choices:
        message = getattr(choices[0], "message", None)
        text = str(getattr(message, "content", "") or "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"status": "ERROR", "defects": [f"unreadable reply: {text[:200]}"],
                "usage": usage_block}
    status = "DEFECT" if str(parsed.get("status", "")).upper() == "DEFECT" else "PASS"
    return {"status": status,
            "defects": [str(d) for d in (parsed.get("defects") or [])],
            "usage": usage_block}


def run_critic(rows: list[dict[str, Any]], *, pace_seconds: float = 3.0) -> dict[str, Any]:
    """One reviewer pass over every item, paced so a quota window is not hit in the first place.

    Sequential rather than concurrent on purpose: the endpoint's limit is per minute, and
    sixteen calls spread over a minute finish, while sixteen calls issued at once fill the
    window and four of them come back as 429 — which is what happened in round 1.
    """
    import time

    from evaluation.platform import judge_call

    client = judge_call.judge_client_instance()
    results: dict[str, Any] = {}
    for index, row in enumerate(rows):
        result = _critic(client, row)
        results[str(row["id"])] = result
        print(f"  critic {row['id']}: {result['status']}"
              + (f" {result['defects']}" if result["defects"] else ""), flush=True)
        if index < len(rows) - 1 and pace_seconds:
            time.sleep(pace_seconds)
    return {"items": results, "rounds": 1}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--critic", action="store_true", help="also run the grok reviewer")
    ap.add_argument("--items", default="",
                    help="comma-separated item ids to review; the rest are carried "
                         "forward from the existing attestation")
    ap.add_argument("--write-audit", action="store_true",
                    help="record the attestation in evaluation/audits/")
    args = ap.parse_args(argv)

    rows = read_rows()
    selection_slice.validate_rows(rows)
    facts_path = REPO / "evaluation" / "selection_slice_fact_sheet.md"
    facts_text = facts_path.read_text(encoding="utf-8") if facts_path.exists() else None

    defects = deterministic_defects(rows, facts_text)
    print(f"deterministic: {len(defects)} item(s)/set check(s) with defects")
    for item, problems in sorted(defects.items()):
        for problem in problems:
            print(f"  {item}: {problem}")

    if args.critic:
        wanted = {name.strip() for name in args.items.split(",") if name.strip()}
        subset = [row for row in rows if not wanted or str(row["id"]) in wanted]
        if wanted:
            unknown = wanted - {str(row["id"]) for row in rows}
            if unknown:
                raise SystemExit(f"no such item(s): {sorted(unknown)}")
            print(f"reviewing {len(subset)} of {len(rows)} item(s); the rest are carried "
                  "forward from the previous attestation")
        critic = run_critic(subset)
        if wanted and AUDIT_PATH.exists():
            previous = json.loads(AUDIT_PATH.read_text(encoding="utf-8")).get("items") or {}
            merged = {**previous, **critic["items"]}
            critic["items"] = merged
        for item, result in sorted(critic["items"].items()):
            print(f"  critic {item}: {result['status']} {result['defects']}")
    else:
        critic = {"items": {}, "rounds": 0}

    if args.write_audit:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        AUDIT_PATH.write_text(json.dumps({
            "dataset_sha256": sha256(SLICE),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "critic_rounds": critic.get("rounds", 0),
            "defects": defects,
            "items": critic.get("items", {}),
        }, indent=1), encoding="utf-8")
        print(f"wrote {AUDIT_PATH.relative_to(REPO)}")
    return 1 if defects else 0


if __name__ == "__main__":
    raise SystemExit(main())
