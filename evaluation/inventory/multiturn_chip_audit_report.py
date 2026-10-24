"""Render the §5.4 deliverable from two audit runs: before and after.

The audit itself (`multiturn_chip_audit.py`) writes JSON; this turns the pair into the
markdown the directive asks for — the answerability table with the columns it names,
and the three-turn progression that proves a clicked chip is consumed.

    ./.venv/bin/python evaluation/inventory/multiturn_chip_audit_report.py \
        --before evaluation/inventory/multiturn_chip_audit_before.json \
        --after  evaluation/inventory/multiturn_chip_audit_after.json \
        --out    evaluation/results/phase05_chip_answerability.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _rows(data: dict) -> list[dict]:
    return [row for key in sorted(data) for row in data[key]["rows"]]


def _tally(rows: list[dict]) -> tuple[int, int, Counter]:
    good = sum(1 for row in rows if row["verdict"] == "GOOD")
    return good, len(rows) - good, Counter(row["reason"] for row in rows
                                           if row["verdict"] == "BAD")


def _cell(text: str, limit: int = 90) -> str:
    flat = " ".join(str(text or "").split()).replace("|", "\\|")
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _table(rows: list[dict]) -> list[str]:
    out = ["| Record | Turn 1 | Chip | Turn 2 tools | Citations | Verdict |",
           "|---|---|---|---|---|---|"]
    for row in rows:
        citations = ("—" if not row["turn_2_sources"]
                     else f"{row['turn_2_sources']} resolved")
        verdict = row["verdict"]
        if verdict == "BAD":
            verdict += f" — {row['reason']}"
        out.append("| {} | {} | {} | {} | {} | {} |".format(
            row["record"], row["turn_1_cmd"], _cell(row["chip"], 70),
            _cell(row["turn_2_tools"], 46), citations, verdict,
        ))
    return out


def _progressions(data: dict) -> list[str]:
    out = ["| Record | Turn 1 | Turn 2 clicked | Turn 2 offered | Turn 3 clicked | "
           "Turn 3 offered | Advanced? |",
           "|---|---|---|---|---|---|---|"]
    for key in sorted(data):
        turns = data[key]["progression"]["turns"]
        record = data[key]["progression"]["record"]
        command = data[key]["progression"]["turn_1_cmd"]
        if not turns:
            out.append(f"| {record} | {command} | — | no chip offered | — | — | — |")
            continue
        first = turns[0]
        if len(turns) < 2:
            out.append("| {} | {} | {} | {} | — | — | no chip offered |".format(
                record, command, _cell(first["question"], 46),
                _cell("; ".join(first["follow_ups"]), 70)))
            continue
        second = turns[1]
        advanced = "yes" if second["question"] != first["question"] else "NO — repeated"
        out.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            record, command,
            _cell(first["question"], 46),
            _cell("; ".join(second["follow_ups"]), 70),
            _cell(second["question"], 46),
            _cell("; ".join(first["follow_ups"]), 70),
            advanced,
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision-before", default="")
    ap.add_argument("--revision-after", default="")
    args = ap.parse_args()

    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())
    b_rows, a_rows = _rows(before), _rows(after)
    b_good, b_bad, b_why = _tally(b_rows)
    a_good, a_bad, a_why = _tally(a_rows)

    lines = [
        "# Multi-turn chip answerability — before and after",
        "",
        f"- Before: `{args.revision_before or 'unrecorded'}` — {b_good}/{len(b_rows)} "
        f"GOOD, {b_bad} BAD",
        f"- After: `{args.revision_after or 'unrecorded'}` — {a_good}/{len(a_rows)} "
        f"GOOD, {a_bad} BAD",
        "",
        "## Before",
        "",
        f"Bad reasons: {dict(b_why)}",
        "",
        *_table(b_rows),
        "",
        "## After",
        "",
        f"Bad reasons: {dict(a_why)}",
        "",
        *_table(a_rows),
        "",
        "## Three-turn progression — before",
        "",
        *_progressions(before),
        "",
        "## Three-turn progression — after",
        "",
        *_progressions(after),
        "",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out}: before {b_good}/{len(b_rows)} GOOD, "
          f"after {a_good}/{len(a_rows)} GOOD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
