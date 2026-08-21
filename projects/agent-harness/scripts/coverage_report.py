"""coverage_report.py — per-note section + chip coverage after section mapping.

Writes a compact summary to stdout. Also writes data/mtsamples/coverage.json
keyed by sample id: {section_count, sections, chip_support, sparse}.
"""

import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.sections import parse_note  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "mtsamples"

CHIPS = {
    "meds_section": ("discharge_medications",),
    "summarize": ("brief_hospital_course",),
    "risk": ("discharge_condition", "discharge_diagnosis"),
    "citations": (
        "history_of_present_illness",
        "discharge_instructions",
        "physical_exam",
    ),
}

# Notes whose whole body is one unbroken section (no sectioning to cite).
SPARSE_THRESHOLD = 1  # sections <= this count a note "structurally thin"


def main() -> None:
    zero = []
    cov_by_sections: dict[int, list[str]] = {}
    chip_counter: Counter[str] = Counter()
    per_note: dict[str, dict] = {}

    for path in sorted(DATA_DIR.glob("*.txt")):
        sid = path.stem
        parsed = parse_note(path.read_text(encoding="utf-8"))
        n = len(parsed.sections)
        names = [s.name for s in parsed.sections]
        cov_by_sections.setdefault(n, []).append(sid)
        if n == 0:
            zero.append(sid)
        supported = [
            chip for chip, secs in CHIPS.items() if any(x in names for x in secs)
        ]
        for chip in supported:
            chip_counter[chip] += 1
        per_note[sid] = {
            "section_count": n,
            "sections": names,
            "chip_support": supported,
            "sparse": n <= SPARSE_THRESHOLD,
        }

    print("notes with 0 recognised sections:", len(zero), zero)
    print("notes by # recognised sections:",
          {k: len(v) for k, v in sorted(cov_by_sections.items())})
    print("chip support (of 108):", dict(sorted(chip_counter.items())))

    DATA_DIR.joinpath("coverage.json").write_text(
        json.dumps(per_note, indent=1), encoding="utf-8"
    )
    print("wrote data/mtsamples/coverage.json")


if __name__ == "__main__":
    main()
