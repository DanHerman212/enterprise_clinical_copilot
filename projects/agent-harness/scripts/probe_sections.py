"""A1: measure section-parse quality and per-section redaction on the corpus.

This probe settles two questions before any concept tagging happens:

  1. Does rag/sections.py actually parse real MIMIC notes, not just the test
     fixture? Reported as coverage distribution + ranked unknown headings, so a
     forgotten section header is a countable line item, not a silent merge.

  2. How much of each section survives de-identification? This replaces the
     earlier corpus-wide "95.5% of Social History is redacted" figure, which was
     an artifact of a regex that truncated sections at their own sub-fields.

Reads the local note cache (scripts/fetch_note_cache.py); touches no cloud
services. Writes docs/probes/sections_probe.json — aggregate counts only, no
note text, so the artifact is safe to commit.

Usage:
    python scripts/probe_sections.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.notes import iter_notes, read_manifest  # noqa: E402
from rag.sections import KNOWN_HEADINGS, parse_note, redaction_profile  # noqa: E402

ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "docs" / "probes" / "sections_probe.json"

LOW_COVERAGE = 0.80


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="analyse only the first N notes (for a quick pass)")
    args = parser.parse_args()

    manifest = read_manifest()
    print(f"Cache: {manifest['note_count']} notes, created {manifest['created_utc']}")

    coverages: list[float] = []
    zero_section_notes = 0
    low_coverage_notes = 0
    unknown_headings: Counter[str] = Counter()

    # per section: how often present, and what redaction leaves behind
    present: Counter[str] = Counter()
    placeholder_only: Counter[str] = Counter()
    empty_body: Counter[str] = Counter()
    has_content: Counter[str] = Counter()
    informative_chars: dict[str, list[int]] = {name: [] for name in KNOWN_HEADINGS}

    analysed = 0
    for note in iter_notes():
        if args.limit is not None and analysed >= args.limit:
            break
        analysed += 1
        parsed = parse_note(note["text"])

        coverages.append(parsed.coverage)
        if not parsed.sections:
            zero_section_notes += 1
        if parsed.coverage < LOW_COVERAGE:
            low_coverage_notes += 1
        for heading in parsed.unknown_headings:
            unknown_headings[heading.lower()] += 1

        seen: set[str] = set()
        for section in parsed.sections:
            if section.name in seen:
                continue  # count each section once per note, first occurrence
            seen.add(section.name)
            present[section.name] += 1
            profile = redaction_profile(section.body)
            if profile.is_empty:
                empty_body[section.name] += 1
            elif profile.is_placeholder_only:
                placeholder_only[section.name] += 1
            else:
                has_content[section.name] += 1
                informative_chars[section.name].append(profile.informative_chars)

        if analysed % 5000 == 0:
            print(f"  …{analysed} notes")

    if args.limit is None and analysed != manifest["note_count"]:
        print(f"FAILED: analysed {analysed} notes but cache holds "
              f"{manifest['note_count']}.")
        return 1

    coverages.sort()
    sections_report = {}
    for name in KNOWN_HEADINGS:
        n_present = present[name]
        chars = informative_chars[name]
        sections_report[name] = {
            "present": n_present,
            "present_pct": round(100 * n_present / analysed, 1),
            "empty": empty_body[name],
            "placeholder_only": placeholder_only[name],
            "has_content": has_content[name],
            "has_content_pct_of_present": (
                round(100 * has_content[name] / n_present, 1) if n_present else None
            ),
            "median_informative_chars": (
                int(statistics.median(chars)) if chars else None
            ),
        }

    artifact = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cache_created_utc": manifest["created_utc"],
        "notes_analysed": analysed,
        "parse": {
            "coverage_mean": round(statistics.mean(coverages), 4),
            "coverage_median": round(statistics.median(coverages), 4),
            "coverage_p10": round(coverages[int(0.10 * len(coverages))], 4),
            "coverage_p01": round(coverages[int(0.01 * len(coverages))], 4),
            "zero_section_notes": zero_section_notes,
            f"notes_below_{LOW_COVERAGE}": low_coverage_notes,
            "distinct_unknown_headings": len(unknown_headings),
            "top_unknown_headings": unknown_headings.most_common(40),
        },
        "sections": sections_report,
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\nWrote {ARTIFACT_PATH}")

    # -- console summary -------------------------------------------------
    p = artifact["parse"]
    print(f"\nParse quality over {analysed} notes:")
    print(f"  coverage mean/median: {p['coverage_mean']}/{p['coverage_median']}")
    print(f"  coverage p10/p01:     {p['coverage_p10']}/{p['coverage_p01']}")
    print(f"  zero-section notes:   {zero_section_notes}")
    print(f"  below {LOW_COVERAGE} coverage:   {low_coverage_notes}")
    print("\n  top unknown headings:")
    for heading, count in unknown_headings.most_common(15):
        print(f"    {count:>8}  {heading}")

    print(f"\n{'section':<28}{'present%':>9}{'content%':>10}{'med chars':>11}")
    for name, row in sorted(sections_report.items(),
                            key=lambda item: -item[1]["present"]):
        if row["present"] == 0:
            continue
        content_pct = row["has_content_pct_of_present"]
        med = row["median_informative_chars"]
        print(f"{name:<28}{row['present_pct']:>8}%"
              f"{content_pct if content_pct is not None else '-':>9}%"
              f"{med if med is not None else '-':>11}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
