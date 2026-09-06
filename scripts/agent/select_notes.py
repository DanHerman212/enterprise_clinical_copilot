"""select_notes.py — Step 2.8: pick ~24 notes spanning low/borderline/high.

Merges the story-anchored band fill (fill.json) with per-note section/chip
support (coverage.json) and picks 8/8/8 notes per band that maximise chip
support, section count, and story coherence (prefer notes where race was
parsed, so the race_unknown card artifact is minimised).

Output: stdout selection + data/mtsamples/selection_24.json
"""

import json
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
DATA_DIR = HARNESS / "data" / "mtsamples"

CHIP_W = {"meds_section": 3, "summarize": 3, "citations": 2, "risk": 2}
PER_BAND = 8


def main() -> None:
    fill = json.loads((DATA_DIR / "fill.json").read_text())
    cov = json.loads((DATA_DIR / "coverage.json").read_text())

    rows = []
    for sid, r in fill.items():
        c = cov.get(sid, {})
        race_parsed = r["provenance"].get("race", ["", ""])[0] == "parsed"
        preview = (DATA_DIR / f"{sid}.txt").read_text(encoding="utf-8")[:70].replace("\n", " ")
        rows.append({
            "sid": sid, "prob": r["probability"], "band": r["band"],
            "sections": len(c.get("sections", [])),
            "chips": c.get("chip_support", []),
            "race_parsed": race_parsed, "preview": preview,
        })

    def sel_score(r: dict) -> float:
        s = sum(CHIP_W.get(ch, 0) for ch in r["chips"])
        s += min(r["sections"], 6)
        s += 1.5 if r["race_parsed"] else 0
        return s

    by_band: dict[str, list[dict]] = {"low": [], "borderline": [], "high": []}
    for r in rows:
        r["score"] = sel_score(r)
        by_band[r["band"]].append(r)
    for b in by_band:
        by_band[b].sort(key=lambda x: (-x["score"], x["prob"]))

    selected = []
    for b in ("low", "borderline", "high"):
        pick = by_band[b][:PER_BAND]
        selected.extend(pick)
        print(f"--- {b} (pick {len(pick)} of {len(by_band[b])}) ---")
        for r in pick:
            chips = ",".join(c[0].upper() for c in r["chips"])
            print(f"  {r['sid']:>5} p={r['prob']:.4f} sec={r['sections']:2d} "
                  f"race={'Y' if r['race_parsed'] else 'n'} chips=[{chips}]")
            print(f"        {r['preview'][:66]}")

    out = [{"sid": r["sid"], "prob": round(r["prob"], 4), "band": r["band"],
            "sections": r["sections"], "chips": r["chips"],
            "race_parsed": r["race_parsed"]} for r in selected]
    (DATA_DIR / "selection_24.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {DATA_DIR / 'selection_24.json'} ({len(out)} notes)")


if __name__ == "__main__":
    main()
