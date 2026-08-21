"""check_coherence.py — trace each selected note's top factor to its basis.

The card shows top_factors from the real model. Coherence = the factor's
PROVENANCE is traceable to the note:

  * parsed factor  -> the raw value must appear in the note text (OK/MISS by
                      text mention).
  * filled factor  -> the fill BASIS must be defensible from the note (e.g.
                      prior_inpatient_days filled "~14d x N status-post/prior
                      mentions" -> the note must contain those mentions; or
                      "no prior signal -> 0" -> coherent because the note is
                      silent on prior admissions).
  * race artifact  -> top factor is race but race was NOT parsed (filled
                      race_unknown). This is the known card artifact: the
                      model's race_unknown attribution shows up as a top
                      factor on notes that never mention race. FLAGGED, not a
                      story mismatch — a presentation decision for the build.

Outputs one line per selected note + a summary of artifact vs story-mismatch.
"""

import json
import re
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
DATA_DIR = HARNESS / "data" / "mtsamples"

# For a filled prior_admission/prior_inpatient factor to be coherent, the note
# must mention the signal the fill counted.
PRIOR_BASIS_WORDS = ["status post", "prior", "previous", "re-admission",
                     "readmission", "had been admitted", "was admitted"]
ED_BASIS_WORDS = ["emergency room", "emergency department", "er", "ed"]


def _basis_ok(text: str, basis: str) -> bool:
    b = basis.lower()
    if "no prior" in b or "no ed" in b:
        # Fill explicitly chose 0 because the note is silent -> coherent.
        return True
    if "status post" in b or "mention" in b or "prior admission signal" in b:
        return any(w in text.lower() for w in PRIOR_BASIS_WORDS)
    if "ed mention" in b:
        return any(w in text.lower() for w in ED_BASIS_WORDS)
    return True  # numeric/other basis assumed coherent


def _parse_ok(text: str, prov) -> bool:
    """For a parsed factor, check the parsed value plausibly appears in text."""
    if not isinstance(prov, list) or not prov:
        return True
    raw = str(prov[-1])
    if raw.lower() in text:
        return True
    # numeric values like "89.0" -> find the bare number in text
    m = re.match(r"^(\d+)(?:\.0)?$", raw)
    if m:
        return (" " + m.group(1)) in text or text.startswith(m.group(1))
    return False


def main() -> None:
    fill = json.loads((DATA_DIR / "fill.json").read_text())
    sel = json.loads((DATA_DIR / "selection_24.json").read_text())

    artifact = story_miss = coherent = 0
    for s in sel:
        sid = s["sid"]
        r = fill[sid]
        text = (DATA_DIR / f"{sid}.txt").read_text(encoding="utf-8").lower()
        tf = r["top_factors"][0]
        feat = tf["feature"]
        prov = r["provenance"].get(feat)
        kind = prov[0] if isinstance(prov, list) and prov else "filled"
        basis = prov[1] if isinstance(prov, list) and len(prov) > 1 else str(prov)

        # race artifact: top factor is race but race was filled (unknown)
        if feat == "race" and kind == "filled":
            artifact += 1
            status = "ARTIFACT"
        elif kind == "filled" and feat in ("prior_inpatient_days",
                                            "prior_admission_count",
                                            "recent_ed_visits",
                                            "index_los_days"):
            good = _basis_ok(text, basis)
            coherent += int(good)
            story_miss += int(not good)
            status = "OK" if good else "STORY-MISS"
        else:
            good = _parse_ok(text, prov)
            coherent += int(good)
            story_miss += int(not good)
            status = "OK" if good else "STORY-MISS"

        print(f"  {status:>11} {sid:>5} {r['band']:<10} top={feat:<22} "
              f"[{kind}] basis={str(basis)[:46]}")

    print(f"\nsummary of 24: {coherent} coherent, {story_miss} story-miss, "
          f"{artifact} race-fill artifact")


if __name__ == "__main__":
    main()
