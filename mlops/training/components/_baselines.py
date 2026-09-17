"""
_baselines — the thresholds the gates compare against, read where the gate runs.

These were pipeline parameters. A parameter is answered by whoever submits the
run, so every gate was only as strong as the submitter's discipline: one
plausible-looking number (`hospital_aucpr=0.01`) passed validation and
neutralised the benchmark gate and the test gate together, after which the bundle
faithfully recorded that a model which beat nothing had passed.

The baseline is now read from the versioned artifact inside the gate container
itself. That is why `artifacts/hospital_baseline.json` is shipped in the training
image rather than left out of the build context: the gate cannot hand its
threshold to anyone else if the only copy it will read is the one that travels in
the image the run records. Changing the number is a commit against the artifact,
not an argument at submit time.

The remaining thresholds in the gates (`MIN_GATE_MARGIN`,
`MAX_VAL_TEST_DEGRADATION`) were already code, and stay where they are.
"""

import json
from pathlib import Path

BASELINE_ARTIFACT = (
    Path(__file__).resolve().parents[2] / "artifacts" / "hospital_baseline.json"
)

# Share of columns Evidently may call drifted before the drift gate fails. A
# judgement call, so it is written down with its reasoning instead of being
# offered as an input: 20% of the 49 columns tolerates a handful of genuinely
# moving lab values without waving through a cohort that has shifted underneath
# the model. Raise it here, in review, with the reason.
MAX_DRIFTED_SHARE = 0.2


def hospital_aucpr() -> float:
    """The HOSPITAL baseline AUCPR, read from the versioned artifact.

    Fails closed. The old guard rejected an implausible *parameter*; with no
    parameter to reject, the same guard now protects the artifact — a missing,
    unreadable or implausible baseline stops the gate rather than gating against
    a number nobody can account for.
    """
    if not BASELINE_ARTIFACT.exists():
        raise FileNotFoundError(
            f"HOSPITAL baseline artifact not found at {BASELINE_ARTIFACT}. It is "
            "shipped in the training image (see mlops/.gcloudignore); its absence "
            "means this container cannot gate anything, so it refuses to try."
        )
    record = json.loads(BASELINE_ARTIFACT.read_text())
    value = float(record["aucpr"])
    if not (0.0 < value < 1.0):
        raise ValueError(
            f"HOSPITAL baseline AUCPR ({value}) is not a plausible score in (0, 1) "
            "— the artifact is corrupt. Refusing to gate."
        )
    return value
