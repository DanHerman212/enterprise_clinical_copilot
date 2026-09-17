"""
benchmark_gate — Fail pipeline if benchmark doesn't beat HOSPITAL by a margin.

The HOSPITAL baseline is not an input. It is read from the versioned artifact
inside this container (`_baselines.hospital_aucpr`), because a baseline passed in
as a parameter is a baseline the submitter chooses: `hospital_aucpr=0.01` used to
pass validation and neutralize this gate and the test gate together (ECC-65). The
gate still fails closed on an implausible baseline — the guard now protects the
artifact rather than a parameter — and a minimum margin is required so a 0.0001
"improvement" cannot pass.
"""

from typing import NamedTuple

from kfp import dsl
from ._baselines import hospital_aucpr
from ._image import TRAINING_IMAGE, component

# Minimum absolute AUCPR improvement over the baseline for a gate to pass.
MIN_GATE_MARGIN = 0.01


def validate_baseline(hospital_aucpr: float) -> None:
    """Fail closed on a missing or corrupt baseline (ECC-65)."""
    if not (0.0 < hospital_aucpr < 1.0):
        raise ValueError(
            f"HOSPITAL baseline AUCPR ({hospital_aucpr}) is not a plausible "
            "score in (0, 1) — the baseline artifact is missing or corrupt. "
            "Refusing to gate."
        )


def run_benchmark_gate(
    *,
    benchmark_aucpr: float,
    hospital_aucpr: float,
    min_margin: float = MIN_GATE_MARGIN,
) -> bool:
    """Return True if benchmark beats HOSPITAL by the margin. Raises on failure."""
    validate_baseline(hospital_aucpr)
    passed = benchmark_aucpr > hospital_aucpr + min_margin
    print(f"  Benchmark:  {benchmark_aucpr:.4f}")
    print(f"  HOSPITAL:   {hospital_aucpr:.4f}  (+ margin {min_margin})")
    print(f"  Gate:       {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise ValueError(
            f"Benchmark AUCPR ({benchmark_aucpr:.4f}) did not beat "
            f"HOSPITAL baseline ({hospital_aucpr:.4f}) by the required "
            f"margin ({min_margin})."
        )
    return passed


@component(base_image=TRAINING_IMAGE, packages_to_install=[])
def benchmark_gate(
    benchmark_aucpr: float,
) -> NamedTuple("GateOutputs", [("passed", bool)]):
    """KFP component: gate on benchmark > HOSPITAL.

    Takes no baseline: there is deliberately nothing here for a submitter to set.
    """
    from mlops.training.components.benchmark_gate import run_benchmark_gate

    passed = run_benchmark_gate(
        benchmark_aucpr=benchmark_aucpr, hospital_aucpr=hospital_aucpr(),
    )
    return (passed,)
