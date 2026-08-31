"""
benchmark_gate — Fail pipeline if benchmark doesn't beat HOSPITAL by a margin.

The HOSPITAL baseline AUCPR is passed in (single source of truth: the pipeline
resolves it from artifacts/hospital_baseline.json at build time) rather than
hardcoded here. The gate fails closed on an implausible baseline (ECC-65): a
submitter passing 0.0 as a runtime override used to neutralize the gate; and a
minimum margin is required so a 0.0001 "improvement" cannot pass.
"""

from typing import NamedTuple

from kfp import dsl
from ._image import TRAINING_IMAGE, component

# Minimum absolute AUCPR improvement over the baseline for a gate to pass.
MIN_GATE_MARGIN = 0.01


def validate_baseline(hospital_aucpr: float) -> None:
    """Fail closed on a missing/neutralized baseline (ECC-65)."""
    if not (0.0 < hospital_aucpr < 1.0):
        raise ValueError(
            f"HOSPITAL baseline AUCPR ({hospital_aucpr}) is not a plausible "
            "score in (0, 1) — the baseline artifact is missing or the gate "
            "is being neutralized by a runtime override. Refusing to gate."
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
    hospital_aucpr: float,
) -> NamedTuple("GateOutputs", [("passed", bool)]):
    """KFP component: gate on benchmark > HOSPITAL."""
    from pipelines.components.benchmark_gate import run_benchmark_gate

    passed = run_benchmark_gate(
        benchmark_aucpr=benchmark_aucpr, hospital_aucpr=hospital_aucpr,
    )
    return (passed,)
