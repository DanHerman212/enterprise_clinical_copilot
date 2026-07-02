"""
benchmark_gate — Fail pipeline if benchmark doesn't beat HOSPITAL.

The HOSPITAL baseline AUCPR is passed in (single source of truth: the pipeline
resolves it from artifacts/hospital_baseline.json at build time) rather than
hardcoded here.
"""

from typing import NamedTuple

from kfp import dsl
from ._image import TRAINING_IMAGE


def run_benchmark_gate(*, benchmark_aucpr: float, hospital_aucpr: float) -> bool:
    """Return True if benchmark beats HOSPITAL baseline.  Raises on failure."""
    passed = benchmark_aucpr > hospital_aucpr
    print(f"  Benchmark:  {benchmark_aucpr:.4f}")
    print(f"  HOSPITAL:   {hospital_aucpr:.4f}")
    print(f"  Gate:       {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise ValueError(
            f"Benchmark AUCPR ({benchmark_aucpr:.4f}) did not beat "
            f"HOSPITAL baseline ({hospital_aucpr:.4f})."
        )
    return passed


@dsl.component(base_image=TRAINING_IMAGE, packages_to_install=[])
def benchmark_gate(
    benchmark_aucpr: float,
    hospital_aucpr: float,
) -> NamedTuple("GateOutputs", [("passed", bool)]):
    """KFP component: gate on benchmark > HOSPITAL."""
    passed = run_benchmark_gate(
        benchmark_aucpr=benchmark_aucpr, hospital_aucpr=hospital_aucpr,
    )
    return (passed,)
