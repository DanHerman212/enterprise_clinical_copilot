"""
benchmark_gate — Fail pipeline if benchmark doesn't beat HOSPITAL.
"""

from __future__ import annotations

from typing import NamedTuple

from kfp import dsl
from ._image import TRAINING_IMAGE

HOSPITAL_AUCPR = 0.3325  # pre-computed, immutable


def run_benchmark_gate(*, benchmark_aucpr: float) -> bool:
    """Return True if benchmark beats HOSPITAL baseline.  Raises on failure."""
    passed = benchmark_aucpr > HOSPITAL_AUCPR
    print(f"  Benchmark:  {benchmark_aucpr:.4f}")
    print(f"  HOSPITAL:   {HOSPITAL_AUCPR:.4f}")
    print(f"  Gate:       {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise ValueError(
            f"Benchmark AUCPR ({benchmark_aucpr:.4f}) did not beat "
            f"HOSPITAL baseline ({HOSPITAL_AUCPR:.4f})."
        )
    return passed


@dsl.component(base_image=TRAINING_IMAGE, packages_to_install=[])
def benchmark_gate(
    benchmark_aucpr: float,
) -> NamedTuple("GateOutputs", [("passed", bool)]):
    """KFP component: gate on benchmark > HOSPITAL."""
    passed = run_benchmark_gate(benchmark_aucpr=benchmark_aucpr)
    return (passed,)
