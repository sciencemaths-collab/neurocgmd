"""Benchmark helpers for repeatable early-suite measurements."""

from benchmarks.reference_cases import (
    ExperimentalReferenceCase,
    ReferenceComparisonMetric,
    ReferenceComparisonReport,
    ReferenceObservable,
    ReferenceSource,
    ReferenceStructureTarget,
    StructuralReference,
    StructureLandmarkTarget,
    InterfaceContactTarget,
    barnase_barstar_reference_case,
    barnase_barstar_structure_target,
    build_barnase_barstar_proxy_report,
    build_spike_ace2_proxy_report,
    spike_ace2_reference_case,
    spike_ace2_structure_target,
)

# Deferred to avoid circular import: baseline_suite → ai_control → ml → topology
_DEFERRED_BASELINE = {
    "BaselineBenchmarkSuite",
    "BenchmarkCaseResult",
    "BenchmarkReport",
    "SmallProteinBenchmarkReport",
    "SmallProteinBenchmarkRunner",
    "SmallProteinBenchmarkSpec",
}


def __getattr__(name: str):
    if name in _DEFERRED_BASELINE:
        from benchmarks.baseline_suite import (  # noqa: F811
            BaselineBenchmarkSuite,
            BenchmarkCaseResult,
            BenchmarkReport,
        )
        from benchmarks.small_protein import (  # noqa: F811
            SmallProteinBenchmarkReport,
            SmallProteinBenchmarkRunner,
            SmallProteinBenchmarkSpec,
        )
        _cache = {
            "BaselineBenchmarkSuite": BaselineBenchmarkSuite,
            "BenchmarkCaseResult": BenchmarkCaseResult,
            "BenchmarkReport": BenchmarkReport,
            "SmallProteinBenchmarkReport": SmallProteinBenchmarkReport,
            "SmallProteinBenchmarkRunner": SmallProteinBenchmarkRunner,
            "SmallProteinBenchmarkSpec": SmallProteinBenchmarkSpec,
        }
        globals().update(_cache)
        return _cache[name]
    raise AttributeError(f"module 'benchmarks' has no attribute {name!r}")


__all__ = [
    "BaselineBenchmarkSuite",
    "BenchmarkCaseResult",
    "BenchmarkReport",
    "SmallProteinBenchmarkReport",
    "SmallProteinBenchmarkRunner",
    "SmallProteinBenchmarkSpec",
    "ExperimentalReferenceCase",
    "ReferenceComparisonMetric",
    "ReferenceComparisonReport",
    "ReferenceObservable",
    "ReferenceSource",
    "ReferenceStructureTarget",
    "StructuralReference",
    "StructureLandmarkTarget",
    "InterfaceContactTarget",
    "barnase_barstar_reference_case",
    "barnase_barstar_structure_target",
    "build_barnase_barstar_proxy_report",
    "build_spike_ace2_proxy_report",
    "spike_ace2_reference_case",
    "spike_ace2_structure_target",
]
