"""Real-world benchmark targets with known experimental answers."""

from benchmarks.reference_cases.barnase_barstar import barnase_barstar_reference_case
from benchmarks.reference_cases.barnase_barstar_structure_targets import barnase_barstar_structure_target
from benchmarks.reference_cases.comparison import (
    ReferenceComparisonMetric,
    ReferenceComparisonReport,
    build_barnase_barstar_proxy_report,
    build_spike_ace2_proxy_report,
)
from benchmarks.reference_cases.models import (
    ExperimentalReferenceCase,
    ReferenceObservable,
    ReferenceSource,
    StructuralReference,
)
from benchmarks.reference_cases.spike_ace2 import spike_ace2_reference_case
from benchmarks.reference_cases.spike_ace2_structure_targets import spike_ace2_structure_target
from benchmarks.reference_cases.structure_targets import (
    InterfaceContactTarget,
    ReferenceStructureTarget,
    StructureLandmarkTarget,
)

__all__ = [
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
