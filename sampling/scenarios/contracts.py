"""Protocols for live-dashboard scientific scenarios."""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

from benchmarks.reference_cases import ReferenceComparisonReport
from benchmarks.reference_cases.models import ExperimentalReferenceCase
from core.state import SimulationState
from graph.graph_manager import ConnectivityGraph
from physics.forces.composite import ForceEvaluation
from qcloud.qcloud_coupling import QCloudCorrectionModel
from validation import FidelityComparisonReport, StructureComparisonReport

if TYPE_CHECKING:
    from sampling.scenarios.complex_assembly import ComplexAssemblyProgress, ComplexAssemblySetup


class DashboardScenario(Protocol):
    """Protocol for concrete live-dashboard scenarios."""

    name: str
    classification: str

    def build_setup(self) -> "ComplexAssemblySetup":
        """Return the concrete initial conditions and runtime metadata."""

    def measure_progress(
        self,
        state: SimulationState,
        *,
        graph: ConnectivityGraph | None = None,
    ) -> "ComplexAssemblyProgress":
        """Return a structured scenario-progress report."""

    def reference_case(self) -> ExperimentalReferenceCase | None:
        """Return an experimental reference case when one exists."""

    def build_reference_report(
        self,
        progress: "ComplexAssemblyProgress",
    ) -> ReferenceComparisonReport | None:
        """Return a scenario-specific reference comparison payload when available."""

    def build_structure_report(
        self,
        state: SimulationState,
        *,
        progress: "ComplexAssemblyProgress" | None = None,
    ) -> StructureComparisonReport | None:
        """Return a scenario-specific structural-comparison payload when available."""

    def build_qcloud_correction_model(self) -> QCloudCorrectionModel | None:
        """Return a scenario-specific qcloud correction model when available."""

    def build_fidelity_report(
        self,
        state: SimulationState,
        *,
        baseline_evaluation: ForceEvaluation,
        corrected_evaluation: ForceEvaluation,
        progress: "ComplexAssemblyProgress" | None = None,
    ) -> FidelityComparisonReport | None:
        """Return a scenario-specific fidelity payload when available."""
