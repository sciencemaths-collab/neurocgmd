"""Deterministic benchmark helpers for baseline scientific and control slices."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter

from ai_control.controller import ExecutiveController
from chemistry.interface_logic import ChemistryInterfaceReport
from compartments.registry import CompartmentRegistry
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import CompartmentId, FrozenMetadata, StateId, coerce_scalar
from forcefields.base_forcefield import BaseForceField
from graph.graph_manager import ConnectivityGraph, ConnectivityGraphManager
from integrators.base import ForceEvaluator
from memory.episode_registry import EpisodeRegistry
from memory.replay_buffer import ReplayBuffer
from memory.trace_store import TraceRecord
from ml.live_features import LiveFeatureVector
from ml.residual_model import ResidualModel
from ml.uncertainty_model import UncertaintyEstimate
from physics.forces.composite import ForceEvaluation
from qcloud.qcloud_coupling import QCloudCorrectionModel, QCloudCouplingResult, QCloudForceCoupler
from qcloud.region_selector import LocalRegionSelector
from topology.system_topology import SystemTopology


@dataclass(frozen=True, slots=True)
class BenchmarkCaseResult(ValidatableComponent):
    """One measured benchmark case with light metadata."""

    name: str
    iterations: int
    elapsed_seconds: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "elapsed_seconds", coerce_scalar(self.elapsed_seconds, "elapsed_seconds"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def average_seconds_per_iteration(self) -> float:
        return self.elapsed_seconds / self.iterations

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name.strip():
            issues.append("name must be a non-empty string.")
        if self.iterations <= 0:
            issues.append("iterations must be strictly positive.")
        if self.elapsed_seconds < 0.0:
            issues.append("elapsed_seconds must be non-negative.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class BenchmarkReport(ValidatableComponent):
    """Aggregated report for one benchmark-suite run."""

    suite_name: str
    cases: tuple[BenchmarkCaseResult, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def case_names(self) -> tuple[str, ...]:
        return tuple(case.name for case in self.cases)

    def case_for(self, name: str) -> BenchmarkCaseResult:
        for case in self.cases:
            if case.name == name:
                return case
        raise KeyError(f"Unknown benchmark case: {name}")

    def total_elapsed_seconds(self) -> float:
        return sum(case.elapsed_seconds for case in self.cases)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.suite_name.strip():
            issues.append("suite_name must be a non-empty string.")
        if not self.cases:
            issues.append("cases must contain at least one benchmark case.")
        if len(self.case_names()) != len(set(self.case_names())):
            issues.append("benchmark case names must be unique within one report.")
        return tuple(issues)


@dataclass(slots=True)
class BaselineBenchmarkSuite(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Measure repeatable observer-side timings for baseline execution slices."""

    default_repeats: int = 3
    name: str = "baseline_benchmark_suite"
    classification: str = "[established]"

    def describe_role(self) -> str:
        return (
            "Measures repeatable wall-clock timings for force, graph, qcloud, ML, "
            "and controller pathways without mutating subsystem ownership."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "physics/forces/composite.py",
            "graph/graph_manager.py",
            "qcloud/qcloud_coupling.py",
            "ml/residual_model.py",
            "ai_control/controller.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/validation_and_benchmarking.md",
            "docs/sections/section_13_validation_and_benchmarking_suite.md",
        )

    def validate(self) -> tuple[str, ...]:
        if self.default_repeats <= 0:
            return ("default_repeats must be strictly positive.",)
        return ()

    def _resolve_repeats(self, repeats: int | None) -> int:
        resolved = self.default_repeats if repeats is None else repeats
        if resolved <= 0:
            raise ContractValidationError("Benchmark repeats must be strictly positive.")
        return resolved

    def _measure_case(
        self,
        *,
        name: str,
        repeats: int,
        operation: Callable[[], object],
        metadata_factory: Callable[[object], FrozenMetadata | dict[str, object] | None],
    ) -> BenchmarkCaseResult:
        last_result: object | None = None
        started_at = perf_counter()
        for _ in range(repeats):
            last_result = operation()
        elapsed = perf_counter() - started_at
        metadata = metadata_factory(last_result) if last_result is not None else {}
        return BenchmarkCaseResult(
            name=name,
            iterations=repeats,
            elapsed_seconds=elapsed,
            metadata=metadata if isinstance(metadata, FrozenMetadata) else FrozenMetadata(metadata),
        )

    def benchmark_force_evaluation(
        self,
        *,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
        force_evaluator: ForceEvaluator,
        repeats: int | None = None,
    ) -> BenchmarkCaseResult:
        resolved_repeats = self._resolve_repeats(repeats)
        return self._measure_case(
            name="force_evaluation",
            repeats=resolved_repeats,
            operation=lambda: force_evaluator.evaluate(state, topology, forcefield),
            metadata_factory=lambda result: {
                "force_evaluator": force_evaluator.name,
                "particle_count": state.particle_count,
                "potential_energy": result.potential_energy,
            },
        )

    def benchmark_graph_update(
        self,
        *,
        state: SimulationState,
        topology: SystemTopology,
        graph_manager: ConnectivityGraphManager,
        previous_graph: ConnectivityGraph,
        repeats: int | None = None,
    ) -> BenchmarkCaseResult:
        resolved_repeats = self._resolve_repeats(repeats)
        return self._measure_case(
            name="graph_update",
            repeats=resolved_repeats,
            operation=lambda: graph_manager.update(state, topology, previous_graph),
            metadata_factory=lambda result: {
                "graph_manager": graph_manager.name,
                "active_edge_count": len(result.active_edges()),
                "adaptive_edge_count": len(result.adaptive_edges()),
            },
        )

    def benchmark_qcloud_coupling(
        self,
        *,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
        base_force_evaluator: ForceEvaluator,
        coupler: QCloudForceCoupler,
        correction_model: QCloudCorrectionModel,
        region_selector: LocalRegionSelector,
        graph: ConnectivityGraph,
        compartments: CompartmentRegistry | None = None,
        trace_record: TraceRecord | None = None,
        focus_compartments: Sequence[CompartmentId | str] = (),
        repeats: int | None = None,
    ) -> BenchmarkCaseResult:
        resolved_repeats = self._resolve_repeats(repeats)
        return self._measure_case(
            name="qcloud_coupling",
            repeats=resolved_repeats,
            operation=lambda: coupler.evaluate_with_selector(
                state=state,
                topology=topology,
                forcefield=forcefield,
                base_force_evaluator=base_force_evaluator,
                correction_model=correction_model,
                region_selector=region_selector,
                graph=graph,
                compartments=compartments,
                trace_record=trace_record,
                focus_compartments=focus_compartments,
            ),
            metadata_factory=lambda result: {
                "qcloud_coupler": coupler.name,
                "selected_region_count": len(result.selected_regions),
                "applied_correction_count": len(result.applied_corrections),
            },
        )

    def benchmark_residual_prediction(
        self,
        *,
        state: SimulationState,
        base_evaluation: ForceEvaluation,
        residual_model: ResidualModel,
        repeats: int | None = None,
    ) -> BenchmarkCaseResult:
        resolved_repeats = self._resolve_repeats(repeats)
        return self._measure_case(
            name="residual_prediction",
            repeats=resolved_repeats,
            operation=lambda: residual_model.predict(state, base_evaluation),
            metadata_factory=lambda result: {
                "residual_model": residual_model.name,
                "predicted_energy_delta": result.predicted_energy_delta,
                "prediction_confidence": result.confidence,
            },
        )

    def benchmark_controller_decision(
        self,
        *,
        state: SimulationState,
        graph: ConnectivityGraph,
        controller: ExecutiveController,
        trace_record: TraceRecord | None = None,
        uncertainty_estimate: UncertaintyEstimate | None = None,
        episode_registry: EpisodeRegistry | None = None,
        replay_buffer: ReplayBuffer | None = None,
        qcloud_result: QCloudCouplingResult | None = None,
        chemistry_report: ChemistryInterfaceReport | None = None,
        live_features: LiveFeatureVector | None = None,
        repeats: int | None = None,
    ) -> BenchmarkCaseResult:
        resolved_repeats = self._resolve_repeats(repeats)
        return self._measure_case(
            name="controller_decision",
            repeats=resolved_repeats,
            operation=lambda: controller.decide(
                state,
                graph,
                trace_record=trace_record,
                uncertainty_estimate=uncertainty_estimate,
                episode_registry=episode_registry,
                replay_buffer=replay_buffer,
                qcloud_result=qcloud_result,
                chemistry_report=chemistry_report,
                live_features=live_features,
            ),
            metadata_factory=lambda result: {
                "controller": controller.name,
                "highest_priority_action": result.highest_priority_action().kind.value,
                "action_count": len(result.actions),
            },
        )

    def run_foundation_suite(
        self,
        *,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
        force_evaluator: ForceEvaluator,
        graph_manager: ConnectivityGraphManager,
        controller: ExecutiveController,
        previous_graph: ConnectivityGraph | None = None,
        qcloud_coupler: QCloudForceCoupler | None = None,
        qcloud_correction_model: QCloudCorrectionModel | None = None,
        qcloud_region_selector: LocalRegionSelector | None = None,
        compartments: CompartmentRegistry | None = None,
        trace_record: TraceRecord | None = None,
        focus_compartments: Sequence[CompartmentId | str] = (),
        uncertainty_estimate: UncertaintyEstimate | None = None,
        episode_registry: EpisodeRegistry | None = None,
        replay_buffer: ReplayBuffer | None = None,
        qcloud_result: QCloudCouplingResult | None = None,
        residual_model: ResidualModel | None = None,
        chemistry_report: ChemistryInterfaceReport | None = None,
        live_features: LiveFeatureVector | None = None,
        repeats: int | None = None,
    ) -> BenchmarkReport:
        graph_seed = previous_graph or graph_manager.initialize(state, topology)
        cases = [
            self.benchmark_force_evaluation(
                state=state,
                topology=topology,
                forcefield=forcefield,
                force_evaluator=force_evaluator,
                repeats=repeats,
            ),
            self.benchmark_graph_update(
                state=state,
                topology=topology,
                graph_manager=graph_manager,
                previous_graph=graph_seed,
                repeats=repeats,
            ),
        ]

        effective_qcloud_result = qcloud_result
        if qcloud_coupler is not None and qcloud_correction_model is not None and qcloud_region_selector is not None:
            cases.append(
                self.benchmark_qcloud_coupling(
                    state=state,
                    topology=topology,
                    forcefield=forcefield,
                    base_force_evaluator=force_evaluator,
                    coupler=qcloud_coupler,
                    correction_model=qcloud_correction_model,
                    region_selector=qcloud_region_selector,
                    graph=graph_seed,
                    compartments=compartments,
                    trace_record=trace_record,
                    focus_compartments=focus_compartments,
                    repeats=repeats,
                )
            )
            effective_qcloud_result = qcloud_coupler.evaluate_with_selector(
                state=state,
                topology=topology,
                forcefield=forcefield,
                base_force_evaluator=force_evaluator,
                correction_model=qcloud_correction_model,
                region_selector=qcloud_region_selector,
                graph=graph_seed,
                compartments=compartments,
                trace_record=trace_record,
                focus_compartments=focus_compartments,
            )

        if residual_model is not None:
            base_evaluation = force_evaluator.evaluate(state, topology, forcefield)
            cases.append(
                self.benchmark_residual_prediction(
                    state=state,
                    base_evaluation=base_evaluation,
                    residual_model=residual_model,
                    repeats=repeats,
                )
            )

        cases.append(
            self.benchmark_controller_decision(
                state=state,
                graph=graph_seed,
                controller=controller,
                trace_record=trace_record,
                uncertainty_estimate=uncertainty_estimate,
                episode_registry=episode_registry,
                replay_buffer=replay_buffer,
                qcloud_result=effective_qcloud_result,
                chemistry_report=chemistry_report,
                live_features=live_features,
                repeats=repeats,
            )
        )

        return BenchmarkReport(
            suite_name=self.name,
            cases=tuple(cases),
            metadata=FrozenMetadata(
                {
                    "state_id": str(StateId(str(state.provenance.state_id))),
                    "repeat_count": self._resolve_repeats(repeats),
                    "case_count": len(cases),
                }
            ),
        )
