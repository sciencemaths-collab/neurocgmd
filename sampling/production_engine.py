"""Integrated hybrid production engine for coordinated MD runtime orchestration.

[proposed novel]

This module is the repo-native orchestration layer above the classical compute
spine. It does not replace established force, chemistry, ML, qcloud, or
control modules. It coordinates them so they act as one production runtime:

- classical kernels provide the stable force substrate
- graph, chemistry, and structure observers provide context
- memory stores and replay make that context persistent
- qcloud refinement is requested through explicit control decisions
- ML residuals learn bounded corrections instead of owning the dynamics
- validation and benchmark observers remain attached to the same state

The goal is coherent cooperation, not subsystem competition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ai_control import ControllerActionKind, ControllerDecision, ExecutiveController
from benchmarks import BaselineBenchmarkSuite, BenchmarkReport
from chemistry import ChemistryInterfaceAnalyzer, ChemistryInterfaceReport
from compartments import CompartmentRegistry
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.state_registry import LifecycleStage, SimulationStateRegistry
from core.types import FrozenMetadata, StateId
from forcefields import BaseForceField, HybridForceEngine, HybridForceResult
from graph import ConnectivityGraph, ConnectivityGraphManager
from integrators.base import ForceEvaluator, IntegratorStepResult
from integrators.langevin import LangevinIntegrator
from memory import EpisodeKind, EpisodeRegistry, ReplayBuffer, ReplayItem, TraceRecord, TraceStore
from ml import (
    HeuristicUncertaintyModel,
    LiveFeatureEncoder,
    LiveFeatureVector,
    ResidualMemoryModel,
    ResidualPrediction,
    ResidualTarget,
    ResidualModel,
    StateAwareResidualModel,
    UncertaintyEstimate,
)
from physics.forces.composite import BaselineForceEvaluator, ForceEvaluation
from qcloud import (
    LocalRegionSelector,
    NullQCloudCorrectionModel,
    QCloudCorrectionModel,
    QCloudCouplingResult,
    RegionSelectionPolicy,
)
from qcloud.event_analyzer import QCloudEventAnalyzer
from validation import FoundationSanityChecker, SanityCheckReport, TrajectoryDriftChecker

if TYPE_CHECKING:
    from sampling.scenarios import DashboardScenario
    from sampling.scenarios.complex_assembly import ComplexAssemblyProgress, ComplexAssemblySetup
    from validation import DriftCheckReport, FidelityComparisonReport, StructureComparisonReport


def _coerce_metadata(
    metadata: FrozenMetadata | dict[str, object] | None,
) -> FrozenMetadata:
    return metadata if isinstance(metadata, FrozenMetadata) else FrozenMetadata(metadata)


def _normalize_focus_compartments(compartment_ids: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_identifier in compartment_ids or ():
        identifier = str(raw_identifier).strip()
        if not identifier or identifier in seen:
            continue
        normalized.append(identifier)
        seen.add(identifier)
    return tuple(normalized)


def _trace_tags_for_state(
    scenario_name: str,
    progress: "ComplexAssemblyProgress",
    graph: ConnectivityGraph,
) -> tuple[tuple[str, ...], FrozenMetadata]:
    active_edge_count = max(1, len(graph.active_edges()))
    adaptive_ratio = len(graph.adaptive_edges()) / active_edge_count
    tags = ["production", "live", scenario_name]
    if not progress.bound:
        tags.append("search")
    if adaptive_ratio >= 0.45 or not progress.bound:
        tags.extend(("priority", "qcloud", "refine"))
    elif adaptive_ratio >= 0.25:
        tags.append("priority")
    return (
        tuple(dict.fromkeys(tags)),
        FrozenMetadata(
            {
                "adaptive_edge_ratio": adaptive_ratio,
                "qcloud_priority": adaptive_ratio >= 0.45 or not progress.bound,
                "problem_stage": progress.stage_label,
                "assembly_score": progress.assembly_score,
                "bound": progress.bound,
            }
        ),
    )


@dataclass(frozen=True, slots=True)
class ProductionCycleReport(ValidatableComponent):
    """One fully integrated production-cycle report."""

    state: SimulationState
    graph: ConnectivityGraph
    progress: object
    classical_evaluation: ForceEvaluation
    final_evaluation: ForceEvaluation
    hybrid_result: HybridForceResult
    qcloud_result: QCloudCouplingResult | None
    trace_record: TraceRecord
    replay_item: ReplayItem | None
    structure_report: object | None
    fidelity_report: object | None
    chemistry_report: ChemistryInterfaceReport | None
    residual_prediction: ResidualPrediction
    live_features: LiveFeatureVector
    preliminary_uncertainty: UncertaintyEstimate
    final_uncertainty: UncertaintyEstimate
    preliminary_decision: ControllerDecision
    final_decision: ControllerDecision
    sanity_report: SanityCheckReport
    drift_report: object | None = None
    benchmark_report: BenchmarkReport | None = None
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    @property
    def state_id(self) -> StateId:
        return self.state.provenance.state_id

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.graph.particle_count != self.state.particle_count:
            issues.append("graph.particle_count must match the report state particle count.")
        if self.graph.step != self.state.step:
            issues.append("graph.step must match the report state step.")
        if self.trace_record.state_id != self.state.provenance.state_id:
            issues.append("trace_record.state_id must match the report state_id.")
        if self.residual_prediction.state_id != self.state.provenance.state_id:
            issues.append("residual_prediction.state_id must match the report state_id.")
        if self.preliminary_uncertainty.state_id != self.state.provenance.state_id:
            issues.append("preliminary_uncertainty.state_id must match the report state_id.")
        if self.final_uncertainty.state_id != self.state.provenance.state_id:
            issues.append("final_uncertainty.state_id must match the report state_id.")
        if self.preliminary_decision.state_id != self.state.provenance.state_id:
            issues.append("preliminary_decision.state_id must match the report state_id.")
        if self.final_decision.state_id != self.state.provenance.state_id:
            issues.append("final_decision.state_id must match the report state_id.")
        return tuple(issues)


@dataclass(slots=True)
class _PreviewForceEvaluator(ForceEvaluator):
    """Side-effect-free production-force preview for integration substeps."""

    engine: "HybridProductionEngine"
    name: str = "hybrid_production_preview_evaluator"
    classification: str = "[hybrid]"

    def evaluate(
        self,
        state: SimulationState,
        topology,
        forcefield,
    ) -> ForceEvaluation:
        del topology, forcefield
        return self.engine.preview_force_evaluation(state)


@dataclass(slots=True)
class _LightweightForceEvaluator(ForceEvaluator):
    """Classical-only force evaluator for intermediate steps (skips ML/control/graph)."""

    engine: "HybridProductionEngine"
    name: str = "lightweight_classical_evaluator"
    classification: str = "[classical]"

    def evaluate(
        self,
        state: SimulationState,
        topology,
        forcefield,
    ) -> ForceEvaluation:
        del topology, forcefield
        return self.engine._lightweight_force_evaluation(state)


@dataclass(slots=True)
class HybridProductionEngine(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Coordinate the classical, qcloud, chemistry, ML, memory, and control stack."""

    scenario: "DashboardScenario"
    setup: "ComplexAssemblySetup"
    registry: SimulationStateRegistry
    integrator: LangevinIntegrator
    compartments: CompartmentRegistry
    hybrid_engine: HybridForceEngine = field(default_factory=HybridForceEngine)
    graph_manager: ConnectivityGraphManager = field(default_factory=ConnectivityGraphManager)
    trace_store: TraceStore = field(default_factory=TraceStore)
    replay_buffer: ReplayBuffer = field(default_factory=ReplayBuffer)
    episode_registry: EpisodeRegistry = field(default_factory=EpisodeRegistry)
    residual_model: ResidualModel = field(default_factory=ResidualMemoryModel)
    uncertainty_model: HeuristicUncertaintyModel = field(default_factory=HeuristicUncertaintyModel)
    chemistry_analyzer: ChemistryInterfaceAnalyzer = field(default_factory=ChemistryInterfaceAnalyzer)
    live_feature_encoder: LiveFeatureEncoder = field(default_factory=LiveFeatureEncoder)
    controller: ExecutiveController = field(default_factory=ExecutiveController)
    qcloud_selector: LocalRegionSelector = field(
        default_factory=lambda: LocalRegionSelector(
            policy=RegionSelectionPolicy(max_regions=2, max_region_size=4, min_region_score=0.1)
        )
    )
    qcloud_correction_model: QCloudCorrectionModel | None = None
    qcloud_event_analyzer: QCloudEventAnalyzer = field(default_factory=QCloudEventAnalyzer)
    benchmark_suite: BaselineBenchmarkSuite = field(default_factory=BaselineBenchmarkSuite)
    sanity_checker: FoundationSanityChecker = field(default_factory=FoundationSanityChecker)
    drift_checker: TrajectoryDriftChecker = field(default_factory=TrajectoryDriftChecker)
    reference_force_evaluator: BaselineForceEvaluator = field(default_factory=BaselineForceEvaluator)
    default_benchmark_repeats: int = 1
    name: str = "hybrid_production_engine"
    classification: str = "[proposed novel]"
    _last_graph: ConnectivityGraph | None = field(default=None, init=False, repr=False)
    _trajectory_episode_id: str | None = field(default=None, init=False, repr=False)
    _last_cycle_report: ProductionCycleReport | None = field(default=None, init=False, repr=False)
    _cached_force_evaluation: "ForceEvaluation | None" = field(default=None, init=False, repr=False)

    def describe_role(self) -> str:
        return (
            "Runs one coordinated production cycle where classical physics, chemistry, "
            "adaptive graph context, qcloud refinement, ML residuals, memory, and control "
            "work through explicit contracts instead of acting as disconnected demos."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "forcefields/hybrid_engine.py",
            "graph/graph_manager.py",
            "chemistry/interface_logic.py",
            "memory/trace_store.py",
            "memory/replay_buffer.py",
            "memory/episode_registry.py",
            "ml/live_features.py",
            "ml/uncertainty_model.py",
            "qcloud/region_selector.py",
            "ai_control/controller.py",
            "validation/sanity_checks.py",
            "validation/drift_checks.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/hybrid_production_engine.md",)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.setup.topology.particle_count != self.setup.initial_particles.particle_count:
            issues.append("setup.topology must align with setup.initial_particles.")
        if self.compartments.particle_count != self.setup.topology.particle_count:
            issues.append("compartments must align with the production topology.")
        if self.default_benchmark_repeats <= 0:
            issues.append("default_benchmark_repeats must be strictly positive.")
        issues.extend(self.hybrid_engine.validate())
        issues.extend(self.live_feature_encoder.validate())
        issues.extend(self.controller.validate())
        issues.extend(self.uncertainty_model.validate())
        issues.extend(self.chemistry_analyzer.validate())
        issues.extend(self.benchmark_suite.validate())
        issues.extend(self.sanity_checker.validate())
        issues.extend(self.drift_checker.validate())
        return tuple(issues)

    def current_state(self) -> SimulationState:
        return self.registry.latest_state()

    def current_graph(self) -> ConnectivityGraph:
        state = self.current_state()
        if self._last_graph is None:
            self._last_graph = self.graph_manager.initialize(state, self.setup.topology)
        elif self._last_graph.particle_count != state.particle_count:
            self._last_graph = self.graph_manager.initialize(state, self.setup.topology)
        elif self._last_graph.step < state.step:
            self._last_graph = self.graph_manager.update(state, self.setup.topology, self._last_graph)
        elif self._last_graph.step > state.step:
            self._last_graph = self.graph_manager.initialize(state, self.setup.topology)
        return self._last_graph

    def preview_force_evaluation(self, state: SimulationState) -> ForceEvaluation:
        if self._cached_force_evaluation is not None:
            cached = self._cached_force_evaluation
            self._cached_force_evaluation = None
            return cached
        return self._lightweight_force_evaluation(state)

    def _lightweight_force_evaluation(self, state: SimulationState) -> ForceEvaluation:
        """Classical-only force evaluation skipping graph/ML/control/drift."""
        result = self.hybrid_engine.evaluate_detailed(
            state,
            self.setup.topology,
            self.setup.forcefield,
        )
        return result.classical_evaluation

    def collect_cycle(
        self,
        *,
        benchmark_repeats: int | None = None,
    ) -> ProductionCycleReport:
        report = self._evaluate_state(self.current_state(), record=True, benchmark_repeats=benchmark_repeats)
        self._last_cycle_report = report
        return report

    def advance(
        self,
        steps: int = 1,
        *,
        record_final_state: bool = False,
        benchmark_repeats: int | None = None,
        full_eval: bool = True,
    ) -> ProductionCycleReport | None:
        if steps < 0:
            raise ContractValidationError("steps must be non-negative.")
        if steps == 0:
            return self.collect_cycle(benchmark_repeats=benchmark_repeats) if record_final_state else self._last_cycle_report

        preview_evaluator = _PreviewForceEvaluator(self)
        lightweight_evaluator = _LightweightForceEvaluator(self)
        current_state = self.current_state()
        for _ in range(steps):
            if full_eval:
                report = self._evaluate_state(current_state, record=True, benchmark_repeats=None)
                self._cached_force_evaluation = report.final_evaluation
                active_evaluator = preview_evaluator
            else:
                self._cached_force_evaluation = self._lightweight_force_evaluation(current_state)
                active_evaluator = lightweight_evaluator
            step_result = self.integrator.step(
                current_state,
                self.setup.topology,
                self.setup.forcefield,
                active_evaluator,
            )
            self._cached_force_evaluation = None
            current_state = self.registry.derive_state(
                current_state,
                particles=step_result.particles,
                time=step_result.time,
                step=step_result.step,
                potential_energy=step_result.potential_energy,
                observables=step_result.observables,
                stage=LifecycleStage.INTEGRATION,
                notes="hybrid production step",
                metadata=FrozenMetadata(
                    {
                        "integrator": self.integrator.name,
                        "force_evaluator": active_evaluator.name,
                        **step_result.metadata.to_dict(),
                    }
                ),
                created_by=self.name,
            )
            # Graph cache is preserved across steps; _graph_for_state() uses
            # graph_manager.update() to refresh it incrementally.
        return self.collect_cycle(benchmark_repeats=benchmark_repeats) if record_final_state else None

    def _evaluate_state(
        self,
        state: SimulationState,
        *,
        record: bool,
        benchmark_repeats: int | None,
    ) -> ProductionCycleReport:
        graph = self._graph_for_state(state)
        progress = self.scenario.measure_progress(state, graph=graph)
        self._ensure_trajectory_episode(state, record=record)
        chemistry_report = self._build_chemistry_report(state, progress)
        structure_report = self.scenario.build_structure_report(state, progress=progress)
        trace_record = self._trace_record_for_state(state, graph, progress, record=record)
        replay_item = self._replay_item_for_trace(trace_record, progress, record=record)

        classical_result = self.hybrid_engine.evaluate_detailed(
            state,
            self.setup.topology,
            self.setup.forcefield,
        )
        preliminary_prediction = self.residual_model.predict(
            state,
            classical_result.classical_evaluation,
        )
        preliminary_live_features = self.live_feature_encoder.encode(
            state,
            graph,
            progress=progress,
            chemistry_report=chemistry_report,
            structure_report=structure_report,
            fidelity_report=None,
        )
        preliminary_uncertainty = self.uncertainty_model.estimate(
            preliminary_prediction,
            trace_record=trace_record,
            replay_item=replay_item,
            live_features=preliminary_live_features,
        )
        preliminary_decision = self.controller.decide(
            state,
            graph,
            trace_record=trace_record,
            uncertainty_estimate=preliminary_uncertainty,
            episode_registry=self.episode_registry,
            replay_buffer=self.replay_buffer,
            chemistry_report=chemistry_report,
            live_features=preliminary_live_features,
        )

        focus_compartments = _normalize_focus_compartments(
            preliminary_decision.allocation.focus_compartments or self.setup.focus_compartments
        )
        production_result = self._run_hybrid_force_path(
            state,
            graph=graph,
            trace_record=trace_record,
            focus_compartments=focus_compartments,
            qcloud_budget=preliminary_decision.allocation.qcloud_region_budget,
            cached_classical=classical_result,
        )
        fidelity_report = self.scenario.build_fidelity_report(
            state,
            baseline_evaluation=classical_result.classical_evaluation,
            corrected_evaluation=production_result.final_evaluation,
            progress=progress,
        )
        residual_prediction = production_result.residual_prediction or preliminary_prediction
        final_live_features = self.live_feature_encoder.encode(
            state,
            graph,
            progress=progress,
            chemistry_report=chemistry_report,
            structure_report=structure_report,
            fidelity_report=fidelity_report,
        )
        replay_item = self._replay_item_for_final_result(
            trace_record,
            preliminary_uncertainty=preliminary_uncertainty,
            qcloud_result=production_result.qcloud_result,
            focus_compartments=focus_compartments,
            record=record,
        )
        final_uncertainty = self.uncertainty_model.estimate(
            residual_prediction,
            trace_record=trace_record,
            replay_item=replay_item,
            live_features=final_live_features,
        )
        final_decision = self.controller.decide(
            state,
            graph,
            trace_record=trace_record,
            uncertainty_estimate=final_uncertainty,
            episode_registry=self.episode_registry,
            replay_buffer=self.replay_buffer,
            qcloud_result=production_result.qcloud_result,
            chemistry_report=chemistry_report,
            live_features=final_live_features,
        )
        if record:
            self._apply_post_cycle_learning(
                state,
                production_result,
                final_decision,
            )
            self._apply_control_episodes(
                state,
                final_decision,
            )
            # Feed QCloud results to the event analyzer for structural event
            # detection and adaptive region feedback.
            self.qcloud_event_analyzer.record_qcloud_result(
                production_result.qcloud_result,
                step=state.step,
                time=state.time,
            )

        _DRIFT_WINDOW = 50
        all_ids = self.registry.state_ids()
        window_ids = all_ids[-_DRIFT_WINDOW:] if len(all_ids) > _DRIFT_WINDOW else all_ids
        states = tuple(self.registry.get_state(sid) for sid in window_ids)
        drift_report = self.drift_checker.assess(states) if states else None
        sanity_report = self.sanity_checker.run(
            state,
            topology=self.setup.topology,
            force_evaluation=production_result.final_evaluation,
            graph=graph,
            qcloud_result=production_result.qcloud_result,
            residual_prediction=residual_prediction,
            controller_decision=final_decision,
        )
        benchmark_report = None
        resolved_benchmark_repeats = benchmark_repeats if benchmark_repeats is not None else self.default_benchmark_repeats
        if record and resolved_benchmark_repeats > 0:
            benchmark_report = self.benchmark_suite.run_foundation_suite(
                state=state,
                topology=self.setup.topology,
                forcefield=self.setup.forcefield,
                force_evaluator=self.reference_force_evaluator,
                graph_manager=self.graph_manager,
                controller=self.controller,
                previous_graph=graph,
                qcloud_coupler=self.hybrid_engine.qcloud_coupler,
                qcloud_correction_model=self._effective_qcloud_correction_model(),
                qcloud_region_selector=self.qcloud_selector,
                compartments=self.compartments,
                trace_record=trace_record,
                focus_compartments=focus_compartments,
                uncertainty_estimate=final_uncertainty,
                episode_registry=self.episode_registry,
                replay_buffer=self.replay_buffer,
                qcloud_result=production_result.qcloud_result,
                residual_model=self.residual_model,
                chemistry_report=chemistry_report,
                live_features=final_live_features,
                repeats=resolved_benchmark_repeats,
            )

        return ProductionCycleReport(
            state=state,
            graph=graph,
            progress=progress,
            classical_evaluation=classical_result.classical_evaluation,
            final_evaluation=production_result.final_evaluation,
            hybrid_result=production_result,
            qcloud_result=production_result.qcloud_result,
            trace_record=trace_record,
            replay_item=replay_item,
            structure_report=structure_report,
            fidelity_report=fidelity_report,
            chemistry_report=chemistry_report,
            residual_prediction=residual_prediction,
            live_features=final_live_features,
            preliminary_uncertainty=preliminary_uncertainty,
            final_uncertainty=final_uncertainty,
            preliminary_decision=preliminary_decision,
            final_decision=final_decision,
            sanity_report=sanity_report,
            drift_report=drift_report,
            benchmark_report=benchmark_report,
            metadata=FrozenMetadata(
                {
                    "scenario": self.scenario.name,
                    "recorded": record,
                    "focus_compartments": focus_compartments,
                    "qcloud_applied": production_result.qcloud_result is not None,
                    "selected_region_count": (
                        len(production_result.qcloud_result.selected_regions)
                        if production_result.qcloud_result is not None
                        else 0
                    ),
                    "replay_buffer_size": len(self.replay_buffer),
                    "trace_record_count": len(self.trace_store),
                }
            ),
        )

    def _effective_qcloud_correction_model(self) -> QCloudCorrectionModel:
        return self.qcloud_correction_model or NullQCloudCorrectionModel()

    def _graph_for_state(self, state: SimulationState) -> ConnectivityGraph:
        if self._last_graph is None:
            self._last_graph = self.graph_manager.initialize(state, self.setup.topology)
        elif self._last_graph.particle_count != state.particle_count:
            self._last_graph = self.graph_manager.initialize(state, self.setup.topology)
        elif self._last_graph.step < state.step:
            self._last_graph = self.graph_manager.update(state, self.setup.topology, self._last_graph)
        elif self._last_graph.step > state.step:
            self._last_graph = self.graph_manager.initialize(state, self.setup.topology)
        return self._last_graph

    def _build_chemistry_report(
        self,
        state: SimulationState,
        progress: "ComplexAssemblyProgress",
    ) -> ChemistryInterfaceReport | None:
        focus_compartments = _normalize_focus_compartments(self.setup.focus_compartments)
        if len(focus_compartments) < 2:
            return None
        distance_cutoff = max(progress.capture_distance, progress.contact_distance + 0.25)
        return self.chemistry_analyzer.assess(
            state,
            self.setup.topology,
            self.compartments,
            compartment_ids=(focus_compartments[0], focus_compartments[1]),
            distance_cutoff=distance_cutoff,
        )

    def _trace_record_for_state(
        self,
        state: SimulationState,
        graph: ConnectivityGraph,
        progress: "ComplexAssemblyProgress",
        *,
        record: bool,
    ) -> TraceRecord:
        if state.provenance.state_id in self.trace_store.state_ids():
            return self.trace_store.get_record(state.provenance.state_id)
        tags, metadata = _trace_tags_for_state(self.scenario.name, progress, graph)
        reference_case = self.scenario.reference_case()
        payload = metadata.with_updates(
            {
                "reference_case": reference_case.name if reference_case is not None else None,
            }
        )
        if record:
            return self.trace_store.append_state(
                state,
                graph=graph,
                compartments=self.compartments,
                tags=tags,
                metadata=payload,
            )
        return TraceRecord(
            record_id=f"preview-{state.step:08d}",
            simulation_id=state.provenance.simulation_id,
            state_id=state.provenance.state_id,
            parent_state_id=state.provenance.parent_state_id,
            stage=state.provenance.stage,
            step=state.step,
            time=state.time,
            particle_count=state.particle_count,
            kinetic_energy=state.kinetic_energy(),
            potential_energy=state.potential_energy,
            active_edge_count=len(graph.active_edges()),
            structural_edge_count=len(graph.structural_edges()),
            adaptive_edge_count=len(graph.adaptive_edges()),
            compartment_ids=tuple(str(domain.compartment_id) for domain in self.compartments.domains),
            tags=tags,
            metadata=payload,
        )

    def _replay_item_for_trace(
        self,
        trace_record: TraceRecord,
        progress: "ComplexAssemblyProgress",
        *,
        record: bool,
    ) -> ReplayItem | None:
        if not record:
            return None
        return self.replay_buffer.add_from_record(
            trace_record,
            score=max(1.0, 1.0 - progress.assembly_score + progress.interface_distance),
            tags=("production",),
            metadata={
                "phase": "preliminary",
                "assembly_score": progress.assembly_score,
                "interface_distance": progress.interface_distance,
            },
        )

    def _run_hybrid_force_path(
        self,
        state: SimulationState,
        *,
        graph: ConnectivityGraph,
        trace_record: TraceRecord,
        focus_compartments: tuple[str, ...],
        qcloud_budget: int,
        cached_classical: "HybridForceResult | None" = None,
    ) -> HybridForceResult:
        correction_model = self._effective_qcloud_correction_model()
        if qcloud_budget <= 0:
            return self.hybrid_engine.evaluate_detailed(
                state,
                self.setup.topology,
                self.setup.forcefield,
                residual_model=self.residual_model,
                cached_classical=cached_classical,
            )
        selector = self.qcloud_selector
        if qcloud_budget < self.qcloud_selector.policy.max_regions:
            selector = LocalRegionSelector(
                policy=RegionSelectionPolicy(
                    max_regions=max(1, qcloud_budget),
                    max_region_size=max(2, self.qcloud_selector.policy.max_region_size),
                    bonded_neighbor_hops=self.qcloud_selector.policy.bonded_neighbor_hops,
                    min_region_score=self.qcloud_selector.policy.min_region_score,
                    inter_compartment_bonus=self.qcloud_selector.policy.inter_compartment_bonus,
                    long_range_bonus=self.qcloud_selector.policy.long_range_bonus,
                    memory_priority_bonus=self.qcloud_selector.policy.memory_priority_bonus,
                    focus_compartment_bonus=self.qcloud_selector.policy.focus_compartment_bonus,
                )
            )
        # Feed correction history back to guide region selection —
        # particles that previously received large corrections get priority.
        priority_scores = self.qcloud_event_analyzer.get_particle_priority_scores()
        return self.hybrid_engine.evaluate_detailed(
            state,
            self.setup.topology,
            self.setup.forcefield,
            graph=graph,
            compartments=self.compartments,
            trace_record=trace_record,
            focus_compartments=focus_compartments,
            region_selector=selector,
            correction_model=correction_model,
            residual_model=self.residual_model,
            cached_classical=cached_classical,
            correction_priority_scores=priority_scores or None,
        )

    def _replay_item_for_final_result(
        self,
        trace_record: TraceRecord,
        *,
        preliminary_uncertainty: UncertaintyEstimate,
        qcloud_result: QCloudCouplingResult | None,
        focus_compartments: tuple[str, ...],
        record: bool,
    ) -> ReplayItem | None:
        if not record:
            return None
        return self.replay_buffer.add_from_record(
            trace_record,
            score=max(preliminary_uncertainty.total_uncertainty, 0.5),
            tags=("qcloud",) if qcloud_result is not None else ("production",),
            metadata={
                "phase": "final",
                "selected_region_count": len(qcloud_result.selected_regions) if qcloud_result is not None else 0,
                "focus_compartments": focus_compartments,
            },
        )

    def _apply_post_cycle_learning(
        self,
        state: SimulationState,
        hybrid_result: HybridForceResult,
        final_decision: ControllerDecision,
    ) -> None:
        if hybrid_result.qcloud_result is None or not hybrid_result.qcloud_result.applied_corrections:
            return
        if final_decision.allocation.ml_example_budget <= 0:
            return
        target = ResidualTarget.from_corrections(
            state.provenance.state_id,
            hybrid_result.qcloud_result.applied_corrections,
            metadata={"source": self.name},
        )
        if isinstance(self.residual_model, StateAwareResidualModel):
            self.residual_model.observe_state(
                state,
                hybrid_result.classical_evaluation,
                target,
                sample_weight=1.0,
            )
            return
        self.residual_model.observe(target, sample_weight=1.0)

    def _apply_control_episodes(
        self,
        state: SimulationState,
        final_decision: ControllerDecision,
    ) -> None:
        action_kinds = {action.kind for action in final_decision.actions}
        if ControllerActionKind.OPEN_INSTABILITY_EPISODE in action_kinds:
            if not any(
                episode.kind == EpisodeKind.INSTABILITY and episode.contains_state(state.provenance.state_id)
                for episode in self.episode_registry.open_episodes()
            ):
                self.episode_registry.open_episode(
                    state,
                    kind=EpisodeKind.INSTABILITY,
                    tags=("instability", "production"),
                    metadata={"source": self.name},
                )
        if ControllerActionKind.REVIEW_COMPARTMENT_FOCUS in action_kinds:
            if not any(
                episode.kind == EpisodeKind.COMPARTMENT_FOCUS and episode.contains_state(state.provenance.state_id)
                for episode in self.episode_registry.open_episodes()
            ):
                self.episode_registry.open_episode(
                    state,
                    kind=EpisodeKind.COMPARTMENT_FOCUS,
                    tags=("compartment_focus", "production"),
                    metadata={"source": self.name},
                )

    def _ensure_trajectory_episode(
        self,
        state: SimulationState,
        *,
        record: bool,
    ) -> None:
        if not record:
            return
        if self._trajectory_episode_id is None:
            episode = self.episode_registry.open_episode(
                state,
                kind=EpisodeKind.TRAJECTORY,
                tags=("trajectory", "production"),
                metadata={"source": self.name},
            )
            self._trajectory_episode_id = episode.episode_id
            return
        episode = self.episode_registry.get_episode(self._trajectory_episode_id)
        if not episode.contains_state(state.provenance.state_id):
            self.episode_registry.append_state(self._trajectory_episode_id, state)


__all__ = [
    "HybridProductionEngine",
    "ProductionCycleReport",
]
