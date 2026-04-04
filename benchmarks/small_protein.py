"""Fast small-protein benchmark built on the arbitrary-protein import pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from ai_control import ControllerActionKind, ExecutiveController
from compartments import CompartmentRegistry
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.types import FrozenMetadata, SimulationId, StateId
from forcefields import HybridForceEngine, ImportedProteinForceFieldBuilder
from graph import ConnectivityGraph, ConnectivityGraphManager
from integrators import LangevinIntegrator
from integrators.base import ForceEvaluator
from memory import EpisodeKind, EpisodeRegistry, ReplayBuffer, ReplayItem, TraceStore
from ml import HeuristicUncertaintyModel, ScalableResidualModel, UncertaintyEstimate
from optimization import BackendExecutionPlan, BackendExecutionPlanner, BackendExecutionRequest
from physics.forces.composite import BaselineForceEvaluator, ForceEvaluation
from qcloud import LocalRegionSelector, RefinementRegion, RegionSelectionPolicy, RegionTriggerKind
from topology import ImportedProteinSystem, ProteinCoarseMapper, SystemTopology
from validation import BackendParityReport, BackendParityValidator

from benchmarks.baseline_suite import BenchmarkCaseResult, BenchmarkReport
from config import ProteinEntityGroup, ProteinMappingConfig
from forcefields.base_forcefield import BaseForceField
from qcloud.protein_shadow_tuning import ProteinShadowRuntimeBundle, ProteinShadowTuner

_CLASSICAL_ONLY_CASE = "classical_only"
_HYBRID_PRODUCTION_CASE = "production_hybrid_engine"
_ENGINE_MODE_CASES = (_CLASSICAL_ONLY_CASE, _HYBRID_PRODUCTION_CASE)


def _default_barnase_pdb_path() -> str:
    return str((Path(__file__).resolve().parent / "reference_cases" / "data" / "1BRS.pdb").resolve())


def _coerce_metadata(
    metadata: FrozenMetadata | dict[str, object] | None,
) -> FrozenMetadata:
    return metadata if isinstance(metadata, FrozenMetadata) else FrozenMetadata(metadata)


def _trace_tags_for_graph(
    graph: ConnectivityGraph,
) -> tuple[tuple[str, ...], FrozenMetadata]:
    active_edge_count = max(1, len(graph.active_edges()))
    adaptive_ratio = len(graph.adaptive_edges()) / active_edge_count
    tags = ["small_protein", "production"]
    if adaptive_ratio >= 0.45:
        tags.extend(("priority", "qcloud", "refine"))
    elif adaptive_ratio >= 0.25:
        tags.append("priority")
    return (
        tuple(tags),
        FrozenMetadata(
            {
                "adaptive_edge_ratio": adaptive_ratio,
                "qcloud_priority": adaptive_ratio >= 0.45,
            }
        ),
    )


@dataclass(frozen=True, slots=True)
class SmallProteinBenchmarkSpec(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """[hybrid] Serializable specification for a fast small-protein benchmark."""

    name: str = "barnase_small_protein_benchmark"
    classification: str = "[hybrid]"
    pdb_path: str = field(default_factory=_default_barnase_pdb_path)
    entity_group: ProteinEntityGroup = field(
        default_factory=lambda: ProteinEntityGroup(
            entity_id="barnase",
            chain_ids=("A",),
            description="Single-chain small-protein benchmark entity.",
        )
    )
    residues_per_bead: int = 10
    repeats: int = 20
    rollout_steps: int = 4
    warmup_training_passes: int = 6
    base_time_step: float = 0.02
    base_friction: float = 0.6
    manual_region_size: int = 6
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Defines one fast, real-structure small-protein benchmark that exercises "
            "the importer, classical kernels, hybrid force engine, and short-step runtime."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "benchmarks/reference_cases/data/1BRS.pdb",
            "topology/protein_coarse_mapping.py",
            "forcefields/hybrid_engine.py",
            "validation/backend_parity.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/small_protein_benchmark.md",)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name.strip():
            issues.append("name must be a non-empty string.")
        if not self.pdb_path.strip():
            issues.append("pdb_path must be a non-empty string.")
        issues.extend(self.entity_group.validate())
        if self.residues_per_bead <= 0:
            issues.append("residues_per_bead must be strictly positive.")
        if self.repeats <= 0:
            issues.append("repeats must be strictly positive.")
        if self.rollout_steps <= 0:
            issues.append("rollout_steps must be strictly positive.")
        if self.warmup_training_passes < 0:
            issues.append("warmup_training_passes must be non-negative.")
        if self.base_time_step <= 0.0:
            issues.append("base_time_step must be strictly positive.")
        if self.base_friction < 0.0:
            issues.append("base_friction must be non-negative.")
        if self.manual_region_size <= 0:
            issues.append("manual_region_size must be strictly positive.")
        return tuple(issues)

    def mapping_config(self) -> ProteinMappingConfig:
        return ProteinMappingConfig(residues_per_bead=self.residues_per_bead)


@dataclass(frozen=True, slots=True)
class SmallProteinBenchmarkReport(ValidatableComponent):
    """Structured report for one small-protein benchmark run."""

    spec: SmallProteinBenchmarkSpec
    structure_id: str
    entity_id: str
    residue_count: int
    bead_count: int
    benchmark_report: BenchmarkReport
    parity_report: BackendParityReport
    execution_plan: BackendExecutionPlan
    recommended_time_step: float
    recommended_friction: float
    selected_region_size: int
    trained_state_count: int
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.structure_id.strip():
            issues.append("structure_id must be a non-empty string.")
        if not self.entity_id.strip():
            issues.append("entity_id must be a non-empty string.")
        if self.residue_count <= 0:
            issues.append("residue_count must be strictly positive.")
        if self.bead_count <= 0:
            issues.append("bead_count must be strictly positive.")
        if self.recommended_time_step <= 0.0:
            issues.append("recommended_time_step must be strictly positive.")
        if self.recommended_friction < 0.0:
            issues.append("recommended_friction must be non-negative.")
        if self.selected_region_size <= 0:
            issues.append("selected_region_size must be strictly positive.")
        if self.trained_state_count < 0:
            issues.append("trained_state_count must be non-negative.")
        issues.extend(self.benchmark_report.validate())
        issues.extend(self.parity_report.validate())
        issues.extend(self.execution_plan.validate())
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        engine_modes = self.engine_mode_summary()
        diagnostic_cases = self.diagnostic_cases()
        return {
            "spec": {
                "name": self.spec.name,
                "pdb_path": self.spec.pdb_path,
                "entity_group": self.spec.entity_group.to_dict(),
                "residues_per_bead": self.spec.residues_per_bead,
                "repeats": self.spec.repeats,
                "rollout_steps": self.spec.rollout_steps,
                "warmup_training_passes": self.spec.warmup_training_passes,
                "base_time_step": self.spec.base_time_step,
                "base_friction": self.spec.base_friction,
                "manual_region_size": self.spec.manual_region_size,
                "metadata": self.spec.metadata.to_dict(),
            },
            "structure_id": self.structure_id,
            "entity_id": self.entity_id,
            "residue_count": self.residue_count,
            "bead_count": self.bead_count,
            "recommended_time_step": self.recommended_time_step,
            "recommended_friction": self.recommended_friction,
            "selected_region_size": self.selected_region_size,
            "trained_state_count": self.trained_state_count,
            "engine_modes": {
                mode: {
                    "display_name": summary["display_name"],
                    "case_name": summary["case_name"],
                    "average_seconds_per_iteration": summary["average_seconds_per_iteration"],
                    "relative_to_classical_only": summary["relative_to_classical_only"],
                }
                for mode, summary in engine_modes.items()
            },
            "diagnostic_cases": [
                {
                    "name": case.name,
                    "iterations": case.iterations,
                    "elapsed_seconds": case.elapsed_seconds,
                    "average_seconds_per_iteration": case.average_seconds_per_iteration(),
                    "metadata": case.metadata.to_dict(),
                }
                for case in diagnostic_cases
            ],
            "benchmark_report": {
                "suite_name": self.benchmark_report.suite_name,
                "metadata": self.benchmark_report.metadata.to_dict(),
                "cases": [
                    {
                        "name": case.name,
                        "iterations": case.iterations,
                        "elapsed_seconds": case.elapsed_seconds,
                        "average_seconds_per_iteration": case.average_seconds_per_iteration(),
                        "metadata": case.metadata.to_dict(),
                    }
                    for case in self.benchmark_report.cases
                ],
            },
            "parity_report": {
                "target_component": self.parity_report.target_component,
                "backend_name": self.parity_report.backend_name,
                "all_passed": self.parity_report.all_passed(),
                "metadata": self.parity_report.metadata.to_dict(),
                "metrics": [
                    {
                        "label": metric.label,
                        "reference_value": metric.reference_value,
                        "candidate_value": metric.candidate_value,
                        "absolute_error": metric.absolute_error,
                        "tolerance": metric.tolerance,
                        "passed": metric.passed,
                    }
                    for metric in self.parity_report.metrics
                ],
            },
            "execution_plan": {
                "execution_mode": self.execution_plan.execution_mode,
                "selected_backend": self.execution_plan.selection.selected_backend,
                "selection_rationale": self.execution_plan.selection.rationale,
                "partition": {
                    "chunk_count": self.execution_plan.partition.chunk_count,
                    "chunk_size": self.execution_plan.partition.chunk_size,
                    "vector_width": self.execution_plan.partition.vector_width,
                    "metadata": self.execution_plan.partition.metadata.to_dict(),
                },
                "metadata": self.execution_plan.metadata.to_dict(),
            },
            "metadata": self.metadata.to_dict(),
        }

    def engine_mode_summary(self) -> dict[str, dict[str, object]]:
        classical_case = self.benchmark_report.case_for(_CLASSICAL_ONLY_CASE)
        production_case = self.benchmark_report.case_for(_HYBRID_PRODUCTION_CASE)
        classical_seconds = classical_case.average_seconds_per_iteration()
        production_seconds = production_case.average_seconds_per_iteration()
        return {
            "classical_only": {
                "display_name": "classical_only",
                "case_name": classical_case.name,
                "average_seconds_per_iteration": classical_seconds,
                "relative_to_classical_only": 1.0,
            },
            "hybrid_production": {
                "display_name": "production_hybrid_engine",
                "case_name": production_case.name,
                "average_seconds_per_iteration": production_seconds,
                "relative_to_classical_only": (
                    production_seconds / classical_seconds if classical_seconds > 0.0 else 0.0
                ),
            },
        }

    def diagnostic_cases(self) -> tuple[BenchmarkCaseResult, ...]:
        return tuple(
            case
            for case in self.benchmark_report.cases
            if case.name not in _ENGINE_MODE_CASES
        )


@dataclass(slots=True)
class _HybridBenchmarkEvaluator(ForceEvaluator):
    """Fixed-configuration hybrid evaluator for timing short integration steps."""

    hybrid_engine: HybridForceEngine
    topology: SystemTopology
    forcefield: BaseForceField
    selected_regions: tuple[RefinementRegion, ...]
    correction_model: object
    residual_model: ScalableResidualModel | None = None
    name: str = "hybrid_benchmark_evaluator"
    classification: str = "[test]"

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
    ) -> ForceEvaluation:
        del topology, forcefield
        return self.hybrid_engine.evaluate_detailed(
            state,
            self.topology,
            self.forcefield,
            selected_regions=self.selected_regions,
            correction_model=self.correction_model,
            residual_model=self.residual_model,
        ).final_evaluation


@dataclass(frozen=True, slots=True)
class _ProductionCycleResult:
    """One interconnected production-cycle result for the benchmark harness."""

    force_evaluation: ForceEvaluation
    selected_region_count: int
    qcloud_requested: bool
    qcloud_applied: bool
    focus_compartments: tuple[str, ...]
    preliminary_uncertainty: UncertaintyEstimate
    final_uncertainty: UncertaintyEstimate
    preliminary_action: str
    final_action: str
    replay_item: ReplayItem | None
    trace_record_count: int
    replay_buffer_size: int
    open_episode_count: int


@dataclass(slots=True)
class _InterconnectedProductionEvaluator(ForceEvaluator):
    """Production evaluator that routes through graph, memory, controller, qcloud, and ML."""

    topology: SystemTopology
    forcefield: BaseForceField
    compartments: CompartmentRegistry
    default_focus_compartments: tuple[str, ...]
    hybrid_engine: HybridForceEngine = field(default_factory=HybridForceEngine)
    graph_manager: ConnectivityGraphManager = field(default_factory=ConnectivityGraphManager)
    controller: ExecutiveController = field(default_factory=ExecutiveController)
    uncertainty_model: HeuristicUncertaintyModel = field(default_factory=HeuristicUncertaintyModel)
    residual_model: ScalableResidualModel = field(default_factory=ScalableResidualModel)
    correction_model: object | None = None
    max_region_size: int = 6
    name: str = "interconnected_production_evaluator"
    classification: str = "[hybrid]"
    trace_store: TraceStore = field(init=False, repr=False)
    replay_buffer: ReplayBuffer = field(init=False, repr=False)
    episode_registry: EpisodeRegistry = field(init=False, repr=False)
    _trajectory_episode_id: str | None = field(default=None, init=False, repr=False)
    last_cycle_result: _ProductionCycleResult | None = field(default=None, init=False, repr=False)
    _last_graph: ConnectivityGraph | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.trace_store = TraceStore()
        self.replay_buffer = ReplayBuffer()
        self.episode_registry = EpisodeRegistry()

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
    ) -> ForceEvaluation:
        del topology, forcefield
        cycle_result = self.run_cycle(state)
        self.last_cycle_result = cycle_result
        return cycle_result.force_evaluation

    def run_cycle(self, state: SimulationState) -> _ProductionCycleResult:
        graph = self._graph_for_state(state)
        self._ensure_trajectory_episode(state)

        trace_record = self._trace_record_for_state(state, graph)
        adaptive_ratio = float(trace_record.metadata["adaptive_edge_ratio"])
        replay_item = self.replay_buffer.add_from_record(
            trace_record,
            score=max(1.0, adaptive_ratio),
            tags=("production",),
            metadata={"phase": "preliminary", "adaptive_edge_ratio": adaptive_ratio},
        )

        classical_result = self.hybrid_engine.evaluate_detailed(
            state,
            self.topology,
            self.forcefield,
        )
        preliminary_prediction = self.residual_model.predict(
            state,
            classical_result.classical_evaluation,
        )
        preliminary_uncertainty = self.uncertainty_model.estimate(
            preliminary_prediction,
            trace_record=trace_record,
            replay_item=replay_item,
        )
        preliminary_decision = self.controller.decide(
            state,
            graph,
            trace_record=trace_record,
            uncertainty_estimate=preliminary_uncertainty,
            episode_registry=self.episode_registry,
            replay_buffer=self.replay_buffer,
        )

        focus_compartments = (
            preliminary_decision.allocation.focus_compartments or self.default_focus_compartments
        )
        qcloud_requested = preliminary_decision.allocation.qcloud_region_budget > 0
        production_result = classical_result
        if qcloud_requested and self.correction_model is not None:
            region_selector = LocalRegionSelector(
                policy=RegionSelectionPolicy(
                    max_regions=max(1, preliminary_decision.allocation.qcloud_region_budget),
                    max_region_size=max(2, self.max_region_size),
                    min_region_score=0.0,
                )
            )
            production_result = self.hybrid_engine.evaluate_detailed(
                state,
                self.topology,
                self.forcefield,
                graph=graph,
                compartments=self.compartments,
                trace_record=trace_record,
                focus_compartments=focus_compartments,
                region_selector=region_selector,
                correction_model=self.correction_model,
                residual_model=self.residual_model,
            )
        elif self.correction_model is not None:
            production_result = self.hybrid_engine.evaluate_detailed(
                state,
                self.topology,
                self.forcefield,
                residual_model=self.residual_model,
            )

        replay_item = self.replay_buffer.add_from_record(
            trace_record,
            score=max(preliminary_uncertainty.total_uncertainty, 1.0 if qcloud_requested else 0.5),
            tags=("qcloud",) if production_result.qcloud_result is not None else ("production",),
            metadata={
                "phase": "final",
                "selected_region_count": (
                    len(production_result.qcloud_result.selected_regions)
                    if production_result.qcloud_result is not None
                    else 0
                ),
            },
        )

        final_prediction = production_result.residual_prediction or preliminary_prediction
        final_uncertainty = self.uncertainty_model.estimate(
            final_prediction,
            trace_record=trace_record,
            replay_item=replay_item,
        )
        final_decision = self.controller.decide(
            state,
            graph,
            trace_record=trace_record,
            uncertainty_estimate=final_uncertainty,
            episode_registry=self.episode_registry,
            replay_buffer=self.replay_buffer,
            qcloud_result=production_result.qcloud_result,
        )

        if (
            production_result.residual_target is not None
            and final_decision.allocation.ml_example_budget > 0
        ):
            self.residual_model.observe_state(
                state,
                production_result.classical_evaluation,
                production_result.residual_target,
                sample_weight=1.0,
            )

        if any(
            action.kind == ControllerActionKind.OPEN_INSTABILITY_EPISODE
            for action in final_decision.actions
        ):
            self.episode_registry.open_episode(
                state,
                kind=EpisodeKind.INSTABILITY,
                tags=("instability", "production"),
                metadata={"source": self.name},
            )

        return _ProductionCycleResult(
            force_evaluation=production_result.final_evaluation,
            selected_region_count=(
                len(production_result.qcloud_result.selected_regions)
                if production_result.qcloud_result is not None
                else 0
            ),
            qcloud_requested=qcloud_requested,
            qcloud_applied=production_result.qcloud_result is not None,
            focus_compartments=focus_compartments,
            preliminary_uncertainty=preliminary_uncertainty,
            final_uncertainty=final_uncertainty,
            preliminary_action=preliminary_decision.highest_priority_action().kind.value,
            final_action=final_decision.highest_priority_action().kind.value,
            replay_item=replay_item,
            trace_record_count=len(self.trace_store),
            replay_buffer_size=len(self.replay_buffer),
            open_episode_count=len(self.episode_registry.open_episodes()),
        )

    def _ensure_trajectory_episode(self, state: SimulationState) -> None:
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
        if episode.status.value != "open":
            return
        if not episode.contains_state(state.provenance.state_id):
            self.episode_registry.append_state(self._trajectory_episode_id, state)

    def _graph_for_state(self, state: SimulationState) -> ConnectivityGraph:
        if self._last_graph is None:
            self._last_graph = self.graph_manager.initialize(state, self.topology)
        elif self._last_graph.step == state.step and self._last_graph.particle_count == state.particle_count:
            return self._last_graph
        elif self._last_graph.particle_count == state.particle_count and self._last_graph.step < state.step:
            self._last_graph = self.graph_manager.update(state, self.topology, self._last_graph)
        else:
            self._last_graph = self.graph_manager.initialize(state, self.topology)
        return self._last_graph

    def _trace_record_for_state(
        self,
        state: SimulationState,
        graph: ConnectivityGraph,
    ) -> TraceRecord:
        if state.provenance.state_id in self.trace_store.state_ids():
            return self.trace_store.get_record(state.provenance.state_id)
        trace_tags, trace_metadata = _trace_tags_for_graph(graph)
        return self.trace_store.append_state(
            state,
            graph=graph,
            compartments=self.compartments,
            tags=trace_tags,
            metadata=trace_metadata,
        )


@dataclass(slots=True)
class SmallProteinBenchmarkRunner(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Run a fast real-structure benchmark on one small protein chain."""

    spec: SmallProteinBenchmarkSpec = field(default_factory=SmallProteinBenchmarkSpec)
    mapper: ProteinCoarseMapper = field(default_factory=ProteinCoarseMapper)
    forcefield_builder: ImportedProteinForceFieldBuilder = field(default_factory=ImportedProteinForceFieldBuilder)
    shadow_tuner: ProteinShadowTuner = field(default_factory=ProteinShadowTuner)
    graph_manager: ConnectivityGraphManager = field(default_factory=ConnectivityGraphManager)
    parity_validator: BackendParityValidator = field(default_factory=BackendParityValidator)
    name: str = "small_protein_benchmark_runner"
    classification: str = "[hybrid]"

    def describe_role(self) -> str:
        return (
            "Runs a small, fast, real-structure benchmark so classical, hybrid, and "
            "future backend paths can be compared on the same imported single-chain protein."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "benchmarks/reference_cases/data/1BRS.pdb",
            "topology/protein_coarse_mapping.py",
            "forcefields/hybrid_engine.py",
            "validation/backend_parity.py",
            "optimization/backend_execution.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/small_protein_benchmark.md",)

    def validate(self) -> tuple[str, ...]:
        issues = list(self.spec.validate())
        issues.extend(self.mapper.validate())
        issues.extend(self.forcefield_builder.validate())
        issues.extend(self.shadow_tuner.validate())
        issues.extend(self.parity_validator.validate())
        return tuple(issues)

    def run(self) -> SmallProteinBenchmarkReport:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

        imported = self.mapper.import_from_pdb(
            pdb_path=self.spec.pdb_path,
            entity_groups=(self.spec.entity_group,),
            structure_id=self.spec.name,
        )
        state = self._build_state(imported)
        forcefield = self.forcefield_builder.build(
            imported.topology,
            scenario_label=self.spec.name,
            reference_label=imported.reference_target.source_pdb_id,
            metadata={"benchmark_kind": "small_protein"},
        )
        runtime_bundle = self.shadow_tuner.build_runtime_bundle(
            topology=imported.topology,
            scenario_label=self.spec.name,
            base_time_step=self.spec.base_time_step,
            base_friction=self.spec.base_friction,
            reference_label=imported.reference_target.source_pdb_id,
            metadata={"benchmark_kind": "small_protein"},
        )
        correction_model = runtime_bundle.build_correction_model()
        selected_region = self._build_region(state, imported)
        hybrid_engine = HybridForceEngine()
        residual_model = self._warm_residual_model(
            state=state,
            topology=imported.topology,
            forcefield=forcefield,
            hybrid_engine=hybrid_engine,
            correction_model=correction_model,
            selected_region=selected_region,
        )
        graph = self.graph_manager.initialize(state, imported.topology)
        execution_plan = self._execution_plan(hybrid_engine, state)
        parity_report = self.parity_validator.compare_hybrid_classical(
            state=state,
            topology=imported.topology,
            forcefield=forcefield,
            reference_provider=BaselineForceEvaluator(),
            hybrid_engine=hybrid_engine,
        )
        benchmark_report = self._benchmark_report(
            state=state,
            topology=imported.topology,
            forcefield=forcefield,
            compartments=imported.compartments,
            graph=graph,
            hybrid_engine=hybrid_engine,
            correction_model=correction_model,
            residual_model=residual_model,
            selected_region=selected_region,
            runtime_bundle=runtime_bundle,
        )

        return SmallProteinBenchmarkReport(
            spec=self.spec,
            structure_id=Path(self.spec.pdb_path).stem.upper(),
            entity_id=self.spec.entity_group.entity_id,
            residue_count=len(imported.residues),
            bead_count=len(imported.bead_blocks),
            benchmark_report=benchmark_report,
            parity_report=parity_report,
            execution_plan=execution_plan,
            recommended_time_step=runtime_bundle.dynamics_recommendation.time_step,
            recommended_friction=runtime_bundle.dynamics_recommendation.friction_coefficient,
            selected_region_size=selected_region.size,
            trained_state_count=residual_model.trained_state_count(),
            metadata=FrozenMetadata(
                {
                    "particle_count": state.particle_count,
                    "source_pdb_id": imported.reference_target.source_pdb_id,
                    "region_particle_indices": selected_region.particle_indices,
                }
            ),
        )

    def _build_state(self, imported: ImportedProteinSystem) -> SimulationState:
        return SimulationState(
            units=UnitSystem.md_nano(),
            particles=imported.particles,
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId(f"{self.spec.name}-sim"),
                state_id=StateId(f"{self.spec.name}-state-0"),
                parent_state_id=None,
                created_by=self.name,
                stage="benchmark_seed",
                metadata={"source_path": imported.source_path},
            ),
            time=0.0,
            step=0,
            potential_energy=0.0,
            observables={"benchmark": self.spec.name},
        )

    def _build_region(
        self,
        state: SimulationState,
        imported: ImportedProteinSystem,
    ) -> RefinementRegion:
        particle_count = state.particle_count
        region_size = min(self.spec.manual_region_size, particle_count)
        start = max(0, (particle_count - region_size) // 2)
        particle_indices = tuple(range(start, start + region_size))
        return RefinementRegion(
            region_id=f"{self.spec.name}_region_core",
            state_id=state.provenance.state_id,
            particle_indices=particle_indices,
            compartment_ids=(self.spec.entity_group.entity_id,),
            trigger_kinds=(RegionTriggerKind.MANUAL,),
            score=1.0,
            metadata={
                "benchmark_kind": "small_protein",
                "structure_id": imported.structure_id,
            },
        )

    def _warm_residual_model(
        self,
        *,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
        hybrid_engine: HybridForceEngine,
        correction_model,
        selected_region: RefinementRegion,
    ) -> ScalableResidualModel:
        residual_model = ScalableResidualModel()
        if self.spec.warmup_training_passes == 0:
            return residual_model
        for _ in range(self.spec.warmup_training_passes):
            result = hybrid_engine.evaluate_detailed(
                state,
                topology,
                forcefield,
                selected_regions=(selected_region,),
                correction_model=correction_model,
            )
            if result.residual_target is None:
                break
            residual_model.observe_state(
                state,
                result.classical_evaluation,
                result.residual_target,
                sample_weight=1.0,
            )
        return residual_model

    def _execution_plan(
        self,
        hybrid_engine: HybridForceEngine,
        state: SimulationState,
    ) -> BackendExecutionPlan:
        planner = BackendExecutionPlanner(
            backend_registry=hybrid_engine.dispatch_boundary.backend_registry,
        )
        pair_count = state.particle_count * max(state.particle_count - 1, 0) // 2
        return planner.plan(
            BackendExecutionRequest(
                target_component="forcefields/hybrid_engine.py",
                particle_count=state.particle_count,
                pair_count=pair_count,
                required_capabilities=("neighbor_list", "pairwise", "tensor"),
            )
        )

    def _measure_case(
        self,
        *,
        name: str,
        repeats: int,
        operation,
        metadata_factory,
    ) -> BenchmarkCaseResult:
        last_result = None
        started_at = perf_counter()
        for _ in range(repeats):
            last_result = operation()
        elapsed = perf_counter() - started_at
        metadata = metadata_factory(last_result) if last_result is not None else {}
        return BenchmarkCaseResult(
            name=name,
            iterations=repeats,
            elapsed_seconds=elapsed,
            metadata=_coerce_metadata(metadata),
        )

    def _benchmark_report(
        self,
        *,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
        compartments: CompartmentRegistry,
        graph: ConnectivityGraph,
        hybrid_engine: HybridForceEngine,
        correction_model,
        residual_model: ScalableResidualModel,
        selected_region: RefinementRegion,
        runtime_bundle: ProteinShadowRuntimeBundle,
    ) -> BenchmarkReport:
        repeats = self.spec.repeats
        baseline_evaluator = BaselineForceEvaluator()
        integrator = LangevinIntegrator(
            time_step=runtime_bundle.dynamics_recommendation.time_step,
            friction_coefficient=runtime_bundle.dynamics_recommendation.friction_coefficient,
            stochastic=False,
        )
        production_evaluator = self._production_evaluator(
            topology=topology,
            forcefield=forcefield,
            compartments=compartments,
            hybrid_engine=hybrid_engine,
            correction_model=correction_model,
            residual_model=residual_model,
        )

        def _run_short_rollout():
            current_state = state
            last_result = None
            rollout_evaluator = self._production_evaluator(
                topology=topology,
                forcefield=forcefield,
                compartments=compartments,
                hybrid_engine=hybrid_engine,
                correction_model=correction_model,
                residual_model=residual_model,
            )
            for _ in range(self.spec.rollout_steps):
                last_result = integrator.step(current_state, topology, forcefield, rollout_evaluator)
                current_state = self._rollout_state_from_step(
                    previous_state=current_state,
                    step_result=last_result,
                )
            return last_result

        cases = (
            self._measure_case(
                name="diagnostic_reference_classical_baseline",
                repeats=repeats,
                operation=lambda: baseline_evaluator.evaluate(state, topology, forcefield),
                metadata_factory=lambda result: {
                    "potential_energy": result.potential_energy,
                    "force_evaluator": baseline_evaluator.name,
                    "diagnostic_role": "reference_baseline",
                },
            ),
            self._measure_case(
                name=_CLASSICAL_ONLY_CASE,
                repeats=repeats,
                operation=lambda: hybrid_engine.evaluate_detailed(state, topology, forcefield),
                metadata_factory=lambda result: {
                    "backend_name": result.backend_name,
                    "potential_energy": result.final_evaluation.potential_energy,
                    "qcloud_applied": False,
                    "residual_applied": False,
                    "engine_mode": "classical_only",
                },
            ),
            self._measure_case(
                name="diagnostic_shadow_only",
                repeats=repeats,
                operation=lambda: hybrid_engine.evaluate_detailed(
                    state,
                    topology,
                    forcefield,
                    selected_regions=(selected_region,),
                    correction_model=correction_model,
                ),
                metadata_factory=lambda result: {
                    "backend_name": result.backend_name,
                    "potential_energy": result.final_evaluation.potential_energy,
                    "selected_region_size": selected_region.size,
                    "qcloud_correction_count": len(result.qcloud_result.applied_corrections) if result.qcloud_result else 0,
                    "diagnostic_role": "shadow_only",
                },
            ),
            self._measure_case(
                name=_HYBRID_PRODUCTION_CASE,
                repeats=repeats,
                operation=lambda: production_evaluator.run_cycle(state),
                metadata_factory=lambda result: {
                    "backend_name": result.force_evaluation.metadata["backend"],
                    "engine_name": production_evaluator.hybrid_engine.name,
                    "potential_energy": result.force_evaluation.potential_energy,
                    "selected_region_size": selected_region.size,
                    "selected_region_count": result.selected_region_count,
                    "qcloud_requested": result.qcloud_requested,
                    "qcloud_applied": result.qcloud_applied,
                    "focus_compartments": result.focus_compartments,
                    "preliminary_action": result.preliminary_action,
                    "final_action": result.final_action,
                    "preliminary_uncertainty": result.preliminary_uncertainty.total_uncertainty,
                    "final_uncertainty": result.final_uncertainty.total_uncertainty,
                    "trace_record_count": result.trace_record_count,
                    "replay_buffer_size": result.replay_buffer_size,
                    "open_episode_count": result.open_episode_count,
                    "trained_state_count": residual_model.trained_state_count(),
                    "engine_mode": "hybrid_production",
                    "display_name": "production_hybrid_engine",
                },
            ),
            self._measure_case(
                name="diagnostic_graph_update_single_chain",
                repeats=repeats,
                operation=lambda: self.graph_manager.update(state, topology, graph),
                metadata_factory=lambda result: {
                    "active_edge_count": len(result.active_edges()),
                    "adaptive_edge_count": len(result.adaptive_edges()),
                    "diagnostic_role": "graph_update",
                },
            ),
            self._measure_case(
                name="diagnostic_production_rollout",
                repeats=repeats,
                operation=_run_short_rollout,
                metadata_factory=lambda result: {
                    "step": result.step,
                    "time": result.time,
                    "potential_energy": result.potential_energy,
                    "integrator": integrator.name,
                    "rollout_steps": self.spec.rollout_steps,
                    "interconnected_production": True,
                    "diagnostic_role": "production_rollout",
                },
            ),
        )
        return BenchmarkReport(
            suite_name=self.spec.name,
            cases=cases,
            metadata=FrozenMetadata(
                {
                    "structure_id": self.spec.name,
                    "repeat_count": repeats,
                    "case_count": len(cases),
                    "particle_count": state.particle_count,
                    "selected_region_size": selected_region.size,
                    "recommended_time_step": runtime_bundle.dynamics_recommendation.time_step,
                }
            ),
        )

    def _production_evaluator(
        self,
        *,
        topology: SystemTopology,
        forcefield: BaseForceField,
        compartments: CompartmentRegistry,
        hybrid_engine: HybridForceEngine,
        correction_model,
        residual_model: ScalableResidualModel,
    ) -> _InterconnectedProductionEvaluator:
        return _InterconnectedProductionEvaluator(
            topology=topology,
            forcefield=forcefield,
            compartments=compartments,
            default_focus_compartments=(self.spec.entity_group.entity_id,),
            hybrid_engine=hybrid_engine,
            correction_model=correction_model,
            residual_model=residual_model,
            max_region_size=self.spec.manual_region_size,
        )

    def _rollout_state_from_step(
        self,
        *,
        previous_state: SimulationState,
        step_result,
    ) -> SimulationState:
        return SimulationState(
            units=previous_state.units,
            particles=step_result.particles,
            thermodynamics=previous_state.thermodynamics,
            provenance=StateProvenance(
                simulation_id=previous_state.provenance.simulation_id,
                state_id=StateId(f"{self.spec.name}-state-{step_result.step}"),
                parent_state_id=previous_state.provenance.state_id,
                created_by=self.name,
                stage="benchmark_rollout",
                metadata={
                    "benchmark_kind": "small_protein",
                    "source_state_id": str(previous_state.provenance.state_id),
                },
            ),
            cell=previous_state.cell,
            time=step_result.time,
            step=step_result.step,
            potential_energy=step_result.potential_energy,
            observables=step_result.observables,
        )


__all__ = [
    "SmallProteinBenchmarkReport",
    "SmallProteinBenchmarkRunner",
    "SmallProteinBenchmarkSpec",
]
