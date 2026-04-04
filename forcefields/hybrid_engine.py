"""Hybrid force engine that composes classical kernels, qcloud corrections, and ML residuals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from compartments.registry import CompartmentRegistry
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import CompartmentId, FrozenMetadata, StateId
from forcefields.base_forcefield import BaseForceField
from graph.graph_manager import ConnectivityGraph
from memory.trace_store import TraceRecord
from ml.residual_model import ResidualPrediction, ResidualTarget, ResidualModel, StateAwareResidualModel
from physics.backends.dispatch import KernelDispatchBoundary, KernelDispatchRequest
from physics.forces.composite import ForceEvaluation
from physics.kernels.bonded import HarmonicBondKernel
from physics.kernels.electrostatics import CoulombElectrostaticKernel, ElectrostaticKernelPolicy
from physics.kernels.nonbonded import LennardJonesNonbondedKernel, NonbondedKernelPolicy
from qcloud import (
    LocalRegionSelector,
    QCloudCorrectionModel,
    QCloudCouplingResult,
    QCloudForceCoupler,
    RefinementRegion,
)
from qcloud.qcloud_coupling import ForceEvaluationProvider
from topology.system_topology import SystemTopology


def _sum_force_blocks(*force_blocks) -> tuple[tuple[float, float, float], ...]:
    if not force_blocks:
        return ()
    particle_count = len(force_blocks[0])
    accumulated = [[0.0, 0.0, 0.0] for _ in range(particle_count)]
    for block in force_blocks:
        if len(block) != particle_count:
            raise ContractValidationError("All force blocks must have the same particle count.")
        for particle_index, vector in enumerate(block):
            for axis, value in enumerate(vector):
                accumulated[particle_index][axis] += value
    return tuple(tuple(vector) for vector in accumulated)


def _force_deltas_to_block(
    particle_count: int,
    deltas,
) -> tuple[tuple[float, float, float], ...]:
    accumulated = [[0.0, 0.0, 0.0] for _ in range(particle_count)]
    for force_delta in deltas:
        for axis, value in enumerate(force_delta.delta_force):
            accumulated[force_delta.particle_index][axis] += value
    return tuple(tuple(vector) for vector in accumulated)


def _scale_prediction(
    prediction: ResidualPrediction,
    *,
    scale: float,
) -> ResidualPrediction:
    return ResidualPrediction(
        state_id=prediction.state_id,
        predicted_energy_delta=prediction.predicted_energy_delta * scale,
        force_deltas=tuple(
            type(force_delta)(
                particle_index=force_delta.particle_index,
                delta_force=tuple(component * scale for component in force_delta.delta_force),
                metadata=force_delta.metadata,
            )
            for force_delta in prediction.force_deltas
        ),
        confidence=prediction.confidence,
        metadata=prediction.metadata.with_updates({"applied_scale": scale}),
    )


@dataclass(frozen=True, slots=True)
class HybridClassicalKernelPolicy(ValidatableComponent):
    """Policy for the classical kernel slice of the hybrid engine."""

    nonbonded_skin: float = 0.3
    exclude_bonded_nonbonded_pairs: bool = True
    include_electrostatics: bool = False

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.nonbonded_skin < 0.0:
            issues.append("nonbonded_skin must be non-negative.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class HybridForceEnginePolicy(ValidatableComponent):
    """Policy for composing classical, qcloud, and ML residual terms."""

    residual_reference_frame: str = "corrected"
    residual_mix_when_qcloud_active: float = 0.35
    train_residual_from_qcloud: bool = False

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.residual_reference_frame not in {"classical", "corrected"}:
            issues.append("residual_reference_frame must be either 'classical' or 'corrected'.")
        if not 0.0 <= self.residual_mix_when_qcloud_active <= 1.0:
            issues.append("residual_mix_when_qcloud_active must lie in [0, 1].")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class HybridForceResult(ValidatableComponent):
    """Detailed hybrid force-evaluation report."""

    classical_evaluation: ForceEvaluation
    final_evaluation: ForceEvaluation
    backend_name: str
    qcloud_result: QCloudCouplingResult | None = None
    residual_prediction: ResidualPrediction | None = None
    residual_target: ResidualTarget | None = None
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.backend_name.strip():
            issues.append("backend_name must be a non-empty string.")
        return tuple(issues)


@dataclass(slots=True)
class HybridForceEngine(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Compose the new compute spine with qcloud and ML residual layers."""

    dispatch_boundary: KernelDispatchBoundary = field(default_factory=KernelDispatchBoundary)
    qcloud_coupler: QCloudForceCoupler = field(default_factory=QCloudForceCoupler)
    classical_policy: HybridClassicalKernelPolicy = field(default_factory=HybridClassicalKernelPolicy)
    hybrid_policy: HybridForceEnginePolicy = field(default_factory=HybridForceEnginePolicy)
    charges: tuple[float, ...] | None = None
    name: str = "hybrid_force_engine"
    classification: str = "[hybrid]"

    def describe_role(self) -> str:
        return (
            "Composes classical bonded/nonbonded kernels, optional electrostatics, "
            "shadow/spatial qcloud corrections, and ML residual corrections into one explicit force path."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "physics/backends/dispatch.py",
            "physics/kernels/bonded.py",
            "physics/kernels/nonbonded.py",
            "physics/kernels/electrostatics.py",
            "qcloud/qcloud_coupling.py",
            "ml/residual_model.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/backend_compute_spine.md",)

    def validate(self) -> tuple[str, ...]:
        issues = list(self.classical_policy.validate())
        issues.extend(self.hybrid_policy.validate())
        return tuple(issues)

    def _classical_evaluation(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
    ) -> tuple[ForceEvaluation, str]:
        dispatch, backend = self.dispatch_boundary.resolve(
            KernelDispatchRequest(
                target_component="forcefields/hybrid_engine.py",
                required_capabilities=("neighbor_list", "pairwise", "tensor"),
            )
        )
        bonded_result = HarmonicBondKernel().evaluate(state, topology, forcefield)
        nonbonded_result = LennardJonesNonbondedKernel(
            backend=backend,
            policy=NonbondedKernelPolicy(
                skin=self.classical_policy.nonbonded_skin,
                exclude_bonded_pairs=self.classical_policy.exclude_bonded_nonbonded_pairs,
            ),
        ).evaluate(state, topology, forcefield)
        force_blocks = [
            _force_deltas_to_block(state.particle_count, bonded_result.force_deltas),
            _force_deltas_to_block(state.particle_count, nonbonded_result.force_deltas),
        ]
        component_energies = {
            "bonded": bonded_result.energy_delta,
            "nonbonded": nonbonded_result.energy_delta,
        }
        if self.classical_policy.include_electrostatics and self.charges is not None:
            electrostatic_result = CoulombElectrostaticKernel(
                backend=backend,
                charges=self.charges,
                policy=ElectrostaticKernelPolicy(
                    cutoff=max(parameter.cutoff for parameter in forcefield.nonbonded_parameters),
                ),
            ).evaluate(state, topology)
            force_blocks.append(_force_deltas_to_block(state.particle_count, electrostatic_result.force_deltas))
            component_energies["electrostatics"] = electrostatic_result.energy_delta
        total_forces = _sum_force_blocks(*force_blocks)
        return (
            ForceEvaluation(
                forces=total_forces,
                potential_energy=sum(component_energies.values()),
                component_energies=FrozenMetadata(component_energies),
                metadata=FrozenMetadata(
                    {
                        "backend": dispatch.backend_name,
                        "bonded_count": bonded_result.evaluated_bond_count,
                        "nonbonded_pair_count": nonbonded_result.evaluated_pair_count,
                    }
                ),
            ),
            dispatch.backend_name,
        )

    def build_residual_target(
        self,
        *,
        state_id: StateId | str,
        qcloud_result: QCloudCouplingResult | None,
    ) -> ResidualTarget | None:
        if qcloud_result is None or not qcloud_result.applied_corrections:
            return None
        return ResidualTarget.from_corrections(
            state_id,
            qcloud_result.applied_corrections,
            metadata={"source": self.name},
        )

    def evaluate_detailed(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
        *,
        graph: ConnectivityGraph | None = None,
        compartments: CompartmentRegistry | None = None,
        trace_record: TraceRecord | None = None,
        focus_compartments: Sequence[CompartmentId | str] = (),
        region_selector: LocalRegionSelector | None = None,
        correction_model: QCloudCorrectionModel | None = None,
        residual_model: ResidualModel | None = None,
        selected_regions: Sequence[RefinementRegion] | None = None,
        cached_classical: "HybridForceResult | None" = None,
        correction_priority_scores: dict[int, float] | None = None,
    ) -> HybridForceResult:
        if cached_classical is not None:
            classical_evaluation = cached_classical.classical_evaluation
            backend_name = cached_classical.backend_name
        else:
            classical_evaluation, backend_name = self._classical_evaluation(state, topology, forcefield)
        qcloud_result: QCloudCouplingResult | None = None

        if correction_model is not None:
            if selected_regions is not None:
                qcloud_result = self.qcloud_coupler.couple(
                    classical_evaluation,
                    state,
                    topology,
                    selected_regions,
                    correction_model,
                )
            elif graph is not None and region_selector is not None:
                qcloud_result = self.qcloud_coupler.evaluate_with_selector(
                    state=state,
                    topology=topology,
                    forcefield=forcefield,
                    base_force_evaluator=_StaticForceEvaluationProvider(classical_evaluation),
                    correction_model=correction_model,
                    region_selector=region_selector,
                    graph=graph,
                    compartments=compartments,
                    trace_record=trace_record,
                    focus_compartments=focus_compartments,
                    correction_priority_scores=correction_priority_scores,
                )

        base_for_residual = classical_evaluation
        final_evaluation = classical_evaluation
        if qcloud_result is not None:
            final_evaluation = qcloud_result.force_evaluation
            if self.hybrid_policy.residual_reference_frame == "corrected":
                base_for_residual = qcloud_result.force_evaluation

        residual_prediction = None
        if residual_model is not None:
            residual_prediction = residual_model.predict(state, base_for_residual)
            if qcloud_result is not None:
                residual_prediction = _scale_prediction(
                    residual_prediction,
                    scale=self.hybrid_policy.residual_mix_when_qcloud_active,
                )
            ml_force_block = _force_deltas_to_block(state.particle_count, residual_prediction.force_deltas)
            final_evaluation = ForceEvaluation(
                forces=_sum_force_blocks(final_evaluation.forces, ml_force_block),
                potential_energy=final_evaluation.potential_energy + residual_prediction.predicted_energy_delta,
                component_energies=final_evaluation.component_energies.with_updates(
                    {"ml_residual": residual_prediction.predicted_energy_delta}
                ),
                metadata=final_evaluation.metadata.with_updates({"residual_model": residual_model.name}),
            )

        residual_target = self.build_residual_target(
            state_id=state.provenance.state_id,
            qcloud_result=qcloud_result,
        )
        if (
            residual_target is not None
            and residual_model is not None
            and self.hybrid_policy.train_residual_from_qcloud
        ):
            if isinstance(residual_model, StateAwareResidualModel):
                residual_model.observe_state(state, classical_evaluation, residual_target, sample_weight=1.0)
            else:
                residual_model.observe(residual_target, sample_weight=1.0)

        return HybridForceResult(
            classical_evaluation=classical_evaluation,
            final_evaluation=final_evaluation,
            backend_name=backend_name,
            qcloud_result=qcloud_result,
            residual_prediction=residual_prediction,
            residual_target=residual_target,
            metadata=FrozenMetadata(
                {
                    "qcloud_applied": qcloud_result is not None,
                    "residual_applied": residual_prediction is not None,
                    "residual_reference_frame": self.hybrid_policy.residual_reference_frame,
                }
            ),
        )

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
    ) -> ForceEvaluation:
        return self.evaluate_detailed(state, topology, forcefield).final_evaluation


@dataclass(slots=True)
class _StaticForceEvaluationProvider(ForceEvaluationProvider):
    """Expose one precomputed force evaluation through the provider protocol."""

    evaluation: ForceEvaluation
    name: str = "static_force_evaluation_provider"
    classification: str = "[test]"

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
    ) -> ForceEvaluation:
        del state, topology, forcefield
        return self.evaluation
