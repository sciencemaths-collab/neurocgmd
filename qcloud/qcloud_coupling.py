"""Bounded coupling of local qcloud corrections into force evaluations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import sqrt
from typing import Protocol, runtime_checkable

from compartments.registry import CompartmentRegistry
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import CompartmentId, FrozenMetadata, Vector3
from forcefields.base_forcefield import BaseForceField
from memory.trace_store import TraceRecord
from physics.forces.composite import ForceEvaluation
from qcloud.cloud_state import ParticleForceDelta, QCloudCorrection, RefinementRegion
from qcloud.region_selector import LocalRegionSelector
from topology.system_topology import SystemTopology
from graph.graph_manager import ConnectivityGraph


def _vector_magnitude(vector: Vector3) -> float:
    return sqrt(sum(component * component for component in vector))


def _cap_vector(vector: Vector3, max_magnitude: float) -> Vector3:
    magnitude = _vector_magnitude(vector)
    if magnitude <= max_magnitude or magnitude == 0.0:
        return vector
    scale = max_magnitude / magnitude
    return tuple(component * scale for component in vector)


@runtime_checkable
class ForceEvaluationProvider(Protocol):
    """Protocol for baseline force providers that qcloud can augment."""

    name: str
    classification: str

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
    ) -> ForceEvaluation:
        """Return a baseline force evaluation before qcloud corrections are applied."""


@runtime_checkable
class QCloudCorrectionModel(Protocol):
    """Protocol for local qcloud correction providers."""

    name: str
    classification: str

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        region: RefinementRegion,
    ) -> QCloudCorrection:
        """Return a bounded local correction for the supplied refinement region."""


class NullQCloudCorrectionModel:
    """No-op correction provider used while the qcloud stack is still being scaffolded."""

    name = "null_qcloud_correction_model"
    classification = "[adapted]"

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        region: RefinementRegion,
    ) -> QCloudCorrection:
        del state, topology
        return QCloudCorrection(
            region_id=region.region_id,
            method_label=self.name,
            energy_delta=0.0,
            force_deltas=(),
            confidence=1.0,
        )


@dataclass(frozen=True, slots=True)
class QCloudCouplingResult:
    """Combined force evaluation plus the selected regions and applied corrections."""

    force_evaluation: ForceEvaluation
    selected_regions: tuple[RefinementRegion, ...]
    applied_corrections: tuple[QCloudCorrection, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)


@dataclass(slots=True)
class QCloudForceCoupler(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Apply bounded qcloud corrections on top of an existing force evaluation."""

    max_energy_delta_magnitude: float = 5.0
    max_force_delta_magnitude: float = 5.0
    name: str = "qcloud_force_coupler"
    classification: str = "[hybrid]"

    def describe_role(self) -> str:
        return (
            "Applies bounded local correction payloads to baseline force evaluations "
            "without replacing the underlying classical force evaluator."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "physics/forces/composite.py",
            "qcloud/cloud_state.py",
            "qcloud/region_selector.py",
            "memory/trace_store.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/qcloud_framework.md",
            "docs/sections/section_10_quantum_cloud_framework.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.max_energy_delta_magnitude < 0.0:
            issues.append("max_energy_delta_magnitude must be non-negative.")
        if self.max_force_delta_magnitude < 0.0:
            issues.append("max_force_delta_magnitude must be non-negative.")
        return tuple(issues)

    def couple(
        self,
        base_evaluation: ForceEvaluation,
        state: SimulationState,
        topology: SystemTopology,
        regions: Sequence[RefinementRegion],
        correction_model: QCloudCorrectionModel,
    ) -> QCloudCouplingResult:
        if len(base_evaluation.forces) != state.particle_count:
            raise ContractValidationError("base_evaluation.forces must match the SimulationState particle count.")
        if topology.particle_count != state.particle_count:
            raise ContractValidationError("SystemTopology particle_count must match the SimulationState particle count.")

        accumulated_forces = [list(vector) for vector in base_evaluation.forces]
        total_energy_delta = 0.0
        applied_corrections: list[QCloudCorrection] = []

        for region in regions:
            if region.state_id != state.provenance.state_id:
                raise ContractValidationError("Each refinement region must target the current SimulationState state_id.")
            correction = correction_model.evaluate(state, topology, region)
            if correction.region_id != region.region_id:
                raise ContractValidationError("Correction region_id must match the supplied refinement region.")

            bounded_energy_delta = max(
                -self.max_energy_delta_magnitude,
                min(self.max_energy_delta_magnitude, correction.energy_delta),
            )
            total_energy_delta += bounded_energy_delta

            for force_delta in correction.force_deltas:
                if force_delta.particle_index not in region.particle_indices:
                    raise ContractValidationError(
                        "Correction force_deltas may only reference particles inside the refinement region."
                    )
                if force_delta.particle_index >= state.particle_count:
                    raise ContractValidationError("Correction particle_index exceeds the SimulationState particle count.")
                bounded_vector = _cap_vector(force_delta.delta_force, self.max_force_delta_magnitude)
                for axis, value in enumerate(bounded_vector):
                    accumulated_forces[force_delta.particle_index][axis] += value

            applied_corrections.append(correction)

        updated_force_evaluation = ForceEvaluation(
            forces=tuple(tuple(vector) for vector in accumulated_forces),
            potential_energy=base_evaluation.potential_energy + total_energy_delta,
            component_energies=base_evaluation.component_energies.with_updates({"qcloud": total_energy_delta}),
            metadata=base_evaluation.metadata.with_updates(
                {
                    "qcloud_region_count": len(tuple(regions)),
                    "qcloud_correction_model": correction_model.name,
                }
            ),
        )
        return QCloudCouplingResult(
            force_evaluation=updated_force_evaluation,
            selected_regions=tuple(regions),
            applied_corrections=tuple(applied_corrections),
            metadata=FrozenMetadata(
                {
                    "applied_region_count": len(tuple(regions)),
                    "applied_correction_count": len(applied_corrections),
                    "total_qcloud_energy_delta": total_energy_delta,
                }
            ),
        )

    def evaluate_with_selector(
        self,
        *,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
        base_force_evaluator: ForceEvaluationProvider,
        correction_model: QCloudCorrectionModel,
        region_selector: LocalRegionSelector,
        graph: ConnectivityGraph,
        compartments: CompartmentRegistry | None = None,
        trace_record: TraceRecord | None = None,
        focus_compartments: Sequence[CompartmentId | str] = (),
        correction_priority_scores: dict[int, float] | None = None,
    ) -> QCloudCouplingResult:
        base_evaluation = base_force_evaluator.evaluate(state, topology, forcefield)
        selected_regions = region_selector.select_regions(
            state,
            topology,
            graph,
            compartments=compartments,
            trace_record=trace_record,
            focus_compartments=focus_compartments,
            correction_priority_scores=correction_priority_scores,
        )
        result = self.couple(
            base_evaluation,
            state,
            topology,
            selected_regions,
            correction_model,
        )
        return QCloudCouplingResult(
            force_evaluation=result.force_evaluation,
            selected_regions=result.selected_regions,
            applied_corrections=result.applied_corrections,
            metadata=result.metadata.with_updates(
                {
                    "base_force_evaluator": base_force_evaluator.name,
                    "region_selector": region_selector.name,
                }
            ),
        )
