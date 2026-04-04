"""Parity checks between backend-spine evaluations and trusted in-repo references."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata, coerce_scalar
from forcefields.base_forcefield import BaseForceField
from forcefields.hybrid_engine import HybridForceEngine
from physics.forces.composite import ForceEvaluation
from topology.system_topology import SystemTopology


def _force_rms_error(left, right) -> float:
    particle_count = max(len(left), 1)
    total_sq = 0.0
    for vector_left, vector_right in zip(left, right):
        for axis in range(3):
            delta = vector_left[axis] - vector_right[axis]
            total_sq += delta * delta
    return sqrt(total_sq / particle_count)


def _max_force_component_error(left, right) -> float:
    maximum = 0.0
    for vector_left, vector_right in zip(left, right):
        for axis in range(3):
            maximum = max(maximum, abs(vector_left[axis] - vector_right[axis]))
    return maximum


@dataclass(frozen=True, slots=True)
class BackendParityMetric(ValidatableComponent):
    """One backend-parity metric."""

    label: str
    reference_value: float
    candidate_value: float
    absolute_error: float
    tolerance: float
    passed: bool

    def __post_init__(self) -> None:
        for field_name in ("reference_value", "candidate_value", "absolute_error", "tolerance"):
            object.__setattr__(self, field_name, coerce_scalar(getattr(self, field_name), field_name))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.label.strip():
            issues.append("label must be a non-empty string.")
        if self.absolute_error < 0.0:
            issues.append("absolute_error must be non-negative.")
        if self.tolerance < 0.0:
            issues.append("tolerance must be non-negative.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class BackendParityReport(ValidatableComponent):
    """Parity report for one backend-evaluation comparison."""

    target_component: str
    backend_name: str
    metrics: tuple[BackendParityMetric, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def all_passed(self) -> bool:
        return all(metric.passed for metric in self.metrics)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.target_component.strip():
            issues.append("target_component must be a non-empty string.")
        if not self.backend_name.strip():
            issues.append("backend_name must be a non-empty string.")
        return tuple(issues)


@dataclass(slots=True)
class BackendParityValidator(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Compare backend-spine outputs against trusted in-repo reference evaluators."""

    energy_tolerance: float = 1e-9
    force_rms_tolerance: float = 1e-9
    max_force_component_tolerance: float = 1e-9
    name: str = "backend_parity_validator"
    classification: str = "[adapted]"

    def describe_role(self) -> str:
        return (
            "Compares new backend-spine force outputs against trusted in-repo reference evaluators "
            "so performance architecture changes stay scientifically honest."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "forcefields/hybrid_engine.py",
            "physics/forces/composite.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/backend_compute_spine.md",)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.energy_tolerance < 0.0:
            issues.append("energy_tolerance must be non-negative.")
        if self.force_rms_tolerance < 0.0:
            issues.append("force_rms_tolerance must be non-negative.")
        if self.max_force_component_tolerance < 0.0:
            issues.append("max_force_component_tolerance must be non-negative.")
        return tuple(issues)

    def compare_force_evaluations(
        self,
        reference: ForceEvaluation,
        candidate: ForceEvaluation,
        *,
        target_component: str,
        backend_name: str,
    ) -> BackendParityReport:
        if len(reference.forces) != len(candidate.forces):
            raise ContractValidationError("reference and candidate force blocks must have the same particle count.")
        energy_error = abs(reference.potential_energy - candidate.potential_energy)
        force_rms = _force_rms_error(reference.forces, candidate.forces)
        max_force_error = _max_force_component_error(reference.forces, candidate.forces)
        return BackendParityReport(
            target_component=target_component,
            backend_name=backend_name,
            metrics=(
                BackendParityMetric(
                    label="energy_abs_error",
                    reference_value=reference.potential_energy,
                    candidate_value=candidate.potential_energy,
                    absolute_error=energy_error,
                    tolerance=self.energy_tolerance,
                    passed=energy_error <= self.energy_tolerance,
                ),
                BackendParityMetric(
                    label="force_rms_error",
                    reference_value=0.0,
                    candidate_value=force_rms,
                    absolute_error=force_rms,
                    tolerance=self.force_rms_tolerance,
                    passed=force_rms <= self.force_rms_tolerance,
                ),
                BackendParityMetric(
                    label="max_force_component_error",
                    reference_value=0.0,
                    candidate_value=max_force_error,
                    absolute_error=max_force_error,
                    tolerance=self.max_force_component_tolerance,
                    passed=max_force_error <= self.max_force_component_tolerance,
                ),
            ),
            metadata=FrozenMetadata({"particle_count": len(reference.forces)}),
        )

    def compare_providers(
        self,
        *,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
        reference_provider,
        candidate_provider,
        target_component: str,
        backend_name: str,
    ) -> BackendParityReport:
        reference = reference_provider.evaluate(state, topology, forcefield)
        candidate = candidate_provider.evaluate(state, topology, forcefield)
        return self.compare_force_evaluations(
            reference,
            candidate,
            target_component=target_component,
            backend_name=backend_name,
        )

    def compare_hybrid_classical(
        self,
        *,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
        reference_provider,
        hybrid_engine: HybridForceEngine,
    ) -> BackendParityReport:
        reference = reference_provider.evaluate(state, topology, forcefield)
        result = hybrid_engine.evaluate_detailed(state, topology, forcefield)
        return self.compare_force_evaluations(
            reference,
            result.classical_evaluation,
            target_component="forcefields/hybrid_engine.py",
            backend_name=result.backend_name,
        )
