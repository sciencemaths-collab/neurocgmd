"""Constraint-kernel entrypoints for backend-neutral position projection."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, VectorTuple


@dataclass(frozen=True, slots=True)
class DistanceConstraintSpec(ValidatableComponent):
    """One fixed-distance constraint."""

    particle_index_a: int
    particle_index_b: int
    target_distance: float
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.particle_index_a < 0 or self.particle_index_b < 0:
            issues.append("constraint particle indices must be non-negative.")
        if self.particle_index_a == self.particle_index_b:
            issues.append("constraint particle indices must differ.")
        if self.target_distance <= 0.0:
            issues.append("target_distance must be strictly positive.")
        if self.tolerance < 0.0:
            issues.append("tolerance must be non-negative.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ConstraintKernelResult(ValidatableComponent):
    """Position-projection result for a constraint step."""

    corrected_positions: VectorTuple
    applied_constraint_count: int
    max_violation: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.applied_constraint_count < 0:
            issues.append("applied_constraint_count must be non-negative.")
        if self.max_violation < 0.0:
            issues.append("max_violation must be non-negative.")
        return tuple(issues)


@dataclass(slots=True)
class DistanceConstraintKernel(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Reference distance-projection kernel."""

    name: str = "distance_constraint_kernel"
    classification: str = "[adapted]"

    def describe_role(self) -> str:
        return "Projects positions back onto explicit distance constraints."

    def declared_dependencies(self) -> tuple[str, ...]:
        return ("physics/kernels/constraints.py",)

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/backend_compute_spine.md",)

    def validate(self) -> tuple[str, ...]:
        return ()

    def apply(
        self,
        positions: VectorTuple,
        constraints: tuple[DistanceConstraintSpec, ...],
    ) -> ConstraintKernelResult:
        corrected = [list(vector) for vector in positions]
        max_violation = 0.0
        applied = 0
        for constraint in constraints:
            left = corrected[constraint.particle_index_a]
            right = corrected[constraint.particle_index_b]
            dx = right[0] - left[0]
            dy = right[1] - left[1]
            dz = right[2] - left[2]
            distance = sqrt(dx * dx + dy * dy + dz * dz)
            if distance <= 1e-12:
                continue
            violation = distance - constraint.target_distance
            max_violation = max(max_violation, abs(violation))
            if abs(violation) <= constraint.tolerance:
                continue
            correction_scale = 0.5 * violation / distance
            correction = (correction_scale * dx, correction_scale * dy, correction_scale * dz)
            for axis in range(3):
                corrected[constraint.particle_index_a][axis] += correction[axis]
                corrected[constraint.particle_index_b][axis] -= correction[axis]
            applied += 1
        return ConstraintKernelResult(
            corrected_positions=tuple(tuple(vector) for vector in corrected),
            applied_constraint_count=applied,
            max_violation=max_violation,
            metadata=FrozenMetadata({"kernel": self.name}),
        )
