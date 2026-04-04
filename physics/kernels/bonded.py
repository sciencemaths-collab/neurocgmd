"""Backend-ready bonded kernel entrypoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata
from forcefields.base_forcefield import BaseForceField
from qcloud.cloud_state import ParticleForceDelta
from topology.system_topology import SystemTopology


@dataclass(frozen=True, slots=True)
class BondedKernelResult(ValidatableComponent):
    """Energy and force contribution from bonded interactions."""

    energy_delta: float
    force_deltas: tuple[ParticleForceDelta, ...]
    evaluated_bond_count: int
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.evaluated_bond_count < 0:
            issues.append("evaluated_bond_count must be non-negative.")
        return tuple(issues)


@dataclass(slots=True)
class HarmonicBondKernel(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Reference bonded kernel matching the baseline harmonic bond contract."""

    name: str = "harmonic_bond_kernel"
    classification: str = "[established]"

    def describe_role(self) -> str:
        return "Evaluates harmonic bond energies and forces as a stable backend-ready kernel site."

    def declared_dependencies(self) -> tuple[str, ...]:
        return ("forcefields/base_forcefield.py", "topology/system_topology.py")

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/backend_compute_spine.md",)

    def validate(self) -> tuple[str, ...]:
        return ()

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
    ) -> BondedKernelResult:
        if topology.particle_count != state.particle_count:
            raise ContractValidationError("topology and state particle counts must match.")
        accumulated_forces = [[0.0, 0.0, 0.0] for _ in range(state.particle_count)]
        total_energy = 0.0
        evaluated = 0

        for bond in topology.bonds:
            parameter = forcefield.bond_parameter_for(topology, bond)
            equilibrium_distance = (
                bond.equilibrium_distance
                if bond.equilibrium_distance is not None
                else parameter.equilibrium_distance
            )
            stiffness = bond.stiffness if bond.stiffness is not None else parameter.stiffness
            position_a = state.particles.positions[bond.particle_index_a]
            position_b = state.particles.positions[bond.particle_index_b]
            dx = position_b[0] - position_a[0]
            dy = position_b[1] - position_a[1]
            dz = position_b[2] - position_a[2]
            distance = sqrt(dx * dx + dy * dy + dz * dz)
            if distance <= 1e-12:
                continue
            displacement = distance - equilibrium_distance
            total_energy += 0.5 * stiffness * displacement * displacement
            force_scale = stiffness * displacement / distance
            force_on_a = (force_scale * dx, force_scale * dy, force_scale * dz)
            for axis, value in enumerate(force_on_a):
                accumulated_forces[bond.particle_index_a][axis] += value
                accumulated_forces[bond.particle_index_b][axis] -= value
            evaluated += 1

        return BondedKernelResult(
            energy_delta=total_energy,
            force_deltas=tuple(
                ParticleForceDelta(particle_index=index, delta_force=tuple(vector))
                for index, vector in enumerate(accumulated_forces)
                if any(abs(component) > 0.0 for component in vector)
            ),
            evaluated_bond_count=evaluated,
            metadata=FrozenMetadata({"kernel": self.name}),
        )
