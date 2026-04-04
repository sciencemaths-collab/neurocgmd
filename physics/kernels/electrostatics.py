"""Backend-ready electrostatic kernel entrypoints."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata
from forcefields.nonbonded_potentials import COULOMB_CONSTANT
from physics.backends.contracts import PairInteractionRecord, PhysicsBackend, pair_input_from_neighbor_list
from qcloud.cloud_state import ParticleForceDelta
from topology.system_topology import SystemTopology


@dataclass(frozen=True, slots=True)
class ElectrostaticKernelPolicy(ValidatableComponent):
    """Policy for sparse electrostatic execution."""

    cutoff: float
    dielectric: float = 78.5
    skin: float = 0.3

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.cutoff <= 0.0:
            issues.append("cutoff must be strictly positive.")
        if self.dielectric <= 0.0:
            issues.append("dielectric must be strictly positive.")
        if self.skin < 0.0:
            issues.append("skin must be non-negative.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ElectrostaticKernelResult(ValidatableComponent):
    """Energy and force contribution from an electrostatic kernel."""

    energy_delta: float
    force_deltas: tuple[ParticleForceDelta, ...]
    evaluated_pair_count: int
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.evaluated_pair_count < 0:
            issues.append("evaluated_pair_count must be non-negative.")
        return tuple(issues)


@dataclass(slots=True)
class CoulombElectrostaticKernel(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Sparse Coulomb kernel routed through a backend pairwise hook."""

    backend: PhysicsBackend
    charges: tuple[float, ...]
    policy: ElectrostaticKernelPolicy
    name: str = "coulomb_electrostatic_kernel"
    classification: str = "[adapted]"

    def describe_role(self) -> str:
        return "Evaluates sparse Coulomb interactions through the backend-neutral pairwise kernel path."

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "physics/backends/contracts.py",
            "forcefields/nonbonded_potentials.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/backend_compute_spine.md",)

    def validate(self) -> tuple[str, ...]:
        return self.policy.validate()

    def _pair_evaluator(self):
        def evaluate_pair(record: PairInteractionRecord):
            charge_a = self.charges[record.particle_index_a]
            charge_b = self.charges[record.particle_index_b]
            if charge_a == 0.0 and charge_b == 0.0:
                return 0.0, (), None
            energy = COULOMB_CONSTANT * charge_a * charge_b / (self.policy.dielectric * record.distance)
            pair_scale = COULOMB_CONSTANT * charge_a * charge_b / (
                self.policy.dielectric * record.distance * record.distance * record.distance
            )
            force_on_a = tuple(-pair_scale * component for component in record.displacement)
            return (
                energy,
                (
                    (record.particle_index_a, force_on_a),
                    (record.particle_index_b, tuple(-component for component in force_on_a)),
                ),
                {"charge_product": charge_a * charge_b},
            )

        return evaluate_pair

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
    ) -> ElectrostaticKernelResult:
        if len(self.charges) != state.particle_count:
            raise ContractValidationError("charges must match the SimulationState particle count.")
        neighbor_list = self.backend.build_neighbor_list(
            state.particles.positions,
            cutoff=self.policy.cutoff,
            skin=self.policy.skin,
            excluded_pairs=frozenset(bond.normalized_pair() for bond in topology.bonds),
        )
        kernel_input = pair_input_from_neighbor_list(
            state_id=state.provenance.state_id,
            positions=state.particles.positions,
            neighbor_list=neighbor_list,
            metadata={"backend": self.backend.name, "kernel": self.name},
        )
        pair_result = self.backend.execute_pairwise(kernel_input, self._pair_evaluator())
        return ElectrostaticKernelResult(
            energy_delta=pair_result.energy_delta,
            force_deltas=tuple(
                ParticleForceDelta(particle_index=contribution.particle_index, delta_force=contribution.delta_force)
                for contribution in pair_result.force_contributions
            ),
            evaluated_pair_count=pair_result.evaluated_pair_count,
            metadata=pair_result.metadata,
        )
