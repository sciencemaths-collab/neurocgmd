"""Backend-ready nonbonded kernel entrypoints."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata
from forcefields.base_forcefield import BaseForceField
from physics.backends.contracts import (
    PairInteractionRecord,
    PairwiseKernelResult,
    PhysicsBackend,
    pair_input_from_neighbor_list,
)
from qcloud.cloud_state import ParticleForceDelta
from topology.system_topology import SystemTopology


@dataclass(frozen=True, slots=True)
class NonbondedKernelPolicy(ValidatableComponent):
    """Explicit nonbonded kernel policy."""

    skin: float = 0.3
    exclude_bonded_pairs: bool = True

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.skin < 0.0:
            issues.append("skin must be non-negative.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class NonbondedKernelResult(ValidatableComponent):
    """Energy and force contribution from a nonbonded kernel."""

    energy_delta: float
    force_deltas: tuple[ParticleForceDelta, ...]
    evaluated_pair_count: int
    cutoff: float
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
        if self.cutoff <= 0.0:
            issues.append("cutoff must be strictly positive.")
        return tuple(issues)


@dataclass(slots=True)
class LennardJonesNonbondedKernel(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Reference Lennard-Jones nonbonded kernel routed through a backend hook."""

    backend: PhysicsBackend
    policy: NonbondedKernelPolicy = field(default_factory=NonbondedKernelPolicy)
    name: str = "lennard_jones_nonbonded_kernel"
    classification: str = "[established]"

    def describe_role(self) -> str:
        return "Evaluates sparse Lennard-Jones interactions through a backend-neutral pairwise kernel path."

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "physics/backends/contracts.py",
            "physics/neighbor_list.py",
            "forcefields/base_forcefield.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/backend_compute_spine.md",)

    def validate(self) -> tuple[str, ...]:
        return self.policy.validate()

    def _pair_evaluator(
        self,
        topology: SystemTopology,
        forcefield: BaseForceField,
    ):
        def evaluate_pair(record: PairInteractionRecord):
            parameter = forcefield.nonbonded_parameter_for_pair(
                topology,
                record.particle_index_a,
                record.particle_index_b,
            )
            if record.distance > parameter.cutoff:
                return 0.0, (), None
            sigma_over_r = parameter.sigma / record.distance
            sr6 = sigma_over_r**6
            sr12 = sr6 * sr6
            # Shifted LJ: subtract potential at cutoff for energy continuity
            sc = parameter.sigma / parameter.cutoff
            sc6 = sc ** 6
            sc12 = sc6 * sc6
            energy = 4.0 * parameter.epsilon * ((sr12 - sr6) - (sc12 - sc6))
            pair_scale = 24.0 * parameter.epsilon * (2.0 * sr12 - sr6) / (record.distance * record.distance)
            force_on_a = tuple(-pair_scale * component for component in record.displacement)
            return (
                energy,
                (
                    (record.particle_index_a, force_on_a),
                    (record.particle_index_b, tuple(-component for component in force_on_a)),
                ),
                {"sigma": parameter.sigma, "epsilon": parameter.epsilon},
            )

        return evaluate_pair

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
    ) -> NonbondedKernelResult:
        if topology.particle_count != state.particle_count:
            raise ContractValidationError("topology and state particle counts must match.")
        if not forcefield.nonbonded_parameters:
            raise ContractValidationError("forcefield must contain at least one nonbonded parameter.")

        excluded_pairs = (
            frozenset(bond.normalized_pair() for bond in topology.bonds)
            if self.policy.exclude_bonded_pairs
            else None
        )
        cutoff = max(parameter.cutoff for parameter in forcefield.nonbonded_parameters)
        neighbor_list = self.backend.build_neighbor_list(
            state.particles.positions,
            cutoff=cutoff,
            skin=self.policy.skin,
            excluded_pairs=excluded_pairs,
        )
        kernel_input = pair_input_from_neighbor_list(
            state_id=state.provenance.state_id,
            positions=state.particles.positions,
            neighbor_list=neighbor_list,
            metadata={"backend": self.backend.name, "kernel": self.name},
        )
        pair_result: PairwiseKernelResult = self.backend.execute_pairwise(
            kernel_input,
            self._pair_evaluator(topology, forcefield),
        )
        return NonbondedKernelResult(
            energy_delta=pair_result.energy_delta,
            force_deltas=tuple(
                ParticleForceDelta(
                    particle_index=contribution.particle_index,
                    delta_force=contribution.delta_force,
                )
                for contribution in pair_result.force_contributions
            ),
            evaluated_pair_count=pair_result.evaluated_pair_count,
            cutoff=cutoff,
            metadata=pair_result.metadata.with_updates({"neighbor_pair_count": len(neighbor_list.pairs)}),
        )
