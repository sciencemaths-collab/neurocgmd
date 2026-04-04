"""Spatially aware local interaction field for intelligent shadow corrections."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, sqrt

from chemistry.residue_semantics import ProteinChemistryModel, ResidueChemistryDescriptor
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata, Vector3, coerce_scalar
from forcefields.spatial_semantic_profiles import SpatialSemanticParameterSet
from qcloud.cloud_state import ParticleForceDelta, RefinementRegion
from topology.system_topology import SystemTopology


def _subtract(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def _magnitude(vector: Vector3) -> float:
    return sqrt(sum(component * component for component in vector))


def _scale(vector: Vector3, factor: float) -> Vector3:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _normalize(vector: Vector3) -> Vector3:
    magnitude = _magnitude(vector)
    if magnitude == 0.0:
        return (0.0, 0.0, 0.0)
    return _scale(vector, 1.0 / magnitude)


def _dot(left: Vector3, right: Vector3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _centroid(points: tuple[Vector3, ...]) -> Vector3:
    if not points:
        return (0.0, 0.0, 0.0)
    factor = 1.0 / len(points)
    summed = [0.0, 0.0, 0.0]
    for point in points:
        for axis, value in enumerate(point):
            summed[axis] += value
    return (summed[0] * factor, summed[1] * factor, summed[2] * factor)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True, slots=True)
class SpatialSemanticFieldPolicy(ValidatableComponent):
    """Numerical guardrails for the spatial-semantic local field."""

    minimum_distance: float = 0.05
    energy_scale: float = 0.65
    repulsion_scale: float = 1.10
    alignment_floor: float = 0.25
    max_pair_energy_magnitude: float = 2.0
    max_pair_force_magnitude: float = 2.5

    def __post_init__(self) -> None:
        for field_name in (
            "minimum_distance",
            "energy_scale",
            "repulsion_scale",
            "alignment_floor",
            "max_pair_energy_magnitude",
            "max_pair_force_magnitude",
        ):
            object.__setattr__(self, field_name, coerce_scalar(getattr(self, field_name), field_name))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.minimum_distance <= 0.0:
            issues.append("minimum_distance must be strictly positive.")
        if self.energy_scale <= 0.0:
            issues.append("energy_scale must be strictly positive.")
        if self.repulsion_scale <= 0.0:
            issues.append("repulsion_scale must be strictly positive.")
        if not 0.0 <= self.alignment_floor <= 1.0:
            issues.append("alignment_floor must lie in the interval [0, 1].")
        if self.max_pair_energy_magnitude <= 0.0:
            issues.append("max_pair_energy_magnitude must be strictly positive.")
        if self.max_pair_force_magnitude <= 0.0:
            issues.append("max_pair_force_magnitude must be strictly positive.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class SpatialSemanticPairContribution(ValidatableComponent):
    """One spatially aware pair contribution inside a refinement region."""

    particle_index_a: int
    particle_index_b: int
    distance: float
    alignment: float
    chemistry_match: float
    pair_energy: float
    force_magnitude: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        for field_name in ("distance", "alignment", "chemistry_match", "pair_energy", "force_magnitude"):
            object.__setattr__(self, field_name, coerce_scalar(getattr(self, field_name), field_name))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.particle_index_a < 0 or self.particle_index_b < 0:
            issues.append("particle indices must be non-negative.")
        if self.distance <= 0.0:
            issues.append("distance must be strictly positive.")
        if not 0.0 <= self.alignment <= 1.0:
            issues.append("alignment must lie in the interval [0, 1].")
        if not 0.0 <= self.chemistry_match <= 1.0:
            issues.append("chemistry_match must lie in the interval [0, 1].")
        if self.force_magnitude < 0.0:
            issues.append("force_magnitude must be non-negative.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class SpatialSemanticFieldEvaluation(ValidatableComponent):
    """Spatial-semantic energy and force result for one refinement region."""

    energy_delta: float
    force_deltas: tuple[ParticleForceDelta, ...]
    pair_contributions: tuple[SpatialSemanticPairContribution, ...]
    quality_score: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy_delta", coerce_scalar(self.energy_delta, "energy_delta"))
        object.__setattr__(self, "force_deltas", tuple(self.force_deltas))
        object.__setattr__(self, "pair_contributions", tuple(self.pair_contributions))
        object.__setattr__(self, "quality_score", coerce_scalar(self.quality_score, "quality_score"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not 0.0 <= self.quality_score <= 1.0:
            issues.append("quality_score must lie in the interval [0, 1].")
        affected_particles = tuple(force_delta.particle_index for force_delta in self.force_deltas)
        if len(affected_particles) != len(set(affected_particles)):
            issues.append("force_deltas must not contain duplicate particle indices.")
        return tuple(issues)


@dataclass(slots=True)
class SpatialSemanticFieldModel(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Evaluate a local spatially aware interaction field inside one refinement region."""

    parameter_set: SpatialSemanticParameterSet
    chemistry_model: ProteinChemistryModel = field(default_factory=ProteinChemistryModel)
    policy: SpatialSemanticFieldPolicy = field(default_factory=SpatialSemanticFieldPolicy)
    name: str = "spatial_semantic_field_model"
    classification: str = "[proposed novel]"

    def describe_role(self) -> str:
        return (
            "Adds a repository-owned spatially aware and chemistry-aware local interaction field "
            "on top of the coarse substrate so intelligent corrections can remain fast and bounded."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "forcefields/spatial_semantic_profiles.py",
            "chemistry/residue_semantics.py",
            "qcloud/cloud_state.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/spatial_semantic_forcefield.md",)

    def validate(self) -> tuple[str, ...]:
        issues = list(self.parameter_set.validate())
        issues.extend(self.chemistry_model.validate())
        issues.extend(self.policy.validate())
        return tuple(issues)

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        region: RefinementRegion,
    ) -> SpatialSemanticFieldEvaluation:
        if topology.particle_count != state.particle_count:
            raise ContractValidationError("SystemTopology particle_count must match the SimulationState particle count.")
        if region.state_id != state.provenance.state_id:
            raise ContractValidationError("region.state_id must match the SimulationState state_id.")

        assignments = {
            assignment.particle_index: assignment
            for assignment in self.chemistry_model.assignments_for_topology(topology)
        }
        accumulated_forces = {particle_index: [0.0, 0.0, 0.0] for particle_index in region.particle_indices}
        pair_contributions: list[SpatialSemanticPairContribution] = []
        total_energy = 0.0

        particle_indices = tuple(region.particle_indices)
        for left_position, particle_index_a in enumerate(particle_indices):
            for particle_index_b in particle_indices[left_position + 1 :]:
                try:
                    profile = self.parameter_set.profile_for_bead_types(
                        topology.bead_for_particle(particle_index_a).bead_type,
                        topology.bead_for_particle(particle_index_b).bead_type,
                    )
                except KeyError:
                    continue

                position_a = state.particles.positions[particle_index_a]
                position_b = state.particles.positions[particle_index_b]
                displacement = _subtract(position_b, position_a)
                distance = max(_magnitude(displacement), self.policy.minimum_distance)
                if distance > profile.cutoff:
                    continue
                unit_ab = _scale(displacement, 1.0 / distance)

                exposure_a = self._exposure_direction(state, topology, particle_index_a)
                exposure_b = self._exposure_direction(state, topology, particle_index_b)
                alignment = self._pair_alignment(exposure_a, exposure_b, unit_ab)
                chemistry_match = self._chemistry_match(
                    assignments[particle_index_a].descriptor,
                    assignments[particle_index_b].descriptor,
                )

                energy_delta, force_on_a = self._pair_energy_force(
                    profile=profile,
                    distance=distance,
                    unit_ab=unit_ab,
                    alignment=alignment,
                    chemistry_match=chemistry_match,
                )
                if energy_delta == 0.0 and force_on_a == (0.0, 0.0, 0.0):
                    continue

                total_energy += energy_delta
                for axis, value in enumerate(force_on_a):
                    accumulated_forces[particle_index_a][axis] += value
                    accumulated_forces[particle_index_b][axis] -= value
                pair_contributions.append(
                    SpatialSemanticPairContribution(
                        particle_index_a=particle_index_a,
                        particle_index_b=particle_index_b,
                        distance=distance,
                        alignment=alignment,
                        chemistry_match=chemistry_match,
                        pair_energy=energy_delta,
                        force_magnitude=_magnitude(force_on_a),
                        metadata={
                            "bead_type_a": topology.bead_for_particle(particle_index_a).bead_type,
                            "bead_type_b": topology.bead_for_particle(particle_index_b).bead_type,
                            "source_label": profile.source_label,
                        },
                    )
                )

        force_deltas = tuple(
            ParticleForceDelta(
                particle_index=particle_index,
                delta_force=tuple(vector),
                metadata={"spatial_semantic": True},
            )
            for particle_index, vector in sorted(accumulated_forces.items())
            if any(abs(component) > 0.0 for component in vector)
        )
        quality_score = (
            sum((contribution.alignment + contribution.chemistry_match) * 0.5 for contribution in pair_contributions)
            / len(pair_contributions)
            if pair_contributions
            else 0.0
        )
        return SpatialSemanticFieldEvaluation(
            energy_delta=total_energy,
            force_deltas=force_deltas,
            pair_contributions=tuple(pair_contributions),
            quality_score=_clamp(quality_score, 0.0, 1.0),
            metadata={
                "pair_count": len(pair_contributions),
                "source_labels": tuple(sorted(self.parameter_set.source_labels())),
            },
        )

    def _exposure_direction(
        self,
        state: SimulationState,
        topology: SystemTopology,
        particle_index: int,
    ) -> Vector3:
        neighbors = topology.bonded_neighbors(particle_index)
        if not neighbors:
            return (0.0, 0.0, 0.0)
        neighbor_positions = tuple(state.particles.positions[index] for index in neighbors)
        return _normalize(
            _subtract(
                state.particles.positions[particle_index],
                _centroid(neighbor_positions),
            )
        )

    def _pair_alignment(
        self,
        exposure_a: Vector3,
        exposure_b: Vector3,
        unit_ab: Vector3,
    ) -> float:
        if exposure_a == (0.0, 0.0, 0.0) and exposure_b == (0.0, 0.0, 0.0):
            return 0.5
        facing_a = max(0.0, _dot(exposure_a, unit_ab))
        facing_b = max(0.0, _dot(exposure_b, _scale(unit_ab, -1.0)))
        return _clamp(0.5 * (facing_a + facing_b), 0.0, 1.0)

    def _chemistry_match(
        self,
        descriptor_a: ResidueChemistryDescriptor,
        descriptor_b: ResidueChemistryDescriptor,
    ) -> float:
        charge_product = descriptor_a.formal_charge * descriptor_b.formal_charge
        if abs(descriptor_a.formal_charge) >= 0.25 and abs(descriptor_b.formal_charge) >= 0.25:
            charge_score = 1.0 if charge_product < 0.0 else 0.10
        elif abs(descriptor_a.formal_charge) >= 0.25 or abs(descriptor_b.formal_charge) >= 0.25:
            charge_score = 0.70
        else:
            charge_score = 0.55
        hydropathy_score = 1.0 - min(1.0, abs(descriptor_a.hydropathy - descriptor_b.hydropathy) / 2.0)
        aromatic_score = sqrt(max(0.0, descriptor_a.aromaticity * descriptor_b.aromaticity))
        hotspot_score = sqrt(max(0.0, descriptor_a.hotspot_propensity * descriptor_b.hotspot_propensity))
        hydrogen_score = sqrt(
            max(0.0, descriptor_a.hydrogen_bond_capacity * descriptor_b.hydrogen_bond_capacity)
        )
        return _clamp(
            0.28 * charge_score
            + 0.20 * hydropathy_score
            + 0.18 * hydrogen_score
            + 0.18 * aromatic_score
            + 0.16 * hotspot_score,
            0.0,
            1.0,
        )

    def _pair_energy_force(
        self,
        *,
        profile,
        distance: float,
        unit_ab: Vector3,
        alignment: float,
        chemistry_match: float,
    ) -> tuple[float, Vector3]:
        direction_scale = 1.0 + profile.directional_strength * max(0.0, alignment - self.policy.alignment_floor)
        chemistry_scale = 0.55 + profile.chemistry_strength * chemistry_match
        delta = distance - profile.preferred_distance
        tolerance = max(profile.distance_tolerance, self.policy.minimum_distance)
        gaussian = exp(-0.5 * (delta / tolerance) ** 2)

        attractive_energy = -profile.attraction_strength * gaussian * direction_scale * chemistry_scale * self.policy.energy_scale
        attractive_force_magnitude = (
            profile.attraction_strength
            * gaussian
            * abs(delta)
            / tolerance
            * direction_scale
            * chemistry_scale
            * self.policy.energy_scale
        )
        attractive_sign = 1.0 if delta > 0.0 else -1.0
        attractive_force = _scale(unit_ab, attractive_force_magnitude * attractive_sign)

        repulsive_energy = 0.0
        repulsive_force = (0.0, 0.0, 0.0)
        if distance < profile.minimum_distance:
            ratio = profile.minimum_distance / distance
            repulsive_energy = (
                profile.repulsion_strength
                * ((ratio**12) - 1.0)
                * self.policy.repulsion_scale
            )
            repulsive_force_magnitude = (
                profile.repulsion_strength
                * self.policy.repulsion_scale
                * (ratio**12)
                / distance
            )
            repulsive_force = _scale(unit_ab, -repulsive_force_magnitude)

        total_energy = _clamp(
            attractive_energy + repulsive_energy,
            -self.policy.max_pair_energy_magnitude,
            self.policy.max_pair_energy_magnitude,
        )
        total_force = tuple(attractive_force[axis] + repulsive_force[axis] for axis in range(3))
        force_magnitude = _magnitude(total_force)
        if force_magnitude > self.policy.max_pair_force_magnitude and force_magnitude > 0.0:
            total_force = _scale(total_force, self.policy.max_pair_force_magnitude / force_magnitude)
        return total_energy, total_force


__all__ = [
    "SpatialSemanticFieldEvaluation",
    "SpatialSemanticFieldModel",
    "SpatialSemanticFieldPolicy",
    "SpatialSemanticPairContribution",
]
