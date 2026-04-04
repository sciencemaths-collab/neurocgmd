"""Shadow coarse-grained correction model driven by trusted parameter sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata, Vector3, coerce_scalar
from forcefields.trusted_sources import TrustedNonbondedProfile, TrustedParameterSet
from qcloud.cloud_state import ParticleForceDelta, QCloudCorrection, RefinementRegion
from qcloud.spatial_semantic_field import SpatialSemanticFieldModel
from qcloud.shadow_cloud import ShadowCloudBuilder, ShadowSite
from topology.system_topology import SystemTopology


def _vector_difference(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def _vector_magnitude(vector: Vector3) -> float:
    return sqrt(sum(component * component for component in vector))


@dataclass(frozen=True, slots=True)
class ShadowCorrectionPolicy(ValidatableComponent):
    """Numerical guardrails for shadow coarse-grained corrections."""

    minimum_site_distance: float = 0.05
    max_interaction_distance: float | None = None
    energy_scale: float = 1.0
    electrostatic_scale: float = 1.0
    exclude_same_parent_pairs: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_site_distance",
            coerce_scalar(self.minimum_site_distance, "minimum_site_distance"),
        )
        if self.max_interaction_distance is not None:
            object.__setattr__(
                self,
                "max_interaction_distance",
                coerce_scalar(self.max_interaction_distance, "max_interaction_distance"),
            )
        object.__setattr__(self, "energy_scale", coerce_scalar(self.energy_scale, "energy_scale"))
        object.__setattr__(
            self,
            "electrostatic_scale",
            coerce_scalar(self.electrostatic_scale, "electrostatic_scale"),
        )
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.minimum_site_distance <= 0.0:
            issues.append("minimum_site_distance must be strictly positive.")
        if self.max_interaction_distance is not None and self.max_interaction_distance <= 0.0:
            issues.append("max_interaction_distance must be strictly positive when provided.")
        if self.energy_scale <= 0.0:
            issues.append("energy_scale must be strictly positive.")
        if self.electrostatic_scale < 0.0:
            issues.append("electrostatic_scale must be non-negative.")
        return tuple(issues)


@dataclass(slots=True)
class ShadowDrivenCorrectionModel(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Use a mirrored shadow cloud plus trusted parameters to generate local corrections."""

    trusted_parameter_set: TrustedParameterSet
    shadow_builder: ShadowCloudBuilder
    policy: ShadowCorrectionPolicy = field(default_factory=ShadowCorrectionPolicy)
    spatial_field_model: SpatialSemanticFieldModel | None = None
    name: str = "shadow_driven_correction_model"
    classification: str = "[proposed novel]"

    def describe_role(self) -> str:
        return (
            "Uses a mirrored shadow cloud plus an optional spatial-semantic local field "
            "to apply trusted high-fidelity corrections while preserving a fast coarse substrate."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "forcefields/trusted_sources.py",
            "qcloud/shadow_mapping.py",
            "qcloud/shadow_cloud.py",
            "qcloud/spatial_semantic_field.py",
            "qcloud/cloud_state.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/shadow_coarse_grained_fidelity.md",
            "docs/architecture/qcloud_framework.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues = list(self.trusted_parameter_set.validate())
        issues.extend(self.shadow_builder.validate())
        issues.extend(self.policy.validate())
        if self.spatial_field_model is not None:
            issues.extend(self.spatial_field_model.validate())
        return tuple(issues)

    def _site_pair_energy_force(
        self,
        site_a: ShadowSite,
        site_b: ShadowSite,
        profile: TrustedNonbondedProfile,
    ) -> tuple[float, Vector3]:
        displacement = _vector_difference(site_b.position, site_a.position)
        distance = max(_vector_magnitude(displacement), self.policy.minimum_site_distance)
        interaction_cutoff = min(
            profile.cutoff,
            self.policy.max_interaction_distance if self.policy.max_interaction_distance is not None else profile.cutoff,
        )
        if distance > interaction_cutoff:
            return 0.0, (0.0, 0.0, 0.0)

        occupancy_weight = site_a.occupancy * site_b.occupancy
        sigma = profile.sigma * 0.5 * (site_a.sigma_scale + site_b.sigma_scale)
        epsilon = profile.epsilon * sqrt(site_a.epsilon_scale * site_b.epsilon_scale)
        electrostatic_strength = (
            profile.electrostatic_strength
            * site_a.charge_scale
            * site_b.charge_scale
            * self.policy.electrostatic_scale
        )
        sr = sigma / distance
        sr6 = sr**6
        sr12 = sr6 * sr6
        pair_energy = occupancy_weight * self.policy.energy_scale * (
            4.0 * epsilon * (sr12 - sr6) + electrostatic_strength / distance
        )
        pair_scale = occupancy_weight * self.policy.energy_scale * (
            24.0 * epsilon * (2.0 * sr12 - sr6) / (distance * distance)
            + electrostatic_strength / (distance**3)
        )
        force_on_a = tuple(-pair_scale * component for component in displacement)
        return pair_energy, force_on_a

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        region: RefinementRegion,
    ) -> QCloudCorrection:
        if topology.particle_count != state.particle_count:
            raise ContractValidationError("SystemTopology particle_count must match the SimulationState particle count.")
        if region.state_id != state.provenance.state_id:
            raise ContractValidationError("region.state_id must match the SimulationState state_id.")

        shadow_cloud = self.shadow_builder.build(
            state,
            topology,
            particle_indices=region.particle_indices,
        )

        accumulated_forces = {particle_index: [0.0, 0.0, 0.0] for particle_index in region.particle_indices}
        total_energy = 0.0
        active_pair_count = 0
        unresolved_pair_count = 0
        source_labels: set[str] = set(shadow_cloud.source_labels)
        spatial_pair_count = 0
        spatial_quality_score = 0.0

        sites = shadow_cloud.sites
        for left_index, site_a in enumerate(sites):
            for site_b in sites[left_index + 1 :]:
                if self.policy.exclude_same_parent_pairs and (
                    site_a.parent_particle_index == site_b.parent_particle_index
                ):
                    continue
                try:
                    profile = self.trusted_parameter_set.nonbonded_profile_for_bead_types(
                        site_a.bead_type,
                        site_b.bead_type,
                    )
                except KeyError:
                    unresolved_pair_count += 1
                    continue

                pair_energy, force_on_a = self._site_pair_energy_force(site_a, site_b, profile)
                if pair_energy == 0.0 and force_on_a == (0.0, 0.0, 0.0):
                    continue

                total_energy += pair_energy
                active_pair_count += 1
                source_labels.add(profile.source_label)

                for axis, component in enumerate(force_on_a):
                    accumulated_forces[site_a.parent_particle_index][axis] += component
                    accumulated_forces[site_b.parent_particle_index][axis] -= component

        if self.spatial_field_model is not None:
            spatial_field = self.spatial_field_model.evaluate(state, topology, region)
            total_energy += spatial_field.energy_delta
            spatial_pair_count = len(spatial_field.pair_contributions)
            spatial_quality_score = spatial_field.quality_score
            source_labels.update(spatial_field.metadata["source_labels"])
            for force_delta in spatial_field.force_deltas:
                vector = accumulated_forces.setdefault(force_delta.particle_index, [0.0, 0.0, 0.0])
                for axis, value in enumerate(force_delta.delta_force):
                    vector[axis] += value

        force_deltas = tuple(
            ParticleForceDelta(
                particle_index=particle_index,
                delta_force=tuple(vector),
                metadata={"shadow_corrected": True},
            )
            for particle_index, vector in sorted(accumulated_forces.items())
            if any(abs(component) > 0.0 for component in vector)
        )

        pair_capacity = max(1, len(sites) * (len(sites) - 1) // 2)
        confidence = min(1.0, active_pair_count / pair_capacity)
        if not force_deltas and total_energy == 0.0:
            confidence = 0.0 if unresolved_pair_count == 0 else min(confidence, 0.25)

        return QCloudCorrection(
            region_id=region.region_id,
            method_label=self.name,
            energy_delta=total_energy,
            force_deltas=force_deltas,
            confidence=confidence,
            metadata=FrozenMetadata(
                {
                    "trusted_parameter_set": self.trusted_parameter_set.name,
                    "shadow_site_count": shadow_cloud.site_count(),
                    "active_pair_count": active_pair_count,
                    "unresolved_pair_count": unresolved_pair_count,
                    "spatial_semantic_pair_count": spatial_pair_count,
                    "spatial_semantic_quality_score": spatial_quality_score,
                    "source_labels": tuple(sorted(source_labels)),
                }
            ),
        )
