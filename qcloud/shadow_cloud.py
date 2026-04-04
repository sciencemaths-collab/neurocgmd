"""Concrete shadow-cloud snapshots mirrored from the coarse-grained body."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata, StateId, Vector3, coerce_scalar, coerce_vector3
from qcloud.shadow_mapping import ShadowMappingLibrary
from topology.system_topology import SystemTopology


@dataclass(frozen=True, slots=True)
class ShadowSite(ValidatableComponent):
    """One mirrored shadow site derived from a coarse particle."""

    site_id: str
    parent_particle_index: int
    bead_type: str
    site_name: str
    position: Vector3
    sigma_scale: float = 1.0
    epsilon_scale: float = 1.0
    charge_scale: float = 0.0
    occupancy: float = 1.0
    source_label: str = ""
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", coerce_vector3(self.position, "position"))
        object.__setattr__(self, "sigma_scale", coerce_scalar(self.sigma_scale, "sigma_scale"))
        object.__setattr__(self, "epsilon_scale", coerce_scalar(self.epsilon_scale, "epsilon_scale"))
        object.__setattr__(self, "charge_scale", coerce_scalar(self.charge_scale, "charge_scale"))
        object.__setattr__(self, "occupancy", coerce_scalar(self.occupancy, "occupancy"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.site_id.strip():
            issues.append("site_id must be a non-empty string.")
        if self.parent_particle_index < 0:
            issues.append("parent_particle_index must be non-negative.")
        if not self.bead_type.strip():
            issues.append("bead_type must be a non-empty string.")
        if not self.site_name.strip():
            issues.append("site_name must be a non-empty string.")
        if self.sigma_scale <= 0.0:
            issues.append("sigma_scale must be strictly positive.")
        if self.epsilon_scale < 0.0:
            issues.append("epsilon_scale must be non-negative.")
        if self.occupancy <= 0.0:
            issues.append("occupancy must be strictly positive.")
        if not self.source_label.strip():
            issues.append("source_label must be a non-empty string.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ShadowCloudSnapshot(ValidatableComponent):
    """Shadow cloud built for one simulation state and optional local region."""

    state_id: StateId
    particle_indices: tuple[int, ...]
    sites: tuple[ShadowSite, ...]
    source_labels: tuple[str, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", StateId(str(self.state_id)))
        object.__setattr__(self, "particle_indices", tuple(sorted(set(int(index) for index in self.particle_indices))))
        object.__setattr__(self, "sites", tuple(self.sites))
        object.__setattr__(self, "source_labels", tuple(dict.fromkeys(str(label) for label in self.source_labels)))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def site_count(self) -> int:
        return len(self.sites)

    def sites_for_particle(self, particle_index: int) -> tuple[ShadowSite, ...]:
        return tuple(site for site in self.sites if site.parent_particle_index == particle_index)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.particle_indices:
            issues.append("particle_indices must contain at least one particle.")
        if not self.sites:
            issues.append("sites must contain at least one shadow site.")
        if not self.source_labels:
            issues.append("source_labels must contain at least one source label.")
        site_ids = tuple(site.site_id for site in self.sites)
        if len(site_ids) != len(set(site_ids)):
            issues.append("shadow site ids must be unique.")
        return tuple(issues)


@dataclass(slots=True)
class ShadowCloudBuilder(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Build mirrored shadow-cloud snapshots from the canonical coarse state."""

    mapping_library: ShadowMappingLibrary
    name: str = "shadow_cloud_builder"
    classification: str = "[proposed novel]"

    def describe_role(self) -> str:
        return (
            "Builds a mirrored shadow cloud around the coarse-grained body so "
            "trusted high-fidelity correction structure can be attached locally."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "core/state.py",
            "topology/system_topology.py",
            "qcloud/shadow_mapping.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/shadow_coarse_grained_fidelity.md",)

    def validate(self) -> tuple[str, ...]:
        return self.mapping_library.validate()

    def build(
        self,
        state: SimulationState,
        topology: SystemTopology,
        *,
        particle_indices: tuple[int, ...] | list[int] | None = None,
    ) -> ShadowCloudSnapshot:
        if topology.particle_count != state.particle_count:
            raise ContractValidationError("SystemTopology particle_count must match the SimulationState particle count.")

        selected_particles = (
            tuple(sorted(set(int(index) for index in particle_indices)))
            if particle_indices is not None
            else tuple(range(state.particle_count))
        )
        if not selected_particles:
            raise ContractValidationError("particle_indices must contain at least one particle.")

        sites: list[ShadowSite] = []
        source_labels: list[str] = []
        for particle_index in selected_particles:
            if particle_index < 0 or particle_index >= state.particle_count:
                raise ContractValidationError("particle_indices must reference valid particles.")
            bead = topology.bead_for_particle(particle_index)
            try:
                rule = self.mapping_library.rule_for_bead_type(bead.bead_type)
            except KeyError as exc:
                raise ContractValidationError(
                    f"No shadow mapping rule is registered for bead type {bead.bead_type!r}."
                ) from exc
            source_labels.append(rule.source_label)
            anchor = state.particles.positions[particle_index]
            for template in rule.site_templates:
                position = tuple(
                    anchor[axis] + rule.mirror_scale * template.relative_offset[axis]
                    for axis in range(3)
                )
                sites.append(
                    ShadowSite(
                        site_id=f"{state.provenance.state_id}-p{particle_index:04d}-{template.site_name}",
                        parent_particle_index=particle_index,
                        bead_type=bead.bead_type,
                        site_name=template.site_name,
                        position=position,
                        sigma_scale=template.sigma_scale,
                        epsilon_scale=template.epsilon_scale,
                        charge_scale=template.charge_scale,
                        occupancy=template.occupancy,
                        source_label=rule.source_label,
                        metadata=template.metadata.with_updates(
                            {
                                "bead_label": bead.label,
                                "rule_bead_type": rule.bead_type,
                            }
                        ),
                    )
                )

        return ShadowCloudSnapshot(
            state_id=state.provenance.state_id,
            particle_indices=selected_particles,
            sites=tuple(sites),
            source_labels=tuple(source_labels),
            metadata=FrozenMetadata(
                {
                    "site_count": len(sites),
                    "selected_particle_count": len(selected_particles),
                }
            ),
        )
