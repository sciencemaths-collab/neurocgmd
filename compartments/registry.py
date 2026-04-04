"""Registry and membership helpers for compartment overlays."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

from compartments.domain_models import CompartmentDomain
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import CompartmentId, FrozenMetadata
from topology.system_topology import SystemTopology


@dataclass(frozen=True, slots=True)
class CompartmentRegistry(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Registry of compartment domains over particle-aligned state."""

    name: str = "compartment_registry"
    classification: str = "[hybrid]"
    particle_count: int = 0
    domains: tuple[CompartmentDomain, ...] = ()
    allow_overlap: bool = False
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "domains", tuple(self.domains))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Represents a modular compartment overlay on top of particle, topology, "
            "and graph state without replacing those lower-level substrates."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "core/state.py",
            "topology/system_topology.py",
            "compartments/domain_models.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/compartment_system.md",
            "docs/sections/section_08_compartment_system.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.particle_count <= 0:
            issues.append("particle_count must be positive.")
        if not self.domains:
            issues.append("domains must contain at least one compartment.")
        ids = [str(domain.compartment_id) for domain in self.domains]
        if len(ids) != len(set(ids)):
            issues.append("domains must have unique compartment_id values.")

        memberships: dict[int, list[str]] = defaultdict(list)
        for domain in self.domains:
            for particle_index in domain.particle_indices:
                if particle_index >= self.particle_count:
                    issues.append(
                        f"Compartment {domain.compartment_id} references out-of-range particle {particle_index}."
                    )
                memberships[particle_index].append(str(domain.compartment_id))

        if not self.allow_overlap:
            overlapping_particles = {
                index: compartment_ids
                for index, compartment_ids in memberships.items()
                if len(compartment_ids) > 1
            }
            if overlapping_particles:
                issues.append(
                    "Particle membership overlaps are not allowed; overlapping particles: "
                    + ", ".join(
                        f"{particle_index}->{compartment_ids}"
                        for particle_index, compartment_ids in sorted(overlapping_particles.items())
                    )
                )
        return tuple(issues)

    def domain_by_id(self, compartment_id: CompartmentId | str) -> CompartmentDomain:
        identifier = str(compartment_id)
        for domain in self.domains:
            if str(domain.compartment_id) == identifier:
                return domain
        raise KeyError(identifier)

    def domains_for_particle(self, particle_index: int) -> tuple[CompartmentDomain, ...]:
        return tuple(
            sorted(
                (
                    domain
                    for domain in self.domains
                    if domain.contains_particle(particle_index)
                ),
                key=lambda domain: (-domain.priority, str(domain.compartment_id)),
            )
        )

    def primary_domain_for_particle(self, particle_index: int) -> CompartmentDomain | None:
        domains = self.domains_for_particle(particle_index)
        return domains[0] if domains else None

    def membership_map(self) -> dict[int, tuple[CompartmentId, ...]]:
        return {
            particle_index: tuple(domain.compartment_id for domain in self.domains_for_particle(particle_index))
            for particle_index in range(self.particle_count)
        }

    def unassigned_particles(self) -> tuple[int, ...]:
        return tuple(
            particle_index
            for particle_index in range(self.particle_count)
            if not self.domains_for_particle(particle_index)
        )

    def validate_against_topology(self, topology: SystemTopology) -> tuple[str, ...]:
        issues: list[str] = []
        if topology.particle_count != self.particle_count:
            issues.append("CompartmentRegistry particle_count does not match SystemTopology.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "classification": self.classification,
            "particle_count": self.particle_count,
            "allow_overlap": self.allow_overlap,
            "domains": [domain.to_dict() for domain in self.domains],
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CompartmentRegistry":
        return cls(
            name=str(data.get("name", "compartment_registry")),
            classification=str(data.get("classification", "[hybrid]")),
            particle_count=int(data["particle_count"]),
            allow_overlap=bool(data.get("allow_overlap", False)),
            domains=tuple(CompartmentDomain.from_dict(item) for item in data.get("domains", ())),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )

    @classmethod
    def from_topology_hints(
        cls,
        topology: SystemTopology,
        *,
        allow_overlap: bool = False,
    ) -> "CompartmentRegistry":
        """Build a registry from `Bead.compartment_hint` values in the topology."""

        members_by_compartment: dict[str, list[int]] = defaultdict(list)
        for bead in topology.beads:
            if bead.compartment_hint is not None:
                members_by_compartment[str(bead.compartment_hint)].append(bead.particle_index)
        if not members_by_compartment:
            raise ContractValidationError("No topology compartment hints were found.")

        domains = tuple(
            CompartmentDomain.from_members(
                compartment_id=compartment_id,
                name=compartment_id,
                particle_indices=tuple(sorted(indices)),
            )
            for compartment_id, indices in sorted(members_by_compartment.items())
        )
        return cls(
            particle_count=topology.particle_count,
            domains=domains,
            allow_overlap=allow_overlap,
            metadata={"source": "topology_hints"},
        )

