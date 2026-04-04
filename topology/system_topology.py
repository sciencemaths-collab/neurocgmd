"""System-level fixed topology assembly aligned to particle-state indices."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import ParticleState
from core.types import BeadId, FrozenMetadata
from topology.beads import Bead, BeadType
from topology.bonds import Bond, build_neighbor_map, connected_components


@dataclass(frozen=True, slots=True)
class SystemTopology(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Canonical static topology for one simulated coarse-grained system."""

    name: str = "system_topology"
    classification: str = "[adapted]"
    system_id: str = "system"
    bead_types: tuple[BeadType, ...] = ()
    beads: tuple[Bead, ...] = ()
    bonds: tuple[Bond, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bead_types", tuple(self.bead_types))
        object.__setattr__(self, "beads", tuple(self.beads))
        object.__setattr__(self, "bonds", tuple(self.bonds))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    @property
    def particle_count(self) -> int:
        return len(self.beads)

    @property
    def bead_type_names(self) -> tuple[str, ...]:
        return tuple(bead_type.name for bead_type in self.bead_types)

    def describe_role(self) -> str:
        return (
            "Represents the fixed bead-level topology that constrains physical "
            "connectivity before any adaptive graph rewiring is introduced."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "core/types.py",
            "core/state.py",
            "topology/beads.py",
            "topology/bonds.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/topology_model.md",
            "docs/sections/section_03_topology_and_bead_system.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.system_id.strip():
            issues.append("system_id must be a non-empty string.")
        if not self.beads:
            issues.append("SystemTopology must contain at least one bead.")

        bead_type_names = [bead_type.name for bead_type in self.bead_types]
        if len(bead_type_names) != len(set(bead_type_names)):
            issues.append("bead_types must have unique names.")

        bead_ids = [str(bead.bead_id) for bead in self.beads]
        if len(bead_ids) != len(set(bead_ids)):
            issues.append("beads must have unique bead_id values.")

        particle_indices = [bead.particle_index for bead in self.beads]
        if len(particle_indices) != len(set(particle_indices)):
            issues.append("beads must map to unique particle_index values.")
        if particle_indices and sorted(particle_indices) != list(range(len(self.beads))):
            issues.append(
                "bead particle indices must be contiguous and start at zero to align with ParticleState."
            )

        bead_type_lookup = set(bead_type_names)
        missing_types = sorted(
            {bead.bead_type for bead in self.beads if bead.bead_type not in bead_type_lookup}
        )
        if missing_types:
            issues.append(
                "Every bead.bead_type must resolve to a declared BeadType; missing: "
                + ", ".join(missing_types)
            )

        normalized_pairs = [bond.normalized_pair() for bond in self.bonds]
        if len(normalized_pairs) != len(set(normalized_pairs)):
            issues.append("bonds must not contain duplicate undirected endpoint pairs.")
        for pair in normalized_pairs:
            if pair[1] >= len(self.beads):
                issues.append(
                    f"Bond pair {pair} references a particle index outside the bead range."
                )

        return tuple(issues)

    def bead_type_by_name(self, bead_type_name: str) -> BeadType:
        for bead_type in self.bead_types:
            if bead_type.name == bead_type_name:
                return bead_type
        raise KeyError(bead_type_name)

    def bead_by_id(self, bead_id: BeadId) -> Bead:
        bead_id_str = str(bead_id)
        for bead in self.beads:
            if str(bead.bead_id) == bead_id_str:
                return bead
        raise KeyError(bead_id_str)

    def bead_for_particle(self, particle_index: int) -> Bead:
        for bead in self.beads:
            if bead.particle_index == particle_index:
                return bead
        raise KeyError(particle_index)

    def bonded_neighbors(self, particle_index: int) -> tuple[int, ...]:
        return build_neighbor_map(self.particle_count, self.bonds)[particle_index]

    def bonds_for_particle(self, particle_index: int) -> tuple[Bond, ...]:
        return tuple(
            bond
            for bond in self.bonds
            if particle_index in {bond.particle_index_a, bond.particle_index_b}
        )

    def connected_components(self) -> tuple[tuple[int, ...], ...]:
        return connected_components(self.particle_count, self.bonds)

    def validate_against_particle_state(self, particles: ParticleState) -> tuple[str, ...]:
        """Check alignment against the canonical particle-state substrate."""

        issues: list[str] = []
        if particles.particle_count != self.particle_count:
            issues.append(
                "ParticleState particle_count does not match SystemTopology bead count."
            )
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "classification": self.classification,
            "system_id": self.system_id,
            "bead_types": [bead_type.to_dict() for bead_type in self.bead_types],
            "beads": [bead.to_dict() for bead in self.beads],
            "bonds": [bond.to_dict() for bond in self.bonds],
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SystemTopology":
        return cls(
            name=str(data.get("name", "system_topology")),
            classification=str(data.get("classification", "[adapted]")),
            system_id=str(data.get("system_id", "system")),
            bead_types=tuple(BeadType.from_dict(item) for item in data.get("bead_types", ())),
            beads=tuple(Bead.from_dict(item) for item in data.get("beads", ())),
            bonds=tuple(Bond.from_dict(item) for item in data.get("bonds", ())),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )

