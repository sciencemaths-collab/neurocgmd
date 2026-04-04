"""Domain and compartment descriptors for modular molecular organization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from core.exceptions import ContractValidationError
from core.interfaces import ValidatableComponent
from core.types import CompartmentId, FrozenMetadata


class CompartmentRole(StrEnum):
    """High-level role of a compartment in the compartment overlay."""

    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    BRIDGE = "bridge"
    REGULATORY = "regulatory"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class CompartmentDomain(ValidatableComponent):
    """Explicit compartment/domain over a set of particle indices."""

    compartment_id: CompartmentId
    name: str
    role: CompartmentRole = CompartmentRole.STRUCTURAL
    particle_indices: tuple[int, ...] = ()
    priority: int = 0
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "compartment_id", CompartmentId(str(self.compartment_id)))
        object.__setattr__(self, "particle_indices", tuple(sorted(set(self.particle_indices))))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    @property
    def size(self) -> int:
        return len(self.particle_indices)

    def contains_particle(self, particle_index: int) -> bool:
        return particle_index in self.particle_indices

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not str(self.compartment_id).strip():
            issues.append("compartment_id must be a non-empty string.")
        if not self.name.strip():
            issues.append("name must be a non-empty string.")
        if not self.particle_indices:
            issues.append("particle_indices must contain at least one particle.")
        if any(index < 0 for index in self.particle_indices):
            issues.append("particle_indices must be non-negative.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "compartment_id": str(self.compartment_id),
            "name": self.name,
            "role": self.role.value,
            "particle_indices": list(self.particle_indices),
            "priority": self.priority,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CompartmentDomain":
        return cls(
            compartment_id=CompartmentId(str(data["compartment_id"])),
            name=str(data["name"]),
            role=CompartmentRole(data.get("role", CompartmentRole.STRUCTURAL.value)),
            particle_indices=tuple(int(index) for index in data.get("particle_indices", ())),
            priority=int(data.get("priority", 0)),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )

    @classmethod
    def from_members(
        cls,
        compartment_id: CompartmentId | str,
        name: str,
        particle_indices: Sequence[int],
        *,
        role: CompartmentRole = CompartmentRole.STRUCTURAL,
        priority: int = 0,
        metadata: FrozenMetadata | Mapping[str, object] | None = None,
    ) -> "CompartmentDomain":
        """Convenience constructor from a particle-index sequence."""

        return cls(
            compartment_id=CompartmentId(str(compartment_id)),
            name=name,
            role=role,
            particle_indices=tuple(particle_indices),
            priority=priority,
            metadata=metadata if isinstance(metadata, FrozenMetadata) else FrozenMetadata(metadata),
        )

