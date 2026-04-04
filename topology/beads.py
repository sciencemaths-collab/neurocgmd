"""Bead type and bead descriptors for coarse-grained system topology."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from core.exceptions import ContractValidationError
from core.interfaces import ValidatableComponent
from core.types import BeadId, CompartmentId, FrozenMetadata


class BeadRole(StrEnum):
    """Coarse semantic role of a bead in the static topology."""

    GENERIC = "generic"
    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    LINKER = "linker"
    ANCHOR = "anchor"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class BeadType(ValidatableComponent):
    """Reusable static bead type definition."""

    name: str
    role: BeadRole = BeadRole.GENERIC
    description: str = ""
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name.strip():
            issues.append("BeadType name must be a non-empty string.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role.value,
            "description": self.description,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "BeadType":
        return cls(
            name=str(data["name"]),
            role=BeadRole(data.get("role", BeadRole.GENERIC.value)),
            description=str(data.get("description", "")),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class Bead(ValidatableComponent):
    """Static mapping from one topology bead to one particle-state index."""

    bead_id: BeadId
    particle_index: int
    bead_type: str
    label: str
    residue_name: str | None = None
    chain_id: str | None = None
    compartment_hint: CompartmentId | None = None
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bead_id", BeadId(str(self.bead_id)))
        if self.compartment_hint is not None:
            object.__setattr__(
                self,
                "compartment_hint",
                CompartmentId(str(self.compartment_hint)),
            )
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not str(self.bead_id).strip():
            issues.append("bead_id must be a non-empty string.")
        if self.particle_index < 0:
            issues.append("particle_index must be non-negative.")
        if not self.bead_type.strip():
            issues.append("bead_type must be a non-empty string.")
        if not self.label.strip():
            issues.append("label must be a non-empty string.")
        if self.residue_name is not None and not self.residue_name.strip():
            issues.append("residue_name must be non-empty when provided.")
        if self.chain_id is not None and not self.chain_id.strip():
            issues.append("chain_id must be non-empty when provided.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "bead_id": str(self.bead_id),
            "particle_index": self.particle_index,
            "bead_type": self.bead_type,
            "label": self.label,
            "residue_name": self.residue_name,
            "chain_id": self.chain_id,
            "compartment_hint": str(self.compartment_hint) if self.compartment_hint else None,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Bead":
        compartment_hint = data.get("compartment_hint")
        return cls(
            bead_id=BeadId(str(data["bead_id"])),
            particle_index=int(data["particle_index"]),
            bead_type=str(data["bead_type"]),
            label=str(data["label"]),
            residue_name=str(data["residue_name"]) if data.get("residue_name") else None,
            chain_id=str(data["chain_id"]) if data.get("chain_id") else None,
            compartment_hint=(
                CompartmentId(str(compartment_hint)) if compartment_hint else None
            ),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )

