"""Configuration contracts for arbitrary protein import and coarse mapping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, coerce_scalar


@dataclass(frozen=True, slots=True)
class ProteinEntityGroup(ValidatableComponent):
    """Named chain group imported as one semantic entity."""

    entity_id: str
    chain_ids: tuple[str, ...]
    description: str = ""
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "chain_ids", tuple(str(chain_id).strip() for chain_id in self.chain_ids))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.entity_id.strip():
            issues.append("entity_id must be a non-empty string.")
        if not self.chain_ids:
            issues.append("chain_ids must contain at least one chain identifier.")
        if any(not chain_id for chain_id in self.chain_ids):
            issues.append("chain_ids must be non-empty strings.")
        if len(self.chain_ids) != len(set(self.chain_ids)):
            issues.append("chain_ids must be unique within one entity group.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "chain_ids": list(self.chain_ids),
            "description": self.description,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProteinEntityGroup":
        return cls(
            entity_id=str(data["entity_id"]),
            chain_ids=tuple(data.get("chain_ids", ())),
            description=str(data.get("description", "")),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ProteinMappingConfig(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """[hybrid] Bounded defaults for importing arbitrary proteins into the coarse substrate."""

    name: str = "protein_mapping_config"
    classification: str = "[hybrid]"
    residues_per_bead: int = 8
    angstrom_to_nm_scale: float = 0.1
    mean_residue_mass: float = 110.0
    structural_bond_stiffness: float = 85.0
    interface_contact_cutoff_nm: float = 1.45
    interface_contact_tolerance_nm: float = 0.18
    max_interface_contacts: int = 16
    minimum_entity_beads: int = 2
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "angstrom_to_nm_scale", coerce_scalar(self.angstrom_to_nm_scale, "angstrom_to_nm_scale"))
        object.__setattr__(self, "mean_residue_mass", coerce_scalar(self.mean_residue_mass, "mean_residue_mass"))
        object.__setattr__(
            self,
            "structural_bond_stiffness",
            coerce_scalar(self.structural_bond_stiffness, "structural_bond_stiffness"),
        )
        object.__setattr__(
            self,
            "interface_contact_cutoff_nm",
            coerce_scalar(self.interface_contact_cutoff_nm, "interface_contact_cutoff_nm"),
        )
        object.__setattr__(
            self,
            "interface_contact_tolerance_nm",
            coerce_scalar(self.interface_contact_tolerance_nm, "interface_contact_tolerance_nm"),
        )
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Defines reusable import-time choices for converting arbitrary local PDB "
            "structures into the repository's protein-general coarse substrate."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return ("io/pdb_loader.py", "topology/protein_coarse_mapping.py")

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/arbitrary_protein_input_pipeline.md",)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.residues_per_bead <= 0:
            issues.append("residues_per_bead must be strictly positive.")
        if self.angstrom_to_nm_scale <= 0.0:
            issues.append("angstrom_to_nm_scale must be strictly positive.")
        if self.mean_residue_mass <= 0.0:
            issues.append("mean_residue_mass must be strictly positive.")
        if self.structural_bond_stiffness <= 0.0:
            issues.append("structural_bond_stiffness must be strictly positive.")
        if self.interface_contact_cutoff_nm <= 0.0:
            issues.append("interface_contact_cutoff_nm must be strictly positive.")
        if self.interface_contact_tolerance_nm < 0.0:
            issues.append("interface_contact_tolerance_nm must be non-negative.")
        if self.max_interface_contacts <= 0:
            issues.append("max_interface_contacts must be strictly positive.")
        if self.minimum_entity_beads <= 0:
            issues.append("minimum_entity_beads must be strictly positive.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "classification": self.classification,
            "residues_per_bead": self.residues_per_bead,
            "angstrom_to_nm_scale": self.angstrom_to_nm_scale,
            "mean_residue_mass": self.mean_residue_mass,
            "structural_bond_stiffness": self.structural_bond_stiffness,
            "interface_contact_cutoff_nm": self.interface_contact_cutoff_nm,
            "interface_contact_tolerance_nm": self.interface_contact_tolerance_nm,
            "max_interface_contacts": self.max_interface_contacts,
            "minimum_entity_beads": self.minimum_entity_beads,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProteinMappingConfig":
        return cls(
            name=str(data.get("name", "protein_mapping_config")),
            classification=str(data.get("classification", "[hybrid]")),
            residues_per_bead=int(data.get("residues_per_bead", 8)),
            angstrom_to_nm_scale=float(data.get("angstrom_to_nm_scale", 0.1)),
            mean_residue_mass=float(data.get("mean_residue_mass", 110.0)),
            structural_bond_stiffness=float(data.get("structural_bond_stiffness", 85.0)),
            interface_contact_cutoff_nm=float(data.get("interface_contact_cutoff_nm", 1.45)),
            interface_contact_tolerance_nm=float(data.get("interface_contact_tolerance_nm", 0.18)),
            max_interface_contacts=int(data.get("max_interface_contacts", 16)),
            minimum_entity_beads=int(data.get("minimum_entity_beads", 2)),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )
