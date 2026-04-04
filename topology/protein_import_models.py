"""Data contracts for arbitrary protein import and coarse mapping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from benchmarks.reference_cases import ReferenceStructureTarget
from compartments import CompartmentRegistry
from config.protein_mapping import ProteinEntityGroup, ProteinMappingConfig
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import ParticleState
from core.types import FrozenMetadata, Vector3, coerce_scalar, coerce_vector3
from topology.system_topology import SystemTopology


@dataclass(frozen=True, slots=True)
class ImportedResidueRecord(ValidatableComponent):
    """One residue recovered from a local PDB file before coarse grouping."""

    chain_id: str
    residue_sequence: int
    residue_name: str
    atom_count: int
    centroid: Vector3
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "centroid", coerce_vector3(self.centroid, "centroid"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.chain_id.strip():
            issues.append("chain_id must be a non-empty string.")
        if self.atom_count <= 0:
            issues.append("atom_count must be strictly positive.")
        if not self.residue_name.strip():
            issues.append("residue_name must be a non-empty string.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ImportedBeadBlock(ValidatableComponent):
    """One coarse bead block derived from a contiguous residue window."""

    block_id: str
    entity_id: str
    chain_id: str
    residue_ids: tuple[int, ...]
    residue_names: tuple[str, ...]
    bead_type: str
    label: str
    centroid: Vector3
    mass: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "residue_ids", tuple(int(value) for value in self.residue_ids))
        object.__setattr__(self, "residue_names", tuple(str(value) for value in self.residue_names))
        object.__setattr__(self, "centroid", coerce_vector3(self.centroid, "centroid"))
        object.__setattr__(self, "mass", coerce_scalar(self.mass, "mass"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.block_id.strip():
            issues.append("block_id must be a non-empty string.")
        if not self.entity_id.strip():
            issues.append("entity_id must be a non-empty string.")
        if not self.chain_id.strip():
            issues.append("chain_id must be a non-empty string.")
        if not self.residue_ids:
            issues.append("residue_ids must contain at least one residue index.")
        if not self.residue_names:
            issues.append("residue_names must contain at least one residue name.")
        if not self.bead_type.strip():
            issues.append("bead_type must be a non-empty string.")
        if not self.label.strip():
            issues.append("label must be a non-empty string.")
        if self.mass <= 0.0:
            issues.append("mass must be strictly positive.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ImportedProteinSystem(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Materialized arbitrary-protein import result aligned to the global architecture."""

    structure_id: str
    source_path: str
    entity_groups: tuple[ProteinEntityGroup, ...]
    mapping_config: ProteinMappingConfig
    residues: tuple[ImportedResidueRecord, ...]
    bead_blocks: tuple[ImportedBeadBlock, ...]
    topology: SystemTopology
    particles: ParticleState
    compartments: CompartmentRegistry
    reference_target: ReferenceStructureTarget
    classification: str = "[hybrid]"
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_groups", tuple(self.entity_groups))
        object.__setattr__(self, "residues", tuple(self.residues))
        object.__setattr__(self, "bead_blocks", tuple(self.bead_blocks))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Represents an arbitrary local protein structure after it has been "
            "converted into the repository's coarse, compartmented, benchmarkable substrate."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "config/protein_mapping.py",
            "topology/protein_coarse_mapping.py",
            "benchmarks/reference_cases/structure_targets.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/arbitrary_protein_input_pipeline.md",)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.structure_id.strip():
            issues.append("structure_id must be a non-empty string.")
        if not self.source_path.strip():
            issues.append("source_path must be a non-empty string.")
        if not self.entity_groups:
            issues.append("entity_groups must contain at least one imported entity.")
        if not self.residues:
            issues.append("residues must contain at least one imported residue.")
        if not self.bead_blocks:
            issues.append("bead_blocks must contain at least one imported bead block.")
        issues.extend(self.mapping_config.validate())
        if self.topology.validate_against_particle_state(self.particles):
            issues.append("topology must align with particles.")
        if self.compartments.particle_count != self.particles.particle_count:
            issues.append("compartments must align with particles.")
        if len(self.reference_target.landmarks) != self.particles.particle_count:
            issues.append("reference_target landmarks must align with imported particle count.")
        return tuple(issues)

    def entity_ids(self) -> tuple[str, ...]:
        return tuple(group.entity_id for group in self.entity_groups)

    def bead_labels(self) -> tuple[str, ...]:
        return tuple(block.label for block in self.bead_blocks)

    def bead_indices_for_entity(self, entity_id: str) -> tuple[int, ...]:
        return tuple(
            index
            for index, block in enumerate(self.bead_blocks)
            if block.entity_id == entity_id
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "structure_id": self.structure_id,
            "source_path": self.source_path,
            "entity_groups": [group.to_dict() for group in self.entity_groups],
            "mapping_config": self.mapping_config.to_dict(),
            "residues": [
                {
                    "chain_id": residue.chain_id,
                    "residue_sequence": residue.residue_sequence,
                    "residue_name": residue.residue_name,
                    "atom_count": residue.atom_count,
                    "centroid": list(residue.centroid),
                    "metadata": residue.metadata.to_dict(),
                }
                for residue in self.residues
            ],
            "bead_blocks": [
                {
                    "block_id": block.block_id,
                    "entity_id": block.entity_id,
                    "chain_id": block.chain_id,
                    "residue_ids": list(block.residue_ids),
                    "residue_names": list(block.residue_names),
                    "bead_type": block.bead_type,
                    "label": block.label,
                    "centroid": list(block.centroid),
                    "mass": block.mass,
                    "metadata": block.metadata.to_dict(),
                }
                for block in self.bead_blocks
            ],
            "topology": self.topology.to_dict(),
            "particles": {
                "positions": [list(position) for position in self.particles.positions],
                "masses": list(self.particles.masses),
                "velocities": [list(velocity) for velocity in self.particles.velocities],
                "forces": [list(force) for force in self.particles.forces],
                "labels": list(self.particles.labels or ()),
            },
            "reference_target": {
                "name": self.reference_target.name,
                "classification": self.reference_target.classification,
                "title": self.reference_target.title,
                "summary": self.reference_target.summary,
                "source_pdb_id": self.reference_target.source_pdb_id,
                "landmarks": [
                    {
                        "label": landmark.label,
                        "chain_id": landmark.chain_id,
                        "residue_ids": list(landmark.residue_ids),
                        "residue_names": list(landmark.residue_names),
                        "description": landmark.description,
                        "target_position": list(landmark.target_position),
                        "metadata": landmark.metadata.to_dict(),
                    }
                    for landmark in self.reference_target.landmarks
                ],
                "interface_contacts": [
                    {
                        "source_label": contact.source_label,
                        "target_label": contact.target_label,
                        "max_distance": contact.max_distance,
                        "description": contact.description,
                        "metadata": contact.metadata.to_dict(),
                    }
                    for contact in self.reference_target.interface_contacts
                ],
                "metadata": self.reference_target.metadata.to_dict(),
            },
            "metadata": self.metadata.to_dict(),
        }
