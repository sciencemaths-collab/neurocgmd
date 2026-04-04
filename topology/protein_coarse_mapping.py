"""Arbitrary-protein import and coarse mapping from local PDB structures."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.reference_cases import (
    InterfaceContactTarget,
    ReferenceStructureTarget,
    StructureLandmarkTarget,
)
from benchmarks.reference_cases.structure_targets import centroid, distance, load_local_pdb_loader
from chemistry import ProteinChemistryModel
from compartments import CompartmentDomain, CompartmentRegistry
from config.protein_mapping import ProteinEntityGroup, ProteinMappingConfig
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import ParticleState
from core.types import BeadId, FrozenMetadata, Vector3
from topology.beads import Bead, BeadRole, BeadType
from topology.bonds import Bond, BondKind
from topology.protein_import_models import (
    ImportedBeadBlock,
    ImportedProteinSystem,
    ImportedResidueRecord,
)
from topology.system_topology import SystemTopology

if TYPE_CHECKING:
    from project_pdb_loader import PDBAtomRecord, PDBStructure


def _scale_position(position: Vector3, factor: float) -> Vector3:
    return (
        position[0] * factor,
        position[1] * factor,
        position[2] * factor,
    )


@dataclass(slots=True)
class ProteinCoarseMapper(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """[hybrid] Convert arbitrary local proteins into the coarse repository substrate."""

    mapping_config: ProteinMappingConfig = field(default_factory=ProteinMappingConfig)
    chemistry_model: ProteinChemistryModel = field(default_factory=ProteinChemistryModel)
    name: str = "protein_coarse_mapper"
    classification: str = "[hybrid]"

    def describe_role(self) -> str:
        return (
            "Loads local protein structures, groups residues into coarse bead blocks, "
            "creates particle/topology/compartment objects, and derives a matching "
            "reference scaffold for honest downstream validation."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "io/pdb_loader.py",
            "topology/protein_import_models.py",
            "chemistry/residue_semantics.py",
            "benchmarks/reference_cases/structure_targets.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/arbitrary_protein_input_pipeline.md",)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        issues.extend(self.mapping_config.validate())
        issues.extend(self.chemistry_model.validate())
        return tuple(issues)

    def import_from_pdb(
        self,
        *,
        pdb_path: str | Path,
        entity_groups: tuple[ProteinEntityGroup, ...],
        structure_id: str | None = None,
    ) -> ImportedProteinSystem:
        """Load a local PDB file and map it into a coarse imported protein system."""

        issues = list(self.validate())
        if not entity_groups:
            issues.append("entity_groups must contain at least one entity.")
        for group in entity_groups:
            issues.extend(group.validate())
        if issues:
            raise ContractValidationError("; ".join(issues))

        resolved_path = Path(pdb_path).expanduser().resolve()
        if not resolved_path.exists():
            raise ContractValidationError(f"Protein import PDB not found at {resolved_path}.")

        pdb_loader = load_local_pdb_loader()
        structure = pdb_loader.load_pdb_file(resolved_path, structure_id=structure_id or resolved_path.stem)
        return self.import_from_structure(
            structure=structure,
            source_path=str(resolved_path),
            entity_groups=entity_groups,
        )

    def import_from_structure(
        self,
        *,
        structure: "PDBStructure",
        source_path: str,
        entity_groups: tuple[ProteinEntityGroup, ...],
    ) -> ImportedProteinSystem:
        """Convert one parsed structure into the coarse imported-protein substrate."""

        chain_to_entity = self._build_chain_to_entity_map(entity_groups)
        residues = self._extract_residues(structure, chain_to_entity)
        bead_blocks = self._build_bead_blocks(residues, chain_to_entity)
        bead_types = self._bead_types_for_blocks(bead_blocks)
        topology = self._build_topology(structure.structure_id, bead_blocks, bead_types)
        particles = self._build_particles(bead_blocks)
        compartments = self._build_compartments(particles.particle_count, entity_groups, bead_blocks)
        reference_target = self._build_reference_target(
            structure_id=structure.structure_id,
            source_path=source_path,
            entity_groups=entity_groups,
            bead_blocks=bead_blocks,
        )
        return ImportedProteinSystem(
            structure_id=structure.structure_id,
            source_path=source_path,
            entity_groups=entity_groups,
            mapping_config=self.mapping_config,
            residues=residues,
            bead_blocks=bead_blocks,
            topology=topology,
            particles=particles,
            compartments=compartments,
            reference_target=reference_target,
            metadata={
                "chain_count": len({residue.chain_id for residue in residues}),
                "residue_count": len(residues),
                "bead_count": len(bead_blocks),
                "entity_count": len(entity_groups),
            },
        )

    def _build_chain_to_entity_map(
        self,
        entity_groups: tuple[ProteinEntityGroup, ...],
    ) -> dict[str, ProteinEntityGroup]:
        chain_to_entity: dict[str, ProteinEntityGroup] = {}
        for group in entity_groups:
            for chain_id in group.chain_ids:
                if chain_id in chain_to_entity:
                    raise ContractValidationError(
                        f"Chain {chain_id!r} is assigned to multiple protein entities."
                    )
                chain_to_entity[chain_id] = group
        return chain_to_entity

    def _extract_residues(
        self,
        structure: "PDBStructure",
        chain_to_entity: dict[str, ProteinEntityGroup],
    ) -> tuple[ImportedResidueRecord, ...]:
        grouped_atoms: dict[tuple[str, int], list["PDBAtomRecord"]] = defaultdict(list)
        residue_names: dict[tuple[str, int], str] = {}
        for atom in structure.atoms:
            if atom.chain_id not in chain_to_entity or atom.record_type != "ATOM":
                continue
            key = (atom.chain_id, atom.residue_sequence)
            grouped_atoms[key].append(atom)
            residue_names.setdefault(key, atom.residue_name)

        residues = [
            ImportedResidueRecord(
                chain_id=chain_id,
                residue_sequence=residue_sequence,
                residue_name=residue_names[(chain_id, residue_sequence)],
                atom_count=len(atoms),
                centroid=_scale_position(
                    centroid(tuple(atom.coordinates for atom in atoms)),
                    self.mapping_config.angstrom_to_nm_scale,
                ),
                metadata={
                    "entity_id": chain_to_entity[chain_id].entity_id,
                },
            )
            for chain_id, residue_sequence in sorted(grouped_atoms)
            for atoms in (grouped_atoms[(chain_id, residue_sequence)],)
        ]
        if not residues:
            raise ContractValidationError("No matching ATOM records were found for the requested entity groups.")
        return tuple(residues)

    def _build_bead_blocks(
        self,
        residues: tuple[ImportedResidueRecord, ...],
        chain_to_entity: dict[str, ProteinEntityGroup],
    ) -> tuple[ImportedBeadBlock, ...]:
        residues_by_chain: dict[str, list[ImportedResidueRecord]] = defaultdict(list)
        for residue in residues:
            residues_by_chain[residue.chain_id].append(residue)

        blocks: list[ImportedBeadBlock] = []
        particle_index = 0
        for chain_id in sorted(residues_by_chain):
            chain_residues = sorted(residues_by_chain[chain_id], key=lambda residue: residue.residue_sequence)
            entity_id = chain_to_entity[chain_id].entity_id
            for block_offset in range(0, len(chain_residues), self.mapping_config.residues_per_bead):
                chunk = tuple(chain_residues[block_offset : block_offset + self.mapping_config.residues_per_bead])
                residue_names = tuple(residue.residue_name for residue in chunk)
                descriptor = self.chemistry_model.aggregate_descriptor_for_residue_names(
                    residue_names,
                    label=f"{entity_id}_{chain_id}_{chunk[0].residue_sequence}_{chunk[-1].residue_sequence}",
                    metadata={"entity_id": entity_id, "chain_id": chain_id},
                )
                bead_type = self._classify_bead_type(descriptor)
                label = (
                    f"{entity_id}_{bead_type}_{chunk[0].residue_sequence}"
                    f"_{chunk[-1].residue_sequence}"
                )
                blocks.append(
                    ImportedBeadBlock(
                        block_id=f"{entity_id}_{chain_id}_{particle_index}",
                        entity_id=entity_id,
                        chain_id=chain_id,
                        residue_ids=tuple(residue.residue_sequence for residue in chunk),
                        residue_names=residue_names,
                        bead_type=bead_type,
                        label=label,
                        centroid=centroid(tuple(residue.centroid for residue in chunk)),
                        mass=len(chunk) * self.mapping_config.mean_residue_mass,
                        metadata={
                            "descriptor_source": descriptor.descriptor_source,
                            "formal_charge": descriptor.formal_charge,
                            "hydropathy": descriptor.hydropathy,
                            "flexibility": descriptor.flexibility,
                            "aromaticity": descriptor.aromaticity,
                            "hydrogen_bond_capacity": descriptor.hydrogen_bond_capacity,
                            "hotspot_propensity": descriptor.hotspot_propensity,
                        },
                    )
                )
                particle_index += 1

        entity_counts = Counter(block.entity_id for block in blocks)
        undersized = sorted(
            entity_id
            for entity_id, bead_count in entity_counts.items()
            if bead_count < self.mapping_config.minimum_entity_beads
        )
        if undersized:
            raise ContractValidationError(
                "Imported entities must have at least "
                f"{self.mapping_config.minimum_entity_beads} coarse beads; undersized: "
                + ", ".join(undersized)
            )
        return tuple(blocks)

    def _classify_bead_type(self, descriptor) -> str:
        if descriptor.formal_charge >= 0.45:
            return "basic"
        if descriptor.formal_charge <= -0.45:
            return "acidic"
        if descriptor.flexibility >= 0.62:
            return "loop"
        if descriptor.aromaticity >= 0.45 or descriptor.hotspot_propensity >= 0.62:
            return "hotspot"
        if descriptor.hydropathy >= 0.28 and descriptor.flexibility <= 0.42:
            return "core"
        if descriptor.hydropathy <= -0.20 and descriptor.hydrogen_bond_capacity >= 0.68:
            return "shield"
        return "support"

    def _bead_types_for_blocks(
        self,
        bead_blocks: tuple[ImportedBeadBlock, ...],
    ) -> tuple[BeadType, ...]:
        bead_type_names = sorted({block.bead_type for block in bead_blocks})
        role_map = {
            "core": BeadRole.STRUCTURAL,
            "support": BeadRole.STRUCTURAL,
            "hotspot": BeadRole.ANCHOR,
            "basic": BeadRole.FUNCTIONAL,
            "acidic": BeadRole.FUNCTIONAL,
            "loop": BeadRole.LINKER,
            "shield": BeadRole.LINKER,
        }
        descriptions = {
            "core": "Imported buried or compact block.",
            "support": "Imported support or scaffold block.",
            "hotspot": "Imported aromatic or hotspot-enriched block.",
            "basic": "Imported positively charged block.",
            "acidic": "Imported negatively charged block.",
            "loop": "Imported flexible loop or linker block.",
            "shield": "Imported solvent-facing or shielded block.",
        }
        return tuple(
            BeadType(
                name=bead_type_name,
                role=role_map[bead_type_name],
                description=descriptions[bead_type_name],
            )
            for bead_type_name in bead_type_names
        )

    def _build_topology(
        self,
        structure_id: str,
        bead_blocks: tuple[ImportedBeadBlock, ...],
        bead_types: tuple[BeadType, ...],
    ) -> SystemTopology:
        beads = tuple(
            Bead(
                bead_id=BeadId(block.block_id),
                particle_index=index,
                bead_type=block.bead_type,
                label=block.label,
                residue_name=block.residue_names[0],
                chain_id=block.chain_id,
                compartment_hint=block.entity_id,
                metadata=block.metadata.with_updates(
                    {
                        "entity_id": block.entity_id,
                        "residue_ids": block.residue_ids,
                        "residue_names": block.residue_names,
                    }
                ),
            )
            for index, block in enumerate(bead_blocks)
        )

        bonds: list[Bond] = []
        blocks_by_chain: dict[str, list[tuple[int, ImportedBeadBlock]]] = defaultdict(list)
        for index, block in enumerate(bead_blocks):
            blocks_by_chain[block.chain_id].append((index, block))
        for chain_id, chain_blocks in blocks_by_chain.items():
            for (left_index, left_block), (right_index, right_block) in zip(chain_blocks, chain_blocks[1:], strict=False):
                equilibrium_distance = max(distance(left_block.centroid, right_block.centroid), 0.05)
                bonds.append(
                    Bond(
                        particle_index_a=left_index,
                        particle_index_b=right_index,
                        kind=BondKind.STRUCTURAL,
                        bond_id=f"{structure_id}_{chain_id}_{left_index}_{right_index}",
                        equilibrium_distance=equilibrium_distance,
                        stiffness=self.mapping_config.structural_bond_stiffness,
                        metadata={
                            "chain_id": chain_id,
                            "entity_id": left_block.entity_id,
                            "source": "protein_coarse_mapper",
                        },
                    )
                )

        return SystemTopology(
            system_id=structure_id,
            bead_types=bead_types,
            beads=beads,
            bonds=tuple(bonds),
            metadata={
                "imported_structure": structure_id,
                "mapping_config": self.mapping_config.to_dict(),
            },
        )

    def _build_particles(
        self,
        bead_blocks: tuple[ImportedBeadBlock, ...],
    ) -> ParticleState:
        return ParticleState(
            positions=tuple(block.centroid for block in bead_blocks),
            masses=tuple(block.mass for block in bead_blocks),
            labels=tuple(block.label for block in bead_blocks),
        )

    def _build_compartments(
        self,
        particle_count: int,
        entity_groups: tuple[ProteinEntityGroup, ...],
        bead_blocks: tuple[ImportedBeadBlock, ...],
    ) -> CompartmentRegistry:
        entity_members = {
            group.entity_id: tuple(
                index
                for index, block in enumerate(bead_blocks)
                if block.entity_id == group.entity_id
            )
            for group in entity_groups
        }
        return CompartmentRegistry(
            particle_count=particle_count,
            domains=tuple(
                CompartmentDomain.from_members(
                    group.entity_id,
                    group.description or group.entity_id,
                    entity_members[group.entity_id],
                )
                for group in entity_groups
            ),
        )

    def _build_reference_target(
        self,
        *,
        structure_id: str,
        source_path: str,
        entity_groups: tuple[ProteinEntityGroup, ...],
        bead_blocks: tuple[ImportedBeadBlock, ...],
    ) -> ReferenceStructureTarget:
        landmarks = tuple(
            StructureLandmarkTarget(
                label=block.label,
                chain_id=block.chain_id,
                residue_ids=block.residue_ids,
                residue_names=block.residue_names,
                description=(
                    f"Imported coarse residue block for entity {block.entity_id} "
                    f"covering residues {block.residue_ids[0]}-{block.residue_ids[-1]}."
                ),
                target_position=block.centroid,
                metadata={
                    "entity_id": block.entity_id,
                    "bead_type": block.bead_type,
                    "source_file": source_path,
                },
            )
            for block in bead_blocks
        )
        landmark_map = {landmark.label: landmark for landmark in landmarks}

        contact_candidates: list[tuple[float, str, str, str, float]] = []
        for entity_index, left_entity in enumerate(entity_groups):
            left_labels = tuple(block.label for block in bead_blocks if block.entity_id == left_entity.entity_id)
            for right_entity in entity_groups[entity_index + 1 :]:
                right_labels = tuple(block.label for block in bead_blocks if block.entity_id == right_entity.entity_id)
                for left_label in left_labels:
                    for right_label in right_labels:
                        target_distance = distance(
                            landmark_map[left_label].target_position,
                            landmark_map[right_label].target_position,
                        )
                        if target_distance <= self.mapping_config.interface_contact_cutoff_nm:
                            contact_candidates.append(
                                (
                                    target_distance,
                                    left_label,
                                    right_label,
                                    f"Imported {left_entity.entity_id}-{right_entity.entity_id} reference contact.",
                                    target_distance + self.mapping_config.interface_contact_tolerance_nm,
                                )
                            )
        if not contact_candidates and len(entity_groups) >= 2:
            left_entity, right_entity = entity_groups[0], entity_groups[1]
            left_blocks = [block for block in bead_blocks if block.entity_id == left_entity.entity_id]
            right_blocks = [block for block in bead_blocks if block.entity_id == right_entity.entity_id]
            if left_blocks and right_blocks:
                left_block, right_block = min(
                    (
                        (left_block, right_block)
                        for left_block in left_blocks
                        for right_block in right_blocks
                    ),
                    key=lambda pair: distance(pair[0].centroid, pair[1].centroid),
                )
                target_distance = distance(left_block.centroid, right_block.centroid)
                contact_candidates.append(
                    (
                        target_distance,
                        left_block.label,
                        right_block.label,
                        f"Imported dominant {left_entity.entity_id}-{right_entity.entity_id} reference contact.",
                        target_distance + self.mapping_config.interface_contact_tolerance_nm,
                    )
                )

        contact_candidates.sort(key=lambda item: item[0])
        interface_contacts = tuple(
            InterfaceContactTarget(
                source_label=source_label,
                target_label=target_label,
                max_distance=max_distance,
                description=description,
                metadata={
                    "target_distance": target_distance,
                    "tolerance": self.mapping_config.interface_contact_tolerance_nm,
                },
            )
            for target_distance, source_label, target_label, description, max_distance in contact_candidates[
                : self.mapping_config.max_interface_contacts
            ]
        )
        dominant_pair = (
            (interface_contacts[0].source_label, interface_contacts[0].target_label)
            if interface_contacts
            else ()
        )
        return ReferenceStructureTarget(
            name=f"{structure_id}_imported_reference_target",
            classification="[hybrid]",
            title=f"{structure_id} Imported Coarse Reference Scaffold",
            summary=(
                "A local-PDB-derived coarse scaffold built automatically from imported residue "
                "blocks so arbitrary proteins can participate in honest structural and "
                "shadow-fidelity validation without bespoke scenario code."
            ),
            source_pdb_id=structure_id,
            landmarks=landmarks,
            interface_contacts=interface_contacts,
            metadata={
                "source_file": source_path,
                "representation": "imported_residue_block_centroids",
                "dominant_interface_pair": dominant_pair,
                "entity_ids": tuple(group.entity_id for group in entity_groups),
            },
        )
