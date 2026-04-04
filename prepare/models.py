"""Data contracts for manifest-driven system preparation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from config.protein_mapping import ProteinEntityGroup, ProteinMappingConfig
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import ParticleState, SimulationCell, ThermodynamicState, UnitSystem
from core.types import FrozenMetadata, Vector3, coerce_vector3
from sampling.scenarios import ImportedProteinScenarioSpec


@dataclass(frozen=True, slots=True)
class PreparationEntitySummary(ValidatableComponent):
    """Per-entity import summary recorded in the prepared bundle."""

    entity_id: str
    chain_ids: tuple[str, ...]
    residue_count: int
    bead_count: int
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "chain_ids", tuple(self.chain_ids))
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
        if self.residue_count <= 0:
            issues.append("residue_count must be strictly positive.")
        if self.bead_count <= 0:
            issues.append("bead_count must be strictly positive.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "chain_ids": list(self.chain_ids),
            "residue_count": self.residue_count,
            "bead_count": self.bead_count,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PreparationEntitySummary":
        return cls(
            entity_id=str(data["entity_id"]),
            chain_ids=tuple(data.get("chain_ids", ())),
            residue_count=int(data["residue_count"]),
            bead_count=int(data["bead_count"]),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ImportSummary(ValidatableComponent):
    """Compact summary of the imported protein structure."""

    structure_id: str
    source_path: str
    residue_count: int
    bead_count: int
    particle_count: int
    entities: tuple[PreparationEntitySummary, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entities", tuple(self.entities))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.structure_id.strip():
            issues.append("structure_id must be a non-empty string.")
        if not self.source_path.strip():
            issues.append("source_path must be a non-empty string.")
        if self.residue_count <= 0:
            issues.append("residue_count must be strictly positive.")
        if self.bead_count <= 0:
            issues.append("bead_count must be strictly positive.")
        if self.particle_count <= 0:
            issues.append("particle_count must be strictly positive.")
        if not self.entities:
            issues.append("entities must contain at least one preparation entity summary.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "structure_id": self.structure_id,
            "source_path": self.source_path,
            "residue_count": self.residue_count,
            "bead_count": self.bead_count,
            "particle_count": self.particle_count,
            "entities": [entity.to_dict() for entity in self.entities],
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ImportSummary":
        return cls(
            structure_id=str(data["structure_id"]),
            source_path=str(data["source_path"]),
            residue_count=int(data["residue_count"]),
            bead_count=int(data["bead_count"]),
            particle_count=int(data["particle_count"]),
            entities=tuple(
                PreparationEntitySummary.from_dict(item)
                for item in data.get("entities", ())
            ),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ProtonationSitePlan(ValidatableComponent):
    """One residue-level protonation estimate."""

    chain_id: str
    residue_sequence: int
    residue_name: str
    protonation_state: str
    formal_charge: float
    present_hydrogen_count: int
    estimated_total_hydrogen_count: int
    estimated_hydrogens_to_add: int
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.chain_id.strip():
            issues.append("chain_id must be a non-empty string.")
        if not self.residue_name.strip():
            issues.append("residue_name must be a non-empty string.")
        if not self.protonation_state.strip():
            issues.append("protonation_state must be a non-empty string.")
        if self.present_hydrogen_count < 0:
            issues.append("present_hydrogen_count must be non-negative.")
        if self.estimated_total_hydrogen_count < 0:
            issues.append("estimated_total_hydrogen_count must be non-negative.")
        if self.estimated_hydrogens_to_add < 0:
            issues.append("estimated_hydrogens_to_add must be non-negative.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "chain_id": self.chain_id,
            "residue_sequence": self.residue_sequence,
            "residue_name": self.residue_name,
            "protonation_state": self.protonation_state,
            "formal_charge": self.formal_charge,
            "present_hydrogen_count": self.present_hydrogen_count,
            "estimated_total_hydrogen_count": self.estimated_total_hydrogen_count,
            "estimated_hydrogens_to_add": self.estimated_hydrogens_to_add,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProtonationSitePlan":
        return cls(
            chain_id=str(data["chain_id"]),
            residue_sequence=int(data["residue_sequence"]),
            residue_name=str(data["residue_name"]),
            protonation_state=str(data["protonation_state"]),
            formal_charge=float(data["formal_charge"]),
            present_hydrogen_count=int(data["present_hydrogen_count"]),
            estimated_total_hydrogen_count=int(data["estimated_total_hydrogen_count"]),
            estimated_hydrogens_to_add=int(data["estimated_hydrogens_to_add"]),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ProtonationPlan(ValidatableComponent):
    """Residue-semantics-based protonation and hydrogen-addition plan."""

    method: str
    ph: float
    add_hydrogens: bool
    estimated_net_charge: float
    total_present_hydrogen_atoms: int
    total_estimated_hydrogens_to_add: int
    sites: tuple[ProtonationSitePlan, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sites", tuple(self.sites))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.method.strip():
            issues.append("method must be a non-empty string.")
        if self.ph <= 0.0:
            issues.append("ph must be strictly positive.")
        if self.total_present_hydrogen_atoms < 0:
            issues.append("total_present_hydrogen_atoms must be non-negative.")
        if self.total_estimated_hydrogens_to_add < 0:
            issues.append("total_estimated_hydrogens_to_add must be non-negative.")
        if not self.sites:
            issues.append("sites must contain at least one protonation site.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "ph": self.ph,
            "add_hydrogens": self.add_hydrogens,
            "estimated_net_charge": self.estimated_net_charge,
            "total_present_hydrogen_atoms": self.total_present_hydrogen_atoms,
            "total_estimated_hydrogens_to_add": self.total_estimated_hydrogens_to_add,
            "sites": [site.to_dict() for site in self.sites],
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProtonationPlan":
        return cls(
            method=str(data["method"]),
            ph=float(data["ph"]),
            add_hydrogens=bool(data["add_hydrogens"]),
            estimated_net_charge=float(data["estimated_net_charge"]),
            total_present_hydrogen_atoms=int(data["total_present_hydrogen_atoms"]),
            total_estimated_hydrogens_to_add=int(data["total_estimated_hydrogens_to_add"]),
            sites=tuple(ProtonationSitePlan.from_dict(item) for item in data.get("sites", ())),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class SolvationPlan(ValidatableComponent):
    """Preparation-time solvent and simulation-cell plan."""

    mode: str
    water_model: str
    box_type: str
    padding_nm: float
    cell: SimulationCell
    box_volume_nm3: float
    estimated_water_molecules: int
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.mode.strip():
            issues.append("mode must be a non-empty string.")
        if not self.water_model.strip():
            issues.append("water_model must be a non-empty string.")
        if not self.box_type.strip():
            issues.append("box_type must be a non-empty string.")
        if self.padding_nm < 0.0:
            issues.append("padding_nm must be non-negative.")
        if self.box_volume_nm3 <= 0.0:
            issues.append("box_volume_nm3 must be strictly positive.")
        if self.estimated_water_molecules < 0:
            issues.append("estimated_water_molecules must be non-negative.")
        issues.extend(self.cell.validate())
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "water_model": self.water_model,
            "box_type": self.box_type,
            "padding_nm": self.padding_nm,
            "cell": self.cell.to_dict(),
            "box_volume_nm3": self.box_volume_nm3,
            "estimated_water_molecules": self.estimated_water_molecules,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SolvationPlan":
        return cls(
            mode=str(data["mode"]),
            water_model=str(data["water_model"]),
            box_type=str(data["box_type"]),
            padding_nm=float(data["padding_nm"]),
            cell=SimulationCell.from_dict(data["cell"]),
            box_volume_nm3=float(data["box_volume_nm3"]),
            estimated_water_molecules=int(data["estimated_water_molecules"]),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class IonPlacementPlan(ValidatableComponent):
    """Estimated ion counts for neutralization and bulk salt."""

    neutralize: bool
    salt: str
    ionic_strength_molar: float
    cation_name: str
    anion_name: str
    estimated_cation_count: int
    estimated_anion_count: int
    estimated_total_ions: int
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.salt.strip():
            issues.append("salt must be a non-empty string.")
        if not self.cation_name.strip():
            issues.append("cation_name must be a non-empty string.")
        if not self.anion_name.strip():
            issues.append("anion_name must be a non-empty string.")
        if self.ionic_strength_molar < 0.0:
            issues.append("ionic_strength_molar must be non-negative.")
        for field_name in (
            "estimated_cation_count",
            "estimated_anion_count",
            "estimated_total_ions",
        ):
            if getattr(self, field_name) < 0:
                issues.append(f"{field_name} must be non-negative.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "neutralize": self.neutralize,
            "salt": self.salt,
            "ionic_strength_molar": self.ionic_strength_molar,
            "cation_name": self.cation_name,
            "anion_name": self.anion_name,
            "estimated_cation_count": self.estimated_cation_count,
            "estimated_anion_count": self.estimated_anion_count,
            "estimated_total_ions": self.estimated_total_ions,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "IonPlacementPlan":
        return cls(
            neutralize=bool(data["neutralize"]),
            salt=str(data["salt"]),
            ionic_strength_molar=float(data["ionic_strength_molar"]),
            cation_name=str(data["cation_name"]),
            anion_name=str(data["anion_name"]),
            estimated_cation_count=int(data["estimated_cation_count"]),
            estimated_anion_count=int(data["estimated_anion_count"]),
            estimated_total_ions=int(data["estimated_total_ions"]),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ExplicitAtomPlacement(ValidatableComponent):
    """One explicit prepared atom coordinate in nanometer units."""

    atom_name: str
    element: str
    residue_name: str
    chain_id: str
    residue_sequence: int
    coordinates: Vector3
    record_type: str = "ATOM"
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinates", coerce_vector3(self.coordinates, "coordinates"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.record_type not in {"ATOM", "HETATM"}:
            issues.append("record_type must be ATOM or HETATM.")
        for field_name in ("atom_name", "element", "residue_name", "chain_id"):
            if not getattr(self, field_name).strip():
                issues.append(f"{field_name} must be a non-empty string.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "atom_name": self.atom_name,
            "element": self.element,
            "residue_name": self.residue_name,
            "chain_id": self.chain_id,
            "residue_sequence": self.residue_sequence,
            "coordinates": list(self.coordinates),
            "record_type": self.record_type,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ExplicitAtomPlacement":
        return cls(
            atom_name=str(data["atom_name"]),
            element=str(data["element"]),
            residue_name=str(data["residue_name"]),
            chain_id=str(data["chain_id"]),
            residue_sequence=int(data["residue_sequence"]),
            coordinates=data["coordinates"],
            record_type=str(data.get("record_type", "ATOM")),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ExplicitMoleculePlacement(ValidatableComponent):
    """One explicit prepared molecule or ion with its atom placements."""

    molecule_id: str
    residue_name: str
    chain_id: str
    residue_sequence: int
    atoms: tuple[ExplicitAtomPlacement, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", tuple(self.atoms))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for field_name in ("molecule_id", "residue_name", "chain_id"):
            if not getattr(self, field_name).strip():
                issues.append(f"{field_name} must be a non-empty string.")
        if not self.atoms:
            issues.append("atoms must contain at least one explicit atom placement.")
        for atom in self.atoms:
            issues.extend(atom.validate())
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "molecule_id": self.molecule_id,
            "residue_name": self.residue_name,
            "chain_id": self.chain_id,
            "residue_sequence": self.residue_sequence,
            "atoms": [atom.to_dict() for atom in self.atoms],
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ExplicitMoleculePlacement":
        return cls(
            molecule_id=str(data["molecule_id"]),
            residue_name=str(data["residue_name"]),
            chain_id=str(data["chain_id"]),
            residue_sequence=int(data["residue_sequence"]),
            atoms=tuple(ExplicitAtomPlacement.from_dict(item) for item in data.get("atoms", ())),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ExplicitPreparationCoordinates(ValidatableComponent):
    """Explicit prepared coordinates for protein atoms, hydrogens, waters, and ions."""

    coordinate_unit: str
    protein_atoms: tuple[ExplicitAtomPlacement, ...] = ()
    built_hydrogens: tuple[ExplicitAtomPlacement, ...] = ()
    water_molecules: tuple[ExplicitMoleculePlacement, ...] = ()
    ions: tuple[ExplicitMoleculePlacement, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "protein_atoms", tuple(self.protein_atoms))
        object.__setattr__(self, "built_hydrogens", tuple(self.built_hydrogens))
        object.__setattr__(self, "water_molecules", tuple(self.water_molecules))
        object.__setattr__(self, "ions", tuple(self.ions))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    @classmethod
    def empty(cls) -> "ExplicitPreparationCoordinates":
        return cls(coordinate_unit="nm")

    def total_atom_count(self) -> int:
        return (
            len(self.protein_atoms)
            + len(self.built_hydrogens)
            + sum(len(molecule.atoms) for molecule in self.water_molecules)
            + sum(len(molecule.atoms) for molecule in self.ions)
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.coordinate_unit.strip():
            issues.append("coordinate_unit must be a non-empty string.")
        for atom in self.protein_atoms:
            issues.extend(atom.validate())
        for atom in self.built_hydrogens:
            issues.extend(atom.validate())
        for molecule in self.water_molecules:
            issues.extend(molecule.validate())
        for molecule in self.ions:
            issues.extend(molecule.validate())
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "coordinate_unit": self.coordinate_unit,
            "protein_atoms": [atom.to_dict() for atom in self.protein_atoms],
            "built_hydrogens": [atom.to_dict() for atom in self.built_hydrogens],
            "water_molecules": [molecule.to_dict() for molecule in self.water_molecules],
            "ions": [molecule.to_dict() for molecule in self.ions],
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ExplicitPreparationCoordinates":
        return cls(
            coordinate_unit=str(data.get("coordinate_unit", "nm")),
            protein_atoms=tuple(ExplicitAtomPlacement.from_dict(item) for item in data.get("protein_atoms", ())),
            built_hydrogens=tuple(
                ExplicitAtomPlacement.from_dict(item) for item in data.get("built_hydrogens", ())
            ),
            water_molecules=tuple(
                ExplicitMoleculePlacement.from_dict(item) for item in data.get("water_molecules", ())
            ),
            ions=tuple(ExplicitMoleculePlacement.from_dict(item) for item in data.get("ions", ())),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class PreparedRuntimeSeed(ValidatableComponent):
    """Initial runtime state seed created by the preparation pipeline."""

    units: UnitSystem
    particles: ParticleState
    thermodynamics: ThermodynamicState
    cell: SimulationCell | None
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues = list(self.units.validate())
        issues.extend(self.particles.validate())
        issues.extend(self.thermodynamics.validate())
        if self.cell is not None:
            issues.extend(self.cell.validate())
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "units": self.units.to_dict(),
            "particles": self.particles.to_dict(),
            "thermodynamics": self.thermodynamics.to_dict(),
            "cell": self.cell.to_dict() if self.cell is not None else None,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PreparedRuntimeSeed":
        cell = data.get("cell")
        return cls(
            units=UnitSystem.from_dict(data["units"]),
            particles=ParticleState.from_dict(data["particles"]),
            thermodynamics=ThermodynamicState.from_dict(data["thermodynamics"]),
            cell=SimulationCell.from_dict(cell) if cell is not None else None,
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class PreparedSystemBundle(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """[hybrid] Preparation artifact handed from `prepare` into `run` and `analyze`."""

    system_name: str
    manifest_path: str
    structure_path: str
    entity_groups: tuple[ProteinEntityGroup, ...]
    mapping_config: ProteinMappingConfig
    scenario_spec: ImportedProteinScenarioSpec
    import_summary: ImportSummary
    protonation_plan: ProtonationPlan
    solvation_plan: SolvationPlan
    ion_plan: IonPlacementPlan
    runtime_seed: PreparedRuntimeSeed
    explicit_coordinates: ExplicitPreparationCoordinates = field(default_factory=ExplicitPreparationCoordinates.empty)
    classification: str = "[hybrid]"
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_groups", tuple(self.entity_groups))
        if not isinstance(self.mapping_config, ProteinMappingConfig):
            object.__setattr__(self, "mapping_config", ProteinMappingConfig.from_dict(self.mapping_config))
        if not isinstance(self.explicit_coordinates, ExplicitPreparationCoordinates):
            object.__setattr__(
                self,
                "explicit_coordinates",
                ExplicitPreparationCoordinates.from_dict(self.explicit_coordinates),
            )
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Carries the user-facing preparation output that connects manifest inputs, "
            "imported protein geometry, solvent/ion planning, and the production runtime seed."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "config/run_manifest.py",
            "prepare/pipeline.py",
            "sampling/scenarios/imported_protein.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/manifest_driven_md_workflow.md",)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.system_name.strip():
            issues.append("system_name must be a non-empty string.")
        if not self.manifest_path.strip():
            issues.append("manifest_path must be a non-empty string.")
        if not self.structure_path.strip():
            issues.append("structure_path must be a non-empty string.")
        if not self.entity_groups:
            issues.append("entity_groups must contain at least one entity group.")
        for group in self.entity_groups:
            issues.extend(group.validate())
        issues.extend(self.mapping_config.validate())
        issues.extend(self.scenario_spec.validate())
        issues.extend(self.import_summary.validate())
        issues.extend(self.protonation_plan.validate())
        issues.extend(self.solvation_plan.validate())
        issues.extend(self.ion_plan.validate())
        issues.extend(self.explicit_coordinates.validate())
        issues.extend(self.runtime_seed.validate())
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "system_name": self.system_name,
            "manifest_path": self.manifest_path,
            "structure_path": self.structure_path,
            "entity_groups": [group.to_dict() for group in self.entity_groups],
            "mapping_config": self.mapping_config.to_dict(),
            "scenario_spec": self.scenario_spec.to_dict(),
            "import_summary": self.import_summary.to_dict(),
            "protonation_plan": self.protonation_plan.to_dict(),
            "solvation_plan": self.solvation_plan.to_dict(),
            "ion_plan": self.ion_plan.to_dict(),
            "explicit_coordinates": self.explicit_coordinates.to_dict(),
            "runtime_seed": self.runtime_seed.to_dict(),
            "classification": self.classification,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PreparedSystemBundle":
        return cls(
            system_name=str(data["system_name"]),
            manifest_path=str(data["manifest_path"]),
            structure_path=str(data["structure_path"]),
            entity_groups=tuple(
                ProteinEntityGroup.from_dict(item)
                for item in data.get("entity_groups", ())
            ),
            mapping_config=ProteinMappingConfig.from_dict(data.get("mapping_config", {})),
            scenario_spec=ImportedProteinScenarioSpec.from_dict(data["scenario_spec"]),
            import_summary=ImportSummary.from_dict(data["import_summary"]),
            protonation_plan=ProtonationPlan.from_dict(data["protonation_plan"]),
            solvation_plan=SolvationPlan.from_dict(data["solvation_plan"]),
            ion_plan=IonPlacementPlan.from_dict(data["ion_plan"]),
            explicit_coordinates=ExplicitPreparationCoordinates.from_dict(
                data.get("explicit_coordinates", {})
            ),
            runtime_seed=PreparedRuntimeSeed.from_dict(data["runtime_seed"]),
            classification=str(data.get("classification", "[hybrid]")),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )
