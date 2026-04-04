"""Generic PDB-derived structure-target contracts and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, Vector3, coerce_vector3


def load_local_pdb_loader():
    """Load the local PDB parser without turning `io/` into a package."""

    module_path = Path(__file__).resolve().parents[2] / "io" / "pdb_loader.py"
    spec = importlib.util.spec_from_file_location("project_pdb_loader", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load io/pdb_loader.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("project_pdb_loader", module)
    spec.loader.exec_module(module)
    return module


def centroid(points: tuple[Vector3, ...]) -> Vector3:
    """Return the centroid of one or more coordinates."""

    count = len(points)
    if count == 0:
        raise ContractValidationError("At least one coordinate is required to compute a centroid.")
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def distance(left: Vector3, right: Vector3) -> float:
    """Return Euclidean distance between two 3D points."""

    return (
        (left[0] - right[0]) ** 2
        + (left[1] - right[1]) ** 2
        + (left[2] - right[2]) ** 2
    ) ** 0.5


def selection_centroid(
    structure: object,
    *,
    chain_id: str,
    residue_ids: tuple[int, ...],
) -> Vector3:
    """Return the centroid of all ATOM records in one chain/residue selection."""

    coordinates = tuple(
        atom.coordinates
        for atom in structure.atoms
        if atom.chain_id == chain_id and atom.residue_sequence in residue_ids and atom.record_type == "ATOM"
    )
    if not coordinates:
        raise ContractValidationError(
            f"No ATOM records found in {structure.structure_id} for chain {chain_id} and residues {residue_ids!r}."
        )
    return centroid(coordinates)


@dataclass(frozen=True, slots=True)
class StructureLandmarkTarget(ValidatableComponent):
    """One PDB-derived reference landmark tied to a real structural hotspot family."""

    label: str
    chain_id: str
    residue_ids: tuple[int, ...]
    residue_names: tuple[str, ...]
    description: str
    target_position: Vector3
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "residue_ids", tuple(int(value) for value in self.residue_ids))
        object.__setattr__(self, "residue_names", tuple(str(value) for value in self.residue_names))
        object.__setattr__(self, "target_position", coerce_vector3(self.target_position, "target_position"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.label.strip():
            issues.append("label must be a non-empty string.")
        if not self.chain_id.strip():
            issues.append("chain_id must be a non-empty string.")
        if not self.residue_ids:
            issues.append("residue_ids must contain at least one residue index.")
        if not self.residue_names:
            issues.append("residue_names must contain at least one residue name.")
        if not self.description.strip():
            issues.append("description must be a non-empty string.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class InterfaceContactTarget(ValidatableComponent):
    """One expected cross-interface hotspot-family contact."""

    source_label: str
    target_label: str
    max_distance: float
    description: str
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.source_label.strip():
            issues.append("source_label must be a non-empty string.")
        if not self.target_label.strip():
            issues.append("target_label must be a non-empty string.")
        if self.max_distance <= 0.0:
            issues.append("max_distance must be positive.")
        if not self.description.strip():
            issues.append("description must be a non-empty string.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ReferenceStructureTarget(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """PDB-derived structural target used for honest proxy-vs-reference comparison."""

    name: str
    classification: str
    title: str
    summary: str
    source_pdb_id: str
    landmarks: tuple[StructureLandmarkTarget, ...]
    interface_contacts: tuple[InterfaceContactTarget, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "landmarks", tuple(self.landmarks))
        object.__setattr__(self, "interface_contacts", tuple(self.interface_contacts))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Stores a local PDB-derived landmark scaffold tied to a real structure so "
            "proxy simulations can report structural progress without pretending to "
            "be atomistic agreement."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return ("io/pdb_loader.py",)

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/use_cases/spike_ace2_reference_case.md",
            "docs/use_cases/barnase_barstar_reference_case.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name.strip():
            issues.append("name must be a non-empty string.")
        if not self.classification.strip():
            issues.append("classification must be a non-empty string.")
        if not self.title.strip():
            issues.append("title must be a non-empty string.")
        if not self.summary.strip():
            issues.append("summary must be a non-empty string.")
        if not self.source_pdb_id.strip():
            issues.append("source_pdb_id must be a non-empty string.")
        if not self.landmarks:
            issues.append("landmarks must contain at least one target landmark.")
        if len(self.landmark_names()) != len(set(self.landmark_names())):
            issues.append("landmark labels must be unique.")
        return tuple(issues)

    def landmark_names(self) -> tuple[str, ...]:
        return tuple(landmark.label for landmark in self.landmarks)

    def landmark_for(self, label: str) -> StructureLandmarkTarget:
        for landmark in self.landmarks:
            if landmark.label == label:
                return landmark
        raise KeyError(label)
