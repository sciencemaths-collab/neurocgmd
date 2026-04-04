"""Protein-oriented chemistry semantics derived from residue names and bead labels."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, coerce_scalar
from topology.system_topology import SystemTopology


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _normalized_tokens(*values: str | None) -> frozenset[str]:
    tokens: set[str] = set()
    for value in values:
        if value is None:
            continue
        tokens.update(token for token in re.findall(r"[a-z0-9]+", value.lower()) if token)
    return frozenset(tokens)


class ChargeClass(StrEnum):
    """Coarse formal-charge categories for residue or bead chemistry."""

    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class PolarityClass(StrEnum):
    """Coarse polarity categories for residue or bead chemistry."""

    HYDROPHOBIC = "hydrophobic"
    AMPHIPATHIC = "amphipathic"
    POLAR = "polar"


def _charge_class_for_value(formal_charge: float) -> ChargeClass:
    if formal_charge >= 0.25:
        return ChargeClass.POSITIVE
    if formal_charge <= -0.25:
        return ChargeClass.NEGATIVE
    return ChargeClass.NEUTRAL


def _polarity_class_for_hydropathy(hydropathy: float) -> PolarityClass:
    if hydropathy >= 0.35:
        return PolarityClass.HYDROPHOBIC
    if hydropathy <= -0.25:
        return PolarityClass.POLAR
    return PolarityClass.AMPHIPATHIC


_RESIDUE_LIBRARY: dict[str, dict[str, object]] = {
    "ALA": {"formal_charge": 0.0, "hydropathy": 0.55, "flexibility": 0.35, "aromaticity": 0.0, "hydrogen_bond_capacity": 0.15, "hotspot_propensity": 0.20},
    "ARG": {"formal_charge": 1.0, "hydropathy": -0.45, "flexibility": 0.45, "aromaticity": 0.0, "hydrogen_bond_capacity": 0.90, "hotspot_propensity": 0.55},
    "ASN": {"formal_charge": 0.0, "hydropathy": -0.55, "flexibility": 0.45, "aromaticity": 0.0, "hydrogen_bond_capacity": 0.80, "hotspot_propensity": 0.35},
    "ASP": {"formal_charge": -1.0, "hydropathy": -0.70, "flexibility": 0.40, "aromaticity": 0.0, "hydrogen_bond_capacity": 0.75, "hotspot_propensity": 0.35},
    "CYS": {"formal_charge": 0.0, "hydropathy": 0.35, "flexibility": 0.30, "aromaticity": 0.0, "hydrogen_bond_capacity": 0.20, "hotspot_propensity": 0.25},
    "GLN": {"formal_charge": 0.0, "hydropathy": -0.50, "flexibility": 0.50, "aromaticity": 0.0, "hydrogen_bond_capacity": 0.82, "hotspot_propensity": 0.35},
    "GLU": {"formal_charge": -1.0, "hydropathy": -0.68, "flexibility": 0.48, "aromaticity": 0.0, "hydrogen_bond_capacity": 0.75, "hotspot_propensity": 0.35},
    "GLY": {"formal_charge": 0.0, "hydropathy": 0.00, "flexibility": 0.85, "aromaticity": 0.0, "hydrogen_bond_capacity": 0.20, "hotspot_propensity": 0.10},
    "HIS": {"formal_charge": 0.35, "hydropathy": -0.15, "flexibility": 0.35, "aromaticity": 0.45, "hydrogen_bond_capacity": 0.70, "hotspot_propensity": 0.65},
    "ILE": {"formal_charge": 0.0, "hydropathy": 0.88, "flexibility": 0.22, "aromaticity": 0.0, "hydrogen_bond_capacity": 0.05, "hotspot_propensity": 0.18},
    "LEU": {"formal_charge": 0.0, "hydropathy": 0.80, "flexibility": 0.24, "aromaticity": 0.0, "hydrogen_bond_capacity": 0.05, "hotspot_propensity": 0.20},
    "LYS": {"formal_charge": 1.0, "hydropathy": -0.55, "flexibility": 0.52, "aromaticity": 0.0, "hydrogen_bond_capacity": 0.85, "hotspot_propensity": 0.48},
    "MET": {"formal_charge": 0.0, "hydropathy": 0.62, "flexibility": 0.30, "aromaticity": 0.0, "hydrogen_bond_capacity": 0.12, "hotspot_propensity": 0.22},
    "PHE": {"formal_charge": 0.0, "hydropathy": 0.78, "flexibility": 0.25, "aromaticity": 0.90, "hydrogen_bond_capacity": 0.05, "hotspot_propensity": 0.72},
    "PRO": {"formal_charge": 0.0, "hydropathy": 0.20, "flexibility": 0.10, "aromaticity": 0.0, "hydrogen_bond_capacity": 0.05, "hotspot_propensity": 0.15},
    "SER": {"formal_charge": 0.0, "hydropathy": -0.38, "flexibility": 0.48, "aromaticity": 0.0, "hydrogen_bond_capacity": 0.75, "hotspot_propensity": 0.25},
    "THR": {"formal_charge": 0.0, "hydropathy": -0.20, "flexibility": 0.42, "aromaticity": 0.0, "hydrogen_bond_capacity": 0.72, "hotspot_propensity": 0.28},
    "TRP": {"formal_charge": 0.0, "hydropathy": 0.72, "flexibility": 0.20, "aromaticity": 1.00, "hydrogen_bond_capacity": 0.25, "hotspot_propensity": 0.85},
    "TYR": {"formal_charge": 0.0, "hydropathy": 0.45, "flexibility": 0.28, "aromaticity": 0.82, "hydrogen_bond_capacity": 0.55, "hotspot_propensity": 0.80},
    "VAL": {"formal_charge": 0.0, "hydropathy": 0.72, "flexibility": 0.24, "aromaticity": 0.0, "hydrogen_bond_capacity": 0.05, "hotspot_propensity": 0.18},
}

_BEAD_TYPE_PRIORS: dict[str, dict[str, object]] = {
    "core": {"formal_charge": 0.0, "hydropathy": 0.78, "flexibility": 0.16, "aromaticity": 0.10, "hydrogen_bond_capacity": 0.08, "hotspot_propensity": 0.18},
    "helix": {"formal_charge": 0.0, "hydropathy": 0.18, "flexibility": 0.24, "aromaticity": 0.08, "hydrogen_bond_capacity": 0.60, "hotspot_propensity": 0.35},
    "hotspot": {"formal_charge": 0.0, "hydropathy": 0.28, "flexibility": 0.20, "aromaticity": 0.45, "hydrogen_bond_capacity": 0.52, "hotspot_propensity": 0.90},
    "support": {"formal_charge": 0.0, "hydropathy": 0.05, "flexibility": 0.36, "aromaticity": 0.06, "hydrogen_bond_capacity": 0.58, "hotspot_propensity": 0.25},
    "shield": {"formal_charge": 0.0, "hydropathy": -0.35, "flexibility": 0.72, "aromaticity": 0.02, "hydrogen_bond_capacity": 0.82, "hotspot_propensity": 0.08},
}


@dataclass(frozen=True, slots=True)
class ResidueChemistryDescriptor(ValidatableComponent):
    """Chemistry descriptor normalized into bounded protein-interface features."""

    label: str
    descriptor_source: str
    charge_class: ChargeClass
    polarity_class: PolarityClass
    formal_charge: float
    hydropathy: float
    flexibility: float
    aromaticity: float
    hydrogen_bond_capacity: float
    hotspot_propensity: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "formal_charge", coerce_scalar(self.formal_charge, "formal_charge"))
        object.__setattr__(self, "hydropathy", coerce_scalar(self.hydropathy, "hydropathy"))
        object.__setattr__(self, "flexibility", coerce_scalar(self.flexibility, "flexibility"))
        object.__setattr__(self, "aromaticity", coerce_scalar(self.aromaticity, "aromaticity"))
        object.__setattr__(
            self,
            "hydrogen_bond_capacity",
            coerce_scalar(self.hydrogen_bond_capacity, "hydrogen_bond_capacity"),
        )
        object.__setattr__(self, "hotspot_propensity", coerce_scalar(self.hotspot_propensity, "hotspot_propensity"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.label.strip():
            issues.append("label must be a non-empty string.")
        if not self.descriptor_source.strip():
            issues.append("descriptor_source must be a non-empty string.")
        if not -1.0 <= self.formal_charge <= 1.0:
            issues.append("formal_charge must lie in the interval [-1, 1].")
        if not -1.0 <= self.hydropathy <= 1.0:
            issues.append("hydropathy must lie in the interval [-1, 1].")
        for field_name in ("flexibility", "aromaticity", "hydrogen_bond_capacity", "hotspot_propensity"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                issues.append(f"{field_name} must lie in the interval [0, 1].")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class BeadChemistryAssignment(ValidatableComponent):
    """Chemistry descriptor assigned to one coarse bead / particle index."""

    particle_index: int
    bead_label: str
    bead_type: str
    compartment_id: str | None
    descriptor: ResidueChemistryDescriptor
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.particle_index < 0:
            issues.append("particle_index must be non-negative.")
        if not self.bead_label.strip():
            issues.append("bead_label must be a non-empty string.")
        if not self.bead_type.strip():
            issues.append("bead_type must be a non-empty string.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ProteinChemistrySummary(ValidatableComponent):
    """Aggregate chemistry summary over one protein-style topology."""

    system_id: str
    assignments: tuple[BeadChemistryAssignment, ...]
    mean_abs_charge: float
    mean_hydropathy: float
    mean_flexibility: float
    aromatic_fraction: float
    hotspot_fraction: float
    polar_fraction: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", tuple(self.assignments))
        for field_name in (
            "mean_abs_charge",
            "mean_hydropathy",
            "mean_flexibility",
            "aromatic_fraction",
            "hotspot_fraction",
            "polar_fraction",
        ):
            object.__setattr__(self, field_name, coerce_scalar(getattr(self, field_name), field_name))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def assignment_for_particle(self, particle_index: int) -> BeadChemistryAssignment:
        for assignment in self.assignments:
            if assignment.particle_index == particle_index:
                return assignment
        raise KeyError(particle_index)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.system_id.strip():
            issues.append("system_id must be a non-empty string.")
        if not self.assignments:
            issues.append("assignments must contain at least one chemistry assignment.")
        if not -1.0 <= self.mean_hydropathy <= 1.0:
            issues.append("mean_hydropathy must lie in the interval [-1, 1].")
        for field_name in (
            "mean_abs_charge",
            "mean_flexibility",
            "aromatic_fraction",
            "hotspot_fraction",
            "polar_fraction",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                issues.append(f"{field_name} must lie in the interval [0, 1].")
        return tuple(issues)


@dataclass(slots=True)
class ProteinChemistryModel(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """[hybrid] Derive bounded chemistry semantics from residue or bead metadata."""

    name: str = "protein_chemistry_model"
    classification: str = "[hybrid]"

    def describe_role(self) -> str:
        return (
            "Converts residue names, bead labels, and topology semantics into bounded "
            "chemistry descriptors that higher layers can consume explicitly."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return ("topology/system_topology.py", "core/types.py")

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/chemistry_semantics_and_live_control.md",)

    def validate(self) -> tuple[str, ...]:
        return ()

    def descriptor_for_residue_name(
        self,
        residue_name: str,
        *,
        label: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ResidueChemistryDescriptor:
        """Return one bounded descriptor for a residue name or a generic fallback."""

        normalized_name = residue_name.strip().upper()
        if not normalized_name:
            raise ContractValidationError("residue_name must be a non-empty string.")

        if normalized_name in _RESIDUE_LIBRARY:
            return self._descriptor_from_profile(
                label=label or normalized_name,
                descriptor_source=f"residue:{normalized_name}",
                profile=_RESIDUE_LIBRARY[normalized_name],
                metadata=metadata,
            )

        fallback_profile = dict(_BEAD_TYPE_PRIORS["support"])
        return self._descriptor_from_profile(
            label=label or normalized_name,
            descriptor_source=f"fallback:{normalized_name}",
            profile=fallback_profile,
            metadata={
                **(metadata or {}),
                "fallback_residue_name": normalized_name,
            },
        )

    def aggregate_descriptor_for_residue_names(
        self,
        residue_names: tuple[str, ...],
        *,
        label: str,
        metadata: Mapping[str, object] | None = None,
    ) -> ResidueChemistryDescriptor:
        """Aggregate one bounded descriptor across a residue block."""

        if not residue_names:
            raise ContractValidationError("residue_names must contain at least one entry.")

        descriptors = tuple(
            self.descriptor_for_residue_name(
                residue_name,
                label=f"{label}:{index}",
                metadata={"aggregate_member": True, "member_index": index},
            )
            for index, residue_name in enumerate(residue_names)
        )
        count = len(descriptors)
        profile = {
            "formal_charge": sum(item.formal_charge for item in descriptors) / count,
            "hydropathy": sum(item.hydropathy for item in descriptors) / count,
            "flexibility": sum(item.flexibility for item in descriptors) / count,
            "aromaticity": sum(item.aromaticity for item in descriptors) / count,
            "hydrogen_bond_capacity": sum(item.hydrogen_bond_capacity for item in descriptors) / count,
            "hotspot_propensity": sum(item.hotspot_propensity for item in descriptors) / count,
        }
        return self._descriptor_from_profile(
            label=label,
            descriptor_source="residue_block_average",
            profile=profile,
            metadata={
                **(metadata or {}),
                "residue_names": tuple(residue_names),
                "block_size": len(residue_names),
            },
        )

    def descriptor_for_bead(self, topology: SystemTopology, particle_index: int) -> ResidueChemistryDescriptor:
        bead = topology.bead_for_particle(particle_index)
        bead_type = topology.bead_type_by_name(bead.bead_type)
        tokens = _normalized_tokens(bead.label, bead.bead_type, bead.residue_name, bead.chain_id, bead_type.role.value)

        if bead.residue_name is not None:
            residue_key = bead.residue_name.upper()
            if residue_key in _RESIDUE_LIBRARY:
                return self._descriptor_from_profile(
                    label=bead.label,
                    descriptor_source=f"residue:{residue_key}",
                    profile=_RESIDUE_LIBRARY[residue_key],
                    metadata={
                        "particle_index": particle_index,
                        "bead_type": bead.bead_type,
                        "source_kind": "residue_library",
                    },
                )

        base_profile = dict(_BEAD_TYPE_PRIORS.get(bead.bead_type.lower(), _BEAD_TYPE_PRIORS["support"]))
        base_profile["formal_charge"] = float(base_profile["formal_charge"])
        base_profile["hydropathy"] = float(base_profile["hydropathy"])
        base_profile["flexibility"] = float(base_profile["flexibility"])
        base_profile["aromaticity"] = float(base_profile["aromaticity"])
        base_profile["hydrogen_bond_capacity"] = float(base_profile["hydrogen_bond_capacity"])
        base_profile["hotspot_propensity"] = float(base_profile["hotspot_propensity"])

        if {"asp", "glu", "acidic", "acid"} & tokens:
            base_profile["formal_charge"] = -1.0
            base_profile["hydropathy"] -= 0.40
            base_profile["hydrogen_bond_capacity"] += 0.15
        if {"lys", "arg", "his", "basic", "cationic"} & tokens:
            base_profile["formal_charge"] = 1.0
            base_profile["hydropathy"] -= 0.35
            base_profile["hydrogen_bond_capacity"] += 0.18
            base_profile["hotspot_propensity"] += 0.10
        if {"glycan", "shield", "loop", "tail"} & tokens:
            base_profile["hydropathy"] -= 0.28
            base_profile["flexibility"] += 0.18
            base_profile["hydrogen_bond_capacity"] += 0.15
            base_profile["hotspot_propensity"] -= 0.08
        if {"core", "buried"} & tokens:
            base_profile["hydropathy"] += 0.18
            base_profile["flexibility"] -= 0.10
        if {"hotspot", "ridge", "anchor", "interface"} & tokens:
            base_profile["aromaticity"] += 0.15
            base_profile["hotspot_propensity"] += 0.18
            base_profile["hydrogen_bond_capacity"] += 0.05
        if {"aromatic", "phe", "tyr", "trp", "ring"} & tokens:
            base_profile["hydropathy"] += 0.15
            base_profile["aromaticity"] += 0.35
            base_profile["hotspot_propensity"] += 0.20
        if {"alpha", "helix", "beta", "sheet"} & tokens:
            base_profile["flexibility"] -= 0.08
        if {"support", "scaffold"} & tokens:
            base_profile["hydrogen_bond_capacity"] += 0.08
            base_profile["hotspot_propensity"] -= 0.02

        return self._descriptor_from_profile(
            label=bead.label,
            descriptor_source=f"inferred:{bead.bead_type}",
            profile=base_profile,
            metadata={
                "particle_index": particle_index,
                "bead_type": bead.bead_type,
                "source_kind": "inferred_from_topology",
                "tokens": tuple(sorted(tokens)),
            },
        )

    def assignments_for_topology(self, topology: SystemTopology) -> tuple[BeadChemistryAssignment, ...]:
        return tuple(
            BeadChemistryAssignment(
                particle_index=bead.particle_index,
                bead_label=bead.label,
                bead_type=bead.bead_type,
                compartment_id=str(bead.compartment_hint) if bead.compartment_hint is not None else None,
                descriptor=self.descriptor_for_bead(topology, bead.particle_index),
                metadata={
                    "bead_id": str(bead.bead_id),
                    "residue_name": bead.residue_name,
                },
            )
            for bead in topology.beads
        )

    def summarize_topology(self, topology: SystemTopology) -> ProteinChemistrySummary:
        assignments = self.assignments_for_topology(topology)
        assignment_count = len(assignments)
        return ProteinChemistrySummary(
            system_id=topology.system_id,
            assignments=assignments,
            mean_abs_charge=sum(abs(item.descriptor.formal_charge) for item in assignments) / assignment_count,
            mean_hydropathy=sum(item.descriptor.hydropathy for item in assignments) / assignment_count,
            mean_flexibility=sum(item.descriptor.flexibility for item in assignments) / assignment_count,
            aromatic_fraction=sum(item.descriptor.aromaticity >= 0.45 for item in assignments) / assignment_count,
            hotspot_fraction=sum(item.descriptor.hotspot_propensity >= 0.55 for item in assignments) / assignment_count,
            polar_fraction=(
                sum(item.descriptor.polarity_class != PolarityClass.HYDROPHOBIC for item in assignments)
                / assignment_count
            ),
            metadata={
                "assignment_count": assignment_count,
                "classification": self.classification,
            },
        )

    def _descriptor_from_profile(
        self,
        *,
        label: str,
        descriptor_source: str,
        profile: Mapping[str, object],
        metadata: Mapping[str, object] | None = None,
    ) -> ResidueChemistryDescriptor:
        formal_charge = _clamp(float(profile["formal_charge"]), -1.0, 1.0)
        hydropathy = _clamp(float(profile["hydropathy"]), -1.0, 1.0)
        flexibility = _clamp(float(profile["flexibility"]), 0.0, 1.0)
        aromaticity = _clamp(float(profile["aromaticity"]), 0.0, 1.0)
        hydrogen_bond_capacity = _clamp(float(profile["hydrogen_bond_capacity"]), 0.0, 1.0)
        hotspot_propensity = _clamp(float(profile["hotspot_propensity"]), 0.0, 1.0)
        return ResidueChemistryDescriptor(
            label=label,
            descriptor_source=descriptor_source,
            charge_class=_charge_class_for_value(formal_charge),
            polarity_class=_polarity_class_for_hydropathy(hydropathy),
            formal_charge=formal_charge,
            hydropathy=hydropathy,
            flexibility=flexibility,
            aromaticity=aromaticity,
            hydrogen_bond_capacity=hydrogen_bond_capacity,
            hotspot_propensity=hotspot_propensity,
            metadata=FrozenMetadata(metadata or {}),
        )
