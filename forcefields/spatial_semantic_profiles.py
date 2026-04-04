"""Spatially aware protein interaction priors for intelligent shadow corrections."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, coerce_scalar
from forcefields.protein_shadow_profiles import (
    ProteinBeadFamily,
    ProteinShadowProfileFactory,
)
from forcefields.trusted_sources import TrustedScienceSource
from topology.system_topology import SystemTopology


def _normalized_type_pair(type_a: str, type_b: str) -> tuple[str, str]:
    return tuple(sorted((type_a, type_b)))


def _normalized_family_pair(
    left: ProteinBeadFamily,
    right: ProteinBeadFamily,
) -> tuple[ProteinBeadFamily, ProteinBeadFamily]:
    return tuple(sorted((left, right), key=lambda family: family.value))


@dataclass(frozen=True, slots=True)
class SpatialSemanticProfile(ValidatableComponent):
    """Distance- and geometry-aware interaction prior for one bead-type pair."""

    bead_type_a: str
    bead_type_b: str
    preferred_distance: float
    distance_tolerance: float
    minimum_distance: float
    cutoff: float
    attraction_strength: float
    repulsion_strength: float
    directional_strength: float
    chemistry_strength: float
    source_label: str
    fidelity_label: str = "protein_spatial_semantic"
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        for field_name in (
            "preferred_distance",
            "distance_tolerance",
            "minimum_distance",
            "cutoff",
            "attraction_strength",
            "repulsion_strength",
            "directional_strength",
            "chemistry_strength",
        ):
            object.__setattr__(self, field_name, coerce_scalar(getattr(self, field_name), field_name))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def pair_key(self) -> tuple[str, str]:
        return _normalized_type_pair(self.bead_type_a, self.bead_type_b)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.bead_type_a.strip() or not self.bead_type_b.strip():
            issues.append("bead_type_a and bead_type_b must be non-empty strings.")
        if self.preferred_distance <= 0.0:
            issues.append("preferred_distance must be strictly positive.")
        if self.distance_tolerance <= 0.0:
            issues.append("distance_tolerance must be strictly positive.")
        if self.minimum_distance <= 0.0:
            issues.append("minimum_distance must be strictly positive.")
        if self.minimum_distance >= self.cutoff:
            issues.append("minimum_distance must remain below cutoff.")
        if self.preferred_distance >= self.cutoff:
            issues.append("preferred_distance must remain below cutoff.")
        for field_name in ("attraction_strength", "repulsion_strength", "directional_strength", "chemistry_strength"):
            if getattr(self, field_name) < 0.0:
                issues.append(f"{field_name} must be non-negative.")
        if not self.source_label.strip():
            issues.append("source_label must be a non-empty string.")
        if not self.fidelity_label.strip():
            issues.append("fidelity_label must be a non-empty string.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class SpatialSemanticParameterSet(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Trusted spatial-semantic parameter bundle for intelligent local corrections."""

    name: str
    summary: str
    sources: tuple[TrustedScienceSource, ...]
    profiles: tuple[SpatialSemanticProfile, ...]
    classification: str = "[adapted]"
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "profiles", tuple(self.profiles))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Stores repository-owned spatial, distance-window, and chemistry-aware interaction "
            "priors for intelligent local correction layers inspired by proven molecular-simulation norms."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "forcefields/protein_shadow_profiles.py",
            "qcloud/spatial_semantic_field.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/spatial_semantic_forcefield.md",)

    def source_labels(self) -> tuple[str, ...]:
        return tuple(source.label for source in self.sources)

    def profile_for_bead_types(self, bead_type_a: str, bead_type_b: str) -> SpatialSemanticProfile:
        key = _normalized_type_pair(bead_type_a, bead_type_b)
        for profile in self.profiles:
            if profile.pair_key() == key:
                return profile
        raise KeyError(key)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name.strip():
            issues.append("name must be a non-empty string.")
        if not self.summary.strip():
            issues.append("summary must be a non-empty string.")
        if not self.sources:
            issues.append("sources must contain at least one trusted source.")
        if not self.profiles:
            issues.append("profiles must contain at least one spatial-semantic profile.")

        source_labels = self.source_labels()
        if len(source_labels) != len(set(source_labels)):
            issues.append("source labels must be unique.")
        profile_keys = tuple(profile.pair_key() for profile in self.profiles)
        if len(profile_keys) != len(set(profile_keys)):
            issues.append("spatial-semantic profiles must be unique per bead-type pair.")
        missing_sources = sorted(
            {profile.source_label for profile in self.profiles if profile.source_label not in source_labels}
        )
        if missing_sources:
            issues.append(
                "Each spatial-semantic profile must reference a declared source label; missing: "
                + ", ".join(missing_sources)
            )
        return tuple(issues)


_SPATIAL_PRIORS: dict[
    tuple[ProteinBeadFamily, ProteinBeadFamily],
    tuple[float, float, float, float, float, float, float, float],
] = {
    (ProteinBeadFamily.HYDROPHOBIC_CORE, ProteinBeadFamily.HYDROPHOBIC_CORE): (1.08, 0.26, 0.78, 2.70, 0.55, 0.65, 0.20, 0.18),
    (ProteinBeadFamily.HYDROPHOBIC_CORE, ProteinBeadFamily.POLAR_SURFACE): (1.18, 0.28, 0.82, 2.85, 0.42, 0.60, 0.18, 0.25),
    (ProteinBeadFamily.HYDROPHOBIC_CORE, ProteinBeadFamily.BASIC_PATCH): (1.16, 0.27, 0.80, 2.85, 0.48, 0.62, 0.22, 0.35),
    (ProteinBeadFamily.HYDROPHOBIC_CORE, ProteinBeadFamily.ACIDIC_PATCH): (1.16, 0.27, 0.80, 2.85, 0.48, 0.62, 0.22, 0.35),
    (ProteinBeadFamily.HYDROPHOBIC_CORE, ProteinBeadFamily.AROMATIC_HOTSPOT): (1.02, 0.22, 0.74, 2.75, 0.82, 0.78, 0.32, 0.42),
    (ProteinBeadFamily.HYDROPHOBIC_CORE, ProteinBeadFamily.FLEXIBLE_LINKER): (1.24, 0.32, 0.88, 2.85, 0.20, 0.52, 0.10, 0.12),
    (ProteinBeadFamily.HYDROPHOBIC_CORE, ProteinBeadFamily.SHIELDED_SURFACE): (1.26, 0.34, 0.90, 2.90, 0.14, 0.55, 0.08, 0.10),
    (ProteinBeadFamily.POLAR_SURFACE, ProteinBeadFamily.POLAR_SURFACE): (1.12, 0.24, 0.76, 2.80, 0.60, 0.60, 0.25, 0.52),
    (ProteinBeadFamily.POLAR_SURFACE, ProteinBeadFamily.BASIC_PATCH): (1.02, 0.22, 0.74, 2.85, 0.92, 0.72, 0.30, 0.78),
    (ProteinBeadFamily.POLAR_SURFACE, ProteinBeadFamily.ACIDIC_PATCH): (1.02, 0.22, 0.74, 2.85, 0.92, 0.72, 0.30, 0.78),
    (ProteinBeadFamily.POLAR_SURFACE, ProteinBeadFamily.AROMATIC_HOTSPOT): (0.98, 0.20, 0.72, 2.80, 1.02, 0.76, 0.38, 0.70),
    (ProteinBeadFamily.POLAR_SURFACE, ProteinBeadFamily.FLEXIBLE_LINKER): (1.18, 0.28, 0.84, 2.85, 0.28, 0.55, 0.12, 0.22),
    (ProteinBeadFamily.POLAR_SURFACE, ProteinBeadFamily.SHIELDED_SURFACE): (1.24, 0.32, 0.88, 2.90, 0.16, 0.58, 0.10, 0.16),
    (ProteinBeadFamily.BASIC_PATCH, ProteinBeadFamily.BASIC_PATCH): (1.14, 0.26, 0.78, 2.75, 0.18, 1.05, 0.28, 0.35),
    (ProteinBeadFamily.BASIC_PATCH, ProteinBeadFamily.ACIDIC_PATCH): (0.96, 0.18, 0.68, 2.85, 1.35, 0.86, 0.42, 0.90),
    (ProteinBeadFamily.BASIC_PATCH, ProteinBeadFamily.AROMATIC_HOTSPOT): (0.98, 0.18, 0.70, 2.80, 1.10, 0.82, 0.48, 0.82),
    (ProteinBeadFamily.BASIC_PATCH, ProteinBeadFamily.FLEXIBLE_LINKER): (1.16, 0.28, 0.82, 2.85, 0.34, 0.58, 0.12, 0.24),
    (ProteinBeadFamily.BASIC_PATCH, ProteinBeadFamily.SHIELDED_SURFACE): (1.20, 0.30, 0.86, 2.90, 0.12, 0.66, 0.10, 0.18),
    (ProteinBeadFamily.ACIDIC_PATCH, ProteinBeadFamily.ACIDIC_PATCH): (1.14, 0.26, 0.78, 2.75, 0.18, 1.05, 0.28, 0.35),
    (ProteinBeadFamily.ACIDIC_PATCH, ProteinBeadFamily.AROMATIC_HOTSPOT): (0.98, 0.18, 0.70, 2.80, 1.10, 0.82, 0.48, 0.82),
    (ProteinBeadFamily.ACIDIC_PATCH, ProteinBeadFamily.FLEXIBLE_LINKER): (1.16, 0.28, 0.82, 2.85, 0.34, 0.58, 0.12, 0.24),
    (ProteinBeadFamily.ACIDIC_PATCH, ProteinBeadFamily.SHIELDED_SURFACE): (1.20, 0.30, 0.86, 2.90, 0.12, 0.66, 0.10, 0.18),
    (ProteinBeadFamily.AROMATIC_HOTSPOT, ProteinBeadFamily.AROMATIC_HOTSPOT): (0.94, 0.16, 0.66, 2.78, 1.18, 0.88, 0.55, 0.86),
    (ProteinBeadFamily.AROMATIC_HOTSPOT, ProteinBeadFamily.FLEXIBLE_LINKER): (1.10, 0.24, 0.78, 2.82, 0.38, 0.58, 0.18, 0.24),
    (ProteinBeadFamily.AROMATIC_HOTSPOT, ProteinBeadFamily.SHIELDED_SURFACE): (1.18, 0.30, 0.84, 2.88, 0.16, 0.60, 0.12, 0.18),
    (ProteinBeadFamily.FLEXIBLE_LINKER, ProteinBeadFamily.FLEXIBLE_LINKER): (1.26, 0.34, 0.92, 2.88, 0.10, 0.52, 0.06, 0.10),
    (ProteinBeadFamily.FLEXIBLE_LINKER, ProteinBeadFamily.SHIELDED_SURFACE): (1.28, 0.36, 0.94, 2.92, 0.08, 0.54, 0.06, 0.10),
    (ProteinBeadFamily.SHIELDED_SURFACE, ProteinBeadFamily.SHIELDED_SURFACE): (1.30, 0.38, 0.96, 2.95, 0.06, 0.56, 0.04, 0.08),
}


@dataclass(slots=True)
class ProteinSpatialProfileFactory(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Build spatial-semantic interaction priors from shared protein families."""

    family_factory: ProteinShadowProfileFactory = field(default_factory=ProteinShadowProfileFactory)
    source_label: str = "protein_spatial_semantic_priors"
    name: str = "protein_spatial_profile_factory"
    classification: str = "[hybrid]"

    def describe_role(self) -> str:
        return (
            "Builds spatially aware, distance-window, and chemistry-weighted interaction priors "
            "from shared protein bead families so the correction layer stays reusable across proteins."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "forcefields/protein_shadow_profiles.py",
            "chemistry/residue_semantics.py",
            "qcloud/spatial_semantic_field.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/spatial_semantic_forcefield.md",)

    def validate(self) -> tuple[str, ...]:
        issues = list(self.family_factory.validate())
        if not self.source_label.strip():
            issues.append("source_label must be a non-empty string.")
        return tuple(issues)

    def pair_prior_for_families(
        self,
        left: ProteinBeadFamily,
        right: ProteinBeadFamily,
    ) -> tuple[float, float, float, float, float, float, float, float]:
        key = _normalized_family_pair(left, right)
        if key in _SPATIAL_PRIORS:
            return _SPATIAL_PRIORS[key]
        reverse_key = (key[1], key[0])
        if reverse_key in _SPATIAL_PRIORS:
            return _SPATIAL_PRIORS[reverse_key]
        raise KeyError(key)

    def build_parameter_set(
        self,
        topology: SystemTopology,
        *,
        scenario_label: str,
        reference_label: str | None = None,
        preferred_distance_scale: float = 1.0,
        distance_tolerance_scale: float = 1.0,
        attraction_scale: float = 1.0,
        repulsion_scale: float = 1.0,
        directional_scale: float = 1.0,
        chemistry_scale: float = 1.0,
        metadata: FrozenMetadata | dict[str, object] | None = None,
    ) -> SpatialSemanticParameterSet:
        for label, value in (
            ("preferred_distance_scale", preferred_distance_scale),
            ("distance_tolerance_scale", distance_tolerance_scale),
            ("attraction_scale", attraction_scale),
            ("repulsion_scale", repulsion_scale),
            ("directional_scale", directional_scale),
            ("chemistry_scale", chemistry_scale),
        ):
            if coerce_scalar(value, label) <= 0.0:
                raise ContractValidationError(f"{label} must be strictly positive.")
        assignments = {
            assignment.bead_type: ProteinBeadFamily(assignment.family)
            for assignment in self.family_factory.assignments_for_topology(topology)
        }
        sources = [
            TrustedScienceSource(
                label=self.source_label,
                source_type="adapted_spatial_parameterization",
                citation=(
                    "Repository-native spatial-semantic protein priors inspired by proven "
                    "distance-aware contact potentials, electrostatic steering ideas, and "
                    "geometry-sensitive coarse-grained interaction design."
                ),
                adaptation_notes=(
                    "Implemented as repository-owned spatial interaction priors that stay explicit, "
                    "bounded, and architecture-compatible with the shadow coarse-grained system."
                ),
            ),
        ]
        if reference_label is not None:
            sources.append(
                TrustedScienceSource(
                    label=f"{scenario_label}_spatial_reference_anchor",
                    source_type="structure_reference",
                    citation=f"Scenario reference anchor: {reference_label}.",
                    adaptation_notes="Used as provenance for the spatial-semantic parameter bundle only.",
                )
            )

        profiles: list[SpatialSemanticProfile] = []
        for bead_type_a, bead_type_b in combinations_with_replacement(topology.bead_type_names, 2):
            family_a = assignments[bead_type_a]
            family_b = assignments[bead_type_b]
            (
                preferred_distance,
                distance_tolerance,
                minimum_distance,
                cutoff,
                attraction_strength,
                repulsion_strength,
                directional_strength,
                chemistry_strength,
            ) = self.pair_prior_for_families(family_a, family_b)
            profiles.append(
                SpatialSemanticProfile(
                    bead_type_a=bead_type_a,
                    bead_type_b=bead_type_b,
                    preferred_distance=preferred_distance * preferred_distance_scale,
                    distance_tolerance=distance_tolerance * distance_tolerance_scale,
                    minimum_distance=minimum_distance,
                    cutoff=cutoff,
                    attraction_strength=attraction_strength * attraction_scale,
                    repulsion_strength=repulsion_strength * repulsion_scale,
                    directional_strength=directional_strength * directional_scale,
                    chemistry_strength=chemistry_strength * chemistry_scale,
                    source_label=self.source_label,
                    fidelity_label="protein_spatial_semantic",
                    metadata={
                        "family_a": family_a.value,
                        "family_b": family_b.value,
                        "scenario": scenario_label,
                    },
                )
            )

        combined_metadata = {}
        if metadata is not None:
            combined_metadata.update(metadata if isinstance(metadata, dict) else metadata.to_dict())
        combined_metadata.update(
            {
                "scenario": scenario_label,
                "reference_label": reference_label,
                "assignment_count": len(assignments),
                "preferred_distance_scale": preferred_distance_scale,
                "distance_tolerance_scale": distance_tolerance_scale,
                "attraction_scale": attraction_scale,
                "repulsion_scale": repulsion_scale,
                "directional_scale": directional_scale,
                "chemistry_scale": chemistry_scale,
            }
        )
        return SpatialSemanticParameterSet(
            name=f"{scenario_label}_protein_spatial_parameters",
            summary=(
                "Protein-general spatial interaction priors with explicit preferred distances, "
                "directional emphasis, and chemistry sensitivity for intelligent local corrections."
            ),
            sources=tuple(sources),
            profiles=tuple(profiles),
            metadata=combined_metadata,
        )


__all__ = [
    "ProteinSpatialProfileFactory",
    "SpatialSemanticParameterSet",
    "SpatialSemanticProfile",
]
