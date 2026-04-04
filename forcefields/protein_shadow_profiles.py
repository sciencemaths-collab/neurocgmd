"""Protein-general trusted shadow profiles for fast large-step coarse simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations_with_replacement

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata
from forcefields.trusted_sources import TrustedNonbondedProfile, TrustedParameterSet, TrustedScienceSource
from topology import BeadRole, BeadType, SystemTopology


class ProteinBeadFamily(StrEnum):
    """Generalized chemistry families used by the protein shadow layer."""

    HYDROPHOBIC_CORE = "hydrophobic_core"
    POLAR_SURFACE = "polar_surface"
    BASIC_PATCH = "basic_patch"
    ACIDIC_PATCH = "acidic_patch"
    AROMATIC_HOTSPOT = "aromatic_hotspot"
    FLEXIBLE_LINKER = "flexible_linker"
    SHIELDED_SURFACE = "shielded_surface"


@dataclass(frozen=True, slots=True)
class ProteinBeadFamilyAssignment(ValidatableComponent):
    """Family assignment for one concrete topology bead type."""

    bead_type: str
    role: str
    family: str
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.bead_type.strip():
            issues.append("bead_type must be a non-empty string.")
        if not self.role.strip():
            issues.append("role must be a non-empty string.")
        if not self.family.strip():
            issues.append("family must be a non-empty string.")
        return tuple(issues)


_PAIR_PRIORS: dict[tuple[ProteinBeadFamily, ProteinBeadFamily], tuple[float, float, float, float]] = {
    (ProteinBeadFamily.HYDROPHOBIC_CORE, ProteinBeadFamily.HYDROPHOBIC_CORE): (1.00, 0.18, 2.9, -0.02),
    (ProteinBeadFamily.HYDROPHOBIC_CORE, ProteinBeadFamily.POLAR_SURFACE): (1.00, 0.20, 2.9, -0.06),
    (ProteinBeadFamily.HYDROPHOBIC_CORE, ProteinBeadFamily.BASIC_PATCH): (0.98, 0.26, 2.9, -0.18),
    (ProteinBeadFamily.HYDROPHOBIC_CORE, ProteinBeadFamily.ACIDIC_PATCH): (0.98, 0.26, 2.9, -0.18),
    (ProteinBeadFamily.HYDROPHOBIC_CORE, ProteinBeadFamily.AROMATIC_HOTSPOT): (0.95, 0.42, 3.0, -0.12),
    (ProteinBeadFamily.HYDROPHOBIC_CORE, ProteinBeadFamily.FLEXIBLE_LINKER): (1.02, 0.12, 2.8, 0.00),
    (ProteinBeadFamily.HYDROPHOBIC_CORE, ProteinBeadFamily.SHIELDED_SURFACE): (1.04, 0.05, 2.8, 0.02),
    (ProteinBeadFamily.POLAR_SURFACE, ProteinBeadFamily.POLAR_SURFACE): (0.98, 0.32, 2.9, -0.15),
    (ProteinBeadFamily.POLAR_SURFACE, ProteinBeadFamily.BASIC_PATCH): (0.95, 0.72, 3.0, -0.45),
    (ProteinBeadFamily.POLAR_SURFACE, ProteinBeadFamily.ACIDIC_PATCH): (0.95, 0.72, 3.0, -0.45),
    (ProteinBeadFamily.POLAR_SURFACE, ProteinBeadFamily.AROMATIC_HOTSPOT): (0.94, 0.88, 3.0, -0.28),
    (ProteinBeadFamily.POLAR_SURFACE, ProteinBeadFamily.FLEXIBLE_LINKER): (1.00, 0.18, 2.8, -0.05),
    (ProteinBeadFamily.POLAR_SURFACE, ProteinBeadFamily.SHIELDED_SURFACE): (1.03, 0.06, 2.8, 0.02),
    (ProteinBeadFamily.BASIC_PATCH, ProteinBeadFamily.BASIC_PATCH): (0.95, 0.18, 2.8, 0.55),
    (ProteinBeadFamily.BASIC_PATCH, ProteinBeadFamily.ACIDIC_PATCH): (0.92, 1.45, 3.0, -1.05),
    (ProteinBeadFamily.BASIC_PATCH, ProteinBeadFamily.AROMATIC_HOTSPOT): (0.92, 1.05, 3.0, -0.52),
    (ProteinBeadFamily.BASIC_PATCH, ProteinBeadFamily.FLEXIBLE_LINKER): (0.98, 0.24, 2.8, -0.06),
    (ProteinBeadFamily.BASIC_PATCH, ProteinBeadFamily.SHIELDED_SURFACE): (1.00, 0.05, 2.8, 0.08),
    (ProteinBeadFamily.ACIDIC_PATCH, ProteinBeadFamily.ACIDIC_PATCH): (0.95, 0.18, 2.8, 0.55),
    (ProteinBeadFamily.ACIDIC_PATCH, ProteinBeadFamily.AROMATIC_HOTSPOT): (0.92, 1.05, 3.0, -0.52),
    (ProteinBeadFamily.ACIDIC_PATCH, ProteinBeadFamily.FLEXIBLE_LINKER): (0.98, 0.24, 2.8, -0.06),
    (ProteinBeadFamily.ACIDIC_PATCH, ProteinBeadFamily.SHIELDED_SURFACE): (1.00, 0.05, 2.8, 0.08),
    (ProteinBeadFamily.AROMATIC_HOTSPOT, ProteinBeadFamily.AROMATIC_HOTSPOT): (0.90, 1.15, 3.0, -0.25),
    (ProteinBeadFamily.AROMATIC_HOTSPOT, ProteinBeadFamily.FLEXIBLE_LINKER): (0.95, 0.36, 2.8, -0.08),
    (ProteinBeadFamily.AROMATIC_HOTSPOT, ProteinBeadFamily.SHIELDED_SURFACE): (0.99, 0.08, 2.8, 0.04),
    (ProteinBeadFamily.FLEXIBLE_LINKER, ProteinBeadFamily.FLEXIBLE_LINKER): (1.04, 0.10, 2.8, 0.00),
    (ProteinBeadFamily.FLEXIBLE_LINKER, ProteinBeadFamily.SHIELDED_SURFACE): (1.05, 0.04, 2.8, 0.01),
    (ProteinBeadFamily.SHIELDED_SURFACE, ProteinBeadFamily.SHIELDED_SURFACE): (1.08, 0.02, 2.8, 0.02),
}


def _normalized_family_pair(
    left: ProteinBeadFamily,
    right: ProteinBeadFamily,
) -> tuple[ProteinBeadFamily, ProteinBeadFamily]:
    return tuple(sorted((left, right), key=lambda family: family.value))


@dataclass(slots=True)
class ProteinShadowProfileFactory(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Build protein-general trusted shadow profiles from a topology."""

    name: str = "protein_shadow_profile_factory"
    classification: str = "[hybrid]"
    source_label: str = "protein_shadow_family_priors"

    def describe_role(self) -> str:
        return (
            "Builds protein-general trusted shadow interaction profiles from bead "
            "roles and protein-chemistry family heuristics instead of one benchmark's bespoke table."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "topology/system_topology.py",
            "forcefields/trusted_sources.py",
            "qcloud/protein_shadow_tuning.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/shadow_coarse_grained_fidelity.md",
            "docs/architecture/protein_general_shadow_tuning.md",
        )

    def validate(self) -> tuple[str, ...]:
        if not self.source_label.strip():
            return ("source_label must be a non-empty string.",)
        return ()

    def infer_family(self, bead_type: BeadType) -> ProteinBeadFamily:
        normalized_name = bead_type.name.lower()
        if any(token in normalized_name for token in ("acid", "glu", "asp", "carbox")):
            return ProteinBeadFamily.ACIDIC_PATCH
        if any(token in normalized_name for token in ("basic", "lys", "arg", "cation")):
            return ProteinBeadFamily.BASIC_PATCH
        if any(token in normalized_name for token in ("hotspot", "aromatic", "pi", "ridge")):
            return ProteinBeadFamily.AROMATIC_HOTSPOT
        if any(token in normalized_name for token in ("shield", "glycan", "cap")):
            return ProteinBeadFamily.SHIELDED_SURFACE
        if any(token in normalized_name for token in ("loop", "hinge", "tail", "link")):
            return ProteinBeadFamily.FLEXIBLE_LINKER
        if normalized_name == "core":
            return ProteinBeadFamily.HYDROPHOBIC_CORE
        if bead_type.role == BeadRole.STRUCTURAL:
            return ProteinBeadFamily.HYDROPHOBIC_CORE
        if bead_type.role == BeadRole.LINKER:
            return ProteinBeadFamily.FLEXIBLE_LINKER
        if bead_type.role == BeadRole.ANCHOR:
            return ProteinBeadFamily.AROMATIC_HOTSPOT
        if bead_type.role == BeadRole.FUNCTIONAL:
            return ProteinBeadFamily.POLAR_SURFACE
        return ProteinBeadFamily.POLAR_SURFACE

    def assignments_for_topology(self, topology: SystemTopology) -> tuple[ProteinBeadFamilyAssignment, ...]:
        return tuple(
            ProteinBeadFamilyAssignment(
                bead_type=bead_type.name,
                role=bead_type.role.value,
                family=self.infer_family(bead_type).value,
                metadata=bead_type.metadata.with_updates({"description": bead_type.description}),
            )
            for bead_type in topology.bead_types
        )

    def pair_prior_for_families(
        self,
        left: ProteinBeadFamily,
        right: ProteinBeadFamily,
    ) -> tuple[float, float, float, float]:
        key = _normalized_family_pair(left, right)
        if key in _PAIR_PRIORS:
            return _PAIR_PRIORS[key]
        reverse_key = (key[1], key[0])
        if reverse_key in _PAIR_PRIORS:
            return _PAIR_PRIORS[reverse_key]
        raise KeyError(key)

    def build_parameter_set(
        self,
        topology: SystemTopology,
        *,
        scenario_label: str,
        reference_label: str | None = None,
        metadata: FrozenMetadata | dict[str, object] | None = None,
    ) -> TrustedParameterSet:
        assignments = {
            assignment.bead_type: ProteinBeadFamily(assignment.family)
            for assignment in self.assignments_for_topology(topology)
        }
        sources = [
            TrustedScienceSource(
                label=self.source_label,
                source_type="adapted_parameterization",
                citation=(
                    "Repository-native protein shadow family priors derived from well-known "
                    "protein chemistry patterns, coarse-grained force-field design norms, and "
                    "large-step stability constraints."
                ),
                adaptation_notes=(
                    "These profiles are not copied from one external engine. They are repository-owned "
                    "protein-family priors meant to generalize across many protein systems while keeping "
                    "the shadow layer explicit and bounded."
                ),
            ),
        ]
        if reference_label:
            sources.append(
                TrustedScienceSource(
                    label=f"{scenario_label}_reference_anchor",
                    source_type="structure_reference",
                    citation=f"Scenario reference anchor: {reference_label}.",
                    adaptation_notes=(
                        "Used only as scenario-level provenance for the protein-general shadow profile bundle."
                    ),
                )
            )

        profiles: list[TrustedNonbondedProfile] = []
        for bead_type_a, bead_type_b in combinations_with_replacement(topology.bead_type_names, 2):
            family_a = assignments[bead_type_a]
            family_b = assignments[bead_type_b]
            sigma, epsilon, cutoff, electrostatic_strength = self.pair_prior_for_families(family_a, family_b)
            profiles.append(
                TrustedNonbondedProfile(
                    bead_type_a=bead_type_a,
                    bead_type_b=bead_type_b,
                    sigma=sigma,
                    epsilon=epsilon,
                    cutoff=cutoff,
                    electrostatic_strength=electrostatic_strength,
                    source_label=self.source_label,
                    fidelity_label="protein_general_shadow",
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
            }
        )
        return TrustedParameterSet(
            name=f"{scenario_label}_protein_shadow_parameters",
            summary=(
                "Protein-general trusted shadow interaction priors tuned for reusable coarse-to-shadow "
                "correction across many protein systems rather than one benchmark only."
            ),
            sources=tuple(sources),
            nonbonded_profiles=tuple(profiles),
            metadata=combined_metadata,
        )


__all__ = [
    "ProteinBeadFamily",
    "ProteinBeadFamilyAssignment",
    "ProteinShadowProfileFactory",
]
