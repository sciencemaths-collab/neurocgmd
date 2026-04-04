"""Provenance-aware trusted parameter sources for shadow-fidelity correction."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, coerce_scalar


def _normalized_type_pair(type_a: str, type_b: str) -> tuple[str, str]:
    return tuple(sorted((type_a, type_b)))


@dataclass(frozen=True, slots=True)
class TrustedScienceSource(ValidatableComponent):
    """One trusted scientific source or parameter provenance record."""

    label: str
    source_type: str
    citation: str
    adaptation_notes: str
    license_notes: str = ""
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for field_name in ("label", "source_type", "citation", "adaptation_notes"):
            if not str(getattr(self, field_name)).strip():
                issues.append(f"{field_name} must be a non-empty string.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class TrustedNonbondedProfile(ValidatableComponent):
    """Trusted nonbonded interaction profile used by the shadow correction layer."""

    bead_type_a: str
    bead_type_b: str
    sigma: float
    epsilon: float
    cutoff: float
    electrostatic_strength: float = 0.0
    source_label: str = ""
    fidelity_label: str = "trusted_shadow"
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma", coerce_scalar(self.sigma, "sigma"))
        object.__setattr__(self, "epsilon", coerce_scalar(self.epsilon, "epsilon"))
        object.__setattr__(self, "cutoff", coerce_scalar(self.cutoff, "cutoff"))
        object.__setattr__(
            self,
            "electrostatic_strength",
            coerce_scalar(self.electrostatic_strength, "electrostatic_strength"),
        )
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
            issues.append("bead types must be non-empty strings.")
        if self.sigma <= 0.0:
            issues.append("sigma must be strictly positive.")
        if self.epsilon < 0.0:
            issues.append("epsilon must be non-negative.")
        if self.cutoff <= 0.0:
            issues.append("cutoff must be strictly positive.")
        if not self.source_label.strip():
            issues.append("source_label must be a non-empty string.")
        if not self.fidelity_label.strip():
            issues.append("fidelity_label must be a non-empty string.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class TrustedParameterSet(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Trusted parameter bundle that feeds the shadow coarse-grained layer."""

    name: str
    summary: str
    sources: tuple[TrustedScienceSource, ...]
    nonbonded_profiles: tuple[TrustedNonbondedProfile, ...]
    classification: str = "[adapted]"
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "nonbonded_profiles", tuple(self.nonbonded_profiles))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Tracks which trusted science, parameterizations, and adaptation notes "
            "feed the shadow coarse-grained correction layer."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "forcefields/base_forcefield.py",
            "qcloud/shadow_correction.py",
            "validation/fidelity_checks.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/forcefield_foundation.md",
            "docs/architecture/shadow_coarse_grained_fidelity.md",
        )

    def source_labels(self) -> tuple[str, ...]:
        return tuple(source.label for source in self.sources)

    def source_for(self, label: str) -> TrustedScienceSource:
        for source in self.sources:
            if source.label == label:
                return source
        raise KeyError(label)

    def nonbonded_profile_for_bead_types(self, bead_type_a: str, bead_type_b: str) -> TrustedNonbondedProfile:
        key = _normalized_type_pair(bead_type_a, bead_type_b)
        for profile in self.nonbonded_profiles:
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
            issues.append("sources must contain at least one trusted science source.")
        if not self.nonbonded_profiles:
            issues.append("nonbonded_profiles must contain at least one trusted profile.")

        source_labels = self.source_labels()
        if len(source_labels) != len(set(source_labels)):
            issues.append("source labels must be unique.")

        profile_keys = tuple(profile.pair_key() for profile in self.nonbonded_profiles)
        if len(profile_keys) != len(set(profile_keys)):
            issues.append("trusted nonbonded profiles must be unique per bead-type pair.")

        source_lookup = set(source_labels)
        missing_sources = sorted(
            {profile.source_label for profile in self.nonbonded_profiles if profile.source_label not in source_lookup}
        )
        if missing_sources:
            issues.append(
                "Each trusted profile must reference a declared source label; missing: "
                + ", ".join(missing_sources)
            )
        return tuple(issues)
