"""Mapping rules from coarse beads to a shadow correction cloud."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, Vector3, coerce_scalar, coerce_vector3


@dataclass(frozen=True, slots=True)
class ShadowSiteTemplate(ValidatableComponent):
    """One shadow site template mirrored around a coarse bead."""

    site_name: str
    relative_offset: Vector3 = (0.0, 0.0, 0.0)
    sigma_scale: float = 1.0
    epsilon_scale: float = 1.0
    charge_scale: float = 0.0
    occupancy: float = 1.0
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_offset", coerce_vector3(self.relative_offset, "relative_offset"))
        object.__setattr__(self, "sigma_scale", coerce_scalar(self.sigma_scale, "sigma_scale"))
        object.__setattr__(self, "epsilon_scale", coerce_scalar(self.epsilon_scale, "epsilon_scale"))
        object.__setattr__(self, "charge_scale", coerce_scalar(self.charge_scale, "charge_scale"))
        object.__setattr__(self, "occupancy", coerce_scalar(self.occupancy, "occupancy"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.site_name.strip():
            issues.append("site_name must be a non-empty string.")
        if self.sigma_scale <= 0.0:
            issues.append("sigma_scale must be strictly positive.")
        if self.epsilon_scale < 0.0:
            issues.append("epsilon_scale must be non-negative.")
        if self.occupancy <= 0.0:
            issues.append("occupancy must be strictly positive.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ShadowMappingRule(ValidatableComponent):
    """One mapping rule from a coarse bead type to shadow sites."""

    bead_type: str
    source_label: str
    site_templates: tuple[ShadowSiteTemplate, ...]
    mirror_scale: float = 1.0
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "site_templates", tuple(self.site_templates))
        object.__setattr__(self, "mirror_scale", coerce_scalar(self.mirror_scale, "mirror_scale"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.bead_type.strip():
            issues.append("bead_type must be a non-empty string.")
        if not self.source_label.strip():
            issues.append("source_label must be a non-empty string.")
        if not self.site_templates:
            issues.append("site_templates must contain at least one shadow site template.")
        if self.mirror_scale <= 0.0:
            issues.append("mirror_scale must be strictly positive.")
        site_names = tuple(template.site_name for template in self.site_templates)
        if len(site_names) != len(set(site_names)):
            issues.append("site template names must be unique within one mapping rule.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ShadowMappingLibrary(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Registry of shadow mapping rules used by the fidelity layer."""

    rules: tuple[ShadowMappingRule, ...]
    name: str = "shadow_mapping_library"
    classification: str = "[proposed novel]"
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", tuple(self.rules))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Defines how one coarse bead expands into a mirrored shadow cloud that "
            "can carry trusted high-fidelity correction structure."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "topology/system_topology.py",
            "qcloud/shadow_cloud.py",
            "forcefields/trusted_sources.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/shadow_coarse_grained_fidelity.md",)

    def registered_bead_types(self) -> tuple[str, ...]:
        return tuple(rule.bead_type for rule in self.rules)

    def rule_for_bead_type(self, bead_type: str) -> ShadowMappingRule:
        for rule in self.rules:
            if rule.bead_type == bead_type:
                return rule
        raise KeyError(bead_type)

    def validate(self) -> tuple[str, ...]:
        bead_types = self.registered_bead_types()
        if len(bead_types) != len(set(bead_types)):
            return ("Shadow mapping rules must be unique per bead type.",)
        return ()
