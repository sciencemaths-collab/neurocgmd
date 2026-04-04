"""Protein-general shadow tuning presets for large-step fast coarse simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, coerce_scalar
from forcefields import (
    ProteinBeadFamily,
    ProteinShadowProfileFactory,
    ProteinSpatialProfileFactory,
    SpatialSemanticParameterSet,
    TrustedParameterSet,
)
from qcloud.spatial_semantic_field import SpatialSemanticFieldModel, SpatialSemanticFieldPolicy
from qcloud.shadow_cloud import ShadowCloudBuilder
from qcloud.shadow_correction import ShadowCorrectionPolicy, ShadowDrivenCorrectionModel
from qcloud.shadow_mapping import ShadowMappingLibrary, ShadowMappingRule, ShadowSiteTemplate
from topology import SystemTopology


@dataclass(frozen=True, slots=True)
class ProteinShadowDynamicsRecommendation(ValidatableComponent):
    """Recommended large-step dynamics settings for one protein shadow build."""

    time_step: float
    friction_coefficient: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_step", coerce_scalar(self.time_step, "time_step"))
        object.__setattr__(
            self,
            "friction_coefficient",
            coerce_scalar(self.friction_coefficient, "friction_coefficient"),
        )
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.time_step <= 0.0:
            issues.append("time_step must be strictly positive.")
        if self.friction_coefficient < 0.0:
            issues.append("friction_coefficient must be non-negative.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ProteinShadowRuntimeBundle(ValidatableComponent):
    """Reusable protein-shadow runtime bundle for scenarios and validators."""

    parameter_set: TrustedParameterSet
    spatial_parameter_set: SpatialSemanticParameterSet
    mapping_library: ShadowMappingLibrary
    correction_policy: ShadowCorrectionPolicy
    spatial_field_policy: SpatialSemanticFieldPolicy
    dynamics_recommendation: ProteinShadowDynamicsRecommendation
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        issues.extend(self.parameter_set.validate())
        issues.extend(self.spatial_parameter_set.validate())
        issues.extend(self.mapping_library.validate())
        issues.extend(self.correction_policy.validate())
        issues.extend(self.spatial_field_policy.validate())
        issues.extend(self.dynamics_recommendation.validate())
        return tuple(issues)

    def build_correction_model(self) -> ShadowDrivenCorrectionModel:
        return ShadowDrivenCorrectionModel(
            trusted_parameter_set=self.parameter_set,
            shadow_builder=ShadowCloudBuilder(mapping_library=self.mapping_library),
            policy=self.correction_policy,
            spatial_field_model=SpatialSemanticFieldModel(
                parameter_set=self.spatial_parameter_set,
                policy=self.spatial_field_policy,
            ),
        )


@dataclass(frozen=True, slots=True)
class ProteinShadowTuningPreset(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Stability-biased preset for fast large-step protein shadow correction."""

    name: str = "protein_large_step_fast"
    classification: str = "[hybrid]"
    minimum_site_distance: float = 0.10
    max_interaction_distance: float = 2.35
    energy_scale: float = 0.60
    electrostatic_scale: float = 0.78
    spatial_profile_preferred_distance_scale: float = 1.0
    spatial_profile_distance_tolerance_scale: float = 1.0
    spatial_profile_attraction_scale: float = 1.0
    spatial_profile_repulsion_scale: float = 1.0
    spatial_profile_directional_scale: float = 1.0
    spatial_profile_chemistry_scale: float = 1.0
    spatial_energy_scale: float = 0.72
    spatial_repulsion_scale: float = 1.15
    spatial_alignment_floor: float = 0.28
    spatial_max_pair_energy_magnitude: float = 1.8
    spatial_max_pair_force_magnitude: float = 2.2
    time_step_multiplier: float = 1.50
    friction_multiplier: float = 1.35
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        for field_name in (
            "minimum_site_distance",
            "max_interaction_distance",
            "energy_scale",
            "electrostatic_scale",
            "spatial_profile_preferred_distance_scale",
            "spatial_profile_distance_tolerance_scale",
            "spatial_profile_attraction_scale",
            "spatial_profile_repulsion_scale",
            "spatial_profile_directional_scale",
            "spatial_profile_chemistry_scale",
            "spatial_energy_scale",
            "spatial_repulsion_scale",
            "spatial_alignment_floor",
            "spatial_max_pair_energy_magnitude",
            "spatial_max_pair_force_magnitude",
            "time_step_multiplier",
            "friction_multiplier",
        ):
            object.__setattr__(self, field_name, coerce_scalar(getattr(self, field_name), field_name))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Provides one stability-biased large-step preset so the protein-general "
            "shadow layer stays fast and bounded instead of turning into a stiff all-atom clone."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "qcloud/shadow_correction.py",
            "forcefields/protein_shadow_profiles.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/shadow_coarse_grained_fidelity.md",
            "docs/architecture/protein_general_shadow_tuning.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.minimum_site_distance <= 0.0:
            issues.append("minimum_site_distance must be strictly positive.")
        if self.max_interaction_distance <= 0.0:
            issues.append("max_interaction_distance must be strictly positive.")
        if self.energy_scale <= 0.0:
            issues.append("energy_scale must be strictly positive.")
        if self.electrostatic_scale < 0.0:
            issues.append("electrostatic_scale must be non-negative.")
        if self.spatial_profile_preferred_distance_scale <= 0.0:
            issues.append("spatial_profile_preferred_distance_scale must be strictly positive.")
        if self.spatial_profile_distance_tolerance_scale <= 0.0:
            issues.append("spatial_profile_distance_tolerance_scale must be strictly positive.")
        if self.spatial_profile_attraction_scale <= 0.0:
            issues.append("spatial_profile_attraction_scale must be strictly positive.")
        if self.spatial_profile_repulsion_scale <= 0.0:
            issues.append("spatial_profile_repulsion_scale must be strictly positive.")
        if self.spatial_profile_directional_scale <= 0.0:
            issues.append("spatial_profile_directional_scale must be strictly positive.")
        if self.spatial_profile_chemistry_scale <= 0.0:
            issues.append("spatial_profile_chemistry_scale must be strictly positive.")
        if self.spatial_energy_scale <= 0.0:
            issues.append("spatial_energy_scale must be strictly positive.")
        if self.spatial_repulsion_scale <= 0.0:
            issues.append("spatial_repulsion_scale must be strictly positive.")
        if not 0.0 <= self.spatial_alignment_floor <= 1.0:
            issues.append("spatial_alignment_floor must lie in the interval [0, 1].")
        if self.spatial_max_pair_energy_magnitude <= 0.0:
            issues.append("spatial_max_pair_energy_magnitude must be strictly positive.")
        if self.spatial_max_pair_force_magnitude <= 0.0:
            issues.append("spatial_max_pair_force_magnitude must be strictly positive.")
        if self.time_step_multiplier <= 1.0:
            issues.append("time_step_multiplier must exceed 1.0 for the large-step preset.")
        if self.friction_multiplier <= 0.0:
            issues.append("friction_multiplier must be strictly positive.")
        return tuple(issues)

    def build_policy(self) -> ShadowCorrectionPolicy:
        return ShadowCorrectionPolicy(
            minimum_site_distance=self.minimum_site_distance,
            max_interaction_distance=self.max_interaction_distance,
            energy_scale=self.energy_scale,
            electrostatic_scale=self.electrostatic_scale,
        )

    def build_spatial_policy(self) -> SpatialSemanticFieldPolicy:
        return SpatialSemanticFieldPolicy(
            minimum_distance=self.minimum_site_distance,
            energy_scale=self.spatial_energy_scale,
            repulsion_scale=self.spatial_repulsion_scale,
            alignment_floor=self.spatial_alignment_floor,
            max_pair_energy_magnitude=self.spatial_max_pair_energy_magnitude,
            max_pair_force_magnitude=self.spatial_max_pair_force_magnitude,
        )


@dataclass(slots=True)
class ProteinShadowTuner(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Build protein-general shadow runtime bundles from a concrete topology."""

    preset: ProteinShadowTuningPreset = field(default_factory=ProteinShadowTuningPreset)
    profile_factory: ProteinShadowProfileFactory = field(default_factory=ProteinShadowProfileFactory)
    spatial_profile_factory: ProteinSpatialProfileFactory = field(default_factory=ProteinSpatialProfileFactory)
    name: str = "protein_shadow_tuner"
    classification: str = "[hybrid]"

    def describe_role(self) -> str:
        return (
            "Generalizes the shadow coarse-grained layer across many proteins by "
            "combining protein-family priors, spatial-semantic local force priors, mirrored mapping rules, "
            "and large-step stability tuning."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "forcefields/protein_shadow_profiles.py",
            "forcefields/spatial_semantic_profiles.py",
            "qcloud/shadow_mapping.py",
            "qcloud/shadow_correction.py",
            "qcloud/spatial_semantic_field.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/protein_general_shadow_tuning.md",)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        issues.extend(self.preset.validate())
        issues.extend(self.profile_factory.validate())
        issues.extend(self.spatial_profile_factory.validate())
        return tuple(issues)

    def recommend_dynamics(
        self,
        *,
        base_time_step: float,
        base_friction: float,
    ) -> ProteinShadowDynamicsRecommendation:
        if base_time_step <= 0.0:
            raise ContractValidationError("base_time_step must be strictly positive.")
        if base_friction < 0.0:
            raise ContractValidationError("base_friction must be non-negative.")
        return ProteinShadowDynamicsRecommendation(
            time_step=base_time_step * self.preset.time_step_multiplier,
            friction_coefficient=base_friction * self.preset.friction_multiplier,
            metadata={
                "preset_name": self.preset.name,
                "time_step_multiplier": self.preset.time_step_multiplier,
                "friction_multiplier": self.preset.friction_multiplier,
            },
        )

    def build_mapping_library(self, topology: SystemTopology) -> ShadowMappingLibrary:
        assignments = {
            assignment.bead_type: ProteinBeadFamily(assignment.family)
            for assignment in self.profile_factory.assignments_for_topology(topology)
        }
        rules = [
            ShadowMappingRule(
                bead_type=bead_type_name,
                source_label=self.profile_factory.source_label,
                site_templates=self._site_templates_for_family(family),
                mirror_scale=0.85 if family == ProteinBeadFamily.FLEXIBLE_LINKER else 1.0,
                metadata={"protein_family": family.value},
            )
            for bead_type_name, family in assignments.items()
        ]
        return ShadowMappingLibrary(
            rules=tuple(rules),
            metadata={
                "preset_name": self.preset.name,
                "registered_protein_families": sorted({family.value for family in assignments.values()}),
            },
        )

    def build_runtime_bundle(
        self,
        *,
        topology: SystemTopology,
        scenario_label: str,
        base_time_step: float,
        base_friction: float,
        reference_label: str | None = None,
        metadata: FrozenMetadata | dict[str, object] | None = None,
    ) -> ProteinShadowRuntimeBundle:
        parameter_set = self.profile_factory.build_parameter_set(
            topology,
            scenario_label=scenario_label,
            reference_label=reference_label,
            metadata=metadata,
        )
        spatial_parameter_set = self.spatial_profile_factory.build_parameter_set(
            topology,
            scenario_label=scenario_label,
            reference_label=reference_label,
            preferred_distance_scale=self.preset.spatial_profile_preferred_distance_scale,
            distance_tolerance_scale=self.preset.spatial_profile_distance_tolerance_scale,
            attraction_scale=self.preset.spatial_profile_attraction_scale,
            repulsion_scale=self.preset.spatial_profile_repulsion_scale,
            directional_scale=self.preset.spatial_profile_directional_scale,
            chemistry_scale=self.preset.spatial_profile_chemistry_scale,
            metadata=metadata,
        )
        mapping_library = self.build_mapping_library(topology)
        recommendation = self.recommend_dynamics(
            base_time_step=base_time_step,
            base_friction=base_friction,
        )
        return ProteinShadowRuntimeBundle(
            parameter_set=parameter_set,
            spatial_parameter_set=spatial_parameter_set,
            mapping_library=mapping_library,
            correction_policy=self.preset.build_policy(),
            spatial_field_policy=self.preset.build_spatial_policy(),
            dynamics_recommendation=recommendation,
            metadata={
                "scenario": scenario_label,
                "reference_label": reference_label,
                "preset_name": self.preset.name,
                "spatial_forcefield_enabled": True,
                "spatial_profile_attraction_scale": self.preset.spatial_profile_attraction_scale,
                "spatial_profile_repulsion_scale": self.preset.spatial_profile_repulsion_scale,
                "spatial_profile_chemistry_scale": self.preset.spatial_profile_chemistry_scale,
            },
        )

    def _site_templates_for_family(
        self,
        family: ProteinBeadFamily,
    ) -> tuple[ShadowSiteTemplate, ...]:
        if family == ProteinBeadFamily.HYDROPHOBIC_CORE:
            return (
                ShadowSiteTemplate("core_anchor", relative_offset=(0.0, 0.0, 0.0), sigma_scale=1.0, epsilon_scale=0.80, charge_scale=0.10),
                ShadowSiteTemplate("core_flank", relative_offset=(0.0, 0.10, 0.0), sigma_scale=0.94, epsilon_scale=0.55, charge_scale=0.05, occupancy=0.70),
            )
        if family == ProteinBeadFamily.POLAR_SURFACE:
            return (
                ShadowSiteTemplate("surface_core", relative_offset=(0.0, 0.0, 0.0), sigma_scale=0.98, epsilon_scale=0.90, charge_scale=0.45),
                ShadowSiteTemplate("surface_halo", relative_offset=(0.0, 0.12, 0.0), sigma_scale=0.92, epsilon_scale=0.55, charge_scale=0.25, occupancy=0.65),
            )
        if family == ProteinBeadFamily.BASIC_PATCH:
            return (
                ShadowSiteTemplate("basic_core", relative_offset=(0.0, 0.0, 0.0), sigma_scale=0.94, epsilon_scale=1.05, charge_scale=1.00),
                ShadowSiteTemplate("basic_halo", relative_offset=(0.0, 0.12, 0.0), sigma_scale=0.88, epsilon_scale=0.65, charge_scale=0.55, occupancy=0.70),
            )
        if family == ProteinBeadFamily.ACIDIC_PATCH:
            return (
                ShadowSiteTemplate("acidic_core", relative_offset=(0.0, 0.0, 0.0), sigma_scale=0.94, epsilon_scale=1.05, charge_scale=-1.00),
                ShadowSiteTemplate("acidic_halo", relative_offset=(0.0, 0.12, 0.0), sigma_scale=0.88, epsilon_scale=0.65, charge_scale=-0.55, occupancy=0.70),
            )
        if family == ProteinBeadFamily.AROMATIC_HOTSPOT:
            return (
                ShadowSiteTemplate("hotspot_core", relative_offset=(0.0, 0.0, 0.0), sigma_scale=0.90, epsilon_scale=1.15, charge_scale=0.35),
                ShadowSiteTemplate("hotspot_ring", relative_offset=(0.0, 0.10, 0.0), sigma_scale=0.84, epsilon_scale=0.80, charge_scale=0.20, occupancy=0.75),
            )
        if family == ProteinBeadFamily.FLEXIBLE_LINKER:
            return (
                ShadowSiteTemplate("linker_core", relative_offset=(0.0, 0.0, 0.0), sigma_scale=1.02, epsilon_scale=0.45, charge_scale=0.00, occupancy=0.72),
            )
        return (
            ShadowSiteTemplate("shield_surface", relative_offset=(0.0, 0.0, 0.0), sigma_scale=1.06, epsilon_scale=0.28, charge_scale=-0.08, occupancy=0.55),
        )


__all__ = [
    "ProteinShadowDynamicsRecommendation",
    "ProteinShadowRuntimeBundle",
    "ProteinShadowTuner",
    "ProteinShadowTuningPreset",
]
