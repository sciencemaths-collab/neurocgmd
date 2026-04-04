"""Chemistry-aware interface analysis across compartment or domain boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from chemistry.residue_semantics import (
    BeadChemistryAssignment,
    ProteinChemistryModel,
    ProteinChemistrySummary,
)
from compartments.registry import CompartmentRegistry
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata, coerce_scalar
from topology.system_topology import SystemTopology


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    delta_x = left[0] - right[0]
    delta_y = left[1] - right[1]
    delta_z = left[2] - right[2]
    return sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)


@dataclass(frozen=True, slots=True)
class ChemistryPairSignal(ValidatableComponent):
    """One chemically interpreted cross-interface pair."""

    particle_index_a: int
    particle_index_b: int
    label_a: str
    label_b: str
    distance: float
    charge_compatibility: float
    hydropathy_alignment: float
    hydrogen_bond_match: float
    aromatic_hotspot_bonus: float
    flexibility_penalty: float
    pair_score: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        for field_name in (
            "distance",
            "charge_compatibility",
            "hydropathy_alignment",
            "hydrogen_bond_match",
            "aromatic_hotspot_bonus",
            "flexibility_penalty",
            "pair_score",
        ):
            object.__setattr__(self, field_name, coerce_scalar(getattr(self, field_name), field_name))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.particle_index_a < 0 or self.particle_index_b < 0:
            issues.append("particle indices must be non-negative.")
        if not self.label_a.strip() or not self.label_b.strip():
            issues.append("pair labels must be non-empty strings.")
        if self.distance < 0.0:
            issues.append("distance must be non-negative.")
        for field_name in (
            "charge_compatibility",
            "hydropathy_alignment",
            "hydrogen_bond_match",
            "aromatic_hotspot_bonus",
            "flexibility_penalty",
            "pair_score",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                issues.append(f"{field_name} must lie in the interval [0, 1].")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ChemistryInterfaceReport(ValidatableComponent):
    """Aggregate chemistry diagnosis for one active interface."""

    title: str
    compartment_ids: tuple[str, str]
    evaluated_pair_count: int
    favorable_pair_fraction: float
    mean_pair_score: float
    charge_complementarity: float
    hydropathy_alignment: float
    flexibility_pressure: float
    hotspot_pair_fraction: float
    dominant_pairs: tuple[ChemistryPairSignal, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "compartment_ids", tuple(self.compartment_ids))
        object.__setattr__(self, "dominant_pairs", tuple(self.dominant_pairs))
        for field_name in (
            "favorable_pair_fraction",
            "mean_pair_score",
            "charge_complementarity",
            "hydropathy_alignment",
            "flexibility_pressure",
            "hotspot_pair_fraction",
        ):
            object.__setattr__(self, field_name, coerce_scalar(getattr(self, field_name), field_name))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.title.strip():
            issues.append("title must be a non-empty string.")
        if len(self.compartment_ids) != 2 or any(not value.strip() for value in self.compartment_ids):
            issues.append("compartment_ids must contain exactly two non-empty identifiers.")
        if self.evaluated_pair_count <= 0:
            issues.append("evaluated_pair_count must be strictly positive.")
        for field_name in (
            "favorable_pair_fraction",
            "mean_pair_score",
            "charge_complementarity",
            "hydropathy_alignment",
            "flexibility_pressure",
            "hotspot_pair_fraction",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                issues.append(f"{field_name} must lie in the interval [0, 1].")
        return tuple(issues)


@dataclass(slots=True)
class ChemistryInterfaceAnalyzer(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """[hybrid] Diagnose whether an observed interface is chemically plausible."""

    chemistry_model: ProteinChemistryModel = field(default_factory=ProteinChemistryModel)
    interaction_distance: float = 2.6
    fallback_pair_count: int = 4
    favorable_threshold: float = 0.58
    charge_weight: float = 0.22
    hydropathy_weight: float = 0.18
    hydrogen_bond_weight: float = 0.18
    aromatic_weight: float = 0.17
    distance_weight: float = 0.25
    flexibility_penalty_weight: float = 0.18
    name: str = "chemistry_interface_analyzer"
    classification: str = "[hybrid]"

    def describe_role(self) -> str:
        return (
            "Turns bead-level chemistry semantics into bounded interface-plausibility "
            "signals that ML, qcloud, and executive control can consume explicitly."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "chemistry/residue_semantics.py",
            "compartments/registry.py",
            "topology/system_topology.py",
            "core/state.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/chemistry_semantics_and_live_control.md",)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.interaction_distance <= 0.0:
            issues.append("interaction_distance must be positive.")
        if self.fallback_pair_count <= 0:
            issues.append("fallback_pair_count must be strictly positive.")
        if not 0.0 <= self.favorable_threshold <= 1.0:
            issues.append("favorable_threshold must lie in the interval [0, 1].")
        for field_name in (
            "charge_weight",
            "hydropathy_weight",
            "hydrogen_bond_weight",
            "aromatic_weight",
            "distance_weight",
            "flexibility_penalty_weight",
        ):
            if getattr(self, field_name) < 0.0:
                issues.append(f"{field_name} must be non-negative.")
        return tuple(issues)

    def assess(
        self,
        state: SimulationState,
        topology: SystemTopology,
        compartments: CompartmentRegistry,
        *,
        compartment_ids: tuple[str, str],
        distance_cutoff: float | None = None,
    ) -> ChemistryInterfaceReport:
        if state.particle_count != topology.particle_count:
            raise ContractValidationError("SimulationState particle_count must match SystemTopology particle_count.")
        if topology.particle_count != compartments.particle_count:
            raise ContractValidationError("SystemTopology particle_count must match CompartmentRegistry particle_count.")

        resolved_cutoff = distance_cutoff if distance_cutoff is not None else self.interaction_distance
        if resolved_cutoff <= 0.0:
            raise ContractValidationError("distance_cutoff must be positive.")

        domain_a = compartments.domain_by_id(compartment_ids[0])
        domain_b = compartments.domain_by_id(compartment_ids[1])
        chemistry_summary = self.chemistry_model.summarize_topology(topology)
        assignment_map = {
            assignment.particle_index: assignment
            for assignment in chemistry_summary.assignments
        }

        all_signals: list[ChemistryPairSignal] = []
        for particle_index_a in domain_a.particle_indices:
            for particle_index_b in domain_b.particle_indices:
                assignment_a = assignment_map[particle_index_a]
                assignment_b = assignment_map[particle_index_b]
                distance = _distance(
                    state.particles.positions[particle_index_a],
                    state.particles.positions[particle_index_b],
                )
                all_signals.append(
                    self._pair_signal(
                        assignment_a,
                        assignment_b,
                        distance=distance,
                        distance_cutoff=resolved_cutoff,
                    )
                )

        if not all_signals:
            raise ContractValidationError("At least one cross-compartment pair is required for chemistry analysis.")

        selected_signals = tuple(signal for signal in all_signals if signal.distance <= resolved_cutoff)
        if not selected_signals:
            selected_signals = tuple(
                sorted(
                    all_signals,
                    key=lambda signal: (signal.distance, -signal.pair_score),
                )[: self.fallback_pair_count]
            )

        favorable_pair_fraction = sum(signal.pair_score >= self.favorable_threshold for signal in selected_signals) / len(
            selected_signals
        )
        mean_pair_score = sum(signal.pair_score for signal in selected_signals) / len(selected_signals)
        charge_complementarity = sum(signal.charge_compatibility for signal in selected_signals) / len(selected_signals)
        hydropathy_alignment = sum(signal.hydropathy_alignment for signal in selected_signals) / len(selected_signals)
        flexibility_pressure = sum(signal.flexibility_penalty for signal in selected_signals) / len(selected_signals)
        hotspot_pair_fraction = sum(signal.aromatic_hotspot_bonus >= 0.45 for signal in selected_signals) / len(
            selected_signals
        )
        dominant_pairs = tuple(
            sorted(
                selected_signals,
                key=lambda signal: (-signal.pair_score, signal.distance),
            )[:3]
        )

        return ChemistryInterfaceReport(
            title="Interface Chemistry",
            compartment_ids=(str(domain_a.compartment_id), str(domain_b.compartment_id)),
            evaluated_pair_count=len(selected_signals),
            favorable_pair_fraction=favorable_pair_fraction,
            mean_pair_score=mean_pair_score,
            charge_complementarity=charge_complementarity,
            hydropathy_alignment=hydropathy_alignment,
            flexibility_pressure=flexibility_pressure,
            hotspot_pair_fraction=hotspot_pair_fraction,
            dominant_pairs=dominant_pairs,
            metadata={
                "distance_cutoff": resolved_cutoff,
                "candidate_pair_count": len(all_signals),
                "mean_distance": sum(signal.distance for signal in selected_signals) / len(selected_signals),
                "chemistry_model": self.chemistry_model.name,
                "mean_abs_charge": chemistry_summary.mean_abs_charge,
            },
        )

    def _pair_signal(
        self,
        assignment_a: BeadChemistryAssignment,
        assignment_b: BeadChemistryAssignment,
        *,
        distance: float,
        distance_cutoff: float,
    ) -> ChemistryPairSignal:
        descriptor_a = assignment_a.descriptor
        descriptor_b = assignment_b.descriptor
        charge_a = descriptor_a.formal_charge
        charge_b = descriptor_b.formal_charge
        if abs(charge_a) >= 0.25 and abs(charge_b) >= 0.25:
            charge_compatibility = 1.0 if charge_a * charge_b < 0.0 else 0.08
        elif abs(charge_a) >= 0.25 or abs(charge_b) >= 0.25:
            charge_compatibility = 0.68 if descriptor_a.hydrogen_bond_capacity + descriptor_b.hydrogen_bond_capacity >= 1.0 else 0.52
        else:
            charge_compatibility = 0.55

        hydropathy_alignment = 1.0 - min(1.0, abs(descriptor_a.hydropathy - descriptor_b.hydropathy) / 2.0)
        hydrogen_bond_match = sqrt(
            max(0.0, descriptor_a.hydrogen_bond_capacity * descriptor_b.hydrogen_bond_capacity)
        )
        aromatic_hotspot_bonus = _clamp(
            (
                sqrt(max(0.0, descriptor_a.aromaticity * descriptor_b.aromaticity)) * 0.45
                + sqrt(max(0.0, descriptor_a.hotspot_propensity * descriptor_b.hotspot_propensity)) * 0.55
            ),
            0.0,
            1.0,
        )
        flexibility_penalty = _clamp(
            (((descriptor_a.flexibility + descriptor_b.flexibility) * 0.5) - 0.55) / 0.45,
            0.0,
            1.0,
        )
        distance_quality = _clamp(1.0 - max(0.0, distance - 1.0) / max(1.0, distance_cutoff - 1.0), 0.0, 1.0)
        weighted_score = (
            self.charge_weight * charge_compatibility
            + self.hydropathy_weight * hydropathy_alignment
            + self.hydrogen_bond_weight * hydrogen_bond_match
            + self.aromatic_weight * aromatic_hotspot_bonus
            + self.distance_weight * distance_quality
            - self.flexibility_penalty_weight * flexibility_penalty
        )
        normalization = (
            self.charge_weight
            + self.hydropathy_weight
            + self.hydrogen_bond_weight
            + self.aromatic_weight
            + self.distance_weight
        )
        pair_score = _clamp(weighted_score / normalization if normalization else 0.0, 0.0, 1.0)
        return ChemistryPairSignal(
            particle_index_a=assignment_a.particle_index,
            particle_index_b=assignment_b.particle_index,
            label_a=assignment_a.bead_label,
            label_b=assignment_b.bead_label,
            distance=distance,
            charge_compatibility=charge_compatibility,
            hydropathy_alignment=hydropathy_alignment,
            hydrogen_bond_match=hydrogen_bond_match,
            aromatic_hotspot_bonus=aromatic_hotspot_bonus,
            flexibility_penalty=flexibility_penalty,
            pair_score=pair_score,
            metadata={
                "compartments": tuple(
                    value
                    for value in (assignment_a.compartment_id, assignment_b.compartment_id)
                    if value is not None
                ),
                "descriptor_sources": (
                    descriptor_a.descriptor_source,
                    descriptor_b.descriptor_source,
                ),
            },
        )
