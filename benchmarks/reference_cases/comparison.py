"""Comparison helpers between coarse proxies and experimental reference cases."""

from __future__ import annotations

from dataclasses import dataclass, field

from benchmarks.reference_cases.models import ExperimentalReferenceCase
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata


@dataclass(frozen=True, slots=True)
class ReferenceComparisonMetric(ValidatableComponent):
    """One comparison signal between a proxy simulation and a reference target."""

    label: str
    target_value: str
    current_value: str
    detail: str = ""
    status: str = "reference"
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.label.strip():
            issues.append("label must be a non-empty string.")
        if not self.target_value.strip():
            issues.append("target_value must be a non-empty string.")
        if not self.current_value.strip():
            issues.append("current_value must be a non-empty string.")
        if self.status not in {"reference", "tracking", "good", "warn"}:
            issues.append("status must be one of: reference, tracking, good, warn.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "target_value": self.target_value,
            "current_value": self.current_value,
            "detail": self.detail,
            "status": self.status,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ReferenceComparisonReport(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Structured comparison payload for one experimental benchmark target."""

    name: str = "reference_comparison_report"
    classification: str = "[hybrid]"
    title: str = ""
    summary: str = ""
    metrics: tuple[ReferenceComparisonMetric, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Keeps experimentally known answers visible next to the current proxy "
            "simulation state so later comparisons stay explicit and honest."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "benchmarks/reference_cases/models.py",
            "docs/use_cases/barnase_barstar_reference_case.md",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/validation_and_benchmarking.md",
            "docs/use_cases/barnase_barstar_reference_case.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.title.strip():
            issues.append("title must be a non-empty string.")
        if not self.summary.strip():
            issues.append("summary must be a non-empty string.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "classification": self.classification,
            "title": self.title,
            "summary": self.summary,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "metadata": self.metadata.to_dict(),
        }


def build_barnase_barstar_proxy_report(
    *,
    reference_case: ExperimentalReferenceCase,
    current_stage: str,
    interface_distance: float,
    bound_distance: float,
    cross_contact_count: int,
    target_contact_count: int,
    bound: bool,
) -> ReferenceComparisonReport:
    """Build a proxy-vs-reference report for the barnase-barstar benchmark."""

    return ReferenceComparisonReport(
        title="Experimental Reference",
        summary=(
            "Known answers from the real barnase-barstar system. The live run is still "
            "a coarse-grained proxy, so structural and kinetic agreement should be treated "
            "as future comparison targets rather than already solved outputs."
        ),
        metrics=(
            ReferenceComparisonMetric(
                label="Bound Structure",
                target_value=reference_case.structural_reference.bound_complex_pdb_id,
                current_value=current_stage,
                detail="Target bound complex from experiment versus the current proxy docking phase.",
                status="good" if bound else "tracking",
            ),
            ReferenceComparisonMetric(
                label="Free Partners",
                target_value=" / ".join(reference_case.structural_reference.unbound_partner_pdb_ids),
                current_value="reduced proxy",
                detail="Real free-partner structures to compare against later before association.",
                status="reference",
            ),
            ReferenceComparisonMetric(
                label="Target Kd",
                target_value=f"{reference_case.observable_for('dissociation_constant').expected_value:.2e} {reference_case.observable_for('dissociation_constant').units}",
                current_value="not estimated yet",
                detail="Thermodynamic agreement should be estimated only after a calibrated kinetics workflow exists.",
                status="reference",
            ),
            ReferenceComparisonMetric(
                label="Target k_on",
                target_value=f"{reference_case.observable_for('association_rate_constant').expected_value:.2e} {reference_case.observable_for('association_rate_constant').units}",
                current_value="not estimated yet",
                detail="Association kinetics are known experimentally but are not inferred by the current live proxy.",
                status="reference",
            ),
            ReferenceComparisonMetric(
                label="Target k_off",
                target_value=f"{reference_case.observable_for('dissociation_rate_constant').expected_value:.2e} {reference_case.observable_for('dissociation_rate_constant').units}",
                current_value="not estimated yet",
                detail="Dissociation kinetics will need longer controlled sampling and explicit event counting later.",
                status="reference",
            ),
            ReferenceComparisonMetric(
                label="Docking Proxy",
                target_value=f"<= {bound_distance:.2f} reduced units and >= {target_contact_count} proxy contacts",
                current_value=f"{interface_distance:.2f} / {cross_contact_count}",
                detail="Current proxy interface distance and cross-protein contact count.",
                status="good" if bound else "warn",
            ),
        ),
        metadata={
            "reference_case": reference_case.name,
            "bound": bound,
        },
    )


def build_spike_ace2_proxy_report(
    *,
    reference_case: ExperimentalReferenceCase,
    current_stage: str,
    interface_distance: float,
    bound_distance: float,
    cross_contact_count: int,
    target_contact_count: int,
    bound: bool,
) -> ReferenceComparisonReport:
    """Build a proxy-vs-reference report for the harder ACE2-spike benchmark."""

    return ReferenceComparisonReport(
        title="Experimental Reference",
        summary=(
            "Known answers from the real ACE2-spike receptor-recognition system. The live run "
            "is still a reduced coarse-grained proxy, so the dashboard should be read as a "
            "tracked comparison target rather than a solved physical reconstruction."
        ),
        metrics=(
            ReferenceComparisonMetric(
                label="Bound Structure",
                target_value=reference_case.structural_reference.bound_complex_pdb_id,
                current_value=current_stage,
                detail="Target experimental complex versus the current proxy recognition phase.",
                status="good" if bound else "tracking",
            ),
            ReferenceComparisonMetric(
                label="Free Partners",
                target_value=" / ".join(reference_case.structural_reference.unbound_partner_pdb_ids),
                current_value="reduced proxy",
                detail="Real receptor and spike references to compare against later before docking.",
                status="reference",
            ),
            ReferenceComparisonMetric(
                label="Target Apparent Kd",
                target_value=(
                    f"{reference_case.observable_for('apparent_dissociation_constant').expected_value:.2e} "
                    f"{reference_case.observable_for('apparent_dissociation_constant').units}"
                ),
                current_value="not estimated yet",
                detail="Affinity agreement should only be claimed after a construct-matched kinetics workflow exists.",
                status="reference",
            ),
            ReferenceComparisonMetric(
                label="Recognition Mode",
                target_value="ACE2 alpha1 helix vs RBD ridge hotspot network",
                current_value=current_stage,
                detail="The harder target is a distributed receptor-recognition geometry, not a single-point collapse.",
                status="good" if bound else "tracking",
            ),
            ReferenceComparisonMetric(
                label="Docking Proxy",
                target_value=f"<= {bound_distance:.2f} reduced units and >= {target_contact_count} proxy contacts",
                current_value=f"{interface_distance:.2f} / {cross_contact_count}",
                detail="Current proxy interface distance and ACE2-spike cross-contact count.",
                status="good" if bound else "warn",
            ),
        ),
        metadata={
            "reference_case": reference_case.name,
            "bound": bound,
        },
    )
