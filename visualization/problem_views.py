"""Problem-oriented dashboard views for concrete simulation objectives."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata


@dataclass(frozen=True, slots=True)
class ObjectiveMetricView(ValidatableComponent):
    """Serializable metric card for a concrete simulation objective."""

    label: str
    value: str
    detail: str = ""
    status: str = "neutral"
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
        if not self.value.strip():
            issues.append("value must be a non-empty string.")
        if self.status not in {"neutral", "active", "good", "warn"}:
            issues.append("status must be one of: neutral, active, good, warn.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "value": self.value,
            "detail": self.detail,
            "status": self.status,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ProblemStatementView(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Serializable problem statement and objective summary for the dashboard."""

    name: str = "problem_statement_view"
    classification: str = "[adapted]"
    title: str = ""
    summary: str = ""
    objective: str = ""
    stage: str = ""
    metrics: tuple[ObjectiveMetricView, ...] = ()
    reference_title: str = ""
    reference_summary: str = ""
    reference_metrics: tuple[ObjectiveMetricView, ...] = ()
    structure_title: str = ""
    structure_summary: str = ""
    structure_metrics: tuple[ObjectiveMetricView, ...] = ()
    chemistry_title: str = ""
    chemistry_summary: str = ""
    chemistry_metrics: tuple[ObjectiveMetricView, ...] = ()
    fidelity_title: str = ""
    fidelity_summary: str = ""
    fidelity_metrics: tuple[ObjectiveMetricView, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "reference_metrics", tuple(self.reference_metrics))
        object.__setattr__(self, "structure_metrics", tuple(self.structure_metrics))
        object.__setattr__(self, "chemistry_metrics", tuple(self.chemistry_metrics))
        object.__setattr__(self, "fidelity_metrics", tuple(self.fidelity_metrics))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Carries the concrete simulation objective, current stage, and progress "
            "metrics into the local dashboard without owning engine state."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "visualization/trajectory_views.py",
            "docs/use_cases/encounter_complex_dashboard.md",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/visualization_and_diagnostics.md",
            "docs/use_cases/encounter_complex_dashboard.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.title.strip():
            issues.append("title must be a non-empty string.")
        if not self.summary.strip():
            issues.append("summary must be a non-empty string.")
        if not self.objective.strip():
            issues.append("objective must be a non-empty string.")
        if not self.stage.strip():
            issues.append("stage must be a non-empty string.")
        if self.reference_metrics and not self.reference_title.strip():
            issues.append("reference_title must be provided when reference_metrics are attached.")
        if self.structure_metrics and not self.structure_title.strip():
            issues.append("structure_title must be provided when structure_metrics are attached.")
        if self.chemistry_metrics and not self.chemistry_title.strip():
            issues.append("chemistry_title must be provided when chemistry_metrics are attached.")
        if self.fidelity_metrics and not self.fidelity_title.strip():
            issues.append("fidelity_title must be provided when fidelity_metrics are attached.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "classification": self.classification,
            "title": self.title,
            "summary": self.summary,
            "objective": self.objective,
            "stage": self.stage,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "reference_title": self.reference_title,
            "reference_summary": self.reference_summary,
            "reference_metrics": [metric.to_dict() for metric in self.reference_metrics],
            "structure_title": self.structure_title,
            "structure_summary": self.structure_summary,
            "structure_metrics": [metric.to_dict() for metric in self.structure_metrics],
            "chemistry_title": self.chemistry_title,
            "chemistry_summary": self.chemistry_summary,
            "chemistry_metrics": [metric.to_dict() for metric in self.chemistry_metrics],
            "fidelity_title": self.fidelity_title,
            "fidelity_summary": self.fidelity_summary,
            "fidelity_metrics": [metric.to_dict() for metric in self.fidelity_metrics],
            "metadata": self.metadata.to_dict(),
        }
