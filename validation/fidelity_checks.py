"""Observer-side fidelity checks for trusted-target shadow correction workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, VectorTuple, coerce_vector_block
from physics.forces.composite import ForceEvaluation


def _rms_force_error(left: VectorTuple, right: VectorTuple) -> float:
    component_errors = [
        (left_vector[axis] - right_vector[axis]) ** 2
        for left_vector, right_vector in zip(left, right, strict=True)
        for axis in range(3)
    ]
    if not component_errors:
        return 0.0
    return sqrt(sum(component_errors) / len(component_errors))


def _max_force_component_error(left: VectorTuple, right: VectorTuple) -> float:
    return max(
        abs(left_vector[axis] - right_vector[axis])
        for left_vector, right_vector in zip(left, right, strict=True)
        for axis in range(3)
    )


def _metadata_to_dict(metadata: FrozenMetadata | dict[str, object] | object) -> dict[str, object]:
    if isinstance(metadata, FrozenMetadata):
        return metadata.to_dict()
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


@dataclass(frozen=True, slots=True)
class ReferenceForceTarget(ValidatableComponent):
    """Trusted reference force and energy target used for fidelity comparison."""

    label: str
    potential_energy: float
    forces: VectorTuple
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "forces", coerce_vector_block(self.forces, "forces"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def particle_count(self) -> int:
        return len(self.forces)

    def validate(self) -> tuple[str, ...]:
        if not self.label.strip():
            return ("label must be a non-empty string.",)
        return ()


@dataclass(frozen=True, slots=True)
class FidelityMetric(ValidatableComponent):
    """One baseline-vs-corrected fidelity comparison metric."""

    label: str
    baseline_error: float
    corrected_error: float
    improved: bool
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
        if self.baseline_error < 0.0:
            issues.append("baseline_error must be non-negative.")
        if self.corrected_error < 0.0:
            issues.append("corrected_error must be non-negative.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class FidelityComparisonReport(ValidatableComponent):
    """Structured report comparing baseline and corrected outputs to a trusted target."""

    title: str
    target_label: str
    metrics: tuple[FidelityMetric, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def metric_for(self, label: str) -> FidelityMetric:
        for metric in self.metrics:
            if metric.label == label:
                return metric
        raise KeyError(label)

    def improved_metrics(self) -> tuple[FidelityMetric, ...]:
        return tuple(metric for metric in self.metrics if metric.improved)

    def regressed_metrics(self) -> tuple[FidelityMetric, ...]:
        return tuple(metric for metric in self.metrics if not metric.improved)

    def passed(self) -> bool:
        return not self.regressed_metrics()

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.title.strip():
            issues.append("title must be a non-empty string.")
        if not self.target_label.strip():
            issues.append("target_label must be a non-empty string.")
        if not self.metrics:
            issues.append("metrics must contain at least one fidelity metric.")
        labels = tuple(metric.label for metric in self.metrics)
        if len(labels) != len(set(labels)):
            issues.append("metric labels must be unique.")
        return tuple(issues)


@dataclass(slots=True)
class ShadowFidelityAssessor(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Assess whether shadow corrections move the simulation closer to trusted targets."""

    absolute_tolerance: float = 1.0e-12
    name: str = "shadow_fidelity_assessor"
    classification: str = "[adapted]"

    def describe_role(self) -> str:
        return (
            "Measures whether shadow coarse-grained corrections reduce energy and "
            "force error relative to a trusted target without taking ownership of execution."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "physics/forces/composite.py",
            "qcloud/shadow_correction.py",
            "forcefields/trusted_sources.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/shadow_coarse_grained_fidelity.md",)

    def validate(self) -> tuple[str, ...]:
        if self.absolute_tolerance < 0.0:
            return ("absolute_tolerance must be non-negative.",)
        return ()

    def assess(
        self,
        *,
        target: ReferenceForceTarget,
        baseline: ForceEvaluation,
        corrected: ForceEvaluation,
        title: str = "Shadow Fidelity Comparison",
    ) -> FidelityComparisonReport:
        particle_count = target.particle_count()
        if len(baseline.forces) != particle_count or len(corrected.forces) != particle_count:
            raise ContractValidationError("baseline and corrected force blocks must match the target particle count.")

        energy_baseline_error = abs(baseline.potential_energy - target.potential_energy)
        energy_corrected_error = abs(corrected.potential_energy - target.potential_energy)
        force_rms_baseline_error = _rms_force_error(baseline.forces, target.forces)
        force_rms_corrected_error = _rms_force_error(corrected.forces, target.forces)
        max_force_baseline_error = _max_force_component_error(baseline.forces, target.forces)
        max_force_corrected_error = _max_force_component_error(corrected.forces, target.forces)

        metrics = (
            FidelityMetric(
                label="energy_absolute_error",
                baseline_error=energy_baseline_error,
                corrected_error=energy_corrected_error,
                improved=energy_corrected_error <= energy_baseline_error + self.absolute_tolerance,
            ),
            FidelityMetric(
                label="force_rms_error",
                baseline_error=force_rms_baseline_error,
                corrected_error=force_rms_corrected_error,
                improved=force_rms_corrected_error <= force_rms_baseline_error + self.absolute_tolerance,
            ),
            FidelityMetric(
                label="max_force_component_error",
                baseline_error=max_force_baseline_error,
                corrected_error=max_force_corrected_error,
                improved=max_force_corrected_error <= max_force_baseline_error + self.absolute_tolerance,
            ),
        )
        return FidelityComparisonReport(
            title=title,
            target_label=target.label,
            metrics=metrics,
            metadata=FrozenMetadata(
                {
                    "particle_count": particle_count,
                    "baseline_metadata": _metadata_to_dict(baseline.metadata),
                    "corrected_metadata": _metadata_to_dict(corrected.metadata),
                }
            ),
        )
