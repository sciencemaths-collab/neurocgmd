"""Batch scientific-validation reports for repeated benchmark trajectories."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from benchmarks.baseline_suite import BenchmarkReport
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, coerce_scalar
from validation.fidelity_checks import FidelityComparisonReport
from validation.structure_metrics import StructureComparisonReport


def _parse_float(text: str, *, label: str) -> float:
    try:
        return float(text.strip())
    except ValueError as error:
        raise ContractValidationError(f"Unable to parse {label} as a float: {text!r}") from error


def _parse_fraction(text: str, *, label: str) -> float:
    numerator_text, separator, denominator_text = text.partition("/")
    if separator != "/":
        raise ContractValidationError(f"Unable to parse {label} as a fraction: {text!r}")
    numerator = _parse_float(numerator_text, label=label)
    denominator = _parse_float(denominator_text, label=label)
    if denominator <= 0.0:
        raise ContractValidationError(f"{label} denominator must be positive.")
    return numerator / denominator


def _metric_map(metrics: tuple[object, ...]) -> dict[str, object]:
    return {str(metric.label): metric for metric in metrics}


@dataclass(frozen=True, slots=True)
class ScientificValidationSample(ValidatableComponent):
    """One sampled scientific-validation point from a live or batch trajectory."""

    replicate_index: int
    sample_step: int
    state_step: int
    time: float
    assembly_score: float
    interface_gap: float
    cross_contact_fraction: float
    graph_bridge_count: float
    atomistic_centroid_rmsd: float
    contact_recovery_fraction: float
    dominant_pair_error: float | None
    baseline_energy_absolute_error: float
    shadow_energy_absolute_error: float
    baseline_force_rms_error: float
    shadow_force_rms_error: float
    baseline_max_force_component_error: float
    shadow_max_force_component_error: float
    benchmark_total_seconds: float
    benchmark_case_seconds: FrozenMetadata = field(default_factory=FrozenMetadata)
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", coerce_scalar(self.time, "time"))
        object.__setattr__(self, "assembly_score", coerce_scalar(self.assembly_score, "assembly_score"))
        object.__setattr__(self, "interface_gap", coerce_scalar(self.interface_gap, "interface_gap"))
        object.__setattr__(self, "cross_contact_fraction", coerce_scalar(self.cross_contact_fraction, "cross_contact_fraction"))
        object.__setattr__(self, "graph_bridge_count", coerce_scalar(self.graph_bridge_count, "graph_bridge_count"))
        object.__setattr__(self, "atomistic_centroid_rmsd", coerce_scalar(self.atomistic_centroid_rmsd, "atomistic_centroid_rmsd"))
        object.__setattr__(self, "contact_recovery_fraction", coerce_scalar(self.contact_recovery_fraction, "contact_recovery_fraction"))
        if self.dominant_pair_error is not None:
            object.__setattr__(self, "dominant_pair_error", coerce_scalar(self.dominant_pair_error, "dominant_pair_error"))
        object.__setattr__(
            self,
            "baseline_energy_absolute_error",
            coerce_scalar(self.baseline_energy_absolute_error, "baseline_energy_absolute_error"),
        )
        object.__setattr__(
            self,
            "shadow_energy_absolute_error",
            coerce_scalar(self.shadow_energy_absolute_error, "shadow_energy_absolute_error"),
        )
        object.__setattr__(self, "baseline_force_rms_error", coerce_scalar(self.baseline_force_rms_error, "baseline_force_rms_error"))
        object.__setattr__(self, "shadow_force_rms_error", coerce_scalar(self.shadow_force_rms_error, "shadow_force_rms_error"))
        object.__setattr__(
            self,
            "baseline_max_force_component_error",
            coerce_scalar(self.baseline_max_force_component_error, "baseline_max_force_component_error"),
        )
        object.__setattr__(
            self,
            "shadow_max_force_component_error",
            coerce_scalar(self.shadow_max_force_component_error, "shadow_max_force_component_error"),
        )
        object.__setattr__(self, "benchmark_total_seconds", coerce_scalar(self.benchmark_total_seconds, "benchmark_total_seconds"))
        if not isinstance(self.benchmark_case_seconds, FrozenMetadata):
            object.__setattr__(self, "benchmark_case_seconds", FrozenMetadata(self.benchmark_case_seconds))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    @classmethod
    def from_reports(
        cls,
        *,
        replicate_index: int,
        sample_step: int,
        state_step: int,
        time: float,
        progress: Any,
        structure_report: StructureComparisonReport,
        fidelity_report: FidelityComparisonReport,
        benchmark_report: BenchmarkReport,
        metadata: FrozenMetadata | dict[str, object] | None = None,
    ) -> "ScientificValidationSample":
        structure_metrics = _metric_map(structure_report.metrics)
        dominant_metric = structure_metrics.get("Dominant Pair Error")
        benchmark_case_seconds = {
            case.name: case.average_seconds_per_iteration()
            for case in benchmark_report.cases
        }
        return cls(
            replicate_index=replicate_index,
            sample_step=sample_step,
            state_step=state_step,
            time=time,
            assembly_score=progress.assembly_score,
            interface_gap=progress.interface_distance,
            cross_contact_fraction=progress.cross_contact_count / progress.target_contact_count,
            graph_bridge_count=float(progress.graph_bridge_count),
            atomistic_centroid_rmsd=_parse_float(
                structure_metrics["Atomistic Centroid RMSD"].value,
                label="Atomistic Centroid RMSD",
            ),
            contact_recovery_fraction=_parse_fraction(
                structure_metrics["Contact Recovery"].value,
                label="Contact Recovery",
            ),
            dominant_pair_error=(
                _parse_float(dominant_metric.value, label="Dominant Pair Error")
                if dominant_metric is not None
                else None
            ),
            baseline_energy_absolute_error=fidelity_report.metric_for("energy_absolute_error").baseline_error,
            shadow_energy_absolute_error=fidelity_report.metric_for("energy_absolute_error").corrected_error,
            baseline_force_rms_error=fidelity_report.metric_for("force_rms_error").baseline_error,
            shadow_force_rms_error=fidelity_report.metric_for("force_rms_error").corrected_error,
            baseline_max_force_component_error=fidelity_report.metric_for("max_force_component_error").baseline_error,
            shadow_max_force_component_error=fidelity_report.metric_for("max_force_component_error").corrected_error,
            benchmark_total_seconds=benchmark_report.total_elapsed_seconds(),
            benchmark_case_seconds=benchmark_case_seconds,
            metadata=metadata,
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.replicate_index < 0:
            issues.append("replicate_index must be non-negative.")
        if self.sample_step < 0:
            issues.append("sample_step must be non-negative.")
        if self.state_step < 0:
            issues.append("state_step must be non-negative.")
        if self.time < 0.0:
            issues.append("time must be non-negative.")
        for label, value in (
            ("assembly_score", self.assembly_score),
            ("cross_contact_fraction", self.cross_contact_fraction),
            ("contact_recovery_fraction", self.contact_recovery_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                issues.append(f"{label} must be between 0 and 1.")
        for label, value in (
            ("interface_gap", self.interface_gap),
            ("graph_bridge_count", self.graph_bridge_count),
            ("atomistic_centroid_rmsd", self.atomistic_centroid_rmsd),
            ("baseline_energy_absolute_error", self.baseline_energy_absolute_error),
            ("shadow_energy_absolute_error", self.shadow_energy_absolute_error),
            ("baseline_force_rms_error", self.baseline_force_rms_error),
            ("shadow_force_rms_error", self.shadow_force_rms_error),
            ("baseline_max_force_component_error", self.baseline_max_force_component_error),
            ("shadow_max_force_component_error", self.shadow_max_force_component_error),
            ("benchmark_total_seconds", self.benchmark_total_seconds),
        ):
            if value < 0.0:
                issues.append(f"{label} must be non-negative.")
        if self.dominant_pair_error is not None and self.dominant_pair_error < 0.0:
            issues.append("dominant_pair_error must be non-negative when provided.")
        for key, value in self.benchmark_case_seconds.to_dict().items():
            if float(value) < 0.0:
                issues.append(f"benchmark case {key!r} must be non-negative.")
        return tuple(issues)

    def energy_improved(self) -> bool:
        return self.shadow_energy_absolute_error <= self.baseline_energy_absolute_error

    def force_rms_improved(self) -> bool:
        return self.shadow_force_rms_error <= self.baseline_force_rms_error

    def max_force_component_improved(self) -> bool:
        return self.shadow_max_force_component_error <= self.baseline_max_force_component_error

    def all_shadow_metrics_improved(self) -> bool:
        return self.energy_improved() and self.force_rms_improved() and self.max_force_component_improved()

    def to_dict(self) -> dict[str, object]:
        return {
            "replicate_index": self.replicate_index,
            "sample_step": self.sample_step,
            "state_step": self.state_step,
            "time": self.time,
            "assembly_score": self.assembly_score,
            "interface_gap": self.interface_gap,
            "cross_contact_fraction": self.cross_contact_fraction,
            "graph_bridge_count": self.graph_bridge_count,
            "atomistic_centroid_rmsd": self.atomistic_centroid_rmsd,
            "contact_recovery_fraction": self.contact_recovery_fraction,
            "dominant_pair_error": self.dominant_pair_error,
            "baseline_energy_absolute_error": self.baseline_energy_absolute_error,
            "shadow_energy_absolute_error": self.shadow_energy_absolute_error,
            "baseline_force_rms_error": self.baseline_force_rms_error,
            "shadow_force_rms_error": self.shadow_force_rms_error,
            "baseline_max_force_component_error": self.baseline_max_force_component_error,
            "shadow_max_force_component_error": self.shadow_max_force_component_error,
            "benchmark_total_seconds": self.benchmark_total_seconds,
            "benchmark_case_seconds": self.benchmark_case_seconds.to_dict(),
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ValidationSeriesPoint(ValidatableComponent):
    """Aggregate point with mean/min/max bands across replicate samples."""

    step: int
    mean: float
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", coerce_scalar(self.mean, "mean"))
        object.__setattr__(self, "minimum", coerce_scalar(self.minimum, "minimum"))
        object.__setattr__(self, "maximum", coerce_scalar(self.maximum, "maximum"))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.step < 0:
            issues.append("step must be non-negative.")
        tolerance = 1.0e-12
        if self.minimum > self.mean + tolerance or self.mean > self.maximum + tolerance:
            issues.append("minimum <= mean <= maximum must hold for each series point.")
        return tuple(issues)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "step": self.step,
            "mean": self.mean,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class ValidationSeries(ValidatableComponent):
    """Named aggregate series ready for rendering or later comparison."""

    label: str
    unit: str
    points: tuple[ValidationSeriesPoint, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(self.points))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.label.strip():
            issues.append("label must be a non-empty string.")
        if not self.points:
            issues.append("points must contain at least one item.")
        steps = tuple(point.step for point in self.points)
        if steps != tuple(sorted(steps)):
            issues.append("series points must be sorted by step.")
        if len(steps) != len(set(steps)):
            issues.append("series points must have unique steps.")
        return tuple(issues)

    def point_for(self, step: int) -> ValidationSeriesPoint:
        for point in self.points:
            if point.step == step:
                return point
        raise KeyError(step)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "unit": self.unit,
            "points": [point.to_dict() for point in self.points],
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ScientificValidationSummary(ValidatableComponent):
    """Compact summary extracted from one scientific-validation report."""

    final_assembly_score_mean: float
    final_atomistic_centroid_rmsd_mean: float
    final_contact_recovery_mean: float
    energy_improvement_rate: float
    force_rms_improvement_rate: float
    max_force_component_improvement_rate: float
    full_shadow_improvement_rate: float
    mean_benchmark_total_seconds: float
    mean_benchmark_case_seconds: FrozenMetadata = field(default_factory=FrozenMetadata)
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        for field_name in (
            "final_assembly_score_mean",
            "final_atomistic_centroid_rmsd_mean",
            "final_contact_recovery_mean",
            "energy_improvement_rate",
            "force_rms_improvement_rate",
            "max_force_component_improvement_rate",
            "full_shadow_improvement_rate",
            "mean_benchmark_total_seconds",
        ):
            object.__setattr__(self, field_name, coerce_scalar(getattr(self, field_name), field_name))
        if not isinstance(self.mean_benchmark_case_seconds, FrozenMetadata):
            object.__setattr__(self, "mean_benchmark_case_seconds", FrozenMetadata(self.mean_benchmark_case_seconds))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for field_name in (
            "final_assembly_score_mean",
            "final_contact_recovery_mean",
            "energy_improvement_rate",
            "force_rms_improvement_rate",
            "max_force_component_improvement_rate",
            "full_shadow_improvement_rate",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                issues.append(f"{field_name} must be between 0 and 1.")
        if self.final_atomistic_centroid_rmsd_mean < 0.0:
            issues.append("final_atomistic_centroid_rmsd_mean must be non-negative.")
        if self.mean_benchmark_total_seconds < 0.0:
            issues.append("mean_benchmark_total_seconds must be non-negative.")
        for key, value in self.mean_benchmark_case_seconds.to_dict().items():
            if float(value) < 0.0:
                issues.append(f"mean benchmark case {key!r} must be non-negative.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "final_assembly_score_mean": self.final_assembly_score_mean,
            "final_atomistic_centroid_rmsd_mean": self.final_atomistic_centroid_rmsd_mean,
            "final_contact_recovery_mean": self.final_contact_recovery_mean,
            "energy_improvement_rate": self.energy_improvement_rate,
            "force_rms_improvement_rate": self.force_rms_improvement_rate,
            "max_force_component_improvement_rate": self.max_force_component_improvement_rate,
            "full_shadow_improvement_rate": self.full_shadow_improvement_rate,
            "mean_benchmark_total_seconds": self.mean_benchmark_total_seconds,
            "mean_benchmark_case_seconds": self.mean_benchmark_case_seconds.to_dict(),
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ScientificValidationReport(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Structured report for repeated scientific-validation trajectories."""

    scenario_name: str
    replicates: int
    steps_per_replicate: int
    sample_interval: int
    samples: tuple[ScientificValidationSample, ...]
    scientific_series: tuple[ValidationSeries, ...]
    benchmark_series: tuple[ValidationSeries, ...]
    summary: ScientificValidationSummary
    title: str = "Scientific Validation Report"
    classification: str = "[hybrid]"
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "scientific_series", tuple(self.scientific_series))
        object.__setattr__(self, "benchmark_series", tuple(self.benchmark_series))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Aggregates repeated benchmark trajectories into scientific and "
            "architecture-level validation plots without changing subsystem ownership."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "scripts/live_dashboard.py",
            "validation/structure_metrics.py",
            "validation/fidelity_checks.py",
            "benchmarks/baseline_suite.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/scientific_validation_reporting.md",)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.scenario_name.strip():
            issues.append("scenario_name must be a non-empty string.")
        if self.replicates <= 0:
            issues.append("replicates must be strictly positive.")
        if self.steps_per_replicate < 0:
            issues.append("steps_per_replicate must be non-negative.")
        if self.sample_interval <= 0:
            issues.append("sample_interval must be strictly positive.")
        if not self.samples:
            issues.append("samples must contain at least one sample.")
        scientific_labels = tuple(series.label for series in self.scientific_series)
        benchmark_labels = tuple(series.label for series in self.benchmark_series)
        if len(scientific_labels) != len(set(scientific_labels)):
            issues.append("scientific_series labels must be unique.")
        if len(benchmark_labels) != len(set(benchmark_labels)):
            issues.append("benchmark_series labels must be unique.")
        return tuple(issues)

    def series_for_label(self, label: str) -> ValidationSeries:
        for series in self.scientific_series + self.benchmark_series:
            if series.label == label:
                return series
        raise KeyError(label)

    def final_samples(self) -> tuple[ScientificValidationSample, ...]:
        final_step = max(sample.sample_step for sample in self.samples)
        return tuple(sample for sample in self.samples if sample.sample_step == final_step)

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "scenario_name": self.scenario_name,
            "classification": self.classification,
            "replicates": self.replicates,
            "steps_per_replicate": self.steps_per_replicate,
            "sample_interval": self.sample_interval,
            "samples": [sample.to_dict() for sample in self.samples],
            "scientific_series": [series.to_dict() for series in self.scientific_series],
            "benchmark_series": [series.to_dict() for series in self.benchmark_series],
            "summary": self.summary.to_dict(),
            "metadata": self.metadata.to_dict(),
        }


@dataclass(slots=True)
class ScientificValidationRunner(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Run repeated trajectory sampling and aggregate it into a validation report."""

    replicates: int = 3
    steps_per_replicate: int = 80
    sample_interval: int = 4
    name: str = "scientific_validation_runner"
    classification: str = "[hybrid]"

    def describe_role(self) -> str:
        return (
            "Runs repeated benchmark trajectories, samples scientific fidelity over time, "
            "and aggregates results into rendering-ready validation series."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "validation/structure_metrics.py",
            "validation/fidelity_checks.py",
            "benchmarks/baseline_suite.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/scientific_validation_reporting.md",)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.replicates <= 0:
            issues.append("replicates must be strictly positive.")
        if self.steps_per_replicate < 0:
            issues.append("steps_per_replicate must be non-negative.")
        if self.sample_interval <= 0:
            issues.append("sample_interval must be strictly positive.")
        return tuple(issues)

    def run(
        self,
        *,
        scenario_name: str,
        context_factory: Callable[[int], object],
        advance_context: Callable[[object, int], None],
        sample_context: Callable[[object, int, int], ScientificValidationSample],
        metadata: FrozenMetadata | dict[str, object] | None = None,
        title: str = "Scientific Validation Report",
    ) -> ScientificValidationReport:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))
        samples: list[ScientificValidationSample] = []
        sample_steps = self._sample_steps()
        for replicate_index in range(self.replicates):
            context = context_factory(replicate_index)
            advanced_steps = 0
            for sample_step in sample_steps:
                delta = sample_step - advanced_steps
                if delta > 0:
                    advance_context(context, delta)
                    advanced_steps = sample_step
                samples.append(sample_context(context, replicate_index, sample_step))

        scientific_series = (
            self._build_series("Assembly Score", "", samples, lambda sample: sample.assembly_score),
            self._build_series("Interface Gap", "reduced", samples, lambda sample: sample.interface_gap),
            self._build_series("Contact Recovery", "fraction", samples, lambda sample: sample.contact_recovery_fraction),
            self._build_series("Atomistic Centroid RMSD", "angstrom_like", samples, lambda sample: sample.atomistic_centroid_rmsd),
            self._build_series(
                "Energy Error Baseline",
                "reduced",
                samples,
                lambda sample: sample.baseline_energy_absolute_error,
            ),
            self._build_series(
                "Energy Error Shadow",
                "reduced",
                samples,
                lambda sample: sample.shadow_energy_absolute_error,
            ),
            self._build_series(
                "Force RMS Error Baseline",
                "reduced",
                samples,
                lambda sample: sample.baseline_force_rms_error,
            ),
            self._build_series(
                "Force RMS Error Shadow",
                "reduced",
                samples,
                lambda sample: sample.shadow_force_rms_error,
            ),
            self._build_series(
                "Max Force Error Baseline",
                "reduced",
                samples,
                lambda sample: sample.baseline_max_force_component_error,
            ),
            self._build_series(
                "Max Force Error Shadow",
                "reduced",
                samples,
                lambda sample: sample.shadow_max_force_component_error,
            ),
            self._build_series(
                "Benchmark Total Time",
                "seconds",
                samples,
                lambda sample: sample.benchmark_total_seconds,
            ),
        )
        benchmark_case_labels = sorted(
            {
                label
                for sample in samples
                for label in sample.benchmark_case_seconds.to_dict()
            }
        )
        benchmark_series = tuple(
            self._build_series(
                f"{label.replace('_', ' ').title()} Time",
                "seconds",
                samples,
                lambda sample, benchmark_label=label: float(sample.benchmark_case_seconds.to_dict().get(benchmark_label, 0.0)),
            )
            for label in benchmark_case_labels
        )
        summary = self._build_summary(samples)
        return ScientificValidationReport(
            title=title,
            scenario_name=scenario_name,
            replicates=self.replicates,
            steps_per_replicate=self.steps_per_replicate,
            sample_interval=self.sample_interval,
            samples=tuple(samples),
            scientific_series=scientific_series,
            benchmark_series=benchmark_series,
            summary=summary,
            metadata=FrozenMetadata(metadata),
        )

    def _sample_steps(self) -> tuple[int, ...]:
        sample_steps = list(range(0, self.steps_per_replicate + 1, self.sample_interval))
        if sample_steps[-1] != self.steps_per_replicate:
            sample_steps.append(self.steps_per_replicate)
        return tuple(sample_steps)

    def _build_series(
        self,
        label: str,
        unit: str,
        samples: list[ScientificValidationSample],
        accessor: Callable[[ScientificValidationSample], float],
    ) -> ValidationSeries:
        bucketed: dict[int, list[float]] = defaultdict(list)
        for sample in samples:
            bucketed[sample.sample_step].append(accessor(sample))
        points = tuple(
            ValidationSeriesPoint(
                step=step,
                mean=fmean(values),
                minimum=min(values),
                maximum=max(values),
            )
            for step, values in sorted(bucketed.items())
        )
        return ValidationSeries(label=label, unit=unit, points=points)

    def _build_summary(self, samples: list[ScientificValidationSample]) -> ScientificValidationSummary:
        final_step = max(sample.sample_step for sample in samples)
        final_samples = [sample for sample in samples if sample.sample_step == final_step]
        benchmark_case_labels = sorted(
            {
                label
                for sample in samples
                for label in sample.benchmark_case_seconds.to_dict()
            }
        )
        mean_benchmark_case_seconds = {
            label: fmean(
                float(sample.benchmark_case_seconds.to_dict().get(label, 0.0))
                for sample in samples
            )
            for label in benchmark_case_labels
        }
        return ScientificValidationSummary(
            final_assembly_score_mean=fmean(sample.assembly_score for sample in final_samples),
            final_atomistic_centroid_rmsd_mean=fmean(sample.atomistic_centroid_rmsd for sample in final_samples),
            final_contact_recovery_mean=fmean(sample.contact_recovery_fraction for sample in final_samples),
            energy_improvement_rate=fmean(1.0 if sample.energy_improved() else 0.0 for sample in samples),
            force_rms_improvement_rate=fmean(1.0 if sample.force_rms_improved() else 0.0 for sample in samples),
            max_force_component_improvement_rate=fmean(
                1.0 if sample.max_force_component_improved() else 0.0 for sample in samples
            ),
            full_shadow_improvement_rate=fmean(
                1.0 if sample.all_shadow_metrics_improved() else 0.0 for sample in samples
            ),
            mean_benchmark_total_seconds=fmean(sample.benchmark_total_seconds for sample in samples),
            mean_benchmark_case_seconds=mean_benchmark_case_seconds,
            metadata={"sample_count": len(samples), "final_step": final_step},
        )


__all__ = [
    "ScientificValidationReport",
    "ScientificValidationRunner",
    "ScientificValidationSample",
    "ScientificValidationSummary",
    "ValidationSeries",
    "ValidationSeriesPoint",
]
