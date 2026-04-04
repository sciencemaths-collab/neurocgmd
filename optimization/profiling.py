"""Profiling helpers for stable timing measurements of platform call sites."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, coerce_scalar

ProfileMetadataFactory = Callable[[object | None], FrozenMetadata | dict[str, object] | None]


@dataclass(frozen=True, slots=True)
class ProfiledOperation(ValidatableComponent):
    """One named callable that should be profiled as part of a report."""

    name: str
    operation: Callable[[], object]
    metadata_factory: ProfileMetadataFactory | None = None

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name.strip():
            issues.append("name must be a non-empty string.")
        if not callable(self.operation):
            issues.append("operation must be callable.")
        if self.metadata_factory is not None and not callable(self.metadata_factory):
            issues.append("metadata_factory must be callable when provided.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ProfilingMeasurement(ValidatableComponent):
    """Raw timing samples and derived statistics for one named operation."""

    name: str
    samples: tuple[float, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "samples",
            tuple(coerce_scalar(sample, f"samples[{index}]") for index, sample in enumerate(self.samples)),
        )
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def repeats(self) -> int:
        return len(self.samples)

    def total_seconds(self) -> float:
        return sum(self.samples)

    def average_seconds(self) -> float:
        return self.total_seconds() / self.repeats()

    def min_seconds(self) -> float:
        return min(self.samples)

    def max_seconds(self) -> float:
        return max(self.samples)

    def iterations_per_second(self) -> float:
        total = self.total_seconds()
        if total == 0.0:
            return float("inf")
        return self.repeats() / total

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name.strip():
            issues.append("name must be a non-empty string.")
        if not self.samples:
            issues.append("samples must contain at least one timing sample.")
        if any(sample < 0.0 for sample in self.samples):
            issues.append("timing samples must be non-negative.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ProfilingReport(ValidatableComponent):
    """Aggregated profiling measurements for one named report."""

    suite_name: str
    measurements: tuple[ProfilingMeasurement, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "measurements", tuple(self.measurements))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def measurement_names(self) -> tuple[str, ...]:
        return tuple(measurement.name for measurement in self.measurements)

    def measurement_for(self, name: str) -> ProfilingMeasurement:
        for measurement in self.measurements:
            if measurement.name == name:
                return measurement
        raise KeyError(name)

    def total_profiled_seconds(self) -> float:
        return sum(measurement.total_seconds() for measurement in self.measurements)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.suite_name.strip():
            issues.append("suite_name must be a non-empty string.")
        if not self.measurements:
            issues.append("measurements must contain at least one profiling measurement.")
        if len(self.measurement_names()) != len(set(self.measurement_names())):
            issues.append("profiling measurement names must be unique within one report.")
        return tuple(issues)


@dataclass(slots=True)
class ExecutionProfiler(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Profile stable call sites without taking ownership of execution logic."""

    default_repeats: int = 5
    default_warmup_runs: int = 1
    name: str = "execution_profiler"
    classification: str = "[adapted]"

    def describe_role(self) -> str:
        return (
            "Collects repeatable timing samples for stable component boundaries so "
            "later optimization work can compare changes without mutating subsystem ownership."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "physics/forces/composite.py",
            "integrators/base.py",
            "ml/residual_model.py",
            "benchmarks/baseline_suite.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/performance_optimization_and_scaling.md",
            "docs/sections/section_15_performance_optimization_and_scaling_hooks.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.default_repeats <= 0:
            issues.append("default_repeats must be strictly positive.")
        if self.default_warmup_runs < 0:
            issues.append("default_warmup_runs must be non-negative.")
        return tuple(issues)

    def _resolve_repeats(self, repeats: int | None) -> int:
        resolved = self.default_repeats if repeats is None else repeats
        if resolved <= 0:
            raise ContractValidationError("Profiling repeats must be strictly positive.")
        return resolved

    def _resolve_warmup_runs(self, warmup_runs: int | None) -> int:
        resolved = self.default_warmup_runs if warmup_runs is None else warmup_runs
        if resolved < 0:
            raise ContractValidationError("warmup_runs must be non-negative.")
        return resolved

    def profile_operation(
        self,
        *,
        name: str,
        operation: Callable[[], object],
        repeats: int | None = None,
        warmup_runs: int | None = None,
        metadata_factory: ProfileMetadataFactory | None = None,
    ) -> ProfilingMeasurement:
        resolved_repeats = self._resolve_repeats(repeats)
        resolved_warmup_runs = self._resolve_warmup_runs(warmup_runs)

        last_result: object | None = None
        for _ in range(resolved_warmup_runs):
            last_result = operation()

        samples: list[float] = []
        for _ in range(resolved_repeats):
            started_at = perf_counter()
            last_result = operation()
            samples.append(perf_counter() - started_at)

        metadata = metadata_factory(last_result) if metadata_factory is not None else {}
        return ProfilingMeasurement(
            name=name,
            samples=tuple(samples),
            metadata=metadata if isinstance(metadata, FrozenMetadata) else FrozenMetadata(metadata),
        )

    def profile_report(
        self,
        *,
        suite_name: str,
        operations: Sequence[ProfiledOperation],
        repeats: int | None = None,
        warmup_runs: int | None = None,
        metadata: FrozenMetadata | dict[str, object] | None = None,
    ) -> ProfilingReport:
        measurements = tuple(
            self.profile_operation(
                name=operation.name,
                operation=operation.operation,
                repeats=repeats,
                warmup_runs=warmup_runs,
                metadata_factory=operation.metadata_factory,
            )
            for operation in operations
        )
        return ProfilingReport(
            suite_name=suite_name,
            measurements=measurements,
            metadata=metadata if isinstance(metadata, FrozenMetadata) else FrozenMetadata(metadata),
        )
