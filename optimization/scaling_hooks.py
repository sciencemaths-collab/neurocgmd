"""Scaling-oriented extension points for future backend-aware execution paths."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, coerce_scalar
from optimization.backend_registry import BackendSelection


@dataclass(frozen=True, slots=True)
class ScalingWorkload(ValidatableComponent):
    """Observed workload shape for one performance-sensitive component."""

    target_component: str
    particle_count: int = 0
    adaptive_edge_count: int = 0
    replay_batch_size: int = 0
    requested_parallelism: int = 1
    memory_pressure_fraction: float = 0.0
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_pressure_fraction", coerce_scalar(self.memory_pressure_fraction))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.target_component.strip():
            issues.append("target_component must be a non-empty string.")
        if self.particle_count < 0:
            issues.append("particle_count must be non-negative.")
        if self.adaptive_edge_count < 0:
            issues.append("adaptive_edge_count must be non-negative.")
        if self.replay_batch_size < 0:
            issues.append("replay_batch_size must be non-negative.")
        if self.requested_parallelism <= 0:
            issues.append("requested_parallelism must be strictly positive.")
        if not 0.0 <= self.memory_pressure_fraction <= 1.0:
            issues.append("memory_pressure_fraction must be between 0.0 and 1.0 inclusive.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ScalingDirective(ValidatableComponent):
    """One explicit scaling recommendation emitted by a hook."""

    label: str
    action: str
    value: str
    rationale: str
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
        if not self.action.strip():
            issues.append("action must be a non-empty string.")
        if not self.value.strip():
            issues.append("value must be a non-empty string.")
        if not self.rationale.strip():
            issues.append("rationale must be a non-empty string.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ScalingHookResult(ValidatableComponent):
    """One hook's contribution to an aggregated scaling plan."""

    hook_name: str
    recommended_parallelism: int
    recommended_backend: str | None = None
    directives: tuple[ScalingDirective, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "directives", tuple(self.directives))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.hook_name.strip():
            issues.append("hook_name must be a non-empty string.")
        if self.recommended_parallelism <= 0:
            issues.append("recommended_parallelism must be strictly positive.")
        if self.recommended_backend is not None and not self.recommended_backend.strip():
            issues.append("recommended_backend must be a non-empty string when provided.")
        labels = tuple(directive.label for directive in self.directives)
        if len(labels) != len(set(labels)):
            issues.append("directive labels must be unique within one hook result.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ScalingPlan(ValidatableComponent):
    """Aggregated scaling guidance for one workload observation."""

    target_component: str
    recommended_parallelism: int
    recommended_backend: str | None = None
    directives: tuple[ScalingDirective, ...] = ()
    triggered_hooks: tuple[str, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "directives", tuple(self.directives))
        object.__setattr__(self, "triggered_hooks", tuple(self.triggered_hooks))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def directive_labels(self) -> tuple[str, ...]:
        return tuple(directive.label for directive in self.directives)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.target_component.strip():
            issues.append("target_component must be a non-empty string.")
        if self.recommended_parallelism <= 0:
            issues.append("recommended_parallelism must be strictly positive.")
        if self.recommended_backend is not None and not self.recommended_backend.strip():
            issues.append("recommended_backend must be a non-empty string when provided.")
        if len(self.directive_labels()) != len(set(self.directive_labels())):
            issues.append("directive labels must be unique within one scaling plan.")
        if len(self.triggered_hooks) != len(set(self.triggered_hooks)):
            issues.append("triggered_hooks must be unique.")
        return tuple(issues)


@runtime_checkable
class ScalingHook(Protocol):
    """Protocol implemented by scaling hooks that emit explicit plan fragments."""

    name: str
    classification: str

    def target_components(self) -> Sequence[str]:
        """Return component prefixes the hook can evaluate."""

    def evaluate(
        self,
        workload: ScalingWorkload,
        *,
        backend_selection: BackendSelection | None = None,
    ) -> ScalingHookResult | None:
        """Return a plan fragment, or `None` when the hook does not trigger."""


@dataclass(frozen=True, slots=True)
class ThresholdScalingHook(ArchitecturalComponent, ValidatableComponent):
    """A conservative threshold-based scaling hook for future acceleration paths."""

    name: str
    component_prefixes: tuple[str, ...]
    particle_threshold: int | None = None
    edge_threshold: int | None = None
    replay_batch_threshold: int | None = None
    memory_pressure_threshold: float | None = None
    parallelism_step: int = 2
    max_parallelism: int = 8
    preferred_backend: str | None = None
    classification: str = "[adapted]"
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_prefixes", tuple(self.component_prefixes))
        if self.memory_pressure_threshold is not None:
            object.__setattr__(self, "memory_pressure_threshold", coerce_scalar(self.memory_pressure_threshold))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Applies transparent workload thresholds to recommend chunking, "
            "parallelism, and backend preferences without mutating execution ownership."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return ("optimization/backend_registry.py",)

    def target_components(self) -> tuple[str, ...]:
        return self.component_prefixes

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name.strip():
            issues.append("name must be a non-empty string.")
        if any(not prefix.strip() for prefix in self.component_prefixes):
            issues.append("component_prefixes must contain only non-empty strings.")
        if not any(
            threshold is not None
            for threshold in (
                self.particle_threshold,
                self.edge_threshold,
                self.replay_batch_threshold,
                self.memory_pressure_threshold,
            )
        ):
            issues.append("at least one threshold must be configured.")
        for threshold_name, threshold_value in (
            ("particle_threshold", self.particle_threshold),
            ("edge_threshold", self.edge_threshold),
            ("replay_batch_threshold", self.replay_batch_threshold),
        ):
            if threshold_value is not None and threshold_value <= 0:
                issues.append(f"{threshold_name} must be strictly positive when provided.")
        if self.memory_pressure_threshold is not None and not 0.0 <= self.memory_pressure_threshold <= 1.0:
            issues.append("memory_pressure_threshold must be between 0.0 and 1.0 inclusive.")
        if self.parallelism_step <= 0:
            issues.append("parallelism_step must be strictly positive.")
        if self.max_parallelism <= 0:
            issues.append("max_parallelism must be strictly positive.")
        if self.preferred_backend is not None and not self.preferred_backend.strip():
            issues.append("preferred_backend must be a non-empty string when provided.")
        return tuple(issues)

    def _matches_component(self, target_component: str) -> bool:
        return any(
            target_component == prefix
            or target_component.startswith(f"{prefix}/")
            or target_component.startswith(f"{prefix}.")
            for prefix in self.component_prefixes
        )

    def evaluate(
        self,
        workload: ScalingWorkload,
        *,
        backend_selection: BackendSelection | None = None,
    ) -> ScalingHookResult | None:
        if not self._matches_component(workload.target_component):
            return None

        directives: list[ScalingDirective] = []
        recommended_parallelism = workload.requested_parallelism
        triggered = False

        if self.particle_threshold is not None and workload.particle_count >= self.particle_threshold:
            increments = max(1, workload.particle_count // self.particle_threshold)
            recommended_parallelism = min(
                self.max_parallelism,
                max(recommended_parallelism, workload.requested_parallelism + increments * self.parallelism_step),
            )
            directives.append(
                ScalingDirective(
                    label=f"{self.name}_particle_partitioning",
                    action="increase_parallelism",
                    value=str(recommended_parallelism),
                    rationale="Particle count crossed the configured threshold.",
                    metadata={"particle_count": workload.particle_count},
                )
            )
            triggered = True

        if self.edge_threshold is not None and workload.adaptive_edge_count >= self.edge_threshold:
            chunk_count = max(1, workload.adaptive_edge_count // self.edge_threshold)
            recommended_parallelism = min(
                self.max_parallelism,
                max(recommended_parallelism, workload.requested_parallelism + chunk_count * self.parallelism_step),
            )
            directives.append(
                ScalingDirective(
                    label=f"{self.name}_edge_chunking",
                    action="chunk_adaptive_edges",
                    value=str(chunk_count),
                    rationale="Adaptive edge count crossed the configured threshold.",
                    metadata={"adaptive_edge_count": workload.adaptive_edge_count},
                )
            )
            triggered = True

        if self.replay_batch_threshold is not None and workload.replay_batch_size >= self.replay_batch_threshold:
            microbatch_size = max(1, workload.replay_batch_size // max(recommended_parallelism, 1))
            directives.append(
                ScalingDirective(
                    label=f"{self.name}_replay_microbatching",
                    action="enable_microbatching",
                    value=str(microbatch_size),
                    rationale="Replay batch size crossed the configured threshold.",
                    metadata={"replay_batch_size": workload.replay_batch_size},
                )
            )
            triggered = True

        if (
            self.memory_pressure_threshold is not None
            and workload.memory_pressure_fraction >= self.memory_pressure_threshold
        ):
            directives.append(
                ScalingDirective(
                    label=f"{self.name}_memory_guardrail",
                    action="stream_state_windows",
                    value="enabled",
                    rationale="Observed memory pressure crossed the configured threshold.",
                    metadata={"memory_pressure_fraction": workload.memory_pressure_fraction},
                )
            )
            triggered = True

        if not triggered:
            return None

        recommended_backend: str | None = None
        if self.preferred_backend is not None:
            recommended_backend = self.preferred_backend
            directives.append(
                ScalingDirective(
                    label=f"{self.name}_backend_preference",
                    action="prefer_backend",
                    value=self.preferred_backend,
                    rationale="The hook declares a preferred backend for this workload class.",
                )
            )
        elif backend_selection is not None and backend_selection.selected_backend is not None:
            recommended_backend = backend_selection.selected_backend

        return ScalingHookResult(
            hook_name=self.name,
            recommended_parallelism=recommended_parallelism,
            recommended_backend=recommended_backend,
            directives=tuple(directives),
            metadata={
                "target_component": workload.target_component,
                "backend_selection": backend_selection.selected_backend if backend_selection is not None else None,
            },
        )


@dataclass(slots=True)
class ScalingHookManager(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Aggregate scaling hooks into one explicit plan for a workload."""

    hooks: tuple[ScalingHook, ...] = ()
    name: str = "scaling_hook_manager"
    classification: str = "[adapted]"

    def __post_init__(self) -> None:
        self.hooks = tuple(self.hooks)
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Collects workload-sensitive scaling hints from transparent hooks so "
            "future optimization paths can scale beneath the simulation and dashboard layers."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "optimization/backend_registry.py",
            "physics/forces/composite.py",
            "graph/graph_manager.py",
            "memory/replay_buffer.py",
            "ml/residual_model.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/performance_optimization_and_scaling.md",
            "docs/sections/section_15_performance_optimization_and_scaling_hooks.md",
        )

    def validate(self) -> tuple[str, ...]:
        names = self.registered_hook_names()
        if len(names) != len(set(names)):
            return ("hook names must be unique.",)
        return ()

    def registered_hook_names(self) -> tuple[str, ...]:
        return tuple(hook.name for hook in self.hooks)

    def register(self, hook: ScalingHook) -> "ScalingHookManager":
        if hook.name in self.registered_hook_names():
            raise ContractValidationError(f"Scaling hook {hook.name!r} is already registered.")
        return ScalingHookManager(hooks=self.hooks + (hook,), name=self.name, classification=self.classification)

    def evaluate(
        self,
        workload: ScalingWorkload,
        *,
        backend_selection: BackendSelection | None = None,
    ) -> ScalingPlan:
        results = tuple(
            result
            for result in (
                hook.evaluate(workload, backend_selection=backend_selection)
                for hook in self.hooks
            )
            if result is not None
        )

        recommended_parallelism = max(
            (result.recommended_parallelism for result in results),
            default=workload.requested_parallelism,
        )
        recommended_backend = next(
            (result.recommended_backend for result in results if result.recommended_backend is not None),
            backend_selection.selected_backend if backend_selection is not None else None,
        )
        directives = tuple(directive for result in results for directive in result.directives)
        triggered_hooks = tuple(result.hook_name for result in results)

        return ScalingPlan(
            target_component=workload.target_component,
            recommended_parallelism=recommended_parallelism,
            recommended_backend=recommended_backend,
            directives=directives,
            triggered_hooks=triggered_hooks,
            metadata={
                "result_count": len(results),
                "requested_parallelism": workload.requested_parallelism,
                "backend_selection": backend_selection.selected_backend if backend_selection is not None else None,
            },
        )
