"""Acceleration-backend registry for stable component selection decisions."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata


@dataclass(frozen=True, slots=True)
class AccelerationBackend(ValidatableComponent):
    """One concrete execution backend that may accelerate stable call sites."""

    name: str
    execution_model: str
    supported_components: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    available: bool = True
    priority: int = 0
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "supported_components", tuple(self.supported_components))
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def supports_component(self, target_component: str) -> bool:
        if not self.supported_components:
            return True
        return any(
            target_component == supported or target_component.startswith(f"{supported}/") or target_component.startswith(
                f"{supported}."
            )
            for supported in self.supported_components
        )

    def satisfies(self, required_capabilities: tuple[str, ...] | list[str]) -> bool:
        required = tuple(required_capabilities)
        capability_set = set(self.capabilities)
        return all(capability in capability_set for capability in required)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name.strip():
            issues.append("name must be a non-empty string.")
        if not self.execution_model.strip():
            issues.append("execution_model must be a non-empty string.")
        if self.priority < 0:
            issues.append("priority must be non-negative.")
        if any(not component.strip() for component in self.supported_components):
            issues.append("supported_components must contain only non-empty strings.")
        if len(self.supported_components) != len(set(self.supported_components)):
            issues.append("supported_components must be unique.")
        if any(not capability.strip() for capability in self.capabilities):
            issues.append("capabilities must contain only non-empty strings.")
        if len(self.capabilities) != len(set(self.capabilities)):
            issues.append("capabilities must be unique.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class BackendSelection(ValidatableComponent):
    """Structured backend-selection result for one target component."""

    target_component: str
    selected_backend: str | None
    considered_backends: tuple[str, ...]
    unmet_capabilities: tuple[str, ...] = ()
    rationale: str = ""
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "considered_backends", tuple(self.considered_backends))
        object.__setattr__(self, "unmet_capabilities", tuple(self.unmet_capabilities))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def resolved(self) -> bool:
        return self.selected_backend is not None

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.target_component.strip():
            issues.append("target_component must be a non-empty string.")
        if self.selected_backend is not None and not self.selected_backend.strip():
            issues.append("selected_backend must be a non-empty string when provided.")
        if any(not backend.strip() for backend in self.considered_backends):
            issues.append("considered_backends must contain only non-empty strings.")
        if len(self.considered_backends) != len(set(self.considered_backends)):
            issues.append("considered_backends must be unique.")
        if any(not capability.strip() for capability in self.unmet_capabilities):
            issues.append("unmet_capabilities must contain only non-empty strings.")
        if len(self.unmet_capabilities) != len(set(self.unmet_capabilities)):
            issues.append("unmet_capabilities must be unique.")
        if not self.rationale.strip():
            issues.append("rationale must be a non-empty string.")
        return tuple(issues)


@dataclass(slots=True)
class BackendRegistry(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Persistent registry of available acceleration backends."""

    backends: tuple[AccelerationBackend, ...] = ()
    name: str = "backend_registry"
    classification: str = "[adapted]"

    def __post_init__(self) -> None:
        self.backends = tuple(self.backends)
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Registers stable execution backends and resolves transparent backend "
            "choices for force, integrator, ML, and future scaling pathways."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "physics/forces/composite.py",
            "integrators/base.py",
            "ml/residual_model.py",
            "optimization/scaling_hooks.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/performance_optimization_and_scaling.md",
            "docs/sections/section_15_performance_optimization_and_scaling_hooks.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        names = self.backend_names()
        if len(names) != len(set(names)):
            issues.append("backend names must be unique.")
        return tuple(issues)

    def backend_names(self) -> tuple[str, ...]:
        return tuple(backend.name for backend in self.backends)

    def register(self, backend: AccelerationBackend) -> "BackendRegistry":
        if backend.name in self.backend_names():
            raise ContractValidationError(f"Backend {backend.name!r} is already registered.")
        return BackendRegistry(backends=self.backends + (backend,), name=self.name, classification=self.classification)

    def available_backends(self, target_component: str | None = None) -> tuple[AccelerationBackend, ...]:
        candidates = tuple(backend for backend in self.backends if backend.available)
        if target_component is None:
            return candidates
        return tuple(backend for backend in candidates if backend.supports_component(target_component))

    def select_backend(
        self,
        target_component: str,
        *,
        required_capabilities: tuple[str, ...] | list[str] = (),
        preferred_backend: str | None = None,
    ) -> BackendSelection:
        if not target_component.strip():
            raise ContractValidationError("target_component must be a non-empty string.")

        required = tuple(dict.fromkeys(required_capabilities))
        candidates = self.available_backends(target_component)
        considered = tuple(
            backend.name for backend in sorted(candidates, key=lambda backend: (-backend.priority, backend.name))
        )

        if preferred_backend is not None:
            preferred = next((backend for backend in self.backends if backend.name == preferred_backend), None)
            if preferred is None:
                return BackendSelection(
                    target_component=target_component,
                    selected_backend=None,
                    considered_backends=considered,
                    unmet_capabilities=required,
                    rationale=f"Preferred backend {preferred_backend!r} is not registered.",
                    metadata={"required_capabilities": required},
                )
            if not preferred.available:
                return BackendSelection(
                    target_component=target_component,
                    selected_backend=None,
                    considered_backends=considered,
                    unmet_capabilities=required,
                    rationale=f"Preferred backend {preferred_backend!r} is currently unavailable.",
                    metadata={"required_capabilities": required},
                )
            if not preferred.supports_component(target_component):
                return BackendSelection(
                    target_component=target_component,
                    selected_backend=None,
                    considered_backends=considered,
                    unmet_capabilities=required,
                    rationale=f"Preferred backend {preferred_backend!r} does not support {target_component}.",
                    metadata={"required_capabilities": required},
                )
            if not preferred.satisfies(required):
                missing = tuple(capability for capability in required if capability not in set(preferred.capabilities))
                return BackendSelection(
                    target_component=target_component,
                    selected_backend=None,
                    considered_backends=considered,
                    unmet_capabilities=missing,
                    rationale=f"Preferred backend {preferred_backend!r} is missing required capabilities.",
                    metadata={"required_capabilities": required},
                )
            return BackendSelection(
                target_component=target_component,
                selected_backend=preferred.name,
                considered_backends=considered,
                unmet_capabilities=(),
                rationale=f"Preferred backend {preferred.name!r} satisfies the requested component and capabilities.",
                metadata={
                    "execution_model": preferred.execution_model,
                    "priority": preferred.priority,
                    "required_capabilities": required,
                },
            )

        matching = tuple(backend for backend in candidates if backend.satisfies(required))
        if matching:
            selected = sorted(matching, key=lambda backend: (-backend.priority, backend.name))[0]
            return BackendSelection(
                target_component=target_component,
                selected_backend=selected.name,
                considered_backends=considered,
                unmet_capabilities=(),
                rationale=f"Selected highest-priority available backend compatible with {target_component}.",
                metadata={
                    "execution_model": selected.execution_model,
                    "priority": selected.priority,
                    "required_capabilities": required,
                },
            )

        capability_union = {capability for backend in candidates for capability in backend.capabilities}
        unmet = tuple(capability for capability in required if capability not in capability_union)
        if not candidates:
            rationale = f"No available backend supports {target_component}."
        elif required:
            rationale = "Available backends exist, but none satisfy all required capabilities."
        else:
            rationale = "Available backends exist, but none are eligible for the requested component."
        return BackendSelection(
            target_component=target_component,
            selected_backend=None,
            considered_backends=considered,
            unmet_capabilities=unmet if unmet else required,
            rationale=rationale,
            metadata={"required_capabilities": required},
        )
