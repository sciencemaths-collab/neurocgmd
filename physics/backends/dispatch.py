"""Kernel dispatch boundary for backend-neutral physics execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata
from optimization.backend_registry import AccelerationBackend, BackendRegistry, BackendSelection
from physics.backends.contracts import PhysicsBackend
from physics.backends.reference_backend import ReferenceComputeBackend


@dataclass(frozen=True, slots=True)
class KernelDispatchRequest(ValidatableComponent):
    """One backend-selection request for a physics kernel site."""

    target_component: str
    required_capabilities: tuple[str, ...] = ()
    preferred_backend: str | None = None
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_capabilities", tuple(dict.fromkeys(self.required_capabilities)))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.target_component.strip():
            issues.append("target_component must be a non-empty string.")
        if self.preferred_backend is not None and not self.preferred_backend.strip():
            issues.append("preferred_backend must be a non-empty string when provided.")
        if any(not capability.strip() for capability in self.required_capabilities):
            issues.append("required_capabilities must contain only non-empty strings.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ResolvedBackendDispatch(ValidatableComponent):
    """Resolved backend implementation plus the selection record."""

    request: KernelDispatchRequest
    selection: BackendSelection
    backend_name: str
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.backend_name != (self.selection.selected_backend or self.backend_name):
            issues.append("backend_name must agree with the resolved selection.")
        if not self.backend_name.strip():
            issues.append("backend_name must be a non-empty string.")
        return tuple(issues)


@dataclass(slots=True)
class KernelDispatchBoundary(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Resolve stable physics-kernel sites onto concrete backends."""

    backend_registry: BackendRegistry = field(default_factory=lambda: default_kernel_backend_registry())
    backend_implementations: dict[str, PhysicsBackend] = field(default_factory=dict)
    name: str = "kernel_dispatch_boundary"
    classification: str = "[adapted]"

    def __post_init__(self) -> None:
        if not self.backend_implementations:
            self.backend_implementations = {"reference_cpu_backend": ReferenceComputeBackend()}
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Resolves backend-neutral physics kernel sites onto concrete execution backends "
            "without leaking backend ownership into the scientific model layer."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "optimization/backend_registry.py",
            "physics/backends/contracts.py",
            "physics/backends/reference_backend.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/backend_compute_spine.md",)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        implementation_names = set(self.backend_implementations.keys())
        missing = tuple(name for name in self.backend_registry.backend_names() if name not in implementation_names)
        if missing:
            issues.append(f"Backend implementations are missing for: {', '.join(missing)}.")
        return tuple(issues)

    def implementation_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.backend_implementations))

    def resolve(self, request: KernelDispatchRequest) -> tuple[ResolvedBackendDispatch, PhysicsBackend]:
        selection = self.backend_registry.select_backend(
            request.target_component,
            required_capabilities=request.required_capabilities,
            preferred_backend=request.preferred_backend,
        )
        if selection.selected_backend is None:
            raise ContractValidationError(selection.rationale)
        backend = self.backend_implementations.get(selection.selected_backend)
        if backend is None:
            raise ContractValidationError(
                f"Backend {selection.selected_backend!r} is selected but has no implementation registered."
            )
        return (
            ResolvedBackendDispatch(
                request=request,
                selection=selection,
                backend_name=selection.selected_backend,
                metadata=FrozenMetadata(
                    {
                        "target_component": request.target_component,
                        "required_capabilities": request.required_capabilities,
                    }
                ),
            ),
            backend,
        )


def default_kernel_backend_registry() -> BackendRegistry:
    """Return the default backend registry for the new kernel dispatch spine."""
    return BackendRegistry(
        backends=(
            AccelerationBackend(
                name="reference_cpu_backend",
                execution_model="python_loops",
                supported_components=(
                    "physics/kernels",
                    "forcefields/hybrid_engine.py",
                    "validation/backend_parity.py",
                ),
                capabilities=("cpu", "reference", "neighbor_list", "pairwise", "tensor"),
                available=True,
                priority=1,
                metadata={"classification": "[adapted]"},
            ),
        )
    )
