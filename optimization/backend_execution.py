"""Execution planning across CPU, vectorized, and future accelerator backends."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata
from optimization.backend_registry import BackendRegistry, BackendSelection


@dataclass(frozen=True, slots=True)
class BackendExecutionRequest(ValidatableComponent):
    """One backend execution-planning request."""

    target_component: str
    particle_count: int
    pair_count: int = 0
    required_capabilities: tuple[str, ...] = ()
    preferred_backend: str | None = None
    latency_sensitive: bool = False
    differentiation_required: bool = False
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
        if self.particle_count < 0:
            issues.append("particle_count must be non-negative.")
        if self.pair_count < 0:
            issues.append("pair_count must be non-negative.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ExecutionPartition(ValidatableComponent):
    """Planned execution partition for one workload."""

    chunk_count: int
    chunk_size: int
    vector_width: int
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.chunk_count <= 0:
            issues.append("chunk_count must be strictly positive.")
        if self.chunk_size <= 0:
            issues.append("chunk_size must be strictly positive.")
        if self.vector_width <= 0:
            issues.append("vector_width must be strictly positive.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class BackendExecutionPlan(ValidatableComponent):
    """Resolved execution plan for one workload."""

    request: BackendExecutionRequest
    selection: BackendSelection
    execution_mode: str
    partition: ExecutionPartition
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.execution_mode.strip():
            issues.append("execution_mode must be a non-empty string.")
        return tuple(issues)


@dataclass(slots=True)
class BackendExecutionPlanner(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Plan execution mode and partitioning over the stable backend registry."""

    backend_registry: BackendRegistry
    serial_particle_threshold: int = 64
    vectorized_particle_threshold: int = 256
    distributed_pair_threshold: int = 50_000
    default_vector_width: int = 4
    name: str = "backend_execution_planner"
    classification: str = "[adapted]"

    def describe_role(self) -> str:
        return (
            "Translates stable backend selections into execution modes and workload partitions "
            "without forcing the scientific layer to own backend heuristics."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return ("optimization/backend_registry.py", "physics/backends/dispatch.py")

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/backend_compute_spine.md",)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.serial_particle_threshold <= 0:
            issues.append("serial_particle_threshold must be strictly positive.")
        if self.vectorized_particle_threshold <= 0:
            issues.append("vectorized_particle_threshold must be strictly positive.")
        if self.distributed_pair_threshold <= 0:
            issues.append("distributed_pair_threshold must be strictly positive.")
        if self.default_vector_width <= 0:
            issues.append("default_vector_width must be strictly positive.")
        return tuple(issues)

    def plan(self, request: BackendExecutionRequest) -> BackendExecutionPlan:
        required_capabilities = list(request.required_capabilities)
        if request.differentiation_required:
            required_capabilities.append("differentiable")
        if request.pair_count >= self.distributed_pair_threshold:
            required_capabilities.append("distributed_ready")
        elif request.particle_count >= self.vectorized_particle_threshold or request.pair_count >= self.vectorized_particle_threshold:
            required_capabilities.append("vectorized")
        selection = self.backend_registry.select_backend(
            request.target_component,
            required_capabilities=tuple(dict.fromkeys(required_capabilities)),
            preferred_backend=request.preferred_backend,
        )
        selected_backend_name = selection.selected_backend or "unresolved"
        selected_backend = next(
            (backend for backend in self.backend_registry.backends if backend.name == selection.selected_backend),
            None,
        )
        capability_set = set(selected_backend.capabilities) if selected_backend is not None else set()
        if selection.selected_backend is None:
            execution_mode = "unresolved"
        elif "distributed_ready" in capability_set and request.pair_count >= self.distributed_pair_threshold:
            execution_mode = "future_accelerator_partitioned"
        elif "vectorized" in capability_set or request.particle_count >= self.vectorized_particle_threshold:
            execution_mode = "vectorized_cpu"
        else:
            execution_mode = "serial_cpu"

        workload_size = max(request.particle_count, request.pair_count, 1)
        vector_width = self.default_vector_width if execution_mode != "serial_cpu" else 1
        if request.latency_sensitive:
            chunk_count = 1
            chunk_size = workload_size
        else:
            chunk_count = max(1, workload_size // max(self.vectorized_particle_threshold, 1))
            chunk_size = max(1, (workload_size + chunk_count - 1) // chunk_count)

        return BackendExecutionPlan(
            request=request,
            selection=selection,
            execution_mode=execution_mode,
            partition=ExecutionPartition(
                chunk_count=chunk_count,
                chunk_size=chunk_size,
                vector_width=vector_width,
                metadata=FrozenMetadata({"selected_backend": selected_backend_name}),
            ),
            metadata=FrozenMetadata(
                {
                    "selected_backend": selected_backend_name,
                    "capability_request": tuple(dict.fromkeys(required_capabilities)),
                }
            ),
        )
