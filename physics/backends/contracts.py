"""Backend-neutral tensor and pairwise execution contracts for the physics spine."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from core.exceptions import ContractValidationError
from core.interfaces import ValidatableComponent
from core.types import FrozenMetadata, StateId, Vector3, VectorTuple, coerce_scalar
from physics.neighbor_list import NeighborList


@dataclass(frozen=True, slots=True)
class TensorBlock(ValidatableComponent):
    """Backend-neutral numeric block.

    The values remain plain Python tuples so the contract stays stable even when
    future accelerated backends use NumPy, JAX, or custom kernels underneath.
    """

    name: str
    shape: tuple[int, ...]
    values: tuple[tuple[float, ...], ...]
    layout: str = "dense_row_major"
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(value) for value in self.shape))
        object.__setattr__(
            self,
            "values",
            tuple(tuple(coerce_scalar(component, "tensor_component") for component in row) for row in self.values),
        )
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name.strip():
            issues.append("name must be a non-empty string.")
        if len(self.shape) != 2:
            issues.append("shape must be two-dimensional for the current tensor contract.")
        if any(dimension < 0 for dimension in self.shape):
            issues.append("shape dimensions must be non-negative.")
        if len(self.values) != (self.shape[0] if self.shape else 0):
            issues.append("values row count must match shape[0].")
        if self.shape and any(len(row) != self.shape[1] for row in self.values):
            issues.append("every tensor row must match shape[1].")
        if not self.layout.strip():
            issues.append("layout must be a non-empty string.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class PairInteractionRecord(ValidatableComponent):
    """One pairwise interaction candidate supplied to a backend kernel."""

    particle_index_a: int
    particle_index_b: int
    distance: float
    displacement: Vector3
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "distance", coerce_scalar(self.distance, "distance"))
        object.__setattr__(self, "displacement", tuple(self.displacement))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def normalized_pair(self) -> tuple[int, int]:
        return (self.particle_index_a, self.particle_index_b)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.particle_index_a < 0 or self.particle_index_b < 0:
            issues.append("pair particle indices must be non-negative.")
        if self.particle_index_a >= self.particle_index_b:
            issues.append("pair indices must be canonical with particle_index_a < particle_index_b.")
        if self.distance <= 0.0:
            issues.append("distance must be strictly positive.")
        if len(self.displacement) != 3:
            issues.append("displacement must be three-dimensional.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class PairwiseKernelInput(ValidatableComponent):
    """Prepared pairwise input block for one backend kernel launch."""

    state_id: StateId
    positions: TensorBlock
    pair_records: tuple[PairInteractionRecord, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", StateId(str(self.state_id)))
        object.__setattr__(self, "pair_records", tuple(self.pair_records))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.positions.shape[1] != 3:
            issues.append("positions tensor must have width 3.")
        if any(record.particle_index_b >= self.positions.shape[0] for record in self.pair_records):
            issues.append("pair record references a particle outside the positions tensor.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class PairwiseForceContribution(ValidatableComponent):
    """One force contribution emitted by a pairwise kernel."""

    particle_index: int
    delta_force: Vector3
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta_force", tuple(self.delta_force))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.particle_index < 0:
            issues.append("particle_index must be non-negative.")
        if len(self.delta_force) != 3:
            issues.append("delta_force must be three-dimensional.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class PairwiseKernelResult(ValidatableComponent):
    """Aggregated result from a backend pairwise execution hook."""

    energy_delta: float
    force_contributions: tuple[PairwiseForceContribution, ...]
    evaluated_pair_count: int
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy_delta", coerce_scalar(self.energy_delta, "energy_delta"))
        object.__setattr__(self, "force_contributions", tuple(self.force_contributions))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.evaluated_pair_count < 0:
            issues.append("evaluated_pair_count must be non-negative.")
        affected_particles = tuple(contribution.particle_index for contribution in self.force_contributions)
        if len(affected_particles) != len(set(affected_particles)):
            issues.append("force_contributions must be aggregated by particle index.")
        return tuple(issues)


PairwiseEvaluator = Callable[
    [PairInteractionRecord],
    tuple[float, Sequence[tuple[int, Vector3]], Mapping[str, object] | FrozenMetadata | None],
]


def tensor_from_vectors(
    vectors: VectorTuple,
    *,
    name: str = "vector_block",
    metadata: Mapping[str, object] | FrozenMetadata | None = None,
) -> TensorBlock:
    """Build a stable tensor block from a vector tuple."""
    return TensorBlock(
        name=name,
        shape=(len(vectors), 3),
        values=tuple(tuple(vector) for vector in vectors),
        metadata=metadata if isinstance(metadata, FrozenMetadata) else FrozenMetadata(metadata),
    )


def pair_input_from_neighbor_list(
    *,
    state_id: StateId | str,
    positions: VectorTuple,
    neighbor_list: NeighborList,
    metadata: Mapping[str, object] | FrozenMetadata | None = None,
) -> PairwiseKernelInput:
    """Convert a neighbor list into backend-neutral pairwise input records."""
    pair_records = tuple(
        PairInteractionRecord(
            particle_index_a=pair[0],
            particle_index_b=pair[1],
            distance=neighbor_list.distances[index],
            displacement=(
                positions[pair[1]][0] - positions[pair[0]][0],
                positions[pair[1]][1] - positions[pair[0]][1],
                positions[pair[1]][2] - positions[pair[0]][2],
            ),
        )
        for index, pair in enumerate(neighbor_list.pairs)
    )
    return PairwiseKernelInput(
        state_id=StateId(str(state_id)),
        positions=tensor_from_vectors(positions, name="positions"),
        pair_records=pair_records,
        metadata=metadata if isinstance(metadata, FrozenMetadata) else FrozenMetadata(metadata),
    )


@runtime_checkable
class ArrayFactory(Protocol):
    """Minimal tensor/array creation contract."""

    def tensor_from_vectors(
        self,
        vectors: VectorTuple,
        *,
        name: str = "vector_block",
        metadata: Mapping[str, object] | FrozenMetadata | None = None,
    ) -> TensorBlock:
        """Materialize a backend-neutral tensor block."""

    def zeros(
        self,
        *,
        rows: int,
        columns: int,
        name: str = "zeros",
        metadata: Mapping[str, object] | FrozenMetadata | None = None,
    ) -> TensorBlock:
        """Materialize a zero-filled tensor block."""


@runtime_checkable
class NeighborListHook(Protocol):
    """Backend hook for spatial pair discovery."""

    def build_neighbor_list(
        self,
        positions: VectorTuple,
        *,
        cutoff: float,
        skin: float = 0.0,
        excluded_pairs: frozenset[tuple[int, int]] | None = None,
    ) -> NeighborList:
        """Build a neighbor list for one particle block."""


@runtime_checkable
class PairwiseExecutionHook(Protocol):
    """Backend hook for pairwise kernel execution."""

    def execute_pairwise(
        self,
        kernel_input: PairwiseKernelInput,
        evaluator: PairwiseEvaluator,
    ) -> PairwiseKernelResult:
        """Execute a pairwise evaluator across the supplied input block."""


@runtime_checkable
class PhysicsBackend(ArrayFactory, NeighborListHook, PairwiseExecutionHook, Protocol):
    """Stable backend-neutral execution contract for the physics spine."""

    name: str
    execution_model: str
    capabilities: tuple[str, ...]
