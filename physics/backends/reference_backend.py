"""Pure-Python reference backend for the new physics compute spine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from core.types import FrozenMetadata, VectorTuple
from physics.backends.contracts import (
    PairwiseEvaluator,
    PairwiseForceContribution,
    PairwiseKernelInput,
    PairwiseKernelResult,
    TensorBlock,
    tensor_from_vectors,
)
from physics.neighbor_list import NeighborList, NeighborListBuilder


@dataclass(slots=True)
class ReferenceComputeBackend:
    """Deterministic reference backend used by the kernel dispatch boundary."""

    default_skin: float = 0.3
    name: str = "reference_cpu_backend"
    execution_model: str = "python_loops"
    capabilities: tuple[str, ...] = ("cpu", "reference", "neighbor_list", "pairwise", "tensor")

    def tensor_from_vectors(
        self,
        vectors: VectorTuple,
        *,
        name: str = "vector_block",
        metadata: Mapping[str, object] | FrozenMetadata | None = None,
    ) -> TensorBlock:
        return tensor_from_vectors(vectors, name=name, metadata=metadata)

    def zeros(
        self,
        *,
        rows: int,
        columns: int,
        name: str = "zeros",
        metadata: Mapping[str, object] | FrozenMetadata | None = None,
    ) -> TensorBlock:
        return TensorBlock(
            name=name,
            shape=(rows, columns),
            values=tuple(tuple(0.0 for _ in range(columns)) for _ in range(rows)),
            metadata=metadata if isinstance(metadata, FrozenMetadata) else FrozenMetadata(metadata),
        )

    def build_neighbor_list(
        self,
        positions: VectorTuple,
        *,
        cutoff: float,
        skin: float = 0.0,
        excluded_pairs: frozenset[tuple[int, int]] | None = None,
    ) -> NeighborList:
        builder = NeighborListBuilder(cutoff=cutoff, skin=skin or self.default_skin)
        return builder.build(
            positions,
            len(positions),
            excluded_pairs=excluded_pairs,
        )

    def execute_pairwise(
        self,
        kernel_input: PairwiseKernelInput,
        evaluator: PairwiseEvaluator,
    ) -> PairwiseKernelResult:
        accumulated_forces: dict[int, list[float]] = {}
        total_energy = 0.0
        pair_metadata: list[FrozenMetadata] = []

        for record in kernel_input.pair_records:
            energy_delta, contributions, metadata = evaluator(record)
            total_energy += energy_delta
            if metadata is not None:
                pair_metadata.append(metadata if isinstance(metadata, FrozenMetadata) else FrozenMetadata(metadata))
            for particle_index, delta_force in contributions:
                vector = accumulated_forces.setdefault(particle_index, [0.0, 0.0, 0.0])
                for axis, value in enumerate(delta_force):
                    vector[axis] += value

        return PairwiseKernelResult(
            energy_delta=total_energy,
            force_contributions=tuple(
                PairwiseForceContribution(
                    particle_index=particle_index,
                    delta_force=tuple(values),
                )
                for particle_index, values in sorted(accumulated_forces.items())
            ),
            evaluated_pair_count=len(kernel_input.pair_records),
            metadata=FrozenMetadata(
                {
                    "backend": self.name,
                    "execution_model": self.execution_model,
                    "pair_metadata_count": len(pair_metadata),
                }
            ),
        )
