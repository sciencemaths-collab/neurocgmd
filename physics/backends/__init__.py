"""Backend-neutral execution contracts and reference implementations."""

from physics.backends.contracts import (
    ArrayFactory,
    NeighborListHook,
    PairInteractionRecord,
    PairwiseExecutionHook,
    PairwiseForceContribution,
    PairwiseKernelInput,
    PairwiseKernelResult,
    PhysicsBackend,
    TensorBlock,
    pair_input_from_neighbor_list,
    tensor_from_vectors,
)
from physics.backends.dispatch import (
    KernelDispatchBoundary,
    KernelDispatchRequest,
    ResolvedBackendDispatch,
    default_kernel_backend_registry,
)
from physics.backends.reference_backend import ReferenceComputeBackend

__all__ = [
    "ArrayFactory",
    "KernelDispatchBoundary",
    "KernelDispatchRequest",
    "NeighborListHook",
    "PairInteractionRecord",
    "PairwiseExecutionHook",
    "PairwiseForceContribution",
    "PairwiseKernelInput",
    "PairwiseKernelResult",
    "PhysicsBackend",
    "ReferenceComputeBackend",
    "ResolvedBackendDispatch",
    "TensorBlock",
    "default_kernel_backend_registry",
    "pair_input_from_neighbor_list",
    "tensor_from_vectors",
]
