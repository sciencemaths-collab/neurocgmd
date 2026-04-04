"""Performance optimization, backend selection, and scaling hooks."""

from optimization.backend_execution import (
    BackendExecutionPlan,
    BackendExecutionPlanner,
    BackendExecutionRequest,
    ExecutionPartition,
)
from optimization.backend_registry import AccelerationBackend, BackendRegistry, BackendSelection
from optimization.compute_backend import (
    ComputeDevice,
    GPUBackend,
    NumpyStateArrays,
    ParallelNeighborList,
    PerformanceMonitor,
    VectorizedBondForces,
    VectorizedForceEvaluator,
    VectorizedNonbondedForces,
    auto_select_evaluator,
    detect_best_device,
)
from optimization.profiling import ExecutionProfiler, ProfiledOperation, ProfilingMeasurement, ProfilingReport
from optimization.scaling_hooks import (
    ScalingDirective,
    ScalingHook,
    ScalingHookManager,
    ScalingHookResult,
    ScalingPlan,
    ScalingWorkload,
    ThresholdScalingHook,
)

__all__ = [
    "AccelerationBackend",
    "BackendExecutionPlan",
    "BackendExecutionPlanner",
    "BackendExecutionRequest",
    "BackendRegistry",
    "BackendSelection",
    "ComputeDevice",
    "ExecutionPartition",
    "ExecutionProfiler",
    "GPUBackend",
    "NumpyStateArrays",
    "ParallelNeighborList",
    "PerformanceMonitor",
    "ProfiledOperation",
    "ProfilingMeasurement",
    "ProfilingReport",
    "ScalingDirective",
    "ScalingHook",
    "ScalingHookManager",
    "ScalingHookResult",
    "ScalingPlan",
    "ScalingWorkload",
    "ThresholdScalingHook",
    "VectorizedBondForces",
    "VectorizedForceEvaluator",
    "VectorizedNonbondedForces",
    "auto_select_evaluator",
    "detect_best_device",
]
