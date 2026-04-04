# Performance Optimization and Scaling Architecture

## Section 15 Scope

Section 15 adds the first explicit optimization substrate beneath the scientific,
validation, and dashboard layers. This section is classified as `[adapted]`
because profiling, backend selection, and scaling hints are standard software
engineering practices, but the way they are layered here is tuned to preserve
scientific transparency and subsystem ownership.

## Mathematical Role

Section 15 introduces three optimization-side objects:

- profiling measurements
  - `P = {p_i}`
- backend selections
  - `B_sel = sigma(C, R)`
- scaling plans
  - `S_plan = phi(W, B_sel, H)`

Where:

- `C` is a target component boundary
- `R` is a set of required backend capabilities
- `W` is an observed workload summary
- `H` is the set of registered threshold hooks

These objects are advisory rather than generative. They do not replace the
simulation state, physics, controller, or validation layers. They exist so the
platform can profile and scale stable interfaces without hiding the decisions.

## Foundation Design

- `optimization/profiling.py`
  - `[adapted]`
  - defines `ProfiledOperation`, `ProfilingMeasurement`, `ProfilingReport`, and `ExecutionProfiler`
  - records repeat timing samples with explicit warmup control
- `optimization/backend_registry.py`
  - `[adapted]`
  - defines `AccelerationBackend`, `BackendSelection`, and `BackendRegistry`
  - makes backend choice an explicit structured decision rather than a hidden import-side effect
- `optimization/scaling_hooks.py`
  - `[adapted]`
  - defines `ScalingWorkload`, `ScalingDirective`, `ScalingHookResult`, `ScalingPlan`, `ScalingHook`, `ThresholdScalingHook`, and `ScalingHookManager`
  - allows future scaling policies to stay modular and inspectable

## Core Invariants

- optimization remains downstream of stable scientific interfaces
- profiling does not mutate force evaluators, integrators, or ML models
- backend selection is explicit and serializable through `BackendSelection`
- scaling hooks emit directives and plans, not hidden mutations
- visualization and dashboard code remain consumers of structured reports, not optimization owners
- benchmark timings from Section 13 remain local diagnostics; Section 15 complements them with richer profiling structure rather than replacing them

## Validation Thinking

We test Section 15 by checking:

- profilers collect repeat samples and preserve metadata
- profiling reports keep operation names unique and queryable
- backend selection prefers compatible higher-priority backends unless an explicit valid preference is supplied
- missing capabilities are reported explicitly rather than hidden by fallback behavior
- threshold hooks emit explicit directives only when workload thresholds are crossed
- scaling plans preserve the original requested parallelism when no hook triggers
