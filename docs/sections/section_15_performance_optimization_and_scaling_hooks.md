# Section 15: Performance Optimization and Scaling Hooks

## STEP A. Section Name and Objective

Section name: `Performance optimization and scaling hooks`

Objective:

- define stable profiling helpers for performance-sensitive call sites
- define an explicit registry for backend and acceleration selection
- define workload-aware scaling hooks that emit structured directives rather than hidden mutations
- keep optimization beneath the scientific, validation, and visualization layers

Primary classification: `[adapted]`

## STEP B. Mathematical and Architectural Role

Section 15 adds:

- `P`
- `B_sel`
- `S_plan`

Architecturally:

- `optimization/` consumes stable interfaces from force evaluation, integrators, replay, and residual learning
- `profiling.py` measures repeat timing samples without owning execution
- `backend_registry.py` resolves explicit backend choices from registered capabilities
- `scaling_hooks.py` converts workload observations into explicit scaling directives and plans

## STEP C. Folder Structure Created or Extended

- `optimization/`
- `tests/`
- `docs/architecture/`
- `docs/sections/`
- `progress_info/`

## STEP D. Smaller Scripts Added

- `optimization/profiling.py`
- `optimization/backend_registry.py`
- `optimization/scaling_hooks.py`
- `tests/test_optimization_layer.py`
- `docs/architecture/performance_optimization_and_scaling.md`
- `docs/sections/section_15_performance_optimization_and_scaling_hooks.md`

## STEP E. Implementation Notes

- `optimization/profiling.py`
  - defines named operations, raw timing-sample measurements, report aggregation, and a repeat/warmup-aware profiler
  - keeps measurement metadata explicit so future benchmark and CI tooling can persist environment details cleanly
- `optimization/backend_registry.py`
  - defines immutable backend descriptors plus transparent selection results
  - resolves backends by component support, availability, capability requirements, explicit preference, and priority
- `optimization/scaling_hooks.py`
  - defines workload descriptors, directives, per-hook results, and aggregated plans
  - adds `ThresholdScalingHook` as the conservative Section 15 baseline
  - keeps scaling as advisory planning instead of hidden execution ownership

Key design choices:

- Section 15 does not rewrite the Section 5 simulation loop or the Section 13 benchmark suite
- backend choice is represented explicitly in data rather than hidden in global runtime state
- scaling hints are kept separate from AI-control recommendations even though future control policies may consume them
- profiling and scaling stay below the dashboard/export layer so performance work does not leak rendering concerns upward

## STEP F. Validation Strategy

Section 15 validation covers:

- profiling warmup and repeat behavior
- profiling-report aggregation
- backend compatibility, preference, and missing-capability reporting
- threshold-triggered scaling directives and empty-plan behavior when no thresholds are crossed

Commands:

```bash
python3 scripts/validate_scaffold.py
python3 -m unittest discover -s tests
```

## STEP G. Documentation Updated

- `docs/architecture/performance_optimization_and_scaling.md`
- `docs/sections/section_15_performance_optimization_and_scaling_hooks.md`
- `progress_info/*.md`

## STEP H. Progress Files Updated

All mandatory continuity files were updated to reflect Section 15 completion and
the closure of the numbered architecture roadmap.

## STEP I. Handoff Summary

SECTION HANDOFF
- Section completed: Section 15. Performance optimization and scaling hooks
- Purpose: add profiling, backend selection, and scaling-planning interfaces beneath the scientific and dashboard layers
- Files created: optimization modules, optimization tests, architecture docs, section docs, and continuity updates
- Key interfaces: `ExecutionProfiler`, `ProfilingMeasurement`, `BackendRegistry`, `BackendSelection`, `ScalingWorkload`, `ScalingPlan`, `ThresholdScalingHook`, `ScalingHookManager`
- Dependencies: `physics/forces/composite.py`, `integrators/base.py`, `memory/replay_buffer.py`, `ml/residual_model.py`, `benchmarks/baseline_suite.py`
- What now works: repeat timing measurements, explicit backend capability selection, threshold-based scaling plans, and test-covered optimization contracts
- What is still stubbed: real compiled kernels, GPU/distributed runtime adapters, automated backend discovery, and closed-loop scaling execution
- Risks / unresolved items: performance hooks are intentionally advisory at this stage; future acceleration work should preserve explicit reporting rather than hiding behavior inside opaque runtime switches
- Next recommended section: numbered architecture roadmap complete; next work should focus on scientific fidelity, repeated validation, and calibrated comparison workflows
- Important continuity notes: future acceleration work should extend these contracts instead of bypassing them, and the GUI should keep consuming explicit reports rather than owning optimization logic
