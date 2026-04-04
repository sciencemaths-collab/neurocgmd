# Section 13: Validation and Benchmarking Suite

## STEP A. Section Name and Objective

Section name: `Validation and benchmarking suite`

Objective:

- define observer-side sanity checks across subsystem outputs
- define trajectory drift checks over ordered state windows
- define reusable benchmark report carriers and baseline benchmark runs
- keep validation and benchmarking separate from subsystem ownership

Primary classification: `[established]`

## STEP B. Mathematical and Architectural Role

Section 13 adds:

- `V_sanity`
- `V_drift`
- `B`

Architecturally:

- `validation/` observes state, topology, force, graph, qcloud, ML, and controller outputs
- `benchmarks/` measures repeatable execution slices without mutating those subsystems
- this section remains downstream of all scientific and control layers built so far

## STEP C. Folder Structure Created or Extended

- `validation/`
- `benchmarks/`
- `tests/`
- `docs/architecture/`
- `docs/sections/`
- `progress_info/`

## STEP D. Smaller Scripts Added

- `validation/sanity_checks.py`
- `validation/drift_checks.py`
- `benchmarks/baseline_suite.py`
- `tests/test_validation_benchmarks.py`
- `docs/architecture/validation_and_benchmarking.md`
- `docs/sections/section_13_validation_and_benchmarking_suite.md`

## STEP E. Implementation Notes

- `validation/sanity_checks.py`
  - defines `SanityCheckResult`, `SanityCheckReport`, and `FoundationSanityChecker`
  - checks alignment across state, topology, force, graph, qcloud, ML, and controller outputs
- `validation/drift_checks.py`
  - defines `DriftThresholds`, `DriftCheckReport`, and `TrajectoryDriftChecker`
  - checks monotonic time/step progression and coarse energy/position drift
- `benchmarks/baseline_suite.py`
  - defines `BenchmarkCaseResult`, `BenchmarkReport`, and `BaselineBenchmarkSuite`
  - measures force, graph, qcloud, ML, and controller pathways with repeatable structured reporting
- `benchmarks/reference_cases/`
  - now stores experimentally grounded benchmark targets with known answers
  - first real-world target is the barnase-barstar association benchmark
  - now also stores proxy-vs-reference comparison helpers for live benchmarking

Key design choices:

- Section 13 remains observer-only and does not take ownership of scientific state
- benchmark reports are structured and deterministic in naming
- real-world reference targets are stored separately from timing results so performance evidence and biological truth do not get conflated
- wall-clock timings are treated as local diagnostics, not universal claims
- validation checks produce explicit named results rather than unstructured strings

## STEP F. Validation Strategy

Section 13 validation covers:

- aligned subsystem-output sanity reports
- force-energy mismatch detection
- low-drift registry-backed trajectory validation
- explicit step and energy-drift failure reporting
- benchmark-suite report structure across force, graph, qcloud, ML, and controller paths

Commands:

```bash
python3 scripts/validate_scaffold.py
python3 -m unittest discover -s tests
```

## STEP G. Documentation Updated

- `docs/architecture/validation_and_benchmarking.md`
- `docs/sections/section_13_validation_and_benchmarking_suite.md`
- `progress_info/*.md`

## STEP H. Progress Files Updated

All mandatory continuity files were updated to reflect Section 13 completion and
the transition to Section 14.

## Extension Note

Section 13 now also hosts `validation/structure_metrics.py`, which adds the
first honest proxy-vs-reference structural comparison layer for the harder live
benchmark. This extension is still `[adapted]` and explicitly reports reduced
landmark benchmarking rather than claiming atomistic agreement.

## STEP I. Handoff Summary

SECTION HANDOFF
- Section completed: Section 13. Validation and benchmarking suite
- Purpose: add observer-side sanity reports, drift reports, and repeatable benchmark reporting for the built scientific and control layers
- Files created: sanity checker, drift checker, benchmark suite, tests, docs, and continuity updates
- Key interfaces: `SanityCheckReport`, `FoundationSanityChecker`, `DriftCheckReport`, `TrajectoryDriftChecker`, `BenchmarkCaseResult`, `BenchmarkReport`, `BaselineBenchmarkSuite`
- Dependencies: `core/state.py`, `core/state_registry.py`, `physics/forces/composite.py`, `graph/graph_manager.py`, `qcloud/qcloud_coupling.py`, `ml/residual_model.py`, `ai_control/controller.py`
- What now works: cross-layer alignment checks, trajectory drift reports, baseline timing reports for force, graph, qcloud, ML, and controller execution slices, and an experimentally grounded barnase-barstar reference target for later scientific comparison
- What is still stubbed: richer physical validation metrics, stochastic benchmark campaigns, long-horizon statistical summaries, and CI-grade performance baselines
- Risks / unresolved items: benchmark timings remain local environment diagnostics; later performance work must not overinterpret them as cross-machine or publication-grade results without stronger methodology
- Next recommended section: Section 14. Visualization and diagnostics
- Important continuity notes: Section 14 should consume these reports for display and diagnostics, not move validation logic into visualization or hide report structure behind ad hoc rendering
