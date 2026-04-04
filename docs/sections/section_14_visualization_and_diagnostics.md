# Section 14: Visualization and Diagnostics

## STEP A. Section Name and Objective

Section name: `Visualization and diagnostics`

Objective:

- define trajectory- and controller-oriented view adapters
- define graph and compartment diagnostic views
- define an import-safe export boundary for rendered dashboard artifacts
- define a local live-dashboard entrypoint so the current engine state can be inspected in real time

Primary classification: `[adapted]`

## STEP B. Mathematical and Architectural Role

Section 14 adds:

- `R_traj`
- `R_graph`
- `R_dash`

Architecturally:

- `visualization/` consumes state, graph, controller, validation, and benchmark outputs
- `io/` serializes the resulting dashboard payloads without becoming a Python package
- `scripts/live_dashboard.py` exercises the current scientific stack and writes live dashboard artifacts

## STEP C. Folder Structure Created or Extended

- `visualization/`
- `io/`
- `scripts/`
- `tests/`
- `docs/architecture/`
- `docs/sections/`
- `progress_info/`

## STEP D. Smaller Scripts Added

- `visualization/trajectory_views.py`
- `visualization/graph_views.py`
- `io/export_registry.py`
- `scripts/live_dashboard.py`
- `tests/test_visualization.py`
- `docs/architecture/visualization_and_diagnostics.md`
- `docs/sections/section_14_visualization_and_diagnostics.md`

## STEP E. Implementation Notes

- `visualization/trajectory_views.py`
  - defines particle, action, validation, benchmark, trajectory, and dashboard view models
  - now also carries explicit problem statement, reference summary, and objective summaries into the dashboard payload
  - renders the live dashboard HTML shell that auto-refreshes from exported JSON
- `visualization/problem_views.py`
  - defines concrete objective cards and problem statement views for scenario-driven dashboards
- `visualization/graph_views.py`
  - defines node, edge, and graph snapshot views
  - preserves graph counts, edge kinds, and compartment-route labeling
- `io/export_registry.py`
  - defines `ExportArtifact`, `DashboardExportBundle`, and dashboard export/load helpers
  - keeps export logic import-safe by living in the non-package `io/` folder
- `scripts/live_dashboard.py`
  - builds reusable live scenarios from the current engine stack
  - writes `index.html` and `dashboard.json` snapshots repeatedly for real-time local viewing

Key design choices:

- Section 14 stays downstream of structured reports from Sections 12 and 13
- dashboard HTML polls exported JSON rather than reaching into live Python objects
- `io/` remains import-safe through explicit file-path loading
- the live dashboard uses real engine components instead of fake static example payloads
- the default live problem is now a concrete two-trimer encounter-complex assembly case, not a generic toy system
- the default live problem can now be switched to a barnase-barstar proxy benchmark with attached known answers
- the default live problem now upgrades again to a harder spike-ACE2 proxy benchmark with scenario-owned reference reporting
- the dashboard can now render a separate structural-comparison block beside objective and reference metrics

## STEP F. Validation Strategy

Section 14 validation covers:

- graph snapshot rendering and route labeling
- dashboard HTML shell generation with live-refresh behavior
- export-bundle creation for HTML and JSON artifacts
- smoke execution of the live dashboard script

Commands:

```bash
python3 scripts/validate_scaffold.py
python3 -m unittest discover -s tests
python3 scripts/live_dashboard.py --output-dir /tmp/neurocgmd_dashboard --steps 2 --interval 0.0 --refresh-ms 500
```

## STEP G. Documentation Updated

- `docs/architecture/visualization_and_diagnostics.md`
- `docs/sections/section_14_visualization_and_diagnostics.md`
- `progress_info/*.md`

## STEP H. Progress Files Updated

All mandatory continuity files were updated to reflect Section 14 completion and
the transition to Section 15.

## STEP I. Handoff Summary

SECTION HANDOFF
- Section completed: Section 14. Visualization and diagnostics
- Purpose: add structured diagnostic views, import-safe dashboard export, and a local live-dashboard harness
- Files created: trajectory views, graph views, problem views, export registry, live dashboard script, tests, docs, and continuity updates
- Key interfaces: `GraphSnapshotView`, `ProblemStatementView`, `TrajectoryFrameView`, `DashboardSnapshotView`, `ExportArtifact`, `DashboardExportBundle`
- Dependencies: `core/state.py`, `graph/graph_manager.py`, `compartments/registry.py`, `validation/`, `benchmarks/`, `ai_control/controller.py`
- What now works: serializable graph, trajectory, and objective diagnostics, export-safe HTML/JSON dashboard bundles, and a smoke-tested local live dashboard on encounter-complex, barnase-barstar, and harder spike-ACE2 scenarios
- What now works: serializable graph, trajectory, objective, reference, and reduced-structure diagnostics, export-safe HTML/JSON dashboard bundles, and a smoke-tested local live dashboard on encounter-complex, barnase-barstar, and harder spike-ACE2 scenarios
- What is still stubbed: richer plotting, long-horizon trajectory playback, browser-side control actions, and publication-grade rendering
- Risks / unresolved items: current GUI is intentionally zero-dependency and local; later UI work may enrich it, but should keep structured payloads explicit and should not move ownership into the rendering layer
- Next recommended section: Section 15. Performance optimization and scaling hooks
- Important continuity notes: Section 15 should optimize stable interfaces beneath this dashboard rather than coupling performance logic directly into the visualization or export layers
