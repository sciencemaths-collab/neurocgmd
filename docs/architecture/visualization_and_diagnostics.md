# Visualization and Diagnostics Architecture

## Section 14 Scope

Section 14 introduces the first local visualization and diagnostic-rendering
layer. This section is classified as `[adapted]` because the rendering patterns
themselves are standard, while the exact dashboard composition, graph/trajectory
view structure, and import-safe export boundary are adapted to the repository's
modular architecture.

## Mathematical Role

Section 14 adds view operators that transform structured scientific reports into
serializable diagnostic payloads:

- trajectory view
  - `R_traj = phi_traj(S_k, D_k, U_k, Q_k)`
- graph view
  - `R_graph = phi_graph(S_k, G_k, C_k)`
- dashboard payload
  - `R_dash = phi_dash(R_traj, R_graph, V_k, B_k)`

These views are interpretive and export-safe:

- they do not own simulation state
- they do not replace validation logic
- they do not mutate qcloud, ML, graph, or controller outputs

## Foundation Design

- `visualization/trajectory_views.py`
  - serializable particle, controller, problem, validation, benchmark, trajectory, and dashboard views
- `visualization/graph_views.py`
  - serializable node-, edge-, and graph-snapshot views
- `visualization/problem_views.py`
  - serializable problem statement, reference-summary, and objective-metric views for concrete simulation targets
- `io/export_registry.py`
  - import-safe dashboard export utilities loaded by file path rather than `io` package import
- `scripts/live_dashboard.py`
  - zero-dependency local live-dashboard harness that writes auto-refreshing HTML and JSON artifacts
- `sampling/scenarios/complex_assembly.py`
  - `[adapted]` concrete encounter-complex benchmark used by the live dashboard
- `sampling/scenarios/barnase_barstar.py`
  - `[hybrid]` coarse-grained proxy for the real barnase-barstar benchmark
- `sampling/scenarios/spike_ace2.py`
  - `[hybrid]` harder coarse-grained proxy for the real ACE2-spike benchmark
- `validation/structure_metrics.py`
  - local atomistic-centroid structural comparison reports for live benchmark scenarios

## Core Invariants

- visualization remains downstream of structured reports
- `io/` remains a plain folder and is accessed through explicit file-path loading
- the dashboard reads exported JSON rather than owning runtime state directly
- concrete live problems are defined in reusable scenario modules rather than ad hoc script state
- experimentally known answers can be attached to a live problem without pretending the proxy is already fully validated
- scenario modules own their own reference-comparison logic so the dashboard stays benchmark-agnostic
- scenario modules may also attach structural-comparison reports without moving comparison ownership into the renderer
- HTML and JSON artifacts are deterministic and export-safe
- live dashboard smoke runs must use real repository modules rather than fake ad hoc payloads

## Validation Thinking

We test Section 14 by checking:

- graph snapshots preserve node counts, edge counts, and route summaries
- dashboard HTML contains the live refresh shell and the expected JSON endpoint
- problem statements and objective metrics survive serialization into the exported payload
- reference benchmark answers survive serialization into the exported payload
- export helpers write both `index.html` and `dashboard.json`
- the live dashboard script can generate a working snapshot bundle from the real engine stack
