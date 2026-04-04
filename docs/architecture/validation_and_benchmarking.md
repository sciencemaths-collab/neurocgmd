# Validation and Benchmarking Architecture

## Section 13 Scope

Section 13 introduces the first observer-side validation and benchmarking suite.
This section is classified as `[established]` because sanity checks, drift
checks, and benchmark harnesses are standard engineering practice, even though
their exact placement across the physics, graph, qcloud, ML, and control layers
is adapted to this repository's architecture.

## Mathematical Role

Section 13 adds three observer objects:

- sanity-check outputs
  - `V_sanity = h(S_k, T_k, F_k, G_k, Q_k, Y_hat_k, D_k)`
- drift-check outputs
  - `V_drift = g(S_0, S_1, ..., S_n)`
- benchmark outputs
  - `B = {b_force, b_graph, b_qcloud, b_ml, b_control}`

These objects are descriptive rather than generative. They observe:

- alignment across state, topology, force, graph, qcloud, ML, and controller outputs
- coarse drift across ordered state windows
- repeatable local wall-clock timings for baseline execution slices

## Foundation Design

- `FoundationSanityChecker`
  - validates alignment and accounting across subsystem outputs
- `TrajectoryDriftChecker`
  - validates monotonicity and coarse drift over state sequences
- `BaselineBenchmarkSuite`
  - measures repeatable force, graph, qcloud, ML, and controller pathways
- `benchmarks/reference_cases/`
  - stores experimentally grounded target cases whose answers are known before simulation begins
  - now also stores comparison helpers that keep known answers visible beside coarse live proxies
  - now includes both barnase-barstar and the harder spike-ACE2 reference target
- `validation/structure_metrics.py`
  - computes atomistic-centroid structural comparison reports against explicit local PDB-derived reference scaffolds

## Core Invariants

- `validation/` and `benchmarks/` remain observer layers only
- validation does not mutate registry, graph, qcloud, ML, or controller state
- benchmark timings are environment-specific diagnostics, not universal performance claims
- benchmark case names remain explicit and stable
- real-world reference cases are separate from timing benchmarks and preserve source attribution
- validation outputs are structured reports, not loose strings
- early structural comparison must explicitly state when it is atomistic-centroid alignment rather than full all-atom agreement

## Validation Thinking

We test Section 13 by checking:

- aligned subsystem outputs produce passing sanity reports
- force-energy mismatches are flagged explicitly
- low-drift trajectories pass monotonicity and threshold checks
- step jumps and large energy/position drift are reported explicitly
- benchmark reports contain deterministic case names and structured metadata
- reference cases preserve structural IDs, affinity or kinetic targets, and comparison axes for later scientific validation
- atomistic-centroid reports provide a first honest bridge between live proxy motion and later fuller atomistic structure comparison
