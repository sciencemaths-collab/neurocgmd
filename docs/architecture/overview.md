# Architecture Overview

## Current Implemented Scope

Sections 1-15 now establish:

- the repository scaffold and continuity layer
- the section manifest and validation script
- the canonical state model, provenance model, and lifecycle registry
- the fixed topology layer with bead types, beads, bonds, and system assembly
- the baseline force-field parameter layer and first energy/force terms
- the first executable MD stepping layer and lineage-preserving simulation loop
- the first adaptive graph layer separated from fixed topology
- the first bounded plasticity layer for adaptive edge reinforcement, pruning, and growth
- the first compartment overlay and compartment-aware routing summaries
- the first long-horizon memory layer for trace storage, deterministic replay, and episode grouping
- the first qcloud framework for bounded refinement-region selection and local correction coupling
- the first online ML framework for replay-driven residual targets, uncertainty estimation, and additive learned correction
- the first explicit AI executive-control framework for stability assessment, bounded resource allocation, and transparent action recommendation
- the first observer-side validation and benchmark framework for sanity checks, drift reports, and repeatable baseline measurements
- the first serializable diagnostic view layer plus a local live dashboard for real-time inspection
- the first explicit optimization substrate for profiling, backend selection, and scaling-plan generation

Primary classifications:

- Section 1: `[adapted]`
- Section 2: `[adapted]`
- Section 3: `[adapted]`
- Section 4: `[established]`
- Section 5: `[established]`
- Section 6: `[proposed novel]`
- Section 7: `[proposed novel]`
- Section 8: `[hybrid]`
- Section 9: `[hybrid]`
- Section 10: `[hybrid]`
- Section 11: `[hybrid]`
- Section 12: `[proposed novel]`
- Section 13: `[established]`
- Section 14: `[adapted]`
- Section 15: `[adapted]`

## Mathematical Role of the Scaffold

The scaffold defines a dependency graph over subsystems:

- Let `V` be the set of top-level subsystems.
- Let `E` be directed dependency edges from upstream foundations to downstream users.
- Let `c: V -> { [established], [adapted], [hybrid], [proposed novel] }` classify
  scientific and architectural novelty.

At this stage, the main mathematical object is the architecture graph itself plus
the simulation snapshot that later dynamics will evolve:

`G_arch = (V, E, c)`

`S_k = (U, X_k, V_k, F_k, m, T_k, C_k, P_k, O_k)`

This matters because later sections will add physically meaningful state evolution
on top of `G_arch`. The graph makes it possible to ask whether a new subsystem is:

- connected to the right upstream state
- validated at the right boundary
- documented with the right novelty label

The snapshot `S_k` makes it possible to ask whether a downstream subsystem reads
the same positions, forces, thermodynamic controls, units, and provenance as the
rest of the platform.

## Planned Layering

1. `core/` and `config/`
   - `[established]` and `[adapted]`
   - host global constants, contracts, state models, layout discovery, and section metadata
2. `topology/`, `physics/`, `forcefields/`, `integrators/`, `sampling/`
   - largely `[established]` or `[adapted]`
   - define the physical substrate and time evolution
   - `topology/` is now the fixed structural truth aligned to particle indices
   - `forcefields/` and `physics/` now contain the first established interaction terms
   - `integrators/` and `sampling/` now contain the first deterministic MD stepping loop
3. `graph/`, `plasticity/`, `compartments/`, `memory/`
   - `[hybrid]` to `[proposed novel]`
   - add adaptive organization, reinforcement, pruning, and historical traces
   - `graph/` now contains the first dynamic connectivity substrate
   - `plasticity/` now contains the first bounded graph-update rules
   - `compartments/` now contains the first modular overlay and route summaries
   - `memory/` now contains long-horizon trace records, deterministic replay, and episode grouping
4. `qcloud/`, `ml/`, `ai_control/`
   - `[hybrid]` to `[proposed novel]`
   - couple uncertainty-aware refinement, residual learning, and executive control
   - `qcloud/` now contains bounded region selection and additive local correction coupling
   - `ml/` now contains replay-driven residual learning, heuristic uncertainty, and online trainer hooks
   - `ai_control/` now contains explicit monitoring, allocation, policy, and controller orchestration
5. `validation/`, `benchmarks/`, `visualization/`, `io/`
   - `[established]` to `[adapted]`
   - verify, explain, and export system behavior
   - `validation/` now contains sanity and drift checks
   - `benchmarks/` now contains the first baseline benchmark suite
   - `visualization/` now contains trajectory, graph, and dashboard view adapters
   - `io/` now contains import-safe dashboard export helpers
6. `optimization/`
   - `[adapted]`
   - profiles stable interfaces, records backend selections, and emits workload-aware scaling plans
   - keeps performance planning explicit and downstream of the scientific stack
7. post-roadmap scientific fidelity
   - `[adapted]` to `[proposed novel]`
   - adds provenance-aware trusted science plus a mirrored shadow correction cloud
   - keeps the coarse-grained body as the main substrate while using local shadow corrections to move toward higher fidelity

## Dependency Direction

The intended dependency direction is downward-only:

- `core/` should remain the lowest-level internal dependency.
- `config/` may depend on `core/`, but domain packages should not depend on high-level
  orchestration packages unless explicitly justified.
- `ai_control/` is downstream of the physical and adaptive layers, not a hidden owner
  of their internal data structures.
- `validation/` and `benchmarks/` are consumers, not sources of domain truth.
- `visualization/` and `io/` are rendering/export consumers, not simulation owners.

## Foundation Validation

The implemented foundation is considered valid when:

- all required top-level directories exist
- all required `progress_info/` files exist
- the section manifest is contiguous and classification-safe
- the scaffold validator and unit tests pass
