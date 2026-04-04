# Hybrid Production Engine

## Purpose

`sampling/production_engine.py` is the repo-native runtime contract for the
full hybrid MD stack.

Its job is to make these layers cooperate in one explicit cycle:

- classical compute spine
- adaptive graph context
- chemistry-aware interface analysis
- memory and replay
- qcloud refinement
- scalable ML residual correction
- executive control
- sanity, drift, and benchmark observers

This keeps the architecture honest: the subsystems are not separate demos,
they are coordinated participants in one production runtime.

## Classification

- `HybridProductionEngine`: `[proposed novel]`
- `ProductionCycleReport`: `[proposed novel]`

The novelty is the orchestration architecture, not the claim that the current
physics already surpasses mature external engines.

## Runtime Role

The production engine sits above `forcefields/hybrid_engine.py`.

- `HybridForceEngine` owns force composition
- `HybridProductionEngine` owns when and how graph, chemistry, memory, qcloud,
  ML, and controller signals are routed through that force path

This separation matters:

- force accounting stays explicit and testable
- orchestration stays explicit and testable
- new backends can replace kernels without rewriting control/memory logic

## Production Cycle

One production cycle currently does the following:

1. reads the current registry state
2. updates or initializes the adaptive graph
3. measures scenario progress
4. diagnoses interface chemistry when compartments allow it
5. records a trace and replay item for the registered state
6. runs the classical hybrid-engine path
7. predicts a preliminary ML residual
8. estimates preliminary uncertainty from residual, graph, memory, chemistry,
   and live features
9. asks the executive controller for budgets and focus compartments
10. runs qcloud selection plus hybrid correction through the same force engine
11. recomputes live features including structure/fidelity context
12. re-estimates uncertainty and re-runs the controller
13. updates the residual model when qcloud-backed correction targets exist
14. opens trajectory / instability / compartment-focus episodes when needed
15. emits sanity, drift, and optional benchmark reports for the same state

## Mathematical Role

The production runtime treats the final effective force as:

`F_total = F_classical + F_qcloud + F_ml`

with explicit gating and conditioning:

- `F_classical` comes from the backend-neutral bonded/nonbonded kernel spine
- `F_qcloud` is selected from bounded local regions informed by graph,
  compartment, memory, and controller signals
- `F_ml` is an additive residual correction from the scalable piece-local model

Control does not invent a separate force law.
It allocates bounded refinement and learning budgets around the established
force substrate.

## Interfaces

### `ProductionCycleReport`

Carries one state-aligned integrated report:

- state and graph
- progress, structure, fidelity, and chemistry observers
- classical and final force evaluations
- qcloud result
- replay/trace records
- residual prediction and live features
- preliminary and final uncertainty
- preliminary and final controller decisions
- sanity, drift, and benchmark reports

### `HybridProductionEngine`

Important methods:

- `current_state() -> SimulationState`
- `current_graph() -> ConnectivityGraph`
- `preview_force_evaluation(state) -> ForceEvaluation`
- `collect_cycle(benchmark_repeats=None) -> ProductionCycleReport`
- `advance(steps=1, record_final_state=False, benchmark_repeats=None) -> ProductionCycleReport | None`

## Current Consumers

- `scripts/live_dashboard.py` now uses this engine as its primary runtime
- the dashboard is now a view over the production engine, not its own parallel
  orchestration path

## Validation Thinking

The production engine is considered healthy only if:

- graph, memory, qcloud, ML, and control all align to the same `state_id`
- force accounting still passes sanity checks
- registry lineage remains valid during stepping
- repeated cycle collection does not duplicate trace ownership incorrectly
- benchmark observers can still run without mutating engine ownership

## Current Boundary

This module makes the architecture intrinsically coordinated.

It does not yet mean:

- full all-atom predictive fidelity
- calibrated experimental kinetics/thermodynamics
- compiled or GPU backend acceleration

Those remain downstream scientific and performance phases built on this runtime
contract.
