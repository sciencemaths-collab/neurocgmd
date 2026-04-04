# Simulation Loop Architecture

## Section 5 Scope

Section 5 introduces the first time-integration and simulation-loop layer. This
section is classified as `[established]` for the basic MD stepping structure, with
some `[adapted]` choices around software interfaces and reduced-unit stochastic
guardrails.

## Mathematical Role

The loop advances the canonical state:

`S_k -> S_(k+1)`

using:

- a force evaluator `F(S_k, T_fixed, Phi)` built from topology `T_fixed` and
  force-field parameters `Phi`
- an integrator update rule `I`

The current stepping rule is a velocity-Verlet-style Langevin integrator:

1. evaluate forces at `S_k`
2. apply a half-kick to velocities
3. apply deterministic damping, with optional guarded stochastic noise
4. drift positions
5. re-evaluate forces at the predicted state
6. apply the second half-kick

## Architectural Boundaries

### Included in Section 5

- a force-evaluator protocol for integrator input
- a baseline composite evaluator that combines current bonded and nonbonded physics
- a first integrator implementation
- a simulation-loop runner that stores lineage in `SimulationStateRegistry`

### Explicitly Excluded from Section 5

- adaptive graph-driven force corrections
- learned residual force terms
- compartment-aware scheduling
- quantum-cloud refinement
- distributed execution or GPU acceleration

## Design Decisions

- integrators do not own parameter lookup; they receive a force evaluator
- registry insertion happens in `sampling/`, not inside integrators
- stochastic Langevin mode is guarded behind `assume_reduced_units=True` to avoid
  hiding shaky unit assumptions in the default path
- the default deterministic path is intentionally testable and reproducible

## Validation Thinking

We test Section 5 by checking:

- baseline force composition returns the expected force shape and scalar energy
- the integrator advances time and position deterministically under zero force
- the simulation loop records state lineage through the registry
- full test-suite compatibility with Sections 1-4 remains intact

