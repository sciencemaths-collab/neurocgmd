# Section 5: Integrators and Simulation Loop

## STEP A. Section Name and Objective

Section name: `Integrators and simulation loop`

Objective:

- define integrator and force-evaluator interfaces
- implement a first established stepping scheme
- implement the first simulation-loop driver
- preserve clear boundaries between state, topology, parameters, forces, and lineage

Primary classification: `[established]`

## STEP B. Mathematical and Architectural Role

Section 5 introduces the state-transition mechanism:

`S_(k+1) = I(S_k, T_fixed, Phi, F)`

Where:

- `I` is the integrator
- `T_fixed` is the fixed topology
- `Phi` is the force-field parameter set
- `F` is the force evaluator

The current implementation uses a velocity-Verlet-style Langevin integrator with a
deterministic default path and an explicitly guarded stochastic option.

## STEP C. Folder Structure Created or Extended

- `integrators/`
- `sampling/`
- `physics/forces/`
- `tests/`
- `docs/architecture/`
- `docs/sections/`
- `progress_info/`

## STEP D. Smaller Scripts Added

- `integrators/base.py`
- `integrators/langevin.py`
- `sampling/simulation_loop.py`
- `physics/forces/bonded_forces.py`
- `physics/forces/composite.py`
- `tests/test_simulation_loop.py`
- `docs/architecture/simulation_loop.md`
- `docs/sections/section_05_integrators_and_simulation_loop.md`

## STEP E. Implementation Notes

- `integrators/base.py`
  - defines `ForceEvaluator`, `StateIntegrator`, and `IntegratorStepResult`
- `integrators/langevin.py`
  - defines `LangevinIntegrator`
- `physics/forces/bonded_forces.py`
  - adds harmonic bonded force evaluation needed for actual stepping
- `physics/forces/composite.py`
  - defines `ForceEvaluation` and `BaselineForceEvaluator`
- `sampling/simulation_loop.py`
  - defines `SimulationLoop` and `SimulationRunResult`

Key design choices:

- force composition lives in `physics/forces/`, not in the integrator itself
- registry lineage is updated by the simulation loop, not by the integrator
- deterministic stepping is the default for testability and transparency

## STEP F. Validation Strategy

Section 5 validation covers:

- composite force-evaluation shape and energy accounting
- deterministic integrator advancement under controlled conditions
- simulation-loop time/step bookkeeping
- registry lineage preservation across multiple steps

Commands:

```bash
python3 scripts/validate_scaffold.py
python3 -m unittest discover -s tests
```

## STEP G. Documentation Updated

- `docs/architecture/simulation_loop.md`
- `docs/sections/section_05_integrators_and_simulation_loop.md`
- `progress_info/*.md`

## STEP H. Progress Files Updated

All mandatory continuity files were updated to reflect Section 5 completion and the
transition to Section 6.

## STEP I. Handoff Summary

SECTION HANDOFF
- Section completed: Section 5. Integrators and simulation loop
- Purpose: provide the first executable MD stepping path on top of the state, topology, and force-field layers
- Files created: integrator contracts, Langevin integrator, composite force evaluator, simulation loop, tests, docs, and continuity updates
- Key interfaces: `ForceEvaluator`, `StateIntegrator`, `LangevinIntegrator`, `BaselineForceEvaluator`, `SimulationLoop`
- Dependencies: `core/state.py`, `core/state_registry.py`, `topology/system_topology.py`, `forcefields/base_forcefield.py`, `physics/forces/`
- What now works: one-step force evaluation, deterministic time stepping, and lineage-preserving multi-step simulation runs
- What is still stubbed: adaptive graph updates, plasticity, compartments, memory replay, qcloud, ML residuals, and AI control
- Risks / unresolved items: stochastic Langevin mode is guarded by reduced-unit assumptions; performance is correctness-first rather than optimized
- Next recommended section: Section 6. Dynamic graph connectivity layer
- Important continuity notes: the future graph layer should consume `SimulationState` and `SystemTopology` without taking ownership of the deterministic MD loop

