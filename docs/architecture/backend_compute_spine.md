# Backend Compute Spine

## Purpose

This phase adds a real compute-spine boundary beneath the scientific and
architectural layers.

Classification:

- backend-neutral tensor/array contracts: `[adapted]`
- sparse neighbor-list and pairwise execution hooks: `[established]`
- hybrid force composition over classical + qcloud + ML layers: `[hybrid]`
- differentiable parameter hooks over repo-native ML modules: `[adapted]`
- execution planning over stable backend descriptors: `[adapted]`

The intent is not to clone AMBER, GROMACS, OpenMM, or JAX. The intent is to
adopt the parts of their design that are proven:

- explicit backend boundaries
- sparse pair execution
- stable kernel surfaces
- clean force accounting
- separation between scientific logic and execution planning

Then our novelty stays where it belongs:

- shadow coarse-grained correction
- adaptive topology and compartments
- memory-driven correction and control
- intelligent executive orchestration

## Modules

### `physics/backends/`

- `contracts.py`
  - defines backend-neutral tensor blocks and pairwise execution records
- `reference_backend.py`
  - pure-Python reference backend for neighbor lists and pairwise execution
- `dispatch.py`
  - dispatch boundary built on the stable backend registry

### `physics/kernels/`

- `bonded.py`
  - harmonic bonded kernel
- `nonbonded.py`
  - sparse Lennard-Jones kernel through the backend pairwise hook
- `electrostatics.py`
  - sparse Coulomb kernel through the same backend path
- `constraints.py`
  - explicit distance projection kernel
- `integration.py`
  - backend-ready Velocity Verlet step kernel

### `forcefields/hybrid_engine.py`

One explicit composition surface for:

- classical bonded/nonbonded kernels
- optional electrostatics
- qcloud correction payloads
- ML residual corrections

This keeps force accounting explicit instead of spreading composition logic
across scripts or controller code.

### `ml/differentiable_hooks.py`

Provides:

- stable parameter descriptors
- parameter snapshots/restores
- finite-difference gradient estimation
- differentiable adapter for `ScalableResidualModel`

This is gradient-ready infrastructure, not a claim that the whole engine is
already end-to-end autodiff-native.

### `validation/backend_parity.py`

Compares the new backend-spine path against trusted in-repo reference evaluators
using:

- energy absolute error
- force RMS error
- max force-component error

### `optimization/backend_execution.py`

Translates a stable backend selection into:

- execution mode
- chunking plan
- vector-width hint

without making scientific modules own those heuristics.

## Validation Thinking

We validate this phase by checking:

- kernel dispatch resolves a concrete backend correctly
- sparse backend kernels reproduce trusted baseline evaluator outputs
- the hybrid engine keeps component accounting explicit
- differentiable hooks expose stable trainable parameter blocks
- execution planning stays separate from the scientific layer

## Continuity Note

This compute spine should be the place where future stronger backends land.
If a future NumPy, JAX, custom C++, or GPU path is added, it should satisfy the
same backend-neutral contracts rather than bypassing them from scripts.
