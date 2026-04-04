# Section 2: Core Data Models and Simulation State

## STEP A. Section Name and Objective

Section name: `Core data models and simulation state`

Objective:

- define the canonical simulation snapshot structure
- make units, provenance, and validation explicit
- provide lifecycle helpers for creating and deriving states
- leave all downstream modules with a stable state contract

Primary classification: `[adapted]`

## STEP B. Mathematical and Architectural Role

The state representation is the first true mathematical substrate of the platform.
It defines the objects that later sections will evolve.

Primary snapshot:

`S_k = (U, X_k, V_k, F_k, m, T_k, C_k, P_k, O_k)`

This gives us:

- a physical substrate for later integrators and force evaluation
- a provenance substrate for memory, replay, and diagnostics
- a serialization substrate for IO and benchmarking

The design is intentionally topology-agnostic so that Section 3 can add topology
without forcing a rewrite of the state layer.

## STEP C. Folder Structure Created or Extended

- `core/`
- `tests/`
- `docs/architecture/`
- `docs/sections/`
- `progress_info/`

## STEP D. Smaller Scripts Added

- `core/types.py`
- `core/state.py`
- `core/state_registry.py`
- `tests/test_core_types.py`
- `tests/test_core_state.py`
- `docs/architecture/state_model.md`
- `docs/sections/section_02_core_state.md`

## STEP E. Implementation Notes

- `core/types.py`
  - defines stable identifiers, vector aliases, coercion helpers, and `FrozenMetadata`
- `core/state.py`
  - defines `UnitSystem`, `SimulationCell`, `ThermodynamicState`, `ParticleState`,
    `StateProvenance`, and `SimulationState`
- `core/state_registry.py`
  - defines `IdentifierMint`, `LifecycleStage`, `SimulationStateRegistry`, and
    `StateSnapshotSummary`

Key design choices:

- immutable dataclasses for state-bearing objects
- deterministic metadata freezing for reproducibility
- backend-agnostic tuples rather than binding early to NumPy, JAX, or PyTorch
- explicit `to_dict()` / `from_dict()` methods to define the serialization boundary

## STEP F. Validation Strategy

Section 2 validation covers:

- particle shape and mass invariants
- thermodynamic ensemble consistency
- provenance integrity
- simulation state round-trip serialization
- registry lineage correctness and state summaries

Commands:

```bash
python3 scripts/validate_scaffold.py
python3 -m unittest discover -s tests
```

## STEP G. Documentation Updated

- `docs/architecture/state_model.md`
- `docs/sections/section_02_core_state.md`
- `progress_info/*.md`

## STEP H. Progress Files Updated

All mandatory continuity files were updated to reflect Section 2 completion and the
transition to Section 3.

## STEP I. Handoff Summary

SECTION HANDOFF
- Section completed: Section 2. Core data models and simulation state
- Purpose: provide the canonical units-aware, provenance-aware simulation snapshot used by all later sections
- Files created: `core/types.py`, `core/state.py`, `core/state_registry.py`, Section 2 docs, and new tests
- Key interfaces: `FrozenMetadata`, `ParticleState`, `SimulationState`, `StateProvenance`, `SimulationStateRegistry`
- Dependencies: `core/exceptions.py`, `core/interfaces.py`, standard-library Python only
- What now works: immutable state creation, validation, provenance capture, lineage tracking, checkpoint summaries, and dictionary serialization
- What is still stubbed: topology semantics, force terms, integrators, graph logic, plasticity, compartments, replay policy, qcloud, ML, and AI control
- Risks / unresolved items: array backend selection is still deferred; state uses tuple-based containers for correctness and portability rather than speed
- Next recommended section: Section 3. Topology and bead system
- Important continuity notes: topology objects should reference or align with `ParticleState` indices without mutating the Section 2 contracts unnecessarily

