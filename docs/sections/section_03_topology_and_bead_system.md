# Section 3: Topology and Bead System

## STEP A. Section Name and Objective

Section name: `Topology and bead system`

Objective:

- define static bead and bead-type objects
- define fixed bonded connectivity
- assemble a system-level topology aligned to `ParticleState`
- keep physical topology cleanly separate from future adaptive graph logic

Primary classification: `[adapted]`

## STEP B. Mathematical and Architectural Role

Section 3 defines the fixed topological substrate:

`T_fixed = (B, T, p, E_fixed)`

Where:

- `B` is the set of beads
- `T` is the set of bead types
- `p` aligns beads to Section 2 particle indices
- `E_fixed` is the static bond set

Architecturally, this becomes the upstream source of structural truth for later
force fields and graph layers. The key boundary is that `SystemTopology` is fixed
topology, not adaptive connectivity.

## STEP C. Folder Structure Created or Extended

- `topology/`
- `tests/`
- `docs/architecture/`
- `docs/sections/`
- `progress_info/`

## STEP D. Smaller Scripts Added

- `topology/beads.py`
- `topology/bonds.py`
- `topology/system_topology.py`
- `tests/test_topology.py`
- `docs/architecture/topology_model.md`
- `docs/sections/section_03_topology_and_bead_system.md`

## STEP E. Implementation Notes

- `topology/beads.py`
  - defines `BeadRole`, `BeadType`, and `Bead`
- `topology/bonds.py`
  - defines `BondKind`, `Bond`, `build_neighbor_map`, and `connected_components`
- `topology/system_topology.py`
  - defines `SystemTopology` and the alignment check against `ParticleState`

Key design choices:

- one bead per Section 2 particle index
- bead types are declared separately from bead instances
- fixed topology and future adaptive graph topology are explicitly separated
- topology objects are immutable and serialization-ready

## STEP F. Validation Strategy

Section 3 validation covers:

- bead-type declaration integrity
- unique contiguous particle indexing
- duplicate bond rejection
- neighbor and connected-component correctness
- topology alignment against `ParticleState`

Commands:

```bash
python3 scripts/validate_scaffold.py
python3 -m unittest discover -s tests
```

## STEP G. Documentation Updated

- `docs/architecture/topology_model.md`
- `docs/sections/section_03_topology_and_bead_system.md`
- `progress_info/*.md`

## STEP H. Progress Files Updated

All mandatory continuity files were updated to reflect Section 3 completion and the
transition to Section 4.

## STEP I. Handoff Summary

SECTION HANDOFF
- Section completed: Section 3. Topology and bead system
- Purpose: provide the fixed bead-level structural substrate aligned to Section 2 particle state
- Files created: topology models, topology tests, Section 3 docs, and continuity updates
- Key interfaces: `BeadType`, `Bead`, `Bond`, `SystemTopology`
- Dependencies: `core/types.py`, `core/state.py`, standard-library Python only
- What now works: static bead typing, bond representation, neighbor queries, connected components, serialization, and state/topology alignment checks
- What is still stubbed: force-field energetics, integrators, adaptive graph logic, plasticity, compartments, memory, qcloud, ML, and AI control
- Risks / unresolved items: angles, torsions, and constraint groups are deferred; force constants stored on bonds are only structural placeholders until Section 4
- Next recommended section: Section 4. Force field foundation
- Important continuity notes: Section 4 should consume `SystemTopology` as structural truth while keeping force parameters and energy functions in `forcefields/` and `physics/`

