# Section 7: Plasticity and Rewiring Rules

## STEP A. Section Name and Objective

Section name: `Plasticity and rewiring rules`

Objective:

- define pair-trace memory for graph updates
- implement bounded reinforcement and weakening
- implement pruning of weak unsupported adaptive edges
- implement Hebbian-style growth of new adaptive edges

Primary classification: `[proposed novel]`

## STEP B. Mathematical and Architectural Role

Section 7 operates on the adaptive graph rather than on fixed topology:

`(G_dyn(k), M_pair(k)) -> (G_dyn'(k), M_pair(k+1))`

Where:

- `G_dyn(k)` is the adaptive graph snapshot from Section 6
- `M_pair(k)` is the pair-level plasticity trace state

The architecture keeps responsibilities separate:

- `topology/` owns fixed structural truth
- `graph/` owns adaptive connectivity snapshots
- `plasticity/` owns rule-based modification of adaptive connectivity

## STEP C. Folder Structure Created or Extended

- `plasticity/`
- `tests/`
- `docs/architecture/`
- `docs/sections/`
- `progress_info/`

## STEP D. Smaller Scripts Added

- `plasticity/traces.py`
- `plasticity/reinforcement.py`
- `plasticity/pruning.py`
- `plasticity/hebbian.py`
- `plasticity/engine.py`
- `tests/test_plasticity_layer.py`
- `docs/architecture/plasticity_layer.md`
- `docs/sections/section_07_plasticity_and_rewiring.md`

## STEP E. Implementation Notes

- `plasticity/traces.py`
  - defines `PairTraceState` and deterministic trace updates
- `plasticity/reinforcement.py`
  - defines bounded adaptive-edge reinforcement and weakening
- `plasticity/pruning.py`
  - defines adaptive-edge deactivation rules
- `plasticity/hebbian.py`
  - defines bounded Hebbian-style edge growth
- `plasticity/engine.py`
  - composes the full update pass

Key design choices:

- traces are pair-level so future growth can use evidence even when an edge does not yet exist
- structural edges are protected from Section 7 modification
- pruning deactivates rather than deleting adaptive edge state
- edge growth is capped per step to prevent runaway graph expansion

## STEP F. Validation Strategy

Section 7 validation covers:

- deterministic trace updates
- bounded reinforcement of adaptive edges
- pruning behavior for weak unsupported adaptive edges
- Hebbian growth for absent but strongly coactive nearby pairs
- end-to-end engine behavior and invariant preservation

Commands:

```bash
python3 scripts/validate_scaffold.py
python3 -m unittest discover -s tests
```

## STEP G. Documentation Updated

- `docs/architecture/plasticity_layer.md`
- `docs/sections/section_07_plasticity_and_rewiring.md`
- `progress_info/*.md`

## STEP H. Progress Files Updated

All mandatory continuity files were updated to reflect Section 7 completion and the
transition to Section 8.

## STEP I. Handoff Summary

SECTION HANDOFF
- Section completed: Section 7. Plasticity and rewiring rules
- Purpose: provide bounded rule-based modification of adaptive graph connectivity
- Files created: pair-trace memory, reinforcement, pruning, growth, engine, tests, docs, and continuity updates
- Key interfaces: `PairTraceState`, `ReinforcementRule`, `PruningRule`, `HebbianGrowthRule`, `PlasticityEngine`
- Dependencies: `graph/graph_manager.py`, `graph/edge_models.py`, `core/state.py`, `topology/system_topology.py`
- What now works: deterministic trace accumulation, bounded adaptive-edge reinforcement, pruning by weak support, capped Hebbian growth, and composed plasticity updates
- What is still stubbed: compartment-aware routing, dedicated memory/replay integration, qcloud-guided rewiring, ML-informed plasticity, and executive AI control
- Risks / unresolved items: pair traces are intentionally minimal before the dedicated memory section; future sections must decide how uncertainty and replay signals should alter these rules
- Next recommended section: Section 8. Compartment system
- Important continuity notes: Section 8 should consume graph and plasticity outputs without turning compartments into hidden owners of graph state

