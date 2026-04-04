# Section 9: Memory and Replay System

## STEP A. Section Name and Objective

Section name: `Memory and replay system`

Objective:

- define reusable longer-horizon trace records
- define a deterministic replay buffer
- define explicit ordered episode grouping over state sequences
- keep memory distinct from both lineage and short-horizon plasticity traces

Primary classification: `[hybrid]`

## STEP B. Mathematical and Architectural Role

Section 9 adds reusable historical objects:

- `M_trace`
- `M_replay`
- `E`

These live above provenance lineage and above short-horizon pair traces.

Architecturally:

- `core/state_registry.py` owns lineage
- `plasticity/traces.py` owns short-horizon pair traces
- `memory/` owns reusable long-horizon trace, replay, and episode abstractions

## STEP C. Folder Structure Created or Extended

- `memory/`
- `tests/`
- `docs/architecture/`
- `docs/sections/`
- `progress_info/`

## STEP D. Smaller Scripts Added

- `memory/trace_store.py`
- `memory/replay_buffer.py`
- `memory/episode_registry.py`
- `tests/test_memory_layer.py`
- `docs/architecture/memory_replay.md`
- `docs/sections/section_09_memory_and_replay.md`

## STEP E. Implementation Notes

- `memory/trace_store.py`
  - defines `TraceRecord` and `TraceStore`
  - captures stage, energy, graph summary counts, plasticity-trace counts, and optional compartment IDs
- `memory/replay_buffer.py`
  - defines `ReplayItem` and `ReplayBuffer`
  - provides deterministic latest, highest-score, and tag-based replay views
- `memory/episode_registry.py`
  - defines `EpisodeKind`, `EpisodeStatus`, `EpisodeRecord`, and `EpisodeRegistry`
  - keeps ordered state windows explicit with validated step progression

Key design choices:

- trace memory stores one reusable record per state id
- replay ordering is deterministic at the foundation stage
- episode membership is explicit rather than inferred heuristically
- states may belong to multiple episodes across different analytical contexts
- graph, plasticity, and compartment context are summarized into memory, not owned by it

## STEP F. Validation Strategy

Section 9 validation covers:

- trace insertion from registry-backed states
- replay ordering and capacity behavior
- episode integrity and lookup
- clean compatibility with Sections 1-8

Commands:

```bash
python3 scripts/validate_scaffold.py
python3 -m unittest discover -s tests
```

## STEP G. Documentation Updated

- `docs/architecture/memory_replay.md`
- `docs/sections/section_09_memory_and_replay.md`
- `progress_info/*.md`

## STEP H. Progress Files Updated

All mandatory continuity files were updated to reflect Section 9 completion and
the transition to Section 10.

## STEP I. Handoff Summary

SECTION HANDOFF
- Section completed: Section 9. Memory and replay system
- Purpose: provide reusable historical records, deterministic replay, and explicit episode grouping above lineage and plasticity
- Files created: trace store, replay buffer, episode registry, tests, docs, and continuity updates
- Key interfaces: `TraceRecord`, `TraceStore`, `ReplayItem`, `ReplayBuffer`, `EpisodeRecord`, `EpisodeRegistry`
- Dependencies: `core/state_registry.py`, `graph/graph_manager.py`, `plasticity/traces.py`, `compartments/registry.py`
- What now works: registry-backed trace capture, deterministic replay selection, episode opening/appending/closure, and cross-layer context summaries in memory records
- What is still stubbed: qcloud-targeted memory, learned replay prioritization, memory-conditioned control, and long-horizon compression or indexing
- Risks / unresolved items: replay is deterministic and simple by design in Section 9; richer prioritization should come later with explicit validation
- Next recommended section: Section 10. Quantum-cloud framework
- Important continuity notes: Section 10 should consume memory as a client rather than embedding memory ownership into qcloud logic
