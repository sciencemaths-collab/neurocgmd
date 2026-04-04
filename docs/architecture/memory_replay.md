# Memory and Replay Architecture

## Section 9 Scope

Section 9 introduces the longer-horizon memory layer. This section is classified
as `[hybrid]` because replay buffers and episode windows are established software
patterns, while their explicit integration with state lineage, adaptive graph
state, plasticity traces, and compartment overlays is a deliberate architectural
choice for this platform.

## Mathematical Role

Section 9 adds reusable historical objects above the core state lineage:

- trace memory
  - `M_trace = { m_1, m_2, ..., m_t }`
- replay memory
  - `M_replay subseteq M_trace`
- episode memory
  - `E = { e_1, e_2, ..., e_k }`

These are intentionally distinct from:

- provenance lineage in `core/state_registry.py`
- short-horizon pair traces in `plasticity/traces.py`

Lineage answers "which state produced this one?"

Memory answers "which historical objects should later systems revisit, reuse, or
organize into windows?"

## Foundation Design

- `TraceRecord`
  - immutable per-state memory record
  - stores stage, step, time, energy summary, active graph counts, plasticity-trace
    counts, and optional compartment IDs
- `TraceStore`
  - append-only long-horizon store keyed by `state_id`
  - can build records directly from `SimulationStateRegistry`
- `ReplayItem`
  - immutable replay selection record
  - stores replay priority score, temporal location, and optional episode linkage
- `ReplayBuffer`
  - deterministic capacity-bounded replay selection layer
  - supports latest, highest-score, and tag-based retrieval
- `EpisodeRecord`
  - immutable ordered state window with explicit step progression
- `EpisodeRegistry`
  - manages opening, extending, and closing grouped windows such as trajectories
    and instability intervals

## Core Invariants

- memory does not own state evolution
- `SimulationStateRegistry` remains the owner of lineage
- `plasticity/traces.py` remains the owner of short-horizon pair traces
- trace records are unique per `state_id`
- replay selection is deterministic in Section 9
- episode ordering is explicit and validated
- one state may belong to multiple episodes across different analytical contexts

## Validation Thinking

We test Section 9 by checking:

- trace insertion from registry-backed states
- graph, plasticity, and compartment summaries are attached without mutating those layers
- replay ordering is deterministic for latest and highest-score retrieval
- replay capacity bounds are enforced
- episode opening, appending, closure, and lookup remain consistent
