# Dynamic Graph Architecture

## Section 6 Scope

Section 6 introduces the first adaptive connectivity graph layer. The section is
classified as `[proposed novel]` because the long-term novelty of the platform
depends on maintaining a dynamic interaction graph on top of fixed topology rather
than collapsing everything into a single static force-field graph.

## Mathematical Role

The fixed topology from Section 3 remains:

`T_fixed = (B, T, p, E_fixed)`

Section 6 adds a dynamic graph:

`G_dyn(k) = (V, E_dyn(k), w(k), tau(k))`

Where:

- `V = {0, ..., N-1}` are particle-aligned vertices
- `E_dyn(k)` are time-dependent edges
- `w(k)` are edge weights
- `tau(k)` are edge kinds such as structural-local, adaptive-local, or adaptive-long-range

The current foundation rule is conservative:

- structural edges are copied from fixed topology
- adaptive edges are proposed from explicit distance bands
- prior weights can contribute inertia to avoid fully memoryless updates

## What Is Novel vs Conservative

- `[established/adapted]`
  - distance-based neighborhood rules
  - adjacency maps
  - connected-component queries
- `[proposed novel]`
  - the architectural separation between fixed topology and adaptive graph state
  - the intent to let later plasticity, memory, uncertainty, and control layers
    modify `G_dyn(k)` without rewriting the physical substrate

## Core Invariants

- fixed topology edges and adaptive graph edges remain conceptually separate
- each undirected pair appears at most once in the adaptive graph
- graph particle count must match `SimulationState`
- structural-local edges are retained from topology when enabled
- graph updates are explicit and reproducible for a given state and rule configuration

## Validation Thinking

We test Section 6 by checking:

- structural, adaptive-local, and adaptive-long-range edges are created correctly
- duplicate graph edges are rejected
- adjacency maps and connected components are correct
- manager updates respond to state changes without mutating fixed topology

