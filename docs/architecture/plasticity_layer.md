# Plasticity Layer Architecture

## Section 7 Scope

Section 7 introduces bounded plasticity rules on top of the adaptive graph layer.
This section is classified as `[proposed novel]` because it is where the project
begins to express brain-inspired reinforcement, weakening, pruning, and growth as
explicit graph-update mechanisms rather than static connectivity.

## Mathematical Role

Given a graph snapshot `G_dyn(k)` and pair-level traces `M_pair(k)`, Section 7
applies a bounded update:

`(G_dyn(k), M_pair(k)) -> (G_dyn'(k), M_pair(k+1))`

The current foundation pass separates the mechanisms:

- trace accumulation
  - exponentially decayed activity and coactivity memories
- reinforcement / weakening
  - bounded updates to adaptive edge weights
- pruning
  - deactivation of weak unsupported adaptive edges
- Hebbian growth
  - addition of new adaptive edges for strongly coactive nearby pairs

## What Is Established vs Novel

- `[established/adapted]`
  - Hebbian inspiration
  - bounded weight updates
  - decayed activity traces
  - pruning of weak unsupported edges
- `[proposed novel]`
  - expressing these rules as a persistent graph-plasticity layer coupled to the
    molecular simulation architecture
  - keeping plasticity explicit and modular so later memory, uncertainty, and AI
    control can act on the same graph substrate

## Core Invariants

- fixed topology remains untouched
- structural graph edges are never pruned or reweighted by Section 7 rules
- adaptive edge weights remain bounded
- pruning deactivates edges rather than deleting structural truth
- growth is capped per step to avoid uncontrolled graph explosion

## Validation Thinking

We test Section 7 by checking:

- pair traces update deterministically from signals
- reinforcement increases adaptive weights under strong support
- pruning deactivates weak unsupported edges
- Hebbian growth adds new edges only for sufficiently coactive nearby pairs
- the composed plasticity engine preserves graph invariants

