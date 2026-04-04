# Compartment System Architecture

## Section 8 Scope

Section 8 introduces compartments as modular overlays on top of particle, topology,
and graph state. This section is classified as `[hybrid]` because compartmental
organization is a common modeling idea, but here it is being positioned as a
stable architectural layer that later plasticity, memory, and control systems can
interact with explicitly.

## Mathematical Role

The compartment overlay is modeled as:

`C = { C_1, C_2, ..., C_m }`

where each compartment is a subset of particle-aligned vertices:

`C_i subseteq {0, ..., N-1}`

This is distinct from:

- fixed topology connected components
- adaptive graph connected components
- future memory episodes

Compartments are semantic and modular, not merely incidental graph partitions.

## Foundation Design

- compartments are explicit particle-index sets
- a registry manages membership and validates overlay integrity
- a routing layer classifies graph edges as intra- or inter-compartment
- topology hints can bootstrap compartments, but the registry remains the source of truth

## Core Invariants

- compartment IDs are unique
- particle indices must lie inside the particle-count boundary
- overlap is disallowed by default
- compartments do not replace topology or graph state
- routing summaries are descriptive in Section 8, not yet executive control policies

## Validation Thinking

We test Section 8 by checking:

- manual and topology-hint registry construction
- overlap rejection when overlap is disabled
- membership lookup and unassigned-particle reporting
- inter-compartment route aggregation over active graph edges

