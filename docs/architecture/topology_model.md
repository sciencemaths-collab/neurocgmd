# Topology Model Architecture

## Section 3 Scope

Section 3 introduces the static topology substrate that sits on top of the Section 2
particle-state layer. This section is classified as `[adapted]` because it uses
standard graph and coarse-grained topology ideas, but adapts them to the repository's
future distinction between fixed physical topology and adaptive graph plasticity.

## Mathematical Role

The fixed topology is modeled as:

- a bead set `B`
- a particle-index alignment map `p: B -> {0, ..., N-1}`
- a bead-type map `tau: B -> T`
- a static undirected bond set `E_fixed subseteq {0, ..., N-1} x {0, ..., N-1}`

The topology object therefore defines:

`T_fixed = (B, T, p, E_fixed)`

This is intentionally separate from the later adaptive graph layer:

- `T_fixed` encodes the physical or structural substrate
- the future graph layer will encode dynamic or learned interaction pathways

## Core Invariants

- one bead maps to exactly one particle index
- particle indices are contiguous from `0` to `N-1`
- bead IDs are unique
- bead types are declared before they are referenced by beads
- bonds are undirected and duplicate-free
- topology can be checked directly against `ParticleState`

## Included in Section 3

- bead types
- bead descriptors
- static bonds
- neighbor-map construction
- connected-component queries
- system-level topology assembly and validation

## Explicitly Excluded from Section 3

- force constants as force-field truth
- adaptive rewiring or dynamic graph updates
- plasticity rules
- compartment routing policies
- trajectory replay

## Validation Thinking

We test Section 3 by checking:

- unique and contiguous particle-index assignment
- duplicate bond rejection
- connected-component correctness
- round-trip serialization
- alignment checks against Section 2 `ParticleState`

