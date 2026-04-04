# State Model Architecture

## Section 2 Scope

Section 2 defines the canonical simulation state representation that every later
module will consume. This section is classified as `[adapted]` because it uses
standard immutable-data and provenance patterns, but adapts them to the needs of a
long-horizon molecular platform with later graph, memory, ML, and control layers.

## Mathematical Role

At time index `k`, the state snapshot is modeled as:

`S_k = (U, X_k, V_k, F_k, m, T_k, C_k, P_k, O_k)`

Where:

- `U` is the unit system metadata
- `X_k in R^(N x 3)` are particle positions
- `V_k in R^(N x 3)` are particle velocities
- `F_k in R^(N x 3)` are particle forces
- `m in R^N` are particle masses
- `T_k` is the thermodynamic control state
- `C_k` is the optional periodic simulation cell
- `P_k` is provenance and lineage metadata
- `O_k` is a frozen observable/metadata map

This is intentionally topology-agnostic. Section 2 does not decide what particles
mean chemically or graph-theoretically. It only guarantees that downstream
algorithms receive a stable, validated, units-aware snapshot.

## Architectural Boundaries

### Included in Section 2

- immutable particle state
- thermodynamic controls
- unit metadata
- optional periodic cell
- provenance and lineage identity
- checkpoint/registry helpers
- serialization-safe dictionaries

### Explicitly Excluded from Section 2

- bonded topology semantics
- force-field parameters
- graph connectivity and adaptive rewiring
- memory replay policies beyond basic lineage storage
- quantum-cloud regions
- ML tensors and model weights
- AI control policies

## Core Invariants

- particle arrays must have shape `N x 3`
- masses must have length `N` and be strictly positive
- time and step must be non-negative
- provenance identifiers must be non-empty and lineage-consistent
- metadata must be recursively immutable once attached to a state
- a registry child state cannot precede its parent in time or step

## Serialization Boundary

All Section 2 objects expose `to_dict()` and `from_dict()` methods using only
standard JSON-friendly container shapes:

- vectors become lists
- tuples become lists in the serialized form
- immutable metadata is thawed back into ordinary dict/list structures

This keeps the future `io/` layer simple and avoids hiding serialization logic in
ad hoc helper code.

## Validation Thinking

We verify Section 2 through:

- shape checks for particle-resolved arrays
- ensemble rule checks for thermodynamic state
- round-trip serialization tests
- lineage and summary checks in the state registry
- rejection tests for invalid metadata and invalid parent/child ordering

