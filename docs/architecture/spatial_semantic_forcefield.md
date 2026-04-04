# Spatial Semantic Forcefield

## Classification

- `forcefields/spatial_semantic_profiles.py`: `[hybrid]`
- `qcloud/spatial_semantic_field.py`: `[proposed novel]`
- integration through `qcloud/shadow_correction.py`: `[proposed novel]`

## Purpose

This phase adds a new spatially aware, distance-window-aware, chemistry-aware
local force layer below control and above the coarse substrate.

It is not a replacement for the baseline bonded or Lennard-Jones terms. It is a
repository-owned intelligent correction layer designed to capture interaction
logic that mature engines often express through richer parameterization,
distance-sensitive nonbonded behavior, and careful local geometry handling.

## Scientific Intent

The design takes inspiration from established simulation practice without
copying one engine's implementation:

- distance-sensitive pair interactions
- explicit preferred-distance windows
- short-range exclusion / repulsion
- charge- and chemistry-modulated attraction
- local geometry or exposure awareness
- additive correction rather than hidden replacement

The novelty is the combination:

- fast coarse body
- mirrored shadow cloud
- chemistry semantics
- spatial-semantic local field
- explicit AI/qcloud orchestration

## Mathematical Role

Each bead-type pair receives a `SpatialSemanticProfile` with:

- preferred distance
- distance tolerance
- minimum distance
- cutoff
- attraction strength
- repulsion strength
- directional strength
- chemistry strength

Inside one qcloud refinement region, the field model computes for each active
pair:

- `distance`
- `alignment`
- `chemistry_match`

The pair energy is a bounded combination of:

- a preferred-distance attraction well
- a short-range repulsion wall
- alignment scaling from local exposure directions
- chemistry scaling from residue/bead descriptors

The current implementation uses:

- Gaussian-like attraction around the preferred distance
- steep repulsion below the minimum distance
- outward local exposure vectors from bonded neighborhoods
- bounded chemistry match derived from charge, hydropathy, aromaticity, hotspot propensity, and hydrogen-bond capacity

## Architectural Role

The dependency path is:

- `forcefields/protein_shadow_profiles.py` defines reusable protein families
- `forcefields/spatial_semantic_profiles.py` converts those families into spatial interaction priors
- `qcloud/spatial_semantic_field.py` evaluates the new local field inside a refinement region
- `qcloud/shadow_correction.py` merges that field with the mirrored shadow-site correction path
- `qcloud/protein_shadow_tuning.py` packages the whole thing into the protein-general runtime bundle

This keeps the architecture clean:

- baseline coarse physics still owns the substrate
- qcloud/shadow owns high-fidelity local correction
- AI control stays above the force path
- validation remains observer-side

## Interfaces

Key contracts:

- `SpatialSemanticProfile`
- `SpatialSemanticParameterSet`
- `ProteinSpatialProfileFactory`
- `SpatialSemanticFieldPolicy`
- `SpatialSemanticPairContribution`
- `SpatialSemanticFieldEvaluation`
- `SpatialSemanticFieldModel`

## Validation Strategy

Current validation confirms:

- spatial-semantic parameter sets build from protein topologies
- the field model produces nonzero local energy/force corrections
- the protein shadow tuner now carries spatial-semantic parameters and policy
- shadow correction remains compatible with and extendable by the new field layer

## Boundaries

- this is still a bounded local correction layer, not a full all-atom force engine
- the current geometry awareness is based on bonded-neighbor exposure, not full multipole orientation
- the current field should be treated as a scientifically motivated foundation to tune and benchmark, not as a final production-grade universal forcefield claim

## Next Work

- tune spatial priors across multiple protein benchmarks
- compare repeated validation with and without the new spatial field
- expand geometry awareness beyond bonded-neighbor exposure where justified
- connect the new field to backend-parity studies later without giving up repository ownership
