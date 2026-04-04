# Protein-General Shadow Tuning

## Classification

- `ProteinShadowProfileFactory`: `[hybrid]`
- `ProteinShadowTuningPreset`: `[hybrid]`
- `ProteinShadowTuner`: `[hybrid]`
- scenario integration into `spike_ace2` and `barnase_barstar`: `[hybrid]`

## Purpose

This layer moves the shadow coarse-grained architecture away from one
benchmark-specific hand-tuned table and toward a protein-general foundation.

The immediate goals are:

- reuse one shadow architecture across many protein systems
- keep the coarse body as the main dynamical substrate
- keep time steps large enough for fast coarse simulation
- keep shadow corrections bounded enough to avoid turning into a stiff hidden all-atom clone

## What It Adds

### Protein-family priors

`forcefields/protein_shadow_profiles.py` classifies bead types into broad protein
chemistry families such as:

- hydrophobic core
- polar surface
- basic patch
- acidic patch
- aromatic hotspot
- flexible linker
- shielded surface

The factory then generates trusted shadow interaction profiles for the actual bead
types in one topology.

### Large-step stability preset

`qcloud/protein_shadow_tuning.py` adds a stability-biased preset that:

- increases the minimum site distance
- caps interaction range
- damps shadow energy/electrostatic scaling
- recommends a larger coarse time step with higher friction

This is intentionally not a fidelity-maximizing preset. It is a
speed-and-stability preset for the shadow coarse-grained architecture.

### Shared runtime bundle

The tuner emits one reusable runtime bundle containing:

- trusted parameter set
- shadow mapping library
- bounded correction policy
- recommended large-step dynamics settings

That bundle is now used by both:

- `sampling/scenarios/spike_ace2.py`
- `sampling/scenarios/barnase_barstar.py`

## Scientific Boundary

This is not a claim that the system now works for all proteins in a validated,
production-grade sense.

What is true now:

- the shadow architecture is no longer spike-specific
- two different protein benchmarks now share the same protein-general shadow layer
- the coarse simulation path now uses a large-step stability preset rather than only a hand-tuned benchmark table

What is still not true:

- universal protein transferability is not proven
- calibrated thermodynamics are not solved
- atomistic parity is not solved
- production protein-folding or docking accuracy is not solved

## Why This Matters

The project goal is not to clone one existing engine.

The goal is to let established science inform a different architecture:

- coarse body for speed
- shadow cloud for bounded higher-fidelity local correction
- graph/memory/controller layers for adaptive orchestration

General protein shadow tuning is the first step that makes this architecture look
like a reusable protein simulation platform instead of a one-benchmark prototype.
