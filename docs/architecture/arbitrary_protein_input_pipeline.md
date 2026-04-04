# Arbitrary Protein Input Pipeline

## Classification

- import-safe PDB parsing: [established]
- residue-block coarse mapping: [adapted]
- imported protein reference scaffold generation: [hybrid]
- imported-protein live scenario wiring: [hybrid]

## Purpose

This pipeline lets the repository ingest an arbitrary local protein complex PDB,
map it into the project's coarse substrate, generate a matching reference
scaffold, and feed that imported system into the same shadow-fidelity and live
dashboard architecture used by the curated benchmark scenarios.

## Main Components

- `config/protein_mapping.py`
  - bounded import and mapping configuration
- `topology/protein_import_models.py`
  - import-time residue, bead-block, and materialized imported-system contracts
- `topology/protein_coarse_mapping.py`
  - local PDB to coarse bead blocks, topology, particles, compartments, and reference scaffold
- `forcefields/protein_import_forcefield.py`
  - baseline imported-protein force-field builder using protein-general priors
- `sampling/scenarios/imported_protein.py`
  - scenario wrapper that turns imported systems into live dashboard / validation scenarios
- `scripts/import_protein_system.py`
  - CLI entrypoint that writes a reusable JSON import summary

## Data Flow

1. load a local PDB through `io/pdb_loader.py`
2. select user-declared chain groups as semantic entities
3. collapse residues into contiguous bead blocks
4. infer bead families and bead types from aggregated chemistry
5. build `ParticleState`, `SystemTopology`, and `CompartmentRegistry`
6. derive a local imported reference scaffold from the imported bead blocks
7. build a baseline coarse force field plus shared shadow runtime bundle
8. run imported complexes through the same validation and dashboard flow as curated scenarios

## Validation Thinking

- direct mapper tests should verify entity grouping, bead counts, and reference-target alignment
- fresh-process CLI tests should verify import safety outside the test process import graph
- imported scenarios should be able to build `ComplexAssemblySetup` without special-casing the dashboard
- transfer tuning should accept imported scenario specs without changing the shared scoring path

## Current Limits

- the imported reference scaffold is still coarse residue-block centroids, not full all-atom force truth
- imported scenarios currently assume complex-style multi-entity inputs rather than single isolated proteins
- broader transfer claims still require more imported protein systems in the benchmark panel
