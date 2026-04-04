# Manifest-Driven MD Workflow

## Classification

- workflow shell: `[adapted]`
- preparation planning: `[adapted]`
- integrated engine routing: `[hybrid]`
- hybrid subsystem orchestration: `[proposed novel]`

## Purpose

This phase turns NeuroCGMD from a research runtime plus dashboard into a
user-facing MD workflow with one control plane:

- `prepare`
- `run`
- `analyze`

The user now drives the system from one TOML manifest rather than manual
scenario wiring. The workflow borrows the familiar staged feel of production MD
packages while keeping the repo's hybrid architecture explicit.

## Inputs

### Structural Input

- local `PDB`
- current implementation uses local protein structures imported through:
  - `io/pdb_loader.py`
  - `topology/protein_coarse_mapping.py`

### Control Input

- `TOML` run manifest
- parsed by `config/run_manifest.py`

The manifest accepts familiar production-MD sections such as:

- `[system]`
- `[solvent]`
- `[ions]`
- `[constraints]`
- `[nonbonded]`
- `[stages.em]`
- `[stages.nvt]`
- `[stages.npt]`
- `[stages.production]`
- `[hybrid.qcloud]`
- `[hybrid.ml]`

## New Contracts

### `config/run_manifest.py`

- `[hybrid]`
- defines the public run-manifest schema
- key contracts:
  - `RunManifest`
  - `SystemConfig`
  - `PrepareConfig`
  - `ForcefieldConfig`
  - `NeighborListConfig`
  - `MinimizationStageConfig`
  - `DynamicsStageConfig`
  - `HybridConfig`
  - `OutputConfig`
  - `AnalysisConfig`
  - `load_run_manifest(...)`

### `prepare/`

- `[adapted]`
- explicit preparation planning layer
- key modules:
  - `prepare/protonation.py`
  - `prepare/solvation.py`
  - `prepare/ions.py`
  - `prepare/pipeline.py`
- key contracts:
  - `PreparedSystemBundle`
  - `PreparedRuntimeSeed`
  - `ProtonationPlan`
  - `SolvationPlan`
  - `IonPlacementPlan`

### `sampling/stage_runner.py`

- `[hybrid]`
- runs staged EM/NVT/NPT/production execution on top of the shared
  `HybridProductionEngine`
- key contracts:
  - `StageRecord`
  - `ProductionRunSummary`
  - `ProductionStageRunner`

### `scripts/neurocgmd.py`

- `[adapted]`
- unified CLI entry point
- subcommands:
  - `prepare`
  - `run`
  - `analyze`

### `io/trajectory_writer.py` and `io/checkpoint_writer.py`

- `[adapted]`
- import-safe path-loaded artifact writers
- current artifact formats:
  - trajectory: JSONL
  - checkpoint: JSON

## Current Workflow

### Prepare

`prepare` does the following:

1. load the TOML manifest
2. import the PDB into the existing arbitrary-protein pipeline
3. infer entity groups if the user did not provide them
4. estimate protonation and hydrogen addition needs
5. estimate solvent box and water count
6. estimate neutralization and bulk-salt ions
7. emit a `PreparedSystemBundle`

### Run

`run` does the following:

1. load the TOML manifest
2. load or create the prepared bundle
3. build an imported-protein scenario from the prepared system
4. seed the shared `HybridProductionEngine` with the prepared runtime state
5. run staged EM/NVT/NPT/production execution through the shared engine
6. emit trajectory, checkpoint, energy log, run summary, and run log artifacts

### Analyze

`analyze` does the following:

1. load the TOML manifest
2. load the prepared bundle
3. load the latest checkpoint
4. rebuild the same imported-protein scenario
5. run observer-side structure, fidelity, chemistry, and controller analysis
6. emit JSON + HTML analysis reports

## What Is Honest Today

This workflow is real and runnable, but it is intentionally honest about the
current scientific boundary.

### Implemented

- manifest-driven workflow
- imported-protein preparation bundle
- hydrogen / protonation planning
- solvent / water-count planning
- salt / neutralization planning
- shared staged run path through the integrated production engine
- trajectory, checkpoint, energy, summary, and analysis artifacts
- single-entity and multi-entity imported protein support at the scenario level

### Not Yet Implemented

- full atomistic hydrogen placement
- full explicit water packing
- true ion placement coordinates
- full barostat-driven NPT box evolution
- XTC/DCD/GRO/TOP-equivalent production file outputs
- external-forcefield/topology export compatible with other MD packages

So the current `prepare` layer is a scientifically explicit planning layer, not
yet a drop-in replacement for mature atomistic packers.

## Why This Fits The Architecture

The important architectural choice is that the new public workflow does not
create a second engine. It routes through the same internal spine:

- imported-protein mapping
- hybrid force engine
- adaptive graph
- chemistry analysis
- memory / replay
- qcloud refinement
- ML residuals
- executive control
- validation observers

That means future fidelity upgrades should strengthen the same path rather than
forking into separate "CLI mode" and "research mode" implementations.

## Validation

The phase is validated by:

- manifest parsing tests
- preparation bundle tests
- single-entity imported-protein scenario tests
- end-to-end `prepare -> run -> analyze` CLI tests

## Next Correct Upgrades

- replace planning-only hydrogen and solvent estimates with explicit coordinate builders
- add native production output formats beyond JSON/JSONL
- make NPT physically real with box updates rather than metadata-only pressure targets
- let imported-protein manifests enter repeated validation and transfer-tuning directly
- benchmark CLI-run imported proteins across more than the current bundled structures
