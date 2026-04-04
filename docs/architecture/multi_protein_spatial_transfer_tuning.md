# Multi-Protein Spatial Transfer Tuning

## Classification

- shared transfer runner: `[hybrid]`
- score aggregation policy: `[adapted]`
- shared shadow-prior reuse across proteins: `[proposed novel]`

## Purpose

The spatial-semantic shadow layer must not be tuned against one benchmark only.
This phase adds an explicit multi-protein transfer-tuning path so one shared
spatial prior set can be evaluated across:

- `spike_ace2`
- `barnase_barstar`

The goal is not to hide a benchmark-specific sculpture inside the default
preset. The goal is to keep one reusable protein-shadow architecture and make
its transfer quality measurable.

## New Modules

- `benchmarks/reference_cases/barnase_barstar_structure_targets.py`
  - derives a local `1BRS` centroid scaffold from real atom records
- `validation/protein_transfer_tuning.py`
  - runs shared-preset tuning across multiple protein benchmarks
- `scripts/tune_spatial_transfer.py`
  - exports a JSON report for candidate sweeps

## Architectural Role

The new flow stays within the existing ownership boundaries:

- scenarios still own benchmark-specific structure and fidelity targets
- `scripts/live_dashboard.py` still owns live context assembly
- `validation/scientific_validation.py` still owns repeated-sampling contracts
- `validation/protein_transfer_tuning.py` only compares candidate presets across
  multiple scenarios

This prevents transfer tuning from mutating the simulation loop or hiding
scenario logic inside the tuner.

## Shared-Preset Surface

`qcloud/protein_shadow_tuning.py` now exposes explicit shared-preset knobs for
the spatial layer:

- `spatial_profile_preferred_distance_scale`
- `spatial_profile_distance_tolerance_scale`
- `spatial_profile_attraction_scale`
- `spatial_profile_repulsion_scale`
- `spatial_profile_directional_scale`
- `spatial_profile_chemistry_scale`
- existing local-field policy knobs such as `spatial_energy_scale`,
  `spatial_repulsion_scale`, and `spatial_alignment_floor`

`forcefields/spatial_semantic_profiles.py` applies those scales when building
the reusable `SpatialSemanticParameterSet`.

## Scoring Logic

The current transfer score is a weighted blend of:

- force RMS improvement rate
- max force-component improvement rate
- full-shadow improvement rate
- energy improvement rate
- final contact recovery mean
- final assembly score mean
- normalized atomistic-centroid RMSD quality
- runtime efficiency

This is intentionally explicit and bounded. It is not an opaque optimizer.

## First Two-Protein Sweep

The first deterministic sweep on April 3, 2026 used:

- `spike_ace2`
- `barnase_barstar`
- one replicate per scenario
- twelve steps per replicate
- a 16-candidate grid over attraction, repulsion, chemistry, and alignment

The important result was methodological:

- the multi-protein sweep now works end to end
- `barnase_barstar` participates with real `1BRS` structure and shadow-fidelity reports
- the tested candidate band was very flat, with no strong universal improvement
  over the current default preset

That means the next tuning pass should widen the search dimensions rather than
pretending the first sweep already found a universally better preset.

## Wider Policy Sweep

A wider deterministic sweep on April 3, 2026 expanded the panel to include:

- profile attraction scale
- profile repulsion scale
- profile directional scale
- profile chemistry scale
- local-field energy scale
- local-field repulsion scale
- alignment floor
- an explicit baseline candidate

That widened sweep still reported the baseline preset as the best grounded
candidate on the current two-protein panel.

This is an important result, not a failure:

- the wider transfer machinery now works
- local-field policy scales are now part of the shared tuning surface
- the current two-protein score surface remains very flat near baseline
- there is still no honest justification to change the default shared preset

So the next improvement needs to come from:

- broader benchmark coverage
- longer trajectories
- stronger observables
- or larger changes to the trusted-prior architecture

not from pretending a tied candidate is a real upgrade.

## Guardrails

- do not tune one scenario in ways that break the other and still call it
  protein-general
- do not hide tuning logic inside scenario code
- do not let transfer tuning own execution, qcloud selection, or controller
  mutation
- do not overclaim a better default preset unless it beats the current baseline
  by a meaningful margin across multiple proteins

## Recommended Next Extension

The next transfer-tuning pass should expand along at least one of these axes:

- wider spatial profile scales
- local-field energy and repulsion policy scales
- longer trajectories
- additional protein benchmarks beyond the current two-case panel
