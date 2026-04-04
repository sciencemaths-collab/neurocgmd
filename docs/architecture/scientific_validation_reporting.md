# Scientific Validation Reporting

## Classification

- `ScientificValidationRunner`: `[hybrid]`
- `ScientificValidationReport`: `[hybrid]`
- `render_scientific_validation_report(...)`: `[adapted]`

## Purpose

This layer turns the live benchmark stack into a repeatable scientific-validation
artifact instead of a single dashboard frame. It exists to answer:

- how the current architecture behaves over time
- whether shadow correction helps or hurts relative to the trusted target
- how much atomistic-reference agreement the current coarse benchmark is retaining
- how expensive the main architecture pathways are while those metrics evolve

## Mathematical and Architectural Role

The report aggregates sampled trajectory metrics over repeated runs.

For each sampled state it records:

- assembly score and interface gap from the scenario progress model
- atomistic-centroid RMSD and contact recovery from the structure-comparison layer
- baseline versus shadow error against the trusted force target
- architecture timing from the Section 13 benchmark suite

The runner then groups samples by sampled simulation step and computes:

- mean
- minimum
- maximum

for each plotted metric.

This gives a bounded observer-side summary:

- scientific behavior over time
- correction quality over time
- architecture timing over time

without moving ownership away from `sampling/`, `qcloud/`, `validation/`, or
`benchmarks/`.

## Files

- `validation/scientific_validation.py`
- `visualization/validation_report_views.py`
- `scripts/plot_scientific_validation.py`

## Main Interfaces

- `ScientificValidationSample`
  - one sampled state with scientific and timing metrics
- `ValidationSeriesPoint`
  - one aggregate mean/min/max point at one sampled step
- `ValidationSeries`
  - one named plotted metric
- `ScientificValidationSummary`
  - compact final report summary
- `ScientificValidationReport`
  - full structured batch-validation artifact
- `ScientificValidationRunner.run(...)`
  - repeated trajectory sampling and aggregation
- `render_scientific_validation_report(...)`
  - standalone HTML/SVG rendering for local inspection

## Current Scientific Boundary

This reporting layer is honest about what the program is and is not doing.

It currently validates:

- coarse-grained benchmark progress
- local atomistic-reference agreement
- trusted-target shadow-fidelity deltas
- architecture timing from the internal benchmark suite

It does not yet validate:

- full all-atom dynamics
- calibrated thermodynamics
- production kinetics
- experimental observables beyond the current benchmark scaffold

## Default Usage

```bash
python3 scripts/plot_scientific_validation.py \
  --output-dir /tmp/neurocgmd_scientific_validation
```

The script writes:

- `index.html`
- `validation.json`

to the requested output directory.
