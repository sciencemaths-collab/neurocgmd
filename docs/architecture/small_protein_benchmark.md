# Small Protein Benchmark

## Purpose

This benchmark gives the repository one fast, real-structure regression target
that can be run in seconds while still exercising the modern engine stack.

It uses:

- local PDB import through the arbitrary-protein pipeline
- one real small protein chain from `1BRS`
- the imported-protein force-field builder
- the hybrid force engine
- the adaptive connectivity graph
- trace, replay, and episode memory
- executive control for bounded qcloud and ML allocation
- the shadow correction path
- the scalable piece-local residual model
- backend parity checks
- backend execution planning

## Current Default Case

- structure source: `1BRS`
- benchmark entity: barnase chain `A`
- benchmark mode: single-chain stability/runtime benchmark
- default coarse mapping: `10` residues per bead

This is intentionally a small and fast harness. It is not a claim of
experimental folding fidelity. Its role is to provide a compact,
repeatable benchmark for:

- classical vs interconnected hybrid-engine timing
- backend-spine regression checks
- small-protein importer coverage
- future backend parity and acceleration work

## Report Contents

`benchmarks/small_protein.py` returns a structured report with:

- top-line engine-mode summary:
  - `classical_only`
  - `hybrid_production` with display label `production_hybrid_engine`
- imported residue and bead counts
- benchmark timings for:
  - `diagnostic_reference_classical_baseline`
  - `classical_only`
  - `diagnostic_shadow_only`
  - `production_hybrid_engine`
  - `diagnostic_graph_update_single_chain`
  - `diagnostic_production_rollout`
- `production_hybrid_engine` metadata proving whether the graph, replay, controller,
  qcloud selector, and ML residual path all participated in the measured cycle:
  - `preliminary_action`
  - `final_action`
  - `qcloud_requested`
  - `qcloud_applied`
  - `selected_region_count`
  - `trace_record_count`
  - `replay_buffer_size`
  - `open_episode_count`
- backend parity metrics for the hybrid classical path
- backend execution plan metadata
- the recommended large-step dynamics settings from the protein shadow tuner

## Plot Output

`scripts/plot_small_protein_benchmark.py` writes:

- `benchmark.json`
- `index.html`

The HTML plot intentionally shows:

- `classical_only` vs `hybrid_production` as the top-line engine comparison
- the detailed ablation slices underneath for diagnostics

That keeps the benchmark user-facing language aligned to one engine while still
preserving internal performance breakdowns.

## Scientific Honesty Boundary

This benchmark is:

- real-structure-backed
- fast enough for repeated regression use
- useful for backend and hybrid-engine comparisons
- now wired so the top-line `production_hybrid_engine` mode is one interconnected
  benchmark path rather than a disconnected force-stack slice

This benchmark is not:

- a full folding benchmark
- a calibrated thermodynamic observable
- a claim of atomistic predictive power

## Next Recommended Extensions

- add at least one more single-chain imported protein
- compare the same benchmark under future non-reference backends
- track benchmark results over time in validation dashboards
