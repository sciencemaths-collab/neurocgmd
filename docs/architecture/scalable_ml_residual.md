# Scalable ML Residual

## Purpose

This module upgrades the Section 11 residual-learning path from a global
state-level learner toward a piece-local, neighbor-coupled architecture.

Classification:

- local residual feature extraction: `[adapted]`
- cell-list neighborhood execution: `[established]`
- piece-local "Piece + Spring" ML interpretation: `[proposed novel]`
- additive residual coupling to the coarse substrate: `[hybrid]`

The goal is not to replace proven atomistic engines with a vague neural claim.
The goal is to make the repo's architecture capable of the same local-message
execution pattern that modern equivariant force fields rely on, while keeping
the project's coarse substrate, qcloud coupling, memory, and executive-control
layers intact.

## Mathematical Role

For each particle `i`:

1. local encoding
   - raw particle features `x_i` are mapped to a hidden representation `h_i`
2. stacked spring interaction blocks
   - for each interaction pass `t`, every neighbor `j` within cutoff emits a
     bounded message `m_{j -> i}^{(t)}` computed from `h_j^{(t)}`, `h_i^{(t)}`,
     and a Gaussian radial basis expansion of `d_ij`
3. degree-normalized aggregation and update
   - each particle collects the mean local message over its neighbors
   - an interaction-update layer mixes the current piece state with that local
     message through a residual update
4. local correction
   - the final propagated representation is concatenated with the initial local
     encoding and mapped to:
     - `e_i`: local energy correction
     - `s_i`: force-scale correction
5. global residual
   - total energy correction is `sum_i e_i`
   - force delta for particle `i` is `s_i * F_i(base)`

Online training keeps the encoder/message layers fixed and updates the local
correction head from qcloud-derived residual targets. This is intentionally a
bounded online adaptation strategy rather than a claim of fully trained
equivariant force-field fidelity.

## Interfaces

### `ml.scalable_residual.ScalableResidualModel`

- classification: `[proposed novel]`
- protocol compatibility:
  - `predict(state, base_evaluation) -> ResidualPrediction`
  - `observe(target, sample_weight=1.0) -> None`
  - `trained_state_count() -> int`
- state-aware extension:
  - `observe_state(state, base_evaluation, target, sample_weight=1.0) -> None`

### `ml.residual_model.StateAwareResidualModel`

- classification: `[adapted]`
- role: optional protocol extension that allows replay or live runtime code to
  pass the explicit state and baseline force block into residual-model updates

### `ml.online_trainer.ReplayDrivenOnlineTrainer.update_from_replay_with_states(...)`

- classification: `[adapted]`
- role: preserve the original replay-driven trainer while enabling explicit
  state-aware updates for local/neighborhood-aware models

## Runtime Integration

- `scripts/live_dashboard.py` now uses `ScalableResidualModel` as the default
  live residual model for protein scenarios
- live qcloud corrections now train the scalable model through
  `observe_state(...)` when available
- the active dashboard therefore exercises:
  - local piece encoders
  - cell-list neighborhood finding
  - stacked spring message passing
  - degree-normalized local aggregation
  - residual interaction updates
  - local correction learning

## Validation Thinking

### What is tested

- fresh untrained predictions remain neutral
- state-aware online training moves predictions toward observed targets
- legacy target-only `observe(...)` remains compatible after a cached forward pass
- replay-driven state-aware trainer updates work
- live dashboard context resolves to the scalable model

### What can still break

- local-head-only online updates may underfit richer force structure
- fixed encoder/message weights may limit expressivity before future offline or
  replay-batch training phases
- imported-protein systems with very different chemistry may still need better
  feature engineering and richer target design

## Continuity Note

Future fidelity work should treat this module as the scalable ML execution
substrate, not as the final scientific claim. The next meaningful steps are:

- add imported-protein systems into the repeated validation loop
- compare scalable residual vs replay-memory residual vs no-ML ablations
- widen the local feature set with bounded chemistry and compartment cues only
  when those additions remain explicit and testable
