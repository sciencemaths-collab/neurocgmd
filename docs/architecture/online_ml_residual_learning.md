# Online ML Residual Learning Architecture

## Section 11 Scope

Section 11 introduces the first online ML residual-learning layer. This section is
classified as `[hybrid]` because additive residual learning and uncertainty hooks
are established ideas, while their explicit placement above replay memory and
beside qcloud corrections is a deliberate architectural choice for this project.

## Mathematical Role

Section 11 adds three linked objects:

- residual targets
  - `Y_res = { y_1, y_2, ..., y_n }`
- residual predictions
  - `Y_hat_res = f_theta(S_k, F_base)`
- uncertainty estimates
  - `U = g(Y_hat_res, M_replay, M_trace)`

In Section 11 the learned correction remains additive:

- `F_total = F_base + F_ml`
- `E_total = E_base + E_ml`

This keeps learned residuals explicitly separate from:

- established baseline physics in `physics/`
- qcloud corrections in `qcloud/`

## Foundation Design

- `ResidualTarget`
  - one observed additive correction target for one state
- `ResidualPrediction`
  - one predicted additive correction payload for one state
- `ResidualMemoryModel`
  - replay-driven residual model with exact-state recall and global-mean fallback
- `ResidualAugmentedForceEvaluator`
  - wraps a baseline force evaluator with additive ML residual corrections
- `UncertaintyEstimate`
  - structured uncertainty output with qcloud-trigger recommendation
- `HeuristicUncertaintyModel`
  - deterministic uncertainty estimator used before calibration infrastructure exists
- `ReplayDrivenOnlineTrainer`
  - consumes replay-buffer items and aligned residual targets to update the residual model

## Core Invariants

- learned residuals remain additive
- baseline force accounting remains explicit
- qcloud corrections and ML residuals are distinct layers
- replay storage remains owned by `memory/`
- trainer logic consumes replay items but does not own replay storage
- Section 11 uncertainty is heuristic and should not be presented as calibrated

## Validation Thinking

We test Section 11 by checking:

- qcloud corrections can be aggregated into residual targets
- replay-driven residual observations produce deterministic exact-state predictions
- ML residuals attach additively to baseline force evaluations
- heuristic uncertainty can flag low-confidence priority states for qcloud fallback
- online trainer updates residual models from replay-backed examples without mutating memory ownership
