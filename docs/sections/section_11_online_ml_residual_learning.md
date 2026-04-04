# Section 11: Online ML Residual Learning

## STEP A. Section Name and Objective

Section name: `Online ML residual learning`

Objective:

- define additive residual targets and predictions
- define explicit uncertainty-estimation hooks
- define replay-driven online training logic
- keep ML residuals distinct from both baseline physics and qcloud corrections

Primary classification: `[hybrid]`

## STEP B. Mathematical and Architectural Role

Section 11 adds:

- `Y_res`
- `Y_hat_res`
- `U`
- `F_total = F_base + F_ml`
- `E_total = E_base + E_ml`

Architecturally:

- `physics/` remains the established baseline force substrate
- `qcloud/` remains a separate correction pathway
- `memory/` remains the owner of replay and trace storage
- `ml/` owns learned residual targets, predictions, uncertainty estimates, and trainer hooks

## STEP C. Folder Structure Created or Extended

- `ml/`
- `tests/`
- `docs/architecture/`
- `docs/sections/`
- `progress_info/`

## STEP D. Smaller Scripts Added

- `ml/residual_model.py`
- `ml/uncertainty_model.py`
- `ml/online_trainer.py`
- `tests/test_ml_layer.py`
- `docs/architecture/online_ml_residual_learning.md`
- `docs/sections/section_11_online_ml_residual_learning.md`

## STEP E. Implementation Notes

- `ml/residual_model.py`
  - defines `ResidualTarget`, `ResidualPrediction`, `ResidualMemoryModel`, and `ResidualAugmentedForceEvaluator`
  - supports qcloud-derived residual targets and additive learned force evaluation
- `ml/uncertainty_model.py`
  - defines `UncertaintyEstimate`, `UncertaintyModel`, and `HeuristicUncertaintyModel`
  - keeps uncertainty explicit and separate from residual prediction
- `ml/online_trainer.py`
  - defines `ReplayTrainingExample`, `OnlineTrainingReport`, and `ReplayDrivenOnlineTrainer`
  - consumes replay items and aligned residual targets without taking ownership of memory storage

Key design choices:

- Section 11 residuals are additive rather than replacements for baseline physics
- qcloud and ML remain separate correction sources
- replay drives training updates explicitly
- uncertainty remains heuristic until later validation/calibration sections exist

## STEP F. Validation Strategy

Section 11 validation covers:

- qcloud-correction aggregation into residual targets
- exact-state replay-driven residual prediction
- additive learned residual correction on top of a baseline evaluator
- heuristic uncertainty-trigger behavior for priority states
- replay-driven online trainer updates

Commands:

```bash
python3 scripts/validate_scaffold.py
python3 -m unittest discover -s tests
```

## STEP G. Documentation Updated

- `docs/architecture/online_ml_residual_learning.md`
- `docs/sections/section_11_online_ml_residual_learning.md`
- `progress_info/*.md`

## STEP H. Progress Files Updated

All mandatory continuity files were updated to reflect Section 11 completion and
the transition to Section 12.

## STEP I. Handoff Summary

SECTION HANDOFF
- Section completed: Section 11. Online ML residual learning
- Purpose: add replay-driven residual targets, additive learned correction, heuristic uncertainty, and online trainer hooks
- Files created: residual model, uncertainty model, online trainer, tests, docs, and continuity updates
- Key interfaces: `ResidualTarget`, `ResidualPrediction`, `ResidualMemoryModel`, `ResidualAugmentedForceEvaluator`, `UncertaintyEstimate`, `ReplayDrivenOnlineTrainer`
- Dependencies: `memory/replay_buffer.py`, `memory/trace_store.py`, `qcloud/cloud_state.py`, `physics/forces/composite.py`
- What now works: qcloud-derived residual targets, replay-driven exact-state residual prediction, additive learned force augmentation, heuristic uncertainty estimates, and deterministic replay-backed training updates
- What is still stubbed: calibrated uncertainty, generalized residual models beyond exact-state recall, feature learning, and AI-control policies that consume ML outputs
- Risks / unresolved items: Section 11 remains intentionally simple and deterministic; future learned models must extend it without silently replacing the explicit additive-correction contract
- Next recommended section: Section 12. AI executive control layer
- Important continuity notes: AI control should consume memory, qcloud, and ML outputs as separate signals rather than collapsing them into one opaque controller input
