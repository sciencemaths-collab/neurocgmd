"""Tests for differentiable hooks over repo-native ML modules."""

from __future__ import annotations

import unittest

from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.types import SimulationId, StateId
from ml.differentiable_hooks import (
    FiniteDifferenceGradientEstimator,
    ScalableResidualDifferentiableAdapter,
)
from ml.scalable_residual import ScalableResidualModel
from ml.residual_model import ResidualTarget
from physics.forces.composite import ForceEvaluation
from qcloud.cloud_state import ParticleForceDelta


class DifferentiableHookTests(unittest.TestCase):
    """Verify stable parameter access and finite-difference estimation."""

    def _build_state(self) -> SimulationState:
        return SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                velocities=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                masses=(1.0, 1.0),
                labels=("A", "B"),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-diff"),
                state_id=StateId("state-diff"),
                parent_state_id=None,
                created_by="unit-test",
                stage="checkpoint",
            ),
            time=0.0,
            step=0,
            potential_energy=0.0,
        )

    def _build_force_evaluation(self) -> ForceEvaluation:
        return ForceEvaluation(
            forces=((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
            potential_energy=-0.4,
        )

    def test_scalable_adapter_exposes_parameter_blocks(self) -> None:
        adapter = ScalableResidualDifferentiableAdapter(ScalableResidualModel())
        parameter_names = tuple(parameter.name for parameter in adapter.list_parameters())

        self.assertEqual(
            parameter_names,
            ("energy_head_weights", "energy_head_bias", "force_head_weights", "force_head_bias"),
        )

    def test_finite_difference_estimator_returns_gradients(self) -> None:
        state = self._build_state()
        force_evaluation = self._build_force_evaluation()
        model = ScalableResidualModel(learning_rate=0.01, random_seed=13)
        target = ResidualTarget(
            state_id=state.provenance.state_id,
            energy_delta=0.8,
            force_deltas=(ParticleForceDelta(0, (0.25, 0.0, 0.0)),),
            source_label="unit-test",
        )
        for _ in range(10):
            model.observe_state(state, force_evaluation, target)
        adapter = ScalableResidualDifferentiableAdapter(model)
        estimator = FiniteDifferenceGradientEstimator(epsilon=1e-6)

        def objective(module_adapter: ScalableResidualDifferentiableAdapter) -> float:
            prediction = module_adapter.model.predict(state, force_evaluation)
            return prediction.predicted_energy_delta * prediction.predicted_energy_delta

        gradients = estimator.estimate(adapter, objective)

        self.assertEqual(len(gradients), 4)
        self.assertTrue(any(any(abs(value) > 0.0 for value in gradient.values) for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
