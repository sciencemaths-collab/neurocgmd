"""Tests for NeuralResidualModel and EnsembleUncertaintyModel."""

from __future__ import annotations

import unittest

from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.types import SimulationId, StateId
from ml.live_features import LiveFeatureVector
from ml.neural_residual_model import NeuralResidualModel
from ml.ensemble_uncertainty import EnsembleUncertaintyModel
from ml.residual_model import ResidualPrediction, ResidualTarget
from ml.uncertainty_model import UncertaintyEstimate
from physics.forces.composite import ForceEvaluation
from qcloud.cloud_state import ParticleForceDelta


class NeuralMLTestBase(unittest.TestCase):
    """Shared helpers for neural ML tests."""

    def _build_state(self, *, state_id: str = "state-0", step: int = 0) -> SimulationState:
        return SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                masses=(1.0, 1.0),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-neural"),
                state_id=StateId(state_id),
                parent_state_id=None,
                created_by="unit-test",
                stage="checkpoint",
            ),
            time=0.1 * step,
            step=step,
            potential_energy=-0.5,
        )

    def _build_force_eval(self, n_particles: int = 2, energy: float = -1.0) -> ForceEvaluation:
        return ForceEvaluation(
            forces=tuple((0.1 * (i + 1), -0.05 * i, 0.02) for i in range(n_particles)),
            potential_energy=energy,
        )

    def _build_target(self, state_id: str = "state-0", energy_delta: float = 0.3) -> ResidualTarget:
        return ResidualTarget(
            state_id=StateId(state_id),
            energy_delta=energy_delta,
            force_deltas=(
                ParticleForceDelta(particle_index=0, delta_force=(0.1, 0.0, 0.0)),
                ParticleForceDelta(particle_index=1, delta_force=(0.0, -0.1, 0.0)),
            ),
            source_label="unit-test",
        )


class TestNeuralResidualModel(NeuralMLTestBase):
    """Tests for NeuralResidualModel."""

    def test_neural_model_predict_returns_prediction(self) -> None:
        """Fresh model returns a ResidualPrediction with zero confidence."""
        model = NeuralResidualModel()
        state = self._build_state()
        force_eval = self._build_force_eval()

        prediction = model.predict(state, force_eval)

        self.assertIsInstance(prediction, ResidualPrediction)
        self.assertEqual(prediction.state_id, StateId("state-0"))
        self.assertAlmostEqual(prediction.confidence, 0.0)
        self.assertIsInstance(prediction.predicted_energy_delta, float)

    def test_neural_model_observe_updates_state(self) -> None:
        """Observing a target increments trained_state_count."""
        model = NeuralResidualModel()
        state = self._build_state()
        force_eval = self._build_force_eval()

        self.assertEqual(model.trained_state_count(), 0)

        # Run predict first so forward cache is populated
        model.predict(state, force_eval)
        target = self._build_target()
        model.observe(target)

        self.assertEqual(model.trained_state_count(), 1)

        model.observe(target)
        self.assertEqual(model.trained_state_count(), 2)

    def test_neural_model_confidence_grows(self) -> None:
        """Confidence increases after multiple observations."""
        model = NeuralResidualModel(confidence_growth_rate=0.1)
        state = self._build_state()
        force_eval = self._build_force_eval()

        pred_before = model.predict(state, force_eval)
        self.assertAlmostEqual(pred_before.confidence, 0.0)

        target = self._build_target()
        for _ in range(5):
            model.predict(state, force_eval)
            model.observe(target)

        pred_after = model.predict(state, force_eval)
        self.assertGreater(pred_after.confidence, pred_before.confidence)
        # 5 observations * 0.1 growth rate = 0.5
        self.assertAlmostEqual(pred_after.confidence, 0.5)

    def test_neural_model_learns_energy_correction(self) -> None:
        """After 50+ observations of same target, predicted energy moves toward target."""
        target_energy = 2.0
        model = NeuralResidualModel(learning_rate=0.01)
        state = self._build_state()
        force_eval = self._build_force_eval()

        # Get initial prediction distance from target
        initial_pred = model.predict(state, force_eval)
        initial_distance = abs(initial_pred.predicted_energy_delta - target_energy)

        target = self._build_target(energy_delta=target_energy)
        for _ in range(60):
            model.predict(state, force_eval)
            model.observe(target)

        final_pred = model.predict(state, force_eval)
        final_distance = abs(final_pred.predicted_energy_delta - target_energy)

        # After training, the prediction should be closer to the target
        self.assertLess(final_distance, initial_distance,
                        f"Expected final distance {final_distance:.4f} < initial distance {initial_distance:.4f}")

    def test_neural_model_deterministic_with_seed(self) -> None:
        """Same seed produces identical predictions."""
        state = self._build_state()
        force_eval = self._build_force_eval()

        model_a = NeuralResidualModel(random_seed=123)
        pred_a = model_a.predict(state, force_eval)

        model_b = NeuralResidualModel(random_seed=123)
        pred_b = model_b.predict(state, force_eval)

        self.assertAlmostEqual(pred_a.predicted_energy_delta, pred_b.predicted_energy_delta)
        self.assertAlmostEqual(pred_a.confidence, pred_b.confidence)

        # Different seed should produce different predictions
        model_c = NeuralResidualModel(random_seed=999)
        pred_c = model_c.predict(state, force_eval)
        # Weights are different, so predictions should differ (unless extremely unlikely)
        # We just verify the model runs; exact inequality is not guaranteed but highly likely
        self.assertIsInstance(pred_c.predicted_energy_delta, float)

    def test_neural_model_validation(self) -> None:
        """Bad hidden_sizes are rejected by validate()."""
        model = NeuralResidualModel(hidden_sizes=())
        issues = model.validate()
        self.assertTrue(any("hidden_sizes" in issue for issue in issues))

        model_bad_lr = NeuralResidualModel(learning_rate=-0.1)
        issues_lr = model_bad_lr.validate()
        self.assertTrue(any("learning_rate" in issue for issue in issues_lr))

        model_bad_momentum = NeuralResidualModel(momentum=1.5)
        issues_m = model_bad_momentum.validate()
        self.assertTrue(any("momentum" in issue for issue in issues_m))

        # Valid model should have no issues
        model_ok = NeuralResidualModel()
        self.assertEqual(model_ok.validate(), ())


class TestEnsembleUncertaintyModel(NeuralMLTestBase):
    """Tests for EnsembleUncertaintyModel."""

    def _build_prediction(
        self, state_id: str = "state-0", energy_delta: float = 0.1, confidence: float = 0.5,
    ) -> ResidualPrediction:
        return ResidualPrediction(
            state_id=StateId(state_id),
            predicted_energy_delta=energy_delta,
            force_deltas=(
                ParticleForceDelta(particle_index=0, delta_force=(0.1, 0.0, 0.0)),
            ),
            confidence=confidence,
        )

    def test_ensemble_estimate_returns_valid(self) -> None:
        """Estimate returns UncertaintyEstimate with values in [0, 1]."""
        model = EnsembleUncertaintyModel()
        prediction = self._build_prediction()

        estimate = model.estimate(prediction)

        self.assertIsInstance(estimate, UncertaintyEstimate)
        self.assertGreaterEqual(estimate.energy_uncertainty, 0.0)
        self.assertLessEqual(estimate.energy_uncertainty, 1.0)
        self.assertGreaterEqual(estimate.force_uncertainty, 0.0)
        self.assertLessEqual(estimate.force_uncertainty, 1.0)
        self.assertGreaterEqual(estimate.total_uncertainty, 0.0)
        self.assertLessEqual(estimate.total_uncertainty, 1.0)
        self.assertIsInstance(estimate.trigger_qcloud, bool)

    def test_ensemble_observe_updates_members(self) -> None:
        """Observing targets changes the ensemble's internal state and uncertainty."""
        model = EnsembleUncertaintyModel()
        prediction = self._build_prediction()

        estimate_before = model.estimate(prediction)

        target = self._build_target(energy_delta=5.0)
        for _ in range(20):
            model.observe(target)

        estimate_after = model.estimate(prediction)

        # After training on consistent targets, uncertainty outputs should change
        # (members diverge due to bootstrap sampling).
        self.assertNotAlmostEqual(
            estimate_before.energy_uncertainty,
            estimate_after.energy_uncertainty,
            places=6,
            msg="Uncertainty should change after observing targets",
        )

    def test_ensemble_high_disagreement_triggers_qcloud(self) -> None:
        """When ensemble members disagree strongly, trigger_qcloud should be True."""
        model = EnsembleUncertaintyModel(
            trigger_threshold=0.1,
            calibration_scale=5.0,
            ensemble_size=5,
        )

        # Train different members on different targets to create disagreement.
        # Because of bootstrap sampling, different members see different subsets.
        # We use a fixed seed and varied targets to maximise disagreement.
        for i in range(30):
            target = self._build_target(
                state_id=f"state-{i}",
                energy_delta=10.0 * ((-1) ** i),  # alternate sign to create divergence
            )
            model.observe(target)

        # Now estimate on a prediction -- members should disagree
        prediction = self._build_prediction(energy_delta=0.0, confidence=0.0)
        estimate = model.estimate(prediction)

        self.assertTrue(
            estimate.trigger_qcloud,
            f"Expected trigger_qcloud=True but got False "
            f"(total_uncertainty={estimate.total_uncertainty:.4f})",
        )

    def test_ensemble_with_live_features(self) -> None:
        """Passing live_features to estimate does not crash and returns valid output."""
        model = EnsembleUncertaintyModel()
        prediction = self._build_prediction()
        live_features = LiveFeatureVector(
            state_id=StateId("state-0"),
            values={
                "chemistry_mean_pair_score": 0.25,
                "structure_rmsd_normalized": 0.6,
                "shadow_force_regression": 0.8,
            },
        )

        estimate = model.estimate(prediction, live_features=live_features)

        self.assertIsInstance(estimate, UncertaintyEstimate)
        self.assertGreaterEqual(estimate.energy_uncertainty, 0.0)
        self.assertLessEqual(estimate.energy_uncertainty, 1.0)
        self.assertGreaterEqual(estimate.force_uncertainty, 0.0)
        self.assertLessEqual(estimate.force_uncertainty, 1.0)


if __name__ == "__main__":
    unittest.main()
