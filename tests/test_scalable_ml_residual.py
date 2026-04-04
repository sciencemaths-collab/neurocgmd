"""Tests for the scalable SPRING-style residual learning upgrade."""

from __future__ import annotations

import unittest

from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.state_registry import SimulationStateRegistry
from core.types import BeadId, SimulationId, StateId
from forcefields.base_forcefield import BaseForceField
from memory import ReplayBuffer, TraceRecord, TraceStore
from ml import ReplayDrivenOnlineTrainer, ResidualPrediction, ResidualTarget, ScalableResidualModel
from physics.forces.composite import ForceEvaluation
from qcloud.cloud_state import ParticleForceDelta
from scripts.live_dashboard import build_spike_ace2_context
from topology import Bead, BeadRole, BeadType, Bond, SystemTopology


class _FixedForceEvaluator:
    name = "fixed_force_evaluator"
    classification = "[test]"

    def __init__(self, force_evaluation: ForceEvaluation) -> None:
        self.force_evaluation = force_evaluation

    def evaluate(self, state: SimulationState, topology: SystemTopology, forcefield: BaseForceField) -> ForceEvaluation:
        del state, topology, forcefield
        return self.force_evaluation


class ScalableResidualModelTests(unittest.TestCase):
    """Validate the piece-local scalable residual path."""

    def _build_state(self, *, state_id: str = "state-scalable", step: int = 0) -> SimulationState:
        return SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=(
                    (0.0, 0.0, 0.0),
                    (0.9, 0.0, 0.0),
                    (0.0, 0.9, 0.0),
                ),
                velocities=(
                    (0.01, 0.00, 0.00),
                    (0.00, 0.02, 0.00),
                    (0.00, 0.00, 0.03),
                ),
                masses=(1.0, 1.5, 2.0),
                labels=("A", "B", "C"),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-scalable"),
                state_id=StateId(state_id),
                parent_state_id=None,
                created_by="unit-test",
                stage="checkpoint",
            ),
            time=0.1 * step,
            step=step,
            potential_energy=-1.0,
        )

    def _build_force_evaluation(self) -> ForceEvaluation:
        return ForceEvaluation(
            forces=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, -1.0),
            ),
            potential_energy=-0.8,
        )

    def _build_target(self, *, state_id: str = "state-scalable", energy_delta: float = 1.2) -> ResidualTarget:
        return ResidualTarget(
            state_id=StateId(state_id),
            energy_delta=energy_delta,
            force_deltas=(
                ParticleForceDelta(0, (0.5, 0.0, 0.0)),
                ParticleForceDelta(1, (0.0, -0.25, 0.0)),
                ParticleForceDelta(2, (0.0, 0.0, 0.1)),
            ),
            source_label="unit-test",
        )

    def _build_topology(self) -> SystemTopology:
        return SystemTopology(
            system_id="scalable-ml-topology",
            bead_types=(
                BeadType(name="bb", role=BeadRole.STRUCTURAL),
                BeadType(name="site", role=BeadRole.FUNCTIONAL),
            ),
            beads=(
                Bead(bead_id=BeadId("b0"), particle_index=0, bead_type="bb", label="A"),
                Bead(bead_id=BeadId("b1"), particle_index=1, bead_type="site", label="B"),
                Bead(bead_id=BeadId("b2"), particle_index=2, bead_type="site", label="C"),
            ),
            bonds=(Bond(0, 1), Bond(0, 2)),
        )

    def test_predict_returns_neutral_untrained_output(self) -> None:
        model = ScalableResidualModel(random_seed=7)
        prediction = model.predict(self._build_state(), self._build_force_evaluation())

        self.assertIsInstance(prediction, ResidualPrediction)
        self.assertAlmostEqual(prediction.predicted_energy_delta, 0.0)
        self.assertEqual(prediction.force_deltas, ())
        self.assertEqual(prediction.metadata["neighbor_method"], "cell_list")
        self.assertEqual(prediction.metadata["message_passing_steps"], 2)
        self.assertGreater(prediction.metadata["mean_neighbor_count"], 0.0)

    def test_model_validation_rejects_bad_message_passing_configuration(self) -> None:
        with self.assertRaisesRegex(Exception, "message_passing_steps"):
            ScalableResidualModel(message_passing_steps=0)

        with self.assertRaisesRegex(Exception, "residual_blend"):
            ScalableResidualModel(residual_blend=1.5)

    def test_observe_state_moves_prediction_toward_target(self) -> None:
        model = ScalableResidualModel(learning_rate=0.01, random_seed=11)
        state = self._build_state()
        force_evaluation = self._build_force_evaluation()
        target = self._build_target()

        initial = model.predict(state, force_evaluation)
        initial_energy_distance = abs(initial.predicted_energy_delta - target.energy_delta)
        initial_force_distance = abs(0.0 - 0.5)

        for _ in range(80):
            model.observe_state(state, force_evaluation, target)

        trained = model.predict(state, force_evaluation)
        trained_force = dict((delta.particle_index, delta.delta_force) for delta in trained.force_deltas)
        trained_force_x = trained_force[0][0] if 0 in trained_force else 0.0

        self.assertLess(abs(trained.predicted_energy_delta - target.energy_delta), initial_energy_distance)
        self.assertLess(abs(trained_force_x - 0.5), initial_force_distance)
        self.assertGreater(model.trained_state_count(), 0)
        self.assertGreater(trained.confidence, initial.confidence)

    def test_observe_legacy_path_uses_cached_forward_state(self) -> None:
        model = ScalableResidualModel(learning_rate=0.01, random_seed=19)
        state = self._build_state()
        force_evaluation = self._build_force_evaluation()
        target = self._build_target()

        initial = model.predict(state, force_evaluation)
        for _ in range(40):
            model.predict(state, force_evaluation)
            model.observe(target)
        trained = model.predict(state, force_evaluation)

        self.assertLess(abs(trained.predicted_energy_delta - target.energy_delta), abs(initial.predicted_energy_delta - target.energy_delta))

    def test_state_aware_replay_trainer_updates_scalable_model(self) -> None:
        registry = SimulationStateRegistry(created_by="unit-test", simulation_id=SimulationId("sim-scalable"))
        state = self._build_state()
        registry.register_state(state)
        record = TraceRecord(
            record_id="trace-scalable-0",
            simulation_id=state.provenance.simulation_id,
            state_id=state.provenance.state_id,
            parent_state_id=None,
            stage=state.provenance.stage,
            step=state.step,
            time=state.time,
            particle_count=state.particle_count,
            kinetic_energy=state.kinetic_energy(),
            potential_energy=state.potential_energy,
            tags=("train",),
        )
        trace_store = TraceStore()
        trace_store.append(record)
        replay_buffer = ReplayBuffer(capacity=4)
        replay_buffer.add_from_record(record, score=2.0, tags=("train",))
        target = self._build_target()
        trainer = ReplayDrivenOnlineTrainer(max_examples_per_update=2)
        model = ScalableResidualModel(learning_rate=0.01, random_seed=23)
        force_evaluation = self._build_force_evaluation()
        report = trainer.update_from_replay_with_states(
            model,
            replay_buffer,
            trace_store,
            {state.provenance.state_id: target},
            state_registry=registry,
            topology=self._build_topology(),
            forcefield=BaseForceField(name="scalable-ff"),
            force_evaluator=_FixedForceEvaluator(force_evaluation),
        )

        self.assertEqual(report.consumed_examples, 1)
        self.assertTrue(report.metadata["state_aware"])
        self.assertEqual(model.trained_state_count(), 1)

    def test_live_dashboard_uses_scalable_model_for_protein_benchmark(self) -> None:
        context = build_spike_ace2_context()
        self.assertEqual(context.residual_model.name, "scalable_residual_model")


if __name__ == "__main__":
    unittest.main()
