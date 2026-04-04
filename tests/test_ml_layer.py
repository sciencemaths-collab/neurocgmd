"""Tests for Section 11 residual learning, uncertainty, and replay-driven updates."""

from __future__ import annotations

import unittest

from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.types import BeadId, SimulationId, StateId
from forcefields.base_forcefield import BaseForceField
from memory import ReplayBuffer, TraceRecord, TraceStore
from ml import (
    HeuristicUncertaintyModel,
    LiveFeatureVector,
    ReplayDrivenOnlineTrainer,
    ResidualAugmentedForceEvaluator,
    ResidualMemoryModel,
    ResidualTarget,
)
from physics.forces.composite import ForceEvaluation
from qcloud import ParticleForceDelta, QCloudCorrection
from topology import Bead, BeadRole, BeadType, Bond, SystemTopology


class _ZeroForceEvaluator:
    name = "zero_force_evaluator"
    classification = "[test]"

    def evaluate(self, state: SimulationState, topology: SystemTopology, forcefield: BaseForceField) -> ForceEvaluation:
        del topology, forcefield
        return ForceEvaluation(
            forces=tuple((0.0, 0.0, 0.0) for _ in range(state.particle_count)),
            potential_energy=0.0,
        )


class MLLayerTests(unittest.TestCase):
    """Verify the Section 11 ML layer stays explicit, bounded, and replay-driven."""

    def _build_state(self, *, state_id: str, step: int) -> SimulationState:
        return SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                masses=(1.0, 1.0),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-ml"),
                state_id=StateId(state_id),
                parent_state_id=None,
                created_by="unit-test",
                stage="checkpoint",
            ),
            time=0.1 * step,
            step=step,
            potential_energy=-0.5,
        )

    def _build_topology(self) -> SystemTopology:
        return SystemTopology(
            system_id="ml-system",
            bead_types=(
                BeadType(name="bb", role=BeadRole.STRUCTURAL),
                BeadType(name="site", role=BeadRole.FUNCTIONAL),
            ),
            beads=(
                Bead(bead_id=BeadId("b0"), particle_index=0, bead_type="bb", label="B0"),
                Bead(bead_id=BeadId("b1"), particle_index=1, bead_type="site", label="S0"),
            ),
            bonds=(Bond(0, 1),),
        )

    def test_residual_target_aggregates_qcloud_corrections_and_predicts_exactly(self) -> None:
        state = self._build_state(state_id="state-ml-0", step=0)
        corrections = (
            QCloudCorrection(
                region_id="region-0",
                method_label="qcloud-a",
                energy_delta=0.3,
                force_deltas=(ParticleForceDelta(0, (1.0, 0.0, 0.0)),),
                confidence=0.9,
            ),
            QCloudCorrection(
                region_id="region-1",
                method_label="qcloud-b",
                energy_delta=0.2,
                force_deltas=(
                    ParticleForceDelta(0, (0.5, 0.5, 0.0)),
                    ParticleForceDelta(1, (0.0, 1.0, 0.0)),
                ),
                confidence=0.8,
            ),
        )
        target = ResidualTarget.from_corrections(state.provenance.state_id, corrections)

        model = ResidualMemoryModel()
        model.observe(target, sample_weight=2.0)
        prediction = model.predict(
            state,
            ForceEvaluation(forces=((0.0, 0.0, 0.0),) * state.particle_count, potential_energy=0.0),
        )

        self.assertAlmostEqual(target.energy_delta, 0.5)
        self.assertEqual(target.force_deltas[0].delta_force, (1.5, 0.5, 0.0))
        self.assertEqual(target.force_deltas[1].delta_force, (0.0, 1.0, 0.0))
        self.assertAlmostEqual(prediction.predicted_energy_delta, 0.5)
        self.assertEqual(prediction.force_deltas, target.force_deltas)
        self.assertAlmostEqual(prediction.confidence, 2.0 / 3.0)

    def test_residual_augmented_force_evaluator_adds_ml_component(self) -> None:
        state = self._build_state(state_id="state-ml-1", step=1)
        topology = self._build_topology()
        target = ResidualTarget(
            state_id=state.provenance.state_id,
            energy_delta=0.4,
            force_deltas=(
                ParticleForceDelta(0, (1.0, 0.0, 0.0)),
                ParticleForceDelta(1, (0.0, -1.0, 0.0)),
            ),
            source_label="unit-test",
        )
        model = ResidualMemoryModel()
        model.observe(target, sample_weight=3.0)
        evaluator = ResidualAugmentedForceEvaluator(
            base_force_evaluator=_ZeroForceEvaluator(),
            residual_model=model,
        )

        evaluation = evaluator.evaluate(state, topology, BaseForceField(name="ml-ff"))
        self.assertAlmostEqual(evaluation.potential_energy, 0.4)
        self.assertEqual(evaluation.forces[0], (1.0, 0.0, 0.0))
        self.assertEqual(evaluation.forces[1], (0.0, -1.0, 0.0))
        self.assertAlmostEqual(evaluation.component_energies["ml_residual"], 0.4)
        self.assertEqual(evaluation.metadata["residual_model"], "residual_memory_model")

    def test_uncertainty_model_flags_low_confidence_priority_predictions(self) -> None:
        state = self._build_state(state_id="state-ml-2", step=2)
        unseen_model = ResidualMemoryModel()
        prediction = unseen_model.predict(
            state,
            ForceEvaluation(forces=((0.0, 0.0, 0.0),) * state.particle_count, potential_energy=0.0),
        )
        trace_record = TraceRecord(
            record_id="trace-ml-2",
            simulation_id=state.provenance.simulation_id,
            state_id=state.provenance.state_id,
            parent_state_id=None,
            stage=state.provenance.stage,
            step=state.step,
            time=state.time,
            particle_count=state.particle_count,
            kinetic_energy=state.kinetic_energy(),
            potential_energy=state.potential_energy,
            tags=("priority",),
        )
        replay_buffer = ReplayBuffer(capacity=4)
        replay_item = replay_buffer.add_from_record(trace_record, score=2.0, tags=("priority",))

        estimate = HeuristicUncertaintyModel(trigger_threshold=0.6).estimate(
            prediction,
            trace_record=trace_record,
            replay_item=replay_item,
        )

        self.assertTrue(estimate.trigger_qcloud)
        self.assertGreaterEqual(estimate.total_uncertainty, 0.6)
        self.assertIn("priority", estimate.metadata["priority_hits"])

    def test_uncertainty_model_uses_live_features_for_chemistry_and_structure_risk(self) -> None:
        state = self._build_state(state_id="state-ml-chem", step=3)
        prediction = ResidualMemoryModel().predict(
            state,
            ForceEvaluation(forces=((0.0, 0.0, 0.0),) * state.particle_count, potential_energy=0.0),
        )
        live_features = LiveFeatureVector(
            state_id=state.provenance.state_id,
            values={
                "chemistry_mean_pair_score": 0.18,
                "chemistry_favorable_pair_fraction": 0.10,
                "chemistry_flexibility_pressure": 0.72,
                "structure_rmsd_normalized": 0.85,
                "shadow_force_regression": 1.0,
                "shadow_energy_regression": 0.0,
            },
        )
        model = HeuristicUncertaintyModel(trigger_threshold=0.4)

        baseline = model.estimate(prediction)
        chemistry_weighted = model.estimate(prediction, live_features=live_features)

        self.assertGreater(chemistry_weighted.total_uncertainty, baseline.total_uncertainty)
        self.assertGreater(chemistry_weighted.metadata["chemistry_bonus"], 0.0)
        self.assertGreater(chemistry_weighted.metadata["structure_bonus"], 0.0)
        self.assertTrue(chemistry_weighted.trigger_qcloud)

    def test_online_trainer_consumes_replay_examples_and_updates_model(self) -> None:
        state_a = self._build_state(state_id="state-ml-a", step=0)
        state_b = self._build_state(state_id="state-ml-b", step=1)
        record_a = TraceRecord(
            record_id="trace-a",
            simulation_id=state_a.provenance.simulation_id,
            state_id=state_a.provenance.state_id,
            parent_state_id=None,
            stage=state_a.provenance.stage,
            step=state_a.step,
            time=state_a.time,
            particle_count=state_a.particle_count,
            kinetic_energy=state_a.kinetic_energy(),
            potential_energy=state_a.potential_energy,
            tags=("train",),
        )
        record_b = TraceRecord(
            record_id="trace-b",
            simulation_id=state_b.provenance.simulation_id,
            state_id=state_b.provenance.state_id,
            parent_state_id=state_a.provenance.state_id,
            stage=state_b.provenance.stage,
            step=state_b.step,
            time=state_b.time,
            particle_count=state_b.particle_count,
            kinetic_energy=state_b.kinetic_energy(),
            potential_energy=state_b.potential_energy,
            tags=("train", "priority"),
        )
        trace_store = TraceStore()
        trace_store.append(record_a)
        trace_store.append(record_b)

        replay_buffer = ReplayBuffer(capacity=4)
        replay_item_a = replay_buffer.add_from_record(record_a, score=1.0, tags=("train",))
        replay_item_b = replay_buffer.add_from_record(record_b, score=2.0, tags=("train", "priority"))

        residual_targets = {
            record_a.state_id: ResidualTarget(record_a.state_id, energy_delta=0.1, source_label="qcloud"),
            record_b.state_id: ResidualTarget(record_b.state_id, energy_delta=0.3, source_label="qcloud"),
        }
        trainer = ReplayDrivenOnlineTrainer(max_examples_per_update=2)
        examples = trainer.build_examples(replay_buffer, trace_store, residual_targets)
        model = ResidualMemoryModel()
        report = trainer.update_from_replay(model, replay_buffer, trace_store, residual_targets)

        self.assertEqual(examples[0].replay_item.state_id, replay_item_b.state_id)
        self.assertEqual(examples[1].replay_item.state_id, replay_item_a.state_id)
        self.assertEqual(report.processed_items, 2)
        self.assertEqual(report.consumed_examples, 2)
        self.assertEqual(model.trained_state_count(), 2)
        self.assertEqual(report.updated_state_ids, (record_b.state_id, record_a.state_id))


if __name__ == "__main__":
    unittest.main()
