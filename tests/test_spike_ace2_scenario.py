"""Tests for the harder spike-ACE2 live benchmark scenario."""

from __future__ import annotations

import unittest

from core.state_registry import SimulationStateRegistry
from graph import ConnectivityGraphManager
from integrators.langevin import LangevinIntegrator
from physics.forces.composite import BaselineForceEvaluator
from qcloud import QCloudForceCoupler, RefinementRegion, RegionTriggerKind
from sampling.scenarios import SpikeAce2Scenario
from sampling.simulation_loop import SimulationLoop


class SpikeAce2ScenarioTests(unittest.TestCase):
    """Verify the harder ACE2-spike coarse-grained benchmark proxy."""

    def test_setup_tracks_reference_case(self) -> None:
        scenario = SpikeAce2Scenario()
        setup = scenario.build_setup()

        self.assertEqual(setup.metadata["reference_case"], "spike_ace2")
        self.assertEqual(setup.metadata["bound_complex_pdb_id"], "6M0J")
        self.assertEqual(setup.focus_compartments, ("ace2", "spike_rbd"))
        self.assertEqual(setup.topology.validate_against_particle_state(setup.initial_particles), ())
        self.assertGreater(setup.integrator_time_step, scenario.recommended_time_step)
        self.assertGreater(setup.integrator_friction, scenario.recommended_friction)
        self.assertEqual(setup.metadata["shadow_tuning_preset"], "protein_large_step_fast")

    def test_reference_case_is_available(self) -> None:
        case = SpikeAce2Scenario().reference_case()

        self.assertEqual(case.name, "spike_ace2")
        self.assertEqual(case.structural_reference.unbound_partner_pdb_ids, ("1R42", "6VYB"))

    def test_structure_report_is_available(self) -> None:
        scenario = SpikeAce2Scenario()
        setup = scenario.build_setup()
        registry = SimulationStateRegistry(created_by="unit-test")
        state = registry.create_initial_state(
            particles=setup.initial_particles,
            thermodynamics=setup.thermodynamics,
        )

        report = scenario.build_structure_report(state)

        self.assertEqual(report.title, "Atomistic Alignment")
        self.assertIn("local atomistic centroids derived from 6M0J", report.summary)
        self.assertIn("Atomistic Centroid RMSD", {metric.label for metric in report.metrics})

    def test_shadow_fidelity_report_is_available(self) -> None:
        scenario = SpikeAce2Scenario()
        setup = scenario.build_setup()
        registry = SimulationStateRegistry(created_by="unit-test")
        state = registry.create_initial_state(
            particles=setup.initial_particles,
            thermodynamics=setup.thermodynamics,
        )
        baseline = BaselineForceEvaluator().evaluate(state, setup.topology, setup.forcefield)
        corrected = QCloudForceCoupler(
            max_energy_delta_magnitude=10.0,
            max_force_delta_magnitude=10.0,
        ).couple(
            baseline,
            state,
            setup.topology,
            (
                RefinementRegion(
                    region_id="region-shadow-live-001",
                    state_id=state.provenance.state_id,
                    particle_indices=(1, 2, 8, 9),
                    seed_pairs=((2, 8), (3, 9)),
                    trigger_kinds=(RegionTriggerKind.ADAPTIVE_EDGE,),
                    score=1.0,
                ),
            ),
            scenario.build_qcloud_correction_model(),
        ).force_evaluation

        report = scenario.build_fidelity_report(
            state,
            baseline_evaluation=baseline,
            corrected_evaluation=corrected,
        )

        self.assertEqual(report.title, "Shadow Fidelity")
        self.assertEqual(report.target_label, "6M0J Shadow Contact Target")
        self.assertIn("force_rms_error", {metric.label for metric in report.metrics})
        self.assertTrue(any(metric.corrected_error != metric.baseline_error for metric in report.metrics))

    def test_proxy_progress_reaches_harder_capture_phase(self) -> None:
        scenario = SpikeAce2Scenario()
        setup = scenario.build_setup()
        registry = SimulationStateRegistry(created_by="unit-test")
        registry.create_initial_state(
            particles=setup.initial_particles,
            thermodynamics=setup.thermodynamics,
        )
        loop = SimulationLoop(
            topology=setup.topology,
            forcefield=setup.forcefield,
            integrator=LangevinIntegrator(
                time_step=setup.integrator_time_step,
                friction_coefficient=setup.integrator_friction,
            ),
            force_evaluator=BaselineForceEvaluator(),
            registry=registry,
        )
        graph_manager = ConnectivityGraphManager()

        for _ in range(160):
            loop.run(1)

        state = registry.latest_state()
        graph = graph_manager.initialize(state, setup.topology)
        progress = scenario.measure_progress(state, graph=graph)

        self.assertLess(progress.interface_distance, progress.initial_interface_distance)
        self.assertGreaterEqual(progress.cross_contact_count, 5)
        self.assertIn(progress.stage_label, {"Hotspot Capture", "Native-Like Recognition"})
        self.assertGreaterEqual(progress.assembly_score, 0.55)


if __name__ == "__main__":
    unittest.main()
