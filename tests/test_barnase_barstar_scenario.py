"""Tests for the barnase-barstar live benchmark scenario."""

from __future__ import annotations

import unittest

from core.state_registry import SimulationStateRegistry
from graph import ConnectivityGraphManager
from integrators.langevin import LangevinIntegrator
from physics.forces.composite import BaselineForceEvaluator
from qcloud import QCloudForceCoupler, RefinementRegion, RegionTriggerKind
from sampling.scenarios import BarnaseBarstarScenario
from sampling.simulation_loop import SimulationLoop


class BarnaseBarstarScenarioTests(unittest.TestCase):
    """Verify the coarse-grained barnase-barstar proxy scenario."""

    def test_setup_tracks_reference_case(self) -> None:
        scenario = BarnaseBarstarScenario()
        setup = scenario.build_setup()

        self.assertEqual(setup.metadata["reference_case"], "barnase_barstar")
        self.assertEqual(setup.metadata["bound_complex_pdb_id"], "1BRS")
        self.assertEqual(setup.focus_compartments, ("barnase", "barstar"))
        self.assertEqual(setup.topology.validate_against_particle_state(setup.initial_particles), ())
        self.assertGreater(setup.integrator_time_step, scenario.recommended_time_step)
        self.assertGreater(setup.integrator_friction, scenario.recommended_friction)
        self.assertEqual(setup.metadata["shadow_tuning_preset"], "protein_large_step_fast")

    def test_reference_case_is_available(self) -> None:
        case = BarnaseBarstarScenario().reference_case()

        self.assertEqual(case.name, "barnase_barstar")
        self.assertEqual(case.structural_reference.unbound_partner_pdb_ids, ("1A2P", "1A19"))

    def test_structure_report_is_available(self) -> None:
        scenario = BarnaseBarstarScenario()
        setup = scenario.build_setup()
        registry = SimulationStateRegistry(created_by="unit-test")
        state = registry.create_initial_state(
            particles=setup.initial_particles,
            thermodynamics=setup.thermodynamics,
        )

        report = scenario.build_structure_report(state)

        self.assertEqual(report.title, "Atomistic Alignment")
        self.assertIn("local atomistic centroids derived from 1BRS", report.summary)
        self.assertIn("Atomistic Centroid RMSD", {metric.label for metric in report.metrics})

    def test_shadow_correction_model_is_available(self) -> None:
        scenario = BarnaseBarstarScenario()
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
                    region_id="region-barnase-shadow-001",
                    state_id=state.provenance.state_id,
                    particle_indices=(1, 2, 4, 5),
                    seed_pairs=((1, 5), (2, 4)),
                    trigger_kinds=(RegionTriggerKind.ADAPTIVE_EDGE,),
                    score=1.0,
                ),
            ),
            scenario.build_qcloud_correction_model(),
        ).force_evaluation

        self.assertNotEqual(corrected.potential_energy, baseline.potential_energy)

    def test_shadow_fidelity_report_is_available(self) -> None:
        scenario = BarnaseBarstarScenario()
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
                    region_id="region-barnase-shadow-002",
                    state_id=state.provenance.state_id,
                    particle_indices=(1, 2, 4, 5),
                    seed_pairs=((1, 5), (2, 4)),
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
        self.assertEqual(report.target_label, "1BRS Shadow Contact Target")
        self.assertIn("force_rms_error", {metric.label for metric in report.metrics})
        self.assertTrue(any(metric.corrected_error != metric.baseline_error for metric in report.metrics))

    def test_proxy_progress_reaches_native_like_docking(self) -> None:
        scenario = BarnaseBarstarScenario()
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

        for _ in range(100):
            loop.run(1)

        state = registry.latest_state()
        graph = graph_manager.initialize(state, setup.topology)
        progress = scenario.measure_progress(state, graph=graph)

        self.assertLess(progress.interface_distance, progress.initial_interface_distance)
        self.assertGreaterEqual(progress.cross_contact_count, 3)
        self.assertIn(progress.stage_label, {"Encounter Complex", "Native-Like Docking"})
        self.assertGreaterEqual(progress.assembly_score, 0.6)


if __name__ == "__main__":
    unittest.main()
