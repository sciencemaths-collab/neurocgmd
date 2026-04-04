"""Tests for the concrete encounter-complex dashboard scenario."""

from __future__ import annotations

import unittest

from core.state_registry import SimulationStateRegistry
from graph import ConnectivityGraphManager
from integrators.langevin import LangevinIntegrator
from physics.forces.composite import BaselineForceEvaluator
from sampling.scenarios import EncounterComplexScenario
from sampling.simulation_loop import SimulationLoop


class ComplexAssemblyScenarioTests(unittest.TestCase):
    """Validate the concrete encounter-complex setup and progress model."""

    def test_setup_is_topology_aligned(self) -> None:
        scenario = EncounterComplexScenario()
        setup = scenario.build_setup()

        self.assertEqual(setup.initial_particles.particle_count, 6)
        self.assertEqual(setup.focus_compartments, ("A", "B"))
        self.assertEqual(setup.topology.validate_against_particle_state(setup.initial_particles), ())

    def test_initial_progress_is_unbound(self) -> None:
        scenario = EncounterComplexScenario()
        setup = scenario.build_setup()
        registry = SimulationStateRegistry(created_by="unit-test")
        state = registry.create_initial_state(
            particles=setup.initial_particles,
            thermodynamics=setup.thermodynamics,
        )

        progress = scenario.measure_progress(state)

        self.assertFalse(progress.bound)
        self.assertEqual(progress.cross_contact_count, 0)
        self.assertEqual(progress.stage_label, "Separated Search")
        self.assertGreater(progress.interface_distance, progress.capture_distance)

    def test_recommended_dynamics_reach_locked_complex(self) -> None:
        scenario = EncounterComplexScenario()
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

        for _ in range(120):
            loop.run(1)

        state = registry.latest_state()
        graph = graph_manager.initialize(state, setup.topology)
        progress = scenario.measure_progress(state, graph=graph)

        self.assertTrue(progress.bound)
        self.assertLess(progress.interface_distance, progress.initial_interface_distance)
        self.assertGreaterEqual(progress.cross_contact_count, scenario.target_contact_count)
        self.assertGreaterEqual(progress.assembly_score, 0.8)


if __name__ == "__main__":
    unittest.main()
