"""Tests for the BAOAB Langevin splitting integrator."""

from __future__ import annotations

import unittest

from core.exceptions import ContractValidationError
from core.state import EnsembleKind, ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.types import BeadId, SimulationId, StateId
from forcefields.base_forcefield import BaseForceField, BondParameter, NonbondedParameter
from integrators.baoab import BAOABIntegrator
from physics.forces.composite import BaselineForceEvaluator, ForceEvaluation
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


class _CountingForceEvaluator:
    """Force evaluator that counts how many times evaluate() is called."""

    name = "counting_force_evaluator"
    classification = "[test]"

    def __init__(self) -> None:
        self.call_count = 0

    def evaluate(self, state: SimulationState, topology: SystemTopology, forcefield: BaseForceField) -> ForceEvaluation:
        del topology, forcefield
        self.call_count += 1
        return ForceEvaluation(
            forces=tuple((0.0, 0.0, 0.0) for _ in range(state.particle_count)),
            potential_energy=0.0,
        )


class BAOABIntegratorTests(unittest.TestCase):
    """Verify the BAOAB Langevin splitting integrator."""

    def _build_baseline_system(self) -> tuple[SimulationState, SystemTopology, BaseForceField]:
        state = SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (2.4, 0.0, 0.0)),
                masses=(1.0, 1.0, 1.0),
                velocities=((0.1, 0.0, 0.0), (-0.1, 0.0, 0.0), (0.0, 0.1, 0.0)),
            ),
            thermodynamics=ThermodynamicState(
                ensemble=EnsembleKind.NVT,
                target_temperature=1.0,
                friction_coefficient=1.0,
            ),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-baoab"),
                state_id=StateId("state-baoab"),
                parent_state_id=None,
                created_by="unit-test",
                stage="initialization",
            ),
        )
        topology = SystemTopology(
            system_id="baoab-system",
            bead_types=(
                BeadType(name="bb", role=BeadRole.STRUCTURAL),
                BeadType(name="sc", role=BeadRole.FUNCTIONAL),
            ),
            beads=(
                Bead(bead_id=BeadId("b0"), particle_index=0, bead_type="bb", label="B0"),
                Bead(bead_id=BeadId("b1"), particle_index=1, bead_type="bb", label="B1"),
                Bead(bead_id=BeadId("b2"), particle_index=2, bead_type="sc", label="S0"),
            ),
            bonds=(Bond(0, 1),),
        )
        forcefield = BaseForceField(
            name="baoab-ff",
            bond_parameters=(BondParameter("bb", "bb", equilibrium_distance=1.2, stiffness=50.0),),
            nonbonded_parameters=(
                NonbondedParameter("bb", "sc", sigma=1.0, epsilon=0.5, cutoff=3.0),
                NonbondedParameter("bb", "bb", sigma=1.0, epsilon=0.2, cutoff=3.0),
                NonbondedParameter("sc", "sc", sigma=1.1, epsilon=0.4, cutoff=3.0),
            ),
        )
        return state, topology, forcefield

    def test_baoab_step_advances_time_and_step(self) -> None:
        state, topology, forcefield = self._build_baseline_system()
        integrator = BAOABIntegrator(
            time_step=0.01,
            friction_coefficient=1.0,
            assume_reduced_units=True,
            random_seed=42,
        )
        result = integrator.step(state, topology, forcefield, _ZeroForceEvaluator())

        self.assertAlmostEqual(result.time, 0.01)
        self.assertEqual(result.step, 1)

    def test_baoab_conserves_energy_approximately(self) -> None:
        state, topology, forcefield = self._build_baseline_system()
        evaluator = BaselineForceEvaluator()
        integrator = BAOABIntegrator(
            time_step=0.001,
            friction_coefficient=0.0,
            assume_reduced_units=True,
            random_seed=42,
        )

        current_state = state
        # Run initial force evaluation to get starting potential energy.
        initial_eval = evaluator.evaluate(current_state, topology, forcefield)
        initial_ke = 0.5 * sum(
            current_state.particles.masses[i]
            * sum(current_state.particles.velocities[i][ax] ** 2 for ax in range(3))
            for i in range(current_state.particle_count)
        )
        initial_total = initial_ke + (initial_eval.potential_energy or 0.0)

        for _ in range(20):
            result = integrator.step(current_state, topology, forcefield, evaluator)
            current_state = SimulationState(
                units=current_state.units,
                particles=result.particles,
                thermodynamics=current_state.thermodynamics,
                provenance=current_state.provenance,
                time=result.time,
                step=result.step,
                potential_energy=result.potential_energy,
                observables=result.observables,
            )

        final_ke = result.observables["kinetic_energy"]
        final_pe = result.potential_energy or 0.0
        final_total = final_ke + final_pe

        # With gamma=0 and small dt the O-step is identity, so energy should
        # be approximately conserved (not diverge).
        self.assertAlmostEqual(initial_total, final_total, places=1)

    def test_baoab_single_force_evaluation(self) -> None:
        state, topology, forcefield = self._build_baseline_system()
        counter = _CountingForceEvaluator()
        integrator = BAOABIntegrator(
            time_step=0.01,
            friction_coefficient=1.0,
            assume_reduced_units=True,
            random_seed=42,
        )

        integrator.step(state, topology, forcefield, counter)
        self.assertEqual(counter.call_count, 1)

    def test_baoab_with_thermostat(self) -> None:
        state, topology, forcefield = self._build_baseline_system()
        integrator = BAOABIntegrator(
            time_step=0.01,
            friction_coefficient=5.0,
            assume_reduced_units=True,
            random_seed=99,
        )
        result = integrator.step(state, topology, forcefield, _ZeroForceEvaluator())

        # With friction > 0 and temperature > 0, the O-step should modify
        # velocities stochastically.  The metadata should indicate stochastic=True.
        self.assertTrue(result.metadata["stochastic"])
        self.assertGreater(result.metadata["friction_coefficient"], 0.0)

        # Velocities should differ from a simple drift (friction damps them and
        # noise is injected).
        original_vel = state.particles.velocities
        new_vel = result.particles.velocities
        changed = any(
            original_vel[i][ax] != new_vel[i][ax]
            for i in range(state.particle_count)
            for ax in range(3)
        )
        self.assertTrue(changed, "Thermostat should modify velocities.")

    def test_baoab_deterministic_with_seed(self) -> None:
        state, topology, forcefield = self._build_baseline_system()

        def _run_one_step(seed: int) -> tuple:
            integrator = BAOABIntegrator(
                time_step=0.01,
                friction_coefficient=1.0,
                assume_reduced_units=True,
                random_seed=seed,
            )
            result = integrator.step(state, topology, forcefield, _ZeroForceEvaluator())
            return result.particles.positions, result.particles.velocities

        pos_a, vel_a = _run_one_step(seed=12345)
        pos_b, vel_b = _run_one_step(seed=12345)

        self.assertEqual(pos_a, pos_b)
        self.assertEqual(vel_a, vel_b)

    def test_baoab_validation_rejects_bad_parameters(self) -> None:
        with self.assertRaises(ContractValidationError):
            BAOABIntegrator(time_step=-0.01, assume_reduced_units=True)

        with self.assertRaises(ContractValidationError):
            BAOABIntegrator(time_step=0.01, friction_coefficient=-1.0, assume_reduced_units=True)

        with self.assertRaises(ContractValidationError):
            BAOABIntegrator(time_step=0.01, thermal_energy_scale=-1.0, assume_reduced_units=True)

        with self.assertRaises(ContractValidationError):
            BAOABIntegrator(time_step=0.01, assume_reduced_units=False)

    def test_baoab_kinetic_energy_computed(self) -> None:
        state, topology, forcefield = self._build_baseline_system()
        integrator = BAOABIntegrator(
            time_step=0.01,
            friction_coefficient=1.0,
            assume_reduced_units=True,
            random_seed=42,
        )
        result = integrator.step(state, topology, forcefield, _ZeroForceEvaluator())

        self.assertIn("kinetic_energy", result.observables)
        ke = result.observables["kinetic_energy"]
        self.assertIsInstance(ke, float)
        self.assertGreaterEqual(ke, 0.0)

    def test_baoab_positions_change(self) -> None:
        state, topology, forcefield = self._build_baseline_system()
        integrator = BAOABIntegrator(
            time_step=0.01,
            friction_coefficient=1.0,
            assume_reduced_units=True,
            random_seed=42,
        )
        result = integrator.step(state, topology, forcefield, BaselineForceEvaluator())

        # Particles have non-zero initial velocities and/or forces, so
        # positions must change after a step.
        self.assertNotEqual(result.particles.positions, state.particles.positions)


if __name__ == "__main__":
    unittest.main()
