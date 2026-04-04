"""Tests for Section 2 state models and lifecycle helpers."""

from __future__ import annotations

import unittest

from core.exceptions import ContractValidationError
from core.state import (
    EnsembleKind,
    ParticleState,
    SimulationCell,
    SimulationState,
    StateProvenance,
    ThermodynamicState,
    UnitSystem,
)
from core.state_registry import LifecycleStage, SimulationStateRegistry
from core.types import SimulationId, StateId


class ParticleStateTests(unittest.TestCase):
    """Verify particle-resolved state invariants."""

    def test_particle_state_defaults_zero_forces_and_computes_kinetic_energy(self) -> None:
        particles = ParticleState(
            positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            masses=(2.0, 1.0),
            velocities=((1.0, 0.0, 0.0), (0.0, 2.0, 0.0)),
        )

        self.assertEqual(particles.forces, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        self.assertAlmostEqual(particles.kinetic_energy(), 3.0)
        self.assertEqual(particles.center_of_mass(), (1.0 / 3.0, 0.0, 0.0))

    def test_particle_state_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ContractValidationError):
            ParticleState(
                positions=((0.0, 0.0, 0.0),),
                masses=(1.0, 2.0),
            )


class ThermodynamicStateTests(unittest.TestCase):
    """Verify ensemble requirements stay explicit."""

    def test_nvt_requires_temperature(self) -> None:
        with self.assertRaises(ContractValidationError):
            ThermodynamicState(ensemble=EnsembleKind.NVT)

    def test_npt_requires_pressure(self) -> None:
        with self.assertRaises(ContractValidationError):
            ThermodynamicState(
                ensemble=EnsembleKind.NPT,
                target_temperature=300.0,
            )


class SimulationStateTests(unittest.TestCase):
    """Verify complete state objects serialize and validate correctly."""

    def test_simulation_state_roundtrip(self) -> None:
        units = UnitSystem.md_nano()
        particles = ParticleState(
            positions=((0.0, 0.0, 0.0),),
            masses=(12.0,),
            velocities=((0.0, 0.0, 0.0),),
        )
        thermo = ThermodynamicState(
            ensemble=EnsembleKind.NVT,
            target_temperature=310.0,
            friction_coefficient=1.0,
        )
        provenance = StateProvenance(
            simulation_id=SimulationId("sim-test"),
            state_id=StateId("state-test"),
            parent_state_id=None,
            created_by="unit-test",
            stage="initialization",
            metadata={"source": "roundtrip"},
        )
        state = SimulationState(
            units=units,
            particles=particles,
            thermodynamics=thermo,
            provenance=provenance,
            cell=SimulationCell(
                box_vectors=((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)),
                periodic_axes=(True, True, True),
            ),
            time=0.0,
            step=0,
            potential_energy=-1.5,
            observables={"temperature_estimate": 308.2},
        )

        restored = SimulationState.from_dict(state.to_dict())
        self.assertEqual(restored, state)
        self.assertAlmostEqual(restored.total_energy(), -1.5)


class StateRegistryTests(unittest.TestCase):
    """Verify lineage creation and registry safeguards."""

    def test_registry_creates_lineage_and_summaries(self) -> None:
        registry = SimulationStateRegistry(created_by="unit-test")
        initial = registry.create_initial_state(
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                masses=(1.0, 1.0),
                velocities=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ),
            thermodynamics=ThermodynamicState(),
            metadata={"seed": 42},
        )

        propagated_particles = initial.particles.with_positions(
            ((0.1, 0.0, 0.0), (1.1, 0.0, 0.0))
        )
        next_state = registry.derive_state(
            initial,
            particles=propagated_particles,
            time=0.5,
            potential_energy=-3.2,
            stage=LifecycleStage.INTEGRATION,
            metadata={"integrator": "placeholder"},
        )

        self.assertEqual(len(registry), 2)
        self.assertEqual(
            registry.lineage_for(next_state.provenance.state_id),
            (initial.provenance.state_id, next_state.provenance.state_id),
        )
        summary = registry.summaries()[-1]
        self.assertEqual(summary.stage, LifecycleStage.INTEGRATION.value)
        self.assertEqual(summary.step, 1)
        self.assertEqual(summary.particle_count, 2)

    def test_registry_rejects_unregistered_parent(self) -> None:
        registry = SimulationStateRegistry(created_by="unit-test")
        foreign_state = SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0),),
                masses=(1.0,),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=registry.require_simulation_id(),
                state_id=StateId("state-foreign"),
                parent_state_id=None,
                created_by="unit-test",
                stage="foreign",
            ),
        )

        with self.assertRaises(ContractValidationError):
            registry.derive_state(foreign_state)


if __name__ == "__main__":
    unittest.main()
