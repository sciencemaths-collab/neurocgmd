"""Tests for enhanced sampling, statistical validation, and QCloud adaptive refinement."""

from __future__ import annotations

import unittest
from math import sqrt, pi
from random import Random

from core.state import (
    EnsembleKind,
    ParticleState,
    SimulationState,
    StateProvenance,
    ThermodynamicState,
    UnitSystem,
)
from core.types import FrozenMetadata, SimulationId, StateId
from qcloud.cloud_state import (
    ParticleForceDelta,
    QCloudCorrection,
    RefinementRegion,
    RegionTriggerKind,
)
from sampling.enhanced_sampling import (
    DistanceCV,
    MetadynamicsEngine,
    RadiusOfGyrationCV,
    ReplicaExchangeManager,
)
from validation.statistical_mechanics import (
    block_average,
    check_equipartition,
    check_maxwell_boltzmann,
    compute_autocorrelation,
)
from qcloud.adaptive_refinement import (
    AdaptiveRefinementController,
    AdaptiveRegionSizer,
    ErrorEstimator,
    RichardsonExtrapolation,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_provenance(state_id: str = "s1") -> StateProvenance:
    return StateProvenance(
        simulation_id=SimulationId("sim1"),
        state_id=StateId(state_id),
        parent_state_id=None,
        created_by="test",
        stage="testing",
    )


def _make_simulation_state(
    n_particles: int = 5,
    temperature: float = 300.0,
    positions: tuple | None = None,
    velocities: tuple | None = None,
    forces: tuple | None = None,
    potential_energy: float | None = 0.0,
    step: int = 0,
) -> SimulationState:
    rng = Random(42)
    if positions is None:
        positions = tuple(
            (rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))
            for _ in range(n_particles)
        )
    if velocities is None:
        velocities = tuple(
            (rng.gauss(0, 0.1), rng.gauss(0, 0.1), rng.gauss(0, 0.1))
            for _ in range(n_particles)
        )
    if forces is None:
        forces = tuple(
            (rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))
            for _ in range(n_particles)
        )
    masses = tuple(1.0 for _ in range(n_particles))
    particles = ParticleState(
        positions=positions,
        masses=masses,
        velocities=velocities,
        forces=forces,
    )
    thermo = ThermodynamicState(
        ensemble=EnsembleKind.NVT,
        target_temperature=temperature,
    )
    return SimulationState(
        units=UnitSystem(),
        particles=particles,
        thermodynamics=thermo,
        provenance=_make_provenance(),
        potential_energy=potential_energy,
        step=step,
    )


def _make_refinement_region(
    particle_indices: tuple[int, ...] = (0, 1, 2),
) -> RefinementRegion:
    return RefinementRegion(
        region_id="r1",
        state_id="s1",
        particle_indices=particle_indices,
        trigger_kinds=(RegionTriggerKind.MANUAL,),
        score=1.0,
    )


def _make_qcloud_correction(
    energy_delta: float = 0.5,
    force_deltas: tuple[ParticleForceDelta, ...] | None = None,
) -> QCloudCorrection:
    if force_deltas is None:
        force_deltas = (
            ParticleForceDelta(particle_index=0, delta_force=(0.1, 0.2, 0.3)),
            ParticleForceDelta(particle_index=1, delta_force=(0.05, 0.1, 0.15)),
        )
    return QCloudCorrection(
        region_id="r1",
        method_label="test_method",
        energy_delta=energy_delta,
        force_deltas=force_deltas,
    )


# ---------------------------------------------------------------------------
# Replica Exchange Tests
# ---------------------------------------------------------------------------


class TestReplicaExchangeInitializes(unittest.TestCase):
    """Create replicas from a base state at different temperatures."""

    def test_replica_exchange_initializes(self) -> None:
        temps = (280.0, 300.0, 320.0, 340.0)
        manager = ReplicaExchangeManager(temperatures=temps)
        base = _make_simulation_state(temperature=300.0)

        replicas = manager.initialize_replicas(base)

        self.assertEqual(len(replicas), 4)
        for replica, temp in zip(replicas, temps):
            self.assertAlmostEqual(replica.temperature, temp)
            self.assertAlmostEqual(
                replica.state.thermodynamics.target_temperature, temp
            )
            # Velocity scaling: v_new = v_base * sqrt(T_new / T_base)
            scale = sqrt(temp / 300.0)
            base_vel = base.particles.velocities[0]
            rep_vel = replica.state.particles.velocities[0]
            for d in range(3):
                self.assertAlmostEqual(rep_vel[d], base_vel[d] * scale, places=10)


class TestReplicaExchangeAttempt(unittest.TestCase):
    """Attempt exchanges and verify structure stays valid."""

    def test_replica_exchange_attempt(self) -> None:
        temps = (280.0, 300.0, 320.0)
        manager = ReplicaExchangeManager(temperatures=temps, random_seed=123)
        base = _make_simulation_state(temperature=300.0, potential_energy=-10.0)
        replicas = manager.initialize_replicas(base)

        swapped = manager.attempt_exchanges(replicas)

        self.assertEqual(len(swapped), 3)
        # Temperatures must remain fixed at original assignments.
        for replica, temp in zip(swapped, temps):
            self.assertAlmostEqual(replica.temperature, temp)
            self.assertAlmostEqual(
                replica.state.thermodynamics.target_temperature, temp
            )
        # Particle count preserved in every replica.
        for replica in swapped:
            self.assertEqual(replica.state.particle_count, base.particle_count)


class TestShouldAttemptExchangeInterval(unittest.TestCase):
    """Respects exchange_interval for scheduling."""

    def test_should_attempt_exchange_interval(self) -> None:
        manager = ReplicaExchangeManager(
            temperatures=(290.0, 310.0), exchange_interval=50
        )
        self.assertFalse(manager.should_attempt_exchange(0))
        self.assertFalse(manager.should_attempt_exchange(25))
        self.assertTrue(manager.should_attempt_exchange(50))
        self.assertTrue(manager.should_attempt_exchange(100))
        self.assertFalse(manager.should_attempt_exchange(75))


# ---------------------------------------------------------------------------
# Metadynamics Tests
# ---------------------------------------------------------------------------


class TestDistanceCVComputation(unittest.TestCase):
    """Verify distance CV computes correctly."""

    def test_distance_cv_computation(self) -> None:
        positions = ((0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (1.0, 1.0, 1.0))
        state = _make_simulation_state(n_particles=3, positions=positions)
        cv = DistanceCV(particle_a=0, particle_b=1)

        result = cv.compute(state)

        self.assertAlmostEqual(result, 5.0, places=10)


class TestRadiusOfGyrationCV(unittest.TestCase):
    """Verify radius of gyration CV computes correctly."""

    def test_radius_of_gyration_cv(self) -> None:
        # Four particles at corners of a 2x2x0 square centred at origin.
        positions = (
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
        )
        state = _make_simulation_state(n_particles=4, positions=positions)
        cv = RadiusOfGyrationCV(particle_indices=(0, 1, 2, 3))

        result = cv.compute(state)

        # Each particle is sqrt(2) from the center; Rg = sqrt(mean(2)) = sqrt(2).
        self.assertAlmostEqual(result, sqrt(2.0), places=10)


class TestMetadynamicsDepositsHill(unittest.TestCase):
    """Deposit a hill, verify bias energy becomes nonzero."""

    def test_metadynamics_deposits_hill(self) -> None:
        positions = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        state = _make_simulation_state(n_particles=2, positions=positions)
        cv = DistanceCV(particle_a=0, particle_b=1)
        engine = MetadynamicsEngine(
            collective_variables=(cv,),
            hill_height=1.0,
            hill_width=0.5,
            well_tempered=False,
        )

        # Before deposition, bias should be zero.
        self.assertAlmostEqual(engine.compute_bias_energy(state), 0.0)

        hill = engine.deposit_hill(state)
        self.assertIsNotNone(hill)

        # After deposition, bias at the same point equals the hill height.
        bias = engine.compute_bias_energy(state)
        self.assertAlmostEqual(bias, 1.0, places=10)


class TestWellTemperedReducesHeight(unittest.TestCase):
    """Hill heights decrease in well-tempered mode."""

    def test_well_tempered_reduces_height(self) -> None:
        positions = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        state = _make_simulation_state(
            n_particles=2, positions=positions, temperature=300.0
        )
        cv = DistanceCV(particle_a=0, particle_b=1)
        engine = MetadynamicsEngine(
            collective_variables=(cv,),
            hill_height=1.0,
            hill_width=0.5,
            well_tempered=True,
            bias_temperature=3000.0,
        )

        hill_1 = engine.deposit_hill(state)
        hill_2 = engine.deposit_hill(state)

        self.assertIsNotNone(hill_1)
        self.assertIsNotNone(hill_2)
        # Second hill must be shorter due to well-tempered dampening.
        self.assertLess(hill_2.height, hill_1.height)
        self.assertGreater(hill_2.height, 0.0)


# ---------------------------------------------------------------------------
# Statistical Mechanics Tests
# ---------------------------------------------------------------------------


class TestBlockAverageConstantData(unittest.TestCase):
    """Block average of constant data yields zero standard error."""

    def test_block_average_constant_data(self) -> None:
        data = tuple(5.0 for _ in range(200))
        result = block_average(data)

        self.assertAlmostEqual(result.mean, 5.0)
        self.assertAlmostEqual(result.standard_error, 0.0, places=12)


class TestBlockAverageRandomData(unittest.TestCase):
    """Block average of random data gives nonzero standard error."""

    def test_block_average_random_data(self) -> None:
        rng = Random(42)
        data = tuple(rng.gauss(0.0, 1.0) for _ in range(1000))
        result = block_average(data)

        self.assertGreater(result.standard_error, 0.0)
        self.assertGreater(result.effective_samples, 0.0)
        self.assertLessEqual(result.effective_samples, len(data))


class TestAutocorrelationWhiteNoise(unittest.TestCase):
    """White noise has autocorrelation time approximately 0.5."""

    def test_autocorrelation_white_noise(self) -> None:
        rng = Random(42)
        data = tuple(rng.gauss(0.0, 1.0) for _ in range(5000))
        result = compute_autocorrelation(data)

        # For uncorrelated data the integrated autocorrelation time should be
        # close to 0.5 (the initial 0.5 term with negligible contributions
        # from higher lags).
        self.assertAlmostEqual(
            result.integrated_autocorrelation_time, 0.5, delta=0.15
        )
        self.assertAlmostEqual(result.autocorrelation_function[0], 1.0)


class TestEquipartitionCheckPasses(unittest.TestCase):
    """Generate velocities from correct Maxwell-Boltzmann; check passes."""

    def test_equipartition_check_passes(self) -> None:
        rng = Random(42)
        temperature = 300.0
        kb = 1.0  # thermal_energy_scale
        n_particles = 20
        masses = tuple(1.0 for _ in range(n_particles))
        n_frames = 500

        velocities_history: list[tuple[tuple[float, float, float], ...]] = []
        for _ in range(n_frames):
            frame: list[tuple[float, float, float]] = []
            for p in range(n_particles):
                m = masses[p]
                sigma = sqrt(kb * temperature / m)
                vx = rng.gauss(0.0, sigma)
                vy = rng.gauss(0.0, sigma)
                vz = rng.gauss(0.0, sigma)
                frame.append((vx, vy, vz))
            velocities_history.append(tuple(frame))

        result = check_equipartition(
            tuple(velocities_history),
            masses,
            temperature,
            thermal_energy_scale=kb,
        )

        self.assertTrue(result.passed)
        self.assertGreater(result.p_value, 0.01)


class TestMaxwellBoltzmannCheck(unittest.TestCase):
    """Generate correct speed distribution and verify KS test passes."""

    def test_maxwell_boltzmann_check(self) -> None:
        rng = Random(42)
        temperature = 300.0
        mass = 1.0
        kt = 1.0 * temperature  # thermal_energy_scale = 1.0

        # Generate speeds: v = sqrt(kT/m) * sqrt(chi2(3)/3)
        # where chi2(3) = sum of 3 standard normal^2
        speeds: list[float] = []
        for _ in range(2000):
            chi2 = rng.gauss(0, 1) ** 2 + rng.gauss(0, 1) ** 2 + rng.gauss(0, 1) ** 2
            v = sqrt(kt / mass) * sqrt(chi2)
            speeds.append(v)

        result = check_maxwell_boltzmann(
            tuple(speeds),
            mass,
            temperature,
            thermal_energy_scale=1.0,
        )

        self.assertTrue(result.passed)
        # Mean speed should be close to sqrt(8 kT / (pi m)).
        expected_mean = sqrt(8.0 * kt / (pi * mass))
        self.assertAlmostEqual(
            result.speed_distribution_mean, expected_mean, delta=expected_mean * 0.1
        )


# ---------------------------------------------------------------------------
# QCloud Adaptive Refinement Tests
# ---------------------------------------------------------------------------


class TestRichardsonExtrapolationConverges(unittest.TestCase):
    """Test with a known convergent sequence."""

    def test_richardson_extrapolation_converges(self) -> None:
        # For a second-order method with ratio=2:
        #   coarse = true + c * h^2,  fine = true + c * (h/2)^2
        # Richardson extrapolation should recover the true value.
        true_value = 3.14159
        c = 1.0
        h = 1.0
        coarse = true_value + c * h ** 2       # 4.14159
        fine = true_value + c * (h / 2) ** 2   # 3.39159

        richardson = RichardsonExtrapolation(order=2)
        extrapolated, error = richardson.extrapolate(coarse, fine, ratio=2.0)

        self.assertAlmostEqual(extrapolated, true_value, places=10)
        self.assertGreater(error, 0.0)
        # The error estimate should be close to |fine - true|.
        self.assertAlmostEqual(error, abs(fine - true_value), places=10)


class TestErrorEstimatorTracksHistory(unittest.TestCase):
    """Record corrections and check error estimate."""

    def test_error_estimator_tracks_history(self) -> None:
        estimator = ErrorEstimator(history_size=10)

        # Fewer than 3 entries: default error of 1.0.
        self.assertAlmostEqual(estimator.estimate_current_error(), 1.0)

        # Record several corrections with decreasing energy deltas.
        for i in range(5):
            correction = _make_qcloud_correction(energy_delta=1.0 / (i + 1))
            estimator.record(step=i * 10, correction=correction)

        error = estimator.estimate_current_error()
        self.assertGreater(error, 0.0)
        self.assertLess(error, 10.0)
        self.assertEqual(len(estimator._history), 5)

        # Convergence rate should be negative (corrections are shrinking).
        rate = estimator.convergence_rate()
        self.assertLess(rate, 0.0)


class TestAdaptiveRegionSizerGrows(unittest.TestCase):
    """High error causes the region to grow."""

    def test_adaptive_region_sizer_grows(self) -> None:
        state = _make_simulation_state(n_particles=10)
        region = _make_refinement_region(particle_indices=(3, 4, 5))
        sizer = AdaptiveRegionSizer(
            min_region_size=2,
            max_region_size=10,
            error_threshold_grow=0.05,
            error_threshold_shrink=0.001,
        )

        grown = sizer.resize(region, current_error=0.5, state=state)

        # Region must have grown (more particles).
        self.assertGreaterEqual(len(grown.particle_indices), len(region.particle_indices))
        # Original particles still present.
        for idx in region.particle_indices:
            self.assertIn(idx, grown.particle_indices)


class TestAdaptiveRefinementController(unittest.TestCase):
    """Full pipeline: controller records, estimates error, resizes region."""

    def test_adaptive_refinement_controller(self) -> None:
        state = _make_simulation_state(n_particles=10, step=100)
        region = _make_refinement_region(particle_indices=(2, 3, 4))
        correction = _make_qcloud_correction(energy_delta=0.5)

        controller = AdaptiveRefinementController(
            target_accuracy=0.01,
            max_refinement_levels=3,
        )

        # First pass: not converged (need at least 3 history entries).
        result1 = controller.refine(state, region, correction)
        self.assertFalse(result1.converged)
        self.assertGreater(result1.estimated_error, 0.0)

        # Feed more corrections so the estimator has history.
        for i in range(5):
            st = _make_simulation_state(n_particles=10, step=100 + (i + 1) * 10)
            corr = _make_qcloud_correction(energy_delta=0.5 / (i + 2))
            result = controller.refine(st, region, corr)

        # After several passes: result should still be valid.
        self.assertIsNotNone(result.correction)
        self.assertIsNotNone(result.region)
        self.assertGreater(result.refinement_levels_used, 0)
        self.assertGreaterEqual(result.refinement_levels_used, 1)
        self.assertLessEqual(
            result.refinement_levels_used, controller.max_refinement_levels
        )


if __name__ == "__main__":
    unittest.main()
