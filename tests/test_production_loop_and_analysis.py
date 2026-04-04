"""Comprehensive tests for the production simulation loop and adaptive analysis engine.

Tests the full integration of all NeuroCGMD modules working together:
  - TemperatureSchedule, EnergyTracker, ProductionSimulationLoop (sampling.production_loop)
  - RMSDTracker, RDFCalculator, UmbrellaSampler, BindingEnergyEstimator,
    AdaptiveAnalysisEngine, AnalysisReport (validation.adaptive_analysis)
  - End-to-end pipeline: build system -> run production -> analyse -> report
"""

from __future__ import annotations

import unittest
from math import sqrt, pi

from core.state import (
    EnsembleKind,
    ParticleState,
    SimulationCell,
    SimulationState,
    StateProvenance,
    ThermodynamicState,
    UnitSystem,
)
from core.state_registry import SimulationStateRegistry
from core.types import BeadId, FrozenMetadata, SimulationId, StateId
from topology.beads import Bead, BeadRole, BeadType
from topology.bonds import Bond
from topology.system_topology import SystemTopology
from forcefields.base_forcefield import BaseForceField, BondParameter, NonbondedParameter
from physics.forces.composite import BaselineForceEvaluator, ForceEvaluation
from integrators.baoab import BAOABIntegrator


# ---------------------------------------------------------------------------
# Shared 4-particle fixture
# ---------------------------------------------------------------------------


class _FourParticleFixture:
    """Builds a complete 4-particle periodic NVT system for reuse by all tests."""

    def build(self) -> dict:
        """Return a dict with all components needed for a production run."""

        cell = SimulationCell(
            box_vectors=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)),
        )

        thermo = ThermodynamicState(
            ensemble=EnsembleKind.NVT,
            target_temperature=1.0,
            friction_coefficient=2.0,
        )

        particles = ParticleState(
            positions=(
                (0.5, 0.5, 0.5),
                (1.7, 0.5, 0.5),
                (2.9, 0.5, 0.5),
                (2.9, 1.7, 0.5),
            ),
            masses=(1.0, 1.0, 1.0, 1.0),
            velocities=(
                (0.1, 0.0, 0.0),
                (-0.1, 0.05, 0.0),
                (0.0, -0.05, 0.1),
                (0.0, 0.0, -0.1),
            ),
        )

        registry = SimulationStateRegistry(created_by="test-fixture")
        initial_state = registry.create_initial_state(
            particles=particles,
            thermodynamics=thermo,
            cell=cell,
        )

        bead_types = (
            BeadType(name="bb", role=BeadRole.STRUCTURAL),
            BeadType(name="sc", role=BeadRole.FUNCTIONAL),
        )
        beads = (
            Bead(bead_id=BeadId("b0"), particle_index=0, bead_type="bb", label="B0"),
            Bead(bead_id=BeadId("b1"), particle_index=1, bead_type="bb", label="B1"),
            Bead(bead_id=BeadId("b2"), particle_index=2, bead_type="sc", label="S0"),
            Bead(bead_id=BeadId("b3"), particle_index=3, bead_type="sc", label="S1"),
        )
        bonds = (Bond(0, 1), Bond(2, 3))
        topology = SystemTopology(
            system_id="prod-test",
            bead_types=bead_types,
            beads=beads,
            bonds=bonds,
        )

        forcefield = BaseForceField(
            name="prod-test-ff",
            bond_parameters=(
                BondParameter("bb", "bb", equilibrium_distance=1.2, stiffness=50.0),
                BondParameter("sc", "sc", equilibrium_distance=1.2, stiffness=50.0),
            ),
            nonbonded_parameters=(
                NonbondedParameter("bb", "sc", sigma=1.0, epsilon=0.5, cutoff=2.5),
                NonbondedParameter("bb", "bb", sigma=1.0, epsilon=0.2, cutoff=2.5),
                NonbondedParameter("sc", "sc", sigma=1.1, epsilon=0.4, cutoff=2.5),
            ),
        )

        force_evaluator = BaselineForceEvaluator()
        integrator = BAOABIntegrator(
            time_step=0.002,
            friction_coefficient=2.0,
            assume_reduced_units=True,
            random_seed=42,
        )

        return {
            "registry": registry,
            "initial_state": initial_state,
            "topology": topology,
            "forcefield": forcefield,
            "force_evaluator": force_evaluator,
            "integrator": integrator,
            "cell": cell,
            "thermo": thermo,
        }


FIXTURE = _FourParticleFixture()


# ===================================================================
# TestTemperatureSchedule
# ===================================================================


class TestTemperatureSchedule(unittest.TestCase):
    """Tests for sampling.production_loop.TemperatureSchedule."""

    def setUp(self) -> None:
        from sampling.production_loop import TemperatureSchedule
        self.TemperatureSchedule = TemperatureSchedule

    def test_constant(self) -> None:
        sched = self.TemperatureSchedule(
            mode="constant",
            initial_temperature=2.0,
        )
        for step in (0, 50, 500, 999, 1000):
            self.assertAlmostEqual(sched.temperature_at_step(step), 2.0)

    def test_linear_ramp(self) -> None:
        sched = self.TemperatureSchedule(
            mode="linear_ramp",
            initial_temperature=4.0,
            final_temperature=2.0,
            total_steps=100,
        )
        self.assertAlmostEqual(sched.temperature_at_step(0), 4.0)
        self.assertAlmostEqual(sched.temperature_at_step(100), 2.0)
        self.assertAlmostEqual(sched.temperature_at_step(50), 3.0)

    def test_exponential_anneal(self) -> None:
        sched = self.TemperatureSchedule(
            mode="exponential_anneal",
            initial_temperature=4.0,
            final_temperature=1.0,
            total_steps=100,
        )
        self.assertAlmostEqual(sched.temperature_at_step(0), 4.0)

        # Monotonically decreasing.
        prev = sched.temperature_at_step(0)
        for step in range(1, 101):
            t = sched.temperature_at_step(step)
            self.assertLessEqual(t, prev + 1e-12)
            prev = t


# ===================================================================
# TestEnergyTracker
# ===================================================================


class TestEnergyTracker(unittest.TestCase):
    """Tests for sampling.production_loop.EnergyTracker."""

    def setUp(self) -> None:
        from sampling.production_loop import EnergyTracker
        self.EnergyTracker = EnergyTracker

    def test_kinetic_energy_computed(self) -> None:
        """KE = 0.5 * m * v^2 for known velocities."""
        tracker = self.EnergyTracker()
        # Single particle: mass=2.0, velocity=(1.0, 0.0, 0.0) -> KE = 0.5*2*1 = 1.0
        state = _make_simple_state(
            positions=((0.0, 0.0, 0.0),),
            masses=(2.0,),
            velocities=((1.0, 0.0, 0.0),),
        )
        ke = tracker.kinetic_energy(state)
        self.assertAlmostEqual(ke, 1.0, places=10)

    def test_instantaneous_temperature(self) -> None:
        """Temperature should be positive for nonzero velocities."""
        tracker = self.EnergyTracker()
        state = _make_simple_state(
            positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            masses=(1.0, 1.0),
            velocities=((0.5, 0.3, 0.1), (-0.2, 0.4, -0.1)),
        )
        t = tracker.instantaneous_temperature(state)
        self.assertGreater(t, 0.0)

    def test_energy_drift(self) -> None:
        """Record 10 energies and verify drift is computable."""
        tracker = self.EnergyTracker()
        for i in range(10):
            tracker.record(step=i, ke=1.0 + 0.01 * i, pe=-0.5, temperature=300.0)
        drift = tracker.energy_drift()
        self.assertIsInstance(drift, float)

    def test_summary_has_all_keys(self) -> None:
        tracker = self.EnergyTracker()
        for i in range(5):
            tracker.record(step=i, ke=1.0, pe=-0.5, temperature=300.0)
        summary = tracker.summary()
        # Check expected top-level keys.
        for key in ("kinetic_energy", "potential_energy", "total_energy", "temperature", "energy_drift", "samples"):
            self.assertIn(key, summary, f"Missing key: {key}")
        # Each sub-dict should have mean, std, min, max.
        for sub_key in ("kinetic_energy", "potential_energy", "total_energy", "temperature"):
            for stat in ("mean", "std", "min", "max"):
                self.assertIn(stat, summary[sub_key], f"Missing {stat} in {sub_key}")


# ===================================================================
# TestProductionLoop
# ===================================================================


class TestProductionLoop(unittest.TestCase):
    """Tests for sampling.production_loop.ProductionSimulationLoop."""

    def setUp(self) -> None:
        from sampling.production_loop import ProductionSimulationLoop, ProductionRunResult
        self.ProductionSimulationLoop = ProductionSimulationLoop
        self.ProductionRunResult = ProductionRunResult

    def _make_loop(self, **overrides):
        """Build a default ProductionSimulationLoop from the 4-particle fixture."""
        parts = FIXTURE.build()
        kwargs = dict(
            topology=parts["topology"],
            forcefield=parts["forcefield"],
            integrator=parts["integrator"],
            force_evaluator=parts["force_evaluator"],
            registry=parts["registry"],
        )
        kwargs.update(overrides)
        return self.ProductionSimulationLoop(**kwargs), parts

    def test_basic_run(self) -> None:
        loop, parts = self._make_loop(analysis_hooks=[], analysis_interval=1000)
        result = loop.run(20)
        self.assertEqual(result.steps_completed, 20)
        self.assertEqual(result.final_state.step, 20)

    def test_energy_tracking(self) -> None:
        loop, _ = self._make_loop(analysis_hooks=[], analysis_interval=1000)
        result = loop.run(10)
        # Energy summary should have potential_energy and kinetic_energy sections.
        self.assertIn("potential_energy", result.energy_summary)
        self.assertIn("kinetic_energy", result.energy_summary)
        self.assertIn("mean", result.energy_summary["potential_energy"])
        self.assertIn("mean", result.energy_summary["kinetic_energy"])

    def test_pbc_wrapping(self) -> None:
        """Particle outside box should be wrapped after one step with apply_pbc=True."""
        cell = SimulationCell(
            box_vectors=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)),
        )
        thermo = ThermodynamicState(
            ensemble=EnsembleKind.NVT,
            target_temperature=1.0,
            friction_coefficient=2.0,
        )
        # Place particle 0 outside the box at x=6.5 (box is [0, 5)).
        particles = ParticleState(
            positions=(
                (6.5, 0.5, 0.5),
                (1.7, 0.5, 0.5),
                (2.9, 0.5, 0.5),
                (2.9, 1.7, 0.5),
            ),
            masses=(1.0, 1.0, 1.0, 1.0),
            velocities=(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
        )
        registry = SimulationStateRegistry(created_by="pbc-test")
        registry.create_initial_state(
            particles=particles,
            thermodynamics=thermo,
            cell=cell,
        )
        parts = FIXTURE.build()
        loop = self.ProductionSimulationLoop(
            topology=parts["topology"],
            forcefield=parts["forcefield"],
            integrator=BAOABIntegrator(
                time_step=0.002,
                friction_coefficient=2.0,
                assume_reduced_units=True,
                random_seed=99,
            ),
            force_evaluator=parts["force_evaluator"],
            registry=registry,
            apply_pbc=True,
            analysis_hooks=[],
            analysis_interval=1000,
        )
        result = loop.run(1)
        # After one step with PBC, particle 0's x coordinate should be inside [0, 5).
        final_x = result.final_state.particles.positions[0][0]
        self.assertGreaterEqual(final_x, 0.0)
        self.assertLess(final_x, 5.0)

    def test_temperature_schedule(self) -> None:
        """With a linear ramp, temperatures recorded in the tracker should vary."""
        from sampling.production_loop import TemperatureSchedule, EnergyTracker

        schedule = TemperatureSchedule(
            mode="linear_ramp",
            initial_temperature=2.0,
            final_temperature=0.5,
            total_steps=20,
        )
        tracker = EnergyTracker()
        loop, _ = self._make_loop(
            temperature_schedule=schedule,
            energy_tracker=tracker,
            analysis_hooks=[],
            analysis_interval=1000,
        )
        loop.run(20)
        # The tracker should have recorded temperatures.
        summary = tracker.summary()
        self.assertGreater(summary["samples"], 0)

    def test_with_constraints(self) -> None:
        """Run with a SHAKE constraint on bond 0-1 and verify bond length is maintained."""
        from physics.constraints import DistanceConstraint, SHAKESolver

        target_dist = 1.2
        constraints = (DistanceConstraint(particle_a=0, particle_b=1, target_distance=target_dist),)
        solver = SHAKESolver(tolerance=1e-6, max_iterations=200)

        loop, parts = self._make_loop(
            constraints=constraints,
            constraint_solver=solver,
            analysis_hooks=[],
            analysis_interval=1000,
        )
        result = loop.run(10)

        # Check bond length is close to the target.
        pos = result.final_state.particles.positions
        dx = pos[0][0] - pos[1][0]
        dy = pos[0][1] - pos[1][1]
        dz = pos[0][2] - pos[1][2]
        bond_length = sqrt(dx * dx + dy * dy + dz * dz)
        self.assertAlmostEqual(bond_length, target_dist, places=3)

    def test_analysis_hooks(self) -> None:
        """Hook should be called during the run."""
        call_log: list[int] = []

        def hook(state, energy_tracker):
            call_log.append(state.step)

        loop, _ = self._make_loop(
            analysis_hooks=[hook],
            analysis_interval=1,  # call every step
        )
        loop.run(5)
        # Hook should have been called at least once.
        self.assertGreater(len(call_log), 0)


# ===================================================================
# TestRMSDTracker
# ===================================================================


class TestRMSDTracker(unittest.TestCase):
    """Tests for validation.adaptive_analysis.RMSDTracker."""

    def setUp(self) -> None:
        from validation.adaptive_analysis import RMSDTracker
        self.RMSDTracker = RMSDTracker

    def test_rmsd_zero_for_same_positions(self) -> None:
        ref = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        tracker = self.RMSDTracker(reference_positions=ref)
        rmsd = tracker.compute_rmsd(ref)
        self.assertAlmostEqual(rmsd, 0.0, places=10)

    def test_rmsd_positive_for_different(self) -> None:
        ref = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        shifted = ((0.1, 0.0, 0.0), (1.1, 0.0, 0.0))
        tracker = self.RMSDTracker(reference_positions=ref)
        rmsd = tracker.compute_rmsd(shifted)
        self.assertGreater(rmsd, 0.0)

    def test_rmsf_per_particle(self) -> None:
        ref = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        tracker = self.RMSDTracker(reference_positions=ref)
        # Record several frames with small perturbations.
        for i in range(5):
            state = _make_simple_state(
                positions=((0.01 * i, 0.0, 0.0), (1.0 + 0.01 * i, 0.0, 0.0)),
                masses=(1.0, 1.0),
            )
            tracker.record(step=i, state=state)

        rmsf = tracker.rmsf_per_particle()
        self.assertEqual(len(rmsf), 2)
        for val in rmsf:
            self.assertGreaterEqual(val, 0.0)


# ===================================================================
# TestRDFCalculator
# ===================================================================


class TestRDFCalculator(unittest.TestCase):
    """Tests for validation.adaptive_analysis.RDFCalculator."""

    def setUp(self) -> None:
        from validation.adaptive_analysis import RDFCalculator
        self.RDFCalculator = RDFCalculator

    def _make_state_for_rdf(self) -> SimulationState:
        """A simple 4-particle state with a cell for RDF accumulation."""
        return _make_simple_state(
            positions=(
                (0.5, 0.5, 0.5),
                (1.5, 0.5, 0.5),
                (0.5, 1.5, 0.5),
                (1.5, 1.5, 0.5),
            ),
            masses=(1.0, 1.0, 1.0, 1.0),
            cell=SimulationCell(
                box_vectors=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)),
            ),
        )

    def test_rdf_accumulates(self) -> None:
        calc = self.RDFCalculator()
        state = self._make_state_for_rdf()
        for _ in range(5):
            calc.accumulate(state)
        self.assertEqual(calc._n_frames, 5)

    def test_rdf_has_correct_shape(self) -> None:
        calc = self.RDFCalculator(n_bins=50)
        state = self._make_state_for_rdf()
        calc.accumulate(state)
        r_vals, g_r_vals = calc.compute_rdf()
        self.assertEqual(len(r_vals), 50)
        self.assertEqual(len(g_r_vals), 50)


# ===================================================================
# TestUmbrellaSampling
# ===================================================================


class TestUmbrellaSampling(unittest.TestCase):
    """Tests for validation.adaptive_analysis.UmbrellaSampler and UmbrellaSamplingWindow."""

    def setUp(self) -> None:
        from validation.adaptive_analysis import UmbrellaSampler, UmbrellaSamplingWindow
        self.UmbrellaSampler = UmbrellaSampler
        self.UmbrellaSamplingWindow = UmbrellaSamplingWindow

    def test_bias_energy(self) -> None:
        """E = 0.5 * k * (cv - center)^2 = 0.5 * 100 * (1.5 - 1.0)^2 = 12.5."""
        # The actual interface computes bias energy from a SimulationState,
        # so we set up a state where the CV (distance between particles 0 and 1)
        # is 1.5, and the window center is 1.0 with k=100.
        window = self.UmbrellaSamplingWindow(center=1.0, force_constant=100.0)
        sampler = self.UmbrellaSampler(
            windows=(window,),
            cv_particle_a=0,
            cv_particle_b=1,
        )
        # Place particles so distance = 1.5.
        state = _make_simple_state(
            positions=((0.0, 0.0, 0.0), (1.5, 0.0, 0.0)),
            masses=(1.0, 1.0),
        )
        energy = sampler.bias_energy(state)
        self.assertAlmostEqual(energy, 12.5, places=5)

    def test_bias_force_direction(self) -> None:
        """Force should push particles toward the window center."""
        window = self.UmbrellaSamplingWindow(center=1.0, force_constant=100.0)
        sampler = self.UmbrellaSampler(
            windows=(window,),
            cv_particle_a=0,
            cv_particle_b=1,
        )
        # Distance = 2.0 > center = 1.0, so force should push them together.
        state = _make_simple_state(
            positions=((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            masses=(1.0, 1.0),
        )
        forces = sampler.bias_force(state)
        # forces is tuple of (particle_index, force_vector).
        # Particle A (index 0) should get force in +x (toward B which is too far).
        # Particle B (index 1) should get force in -x (toward A).
        fa_idx, fa_vec = forces[0]
        fb_idx, fb_vec = forces[1]
        self.assertEqual(fa_idx, 0)
        self.assertEqual(fb_idx, 1)
        # fa_vec[0] should be positive (pushing A toward B).
        self.assertGreater(fa_vec[0], 0.0)
        # fb_vec[0] should be negative (pushing B toward A).
        self.assertLess(fb_vec[0], 0.0)

    def test_wham_produces_pmf(self) -> None:
        """3 windows with synthetic histogram data should produce a PMF with values."""
        windows = tuple(
            self.UmbrellaSamplingWindow(center=c, force_constant=100.0)
            for c in (1.0, 2.0, 3.0)
        )
        sampler = self.UmbrellaSampler(
            windows=windows,
            n_bins=50,
        )
        # Inject synthetic Gaussian-like histograms centered on each window.
        import math
        lo, hi = sampler._cv_range()
        bin_width = (hi - lo) / 50
        for wi, window in enumerate(windows):
            hist = [0] * 50
            for bi in range(50):
                cv = lo + (bi + 0.5) * bin_width
                hist[bi] = int(100 * math.exp(-0.5 * ((cv - window.center) / 0.3) ** 2))
            sampler._histograms[wi] = hist

        cv_vals, pmf_vals = sampler.compute_pmf_wham(temperature=300.0)
        self.assertEqual(len(cv_vals), 50)
        self.assertEqual(len(pmf_vals), 50)
        # At least some PMF values should be finite and non-negative (shifted to min=0).
        finite_vals = [v for v in pmf_vals if v != float("inf")]
        self.assertGreater(len(finite_vals), 0)
        self.assertAlmostEqual(min(finite_vals), 0.0)


# ===================================================================
# TestBindingEstimator
# ===================================================================


class TestBindingEstimator(unittest.TestCase):
    """Tests for validation.adaptive_analysis.BindingEnergyEstimator."""

    def setUp(self) -> None:
        from validation.adaptive_analysis import BindingEnergyEstimator
        self.BindingEnergyEstimator = BindingEnergyEstimator

    def test_records_distances(self) -> None:
        estimator = self.BindingEnergyEstimator(
            group_a_indices=(0, 1),
            group_b_indices=(2, 3),
        )
        for i in range(10):
            state = _make_simple_state(
                positions=(
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (3.0 + 0.1 * i, 0.0, 0.0),
                    (4.0 + 0.1 * i, 0.0, 0.0),
                ),
                masses=(1.0, 1.0, 1.0, 1.0),
            )
            estimator.record(state)
        self.assertEqual(len(estimator._distance_history), 10)

    def test_estimate_returns_dict(self) -> None:
        estimator = self.BindingEnergyEstimator(
            group_a_indices=(0,),
            group_b_indices=(1,),
        )
        for _ in range(5):
            state = _make_simple_state(
                positions=((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
                masses=(1.0, 1.0),
            )
            estimator.record(state)
        result = estimator.estimate_binding_energy()
        # The actual implementation uses 'mean_interaction_energy' and 'estimated_binding_dG'.
        self.assertIn("mean_interaction_energy", result)
        self.assertIn("estimated_binding_dG", result)


# ===================================================================
# TestAdaptiveAnalysisEngine
# ===================================================================


class TestAdaptiveAnalysisEngine(unittest.TestCase):
    """Tests for validation.adaptive_analysis.AdaptiveAnalysisEngine."""

    def setUp(self) -> None:
        from validation.adaptive_analysis import AdaptiveAnalysisEngine, AnalysisReport
        self.AdaptiveAnalysisEngine = AdaptiveAnalysisEngine
        self.AnalysisReport = AnalysisReport

    def _build_state(self) -> SimulationState:
        return _make_simple_state(
            positions=(
                (0.5, 0.5, 0.5),
                (1.7, 0.5, 0.5),
                (2.9, 0.5, 0.5),
                (2.9, 1.7, 0.5),
            ),
            masses=(1.0, 1.0, 1.0, 1.0),
            cell=SimulationCell(
                box_vectors=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)),
            ),
        )

    def test_auto_configure_sets_up_trackers(self) -> None:
        engine = self.AdaptiveAnalysisEngine()
        state = self._build_state()
        engine.auto_configure(state)
        self.assertIsNotNone(engine.rmsd_tracker)
        self.assertIsNotNone(engine.rdf_calculator)

    def test_collect_increments_counter(self) -> None:
        engine = self.AdaptiveAnalysisEngine(collection_interval=1)
        state = self._build_state()
        engine.auto_configure(state)
        for _ in range(5):
            engine.collect(state, potential_energy=-1.0)
        self.assertEqual(engine._steps_analyzed, 5)

    def test_generate_report_type(self) -> None:
        engine = self.AdaptiveAnalysisEngine(collection_interval=1)
        state = self._build_state()
        engine.auto_configure(state)
        for _ in range(3):
            engine.collect(state, potential_energy=-1.0)
        report = engine.generate_report()
        self.assertIsInstance(report, self.AnalysisReport)

    def test_report_convergence(self) -> None:
        engine = self.AdaptiveAnalysisEngine(collection_interval=1)
        state = self._build_state()
        engine.auto_configure(state)
        for _ in range(10):
            engine.collect(state, potential_energy=-1.0)
        report = engine.generate_report()
        # is_converged should return a bool.
        self.assertIsInstance(report.is_converged(), bool)


# ===================================================================
# TestFullIntegration
# ===================================================================


class TestFullIntegration(unittest.TestCase):
    """End-to-end: build system -> run production loop -> analyse -> report."""

    def test_full_pipeline_runs_and_produces_report(self) -> None:
        from sampling.production_loop import ProductionSimulationLoop, EnergyTracker
        from validation.adaptive_analysis import AdaptiveAnalysisEngine

        # 1. Build the 4-particle system.
        parts = FIXTURE.build()

        # 2. Create the AdaptiveAnalysisEngine and auto-configure it.
        engine = AdaptiveAnalysisEngine(collection_interval=1)
        engine.auto_configure(parts["initial_state"])

        # 3. Wire the analysis engine as a hook into the production loop.
        #    The production loop calls hooks as hook(state, energy_tracker).
        def analysis_hook(state, energy_tracker):
            pe = state.potential_energy if state.potential_energy is not None else 0.0
            engine.collect(state, potential_energy=pe)

        energy_tracker = EnergyTracker()
        loop = ProductionSimulationLoop(
            topology=parts["topology"],
            forcefield=parts["forcefield"],
            integrator=parts["integrator"],
            force_evaluator=parts["force_evaluator"],
            registry=parts["registry"],
            apply_pbc=True,
            energy_tracker=energy_tracker,
            analysis_hooks=[analysis_hook],
            analysis_interval=1,  # call hook every step
        )

        # 4. Run 50 steps.
        result = loop.run(50)
        self.assertEqual(result.steps_completed, 50)

        # 5. Generate the analysis report.
        report = engine.generate_report()

        # 6. Verify: report has RMSD data.
        self.assertIn("mean_rmsd", report.rmsd_summary)
        self.assertIn("rmsf_per_particle", report.rmsd_summary)
        rmsf = report.rmsd_summary["rmsf_per_particle"]
        self.assertEqual(len(rmsf), 4)

        # 7. Verify: report has RDF data.
        self.assertIn("r_values", report.rdf_data)
        self.assertIn("g_r_values", report.rdf_data)
        self.assertGreater(len(report.rdf_data["r_values"]), 0)

        # 8. Verify: energy summary from the production loop.
        self.assertIn("kinetic_energy", result.energy_summary)
        self.assertIn("potential_energy", result.energy_summary)

        # 9. Verify convergence check returns a bool.
        self.assertIsInstance(report.is_converged(), bool)

        # 10. Verify the final state is physically reasonable.
        final = result.final_state
        self.assertEqual(final.step, 50)
        self.assertGreater(final.time, 0.0)
        # All positions should be inside the box after PBC wrapping.
        for pos in final.particles.positions:
            for ax in range(3):
                self.assertGreaterEqual(pos[ax], 0.0 - 1e-10)
                self.assertLess(pos[ax], 5.0 + 1e-10)


# ===================================================================
# Helper: make a minimal SimulationState for unit tests
# ===================================================================


def _make_simple_state(
    *,
    positions: tuple[tuple[float, ...], ...],
    masses: tuple[float, ...],
    velocities: tuple[tuple[float, ...], ...] | None = None,
    cell: SimulationCell | None = None,
    step: int = 0,
) -> SimulationState:
    """Create a minimal SimulationState for testing without a registry."""
    n = len(positions)
    if velocities is None:
        velocities = tuple((0.0, 0.0, 0.0) for _ in range(n))

    return SimulationState(
        units=UnitSystem.md_nano(),
        particles=ParticleState(
            positions=positions,
            masses=masses,
            velocities=velocities,
        ),
        thermodynamics=ThermodynamicState(
            ensemble=EnsembleKind.NVT,
            target_temperature=1.0,
            friction_coefficient=2.0,
        ),
        provenance=StateProvenance(
            simulation_id=SimulationId("sim-test"),
            state_id=StateId(f"state-test-{id(positions)}"),
            parent_state_id=None,
            created_by="unit-test",
            stage="test",
        ),
        cell=cell,
        step=step,
    )


if __name__ == "__main__":
    unittest.main()
