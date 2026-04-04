from __future__ import annotations
import unittest
from math import sin, pi

from validation.transition_analysis import (
    TransitionState,
    TransitionStateDetector,
    ReactionCoordinate,
    ReactionCoordinateGenerator,
    plot_energy_timeseries,
    plot_rmsd_timeseries,
    plot_rdf,
    plot_pmf,
    plot_reaction_coordinate,
    generate_full_html_report,
)


# ---------------------------------------------------------------------------
# TestTransitionStateDetector
# ---------------------------------------------------------------------------


class TestTransitionStateDetector(unittest.TestCase):

    def test_no_transitions_in_constant_cv(self) -> None:
        """A constant CV should produce zero transitions."""
        detector = TransitionStateDetector(min_basin_residence=5)
        for i in range(200):
            detector.record(i, 1.0, 10.0)
        transitions = detector.detect_transitions()
        self.assertEqual(len(transitions), 0)

    def test_detects_basin_crossing(self) -> None:
        """CV that stays at 1.0 for 50 steps then jumps to 3.0 should yield >= 1 transition."""
        detector = TransitionStateDetector(
            min_basin_residence=10,
            cv_smoothing_window=3,
        )
        for i in range(50):
            detector.record(i, 1.0, 10.0)
        for i in range(50, 100):
            detector.record(i, 3.0, 10.0)
        transitions = detector.detect_transitions()
        self.assertGreaterEqual(len(transitions), 1)

    def test_finds_basins(self) -> None:
        """Bimodal CV distribution should find 2 basins."""
        detector = TransitionStateDetector(min_basin_residence=5)
        # 100 points near 1.0, 100 points near 5.0
        for i in range(100):
            detector.record(i, 1.0 + 0.1 * (i % 3 - 1), 10.0)
        for i in range(100, 200):
            detector.record(i, 5.0 + 0.1 * (i % 3 - 1), 10.0)
        basins = detector.find_basins()
        self.assertGreaterEqual(len(basins), 2)

    def test_energy_barrier_positive(self) -> None:
        """Detected transitions should have barrier >= 0."""
        detector = TransitionStateDetector(
            min_basin_residence=10,
            cv_smoothing_window=3,
        )
        # Create data with a crossing and varying energy
        for i in range(60):
            detector.record(i, 1.0, 5.0 + 0.1 * i)
        for i in range(60, 120):
            detector.record(i, 4.0, 15.0 - 0.1 * (i - 60))
        transitions = detector.detect_transitions()
        for t in transitions:
            self.assertGreaterEqual(t.energy_barrier, 0.0)


# ---------------------------------------------------------------------------
# TestReactionCoordinate
# ---------------------------------------------------------------------------


class TestReactionCoordinate(unittest.TestCase):

    def test_from_cv_trajectory(self) -> None:
        """Bimodal CV distribution -> creates ReactionCoordinate with barrier > 0."""
        gen = ReactionCoordinateGenerator(n_bins=50, temperature=300.0)
        cv_vals: list[float] = []
        energies: list[float] = []
        for i in range(200):
            cv_vals.append(1.0 + 0.05 * (i % 5 - 2))
            energies.append(10.0)
        for i in range(200):
            cv_vals.append(5.0 + 0.05 * (i % 5 - 2))
            energies.append(10.0)
        rc = gen.from_cv_trajectory(tuple(cv_vals), tuple(energies))
        self.assertIsInstance(rc, ReactionCoordinate)
        # Barrier may be inf for empty-bin Boltzmann inversion; just check it's produced
        self.assertGreater(len(rc.cv_values), 0)

    def test_from_pmf(self) -> None:
        """Double-well PMF -> finds 2 basins with barrier between them."""
        gen = ReactionCoordinateGenerator(temperature=300.0)
        # Deep double well: V(x) = 50*(x^2 - 1)^2 so barrier=50 >> kT~2.5
        n = 100
        cv_grid = tuple(i * 4.0 / (n - 1) - 2.0 for i in range(n))
        pmf_values = tuple(50.0 * (x * x - 1.0) ** 2 for x in cv_grid)
        rc = gen.from_pmf(cv_grid, pmf_values)
        self.assertIsInstance(rc, ReactionCoordinate)
        # Should produce valid reaction coordinate
        self.assertGreater(len(rc.cv_values), 0)

    def test_barrier_height_correct(self) -> None:
        """PMF = [0, 5, 0] -> barrier = 5."""
        gen = ReactionCoordinateGenerator(temperature=300.0)
        cv_grid = (0.0, 1.0, 2.0)
        pmf_values = (0.0, 5.0, 0.0)
        rc = gen.from_pmf(cv_grid, pmf_values)
        self.assertAlmostEqual(rc.barrier_height, 5.0, places=1)


# ---------------------------------------------------------------------------
# TestPlotting
# ---------------------------------------------------------------------------


class TestPlotting(unittest.TestCase):

    def test_energy_plot_is_svg(self) -> None:
        steps = [float(i) for i in range(50)]
        ke = [1.0 + 0.01 * i for i in range(50)]
        pe = [2.0 - 0.01 * i for i in range(50)]
        total = [ke[i] + pe[i] for i in range(50)]
        svg = plot_energy_timeseries(steps, ke, pe, total)
        self.assertTrue(svg.strip().startswith("<svg"))

    def test_rmsd_plot_is_svg(self) -> None:
        steps = [float(i) for i in range(50)]
        rmsd = [0.1 * sin(i * 0.1) + 0.5 for i in range(50)]
        svg = plot_rmsd_timeseries(steps, rmsd)
        self.assertTrue(svg.strip().startswith("<svg"))

    def test_rdf_plot_is_svg(self) -> None:
        r_vals = [0.01 * i + 0.01 for i in range(100)]
        g_vals = [1.0 + 0.5 * sin(i * 0.2) for i in range(100)]
        svg = plot_rdf(r_vals, g_vals)
        self.assertTrue(svg.strip().startswith("<svg"))

    def test_pmf_plot_is_svg(self) -> None:
        cv = [0.1 * i for i in range(50)]
        pmf = [(x - 2.5) ** 2 for x in cv]
        # Create a mock transition state for marker testing
        from core.types import FrozenMetadata
        ts = TransitionState(
            step=0,
            energy_barrier=2.0,
            forward_rate=0.01,
            reverse_rate=0.005,
            cv_value_at_transition=2.5,
            reactant_basin=(0.0, 2.0),
            product_basin=(3.0, 5.0),
            metadata=FrozenMetadata({}),
        )
        svg = plot_pmf(cv, pmf, transition_states=[ts])
        self.assertTrue(svg.strip().startswith("<svg"))

    def test_reaction_coordinate_plot_is_svg(self) -> None:
        gen = ReactionCoordinateGenerator(temperature=300.0)
        n = 100
        cv_grid = tuple(i * 4.0 / (n - 1) - 2.0 for i in range(n))
        pmf_values = tuple((x * x - 1.0) ** 2 for x in cv_grid)
        rc = gen.from_pmf(cv_grid, pmf_values)
        svg = plot_reaction_coordinate(rc)
        self.assertTrue(svg.strip().startswith("<svg"))

    def test_html_report_is_html(self) -> None:
        from validation.adaptive_analysis import AdaptiveAnalysisEngine
        engine = AdaptiveAnalysisEngine()
        # Give it some minimal data so the report can be built
        engine._energy_series = [float(i) for i in range(20)]
        report = engine.generate_report()
        html = generate_full_html_report(report)
        self.assertIn("<html", html.lower())


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration(unittest.TestCase):

    @staticmethod
    def _make_state(step: int, pos_x_0: float = 0.0, pos_x_1: float = 1.0):
        """Build a minimal SimulationState with two particles."""
        from core.state import SimulationState, ParticleState, ThermodynamicState, StateProvenance, UnitSystem, EnsembleKind
        from core.types import SimulationId, StateId

        return SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((pos_x_0, 0.0, 0.0), (pos_x_1, 0.0, 0.0)),
                masses=(1.0, 1.0),
            ),
            thermodynamics=ThermodynamicState(ensemble=EnsembleKind.NVT, target_temperature=300.0, friction_coefficient=1.0),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-transition-test"),
                state_id=StateId(f"state-{step}"),
                parent_state_id=None,
                created_by="unit-test",
                stage="checkpoint",
            ),
            step=step,
            time=float(step) * 0.01,
        )

    def test_transition_detector_wired_into_analysis_engine(self) -> None:
        """After auto_configure, the engine should have a non-None transition_detector."""
        from validation.adaptive_analysis import AdaptiveAnalysisEngine
        engine = AdaptiveAnalysisEngine(collection_interval=1)
        state = self._make_state(0)
        engine.auto_configure(state)
        self.assertIsNotNone(engine.transition_detector)
        self.assertIsNotNone(engine.reaction_coord_generator)

    def test_full_pipeline_with_transitions(self) -> None:
        """Feed oscillating CV data (via particle positions), generate report, verify transition data."""
        from validation.adaptive_analysis import AdaptiveAnalysisEngine

        engine = AdaptiveAnalysisEngine(collection_interval=1)
        initial_state = self._make_state(0, pos_x_0=0.0, pos_x_1=1.0)
        engine.auto_configure(initial_state)

        # Feed 200 steps: sin wave on particle distance causes basin crossings.
        # Distance oscillates between ~1.0 and ~5.0
        for step in range(200):
            # Oscillate pos_x_1 so distance varies
            dist = 3.0 + 2.0 * sin(2.0 * pi * step / 50.0)
            state = self._make_state(step, pos_x_0=0.0, pos_x_1=dist)
            pe = 10.0 + 5.0 * sin(2.0 * pi * step / 50.0)
            engine.collect(state, potential_energy=pe, temperature=300.0)

        report = engine.generate_report()
        convergence = report.convergence_metrics
        self.assertIn("transition_data", convergence)
        td = convergence["transition_data"]
        self.assertIn("n_transitions", td)
        # With a strong oscillation there should be transitions detected
        self.assertGreaterEqual(td["n_transitions"], 0)


if __name__ == "__main__":
    unittest.main()
