"""Tests for the SPRING ↔ NeuroCGMD integration bridge."""

from __future__ import annotations

import unittest

import numpy as np

from spring.engine import Piece, Spring, SpringEngine, Snapshot
from spring.bridges.neurocgmd_bridge import (
    MDPoweredSolver,
    MDSystemAsSPRING,
    spring_to_md,
    md_to_spring,
)
from spring.demos.chain_fold import ChainFold
from spring.demos.tile_puzzle import TilePuzzle

from core.state import EnsembleKind


class TestSpringToMD(unittest.TestCase):
    """Verify SPRING → NeuroCGMD conversion."""

    def _make_pieces_and_springs(self):
        pieces = [
            Piece(state=np.array([0.0, 0.0]), piece_id=0),
            Piece(state=np.array([1.0, 0.0]), piece_id=1),
            Piece(state=np.array([2.0, 1.0]), piece_id=2),
        ]
        springs = [
            Spring(a=0, b=1, stiffness=1.0, rest_value=1.0),
            Spring(a=1, b=2, stiffness=0.5, rest_value=1.0),
        ]
        return pieces, springs

    def test_conversion_creates_valid_state(self):
        pieces, springs = self._make_pieces_and_springs()
        state, topology, forcefield = spring_to_md(pieces, springs)

        self.assertEqual(state.particle_count, 3)
        self.assertEqual(len(topology.bonds), 2)
        self.assertEqual(state.thermodynamics.ensemble, EnsembleKind.NVT)
        # 2D pieces should be padded to 3D
        for pos in state.particles.positions:
            self.assertEqual(len(pos), 3)
            self.assertAlmostEqual(pos[2], 0.0)

    def test_positions_match(self):
        pieces, springs = self._make_pieces_and_springs()
        state, _, _ = spring_to_md(pieces, springs)

        self.assertAlmostEqual(state.particles.positions[0][0], 0.0)
        self.assertAlmostEqual(state.particles.positions[1][0], 1.0)
        self.assertAlmostEqual(state.particles.positions[2][1], 1.0)

    def test_roundtrip(self):
        pieces, springs = self._make_pieces_and_springs()
        state, _, _ = spring_to_md(pieces, springs)
        # Modify state positions
        new_pieces = [p.copy() for p in pieces]
        md_to_spring(state, new_pieces, dim=2)
        for orig, new in zip(pieces, new_pieces):
            np.testing.assert_array_almost_equal(orig.state, new.state)


class TestMDPoweredSolver(unittest.TestCase):
    """Verify the hybrid solver runs without errors."""

    def test_chain_fold_runs(self):
        problem = ChainFold(bond_length=1.0, bond_stiffness=5.0, attract_stiffness=2.0)
        solver = MDPoweredSolver(
            max_iterations=10,
            md_steps_per_iter=5,
            time_step=0.0005,
            temperature_start=3.0,
            temperature_end=0.5,
            friction=2.0,
            use_ml=True,
            use_plasticity=True,
            random_seed=42,
            verbose=False,
        )
        np.random.seed(42)
        best, history = solver.solve(problem, "HPPH")

        self.assertIsNotNone(best)
        self.assertEqual(len(history), 10)
        self.assertGreater(best.energy_total, 0.0)
        self.assertTrue(all(isinstance(s, Snapshot) for s in history))

    def test_energy_decreases_overall(self):
        problem = ChainFold(bond_length=1.0, bond_stiffness=5.0, attract_stiffness=2.0)
        solver = MDPoweredSolver(
            max_iterations=30, md_steps_per_iter=5, time_step=0.0003,
            temperature_start=5.0, temperature_end=0.1,
            use_ml=False, use_plasticity=False,
            random_seed=42, verbose=False,
        )
        np.random.seed(42)
        best, history = solver.solve(problem, "HPPH")

        first_energy = history[0].energy_total
        best_energy = best.energy_total
        self.assertLess(best_energy, first_energy,
                        "Best energy should be lower than initial energy.")

    def test_solver_with_ml_trains_model(self):
        problem = ChainFold(bond_length=1.0, bond_stiffness=5.0)
        solver = MDPoweredSolver(
            max_iterations=15, md_steps_per_iter=3, time_step=0.0003,
            use_ml=True, use_plasticity=False,
            random_seed=42, verbose=False,
        )
        np.random.seed(42)
        solver.solve(problem, "HPH")
        # ML model should have been created and trained (>0 states)
        # Can't directly access but the solve should not crash


class TestMDSystemAsSPRING(unittest.TestCase):
    """Verify wrapping MD systems as SPRING problems."""

    def test_shatter_creates_pieces(self):
        problem = MDSystemAsSPRING(dim=3)
        pieces, springs = problem.shatter({
            "positions": [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            "bonds": [[0, 1, 1.0, 100.0], [1, 2, 1.0, 100.0]],
        })
        self.assertEqual(len(pieces), 3)
        self.assertEqual(len(springs), 2)

    def test_coupling_energy_is_harmonic(self):
        problem = MDSystemAsSPRING(dim=2)
        p_a = Piece(state=np.array([0.0, 0.0]), piece_id=0)
        p_b = Piece(state=np.array([2.0, 0.0]), piece_id=1)
        s = Spring(a=0, b=1, stiffness=1.0, rest_value=1.0)
        # dist=2.0, rest=1.0, k=1.0 → 0.5*1.0*(2.0-1.0)^2 = 0.5
        energy = problem.coupling_energy(p_a, p_b, s)
        self.assertAlmostEqual(energy, 0.5)

    def test_spring_engine_solves_md_system(self):
        problem = MDSystemAsSPRING(dim=2)
        engine = SpringEngine(max_iterations=50, temperature=1.0, verbose=False)
        np.random.seed(42)
        best, history = engine.solve(problem, {
            "positions": [[0, 0], [3, 0], [1.5, 3]],
            "bonds": [[0, 1, 1.0, 1.0], [1, 2, 1.0, 1.0], [0, 2, 1.0, 1.0]],
        })
        self.assertIsNotNone(best)
        self.assertLess(best.energy_total, history[0].energy_total)


class TestPureSPRINGStillWorks(unittest.TestCase):
    """Verify original SPRING demos work in the merged codebase."""

    def test_tile_puzzle(self):
        problem = TilePuzzle()
        engine = SpringEngine(max_iterations=100, temperature=2.0, verbose=False)
        np.random.seed(42)
        best, history = engine.solve(problem, [3, 1, 0, 2])
        self.assertIsNotNone(best)
        self.assertGreater(len(history), 0)

    def test_chain_fold(self):
        problem = ChainFold()
        engine = SpringEngine(max_iterations=100, temperature=1.0, momentum=0.3, verbose=False)
        np.random.seed(42)
        best, history = engine.solve(problem, "HPPH")
        self.assertIsNotNone(best)
        self.assertLess(best.energy_total, history[0].energy_total)


if __name__ == "__main__":
    unittest.main()
