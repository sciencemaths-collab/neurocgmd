"""Tests for the cell-list neighbor finding and accelerated nonbonded force model."""

from __future__ import annotations

import unittest
from math import sqrt

from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.types import BeadId, SimulationId, StateId
from forcefields.base_forcefield import BaseForceField, BondParameter, NonbondedParameter
from physics.neighbor_list import (
    AcceleratedNonbondedForceModel,
    NeighborListBuilder,
    _build_cell_list,
)
from physics.forces.nonbonded_forces import LennardJonesNonbondedForceModel
from topology import Bead, BeadRole, BeadType, Bond, SystemTopology


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Five particles in a roughly linear arrangement along the x-axis:
#   0 -- 1 -- 2           3        4
#  (0,0,0) (1,0,0) (2,0,0) (5,0,0) (5.5,0,0)
#
# Particles 0-1-2 are close together (distance 1.0 apart).
# Particle 3 is far away (distance 3.0 from particle 2).
# Particle 4 is close to particle 3 (distance 0.5).

_POSITIONS = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (5.0, 0.0, 0.0),
    (5.5, 0.0, 0.0),
)
_PARTICLE_COUNT = len(_POSITIONS)


def _make_state(positions=_POSITIONS) -> SimulationState:
    return SimulationState(
        units=UnitSystem.md_nano(),
        particles=ParticleState(
            positions=positions,
            masses=(1.0,) * len(positions),
            velocities=((0.0, 0.0, 0.0),) * len(positions),
        ),
        thermodynamics=ThermodynamicState(),
        provenance=StateProvenance(
            simulation_id=SimulationId("sim-nbl"),
            state_id=StateId("state-nbl"),
            parent_state_id=None,
            created_by="unit-test",
            stage="initialization",
        ),
    )


def _make_topology(positions=_POSITIONS, bonds=(Bond(0, 1),)) -> SystemTopology:
    n = len(positions)
    return SystemTopology(
        system_id="nbl-system",
        bead_types=(
            BeadType(name="bb", role=BeadRole.STRUCTURAL),
        ),
        beads=tuple(
            Bead(bead_id=BeadId(f"b{i}"), particle_index=i, bead_type="bb", label=f"B{i}")
            for i in range(n)
        ),
        bonds=bonds,
    )


def _make_forcefield(cutoff: float = 3.0) -> BaseForceField:
    return BaseForceField(
        name="nbl-ff",
        bond_parameters=(
            BondParameter("bb", "bb", equilibrium_distance=1.0, stiffness=50.0),
        ),
        nonbonded_parameters=(
            NonbondedParameter("bb", "bb", sigma=1.0, epsilon=0.5, cutoff=cutoff),
        ),
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestCellListConstruction(unittest.TestCase):
    """Verify particles are assigned to the correct cells."""

    def test_cell_list_construction(self) -> None:
        cell_size = 2.0
        cell_list = _build_cell_list(_POSITIONS, _PARTICLE_COUNT, cell_size)

        # Every particle must appear in exactly one cell.
        self.assertEqual(len(cell_list.particle_cells), _PARTICLE_COUNT)

        # Particles in the same spatial region share a cell.
        # Particles 0 (0,0,0) and 1 (1,0,0) are within cell_size=2.0 of the
        # origin, so they should map to the same cell.
        self.assertEqual(cell_list.particle_cells[0], cell_list.particle_cells[1])

        # Particle 3 (5,0,0) is far from particle 0 -- different cell.
        self.assertNotEqual(cell_list.particle_cells[0], cell_list.particle_cells[3])

        # Particles 3 (5,0,0) and 4 (5.5,0,0) should share a cell.
        self.assertEqual(cell_list.particle_cells[3], cell_list.particle_cells[4])

        # Reverse mapping: every particle in cell_particles should agree with
        # particle_cells.
        for cell_idx, particles in cell_list.cell_particles.items():
            for pid in particles:
                self.assertEqual(cell_list.particle_cells[pid], cell_idx)


class TestNeighborListPairDiscovery(unittest.TestCase):
    """Verify that the neighbor list finds close pairs and excludes distant ones."""

    def test_neighbor_list_finds_close_pairs(self) -> None:
        # cutoff=1.5, skin=0.0 => effective cutoff = 1.5
        # Pairs within 1.5: (0,1)=1.0, (1,2)=1.0, (3,4)=0.5
        builder = NeighborListBuilder(cutoff=1.5, skin=0.0)
        nbl = builder.build(_POSITIONS, _PARTICLE_COUNT)

        pair_set = set(nbl.pairs)
        self.assertIn((0, 1), pair_set)
        self.assertIn((1, 2), pair_set)
        self.assertIn((3, 4), pair_set)

    def test_neighbor_list_excludes_distant_pairs(self) -> None:
        # cutoff=1.5, skin=0.0 => effective cutoff = 1.5
        # (0,2)=2.0, (0,3)=5.0, (2,3)=3.0 should all be absent.
        builder = NeighborListBuilder(cutoff=1.5, skin=0.0)
        nbl = builder.build(_POSITIONS, _PARTICLE_COUNT)

        pair_set = set(nbl.pairs)
        self.assertNotIn((0, 2), pair_set)
        self.assertNotIn((0, 3), pair_set)
        self.assertNotIn((2, 3), pair_set)
        self.assertNotIn((0, 4), pair_set)


class TestExcludedPairs(unittest.TestCase):
    """Verify that bonded-pair exclusion works."""

    def test_excluded_pairs_honored(self) -> None:
        builder = NeighborListBuilder(cutoff=1.5, skin=0.0)
        excluded = frozenset({(0, 1)})
        nbl = builder.build(_POSITIONS, _PARTICLE_COUNT, excluded_pairs=excluded)

        pair_set = set(nbl.pairs)
        # (0,1) is within cutoff but should be excluded.
        self.assertNotIn((0, 1), pair_set)
        # Other close pairs remain.
        self.assertIn((1, 2), pair_set)
        self.assertIn((3, 4), pair_set)


class TestRebuildDetection(unittest.TestCase):
    """Verify the skin-based rebuild heuristic."""

    def test_needs_rebuild_when_particles_move(self) -> None:
        builder = NeighborListBuilder(cutoff=1.5, skin=0.4)
        # skin/2 = 0.2 => movement > 0.2 triggers rebuild.
        old_positions = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        new_positions = ((0.3, 0.0, 0.0), (1.0, 0.0, 0.0))  # particle 0 moved 0.3
        self.assertTrue(builder.needs_rebuild(new_positions, old_positions))

    def test_no_rebuild_for_small_motion(self) -> None:
        builder = NeighborListBuilder(cutoff=1.5, skin=0.4)
        # skin/2 = 0.2 => movement of 0.1 does NOT trigger rebuild.
        old_positions = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        new_positions = ((0.1, 0.0, 0.0), (1.0, 0.0, 0.0))  # particle 0 moved 0.1
        self.assertFalse(builder.needs_rebuild(new_positions, old_positions))


class TestNeighborListSymmetry(unittest.TestCase):
    """Verify canonical ordering -- no duplicate (j,i) for existing (i,j)."""

    def test_neighbor_list_symmetry(self) -> None:
        builder = NeighborListBuilder(cutoff=3.0, skin=0.0)
        nbl = builder.build(_POSITIONS, _PARTICLE_COUNT)

        seen: set[tuple[int, int]] = set()
        for a, b in nbl.pairs:
            # Canonical order: a < b.
            self.assertLess(a, b, f"Pair ({a},{b}) not in canonical order")
            # No duplicate.
            self.assertNotIn((a, b), seen, f"Duplicate pair ({a},{b})")
            seen.add((a, b))


class TestAcceleratedNonbondedMatchesNaive(unittest.TestCase):
    """Compare AcceleratedNonbondedForceModel against LennardJonesNonbondedForceModel."""

    def test_accelerated_nonbonded_matches_naive(self) -> None:
        cutoff = 3.0
        state = _make_state()
        topology = _make_topology()
        forcefield = _make_forcefield(cutoff=cutoff)

        # Naive O(N^2) model.
        naive_model = LennardJonesNonbondedForceModel(exclude_bonded_pairs=True)
        naive_report = naive_model.evaluate(state, topology, forcefield)

        # Accelerated cell-list model.  Use skin=0.0 so the effective cutoff
        # matches the forcefield cutoff exactly and produces identical pairs.
        builder = NeighborListBuilder(cutoff=cutoff, skin=0.0)
        accel_model = AcceleratedNonbondedForceModel(
            builder, exclude_bonded_pairs=True
        )
        accel_report = accel_model.evaluate(state, topology, forcefield)

        # Force vectors must match to high precision.
        self.assertEqual(len(naive_report.forces), len(accel_report.forces))
        for i in range(len(naive_report.forces)):
            for axis in range(3):
                self.assertAlmostEqual(
                    naive_report.forces[i][axis],
                    accel_report.forces[i][axis],
                    places=10,
                    msg=f"Force mismatch at particle {i}, axis {axis}",
                )

        # Same set of evaluated pairs (order may differ).
        self.assertEqual(
            set(naive_report.evaluated_pairs),
            set(accel_report.evaluated_pairs),
        )


if __name__ == "__main__":
    unittest.main()
