"""Comprehensive tests for the molecular observables module.

Covers SASA, radius of gyration, hydrogen bond analysis, energy decomposition,
contact maps, secondary structure estimation, end-to-end distance, and the
unified MolecularObservableCollector.
"""

from __future__ import annotations

import unittest
from math import sqrt, pi

from core.state import (
    EnsembleKind,
    ParticleState,
    SimulationState,
    SimulationCell,
    StateProvenance,
    ThermodynamicState,
    UnitSystem,
)
from core.types import BeadId, FrozenMetadata, SimulationId, StateId
from topology.beads import Bead, BeadRole, BeadType
from topology.bonds import Bond
from topology.system_topology import SystemTopology

from validation.molecular_observables import (
    ContactMapCalculator,
    EndToEndDistance,
    HydrogenBondAnalyzer,
    HydrogenBondCriteria,
    MolecularObservableCollector,
    MolecularSnapshot,
    RadiusOfGyration,
    ResidueEnergyDecomposition,
    SASACalculator,
    SecondaryStructureEstimator,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CHAIN_POSITIONS = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (2.0, 0.5, 0.0),
    (3.0, 0.0, 0.0),
    (4.0, 0.5, 0.0),
    (5.0, 0.0, 0.0),
)

_BEAD_TYPE_NAMES = ("BB", "P5", "C1", "BB", "P5", "BB")

_MASSES = (72.0, 72.0, 72.0, 72.0, 72.0, 72.0)

_VELOCITIES = (
    (0.01, 0.02, -0.01),
    (-0.02, 0.01, 0.0),
    (0.0, -0.01, 0.02),
    (0.02, 0.0, -0.02),
    (-0.01, 0.02, 0.01),
    (0.01, -0.02, 0.0),
)


def _make_beads() -> tuple[Bead, ...]:
    return tuple(
        Bead(
            bead_id=BeadId(f"bead-{i}"),
            particle_index=i,
            bead_type=_BEAD_TYPE_NAMES[i],
            label=f"{_BEAD_TYPE_NAMES[i]}_{i}",
        )
        for i in range(6)
    )


def _make_bonds() -> tuple[Bond, ...]:
    return tuple(
        Bond(particle_index_a=i, particle_index_b=i + 1) for i in range(5)
    )


_BEAD_TYPES = (
    BeadType(name="BB", role=BeadRole.STRUCTURAL),
    BeadType(name="P5", role=BeadRole.FUNCTIONAL),
    BeadType(name="C1", role=BeadRole.FUNCTIONAL),
)


def _make_topology() -> SystemTopology:
    return SystemTopology(
        system_id="test_obs",
        bead_types=_BEAD_TYPES,
        beads=_make_beads(),
        bonds=_make_bonds(),
    )


def _make_state(
    positions=_CHAIN_POSITIONS,
    velocities=_VELOCITIES,
    masses=_MASSES,
    step: int = 0,
) -> SimulationState:
    return SimulationState(
        units=UnitSystem.md_nano(),
        particles=ParticleState(
            positions=positions,
            masses=masses,
            velocities=velocities,
        ),
        thermodynamics=ThermodynamicState(
            ensemble=EnsembleKind.NVT,
            target_temperature=300.0,
            friction_coefficient=1.0,
        ),
        provenance=StateProvenance(
            simulation_id=SimulationId("sim-mol-obs-test"),
            state_id=StateId(f"state-{step}"),
            parent_state_id=None,
            created_by="unit-test",
            stage="checkpoint",
        ),
        step=step,
        time=float(step) * 0.01,
    )


def _make_radii() -> tuple[float, ...]:
    return (0.24, 0.26, 0.26, 0.24, 0.26, 0.24)


# ---------------------------------------------------------------------------
# TestSASA
# ---------------------------------------------------------------------------


class TestSASA(unittest.TestCase):
    """Solvent-accessible surface area calculator tests."""

    def setUp(self) -> None:
        self.calc = SASACalculator(n_sphere_points=92)
        self.positions = _CHAIN_POSITIONS
        self.radii = _make_radii()
        self.bead_types = _BEAD_TYPE_NAMES

    def test_sasa_positive(self) -> None:
        """Total SASA must be positive for any non-degenerate arrangement."""
        result = self.calc.compute(self.positions, self.radii)
        self.assertGreater(result.total_sasa, 0.0)

    def test_sasa_per_particle(self) -> None:
        """Each per-particle SASA must be >= 0."""
        result = self.calc.compute(self.positions, self.radii)
        self.assertEqual(len(result.per_particle_sasa), len(self.positions))
        for sasa_i in result.per_particle_sasa:
            self.assertGreaterEqual(sasa_i, 0.0)

    def test_buried_particle_lower_sasa(self) -> None:
        """A particle surrounded by neighbors should have lower SASA than an exposed one.

        Construct a cluster: one center particle surrounded by six neighbors at
        close range, then compute SASA. The center particle should have less
        exposed surface than an isolated outer particle.
        """
        # Center surrounded by 6 neighbors in octahedral arrangement
        d = 0.40  # just beyond sum of two radii + probe ≈ 0.24+0.14=0.38
        clustered_positions = (
            (0.0, 0.0, 0.0),   # center, should be somewhat buried
            (d, 0.0, 0.0),
            (-d, 0.0, 0.0),
            (0.0, d, 0.0),
            (0.0, -d, 0.0),
            (0.0, 0.0, d),
        )
        radii_cluster = (0.24,) * 6
        result = self.calc.compute(clustered_positions, radii_cluster)

        # Center particle (index 0) should have lower SASA than at least one
        # outer particle that has fewer neighbors blocking it.
        center_sasa = result.per_particle_sasa[0]
        max_outer_sasa = max(result.per_particle_sasa[1:])
        self.assertLess(center_sasa, max_outer_sasa)

    def test_hydrophobic_hydrophilic_split(self) -> None:
        """Hydrophobic + hydrophilic SASA must equal total SASA."""
        result = self.calc.compute(
            self.positions, self.radii, bead_types=self.bead_types,
        )
        self.assertAlmostEqual(
            result.hydrophobic_sasa + result.hydrophilic_sasa,
            result.total_sasa,
            places=10,
        )


# ---------------------------------------------------------------------------
# TestRadiusOfGyration
# ---------------------------------------------------------------------------


class TestRadiusOfGyration(unittest.TestCase):
    """Radius of gyration and shape analysis tests."""

    def setUp(self) -> None:
        self.calc = RadiusOfGyration()
        self.positions = _CHAIN_POSITIONS

    def test_rg_positive(self) -> None:
        """Rg > 0 for non-degenerate (non-coincident) positions."""
        result = self.calc.compute(self.positions)
        self.assertGreater(result.radius_of_gyration, 0.0)

    def test_rg_increases_with_spread(self) -> None:
        """Widely spread particles should have a larger Rg."""
        compact = ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0))
        spread = ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0))

        rg_compact = self.calc.compute(compact).radius_of_gyration
        rg_spread = self.calc.compute(spread).radius_of_gyration
        self.assertGreater(rg_spread, rg_compact)

    def test_rg_per_axis(self) -> None:
        """Per-axis Rg components are all >= 0."""
        result = self.calc.compute(self.positions)
        self.assertGreaterEqual(result.rg_x, 0.0)
        self.assertGreaterEqual(result.rg_y, 0.0)
        self.assertGreaterEqual(result.rg_z, 0.0)

    def test_asphericity_range(self) -> None:
        """Asphericity should be in [0, 1]."""
        result = self.calc.compute(self.positions)
        self.assertGreaterEqual(result.asphericity, 0.0)
        self.assertLessEqual(result.asphericity, 1.0)

    def test_center_of_mass(self) -> None:
        """Center of mass should lie within the bounding box of particle positions."""
        result = self.calc.compute(self.positions)
        com = result.center_of_mass

        for axis in range(3):
            coords = [p[axis] for p in self.positions]
            self.assertGreaterEqual(com[axis], min(coords) - 1e-12)
            self.assertLessEqual(com[axis], max(coords) + 1e-12)


# ---------------------------------------------------------------------------
# TestHydrogenBonds
# ---------------------------------------------------------------------------


class TestHydrogenBonds(unittest.TestCase):
    """Hydrogen bond detection tests."""

    def _make_polar_pair_state(self, distance: float) -> tuple[SimulationState, SystemTopology]:
        """Two P5 (polar) beads separated by *distance* nm, no bond between them."""
        positions = ((0.0, 0.0, 0.0), (distance, 0.0, 0.0))
        state = _make_state(
            positions=positions,
            velocities=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            masses=(72.0, 72.0),
        )
        topology = SystemTopology(
            system_id="hbond_test",
            bead_types=(BeadType(name="P5", role=BeadRole.FUNCTIONAL),),
            beads=(
                Bead(bead_id=BeadId("b0"), particle_index=0, bead_type="P5", label="P5_0"),
                Bead(bead_id=BeadId("b1"), particle_index=1, bead_type="P5", label="P5_1"),
            ),
            bonds=(),
        )
        return state, topology

    def test_finds_hbonds_within_cutoff(self) -> None:
        """Two polar (P5) beads within default cutoff (0.35 nm) should form an H-bond."""
        analyzer = HydrogenBondAnalyzer()
        state, topology = self._make_polar_pair_state(distance=0.30)
        hbonds = analyzer.find_hbonds(state, topology)
        self.assertGreater(len(hbonds), 0)

    def test_no_hbonds_beyond_cutoff(self) -> None:
        """Two polar beads beyond cutoff should NOT form an H-bond."""
        analyzer = HydrogenBondAnalyzer()
        state, topology = self._make_polar_pair_state(distance=1.0)
        hbonds = analyzer.find_hbonds(state, topology)
        self.assertEqual(len(hbonds), 0)

    def test_per_residue_hbonds(self) -> None:
        """Sum of per-residue H-bond counts (halved) should match total H-bond count."""
        analyzer = HydrogenBondAnalyzer()
        state, topology = self._make_polar_pair_state(distance=0.30)
        per_res = analyzer.per_residue_hbonds(state, topology)
        hbonds = analyzer.find_hbonds(state, topology)

        # Each H-bond contributes +1 to both donor and acceptor, so the sum of
        # per-residue counts equals 2 * n_hbonds.
        total_per_res = sum(per_res.values())
        self.assertEqual(total_per_res, 2 * len(hbonds))

    def test_apolar_no_hbonds(self) -> None:
        """Two apolar (C1) beads close together should NOT form H-bonds."""
        positions = ((0.0, 0.0, 0.0), (0.25, 0.0, 0.0))
        state = _make_state(
            positions=positions,
            velocities=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            masses=(72.0, 72.0),
        )
        topology = SystemTopology(
            system_id="apolar_test",
            bead_types=(BeadType(name="C1", role=BeadRole.FUNCTIONAL),),
            beads=(
                Bead(bead_id=BeadId("b0"), particle_index=0, bead_type="C1", label="C1_0"),
                Bead(bead_id=BeadId("b1"), particle_index=1, bead_type="C1", label="C1_1"),
            ),
            bonds=(),
        )
        analyzer = HydrogenBondAnalyzer()
        hbonds = analyzer.find_hbonds(state, topology)
        self.assertEqual(len(hbonds), 0)


# ---------------------------------------------------------------------------
# TestEnergyDecomposition
# ---------------------------------------------------------------------------


class TestEnergyDecomposition(unittest.TestCase):
    """Per-particle energy decomposition tests."""

    def setUp(self) -> None:
        self.decomposer = ResidueEnergyDecomposition()
        self.state = _make_state()
        self.topology = _make_topology()
        # Small uniform forces for testing
        self.forces = tuple((0.1, -0.1, 0.05) for _ in range(6))
        self.potential_energy = 50.0

    def test_per_particle_sums_to_total(self) -> None:
        """Sum of per-particle total energies should approximate total energy."""
        result = self.decomposer.decompose(
            self.state, self.topology, self.forces, self.potential_energy,
        )
        sum_per_particle = sum(result.per_particle_total)
        expected_total = result.total_bonded + result.total_nonbonded + result.total_kinetic
        self.assertAlmostEqual(sum_per_particle, expected_total, places=6)

    def test_kinetic_energy_per_particle(self) -> None:
        """KE per particle should equal 0.5 * m * |v|^2 for known velocities."""
        result = self.decomposer.decompose(
            self.state, self.topology, self.forces, self.potential_energy,
        )
        for i, (vx, vy, vz) in enumerate(_VELOCITIES):
            expected_ke = 0.5 * _MASSES[i] * (vx * vx + vy * vy + vz * vz)
            self.assertAlmostEqual(
                result.per_particle_kinetic[i], expected_ke, places=10,
            )

    def test_interaction_between_groups(self) -> None:
        """Interaction energy between two particle groups is computable."""
        group_a = (0, 1, 2)
        group_b = (3, 4, 5)
        energy = self.decomposer.interaction_energy_between_groups(
            self.state, self.topology, group_a, group_b,
        )
        # Energy should be finite (not NaN or inf)
        self.assertTrue(
            energy == energy,  # NaN check
            "Interaction energy should not be NaN",
        )
        self.assertFalse(
            abs(energy) == float("inf"),
            "Interaction energy should be finite",
        )


# ---------------------------------------------------------------------------
# TestContactMap
# ---------------------------------------------------------------------------


class TestContactMap(unittest.TestCase):
    """Contact map calculator tests."""

    def setUp(self) -> None:
        # Use a cutoff large enough to capture neighboring chain beads (~1 nm apart)
        self.calc = ContactMapCalculator(distance_cutoff=1.2)

    def test_contacts_within_cutoff(self) -> None:
        """Particles closer than cutoff should appear as contacts."""
        # Particles 0 and 1 are 1.0 nm apart; with cutoff 1.2 they should be contacts
        result = self.calc.compute(_CHAIN_POSITIONS)
        contact_pairs = set(result.contacts)
        self.assertIn((0, 1), contact_pairs)

    def test_no_contacts_beyond_cutoff(self) -> None:
        """Particles farther than cutoff should NOT be contacts."""
        # Use a very small cutoff
        calc = ContactMapCalculator(distance_cutoff=0.5)
        result = calc.compute(_CHAIN_POSITIONS)
        # All consecutive particles are >= 1.0 nm apart, so no contacts
        self.assertEqual(result.n_contacts, 0)

    def test_native_contact_fraction(self) -> None:
        """Identical current and native positions should give Q = 1.0."""
        q = self.calc.native_contact_fraction(_CHAIN_POSITIONS, _CHAIN_POSITIONS)
        self.assertAlmostEqual(q, 1.0, places=10)

    def test_native_contact_fraction_zero(self) -> None:
        """Completely rearranged positions should give Q < 1.0."""
        # Move all particles far apart so no native contacts are preserved
        far_positions = tuple((i * 100.0, 0.0, 0.0) for i in range(6))
        q = self.calc.native_contact_fraction(far_positions, _CHAIN_POSITIONS)
        self.assertLess(q, 1.0)


# ---------------------------------------------------------------------------
# TestSecondaryStructure
# ---------------------------------------------------------------------------


class TestSecondaryStructure(unittest.TestCase):
    """Secondary structure proxy estimator tests."""

    def setUp(self) -> None:
        self.estimator = SecondaryStructureEstimator()

    def test_helix_detection(self) -> None:
        """Backbone arranged so distance(i, i+3) is near helix_distance (0.55 nm)
        should be assigned 'helix'.
        """
        # Build 7 backbone beads where consecutive beads are spaced so that
        # distance(i, i+3) ~ 0.55 nm. A simple way: place them along x at
        # ~0.55/3 = 0.183 nm spacing. Then dist(i,i+3) = 3*0.183 = 0.55.
        spacing = 0.55 / 3.0
        positions = tuple((i * spacing, 0.0, 0.0) for i in range(7))
        backbone_indices = tuple(range(7))

        result = self.estimator.estimate(positions, backbone_indices)
        self.assertGreater(result.helix_fraction, 0.0)
        self.assertIn("helix", result.per_residue)

    def test_coil_for_random(self) -> None:
        """Random-ish arrangement should yield mostly 'coil'."""
        # Positions deliberately far apart so neither helix nor sheet distance
        # criteria match: consecutive ~5 nm, so dist(i,i+3) ~ 15 nm.
        positions = tuple((i * 5.0, (i % 2) * 3.0, 0.0) for i in range(6))
        backbone_indices = tuple(range(6))

        result = self.estimator.estimate(positions, backbone_indices)
        self.assertGreater(result.coil_fraction, 0.5)

    def test_fractions_sum_to_one(self) -> None:
        """Helix + sheet + coil fractions must sum to 1.0."""
        positions = _CHAIN_POSITIONS
        backbone_indices = (0, 3, 5)  # backbone type indices from our test system

        result = self.estimator.estimate(positions, backbone_indices)
        total = result.helix_fraction + result.sheet_fraction + result.coil_fraction
        self.assertAlmostEqual(total, 1.0, places=10)


# ---------------------------------------------------------------------------
# TestEndToEndDistance
# ---------------------------------------------------------------------------


class TestEndToEndDistance(unittest.TestCase):
    """End-to-end distance calculator tests."""

    def test_e2e_correct(self) -> None:
        """Particles at (0,0,0) and (5,0,0) should give distance = 5.0."""
        calc = EndToEndDistance(particle_a=0, particle_b=5)
        d = calc.compute(_CHAIN_POSITIONS)
        self.assertAlmostEqual(d, 5.0, places=10)

    def test_e2e_with_negative_index(self) -> None:
        """particle_b=-1 should use the last particle."""
        calc = EndToEndDistance(particle_a=0, particle_b=-1)
        d = calc.compute(_CHAIN_POSITIONS)
        # Last particle is at (5,0,0), first at (0,0,0) -> distance 5.0
        self.assertAlmostEqual(d, 5.0, places=10)


# ---------------------------------------------------------------------------
# TestMolecularObservableCollector
# ---------------------------------------------------------------------------


class TestMolecularObservableCollector(unittest.TestCase):
    """Unified collector tests."""

    def setUp(self) -> None:
        self.collector = MolecularObservableCollector()
        self.state = _make_state()
        self.topology = _make_topology()

    def test_collect_all_returns_snapshot(self) -> None:
        """collect_all should return a MolecularSnapshot with all expected fields."""
        snapshot = self.collector.collect_all(self.state, self.topology)

        self.assertIsInstance(snapshot, MolecularSnapshot)
        self.assertIsNotNone(snapshot.sasa)
        self.assertIsNotNone(snapshot.gyration)
        self.assertIsNotNone(snapshot.hbonds)
        self.assertIsNotNone(snapshot.contact_map)
        self.assertIsNotNone(snapshot.secondary_structure)
        self.assertIsInstance(snapshot.end_to_end_distance, float)
        self.assertEqual(snapshot.step, 0)
        self.assertGreater(snapshot.sasa.total_sasa, 0.0)
        self.assertGreater(snapshot.gyration.radius_of_gyration, 0.0)

    def test_record_and_summary(self) -> None:
        """Record 5 frames and verify summary has mean values."""
        for step in range(5):
            state = _make_state(step=step)
            self.collector.record(step, state, self.topology)

        summary = self.collector.summary()
        self.assertEqual(summary["n_frames"], 5)
        self.assertGreater(summary["sasa_mean_nm2"], 0.0)
        self.assertGreater(summary["rg_mean_nm"], 0.0)
        self.assertGreaterEqual(summary["e2e_mean_nm"], 0.0)
        # Standard deviations should be defined (0 is fine if frames identical)
        self.assertGreaterEqual(summary["sasa_std_nm2"], 0.0)
        self.assertGreaterEqual(summary["rg_std_nm"], 0.0)

    def test_all_histories_populated(self) -> None:
        """After recording, all internal history lists should have entries."""
        for step in range(3):
            state = _make_state(step=step)
            self.collector.record(step, state, self.topology)

        self.assertEqual(len(self.collector._sasa_history), 3)
        self.assertEqual(len(self.collector._rg_history), 3)
        self.assertEqual(len(self.collector._hbond_history), 3)
        self.assertEqual(len(self.collector._contact_history), 3)
        self.assertEqual(len(self.collector._e2e_history), 3)


if __name__ == "__main__":
    unittest.main()
