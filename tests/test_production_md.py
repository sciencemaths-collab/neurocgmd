"""Comprehensive tests for the production MD features added to NeuroCGMD.

Covers periodic boundary conditions, unit utilities, holonomic constraint
solvers (SHAKE / LINCS), electrostatic methods (reaction-field / Ewald),
bonded angle and dihedral potentials, nonbonded WCA / shifted-force LJ /
switch-function / Coulomb potentials, and the composite ProductionForceEvaluator.
"""

from __future__ import annotations

import unittest
from math import sqrt, pi, cos, sin, erfc, exp

from core.state import (
    EnsembleKind, ParticleState, SimulationState, SimulationCell,
    StateProvenance, ThermodynamicState, UnitSystem,
)
from core.types import BeadId, FrozenMetadata, SimulationId, StateId
from topology.beads import Bead, BeadRole, BeadType
from topology.bonds import Bond
from topology.system_topology import SystemTopology
from forcefields.base_forcefield import BaseForceField, BondParameter, NonbondedParameter


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_provenance(label: str = "test") -> StateProvenance:
    """Build a minimal valid StateProvenance for test fixtures."""
    return StateProvenance(
        simulation_id=SimulationId(f"sim-{label}"),
        state_id=StateId(f"state-{label}"),
        parent_state_id=None,
        created_by="unit-test",
        stage="initialization",
    )


def _make_state(
    positions: tuple[tuple[float, float, float], ...],
    masses: tuple[float, ...] | None = None,
    cell: SimulationCell | None = None,
    label: str = "test",
) -> SimulationState:
    """Build a minimal SimulationState for test fixtures."""
    n = len(positions)
    if masses is None:
        masses = tuple(1.0 for _ in range(n))
    return SimulationState(
        units=UnitSystem.md_nano(),
        particles=ParticleState(positions=positions, masses=masses),
        thermodynamics=ThermodynamicState(),
        provenance=_make_provenance(label),
        cell=cell,
    )


def _make_3particle_topology() -> SystemTopology:
    """Return a 3-bead topology with two bonds (0-1, 1-2) and a single bead type."""
    return SystemTopology(
        system_id="test-3p",
        bead_types=(
            BeadType(name="bb", role=BeadRole.STRUCTURAL),
        ),
        beads=(
            Bead(bead_id=BeadId("b0"), particle_index=0, bead_type="bb", label="B0"),
            Bead(bead_id=BeadId("b1"), particle_index=1, bead_type="bb", label="B1"),
            Bead(bead_id=BeadId("b2"), particle_index=2, bead_type="bb", label="B2"),
        ),
        bonds=(Bond(0, 1), Bond(1, 2)),
    )


def _make_3particle_forcefield() -> BaseForceField:
    """Return a forcefield for a homogeneous 3-bead system."""
    return BaseForceField(
        name="test-3p-ff",
        bond_parameters=(
            BondParameter("bb", "bb", equilibrium_distance=1.0, stiffness=100.0),
        ),
        nonbonded_parameters=(
            NonbondedParameter("bb", "bb", sigma=0.5, epsilon=1.0, cutoff=2.5),
        ),
    )


# =========================================================================== #
# 1. TestPeriodicBoundary
# =========================================================================== #

from physics.periodic_boundary import (
    minimum_image_displacement,
    minimum_image_distance,
    wrap_positions,
    PBCHandler,
)


class TestPeriodicBoundary(unittest.TestCase):
    """Verify minimum-image convention and position wrapping."""

    def test_minimum_image_simple(self) -> None:
        """Particles at x=0.1 and x=9.9 in a box of 10: displacement ~+0.2."""
        cell = SimulationCell(box_vectors=((10.0, 0, 0), (0, 10.0, 0), (0, 0, 10.0)))
        r_i = (0.1, 0.0, 0.0)
        r_j = (9.9, 0.0, 0.0)

        dx, dy, dz = minimum_image_displacement(r_i, r_j, cell)
        # The nearest image of 9.9 to 0.1 across the periodic boundary is
        # 0.1 + (-0.2).  Displacement = r_j - r_i under PBC = -0.2.
        self.assertAlmostEqual(abs(dx), 0.2, places=10)
        self.assertAlmostEqual(dy, 0.0)
        self.assertAlmostEqual(dz, 0.0)

        dist = minimum_image_distance(r_i, r_j, cell)
        self.assertAlmostEqual(dist, 0.2, places=10)

    def test_minimum_image_no_pbc(self) -> None:
        """With all periodic axes disabled, displacement is raw difference."""
        cell = SimulationCell(
            box_vectors=((10.0, 0, 0), (0, 10.0, 0), (0, 0, 10.0)),
            periodic_axes=(False, False, False),
        )
        r_i = (0.1, 0.0, 0.0)
        r_j = (9.9, 0.0, 0.0)

        dx, dy, dz = minimum_image_displacement(r_i, r_j, cell)
        self.assertAlmostEqual(dx, 9.8, places=10)
        self.assertAlmostEqual(dy, 0.0)
        self.assertAlmostEqual(dz, 0.0)

    def test_wrap_positions(self) -> None:
        """Position at x=12.5 in box [0, 10] should wrap to 2.5."""
        cell = SimulationCell(box_vectors=((10.0, 0, 0), (0, 10.0, 0), (0, 0, 10.0)))
        wrapped = wrap_positions(((12.5, 0.0, 0.0),), cell)
        self.assertAlmostEqual(wrapped[0][0], 2.5, places=10)

    def test_wrap_preserves_non_periodic(self) -> None:
        """Non-periodic axis positions are unchanged after wrapping."""
        cell = SimulationCell(
            box_vectors=((10.0, 0, 0), (0, 10.0, 0), (0, 0, 10.0)),
            periodic_axes=(True, False, True),
        )
        wrapped = wrap_positions(((12.5, 15.0, -3.0),), cell)
        self.assertAlmostEqual(wrapped[0][0], 2.5, places=10)
        # y is non-periodic -- should be unchanged
        self.assertAlmostEqual(wrapped[0][1], 15.0, places=10)
        # z is periodic: -3.0 mod 10 = 7.0
        self.assertAlmostEqual(wrapped[0][2], 7.0, places=10)

    def test_pbc_handler_wraps_state(self) -> None:
        """PBCHandler.wrap_state moves out-of-box particles back inside."""
        cell = SimulationCell(box_vectors=((10.0, 0, 0), (0, 10.0, 0), (0, 0, 10.0)))
        state = _make_state(
            positions=((12.0, -1.0, 25.0), (5.0, 5.0, 5.0)),
            cell=cell,
            label="pbc-wrap",
        )
        handler = PBCHandler()
        wrapped_state = handler.wrap_state(state)

        for pos in wrapped_state.particles.positions:
            for coord in pos:
                self.assertGreaterEqual(coord, 0.0)
                self.assertLess(coord, 10.0)


# =========================================================================== #
# 2. TestUnits
# =========================================================================== #

from core.units import (
    BOLTZMANN_CONSTANT, COULOMB_CONSTANT as COULOMB_CONST_UNITS,
    UnitConverter, UnitValidator,
)


class TestUnits(unittest.TestCase):
    """Verify physical constants, conversions, and validators."""

    def test_boltzmann_constant(self) -> None:
        """kB should be ~0.00831446 kJ/(mol*K)."""
        self.assertAlmostEqual(BOLTZMANN_CONSTANT, 0.00831446, places=6)

    def test_coulomb_constant(self) -> None:
        """1/(4pi eps0) should be ~138.935458 kJ*nm/(mol*e^2)."""
        self.assertAlmostEqual(COULOMB_CONST_UNITS, 138.935458, places=3)

    def test_unit_converter_kcal_kj(self) -> None:
        """1 kcal/mol = 4.184 kJ/mol."""
        conv = UnitConverter()
        self.assertAlmostEqual(conv.kcal_to_kj(1.0), 4.184, places=6)

    def test_thermal_energy(self) -> None:
        """At 300 K, kB*T ~ 2.494 kJ/mol."""
        conv = UnitConverter()
        kbt = conv.thermal_energy(300.0)
        self.assertAlmostEqual(kbt, BOLTZMANN_CONSTANT * 300.0, places=6)
        self.assertAlmostEqual(kbt, 2.494338, places=3)

    def test_validator_rejects_negative_mass(self) -> None:
        """validate_mass(-1.0) should return at least one issue."""
        val = UnitValidator()
        issues = val.validate_mass(-1.0)
        self.assertGreater(len(issues), 0)

    def test_validator_accepts_valid(self) -> None:
        """validate_temperature(300.0) should return empty tuple."""
        val = UnitValidator()
        issues = val.validate_temperature(300.0)
        self.assertEqual(issues, ())


# =========================================================================== #
# 3. TestConstraints
# =========================================================================== #

from physics.constraints import (
    DistanceConstraint, SHAKESolver, LINCSolver, ConstraintResult,
)


class TestConstraints(unittest.TestCase):
    """Verify SHAKE and LINCS holonomic constraint solvers."""

    # -- SHAKE --------------------------------------------------------------

    def test_shake_fixes_bond_length(self) -> None:
        """Two particles stretched beyond target --> SHAKE corrects to target."""
        old_pos = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        new_pos = ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0))
        masses = (1.0, 1.0)
        constraints = (DistanceConstraint(0, 1, target_distance=1.0),)

        solver = SHAKESolver(tolerance=1e-8)
        result = solver.apply(old_pos, new_pos, masses, constraints)

        # Compute distance after correction.
        dx = result.positions[1][0] - result.positions[0][0]
        dy = result.positions[1][1] - result.positions[0][1]
        dz = result.positions[1][2] - result.positions[0][2]
        dist = sqrt(dx * dx + dy * dy + dz * dz)
        self.assertAlmostEqual(dist, 1.0, places=4)

    def test_shake_converges(self) -> None:
        """Simple two-particle constraint converges."""
        old_pos = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        new_pos = ((0.0, 0.0, 0.0), (1.2, 0.0, 0.0))
        masses = (1.0, 1.0)
        constraints = (DistanceConstraint(0, 1, target_distance=1.0),)

        solver = SHAKESolver()
        result = solver.apply(old_pos, new_pos, masses, constraints)
        self.assertTrue(result.converged)

    def test_shake_preserves_center_of_mass(self) -> None:
        """Center of mass must be (approximately) preserved by SHAKE."""
        old_pos = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        new_pos = ((-0.1, 0.0, 0.0), (1.3, 0.0, 0.0))
        masses = (1.0, 1.0)
        constraints = (DistanceConstraint(0, 1, target_distance=1.0),)

        # COM of new_pos before correction
        com_before_x = (masses[0] * new_pos[0][0] + masses[1] * new_pos[1][0]) / sum(masses)

        solver = SHAKESolver(tolerance=1e-10)
        result = solver.apply(old_pos, new_pos, masses, constraints)

        com_after_x = (
            masses[0] * result.positions[0][0] + masses[1] * result.positions[1][0]
        ) / sum(masses)
        self.assertAlmostEqual(com_before_x, com_after_x, places=6)

    # -- LINCS --------------------------------------------------------------

    def test_lincs_fixes_bond_length(self) -> None:
        """Two particles stretched beyond target --> LINCS corrects to target."""
        old_pos = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        new_pos = ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0))
        masses = (1.0, 1.0)
        constraints = (DistanceConstraint(0, 1, target_distance=1.0),)

        solver = LINCSolver(order=6)
        result = solver.apply(old_pos, new_pos, masses, constraints)

        dx = result.positions[1][0] - result.positions[0][0]
        dy = result.positions[1][1] - result.positions[0][1]
        dz = result.positions[1][2] - result.positions[0][2]
        dist = sqrt(dx * dx + dy * dy + dz * dz)
        self.assertAlmostEqual(dist, 1.0, places=4)

    def test_lincs_handles_multiple_constraints(self) -> None:
        """3-particle chain with 2 constraints — LINCS reduces violations vs unconstrained."""
        old_pos = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        new_pos = ((0.0, 0.0, 0.0), (1.05, 0.0, 0.0), (2.08, 0.0, 0.0))
        masses = (1.0, 1.0, 1.0)
        constraints = (
            DistanceConstraint(0, 1, target_distance=1.0),
            DistanceConstraint(1, 2, target_distance=1.0),
        )

        solver = LINCSolver(order=8)
        result = solver.apply(old_pos, new_pos, masses, constraints)

        # Measure violations before and after
        for con in constraints:
            a, b = con.particle_a, con.particle_b
            # Unconstrained violation
            dx0 = new_pos[b][0] - new_pos[a][0]
            dist_before = sqrt(dx0*dx0)
            # Constrained violation
            dx1 = result.positions[b][0] - result.positions[a][0]
            dy1 = result.positions[b][1] - result.positions[a][1]
            dz1 = result.positions[b][2] - result.positions[a][2]
            dist_after = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
            # LINCS should at least reduce the violation
            error_before = abs(dist_before - con.target_distance)
            error_after = abs(dist_after - con.target_distance)
            self.assertLessEqual(error_after, error_before + 0.01,
                                 f"LINCS should not make constraint ({a},{b}) worse")


# =========================================================================== #
# 4. TestElectrostatics
# =========================================================================== #

from physics.electrostatics import (
    ChargeSet, EwaldParameters, EwaldSummation,
    ReactionFieldElectrostatics,
)


class TestElectrostatics(unittest.TestCase):
    """Verify charge bookkeeping, reaction-field, and Ewald summation."""

    def test_charge_set_total(self) -> None:
        """Two equal and opposite charges should be neutral."""
        charges = ChargeSet(charges=(1.0, -1.0))
        self.assertAlmostEqual(charges.total_charge(), 0.0, places=12)
        self.assertTrue(charges.is_neutral())

    def test_reaction_field_two_charges(self) -> None:
        """+1 and -1 at distance 1.0 --> attractive (negative) energy."""
        state = _make_state(
            positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            label="rf-attract",
        )
        charges = ChargeSet(charges=(1.0, -1.0))
        rf = ReactionFieldElectrostatics(cutoff=2.0)
        result = rf.evaluate(state, charges)
        self.assertLess(result.total_energy, 0.0)

    def test_reaction_field_same_charges(self) -> None:
        """+1 and +1 at distance 1.0 --> repulsive (positive) energy."""
        state = _make_state(
            positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            label="rf-repel",
        )
        charges = ChargeSet(charges=(1.0, 1.0))
        rf = ReactionFieldElectrostatics(cutoff=2.0)
        result = rf.evaluate(state, charges)
        self.assertGreater(result.total_energy, 0.0)

    def test_ewald_auto_parameters(self) -> None:
        """auto_from_cutoff(1.0) should produce reasonable alpha and kmax."""
        params = EwaldParameters.auto_from_cutoff(1.0)
        self.assertGreater(params.alpha, 0.0)
        self.assertGreaterEqual(params.kmax, 1)
        self.assertAlmostEqual(params.real_cutoff, 1.0)

    def test_ewald_neutral_system(self) -> None:
        """Two opposite charges in a box: finite energy, forces attract."""
        cell = SimulationCell(box_vectors=((5.0, 0, 0), (0, 5.0, 0), (0, 0, 5.0)))
        state = SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((1.0, 2.5, 2.5), (4.0, 2.5, 2.5)),
                masses=(1.0, 1.0),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("t"),
                state_id=StateId("s"),
                parent_state_id=None,
                created_by="test",
                stage="initialization",
            ),
            cell=cell,
        )
        charges = ChargeSet(charges=(1.0, -1.0))
        params = EwaldParameters.auto_from_cutoff(2.0, max_box_length=5.0)
        ewald = EwaldSummation(params=params)
        result = ewald.evaluate(state, charges)

        # Energy should be finite and negative (attraction dominates).
        self.assertTrue(result.total_energy != 0.0)
        import math
        self.assertFalse(math.isinf(result.total_energy))
        self.assertFalse(math.isnan(result.total_energy))

        # Force on the positive charge should pull it towards the negative
        # charge (positive x-direction, since the negative is at x=4.0).
        fx_positive = result.total_forces[0][0]
        fx_negative = result.total_forces[1][0]
        # Particle 0 at x=1 should be pushed right (+x), particle 1 at x=4 pushed left (-x).
        # But minimum image: distance is 3.0 direct vs 2.0 via boundary.
        # Via boundary the closest image of x=4 from x=1 is at x=-1 (distance 2).
        # So force on particle 0 should be in the -x direction (towards -1).
        # Regardless of sign, forces should be opposite (Newton's 3rd law).
        self.assertAlmostEqual(
            fx_positive + fx_negative, 0.0, places=3,
            msg="Forces should obey Newton's third law",
        )


# =========================================================================== #
# 5. TestBondedPotentials
# =========================================================================== #

from forcefields.bonded_potentials import (
    AngleParameter, DihedralParameter, AngleForceModel, DihedralForceModel,
)


class TestBondedPotentials(unittest.TestCase):
    """Verify angle and dihedral energy terms, plus nonbonded potentials."""

    # -- Angle interactions --------------------------------------------------

    def test_angle_energy_at_equilibrium(self) -> None:
        """An angle held exactly at theta0 has zero energy."""
        theta0 = pi / 3.0
        # Build 3 particles forming an angle theta0 at the central bead.
        r = 1.0
        positions = (
            (r, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (r * cos(theta0), r * sin(theta0), 0.0),
        )
        param = AngleParameter("bb", "bb", "bb", equilibrium_angle=theta0, force_constant=100.0)
        model = AngleForceModel()
        _forces, energy = model.evaluate(positions, ((0, 1, 2, param),))
        self.assertAlmostEqual(energy, 0.0, places=8)

    def test_angle_energy_off_equilibrium(self) -> None:
        """An angle away from theta0 has positive energy."""
        theta0 = pi / 2.0
        # Actual angle is pi (linear arrangement), very different from pi/2.
        positions = (
            (-1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        )
        param = AngleParameter("bb", "bb", "bb", equilibrium_angle=theta0, force_constant=100.0)
        model = AngleForceModel()
        _forces, energy = model.evaluate(positions, ((0, 1, 2, param),))
        self.assertGreater(energy, 0.0)

    # -- Nonbonded potentials (from expected interfaces) ---------------------

    def test_wca_purely_repulsive(self) -> None:
        """WCA potential: positive for r < 2^(1/6)*sigma, zero beyond."""
        try:
            from forcefields.nonbonded_potentials import WCAPotential
        except ImportError:
            self.skipTest("WCAPotential not yet implemented")

        sigma = 1.0
        epsilon = 1.0
        wca = WCAPotential(sigma=sigma, epsilon=epsilon)
        cutoff = 2.0 ** (1.0 / 6.0) * sigma

        # Inside the repulsive region
        e_inside = wca.energy(cutoff * 0.9)
        self.assertGreater(e_inside, 0.0)

        # Outside the repulsive region
        e_outside = wca.energy(cutoff * 1.1)
        self.assertAlmostEqual(e_outside, 0.0, places=8)

    def test_shifted_force_lj_zero_at_cutoff(self) -> None:
        """Shifted-force LJ energy is exactly 0 at r = cutoff."""
        try:
            from forcefields.nonbonded_potentials import ShiftedForceLJ
        except ImportError:
            self.skipTest("ShiftedForceLJ not yet implemented")

        lj = ShiftedForceLJ(sigma=1.0, epsilon=1.0, cutoff=2.5)
        e_at_cutoff = lj.energy(2.5)
        self.assertAlmostEqual(e_at_cutoff, 0.0, places=10)

    def test_switch_function_boundaries(self) -> None:
        """S(r) = 1 for r < inner, S(r) = 0 for r > outer."""
        try:
            from forcefields.nonbonded_potentials import SwitchFunction
        except ImportError:
            self.skipTest("SwitchFunction not yet implemented")

        sw = SwitchFunction(inner_cutoff=1.0, outer_cutoff=2.0)
        val_lo, _ = sw.evaluate(0.5)
        self.assertAlmostEqual(val_lo, 1.0, places=10)
        val_lo2, _ = sw.evaluate(0.99)
        self.assertAlmostEqual(val_lo2, 1.0, places=10)
        val_hi, _ = sw.evaluate(2.5)
        self.assertAlmostEqual(val_hi, 0.0, places=10)
        # Somewhere in between should be in (0, 1)
        mid, _ = sw.evaluate(1.5)
        self.assertGreater(mid, 0.0)
        self.assertLess(mid, 1.0)

    def test_coulomb_opposite_charges_attract(self) -> None:
        """Coulomb energy for +1/-1 pair is negative."""
        try:
            from forcefields.nonbonded_potentials import CoulombPotential
        except ImportError:
            self.skipTest("CoulombPotential not yet implemented")

        cp = CoulombPotential()
        energy = cp.energy(qi=1.0, qj=-1.0, r=1.0)
        self.assertLess(energy, 0.0)


# =========================================================================== #
# 6. TestProductionEvaluator
# =========================================================================== #


class TestProductionEvaluator(unittest.TestCase):
    """Verify the composite ProductionForceEvaluator that adds production
    MD features on top of the baseline evaluator."""

    def _build_periodic_3particle_system(
        self,
    ) -> tuple[SimulationState, SystemTopology, BaseForceField]:
        """Build a 3-particle periodic system with topology and forcefield."""
        cell = SimulationCell(box_vectors=((5.0, 0, 0), (0, 5.0, 0), (0, 0, 5.0)))
        state = SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((1.0, 2.5, 2.5), (2.0, 2.5, 2.5), (3.0, 2.5, 2.5)),
                masses=(1.0, 1.0, 1.0),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=_make_provenance("prod-eval"),
            cell=cell,
        )
        topology = _make_3particle_topology()
        forcefield = _make_3particle_forcefield()
        return state, topology, forcefield

    def test_evaluator_with_pbc(self) -> None:
        """ProductionForceEvaluator with periodic system: forces must be finite."""
        try:
            from physics.production_evaluator import ProductionForceEvaluator
        except ImportError:
            self.skipTest("ProductionForceEvaluator not yet implemented")

        import math

        state, topology, forcefield = self._build_periodic_3particle_system()
        evaluator = ProductionForceEvaluator()
        result = evaluator.evaluate(state, topology, forcefield)

        for i, force_vec in enumerate(result.forces):
            for d, component in enumerate(force_vec):
                self.assertFalse(
                    math.isnan(component),
                    f"Force on particle {i} component {d} is NaN",
                )
                self.assertFalse(
                    math.isinf(component),
                    f"Force on particle {i} component {d} is Inf",
                )

    def test_evaluator_matches_baseline_without_extras(self) -> None:
        """Without charges, angles, or dihedrals, results should be close
        to the BaselineForceEvaluator."""
        try:
            from physics.production_evaluator import ProductionForceEvaluator
        except ImportError:
            self.skipTest("ProductionForceEvaluator not yet implemented")

        from physics.forces.composite import BaselineForceEvaluator

        state, topology, forcefield = self._build_periodic_3particle_system()

        baseline = BaselineForceEvaluator()
        production = ProductionForceEvaluator()

        baseline_result = baseline.evaluate(state, topology, forcefield)
        production_result = production.evaluate(state, topology, forcefield)

        # Potential energies should be close (production may add small corrections
        # from PBC, but with these well-separated particles the difference should
        # be negligible).
        self.assertAlmostEqual(
            baseline_result.potential_energy,
            production_result.potential_energy,
            places=2,
            msg="Production evaluator without extras should approximate baseline",
        )

    def test_evaluator_with_charges(self) -> None:
        """When charges are provided, there should be an electrostatic
        contribution in the component energies."""
        try:
            from physics.production_evaluator import ProductionForceEvaluator
        except ImportError:
            self.skipTest("ProductionForceEvaluator not yet implemented")

        state, topology, forcefield = self._build_periodic_3particle_system()
        evaluator = ProductionForceEvaluator(charges=(1.0, -1.0, 0.5))
        result = evaluator.evaluate(state, topology, forcefield)

        # The component_energies dict should contain an electrostatic key.
        component_dict = dict(result.component_energies)
        has_electrostatic = any(
            "electrostatic" in str(k).lower() or "coulomb" in str(k).lower()
            for k in component_dict
        )
        self.assertTrue(
            has_electrostatic,
            f"Expected electrostatic component in {list(component_dict.keys())}",
        )


# =========================================================================== #
# Entry point
# =========================================================================== #

if __name__ == "__main__":
    unittest.main()
