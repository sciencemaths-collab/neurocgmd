"""Production-grade composite force evaluator integrating PBC, electrostatics,
angles, dihedrals, and shifted-force cutoffs for the NeuroCGMD pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, atan2, cos, floor, sin, sqrt

from core.exceptions import ContractValidationError
from core.state import SimulationCell, SimulationState
from core.types import FrozenMetadata, Vector3, VectorTuple
from forcefields.base_forcefield import BaseForceField
from physics.forces.composite import ForceEvaluation
from topology.system_topology import SystemTopology

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COULOMB_CONSTANT: float = 138.935458  # kJ*nm/(mol*e^2)
_CLAMP_ACOS: float = 1.0 - 1e-12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_box_lengths(cell: SimulationCell) -> Vector3:
    """Return (Lx, Ly, Lz) from diagonal elements of an orthorhombic cell."""
    bv = cell.box_vectors
    return (bv[0][0], bv[1][1], bv[2][2])


def minimum_image_delta(
    ri: Vector3,
    rj: Vector3,
    box_lengths: Vector3,
    periodic: tuple[bool, bool, bool],
) -> Vector3:
    """Compute displacement ri - rj with minimum image convention.

    Parameters
    ----------
    ri, rj : Vector3
        Position vectors.
    box_lengths : Vector3
        Box side lengths (Lx, Ly, Lz) extracted from the diagonal of box_vectors.
    periodic : tuple of 3 bool
        Whether each axis is periodic.

    Returns
    -------
    Vector3
        Displacement vector (dx, dy, dz) with minimum image applied on
        periodic axes.
    """
    dx = ri[0] - rj[0]
    dy = ri[1] - rj[1]
    dz = ri[2] - rj[2]
    if periodic[0]:
        lx = box_lengths[0]
        dx -= lx * round(dx / lx)
    if periodic[1]:
        ly = box_lengths[1]
        dy -= ly * round(dy / ly)
    if periodic[2]:
        lz = box_lengths[2]
        dz -= lz * round(dz / lz)
    return (dx, dy, dz)


def _vec_sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec_norm(a: Vector3) -> float:
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _vec_scale(a: Vector3, s: float) -> Vector3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _vec_add3(a: Vector3, b: Vector3, c: Vector3) -> Vector3:
    return (a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2])


# ---------------------------------------------------------------------------
# ProductionForceEvaluator
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProductionForceEvaluator:
    """Full production force pipeline with PBC, shifted-force LJ,
    reaction-field electrostatics, angle, and dihedral potentials."""

    use_pbc: bool = True
    use_shifted_force: bool = True
    electrostatic_method: str = "reaction_field"
    dielectric: float = 78.5
    charges: tuple[float, ...] | None = None
    angle_interactions: tuple | None = None
    dihedral_interactions: tuple | None = None
    exclude_bonded_nb: bool = True
    name: str = "production_force_evaluator"
    classification: str = "[adapted]"

    # ---- public entry point ------------------------------------------------

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
    ) -> ForceEvaluation:
        """Evaluate the full production force pipeline.

        Computes bond, angle, dihedral, and nonbonded (LJ + electrostatic)
        contributions and returns a single :class:`ForceEvaluation`.
        """
        n = state.particle_count
        positions = state.particles.positions
        forces: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(n)]

        # PBC setup
        pbc_active = self.use_pbc and state.cell is not None
        box_lengths: Vector3 | None = None
        periodic: tuple[bool, bool, bool] | None = None
        if pbc_active:
            assert state.cell is not None
            box_lengths = extract_box_lengths(state.cell)
            periodic = state.cell.periodic_axes

        # ------------------------------------------------------------------
        # (a) Bond forces
        # ------------------------------------------------------------------
        bond_energy = 0.0
        bonded_pairs: set[tuple[int, int]] = set()

        for bond in topology.bonds:
            ia = bond.particle_index_a
            ib = bond.particle_index_b
            bonded_pairs.add(bond.normalized_pair())

            param = forcefield.bond_parameter_for(topology, bond)
            k = param.stiffness
            r0 = param.equilibrium_distance

            if pbc_active:
                assert box_lengths is not None and periodic is not None
                delta = minimum_image_delta(positions[ia], positions[ib], box_lengths, periodic)
            else:
                delta = _vec_sub(positions[ia], positions[ib])

            dist = _vec_norm(delta)
            if dist < 1e-30:
                continue

            dr = dist - r0
            bond_energy += 0.5 * k * dr * dr

            f_mag = -k * dr / dist
            fx = f_mag * delta[0]
            fy = f_mag * delta[1]
            fz = f_mag * delta[2]

            forces[ia][0] += fx
            forces[ia][1] += fy
            forces[ia][2] += fz
            forces[ib][0] -= fx
            forces[ib][1] -= fy
            forces[ib][2] -= fz

        # ------------------------------------------------------------------
        # (b) Angle forces
        # ------------------------------------------------------------------
        angle_energy = 0.0

        if self.angle_interactions is not None:
            for interaction in self.angle_interactions:
                i, j, k_idx, theta0, k_angle = interaction

                if pbc_active:
                    assert box_lengths is not None and periodic is not None
                    r_ji = minimum_image_delta(positions[i], positions[j], box_lengths, periodic)
                    r_jk = minimum_image_delta(positions[k_idx], positions[j], box_lengths, periodic)
                else:
                    r_ji = _vec_sub(positions[i], positions[j])
                    r_jk = _vec_sub(positions[k_idx], positions[j])

                d_ji = _vec_norm(r_ji)
                d_jk = _vec_norm(r_jk)
                if d_ji < 1e-30 or d_jk < 1e-30:
                    continue

                cos_theta = _vec_dot(r_ji, r_jk) / (d_ji * d_jk)
                cos_theta = max(-_CLAMP_ACOS, min(_CLAMP_ACOS, cos_theta))
                theta = acos(cos_theta)

                dtheta = theta - theta0
                angle_energy += 0.5 * k_angle * dtheta * dtheta

                # Gradient: dE/dtheta * dtheta/d(cos_theta) * d(cos_theta)/dr
                sin_theta = sin(theta)
                if abs(sin_theta) < 1e-30:
                    continue

                prefactor = -k_angle * dtheta / sin_theta

                inv_ji = 1.0 / d_ji
                inv_jk = 1.0 / d_jk

                # Unit vectors
                e_ji = _vec_scale(r_ji, inv_ji)
                e_jk = _vec_scale(r_jk, inv_jk)

                # Force on i: prefactor * (e_jk - cos_theta * e_ji) / d_ji
                fi = _vec_scale(
                    (
                        e_jk[0] * inv_ji - cos_theta * e_ji[0] * inv_ji,
                        e_jk[1] * inv_ji - cos_theta * e_ji[1] * inv_ji,
                        e_jk[2] * inv_ji - cos_theta * e_ji[2] * inv_ji,
                    ),
                    prefactor,
                )
                # Force on k: prefactor * (e_ji - cos_theta * e_jk) / d_jk
                fk = _vec_scale(
                    (
                        e_ji[0] * inv_jk - cos_theta * e_jk[0] * inv_jk,
                        e_ji[1] * inv_jk - cos_theta * e_jk[1] * inv_jk,
                        e_ji[2] * inv_jk - cos_theta * e_jk[2] * inv_jk,
                    ),
                    prefactor,
                )

                forces[i][0] += fi[0]
                forces[i][1] += fi[1]
                forces[i][2] += fi[2]
                forces[k_idx][0] += fk[0]
                forces[k_idx][1] += fk[1]
                forces[k_idx][2] += fk[2]
                # Newton's third: force on j = -(fi + fk)
                forces[j][0] -= fi[0] + fk[0]
                forces[j][1] -= fi[1] + fk[1]
                forces[j][2] -= fi[2] + fk[2]

        # ------------------------------------------------------------------
        # (c) Dihedral forces
        # ------------------------------------------------------------------
        dihedral_energy = 0.0

        if self.dihedral_interactions is not None:
            for interaction in self.dihedral_interactions:
                i, j, k_idx, l_idx, k_dih, n_mult, phase = interaction

                if pbc_active:
                    assert box_lengths is not None and periodic is not None
                    b1 = minimum_image_delta(positions[j], positions[i], box_lengths, periodic)
                    b2 = minimum_image_delta(positions[k_idx], positions[j], box_lengths, periodic)
                    b3 = minimum_image_delta(positions[l_idx], positions[k_idx], box_lengths, periodic)
                else:
                    b1 = _vec_sub(positions[j], positions[i])
                    b2 = _vec_sub(positions[k_idx], positions[j])
                    b3 = _vec_sub(positions[l_idx], positions[k_idx])

                # Normal vectors to planes (i,j,k) and (j,k,l)
                n1 = _vec_cross(b1, b2)
                n2 = _vec_cross(b2, b3)

                nn1 = _vec_norm(n1)
                nn2 = _vec_norm(n2)
                nb2 = _vec_norm(b2)

                if nn1 < 1e-30 or nn2 < 1e-30 or nb2 < 1e-30:
                    continue

                # Dihedral angle via atan2 for correct sign
                m1 = _vec_cross(n1, _vec_scale(b2, 1.0 / nb2))
                x = _vec_dot(n1, n2) / (nn1 * nn2)
                y = _vec_dot(m1, n2) / (_vec_norm(m1) if _vec_norm(m1) > 1e-30 else 1.0) / nn2
                phi = atan2(y, x)

                dihedral_energy += k_dih * (1.0 + cos(n_mult * phi - phase))

                # dE/dphi
                dEdphi = -k_dih * n_mult * sin(n_mult * phi - phase)

                # Forces using the cross-product formulation
                # f_i = -dEdphi * |b2| / |n1|^2 * n1
                # f_l =  dEdphi * |b2| / |n2|^2 * n2
                inv_nn1_sq = 1.0 / (nn1 * nn1)
                inv_nn2_sq = 1.0 / (nn2 * nn2)

                fi = _vec_scale(n1, -dEdphi * nb2 * inv_nn1_sq)
                fl = _vec_scale(n2, dEdphi * nb2 * inv_nn2_sq)

                # Projections for j and k forces
                dot_b1_b2 = _vec_dot(b1, b2)
                dot_b3_b2 = _vec_dot(b3, b2)
                inv_nb2_sq = 1.0 / (nb2 * nb2)

                coeff_ji = dot_b1_b2 * inv_nb2_sq
                coeff_jl = dot_b3_b2 * inv_nb2_sq

                fj = (
                    (coeff_ji - 1.0) * fi[0] - coeff_jl * fl[0],
                    (coeff_ji - 1.0) * fi[1] - coeff_jl * fl[1],
                    (coeff_ji - 1.0) * fi[2] - coeff_jl * fl[2],
                )
                fk = (
                    -coeff_ji * fi[0] + (coeff_jl - 1.0) * fl[0],
                    -coeff_ji * fi[1] + (coeff_jl - 1.0) * fl[1],
                    -coeff_ji * fi[2] + (coeff_jl - 1.0) * fl[2],
                )

                for idx, f in ((i, fi), (j, fj), (k_idx, fk), (l_idx, fl)):
                    forces[idx][0] += f[0]
                    forces[idx][1] += f[1]
                    forces[idx][2] += f[2]

        # ------------------------------------------------------------------
        # (d) Nonbonded forces (LJ + optional electrostatics)
        # ------------------------------------------------------------------
        lj_energy = 0.0
        elec_energy = 0.0
        nb_pair_count = 0

        for ia in range(n):
            for ib in range(ia + 1, n):
                # Bonded exclusion
                if self.exclude_bonded_nb:
                    pair = (ia, ib) if ia < ib else (ib, ia)
                    if pair in bonded_pairs:
                        continue

                nb_param = forcefield.nonbonded_parameter_for_pair(topology, ia, ib)
                sigma = nb_param.sigma
                eps = nb_param.epsilon
                rc = nb_param.cutoff

                if pbc_active:
                    assert box_lengths is not None and periodic is not None
                    delta = minimum_image_delta(positions[ia], positions[ib], box_lengths, periodic)
                else:
                    delta = _vec_sub(positions[ia], positions[ib])

                dist = _vec_norm(delta)
                if dist < 1e-30 or dist > rc:
                    continue

                nb_pair_count += 1

                # --- LJ contribution ---
                inv_r = 1.0 / dist
                sr6 = (sigma * inv_r) ** 6
                sr12 = sr6 * sr6

                if self.use_shifted_force:
                    # Shifted-force LJ: smooth both energy and force to zero
                    # at the cutoff rc.
                    # E_lj(r)  = 4*eps*(sr12 - sr6)
                    # F_lj(r)  = 4*eps*(12*sr12 - 6*sr6) / r  (magnitude)
                    #
                    # E_sf(r) = E_lj(r) - E_lj(rc) - (r - rc)*dE_lj/dr|_rc
                    # F_sf(r) = F_lj(r) - F_lj(rc)

                    src6 = (sigma / rc) ** 6
                    src12 = src6 * src6
                    e_lj_r = 4.0 * eps * (sr12 - sr6)
                    e_lj_rc = 4.0 * eps * (src12 - src6)
                    # dE/dr at rc (note: this is negative of force magnitude)
                    de_dr_rc = 4.0 * eps * (-12.0 * src12 + 6.0 * src6) / rc

                    e_pair = e_lj_r - e_lj_rc - (dist - rc) * de_dr_rc

                    # Force magnitude = -dE_sf/dr
                    # dE_sf/dr = dE_lj/dr(r) - dE_lj/dr(rc)
                    de_dr_r = 4.0 * eps * (-12.0 * sr12 + 6.0 * sr6) * inv_r
                    f_mag = -(de_dr_r - de_dr_rc) * inv_r
                else:
                    # Standard hard-cutoff LJ
                    e_pair = 4.0 * eps * (sr12 - sr6)
                    f_mag = 4.0 * eps * (12.0 * sr12 - 6.0 * sr6) * inv_r * inv_r

                lj_energy += e_pair

                fx = f_mag * delta[0]
                fy = f_mag * delta[1]
                fz = f_mag * delta[2]

                forces[ia][0] += fx
                forces[ia][1] += fy
                forces[ia][2] += fz
                forces[ib][0] -= fx
                forces[ib][1] -= fy
                forces[ib][2] -= fz

                # --- Electrostatics contribution ---
                if (
                    self.charges is not None
                    and self.electrostatic_method != "none"
                ):
                    qi = self.charges[ia]
                    qj = self.charges[ib]
                    if abs(qi) > 1e-30 and abs(qj) > 1e-30:
                        qq = qi * qj

                        if self.electrostatic_method == "reaction_field":
                            eps_r = self.dielectric
                            k_rf = (eps_r - 1.0) / (2.0 * eps_r + 1.0) / (rc ** 3)
                            c_rf = 3.0 * eps_r / (2.0 * eps_r + 1.0) / rc

                            e_rf = _COULOMB_CONSTANT * qq * (
                                inv_r + k_rf * dist * dist - c_rf
                            )
                            # F = -dE/dr * r_hat
                            # dE/dr = C*qq*(-1/r^2 + 2*k_rf*r)
                            de_dr = _COULOMB_CONSTANT * qq * (
                                -inv_r * inv_r + 2.0 * k_rf * dist
                            )
                            ef_mag = -de_dr * inv_r

                            elec_energy += e_rf

                        elif self.electrostatic_method == "direct":
                            e_coul = _COULOMB_CONSTANT * qq * inv_r
                            # F = C*qq/r^2 * r_hat
                            ef_mag = _COULOMB_CONSTANT * qq * inv_r * inv_r * inv_r

                            elec_energy += e_coul

                        else:
                            raise ContractValidationError(
                                f"Unknown electrostatic_method: "
                                f"{self.electrostatic_method!r}. "
                                f"Must be 'none', 'reaction_field', or 'direct'."
                            )

                        efx = ef_mag * delta[0]
                        efy = ef_mag * delta[1]
                        efz = ef_mag * delta[2]
                        forces[ia][0] += efx
                        forces[ia][1] += efy
                        forces[ia][2] += efz
                        forces[ib][0] -= efx
                        forces[ib][1] -= efy
                        forces[ib][2] -= efz

        # ------------------------------------------------------------------
        # (e) Assemble result
        # ------------------------------------------------------------------
        total_energy = bond_energy + angle_energy + dihedral_energy + lj_energy + elec_energy

        forces_tuple: VectorTuple = tuple(
            (f[0], f[1], f[2]) for f in forces
        )

        component_energies = FrozenMetadata(
            {
                "bond": bond_energy,
                "angle": angle_energy,
                "dihedral": dihedral_energy,
                "lj": lj_energy,
                "electrostatic": elec_energy,
            }
        )
        metadata = FrozenMetadata(
            {
                "evaluated_bonds": len(topology.bonds),
                "evaluated_angles": (
                    len(self.angle_interactions)
                    if self.angle_interactions is not None
                    else 0
                ),
                "evaluated_dihedrals": (
                    len(self.dihedral_interactions)
                    if self.dihedral_interactions is not None
                    else 0
                ),
                "evaluated_nonbonded_pairs": nb_pair_count,
                "pbc_active": pbc_active,
                "shifted_force": self.use_shifted_force,
                "electrostatic_method": self.electrostatic_method,
            }
        )

        return ForceEvaluation(
            forces=forces_tuple,
            potential_energy=total_energy,
            component_energies=component_energies,
            metadata=metadata,
        )
