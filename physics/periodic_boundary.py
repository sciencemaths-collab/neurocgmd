"""Periodic boundary conditions with minimum image convention.

Provides orthorhombic and triclinic PBC handling for molecular dynamics
simulations, including position wrapping, minimum-image displacement,
and pairwise distance computation under periodic images.

Classification: [established]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor, sqrt

from core.exceptions import ContractValidationError
from core.state import SimulationCell, SimulationState, ParticleState
from core.types import FrozenMetadata, Vector3, VectorTuple


# ---------------------------------------------------------------------------
# Orthorhombic helpers
# ---------------------------------------------------------------------------


def minimum_image_displacement(
    r_i: Vector3,
    r_j: Vector3,
    cell: SimulationCell,
) -> Vector3:
    """Return the minimum-image displacement vector r_j - r_i.

    For an orthorhombic cell the standard nearest-image formula is applied
    independently along each periodic axis:

        dx -= box_x * round(dx / box_x)

    Non-periodic axes are left untouched.
    """
    box = cell.box_vectors
    periodic = cell.periodic_axes

    dx = r_j[0] - r_i[0]
    dy = r_j[1] - r_i[1]
    dz = r_j[2] - r_i[2]

    if periodic[0]:
        box_x = box[0][0]
        dx -= box_x * round(dx / box_x)
    if periodic[1]:
        box_y = box[1][1]
        dy -= box_y * round(dy / box_y)
    if periodic[2]:
        box_z = box[2][2]
        dz -= box_z * round(dz / box_z)

    return (dx, dy, dz)


def minimum_image_distance(
    r_i: Vector3,
    r_j: Vector3,
    cell: SimulationCell,
) -> float:
    """Return the scalar minimum-image distance between two positions."""
    dx, dy, dz = minimum_image_displacement(r_i, r_j, cell)
    return sqrt(dx * dx + dy * dy + dz * dz)


def wrap_positions(
    positions: VectorTuple,
    cell: SimulationCell,
) -> VectorTuple:
    """Wrap all positions into the primary simulation cell.

    For orthorhombic cells the mapping is:

        x -= box_x * floor((x - origin_x) / box_x)

    applied independently to each periodic axis.
    """
    box = cell.box_vectors
    periodic = cell.periodic_axes
    origin = cell.origin

    wrapped: list[Vector3] = []
    for pos in positions:
        x, y, z = pos

        if periodic[0]:
            box_x = box[0][0]
            x -= box_x * floor((x - origin[0]) / box_x)
        if periodic[1]:
            box_y = box[1][1]
            y -= box_y * floor((y - origin[1]) / box_y)
        if periodic[2]:
            box_z = box[2][2]
            z -= box_z * floor((z - origin[2]) / box_z)

        wrapped.append((x, y, z))

    return tuple(wrapped)


# ---------------------------------------------------------------------------
# Orthorhombic PBC handler
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PBCHandler:
    """Orthorhombic periodic boundary condition handler.

    Provides position wrapping and pairwise distance computation using the
    minimum image convention for axis-aligned simulation cells.
    """

    name: str = "pbc_handler"
    classification: str = "[established]"

    def wrap_state(self, state: SimulationState) -> SimulationState:
        """Return a new state with positions wrapped into the primary cell.

        If the state has no simulation cell the state is returned unchanged.
        """
        if state.cell is None:
            return state

        new_positions = wrap_positions(state.particles.positions, state.cell)
        new_particles = state.particles.with_positions(new_positions)

        # Reconstruct the immutable SimulationState with updated particles.
        return SimulationState(
            units=state.units,
            particles=new_particles,
            thermodynamics=state.thermodynamics,
            provenance=state.provenance,
            cell=state.cell,
            time=state.time,
            step=state.step,
            potential_energy=state.potential_energy,
            observables=state.observables,
        )

    def pairwise_distances(
        self, state: SimulationState,
    ) -> tuple[tuple[int, int, float], ...]:
        """Return all unique pairwise minimum-image distances.

        Each entry is ``(i, j, distance)`` with ``i < j``.  If no simulation
        cell is present, plain Euclidean distances are returned.
        """
        positions = state.particles.positions
        n = state.particle_count
        pairs: list[tuple[int, int, float]] = []

        if state.cell is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    d = minimum_image_distance(positions[i], positions[j], state.cell)
                    pairs.append((i, j, d))
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    dx = positions[j][0] - positions[i][0]
                    dy = positions[j][1] - positions[i][1]
                    dz = positions[j][2] - positions[i][2]
                    pairs.append((i, j, sqrt(dx * dx + dy * dy + dz * dz)))

        return tuple(pairs)


# ---------------------------------------------------------------------------
# Triclinic helpers
# ---------------------------------------------------------------------------


def _mat3_inverse(m: tuple[Vector3, Vector3, Vector3]) -> tuple[Vector3, Vector3, Vector3]:
    """Return the inverse of a 3x3 matrix represented as a tuple of row vectors.

    Uses the classical adjugate / determinant formula.
    """
    (a, b, c), (d, e, f), (g, h, k) = m

    det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-30:
        raise ContractValidationError(
            "Box vector matrix is singular or near-singular; cannot invert."
        )

    inv_det = 1.0 / det

    r00 = (e * k - f * h) * inv_det
    r01 = (c * h - b * k) * inv_det
    r02 = (b * f - c * e) * inv_det
    r10 = (f * g - d * k) * inv_det
    r11 = (a * k - c * g) * inv_det
    r12 = (c * d - a * f) * inv_det
    r20 = (d * h - e * g) * inv_det
    r21 = (b * g - a * h) * inv_det
    r22 = (a * e - b * d) * inv_det

    return (
        (r00, r01, r02),
        (r10, r11, r12),
        (r20, r21, r22),
    )


def _mat3_vec_mul(m: tuple[Vector3, Vector3, Vector3], v: Vector3) -> Vector3:
    """Multiply a 3x3 matrix (row-major) by a column vector."""
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


# ---------------------------------------------------------------------------
# Triclinic PBC handler
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TriclinicPBCHandler:
    """Periodic boundary handler for general triclinic simulation cells.

    Positions are converted to fractional coordinates via H^{-1}, wrapped
    into [0, 1) along periodic axes, and converted back via H.  The same
    fractional-coordinate approach is used for minimum-image displacement.
    """

    name: str = "triclinic_pbc_handler"
    classification: str = "[established]"

    # -- position wrapping --------------------------------------------------

    def wrap_positions(
        self,
        positions: VectorTuple,
        cell: SimulationCell,
    ) -> VectorTuple:
        """Wrap positions into the primary triclinic cell.

        The box matrix H is taken directly from ``cell.box_vectors`` (rows are
        the three lattice vectors).  Fractional coordinates are
        ``s = H^{-1} * (r - origin)``; wrapping sets ``s -= floor(s)`` for
        periodic axes, then ``r = H * s + origin``.
        """
        h_matrix = cell.box_vectors
        h_inv = _mat3_inverse(h_matrix)
        periodic = cell.periodic_axes
        origin = cell.origin

        wrapped: list[Vector3] = []
        for pos in positions:
            # Shift to cell-relative coordinates.
            r = (pos[0] - origin[0], pos[1] - origin[1], pos[2] - origin[2])
            s = _mat3_vec_mul(h_inv, r)

            sx = s[0] - floor(s[0]) if periodic[0] else s[0]
            sy = s[1] - floor(s[1]) if periodic[1] else s[1]
            sz = s[2] - floor(s[2]) if periodic[2] else s[2]

            r_new = _mat3_vec_mul(h_matrix, (sx, sy, sz))
            wrapped.append((
                r_new[0] + origin[0],
                r_new[1] + origin[1],
                r_new[2] + origin[2],
            ))

        return tuple(wrapped)

    # -- minimum image ------------------------------------------------------

    def minimum_image_displacement(
        self,
        r_i: Vector3,
        r_j: Vector3,
        cell: SimulationCell,
    ) -> Vector3:
        """Return the minimum-image displacement r_j - r_i in a triclinic cell.

        The displacement is converted to fractional coordinates, each periodic
        component is shifted to [-0.5, 0.5), then converted back to Cartesian.
        """
        h_matrix = cell.box_vectors
        h_inv = _mat3_inverse(h_matrix)
        periodic = cell.periodic_axes

        dr = (r_j[0] - r_i[0], r_j[1] - r_i[1], r_j[2] - r_i[2])
        ds = _mat3_vec_mul(h_inv, dr)

        sx = ds[0] - round(ds[0]) if periodic[0] else ds[0]
        sy = ds[1] - round(ds[1]) if periodic[1] else ds[1]
        sz = ds[2] - round(ds[2]) if periodic[2] else ds[2]

        return _mat3_vec_mul(h_matrix, (sx, sy, sz))

    def minimum_image_distance(
        self,
        r_i: Vector3,
        r_j: Vector3,
        cell: SimulationCell,
    ) -> float:
        """Return the scalar minimum-image distance in a triclinic cell."""
        dx, dy, dz = self.minimum_image_displacement(r_i, r_j, cell)
        return sqrt(dx * dx + dy * dy + dz * dz)

    # -- state-level operations ---------------------------------------------

    def wrap_state(self, state: SimulationState) -> SimulationState:
        """Return a new state with positions wrapped into the triclinic cell."""
        if state.cell is None:
            return state

        new_positions = self.wrap_positions(state.particles.positions, state.cell)
        new_particles = state.particles.with_positions(new_positions)

        return SimulationState(
            units=state.units,
            particles=new_particles,
            thermodynamics=state.thermodynamics,
            provenance=state.provenance,
            cell=state.cell,
            time=state.time,
            step=state.step,
            potential_energy=state.potential_energy,
            observables=state.observables,
        )

    def pairwise_distances(
        self, state: SimulationState,
    ) -> tuple[tuple[int, int, float], ...]:
        """Return all unique pairwise minimum-image distances (triclinic)."""
        positions = state.particles.positions
        n = state.particle_count
        pairs: list[tuple[int, int, float]] = []

        if state.cell is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    d = self.minimum_image_distance(
                        positions[i], positions[j], state.cell,
                    )
                    pairs.append((i, j, d))
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    dx = positions[j][0] - positions[i][0]
                    dy = positions[j][1] - positions[i][1]
                    dz = positions[j][2] - positions[i][2]
                    pairs.append((i, j, sqrt(dx * dx + dy * dy + dz * dz)))

        return tuple(pairs)
