"""Ewald summation and reaction-field electrostatics for periodic systems.

Implements the standard Ewald decomposition of Coulomb interactions into
real-space, reciprocal-space, and self-energy contributions.  All energies
are returned in kJ/mol and forces in kJ/(mol*nm), consistent with the
md_nano unit system used throughout NeuroCGMD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt, pi, exp, erfc, floor, ceil, cos, sin, log

from core.exceptions import ContractValidationError
from core.state import SimulationCell, SimulationState
from core.types import FrozenMetadata, Vector3, VectorTuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COULOMB_CONSTANT: float = 138.935458  # kJ*nm/(mol*e^2) -- 1/(4*pi*eps0) in MD units

_ZERO_VEC: Vector3 = (0.0, 0.0, 0.0)

# ---------------------------------------------------------------------------
# ChargeSet
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChargeSet:
    """Partial charges for every particle in elementary-charge units."""

    charges: tuple[float, ...]

    def validate(self) -> tuple[str, ...]:
        """Return a tuple of validation issue strings (empty when valid)."""
        issues: list[str] = []
        if len(self.charges) == 0:
            issues.append("ChargeSet must contain at least one charge.")
        for idx, q in enumerate(self.charges):
            if not isinstance(q, (int, float)) or isinstance(q, bool):
                issues.append(f"charges[{idx}] must be numeric; received {q!r}.")
        return tuple(issues)

    def total_charge(self) -> float:
        """Return the algebraic sum of all partial charges."""
        return sum(self.charges)

    def is_neutral(self, tolerance: float = 1e-6) -> bool:
        """Return True when the net charge is within *tolerance* of zero."""
        return abs(self.total_charge()) <= tolerance


# ---------------------------------------------------------------------------
# EwaldParameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EwaldParameters:
    """Tuning parameters for the Ewald summation decomposition."""

    alpha: float
    real_cutoff: float
    kmax: int = 5
    name: str = "ewald_summation"
    classification: str = "[established]"

    def __post_init__(self) -> None:
        issues: list[str] = []
        if self.alpha <= 0.0:
            issues.append("alpha must be positive.")
        if self.real_cutoff <= 0.0:
            issues.append("real_cutoff must be positive.")
        if self.kmax < 1:
            issues.append("kmax must be at least 1.")
        if issues:
            raise ContractValidationError("; ".join(issues))

    @classmethod
    def auto_from_cutoff(
        cls,
        cutoff: float,
        tolerance: float = 1e-5,
        max_box_length: float | None = None,
    ) -> EwaldParameters:
        """Derive optimal *alpha* and *kmax* from a real-space cutoff.

        Parameters
        ----------
        cutoff:
            Real-space cutoff in nm.
        tolerance:
            Target error tolerance (dimensionless).
        max_box_length:
            Largest box dimension in nm.  When ``None`` an estimate of
            ``3 * cutoff`` is used to compute *kmax*.
        """
        if cutoff <= 0.0:
            raise ContractValidationError("cutoff must be positive.")
        if tolerance <= 0.0 or tolerance >= 1.0:
            raise ContractValidationError("tolerance must be in (0, 1).")

        neg_log_tol = -log(tolerance)
        alpha = sqrt(neg_log_tol) / cutoff

        box_l = max_box_length if max_box_length is not None else 3.0 * cutoff
        kmax_raw = alpha * box_l * sqrt(neg_log_tol)
        kmax = min(int(ceil(kmax_raw)), 10)
        kmax = max(kmax, 1)

        return cls(alpha=alpha, real_cutoff=cutoff, kmax=kmax)


# ---------------------------------------------------------------------------
# EwaldResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EwaldResult:
    """Container for Ewald-decomposed energies, forces, and metadata."""

    real_energy: float
    reciprocal_energy: float
    self_energy: float
    total_energy: float
    real_forces: VectorTuple
    reciprocal_forces: VectorTuple
    self_forces: VectorTuple
    total_forces: VectorTuple
    metadata: FrozenMetadata


# ---------------------------------------------------------------------------
# Helper: vector arithmetic
# ---------------------------------------------------------------------------


def _vadd(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vsub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vscale(s: float, v: Vector3) -> Vector3:
    return (s * v[0], s * v[1], s * v[2])


def _vnorm(v: Vector3) -> float:
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _vdot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


# ---------------------------------------------------------------------------
# EwaldSummation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EwaldSummation:
    """Full Ewald summation for periodic Coulomb interactions.

    Decomposes the electrostatic energy into real-space, reciprocal-space,
    and self-correction terms.  Assumes an orthorhombic simulation cell.
    """

    params: EwaldParameters
    name: str = "ewald_summation"
    classification: str = "[established]"

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def evaluate(
        self,
        state: SimulationState,
        charges: ChargeSet,
        cell: SimulationCell | None = None,
    ) -> EwaldResult:
        """Compute Ewald electrostatic energy and forces.

        Parameters
        ----------
        state:
            Current simulation state (positions are read from
            ``state.particles.positions``).
        charges:
            Partial charges for every particle.
        cell:
            Override simulation cell.  Falls back to ``state.cell``.

        Returns
        -------
        EwaldResult
        """
        cell = cell if cell is not None else state.cell
        if cell is None:
            raise ContractValidationError(
                "Ewald summation requires a periodic simulation cell."
            )

        positions: VectorTuple = state.particles.positions
        n = state.particle_count
        q = charges.charges

        if len(q) != n:
            raise ContractValidationError(
                f"ChargeSet length ({len(q)}) must match particle count ({n})."
            )

        # Extract box lengths (orthorhombic assumption).
        bv = cell.box_vectors
        Lx, Ly, Lz = abs(bv[0][0]), abs(bv[1][1]), abs(bv[2][2])
        periodic = cell.periodic_axes
        volume = cell.volume()

        alpha = self.params.alpha
        rc = self.params.real_cutoff
        kmax = self.params.kmax

        # ---- Real-space ------------------------------------------------- #
        real_energy, real_forces = self._real_space(
            positions, q, n, Lx, Ly, Lz, periodic, alpha, rc,
        )

        # ---- Reciprocal-space ------------------------------------------- #
        recip_energy, recip_forces = self._reciprocal_space(
            positions, q, n, Lx, Ly, Lz, volume, alpha, kmax,
        )

        # ---- Self-energy correction ------------------------------------- #
        self_energy = -alpha / sqrt(pi) * COULOMB_CONSTANT * sum(qi * qi for qi in q)
        self_forces: VectorTuple = tuple(_ZERO_VEC for _ in range(n))

        # ---- Totals ----------------------------------------------------- #
        total_energy = real_energy + recip_energy + self_energy
        total_forces: VectorTuple = tuple(
            _vadd(_vadd(real_forces[i], recip_forces[i]), self_forces[i])
            for i in range(n)
        )

        metadata = FrozenMetadata({
            "alpha": alpha,
            "real_cutoff": rc,
            "kmax": kmax,
            "particle_count": n,
            "box_lengths": (Lx, Ly, Lz),
            "method": "ewald_summation",
        })

        return EwaldResult(
            real_energy=real_energy,
            reciprocal_energy=recip_energy,
            self_energy=self_energy,
            total_energy=total_energy,
            real_forces=real_forces,
            reciprocal_forces=recip_forces,
            self_forces=self_forces,
            total_forces=total_forces,
            metadata=metadata,
        )

    # --------------------------------------------------------------------- #
    # Real-space contribution
    # --------------------------------------------------------------------- #

    @staticmethod
    def _real_space(
        positions: VectorTuple,
        q: tuple[float, ...],
        n: int,
        Lx: float,
        Ly: float,
        Lz: float,
        periodic: tuple[bool, bool, bool],
        alpha: float,
        rc: float,
    ) -> tuple[float, VectorTuple]:
        energy = 0.0
        forces_accum: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(n)]
        rc2 = rc * rc
        two_alpha_over_sqrt_pi = 2.0 * alpha / sqrt(pi)

        for i in range(n):
            qi = q[i]
            xi, yi, zi = positions[i]
            for j in range(i + 1, n):
                qj = q[j]
                dx = xi - positions[j][0]
                dy = yi - positions[j][1]
                dz = zi - positions[j][2]

                # Minimum image convention
                if periodic[0]:
                    dx -= Lx * round(dx / Lx)
                if periodic[1]:
                    dy -= Ly * round(dy / Ly)
                if periodic[2]:
                    dz -= Lz * round(dz / Lz)

                r2 = dx * dx + dy * dy + dz * dz
                if r2 >= rc2 or r2 == 0.0:
                    continue

                r = sqrt(r2)
                ar = alpha * r
                erfc_val = erfc(ar)
                qq = qi * qj

                # Energy: COULOMB * q_i * q_j * erfc(alpha*r) / r
                energy += qq * erfc_val / r

                # Force magnitude (scalar along r_hat, positive = repulsive)
                force_scalar = (
                    COULOMB_CONSTANT * qq
                    * (erfc_val / r2 + two_alpha_over_sqrt_pi * exp(-ar * ar) / r)
                )
                fx = force_scalar * dx / r
                fy = force_scalar * dy / r
                fz = force_scalar * dz / r

                forces_accum[i][0] += fx
                forces_accum[i][1] += fy
                forces_accum[i][2] += fz
                forces_accum[j][0] -= fx
                forces_accum[j][1] -= fy
                forces_accum[j][2] -= fz

        energy *= COULOMB_CONSTANT
        forces: VectorTuple = tuple(
            (f[0], f[1], f[2]) for f in forces_accum
        )
        return energy, forces

    # --------------------------------------------------------------------- #
    # Reciprocal-space contribution
    # --------------------------------------------------------------------- #

    @staticmethod
    def _reciprocal_space(
        positions: VectorTuple,
        q: tuple[float, ...],
        n: int,
        Lx: float,
        Ly: float,
        Lz: float,
        volume: float,
        alpha: float,
        kmax: int,
    ) -> tuple[float, VectorTuple]:
        energy = 0.0
        forces_accum: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(n)]
        two_pi = 2.0 * pi
        inv_4alpha2 = 1.0 / (4.0 * alpha * alpha)
        prefactor = COULOMB_CONSTANT / volume  # common prefactor

        for kx_idx in range(-kmax, kmax + 1):
            kx_component = two_pi * kx_idx / Lx
            for ky_idx in range(-kmax, kmax + 1):
                ky_component = two_pi * ky_idx / Ly
                for kz_idx in range(-kmax, kmax + 1):
                    if kx_idx == 0 and ky_idx == 0 and kz_idx == 0:
                        continue

                    kz_component = two_pi * kz_idx / Lz
                    k2 = (
                        kx_component * kx_component
                        + ky_component * ky_component
                        + kz_component * kz_component
                    )

                    # Gaussian damping factor
                    gauss = exp(-k2 * inv_4alpha2)
                    coeff = 4.0 * pi / k2 * gauss  # per-k prefactor (excl. V)

                    # Structure factor: sum_i q_i exp(i k.r_i)
                    S_cos = 0.0
                    S_sin = 0.0
                    # Pre-compute k.r for each particle
                    kdotr: list[float] = []
                    for idx in range(n):
                        kr = (
                            kx_component * positions[idx][0]
                            + ky_component * positions[idx][1]
                            + kz_component * positions[idx][2]
                        )
                        kdotr.append(kr)
                        S_cos += q[idx] * cos(kr)
                        S_sin += q[idx] * sin(kr)

                    S2 = S_cos * S_cos + S_sin * S_sin

                    # Energy contribution
                    energy += coeff / volume * S2

                    # Force on particle j
                    force_coeff = prefactor * coeff
                    for j in range(n):
                        # dE/dr_j component
                        scalar = (
                            force_coeff
                            * q[j]
                            * (S_sin * cos(kdotr[j]) - S_cos * sin(kdotr[j]))
                        )
                        forces_accum[j][0] += scalar * kx_component
                        forces_accum[j][1] += scalar * ky_component
                        forces_accum[j][2] += scalar * kz_component

        energy *= COULOMB_CONSTANT
        forces: VectorTuple = tuple(
            (f[0], f[1], f[2]) for f in forces_accum
        )
        return energy, forces


# ---------------------------------------------------------------------------
# ReactionFieldElectrostatics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReactionFieldElectrostatics:
    """Reaction-field electrostatics with a dielectric continuum beyond the cutoff.

    A simpler alternative to Ewald summation suitable for non-periodic or
    truncated-electrostatic simulations.  The surrounding medium is treated
    as a uniform dielectric with permittivity *dielectric*.
    """

    dielectric: float = 78.5
    cutoff: float = 1.2
    name: str = "reaction_field"
    classification: str = "[established]"

    def __post_init__(self) -> None:
        issues: list[str] = []
        if self.dielectric <= 0.0:
            issues.append("dielectric must be positive.")
        if self.cutoff <= 0.0:
            issues.append("cutoff must be positive.")
        if issues:
            raise ContractValidationError("; ".join(issues))

    def evaluate(
        self,
        state: SimulationState,
        charges: ChargeSet,
    ) -> EwaldResult:
        """Evaluate reaction-field electrostatics.

        Returns an :class:`EwaldResult` for interface consistency.  The
        reciprocal and self terms are zero.
        """
        positions: VectorTuple = state.particles.positions
        n = state.particle_count
        q = charges.charges

        if len(q) != n:
            raise ContractValidationError(
                f"ChargeSet length ({len(q)}) must match particle count ({n})."
            )

        eps = self.dielectric
        rc = self.cutoff
        rc3 = rc * rc * rc

        k_rf = (eps - 1.0) / (2.0 * eps + 1.0) / rc3
        c_rf = 3.0 * eps / (2.0 * eps + 1.0) / rc

        energy = 0.0
        forces_accum: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(n)]
        rc2 = rc * rc

        # Apply minimum image if cell is present
        cell = state.cell
        has_cell = cell is not None
        if has_cell:
            bv = cell.box_vectors
            Lx = abs(bv[0][0])
            Ly = abs(bv[1][1])
            Lz = abs(bv[2][2])
            periodic = cell.periodic_axes
        else:
            Lx = Ly = Lz = 0.0
            periodic = (False, False, False)

        for i in range(n):
            qi = q[i]
            xi, yi, zi = positions[i]
            for j in range(i + 1, n):
                qj = q[j]
                dx = xi - positions[j][0]
                dy = yi - positions[j][1]
                dz = zi - positions[j][2]

                if has_cell:
                    if periodic[0]:
                        dx -= Lx * round(dx / Lx)
                    if periodic[1]:
                        dy -= Ly * round(dy / Ly)
                    if periodic[2]:
                        dz -= Lz * round(dz / Lz)

                r2 = dx * dx + dy * dy + dz * dz
                if r2 >= rc2 or r2 == 0.0:
                    continue

                r = sqrt(r2)
                qq = qi * qj

                # E = C * q_i * q_j * [1/r + k_rf * r^2 - c_rf]
                e_pair = COULOMB_CONSTANT * qq * (1.0 / r + k_rf * r2 - c_rf)
                energy += e_pair

                # F = C * q_i * q_j * [1/r^2 - 2*k_rf*r] * r_hat
                force_scalar = COULOMB_CONSTANT * qq * (1.0 / r2 - 2.0 * k_rf * r)
                fx = force_scalar * dx / r
                fy = force_scalar * dy / r
                fz = force_scalar * dz / r

                forces_accum[i][0] += fx
                forces_accum[i][1] += fy
                forces_accum[i][2] += fz
                forces_accum[j][0] -= fx
                forces_accum[j][1] -= fy
                forces_accum[j][2] -= fz

        zero_forces: VectorTuple = tuple(_ZERO_VEC for _ in range(n))
        total_forces: VectorTuple = tuple(
            (f[0], f[1], f[2]) for f in forces_accum
        )

        metadata = FrozenMetadata({
            "dielectric": eps,
            "cutoff": rc,
            "k_rf": k_rf,
            "c_rf": c_rf,
            "particle_count": n,
            "method": "reaction_field",
        })

        return EwaldResult(
            real_energy=energy,
            reciprocal_energy=0.0,
            self_energy=0.0,
            total_energy=energy,
            real_forces=total_forces,
            reciprocal_forces=zero_forces,
            self_forces=zero_forces,
            total_forces=total_forces,
            metadata=metadata,
        )
