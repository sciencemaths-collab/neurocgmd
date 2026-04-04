"""Production bonded interaction terms: angles, proper dihedrals, and improper dihedrals."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, atan2, cos, pi, sin, sqrt

from core.exceptions import ContractValidationError
from core.types import FrozenMetadata, Vector3, VectorTuple


# ---------------------------------------------------------------------------
# Vector helpers (pure-Python, no NumPy dependency)
# ---------------------------------------------------------------------------

def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: Vector3) -> float:
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(v: Vector3, s: float) -> Vector3:
    return (v[0] * s, v[1] * s, v[2] * s)


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _neg(v: Vector3) -> Vector3:
    return (-v[0], -v[1], -v[2])


_ZERO: Vector3 = (0.0, 0.0, 0.0)

# Small value to guard against division by zero in degenerate geometries.
_ANGLE_EPSILON = 1.0e-12


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AngleParameter:
    """Harmonic angle parameter for a bead-type triple (a--b--c, b is central)."""

    bead_type_a: str
    bead_type_b: str
    bead_type_c: str
    equilibrium_angle: float  # radians
    force_constant: float  # kJ/(mol*rad^2)
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for label, value in (
            ("bead_type_a", self.bead_type_a),
            ("bead_type_b", self.bead_type_b),
            ("bead_type_c", self.bead_type_c),
        ):
            if not value.strip():
                issues.append(f"AngleParameter {label} must be a non-empty string.")
        if not (0.0 <= self.equilibrium_angle <= pi):
            issues.append(
                "AngleParameter equilibrium_angle must be in [0, pi] radians."
            )
        if self.force_constant < 0.0:
            issues.append(
                "AngleParameter force_constant must be non-negative."
            )
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class DihedralParameter:
    """Cosine-series proper dihedral parameter for a bead-type quadruple."""

    bead_type_a: str
    bead_type_b: str
    bead_type_c: str
    bead_type_d: str
    force_constant: float  # kJ/mol
    multiplicity: int = 1
    phase: float = 0.0  # radians
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for label, value in (
            ("bead_type_a", self.bead_type_a),
            ("bead_type_b", self.bead_type_b),
            ("bead_type_c", self.bead_type_c),
            ("bead_type_d", self.bead_type_d),
        ):
            if not value.strip():
                issues.append(f"DihedralParameter {label} must be a non-empty string.")
        if self.multiplicity < 1:
            issues.append("DihedralParameter multiplicity must be >= 1.")
        if self.force_constant < 0.0:
            issues.append("DihedralParameter force_constant must be non-negative.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ImproperDihedralParameter:
    """Improper (out-of-plane) dihedral parameter for a bead-type quadruple."""

    bead_type_a: str
    bead_type_b: str
    bead_type_c: str
    bead_type_d: str
    force_constant: float  # kJ/mol
    multiplicity: int = 1
    phase: float = 0.0  # radians
    harmonic: bool = True  # if True use k*(phi-phi0)^2 instead of cosine form
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for label, value in (
            ("bead_type_a", self.bead_type_a),
            ("bead_type_b", self.bead_type_b),
            ("bead_type_c", self.bead_type_c),
            ("bead_type_d", self.bead_type_d),
        ):
            if not value.strip():
                issues.append(
                    f"ImproperDihedralParameter {label} must be a non-empty string."
                )
        if self.multiplicity < 1:
            issues.append("ImproperDihedralParameter multiplicity must be >= 1.")
        if self.force_constant < 0.0:
            issues.append(
                "ImproperDihedralParameter force_constant must be non-negative."
            )
        return tuple(issues)


# ---------------------------------------------------------------------------
# Force models
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AngleForceModel:
    """Harmonic angle force evaluation for coarse-grained beads.

    Energy: E = 0.5 * k * (theta - theta_0)^2
    Classification: [established]
    """

    name: str = "harmonic_angle_force"
    classification: str = "[established]"

    # ---- public API -------------------------------------------------------

    def evaluate(
        self,
        positions: VectorTuple,
        angles: tuple[tuple[int, int, int, AngleParameter], ...],
    ) -> tuple[VectorTuple, float]:
        """Return (forces, total_energy) for all angle interactions.

        Each entry in *angles* is ``(i, j, k, param)`` where *j* is the
        central atom.
        """
        n_particles = len(positions)
        # Accumulate forces in a mutable list, then freeze.
        forces: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(n_particles)]
        total_energy = 0.0

        for i, j, k, param in angles:
            energy, fi, fj, fk = self._single_angle(
                positions[i], positions[j], positions[k], param
            )
            total_energy += energy
            for d in range(3):
                forces[i][d] += fi[d]
                forces[j][d] += fj[d]
                forces[k][d] += fk[d]

        frozen_forces: VectorTuple = tuple(
            (f[0], f[1], f[2]) for f in forces
        )
        return frozen_forces, total_energy

    # ---- internals --------------------------------------------------------

    @staticmethod
    def _single_angle(
        ri: Vector3, rj: Vector3, rk: Vector3, param: AngleParameter
    ) -> tuple[float, Vector3, Vector3, Vector3]:
        r_ji = _sub(ri, rj)
        r_jk = _sub(rk, rj)
        dist_ji = _norm(r_ji)
        dist_jk = _norm(r_jk)

        if dist_ji < _ANGLE_EPSILON or dist_jk < _ANGLE_EPSILON:
            return 0.0, _ZERO, _ZERO, _ZERO

        cos_theta = _dot(r_ji, r_jk) / (dist_ji * dist_jk)
        # Clamp to avoid numerical issues with acos.
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta = acos(cos_theta)
        sin_theta = sin(theta)

        if abs(sin_theta) < _ANGLE_EPSILON:
            # Nearly linear or folded -- gradient is ill-defined; skip.
            energy = 0.5 * param.force_constant * (theta - param.equilibrium_angle) ** 2
            return energy, _ZERO, _ZERO, _ZERO

        energy = 0.5 * param.force_constant * (theta - param.equilibrium_angle) ** 2

        # d(theta)/d(r_i) and d(theta)/d(r_k)
        inv_dist_ji = 1.0 / dist_ji
        inv_dist_jk = 1.0 / dist_jk
        unit_ji = _scale(r_ji, inv_dist_ji)
        unit_jk = _scale(r_jk, inv_dist_jk)
        inv_sin = 1.0 / sin_theta

        # d(theta)/d(r_i)
        dtheta_dri = _scale(
            _sub(_scale(unit_ji, cos_theta), unit_jk),
            inv_dist_ji * inv_sin,
        )
        # d(theta)/d(r_k)
        dtheta_drk = _scale(
            _sub(_scale(unit_jk, cos_theta), unit_ji),
            inv_dist_jk * inv_sin,
        )
        # d(theta)/d(r_j) = -(d(theta)/d(r_i) + d(theta)/d(r_k))
        dtheta_drj = _neg(_add(dtheta_dri, dtheta_drk))

        prefactor = -param.force_constant * (theta - param.equilibrium_angle)
        fi = _scale(dtheta_dri, prefactor)
        fj = _scale(dtheta_drj, prefactor)
        fk = _scale(dtheta_drk, prefactor)

        return energy, fi, fj, fk


@dataclass(frozen=True, slots=True)
class DihedralForceModel:
    """Cosine dihedral force evaluation for coarse-grained beads.

    Energy: E = k * (1 + cos(n*phi - phase))
    Classification: [established]
    """

    name: str = "cosine_dihedral_force"
    classification: str = "[established]"

    # ---- public API -------------------------------------------------------

    def evaluate(
        self,
        positions: VectorTuple,
        dihedrals: tuple[
            tuple[int, int, int, int, DihedralParameter | ImproperDihedralParameter],
            ...,
        ],
    ) -> tuple[VectorTuple, float]:
        """Return (forces, total_energy) for all dihedral interactions."""
        n_particles = len(positions)
        forces: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(n_particles)]
        total_energy = 0.0

        for i, j, k, l, param in dihedrals:
            energy, fi, fj, fk, fl = self._single_dihedral(
                positions[i], positions[j], positions[k], positions[l], param
            )
            total_energy += energy
            for d in range(3):
                forces[i][d] += fi[d]
                forces[j][d] += fj[d]
                forces[k][d] += fk[d]
                forces[l][d] += fl[d]

        frozen_forces: VectorTuple = tuple(
            (f[0], f[1], f[2]) for f in forces
        )
        return frozen_forces, total_energy

    # ---- internals --------------------------------------------------------

    @staticmethod
    def _single_dihedral(
        ri: Vector3,
        rj: Vector3,
        rk: Vector3,
        rl: Vector3,
        param: DihedralParameter | ImproperDihedralParameter,
    ) -> tuple[float, Vector3, Vector3, Vector3, Vector3]:
        # Bond vectors.
        b1 = _sub(rj, ri)
        b2 = _sub(rk, rj)
        b3 = _sub(rl, rk)

        # Normal vectors.
        n1 = _cross(b1, b2)
        n2 = _cross(b2, b3)

        norm_n1 = _norm(n1)
        norm_n2 = _norm(n2)
        norm_b2 = _norm(b2)

        if norm_n1 < _ANGLE_EPSILON or norm_n2 < _ANGLE_EPSILON or norm_b2 < _ANGLE_EPSILON:
            return 0.0, _ZERO, _ZERO, _ZERO, _ZERO

        # Dihedral angle via atan2.
        m1 = _cross(n1, _scale(b2, 1.0 / norm_b2))
        x = _dot(n1, n2)
        y = _dot(m1, n2)
        phi = atan2(y, x)

        # Energy.
        is_harmonic_improper = (
            isinstance(param, ImproperDihedralParameter) and param.harmonic
        )
        if is_harmonic_improper:
            delta = phi - param.phase
            # Wrap to [-pi, pi].
            while delta > pi:
                delta -= 2.0 * pi
            while delta < -pi:
                delta += 2.0 * pi
            energy = param.force_constant * delta * delta
            dE_dphi = 2.0 * param.force_constant * delta
        else:
            n_mult = param.multiplicity
            energy = param.force_constant * (1.0 + cos(n_mult * phi - param.phase))
            dE_dphi = -param.force_constant * n_mult * sin(n_mult * phi - param.phase)

        # Dihedral forces using the standard algorithm.
        # Reference: Bekker et al., J. Comput. Chem., 1995.
        inv_norm_n1_sq = 1.0 / max(_dot(n1, n1), _ANGLE_EPSILON)
        inv_norm_n2_sq = 1.0 / max(_dot(n2, n2), _ANGLE_EPSILON)

        # d(phi)/d(r_i) = -(|b2| / |n1|^2) * n1
        dphi_dri = _scale(n1, -norm_b2 * inv_norm_n1_sq)
        # d(phi)/d(r_l) = (|b2| / |n2|^2) * n2
        dphi_drl = _scale(n2, norm_b2 * inv_norm_n2_sq)

        # d(phi)/d(r_j) and d(phi)/d(r_k) use projection along bonds.
        dot_b1_b2 = _dot(b1, b2)
        dot_b3_b2 = _dot(b3, b2)
        norm_b2_sq = norm_b2 * norm_b2

        # d(phi)/d(r_j) = (dot(b1,b2)/|b2|^2 - 1)*dphi_dri - (dot(b3,b2)/|b2|^2)*dphi_drl
        dphi_drj = _sub(
            _scale(dphi_dri, dot_b1_b2 / norm_b2_sq - 1.0),
            _scale(dphi_drl, dot_b3_b2 / norm_b2_sq),
        )
        # d(phi)/d(r_k) = -(d(phi)/d(r_i) + d(phi)/d(r_j) + d(phi)/d(r_l))
        dphi_drk = _neg(_add(_add(dphi_dri, dphi_drj), dphi_drl))

        fi = _scale(dphi_dri, -dE_dphi)
        fj = _scale(dphi_drj, -dE_dphi)
        fk = _scale(dphi_drk, -dE_dphi)
        fl = _scale(dphi_drl, -dE_dphi)

        return energy, fi, fj, fk, fl
