"""Holonomic constraint solvers for fixed bond-length maintenance.

SHAKE (Ryckaert, Ciccotti & Berendsen 1977) and LINCS (Hess, Bekker,
Berendsen & Fraaije 1997) iteratively correct positions after an
unconstrained integration step so that specified inter-particle distances
are satisfied.  Removing these fast vibrational degrees of freedom allows
significantly larger time-steps in coarse-grained molecular dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from core.exceptions import ContractValidationError
from core.state import ParticleState
from core.types import FrozenMetadata, Vector3, VectorTuple


# ---------------------------------------------------------------------------
# Constraint definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DistanceConstraint:
    """A single holonomic bond-length constraint between two particles."""

    particle_a: int
    particle_b: int
    target_distance: float

    def validate(self) -> tuple[str, ...]:
        """Return a tuple of human-readable issues (empty when valid)."""
        issues: list[str] = []
        if self.particle_a < 0:
            issues.append("particle_a index must be non-negative.")
        if self.particle_b < 0:
            issues.append("particle_b index must be non-negative.")
        if self.particle_a == self.particle_b:
            issues.append("particle_a and particle_b must differ.")
        if self.target_distance <= 0.0:
            issues.append("target_distance must be strictly positive.")
        return tuple(issues)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ConstraintResult:
    """Output produced by a constraint solver after position correction."""

    positions: VectorTuple
    velocities: VectorTuple
    converged: bool
    iterations_used: int
    max_relative_error: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)


# ---------------------------------------------------------------------------
# Vector helpers (pure-Python, no external dependencies)
# ---------------------------------------------------------------------------

def _v3_sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _v3_dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _v3_scale(s: float, v: Vector3) -> Vector3:
    return (s * v[0], s * v[1], s * v[2])


def _v3_add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _v3_norm_sq(v: Vector3) -> float:
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]


# ---------------------------------------------------------------------------
# SHAKE solver
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SHAKESolver:
    """SHAKE constraint solver (Ryckaert, Ciccotti & Berendsen, 1977).

    Classification: [established]
    """

    tolerance: float = 1e-6
    max_iterations: int = 100
    name: str = "shake"
    classification: str = "[established]"

    def apply(
        self,
        old_positions: VectorTuple,
        new_positions: VectorTuple,
        masses: tuple[float, ...],
        constraints: tuple[DistanceConstraint, ...],
        *,
        time_step: float = 0.001,
    ) -> ConstraintResult:
        """Iteratively correct *new_positions* to satisfy all constraints.

        Parameters
        ----------
        old_positions:
            Positions **before** the unconstrained integration step.
        new_positions:
            Positions **after** the unconstrained integration step (to be
            corrected in-place via mutable working copy).
        masses:
            Per-particle masses.
        constraints:
            Bond-length constraints to enforce.
        time_step:
            Integration time-step used to derive corrected velocities.

        Returns
        -------
        ConstraintResult
            Corrected positions, corrected velocities, convergence flag,
            iteration count, and worst relative error.
        """
        if not constraints:
            velocities = tuple(
                _v3_scale(1.0 / time_step, _v3_sub(new_positions[i], old_positions[i]))
                for i in range(len(new_positions))
            )
            return ConstraintResult(
                positions=new_positions,
                velocities=velocities,
                converged=True,
                iterations_used=0,
                max_relative_error=0.0,
                metadata=FrozenMetadata({"solver": "shake"}),
            )

        # Work with mutable lists for the iterative loop.
        pos: list[list[float]] = [[p[0], p[1], p[2]] for p in new_positions]

        converged = False
        iterations_used = 0
        max_rel_error = 0.0

        for iteration in range(1, self.max_iterations + 1):
            iterations_used = iteration
            max_rel_error = 0.0
            all_satisfied = True

            for con in constraints:
                ia, ib = con.particle_a, con.particle_b
                d0 = con.target_distance
                target_sq = d0 * d0

                # Current separation in the working (new) positions.
                rx = pos[ia][0] - pos[ib][0]
                ry = pos[ia][1] - pos[ib][1]
                rz = pos[ia][2] - pos[ib][2]
                current_sq = rx * rx + ry * ry + rz * rz

                rel_err = abs(current_sq - target_sq) / target_sq
                if rel_err > max_rel_error:
                    max_rel_error = rel_err

                if rel_err < self.tolerance:
                    continue

                all_satisfied = False

                # Reference direction from old positions.
                sx = old_positions[ia][0] - old_positions[ib][0]
                sy = old_positions[ia][1] - old_positions[ib][1]
                sz = old_positions[ia][2] - old_positions[ib][2]

                dot_rs = rx * sx + ry * sy + rz * sz
                inv_mass_sum = 1.0 / masses[ia] + 1.0 / masses[ib]
                lam = (current_sq - target_sq) / (2.0 * dot_rs * inv_mass_sum)

                factor_a = lam / masses[ia]
                factor_b = lam / masses[ib]
                pos[ia][0] -= factor_a * sx
                pos[ia][1] -= factor_a * sy
                pos[ia][2] -= factor_a * sz
                pos[ib][0] += factor_b * sx
                pos[ib][1] += factor_b * sy
                pos[ib][2] += factor_b * sz

            if all_satisfied:
                converged = True
                break

        # Build immutable corrected position tuple.
        corrected_positions: VectorTuple = tuple(
            (p[0], p[1], p[2]) for p in pos
        )

        # Derive corrected velocities from position displacement.
        inv_dt = 1.0 / time_step
        corrected_velocities: VectorTuple = tuple(
            (
                (pos[i][0] - old_positions[i][0]) * inv_dt,
                (pos[i][1] - old_positions[i][1]) * inv_dt,
                (pos[i][2] - old_positions[i][2]) * inv_dt,
            )
            for i in range(len(pos))
        )

        return ConstraintResult(
            positions=corrected_positions,
            velocities=corrected_velocities,
            converged=converged,
            iterations_used=iterations_used,
            max_relative_error=max_rel_error,
            metadata=FrozenMetadata({
                "solver": "shake",
                "tolerance": self.tolerance,
            }),
        )


# ---------------------------------------------------------------------------
# LINCS solver
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LINCSolver:
    """LINCS constraint solver (Hess, Bekker, Berendsen & Fraaije, 1997).

    Classification: [established]
    """

    order: int = 4
    warn_angle: float = 30.0
    name: str = "lincs"
    classification: str = "[established]"

    def apply(
        self,
        old_positions: VectorTuple,
        new_positions: VectorTuple,
        masses: tuple[float, ...],
        constraints: tuple[DistanceConstraint, ...],
        *,
        time_step: float = 0.001,
    ) -> ConstraintResult:
        """Apply the LINCS algorithm to correct *new_positions*.

        Parameters
        ----------
        old_positions:
            Positions before the unconstrained integration step.
        new_positions:
            Unconstrained positions after integration.
        masses:
            Per-particle masses.
        constraints:
            Bond-length constraints to enforce.
        time_step:
            Integration time-step for velocity derivation.

        Returns
        -------
        ConstraintResult
        """
        if not constraints:
            velocities = tuple(
                _v3_scale(1.0 / time_step, _v3_sub(new_positions[i], old_positions[i]))
                for i in range(len(new_positions))
            )
            return ConstraintResult(
                positions=new_positions,
                velocities=velocities,
                converged=True,
                iterations_used=0,
                max_relative_error=0.0,
                metadata=FrozenMetadata({"solver": "lincs"}),
            )

        # Mutable working copy.
        pos: list[list[float]] = [[p[0], p[1], p[2]] for p in new_positions]

        # Step 1 -- reference direction vectors B_n from old positions.
        directions: list[Vector3] = []
        for con in constraints:
            ia, ib = con.particle_a, con.particle_b
            d0 = con.target_distance
            dx = old_positions[ia][0] - old_positions[ib][0]
            dy = old_positions[ia][1] - old_positions[ib][1]
            dz = old_positions[ia][2] - old_positions[ib][2]
            inv_d0 = 1.0 / d0
            directions.append((dx * inv_d0, dy * inv_d0, dz * inv_d0))

        # Step 3-6 -- initial correction (iterated for coupled constraints).
        for _coupling_pass in range(3):
            for n, con in enumerate(constraints):
                ia, ib = con.particle_a, con.particle_b
                d0 = con.target_distance
                bx, by, bz = directions[n]

                # Projection of unconstrained bond onto reference direction.
                rx = pos[ia][0] - pos[ib][0]
                ry = pos[ia][1] - pos[ib][1]
                rz = pos[ia][2] - pos[ib][2]
                p_n = bx * rx + by * ry + bz * rz

                # Violation.
                l_n = d0 - p_n
                inv_mass_sum = 1.0 / masses[ia] + 1.0 / masses[ib]
                lam = l_n / inv_mass_sum

                fa = lam / masses[ia]
                fb = lam / masses[ib]
                pos[ia][0] += fa * bx
                pos[ia][1] += fa * by
                pos[ia][2] += fa * bz
                pos[ib][0] -= fb * bx
                pos[ib][1] -= fb * by
                pos[ib][2] -= fb * bz

        # Step 7 -- iterative correction using actual bond vectors.
        # This replaces the original LINCS rotation formula with a more
        # robust iterative approach that handles collinear and coupled
        # constraints correctly.
        for _k in range(self.order):
            for n, con in enumerate(constraints):
                ia, ib = con.particle_a, con.particle_b
                d0 = con.target_distance

                rx = pos[ia][0] - pos[ib][0]
                ry = pos[ia][1] - pos[ib][1]
                rz = pos[ia][2] - pos[ib][2]
                current_dist = sqrt(rx * rx + ry * ry + rz * rz)

                if current_dist < 1e-12:
                    continue

                # Direct length correction along current bond vector
                correction = (current_dist - d0) / current_dist
                inv_mass_sum = 1.0 / masses[ia] + 1.0 / masses[ib]
                scale_a = correction / (masses[ia] * inv_mass_sum)
                scale_b = correction / (masses[ib] * inv_mass_sum)

                pos[ia][0] -= scale_a * rx
                pos[ia][1] -= scale_a * ry
                pos[ia][2] -= scale_a * rz
                pos[ib][0] += scale_b * rx
                pos[ib][1] += scale_b * ry
                pos[ib][2] += scale_b * rz

        # Compute worst relative error.
        max_rel_error = 0.0
        for con in constraints:
            ia, ib = con.particle_a, con.particle_b
            d0 = con.target_distance
            dx = pos[ia][0] - pos[ib][0]
            dy = pos[ia][1] - pos[ib][1]
            dz = pos[ia][2] - pos[ib][2]
            actual = sqrt(dx * dx + dy * dy + dz * dz)
            rel_err = abs(actual - d0) / d0
            if rel_err > max_rel_error:
                max_rel_error = rel_err

        converged = max_rel_error < 1e-6
        total_iterations = 1 + self.order  # initial pass + rotation corrections

        # Build immutable corrected positions.
        corrected_positions: VectorTuple = tuple(
            (p[0], p[1], p[2]) for p in pos
        )

        # Derive corrected velocities.
        inv_dt = 1.0 / time_step
        corrected_velocities: VectorTuple = tuple(
            (
                (pos[i][0] - old_positions[i][0]) * inv_dt,
                (pos[i][1] - old_positions[i][1]) * inv_dt,
                (pos[i][2] - old_positions[i][2]) * inv_dt,
            )
            for i in range(len(pos))
        )

        return ConstraintResult(
            positions=corrected_positions,
            velocities=corrected_velocities,
            converged=converged,
            iterations_used=total_iterations,
            max_relative_error=max_rel_error,
            metadata=FrozenMetadata({
                "solver": "lincs",
                "order": self.order,
                "warn_angle": self.warn_angle,
            }),
        )


# ---------------------------------------------------------------------------
# Wrapper that applies constraints after any integrator step
# ---------------------------------------------------------------------------

from integrators.base import IntegratorStepResult, ForceEvaluator  # noqa: E402
from forcefields.base_forcefield import BaseForceField  # noqa: E402
from topology.system_topology import SystemTopology  # noqa: E402


@dataclass(frozen=True, slots=True)
class ConstrainedIntegratorWrapper:
    """Wraps an arbitrary integrator to apply constraint corrections.

    After the inner integrator produces an unconstrained step, the
    configured constraint solver (SHAKE or LINCS) adjusts positions and
    velocities to satisfy the supplied distance constraints.

    Classification: [adapted]
    """

    integrator: object
    constraint_solver: SHAKESolver | LINCSolver
    constraints: tuple[DistanceConstraint, ...]
    classification: str = "[adapted]"

    @property
    def name(self) -> str:
        return getattr(self.integrator, "name", "unknown")

    @property
    def time_step(self) -> float:
        return getattr(self.integrator, "time_step", 0.001)

    def step(
        self,
        state: object,
        topology: SystemTopology,
        forcefield: BaseForceField,
        force_evaluator: ForceEvaluator,
    ) -> IntegratorStepResult:
        """Advance state by one constrained integration step.

        1. Delegate to the inner integrator for an unconstrained step.
        2. Apply the constraint solver to correct positions/velocities.
        3. Return an ``IntegratorStepResult`` with corrected data.
        """
        # 1. Unconstrained step.
        inner_result: IntegratorStepResult = self.integrator.step(  # type: ignore[union-attr]
            state, topology, forcefield, force_evaluator,
        )

        if not self.constraints:
            return inner_result

        # Recover old positions from the incoming state.
        old_positions: VectorTuple = state.particles.positions  # type: ignore[union-attr]

        # 2. Constraint correction.
        constraint_result = self.constraint_solver.apply(
            old_positions=old_positions,
            new_positions=inner_result.particles.positions,
            masses=inner_result.particles.masses,
            constraints=self.constraints,
            time_step=self.time_step,
        )

        # 3. Build corrected ParticleState.
        corrected_particles = ParticleState(
            positions=constraint_result.positions,
            masses=inner_result.particles.masses,
            velocities=constraint_result.velocities,
            labels=inner_result.particles.labels,
        )

        return IntegratorStepResult(
            particles=corrected_particles,
            time=inner_result.time,
            step=inner_result.step,
            potential_energy=inner_result.potential_energy,
            observables=inner_result.observables,
            metadata=inner_result.metadata,
        )
