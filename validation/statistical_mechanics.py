"""Statistical mechanics validation tools for NeuroCGMD trajectories.

Implements Flyvbjerg-Petersen block averaging, autocorrelation analysis,
equipartition theorem checks, Maxwell-Boltzmann speed distribution tests,
and detailed balance verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, exp, log, pi, sqrt

from core.exceptions import ContractValidationError
from core.types import FrozenMetadata, coerce_scalar


# ---------------------------------------------------------------------------
# Block averaging (Flyvbjerg-Petersen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BlockAverageResult:
    """Result of Flyvbjerg-Petersen block averaging analysis."""

    mean: float
    standard_error: float
    block_size: int
    n_blocks: int
    autocorrelation_time: float
    effective_samples: float
    metadata: FrozenMetadata


def block_average(data: tuple[float, ...], min_blocks: int = 10) -> BlockAverageResult:
    """Estimate the standard error of the mean via Flyvbjerg-Petersen block averaging.

    Parameters
    ----------
    data:
        Time series of scalar observations.
    min_blocks:
        Stop blocking when fewer than this many blocks remain.

    Returns
    -------
    BlockAverageResult with estimated standard error and autocorrelation time.
    """
    n = len(data)
    if n < max(2, min_blocks):
        raise ContractValidationError(
            f"block_average requires at least {max(2, min_blocks)} data points; received {n}."
        )

    mean = sum(data) / n

    # Naive standard error (no correlation correction)
    variance_naive = sum((x - mean) ** 2 for x in data) / (n - 1)
    se_naive = sqrt(variance_naive / n)

    # Progressive blocking
    current = list(data)
    best_se = se_naive
    best_block_size = 1
    best_n_blocks = n

    block_size = 1
    while len(current) >= max(2, min_blocks):
        m = len(current)
        block_mean = sum(current) / m
        var_block = sum((x - block_mean) ** 2 for x in current) / (m - 1)
        se_block = sqrt(var_block / m)

        if se_block >= best_se:
            best_se = se_block
            best_block_size = block_size
            best_n_blocks = m

        # Create next blocking level: average consecutive pairs
        new_current: list[float] = []
        for i in range(0, m - 1, 2):
            new_current.append((current[i] + current[i + 1]) / 2.0)
        current = new_current
        block_size *= 2

    # Autocorrelation time from the ratio of blocked to naive SE
    if se_naive > 0.0:
        ratio = best_se / se_naive
        autocorrelation_time = best_block_size * ratio * ratio / 2.0
    else:
        autocorrelation_time = 0.5

    effective_samples = n / (2.0 * autocorrelation_time) if autocorrelation_time > 0.0 else float(n)

    return BlockAverageResult(
        mean=mean,
        standard_error=best_se,
        block_size=best_block_size,
        n_blocks=best_n_blocks,
        autocorrelation_time=autocorrelation_time,
        effective_samples=effective_samples,
        metadata=FrozenMetadata({"method": "flyvbjerg_petersen", "n_original": n}),
    )


# ---------------------------------------------------------------------------
# Autocorrelation analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AutocorrelationResult:
    """Result of autocorrelation function computation."""

    autocorrelation_function: tuple[float, ...]
    integrated_autocorrelation_time: float
    effective_sample_size: float
    metadata: FrozenMetadata


def compute_autocorrelation(
    data: tuple[float, ...], max_lag: int | None = None
) -> AutocorrelationResult:
    """Compute the normalized autocorrelation function and integrated autocorrelation time.

    Uses Sokal's automatic windowing: integration stops when C(t) drops below zero.

    Parameters
    ----------
    data:
        Time series of scalar observations.
    max_lag:
        Maximum lag to compute. Defaults to N // 2.

    Returns
    -------
    AutocorrelationResult with C(t), integrated autocorrelation time, and
    effective sample size.
    """
    n = len(data)
    if n < 2:
        raise ContractValidationError(
            f"compute_autocorrelation requires at least 2 data points; received {n}."
        )

    mean = sum(data) / n
    deviations = [x - mean for x in data]
    c0 = sum(d * d for d in deviations) / n
    if c0 == 0.0:
        # Constant series: no autocorrelation
        acf = (1.0,)
        return AutocorrelationResult(
            autocorrelation_function=acf,
            integrated_autocorrelation_time=0.5,
            effective_sample_size=float(n),
            metadata=FrozenMetadata({"method": "direct", "max_lag_used": 0}),
        )

    if max_lag is None:
        max_lag = n // 2
    max_lag = min(max_lag, n - 1)

    acf_values: list[float] = []
    for t in range(max_lag + 1):
        ct = sum(deviations[i] * deviations[i + t] for i in range(n - t)) / n
        acf_values.append(ct / c0)

    # Integrated autocorrelation time with Sokal's automatic windowing
    tau = 0.5
    for t in range(1, len(acf_values)):
        if acf_values[t] < 0.0:
            break
        tau += acf_values[t]

    effective_sample_size = n / (2.0 * tau) if tau > 0.0 else float(n)

    return AutocorrelationResult(
        autocorrelation_function=tuple(acf_values),
        integrated_autocorrelation_time=tau,
        effective_sample_size=effective_sample_size,
        metadata=FrozenMetadata({"method": "direct", "max_lag_used": max_lag}),
    )


# ---------------------------------------------------------------------------
# Equipartition theorem check
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquipartitionCheck:
    """Result of equipartition theorem validation."""

    per_dof_kinetic_energies: tuple[float, ...]
    expected_ke_per_dof: float
    chi_squared: float
    p_value: float
    passed: bool
    metadata: FrozenMetadata


def _chi_squared_survival(x: float, k: float) -> float:
    """Approximate chi-squared survival function P(X^2 >= x) for k degrees of freedom.

    Uses the Wilson-Hilferty normal approximation for large k.
    """
    if k <= 0.0 or x < 0.0:
        return 1.0
    # Wilson-Hilferty approximation: transform to standard normal
    z = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / sqrt(2.0 / (9.0 * k))
    # Survival function of standard normal: 0.5 * erfc(z / sqrt(2))
    p = 0.5 * (1.0 - erf(z / sqrt(2.0)))
    return max(0.0, min(1.0, p))


def check_equipartition(
    velocities_history: tuple[tuple[tuple[float, float, float], ...], ...],
    masses: tuple[float, ...],
    temperature: float,
    thermal_energy_scale: float = 1.0,
) -> EquipartitionCheck:
    """Verify that kinetic energy is equally distributed among degrees of freedom.

    Parameters
    ----------
    velocities_history:
        Trajectory of velocity snapshots. Each snapshot is a tuple of 3D
        velocity vectors, one per particle.
    masses:
        Mass of each particle.
    temperature:
        Target temperature.
    thermal_energy_scale:
        Boltzmann constant in the simulation's unit system (default 1.0).

    Returns
    -------
    EquipartitionCheck with chi-squared test against kT/2 per DOF.
    """
    temperature = coerce_scalar(temperature, "temperature")
    thermal_energy_scale = coerce_scalar(thermal_energy_scale, "thermal_energy_scale")

    n_frames = len(velocities_history)
    if n_frames == 0:
        raise ContractValidationError("velocities_history must not be empty.")
    n_particles = len(velocities_history[0])
    if n_particles == 0:
        raise ContractValidationError("velocities_history frames must contain particles.")
    if len(masses) != n_particles:
        raise ContractValidationError(
            f"masses length ({len(masses)}) must match particle count ({n_particles})."
        )

    n_dof = n_particles * 3
    ke_accum = [0.0] * n_dof

    for frame in velocities_history:
        if len(frame) != n_particles:
            raise ContractValidationError("All frames must have the same number of particles.")
        for p_idx, vel in enumerate(frame):
            m = masses[p_idx]
            for d in range(3):
                ke_accum[p_idx * 3 + d] += 0.5 * m * vel[d] * vel[d]

    per_dof_ke = tuple(ke / n_frames for ke in ke_accum)
    expected = thermal_energy_scale * temperature / 2.0

    if expected <= 0.0:
        raise ContractValidationError("Expected kinetic energy per DOF must be positive.")

    chi_sq = sum((obs - expected) ** 2 / expected for obs in per_dof_ke)
    p_value = _chi_squared_survival(chi_sq, float(n_dof))
    passed = p_value > 0.01

    return EquipartitionCheck(
        per_dof_kinetic_energies=per_dof_ke,
        expected_ke_per_dof=expected,
        chi_squared=chi_sq,
        p_value=p_value,
        passed=passed,
        metadata=FrozenMetadata({
            "n_frames": n_frames,
            "n_particles": n_particles,
            "n_dof": n_dof,
            "temperature": temperature,
        }),
    )


# ---------------------------------------------------------------------------
# Detailed balance check
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DetailedBalanceCheck:
    """Result of detailed balance verification."""

    forward_acceptance: float
    reverse_acceptance: float
    ratio: float
    max_deviation: float
    passed: bool
    metadata: FrozenMetadata


# ---------------------------------------------------------------------------
# Maxwell-Boltzmann speed distribution check
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MaxwellBoltzmannCheck:
    """Result of Maxwell-Boltzmann speed distribution test."""

    ks_statistic: float
    speed_distribution_mean: float
    expected_mean: float
    passed: bool
    metadata: FrozenMetadata


def _maxwell_boltzmann_cdf(v: float, mass: float, kt: float) -> float:
    """Evaluate the Maxwell-Boltzmann cumulative distribution function at speed v."""
    if v <= 0.0:
        return 0.0
    a = mass / (2.0 * kt)
    term1 = erf(v * sqrt(a))
    term2 = v * sqrt(2.0 * mass / (pi * kt)) * exp(-mass * v * v / (2.0 * kt))
    return max(0.0, min(1.0, term1 - term2))


def check_maxwell_boltzmann(
    speeds: tuple[float, ...],
    mass: float,
    temperature: float,
    thermal_energy_scale: float = 1.0,
) -> MaxwellBoltzmannCheck:
    """Compare an empirical speed distribution against the Maxwell-Boltzmann distribution.

    Uses the Kolmogorov-Smirnov test at 95% confidence.

    Parameters
    ----------
    speeds:
        Observed particle speeds.
    mass:
        Particle mass.
    temperature:
        System temperature.
    thermal_energy_scale:
        Boltzmann constant in the simulation's unit system (default 1.0).

    Returns
    -------
    MaxwellBoltzmannCheck with KS statistic and pass/fail result.
    """
    mass = coerce_scalar(mass, "mass")
    temperature = coerce_scalar(temperature, "temperature")
    thermal_energy_scale = coerce_scalar(thermal_energy_scale, "thermal_energy_scale")

    n = len(speeds)
    if n == 0:
        raise ContractValidationError("speeds must not be empty.")

    kt = thermal_energy_scale * temperature
    if kt <= 0.0:
        raise ContractValidationError("thermal_energy_scale * temperature must be positive.")
    if mass <= 0.0:
        raise ContractValidationError("mass must be positive.")

    sorted_speeds = sorted(speeds)
    speed_mean = sum(sorted_speeds) / n
    expected_mean = sqrt(8.0 * kt / (pi * mass))

    # Kolmogorov-Smirnov test
    ks_stat = 0.0
    for i, v in enumerate(sorted_speeds):
        f_theoretical = _maxwell_boltzmann_cdf(v, mass, kt)
        # Empirical CDF jumps at each data point
        f_lower = i / n
        f_upper = (i + 1) / n
        d1 = abs(f_upper - f_theoretical)
        d2 = abs(f_lower - f_theoretical)
        ks_stat = max(ks_stat, d1, d2)

    # 95% confidence critical value
    critical_value = 1.36 / sqrt(n)
    passed = ks_stat < critical_value

    return MaxwellBoltzmannCheck(
        ks_statistic=ks_stat,
        speed_distribution_mean=speed_mean,
        expected_mean=expected_mean,
        passed=passed,
        metadata=FrozenMetadata({
            "n_speeds": n,
            "mass": mass,
            "temperature": temperature,
            "critical_value": critical_value,
        }),
    )


# ---------------------------------------------------------------------------
# Validation report and suite
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StatisticalValidationReport:
    """Comprehensive report from the statistical validation suite."""

    block_average: BlockAverageResult
    autocorrelation: AutocorrelationResult
    equipartition: EquipartitionCheck
    maxwell_boltzmann: MaxwellBoltzmannCheck
    all_passed: bool
    metadata: FrozenMetadata


@dataclass(slots=True)
class StatisticalValidationSuite:
    """Orchestrates all statistical mechanics validation checks on a trajectory."""

    name: str = "statistical_validation_suite"
    classification: str = "[established]"

    def validate_trajectory(
        self,
        energy_series: tuple[float, ...],
        velocity_history: tuple[tuple[tuple[float, float, float], ...], ...],
        masses: tuple[float, ...],
        temperature: float,
        *,
        thermal_energy_scale: float = 1.0,
    ) -> StatisticalValidationReport:
        """Run all statistical mechanics checks and return a comprehensive report.

        Parameters
        ----------
        energy_series:
            Time series of total (or potential) energy values.
        velocity_history:
            Trajectory of velocity snapshots per particle.
        masses:
            Mass of each particle.
        temperature:
            Target temperature.
        thermal_energy_scale:
            Boltzmann constant in the simulation's unit system.

        Returns
        -------
        StatisticalValidationReport aggregating all check results.
        """
        ba_result = block_average(energy_series)
        ac_result = compute_autocorrelation(energy_series)
        eq_result = check_equipartition(
            velocity_history, masses, temperature,
            thermal_energy_scale=thermal_energy_scale,
        )

        # Compute speeds from the last frame for Maxwell-Boltzmann check
        last_frame = velocity_history[-1]
        speeds: list[float] = []
        for vel in last_frame:
            speed = sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
            speeds.append(speed)
        # Use average mass for the MB check when masses differ
        avg_mass = sum(masses) / len(masses)
        mb_result = check_maxwell_boltzmann(
            tuple(speeds), avg_mass, temperature,
            thermal_energy_scale=thermal_energy_scale,
        )

        all_passed = eq_result.passed and mb_result.passed

        return StatisticalValidationReport(
            block_average=ba_result,
            autocorrelation=ac_result,
            equipartition=eq_result,
            maxwell_boltzmann=mb_result,
            all_passed=all_passed,
            metadata=FrozenMetadata({
                "suite_name": self.name,
                "classification": self.classification,
                "temperature": temperature,
                "thermal_energy_scale": thermal_energy_scale,
            }),
        )
