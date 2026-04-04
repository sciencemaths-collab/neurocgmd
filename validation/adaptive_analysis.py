"""Adaptive analysis engine for NeuroCGMD simulations.

Automatically computes key observables (PMF, binding energy, RMSD, RDF,
autocorrelation) during and after simulation without requiring explicit
user requests.  This is the intelligence layer that watches the simulation
and produces comprehensive results automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt, pi, exp, log

from core.exceptions import ContractValidationError
from core.state import SimulationState
from core.types import FrozenMetadata, Vector3, VectorTuple


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _distance_sq(a: Vector3, b: Vector3) -> float:
    """Squared Euclidean distance between two 3-D points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def _distance(a: Vector3, b: Vector3) -> float:
    """Euclidean distance between two 3-D points."""
    return sqrt(_distance_sq(a, b))


def _minimum_image_distance(
    a: Vector3, b: Vector3, box_lengths: tuple[float, float, float]
) -> float:
    """Distance under minimum image convention for an orthorhombic cell."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    lx, ly, lz = box_lengths
    dx -= lx * round(dx / lx)
    dy -= ly * round(dy / ly)
    dz -= lz * round(dz / lz)
    return sqrt(dx * dx + dy * dy + dz * dz)


def _box_lengths_from_cell(cell) -> tuple[float, float, float] | None:
    """Extract orthorhombic box lengths from a SimulationCell."""
    if cell is None:
        return None
    bv = cell.box_vectors
    return (abs(bv[0][0]), abs(bv[1][1]), abs(bv[2][2]))


def _center_of_mass_subset(
    positions: VectorTuple, masses: tuple[float, ...], indices: tuple[int, ...]
) -> Vector3:
    """Compute the center of mass for a subset of particles."""
    total_mass = 0.0
    cx = cy = cz = 0.0
    for i in indices:
        m = masses[i]
        total_mass += m
        cx += m * positions[i][0]
        cy += m * positions[i][1]
        cz += m * positions[i][2]
    if total_mass == 0.0:
        raise ContractValidationError("Subset total mass is zero.")
    return (cx / total_mass, cy / total_mass, cz / total_mass)


# ---------------------------------------------------------------------------
# RMSDTracker
# ---------------------------------------------------------------------------

_MAX_POSITION_FRAMES = 200


@dataclass(slots=True)
class RMSDTracker:
    """Tracks RMSD and per-particle RMSF over the course of a simulation."""

    reference_positions: VectorTuple
    _history: list[tuple[int, float]] = field(default_factory=list)
    _position_frames: list[VectorTuple] = field(default_factory=list)

    def compute_rmsd(self, positions: VectorTuple) -> float:
        """Compute RMSD between *positions* and the stored reference.

        RMSD = sqrt(1/N * sum(|r_i - r_ref_i|^2))
        """
        n = len(self.reference_positions)
        if len(positions) != n:
            raise ContractValidationError(
                f"Position count mismatch: expected {n}, got {len(positions)}."
            )
        total = 0.0
        for i in range(n):
            total += _distance_sq(positions[i], self.reference_positions[i])
        return sqrt(total / n)

    def record(self, step: int, state: SimulationState) -> None:
        """Compute RMSD for *state* and store the result."""
        positions = state.particles.positions
        rmsd = self.compute_rmsd(positions)
        self._history.append((step, rmsd))
        # Keep the last _MAX_POSITION_FRAMES frames for RMSF calculation.
        if len(self._position_frames) >= _MAX_POSITION_FRAMES:
            self._position_frames.pop(0)
        self._position_frames.append(positions)

    def rmsf_per_particle(self) -> tuple[float, ...]:
        """Root-mean-square fluctuation per particle from recorded frames.

        RMSF_i = sqrt(<|r_i - <r_i>|^2>)
        """
        n_frames = len(self._position_frames)
        if n_frames == 0:
            return ()
        n_particles = len(self._position_frames[0])

        # Compute mean position per particle.
        mean_x = [0.0] * n_particles
        mean_y = [0.0] * n_particles
        mean_z = [0.0] * n_particles
        for frame in self._position_frames:
            for i in range(n_particles):
                mean_x[i] += frame[i][0]
                mean_y[i] += frame[i][1]
                mean_z[i] += frame[i][2]
        inv_n = 1.0 / n_frames
        for i in range(n_particles):
            mean_x[i] *= inv_n
            mean_y[i] *= inv_n
            mean_z[i] *= inv_n

        # Compute mean squared fluctuation per particle.
        rmsf: list[float] = [0.0] * n_particles
        for frame in self._position_frames:
            for i in range(n_particles):
                dx = frame[i][0] - mean_x[i]
                dy = frame[i][1] - mean_y[i]
                dz = frame[i][2] - mean_z[i]
                rmsf[i] += dx * dx + dy * dy + dz * dz
        return tuple(sqrt(v * inv_n) for v in rmsf)

    def summary(self) -> dict:
        """Return summary statistics: mean, std, min, max RMSD and per-particle RMSF."""
        if not self._history:
            return {
                "mean_rmsd": 0.0,
                "std_rmsd": 0.0,
                "min_rmsd": 0.0,
                "max_rmsd": 0.0,
                "rmsf_per_particle": (),
            }
        values = [v for _, v in self._history]
        n = len(values)
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
        return {
            "mean_rmsd": mean,
            "std_rmsd": sqrt(var),
            "min_rmsd": min(values),
            "max_rmsd": max(values),
            "rmsf_per_particle": self.rmsf_per_particle(),
        }


# ---------------------------------------------------------------------------
# RDFCalculator
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RDFCalculator:
    """Accumulates pair distance histograms and computes g(r)."""

    cutoff: float = 2.0
    n_bins: int = 100
    _histogram: list[int] = field(default_factory=list)
    _n_frames: int = 0
    _n_particles: int = 0
    _volume: float = 0.0

    def __post_init__(self) -> None:
        if not self._histogram:
            self._histogram = [0] * self.n_bins

    def accumulate(self, state: SimulationState) -> None:
        """Bin all pair distances from the current frame."""
        positions = state.particles.positions
        n = len(positions)
        dr = self.cutoff / self.n_bins
        box = _box_lengths_from_cell(state.cell)

        # Store particle count and volume for RDF normalisation.
        self._n_particles = n
        if state.cell is not None:
            self._volume = state.cell.volume()
        elif self._volume <= 0.0:
            # Estimate a fixed volume from the first frame's bounding box with 20% padding.
            # This stays constant across frames for consistent normalization.
            mins = [min(positions[k][d] for k in range(n)) for d in range(3)]
            maxs = [max(positions[k][d] for k in range(n)) for d in range(3)]
            pad = 0.2
            self._volume = max(
                (maxs[0] - mins[0] + pad) * (maxs[1] - mins[1] + pad) * (maxs[2] - mins[2] + pad),
                1e-30,
            )

        for i in range(n):
            for j in range(i + 1, n):
                if box is not None:
                    dist = _minimum_image_distance(positions[i], positions[j], box)
                else:
                    dist = _distance(positions[i], positions[j])
                if dist < self.cutoff:
                    bin_idx = int(dist / dr)
                    if bin_idx < self.n_bins:
                        self._histogram[bin_idx] += 1
        self._n_frames += 1

    def compute_rdf(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return (r_values, g_r_values).

        g(r) = histogram / (n_frames * N * rho * 4*pi*r^2*dr)
        """
        if self._n_frames == 0:
            r_vals = tuple(
                (i + 0.5) * self.cutoff / self.n_bins for i in range(self.n_bins)
            )
            return r_vals, tuple(0.0 for _ in range(self.n_bins))

        dr = self.cutoff / self.n_bins
        r_values: list[float] = []
        g_r_values: list[float] = []

        n_particles = self._n_particles if self._n_particles > 0 else 2
        volume = self._volume

        if volume <= 0.0:
            # Estimate from cutoff sphere (rough fallback).
            volume = (4.0 / 3.0) * pi * self.cutoff ** 3

        rho = n_particles / volume

        for i in range(self.n_bins):
            r = (i + 0.5) * dr
            shell_volume = 4.0 * pi * r * r * dr
            ideal_count = self._n_frames * n_particles * rho * shell_volume / 2.0
            g = self._histogram[i] / ideal_count if ideal_count > 0.0 else 0.0
            r_values.append(r)
            g_r_values.append(g)

        return tuple(r_values), tuple(g_r_values)

    def summary(self) -> dict:
        """Peak position, peak height, and coordination number (integral to first minimum)."""
        r_vals, g_vals = self.compute_rdf()
        if not g_vals or max(g_vals) == 0.0:
            return {
                "peak_position": 0.0,
                "peak_height": 0.0,
                "coordination_number": 0.0,
            }

        peak_idx = 0
        peak_val = g_vals[0]
        for i, g in enumerate(g_vals):
            if g > peak_val:
                peak_val = g
                peak_idx = i

        # Coordination number: integrate 4*pi*r^2*rho*g(r)*dr up to first minimum after peak.
        dr = self.cutoff / self.n_bins
        rho = self._n_particles / self._volume if self._volume > 0.0 else 0.0
        coord_number = 0.0
        first_min_idx = peak_idx + 1
        # Walk past peak to find the first local minimum.
        for i in range(peak_idx + 1, len(g_vals) - 1):
            if g_vals[i] <= g_vals[i - 1] and g_vals[i] <= g_vals[i + 1]:
                first_min_idx = i
                break
        else:
            first_min_idx = len(g_vals)

        for i in range(first_min_idx):
            r = r_vals[i]
            coord_number += 4.0 * pi * r * r * rho * g_vals[i] * dr

        return {
            "peak_position": r_vals[peak_idx],
            "peak_height": peak_val,
            "coordination_number": coord_number,
        }


# ---------------------------------------------------------------------------
# Umbrella sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UmbrellaSamplingWindow:
    """Definition of a single umbrella-sampling bias window."""

    center: float
    force_constant: float
    cv_name: str = "distance"


@dataclass(slots=True)
class UmbrellaSampler:
    """Umbrella sampling manager with WHAM-based PMF reconstruction."""

    windows: tuple[UmbrellaSamplingWindow, ...]
    cv_particle_a: int = 0
    cv_particle_b: int = 1
    n_bins: int = 100
    _histograms: list[list[int]] = field(default_factory=list)
    _current_window: int = 0

    def __post_init__(self) -> None:
        if not self._histograms:
            self._histograms = [[0] * self.n_bins for _ in self.windows]

    # -- CV helpers ----------------------------------------------------------

    def _compute_cv(self, state: SimulationState) -> float:
        """Compute the collective variable (distance between two particles)."""
        positions = state.particles.positions
        return _distance(positions[self.cv_particle_a], positions[self.cv_particle_b])

    def _cv_unit_vector(self, state: SimulationState) -> Vector3:
        """Unit vector from particle_a to particle_b."""
        pa = state.particles.positions[self.cv_particle_a]
        pb = state.particles.positions[self.cv_particle_b]
        dx = pb[0] - pa[0]
        dy = pb[1] - pa[1]
        dz = pb[2] - pa[2]
        dist = sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 1e-30:
            return (0.0, 0.0, 0.0)
        inv = 1.0 / dist
        return (dx * inv, dy * inv, dz * inv)

    def _cv_range(self) -> tuple[float, float]:
        """Return the CV range spanned by all windows."""
        centres = [w.center for w in self.windows]
        margin = 0.5  # nm padding
        return (min(centres) - margin, max(centres) + margin)

    # -- Bias ----------------------------------------------------------------

    def bias_energy(self, state: SimulationState) -> float:
        """Harmonic bias energy for the current window: E = 0.5*k*(cv - center)^2."""
        cv = self._compute_cv(state)
        w = self.windows[self._current_window]
        delta = cv - w.center
        return 0.5 * w.force_constant * delta * delta

    def bias_force(
        self, state: SimulationState
    ) -> tuple[tuple[int, Vector3], ...]:
        """Bias forces on the two CV particles.

        F_bias = -k*(cv - center) * d(cv)/d(r)
        For a distance CV the gradient on particle_a is -r_hat and on
        particle_b is +r_hat, so:
          F_a = +k*(cv - center)*r_hat
          F_b = -k*(cv - center)*r_hat
        """
        cv = self._compute_cv(state)
        w = self.windows[self._current_window]
        delta = cv - w.center
        r_hat = self._cv_unit_vector(state)
        scale = w.force_constant * delta
        fa: Vector3 = (scale * r_hat[0], scale * r_hat[1], scale * r_hat[2])
        fb: Vector3 = (-scale * r_hat[0], -scale * r_hat[1], -scale * r_hat[2])
        return (
            (self.cv_particle_a, fa),
            (self.cv_particle_b, fb),
        )

    # -- Histogram collection ------------------------------------------------

    def record_cv(self, state: SimulationState) -> None:
        """Bin the current CV value into the active window's histogram."""
        cv = self._compute_cv(state)
        lo, hi = self._cv_range()
        if hi <= lo:
            return
        bin_width = (hi - lo) / self.n_bins
        bin_idx = int((cv - lo) / bin_width)
        if 0 <= bin_idx < self.n_bins:
            self._histograms[self._current_window][bin_idx] += 1

    # -- WHAM ----------------------------------------------------------------

    def compute_pmf_wham(
        self, temperature: float = 300.0
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Reconstruct the PMF via the Weighted Histogram Analysis Method.

        Returns (cv_values, pmf_values).
        """
        from core.units import BOLTZMANN_CONSTANT

        beta = 1.0 / (BOLTZMANN_CONSTANT * temperature)
        n_windows = len(self.windows)
        n_bins = self.n_bins
        lo, hi = self._cv_range()
        bin_width = (hi - lo) / n_bins

        cv_values = tuple(lo + (i + 0.5) * bin_width for i in range(n_bins))

        # Total counts per window.
        n_counts = [sum(h) for h in self._histograms]

        # Pre-compute bias energies for each window at each bin centre.
        bias: list[list[float]] = []
        for wi in range(n_windows):
            w = self.windows[wi]
            row: list[float] = []
            for bi in range(n_bins):
                delta = cv_values[bi] - w.center
                row.append(0.5 * w.force_constant * delta * delta)
            bias.append(row)

        # Initial free energies (all zero).
        f = [0.0] * n_windows
        max_iterations = 500
        tolerance = 1e-8

        for _ in range(max_iterations):
            # Compute unbiased probability P(xi).
            p_unbias = [0.0] * n_bins
            for bi in range(n_bins):
                numerator = 0.0
                for wi in range(n_windows):
                    numerator += self._histograms[wi][bi]
                denominator = 0.0
                for wi in range(n_windows):
                    denominator += n_counts[wi] * exp(-beta * bias[wi][bi] + f[wi])
                p_unbias[bi] = numerator / denominator if denominator > 0.0 else 0.0

            # Update free energies.
            f_new = [0.0] * n_windows
            max_diff = 0.0
            for wi in range(n_windows):
                s = 0.0
                for bi in range(n_bins):
                    s += p_unbias[bi] * exp(-beta * bias[wi][bi])
                f_new[wi] = -log(s) if s > 0.0 else 0.0
                max_diff = max(max_diff, abs(f_new[wi] - f[wi]))

            # Shift so f[0] == 0.
            shift = f_new[0]
            f = [fi - shift for fi in f_new]

            if max_diff < tolerance:
                break

        # PMF = -kT * ln(P_unbias), shifted so minimum is zero.
        pmf_raw: list[float] = []
        for bi in range(n_bins):
            if p_unbias[bi] > 0.0:
                pmf_raw.append(-1.0 / beta * log(p_unbias[bi]))
            else:
                pmf_raw.append(float("inf"))

        # Shift minimum to zero.
        finite_vals = [v for v in pmf_raw if v != float("inf")]
        if finite_vals:
            pmf_min = min(finite_vals)
            pmf_values = tuple(v - pmf_min if v != float("inf") else float("inf") for v in pmf_raw)
        else:
            pmf_values = tuple(pmf_raw)

        return cv_values, pmf_values


# ---------------------------------------------------------------------------
# BindingEnergyEstimator
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BindingEnergyEstimator:
    """Estimates binding free energy from interaction energies and distance fluctuations."""

    group_a_indices: tuple[int, ...]
    group_b_indices: tuple[int, ...]
    _interaction_energies: list[float] = field(default_factory=list)
    _distance_history: list[float] = field(default_factory=list)

    def compute_interaction_energy(
        self, state: SimulationState, forces_a_on_b: VectorTuple | None = None
    ) -> float:
        """Estimate interaction energy between groups.

        Uses force-dot-displacement as a work proxy when *forces_a_on_b* is
        supplied, otherwise falls back to a simple distance-based LJ-like
        estimate.
        """
        positions = state.particles.positions
        if forces_a_on_b is not None:
            # Sum F . dr for each particle in group_b relative to COM of group_a.
            com_a = _center_of_mass_subset(
                positions, state.particles.masses, self.group_a_indices
            )
            energy = 0.0
            for idx, force in zip(self.group_b_indices, forces_a_on_b, strict=False):
                dx = positions[idx][0] - com_a[0]
                dy = positions[idx][1] - com_a[1]
                dz = positions[idx][2] - com_a[2]
                energy += force[0] * dx + force[1] * dy + force[2] * dz
            return energy

        # Simplified pairwise inverse-distance proxy (unitless score).
        energy = 0.0
        for i in self.group_a_indices:
            for j in self.group_b_indices:
                r = _distance(positions[i], positions[j])
                if r > 1e-10:
                    # Approximate LJ-like: -1/r^6 attractive contribution.
                    r6 = r ** 6
                    energy -= 1.0 / r6
        return energy

    def record(self, state: SimulationState) -> None:
        """Record centre-of-mass distance and interaction energy."""
        positions = state.particles.positions
        masses = state.particles.masses
        com_a = _center_of_mass_subset(positions, masses, self.group_a_indices)
        com_b = _center_of_mass_subset(positions, masses, self.group_b_indices)
        dist = _distance(com_a, com_b)
        self._distance_history.append(dist)
        self._interaction_energies.append(
            self.compute_interaction_energy(state)
        )

    def estimate_binding_energy(self) -> dict:
        """Estimate binding free energy components.

        dG ~ dE_interaction - T*dS_config
        where the entropic correction is estimated from distance fluctuations.
        """
        if not self._interaction_energies:
            return {
                "mean_interaction_energy": 0.0,
                "entropic_correction": 0.0,
                "estimated_binding_dG": 0.0,
                "mean_distance": 0.0,
                "distance_std": 0.0,
                "n_samples": 0,
            }

        n = len(self._interaction_energies)
        mean_e = sum(self._interaction_energies) / n
        mean_d = sum(self._distance_history) / n
        var_d = (
            sum((d - mean_d) ** 2 for d in self._distance_history) / n
            if n > 1
            else 0.0
        )
        std_d = sqrt(var_d)

        # Simple MM/PBSA-inspired entropic correction.
        # -T*dS ~ kT * ln(sigma_d / d_ref) where d_ref is the mean distance.
        # Using room temperature (300 K) and BOLTZMANN_CONSTANT from core.units.
        from core.units import BOLTZMANN_CONSTANT

        temperature = 300.0
        kt = BOLTZMANN_CONSTANT * temperature
        if mean_d > 1e-10 and std_d > 1e-10:
            entropic_correction = kt * log(std_d / mean_d)
        else:
            entropic_correction = 0.0

        estimated_dg = mean_e - entropic_correction

        return {
            "mean_interaction_energy": mean_e,
            "entropic_correction": entropic_correction,
            "estimated_binding_dG": estimated_dg,
            "mean_distance": mean_d,
            "distance_std": std_d,
            "n_samples": n,
        }


# ---------------------------------------------------------------------------
# TemperatureSchedule (utility for annealing / replica exchange)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TemperatureSchedule:
    """Immutable schedule of target temperatures for simulated annealing or REMD."""

    temperatures: tuple[float, ...]
    steps_per_temperature: int = 1000

    def __post_init__(self) -> None:
        if not self.temperatures:
            raise ContractValidationError("TemperatureSchedule requires at least one temperature.")
        if any(t <= 0.0 for t in self.temperatures):
            raise ContractValidationError("All temperatures must be strictly positive.")
        if self.steps_per_temperature <= 0:
            raise ContractValidationError("steps_per_temperature must be positive.")

    def temperature_at_step(self, step: int) -> float:
        """Return the target temperature for the given simulation step."""
        idx = step // self.steps_per_temperature
        idx = min(idx, len(self.temperatures) - 1)
        return self.temperatures[idx]


# ---------------------------------------------------------------------------
# AnalysisReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    """Comprehensive report produced by the AdaptiveAnalysisEngine."""

    steps_analyzed: int
    rmsd_summary: dict
    rdf_data: dict
    pmf_data: dict
    binding_energy: dict
    energy_autocorrelation: dict
    energy_block_average: dict
    convergence_metrics: dict
    metadata: FrozenMetadata

    def is_converged(
        self, rmsd_threshold: float = 0.1, energy_drift_threshold: float = 0.01
    ) -> bool:
        """True if RMSD is stable AND energy drift is low."""
        rmsd_stable = self.convergence_metrics.get("rmsd_stable", False)
        energy_drift = self.convergence_metrics.get("energy_drift", float("inf"))
        rmsd_std = self.rmsd_summary.get("std_rmsd", float("inf"))
        return rmsd_std < rmsd_threshold and abs(energy_drift) < energy_drift_threshold

    def to_dict(self) -> dict:
        """Serialize the report for JSON output."""
        return {
            "steps_analyzed": self.steps_analyzed,
            "rmsd_summary": self.rmsd_summary,
            "rdf_data": self.rdf_data,
            "pmf_data": self.pmf_data,
            "binding_energy": self.binding_energy,
            "energy_autocorrelation": self.energy_autocorrelation,
            "energy_block_average": self.energy_block_average,
            "convergence_metrics": self.convergence_metrics,
            "metadata": self.metadata.to_dict() if isinstance(self.metadata, FrozenMetadata) else self.metadata,
        }


# ---------------------------------------------------------------------------
# AdaptiveAnalysisEngine
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AdaptiveAnalysisEngine:
    """Intelligence layer that watches a simulation and produces comprehensive
    analysis automatically.

    Computes RMSD, RDF, PMF (via umbrella sampling), binding energy, and
    autocorrelation without the user needing to request them.
    """

    rmsd_tracker: RMSDTracker | None = None
    rdf_calculator: RDFCalculator | None = None
    umbrella_sampler: UmbrellaSampler | None = None
    binding_estimator: BindingEnergyEstimator | None = None
    transition_detector: object | None = None  # TransitionStateDetector
    reaction_coord_generator: object | None = None  # ReactionCoordinateGenerator
    _energy_series: list[float] = field(default_factory=list)
    _temperature_series: list[float] = field(default_factory=list)
    _steps_analyzed: int = 0
    collection_interval: int = 10
    name: str = "adaptive_analysis_engine"
    classification: str = "[proposed novel]"

    # -- Auto-configuration --------------------------------------------------

    def auto_configure(
        self,
        state: SimulationState,
        *,
        group_a: tuple[int, ...] | None = None,
        group_b: tuple[int, ...] | None = None,
    ) -> None:
        """Set up all analysis sub-engines based on the initial state.

        Always configures RMSD tracking and RDF calculation.  Conditionally
        sets up umbrella sampling (if a periodic cell exists) and binding
        energy estimation (if groups are provided or can be guessed).
        """
        positions = state.particles.positions

        # Always: RMSD tracker with current positions as reference.
        self.rmsd_tracker = RMSDTracker(reference_positions=positions)

        # Always: RDF calculator.
        self.rdf_calculator = RDFCalculator()

        # Umbrella sampling if cell exists.
        if state.cell is not None:
            box = _box_lengths_from_cell(state.cell)
            if box is not None:
                # Find the biggest dimension for umbrella windows.
                max_dim = max(box)
                n_windows = 10
                spacing = max_dim / (n_windows + 1)
                windows = tuple(
                    UmbrellaSamplingWindow(
                        center=(i + 1) * spacing,
                        force_constant=1000.0,
                        cv_name="distance",
                    )
                    for i in range(n_windows)
                )
                self.umbrella_sampler = UmbrellaSampler(windows=windows)

        # Binding energy estimator.
        n = state.particle_count
        if group_a is not None and group_b is not None:
            self.binding_estimator = BindingEnergyEstimator(
                group_a_indices=group_a, group_b_indices=group_b
            )
        elif n >= 4:
            # Guess: first half vs second half.
            half = n // 2
            self.binding_estimator = BindingEnergyEstimator(
                group_a_indices=tuple(range(half)),
                group_b_indices=tuple(range(half, n)),
            )

        # Transition state detection.
        from validation.transition_analysis import TransitionStateDetector, ReactionCoordinateGenerator
        target_temp = getattr(state.thermodynamics, 'target_temperature', None) if hasattr(state, 'thermodynamics') else None
        if target_temp is None:
            target_temp = 300.0
        self.transition_detector = TransitionStateDetector()
        self.reaction_coord_generator = ReactionCoordinateGenerator(temperature=target_temp)

    # -- Collection ----------------------------------------------------------

    def collect(
        self,
        state: SimulationState,
        potential_energy: float | None = None,
        temperature: float | None = None,
    ) -> None:
        """Called each step by the simulation loop.  Skips unless on-interval."""
        if state.step % self.collection_interval != 0:
            return

        if potential_energy is not None:
            self._energy_series.append(potential_energy)
        if temperature is not None:
            self._temperature_series.append(temperature)

        if self.rmsd_tracker is not None:
            self.rmsd_tracker.record(state.step, state)

        if self.rdf_calculator is not None:
            self.rdf_calculator.accumulate(state)

        if self.umbrella_sampler is not None:
            self.umbrella_sampler.record_cv(state)

        if self.binding_estimator is not None:
            self.binding_estimator.record(state)

        if self.transition_detector is not None and hasattr(self.transition_detector, 'record'):
            cv_val = 0.0
            if self.umbrella_sampler is not None:
                # Use the umbrella CV (distance between umbrella particles).
                pa = state.particles.positions[self.umbrella_sampler.cv_particle_a]
                pb = state.particles.positions[self.umbrella_sampler.cv_particle_b]
                cv_val = _distance(pa, pb)
            elif state.particle_count >= 2:
                # Default CV: distance between first two particles.
                cv_val = _distance(state.particles.positions[0], state.particles.positions[1])
            pe = potential_energy if potential_energy is not None else 0.0
            self.transition_detector.record(state.step, cv_val, pe)

        self._steps_analyzed += 1

    # -- Report generation ---------------------------------------------------

    def generate_report(self) -> AnalysisReport:
        """Compile results from all sub-analysers into a comprehensive report."""
        from validation.statistical_mechanics import (
            compute_autocorrelation,
            block_average,
        )

        # RMSD summary.
        rmsd_summary = self.rmsd_tracker.summary() if self.rmsd_tracker else {}

        # RDF data.
        if self.rdf_calculator is not None:
            r_vals, g_vals = self.rdf_calculator.compute_rdf()
            rdf_data = {
                "r_values": r_vals,
                "g_r_values": g_vals,
                **self.rdf_calculator.summary(),
            }
        else:
            rdf_data = {}

        # PMF data.
        if self.umbrella_sampler is not None:
            try:
                cv_vals, pmf_vals = self.umbrella_sampler.compute_pmf_wham()
                pmf_data = {"cv_values": cv_vals, "pmf_values": pmf_vals}
            except Exception:
                pmf_data = {"error": "PMF computation failed — insufficient sampling."}
        else:
            pmf_data = {}

        # Binding energy.
        if self.binding_estimator is not None:
            binding_energy = self.binding_estimator.estimate_binding_energy()
        else:
            binding_energy = {}

        # Energy autocorrelation.
        energy_autocorrelation: dict = {}
        energy_block_avg: dict = {}
        if len(self._energy_series) >= 10:
            energy_data = tuple(self._energy_series)
            try:
                acf_result = compute_autocorrelation(energy_data)
                energy_autocorrelation = {
                    "integrated_autocorrelation_time": acf_result.integrated_time,
                    "effective_samples": acf_result.effective_samples,
                }
            except Exception:
                energy_autocorrelation = {"error": "Autocorrelation computation failed."}

            try:
                ba_result = block_average(energy_data)
                energy_block_avg = {
                    "mean": ba_result.mean,
                    "standard_error": ba_result.standard_error,
                }
            except Exception:
                energy_block_avg = {"error": "Block average computation failed."}

        # Convergence metrics.
        convergence: dict = {}
        if self._energy_series:
            # Energy drift: linear regression slope normalised by mean.
            n = len(self._energy_series)
            if n >= 2:
                mean_e = sum(self._energy_series) / n
                mean_t = (n - 1) / 2.0
                num = sum(
                    (i - mean_t) * (e - mean_e)
                    for i, e in enumerate(self._energy_series)
                )
                den = sum((i - mean_t) ** 2 for i in range(n))
                slope = num / den if den > 0.0 else 0.0
                drift = slope / abs(mean_e) if abs(mean_e) > 1e-30 else 0.0
                convergence["energy_drift"] = drift
            else:
                convergence["energy_drift"] = 0.0

        # RMSD plateau detection.
        if rmsd_summary and rmsd_summary.get("std_rmsd", float("inf")) < 0.1:
            convergence["rmsd_stable"] = True
        else:
            convergence["rmsd_stable"] = False

        # Transition state data.
        transition_data: dict = {}
        if self.transition_detector is not None and hasattr(self.transition_detector, 'detect_transitions'):
            transitions = self.transition_detector.detect_transitions()
            transition_data["n_transitions"] = len(transitions)
            transition_data["transitions"] = [
                {"step": t.step, "barrier": t.energy_barrier, "cv": t.cv_value_at_transition}
                for t in transitions
            ]
            # Generate reaction coordinate if we have CV data.
            if self.reaction_coord_generator is not None and hasattr(self.transition_detector, '_cv_history'):
                cv_vals = tuple(v for _, v in self.transition_detector._cv_history)
                energies = tuple(e for _, e in self.transition_detector._energy_history)
                if len(cv_vals) >= 10 and max(cv_vals) - min(cv_vals) > 1e-8:
                    try:
                        rc = self.reaction_coord_generator.from_cv_trajectory(cv_vals, energies)
                        transition_data["reaction_coordinate"] = {
                            "barrier_height": rc.barrier_height,
                            "reaction_free_energy": rc.reaction_free_energy,
                            "n_cv_bins": len(rc.cv_values),
                        }
                    except Exception:
                        pass  # degenerate CV data — skip RC generation
        convergence["transition_data"] = transition_data

        return AnalysisReport(
            steps_analyzed=self._steps_analyzed,
            rmsd_summary=rmsd_summary,
            rdf_data=rdf_data,
            pmf_data=pmf_data,
            binding_energy=binding_energy,
            energy_autocorrelation=energy_autocorrelation,
            energy_block_average=energy_block_avg,
            convergence_metrics=convergence,
            metadata=FrozenMetadata(
                {
                    "engine": self.name,
                    "classification": self.classification,
                    "collection_interval": self.collection_interval,
                }
            ),
        )
