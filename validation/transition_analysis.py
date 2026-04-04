"""Transition state detection, reaction coordinate generation, and SVG plotting.

Detects transition states from trajectory data, generates reaction coordinates
(via Boltzmann inversion or pre-computed PMF), and produces inline SVG plots of
all key observables.  Pure Python -- no matplotlib dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt, pi, exp, log, floor, ceil, inf

from core.exceptions import ContractValidationError
from core.types import FrozenMetadata, Vector3, VectorTuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KB = 0.00831446  # Boltzmann constant in kJ/(mol*K)


# ===================================================================
# Data containers
# ===================================================================


@dataclass(frozen=True, slots=True)
class TransitionState:
    """A single detected transition state."""

    step: int
    energy_barrier: float
    forward_rate: float
    reverse_rate: float
    cv_value_at_transition: float
    reactant_basin: tuple[float, float]
    product_basin: tuple[float, float]
    metadata: FrozenMetadata


@dataclass(frozen=True, slots=True)
class ReactionCoordinate:
    """Free-energy profile along a collective variable."""

    cv_values: tuple[float, ...]
    free_energy: tuple[float, ...]
    probability: tuple[float, ...]
    transition_states: tuple[TransitionState, ...]
    reactant_free_energy: float
    product_free_energy: float
    barrier_height: float
    reaction_free_energy: float
    metadata: FrozenMetadata


# ===================================================================
# TransitionStateDetector
# ===================================================================


@dataclass(slots=True)
class TransitionStateDetector:
    """Monitors a collective-variable time series and detects basin crossings."""

    energy_window: float = 5.0
    cv_smoothing_window: int = 10
    min_basin_residence: int = 20
    name: str = "transition_state_detector"
    classification: str = "[proposed novel]"

    _cv_history: list[tuple[int, float]] = field(default_factory=list)
    _energy_history: list[tuple[int, float]] = field(default_factory=list)
    _detected_transitions: list[TransitionState] = field(default_factory=list)
    _current_basin: str = "unknown"
    _basin_entry_step: int = 0

    # ----- helpers -----

    def _smooth_cv(self, values: list[float]) -> list[float]:
        """Rolling mean of *values* with *cv_smoothing_window*."""
        w = self.cv_smoothing_window
        if w < 2 or len(values) < w:
            return list(values)
        smoothed: list[float] = []
        running = sum(values[:w])
        for i in range(len(values)):
            if i < w:
                smoothed.append(sum(values[: i + 1]) / (i + 1))
            else:
                running += values[i] - values[i - w]
                smoothed.append(running / w)
        return smoothed

    # ----- public API -----

    def record(self, step: int, cv_value: float, energy: float) -> None:
        """Append a single observation and attempt incremental detection."""
        self._cv_history.append((step, cv_value))
        self._energy_history.append((step, energy))

        # Incremental smoothing -- recompute tail of smoothed series
        raw_cv = [v for _, v in self._cv_history]
        smoothed = self._smooth_cv(raw_cv)

        n = len(smoothed)
        if n < self.min_basin_residence:
            return

        basins = self.find_basins()
        if len(basins) < 2:
            return

        current_cv = smoothed[-1]
        basin_label = self._classify_cv(current_cv, basins)

        if self._current_basin == "unknown":
            self._current_basin = basin_label
            self._basin_entry_step = step
            return

        residence = step - self._basin_entry_step
        if basin_label != self._current_basin and basin_label != "unknown" and residence >= self.min_basin_residence:
            # Crossing detected
            origin_label = self._current_basin
            dest_label = basin_label

            origin_basin = basins[0] if origin_label == "reactant" else basins[-1]
            dest_basin = basins[-1] if dest_label == "product" else basins[0]

            # Energy barrier: scan back to basin_entry_step
            crossing_energies: list[float] = []
            origin_energies: list[float] = []
            for s, e in self._energy_history:
                if self._basin_entry_step <= s <= step:
                    crossing_energies.append(e)
                if s < self._basin_entry_step:
                    origin_energies.append(e)

            if crossing_energies and origin_energies:
                barrier = max(crossing_energies) - min(origin_energies)
            elif crossing_energies:
                barrier = max(crossing_energies) - min(crossing_energies)
            else:
                barrier = 0.0

            # Rough rate estimate
            total_steps = max(1, len(self._cv_history))
            n_transitions_so_far = len(self._detected_transitions) + 1
            fwd_rate = n_transitions_so_far / max(1.0, float(total_steps))
            rev_rate = fwd_rate * 0.5  # placeholder symmetry assumption

            ts = TransitionState(
                step=step,
                energy_barrier=max(0.0, barrier),
                forward_rate=fwd_rate,
                reverse_rate=rev_rate,
                cv_value_at_transition=current_cv,
                reactant_basin=origin_basin,
                product_basin=dest_basin,
                metadata=FrozenMetadata({"origin": origin_label, "destination": dest_label}),
            )
            self._detected_transitions.append(ts)
            self._current_basin = basin_label
            self._basin_entry_step = step

    def detect_transitions(self) -> tuple[TransitionState, ...]:
        """Analyse the full CV trajectory and return all basin crossings."""
        if len(self._cv_history) < self.min_basin_residence * 2:
            return tuple(self._detected_transitions)

        raw_cv = [v for _, v in self._cv_history]
        smoothed = self._smooth_cv(raw_cv)
        steps = [s for s, _ in self._cv_history]
        energies = [e for _, e in self._energy_history]

        basins = self.find_basins()
        if len(basins) < 2:
            return tuple(self._detected_transitions)

        transitions: list[TransitionState] = []
        current_basin_label = self._classify_cv(smoothed[0], basins)
        basin_entry_idx = 0
        residence_counts: dict[str, int] = {}

        for i in range(1, len(smoothed)):
            label = self._classify_cv(smoothed[i], basins)
            if label == "unknown" or label == current_basin_label:
                continue

            residence = i - basin_entry_idx
            if residence < self.min_basin_residence:
                continue

            residence_counts[current_basin_label] = residence_counts.get(current_basin_label, 0) + residence

            origin_basin = basins[0] if current_basin_label == "reactant" else basins[-1]
            dest_basin = basins[-1] if label == "product" else basins[0]

            crossing_e = energies[basin_entry_idx: i + 1]
            origin_e = energies[max(0, basin_entry_idx - self.min_basin_residence): basin_entry_idx]
            if crossing_e and origin_e:
                barrier = max(crossing_e) - min(origin_e)
            elif crossing_e:
                barrier = max(crossing_e) - min(crossing_e)
            else:
                barrier = 0.0

            # Find the step of maximum energy during crossing (transition state)
            ts_local_idx = basin_entry_idx + crossing_e.index(max(crossing_e))
            ts_step = steps[ts_local_idx] if ts_local_idx < len(steps) else steps[i]
            ts_cv = smoothed[ts_local_idx] if ts_local_idx < len(smoothed) else smoothed[i]

            transitions.append(TransitionState(
                step=ts_step,
                energy_barrier=max(0.0, barrier),
                forward_rate=0.0,  # updated below
                reverse_rate=0.0,
                cv_value_at_transition=ts_cv,
                reactant_basin=origin_basin,
                product_basin=dest_basin,
                metadata=FrozenMetadata({"origin": current_basin_label, "destination": label}),
            ))

            current_basin_label = label
            basin_entry_idx = i

        # Compute rates from residence times
        total_time = float(len(smoothed))
        fwd_count = sum(1 for t in transitions if t.metadata.get("origin", "") == "reactant")
        rev_count = sum(1 for t in transitions if t.metadata.get("origin", "") == "product")
        reactant_time = float(residence_counts.get("reactant", 1))
        product_time = float(residence_counts.get("product", 1))

        fwd_rate = fwd_count / reactant_time if reactant_time > 0 else 0.0
        rev_rate = rev_count / product_time if product_time > 0 else 0.0

        rated: list[TransitionState] = []
        for t in transitions:
            rated.append(TransitionState(
                step=t.step,
                energy_barrier=t.energy_barrier,
                forward_rate=fwd_rate,
                reverse_rate=rev_rate,
                cv_value_at_transition=t.cv_value_at_transition,
                reactant_basin=t.reactant_basin,
                product_basin=t.product_basin,
                metadata=t.metadata,
            ))

        self._detected_transitions = rated
        return tuple(rated)

    def find_basins(self, n_bins: int = 50) -> list[tuple[float, float]]:
        """Histogram the CV values and return basins as (cv_min, cv_max) intervals."""
        if not self._cv_history:
            return []

        raw = [v for _, v in self._cv_history]
        cv_min, cv_max = min(raw), max(raw)
        if cv_max == cv_min:
            return [(cv_min, cv_max)]

        bin_width = (cv_max - cv_min) / n_bins
        counts = [0] * n_bins
        for v in raw:
            idx = min(int((v - cv_min) / bin_width), n_bins - 1)
            counts[idx] += 1

        mean_count = sum(counts) / n_bins
        threshold = mean_count * 1.5

        # Identify peak bins
        peak_bins: list[int] = []
        for i, c in enumerate(counts):
            if c > threshold:
                peak_bins.append(i)

        if not peak_bins:
            # Fall back: use the two highest bins
            sorted_indices = sorted(range(n_bins), key=lambda k: counts[k], reverse=True)
            if len(sorted_indices) >= 2:
                peak_bins = sorted(sorted_indices[:2])
            elif sorted_indices:
                peak_bins = sorted_indices[:1]

        # Merge adjacent peak bins into basins
        basins: list[tuple[float, float]] = []
        if not peak_bins:
            return []

        group_start = peak_bins[0]
        group_end = peak_bins[0]
        for pb in peak_bins[1:]:
            if pb <= group_end + 1:
                group_end = pb
            else:
                lo = cv_min + group_start * bin_width
                hi = cv_min + (group_end + 1) * bin_width
                basins.append((lo, hi))
                group_start = pb
                group_end = pb
        lo = cv_min + group_start * bin_width
        hi = cv_min + (group_end + 1) * bin_width
        basins.append((lo, hi))

        return basins

    # ----- internal helpers -----

    def _classify_cv(self, cv: float, basins: list[tuple[float, float]]) -> str:
        """Label a CV value as 'reactant', 'product', or 'unknown'."""
        if not basins:
            return "unknown"
        if basins[0][0] <= cv <= basins[0][1]:
            return "reactant"
        if len(basins) > 1 and basins[-1][0] <= cv <= basins[-1][1]:
            return "product"
        return "unknown"


# ===================================================================
# ReactionCoordinateGenerator
# ===================================================================


@dataclass(slots=True)
class ReactionCoordinateGenerator:
    """Builds a :class:`ReactionCoordinate` from trajectory or PMF data."""

    n_bins: int = 100
    temperature: float = 300.0
    name: str = "reaction_coordinate_generator"
    classification: str = "[proposed novel]"

    def from_cv_trajectory(
        self,
        cv_values: tuple[float, ...],
        energies: tuple[float, ...] | None = None,
        temperature: float | None = None,
    ) -> ReactionCoordinate:
        """Construct a free-energy profile via Boltzmann inversion of *cv_values*."""
        if not cv_values:
            raise ContractValidationError("cv_values must be non-empty.")

        temp = temperature if temperature is not None else self.temperature
        kT = _KB * temp

        cv_min, cv_max = min(cv_values), max(cv_values)
        if cv_max == cv_min:
            raise ContractValidationError("All CV values are identical; cannot histogram.")

        bin_width = (cv_max - cv_min) / self.n_bins
        counts = [0] * self.n_bins
        for v in cv_values:
            idx = min(int((v - cv_min) / bin_width), self.n_bins - 1)
            counts[idx] += 1

        total = float(len(cv_values))
        prob = tuple(c / total if total > 0 else 0.0 for c in counts)

        # Free energy via Boltzmann inversion: F = -kT ln(P)
        fe_raw: list[float] = []
        for p in prob:
            if p > 0:
                fe_raw.append(-kT * log(p))
            else:
                fe_raw.append(inf)

        # Shift so minimum is zero
        fe_min = min(v for v in fe_raw if v < inf)
        free_energy = tuple((v - fe_min) if v < inf else 100.0 * kT for v in fe_raw)

        cv_grid = tuple(cv_min + (i + 0.5) * bin_width for i in range(self.n_bins))

        # Find basins and transition states from the free-energy profile
        basins, ts_list = self._find_fe_features(cv_grid, free_energy, kT)

        reactant_basin = basins[0] if basins else (cv_grid[0], cv_grid[-1])
        product_basin = basins[-1] if len(basins) > 1 else reactant_basin

        r_fe = self._basin_min_fe(cv_grid, free_energy, reactant_basin)
        p_fe = self._basin_min_fe(cv_grid, free_energy, product_basin)
        barrier = max(free_energy) - r_fe if free_energy else 0.0
        if ts_list:
            barrier = max(t.energy_barrier for t in ts_list)

        return ReactionCoordinate(
            cv_values=cv_grid,
            free_energy=free_energy,
            probability=prob,
            transition_states=tuple(ts_list),
            reactant_free_energy=r_fe,
            product_free_energy=p_fe,
            barrier_height=barrier,
            reaction_free_energy=p_fe - r_fe,
            metadata=FrozenMetadata({"temperature": temp, "n_bins": self.n_bins, "method": "boltzmann_inversion"}),
        )

    def from_pmf(
        self,
        cv_grid: tuple[float, ...],
        pmf_values: tuple[float, ...],
    ) -> ReactionCoordinate:
        """Build a :class:`ReactionCoordinate` from a pre-computed PMF."""
        if len(cv_grid) != len(pmf_values):
            raise ContractValidationError("cv_grid and pmf_values must have the same length.")
        if not cv_grid:
            raise ContractValidationError("cv_grid must be non-empty.")

        kT = _KB * self.temperature

        # Shift PMF minimum to zero
        pmf_min = min(pmf_values)
        fe = tuple(v - pmf_min for v in pmf_values)

        # Probability from PMF
        boltz = [exp(-f / kT) if f / kT < 500 else 0.0 for f in fe]
        z = sum(boltz)
        prob = tuple(b / z if z > 0 else 0.0 for b in boltz)

        basins, ts_list = self._find_fe_features(cv_grid, fe, kT)

        reactant_basin = basins[0] if basins else (cv_grid[0], cv_grid[-1])
        product_basin = basins[-1] if len(basins) > 1 else reactant_basin

        r_fe = self._basin_min_fe(cv_grid, fe, reactant_basin)
        p_fe = self._basin_min_fe(cv_grid, fe, product_basin)
        barrier = 0.0
        if ts_list:
            barrier = max(t.energy_barrier for t in ts_list)
        else:
            barrier = max(fe) - r_fe

        return ReactionCoordinate(
            cv_values=cv_grid,
            free_energy=fe,
            probability=prob,
            transition_states=tuple(ts_list),
            reactant_free_energy=r_fe,
            product_free_energy=p_fe,
            barrier_height=barrier,
            reaction_free_energy=p_fe - r_fe,
            metadata=FrozenMetadata({"temperature": self.temperature, "method": "pmf_direct"}),
        )

    # ----- internal helpers -----

    @staticmethod
    def _find_fe_features(
        cv_grid: tuple[float, ...],
        fe: tuple[float, ...],
        kT: float,
    ) -> tuple[list[tuple[float, float]], list[TransitionState]]:
        """Locate basins (minima) and transition states (maxima) in a 1-D FE profile."""
        n = len(fe)
        if n < 3:
            return [], []

        # Find local minima and maxima
        minima: list[int] = []
        maxima: list[int] = []
        for i in range(1, n - 1):
            if fe[i] <= fe[i - 1] and fe[i] <= fe[i + 1]:
                minima.append(i)
            if fe[i] >= fe[i - 1] and fe[i] >= fe[i + 1]:
                maxima.append(i)

        if not minima:
            # Use global min as single basin
            gmin = fe.index(min(fe))
            minima = [gmin]

        # Build basins around each minimum -- extend until FE rises by kT
        basins: list[tuple[float, float]] = []
        for mi in minima:
            lo = mi
            while lo > 0 and fe[lo - 1] - fe[mi] < kT:
                lo -= 1
            hi = mi
            while hi < n - 1 and fe[hi + 1] - fe[mi] < kT:
                hi += 1
            basins.append((cv_grid[lo], cv_grid[hi]))

        # Find transition states between consecutive basins
        ts_list: list[TransitionState] = []
        for bi in range(len(basins) - 1):
            # Region between basin[bi] and basin[bi+1]
            b1_hi = basins[bi][1]
            b2_lo = basins[bi + 1][0]
            # Find max FE in this region
            max_fe = -inf
            max_idx = 0
            for i in range(n):
                if b1_hi <= cv_grid[i] <= b2_lo:
                    if fe[i] > max_fe:
                        max_fe = fe[i]
                        max_idx = i

            reactant_min_fe = min(fe[i] for i in range(n) if basins[bi][0] <= cv_grid[i] <= basins[bi][1])
            product_min_fe = min(fe[i] for i in range(n) if basins[bi + 1][0] <= cv_grid[i] <= basins[bi + 1][1])

            barrier = max_fe - reactant_min_fe
            fwd_rate = exp(-barrier / kT) if barrier / kT < 500 else 0.0
            rev_barrier = max_fe - product_min_fe
            rev_rate = exp(-rev_barrier / kT) if rev_barrier / kT < 500 else 0.0

            ts_list.append(TransitionState(
                step=0,
                energy_barrier=barrier,
                forward_rate=fwd_rate,
                reverse_rate=rev_rate,
                cv_value_at_transition=cv_grid[max_idx],
                reactant_basin=basins[bi],
                product_basin=basins[bi + 1],
                metadata=FrozenMetadata({"basin_pair": bi}),
            ))

        return basins, ts_list

    @staticmethod
    def _basin_min_fe(
        cv_grid: tuple[float, ...],
        fe: tuple[float, ...],
        basin: tuple[float, float],
    ) -> float:
        """Return the minimum free energy inside a basin interval."""
        vals = [fe[i] for i in range(len(cv_grid)) if basin[0] <= cv_grid[i] <= basin[1]]
        return min(vals) if vals else 0.0


# ===================================================================
# SVG Plotting Helpers
# ===================================================================


def _svg_header(width: int, height: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}">'
    )


def _svg_polyline(points: list[tuple[float, float]], color: str, stroke_width: int = 2) -> str:
    pts = " ".join(f"{x},{y}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{stroke_width}"/>'


def _svg_text(x: float, y: float, text: str, font_size: int = 12, anchor: str = "middle") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{font_size}" text-anchor="{anchor}"'
        f' font-family="monospace">{text}</text>'
    )


def _svg_line(
    x1: float, y1: float, x2: float, y2: float,
    color: str, stroke_width: int = 1, dash: str = "",
) -> str:
    style = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}"'
        f' stroke-width="{stroke_width}"{style}/>'
    )


def _svg_circle(cx: float, cy: float, r: float, fill: str) -> str:
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"/>'


def _svg_rect(x: float, y: float, width: float, height: float, fill: str) -> str:
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}"/>'


def _scale_to_viewport(values: list[float], min_px: float, max_px: float) -> list[float]:
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [0.5 * (min_px + max_px)] * len(values)
    return [min_px + (v - vmin) / (vmax - vmin) * (max_px - min_px) for v in values]


def _nice_ticks(vmin: float, vmax: float, n_ticks: int = 5) -> list[float]:
    """Generate approximately *n_ticks* human-friendly tick values."""
    if vmax == vmin:
        return [vmin]
    raw_step = (vmax - vmin) / max(1, n_ticks - 1)
    # Round to 1-2-5 sequence
    mag = 10 ** floor(log(raw_step) / log(10)) if raw_step > 0 else 1.0
    candidates = [mag, 2 * mag, 5 * mag, 10 * mag]
    step = min(candidates, key=lambda s: abs(s - raw_step))
    start = ceil(vmin / step) * step
    ticks: list[float] = []
    v = start
    while v <= vmax + step * 0.01:
        ticks.append(v)
        v += step
    return ticks if ticks else [vmin, vmax]


# Margins used by all plots
_ML, _MR, _MT, _MB = 70, 20, 40, 50  # left, right, top, bottom


def _axes_svg(
    width: int, height: int, title: str,
    x_label: str, y_label: str,
    x_values: list[float], y_values: list[float],
) -> tuple[str, list[float], list[float]]:
    """Return SVG elements for axes/labels and the pixel-mapped x/y lists."""
    parts: list[str] = []
    parts.append(_svg_header(width, height, title))

    # White background
    parts.append(_svg_rect(0, 0, float(width), float(height), "white"))

    pw = width - _ML - _MR
    ph = height - _MT - _MB

    x_px = _scale_to_viewport(x_values, float(_ML), float(_ML + pw))
    # Flip y so higher values go up
    y_px = _scale_to_viewport(y_values, float(_MT + ph), float(_MT))

    # Axes lines
    parts.append(_svg_line(float(_ML), float(_MT), float(_ML), float(_MT + ph), "black", 1))
    parts.append(_svg_line(float(_ML), float(_MT + ph), float(_ML + pw), float(_MT + ph), "black", 1))

    # Title
    parts.append(_svg_text(width / 2.0, 20.0, title, font_size=14))

    # Axis labels
    parts.append(_svg_text(width / 2.0, float(height - 5), x_label, font_size=11))
    parts.append(
        f'<text x="15" y="{height / 2}" font-size="11" text-anchor="middle"'
        f' font-family="monospace" transform="rotate(-90 15 {height / 2})">{y_label}</text>'
    )

    # X ticks
    x_ticks = _nice_ticks(min(x_values), max(x_values))
    for tv in x_ticks:
        px = _scale_to_viewport([tv], float(_ML), float(_ML + pw))[0]
        parts.append(_svg_line(px, float(_MT + ph), px, float(_MT + ph + 5), "black"))
        label = f"{tv:.4g}"
        parts.append(_svg_text(px, float(_MT + ph + 18), label, font_size=9))

    # Y ticks
    y_ticks = _nice_ticks(min(y_values), max(y_values))
    for tv in y_ticks:
        py = _scale_to_viewport([tv], float(_MT + ph), float(_MT))[0]
        parts.append(_svg_line(float(_ML - 5), py, float(_ML), py, "black"))
        label = f"{tv:.4g}"
        parts.append(_svg_text(float(_ML - 8), py + 3, label, font_size=9, anchor="end"))

    return "\n".join(parts), x_px, y_px


def _close_svg() -> str:
    return "</svg>"


# ===================================================================
# Public Plot Functions
# ===================================================================


def plot_energy_timeseries(
    steps: list[float],
    ke: list[float],
    pe: list[float],
    total: list[float],
    *,
    width: int = 800,
    height: int = 400,
) -> str:
    """SVG plot of KE, PE, and total energy vs. simulation step."""
    all_e = ke + pe + total
    axes, x_px, _ = _axes_svg(width, height, "Energy vs. Time", "Step", "Energy (kJ/mol)", steps, all_e)

    pw = width - _ML - _MR
    ph = height - _MT - _MB
    parts = [axes]

    for series, color, label in [(ke, "#2196F3", "KE"), (pe, "#F44336", "PE"), (total, "#000000", "Total")]:
        y_px = _scale_to_viewport(series, float(_MT + ph), float(_MT))
        points = list(zip(x_px, y_px))
        parts.append(_svg_polyline(points, color))

    # Legend
    lx = float(_ML + 10)
    for i, (color, label) in enumerate([("#2196F3", "KE"), ("#F44336", "PE"), ("#000000", "Total")]):
        ly = float(_MT + 15 + i * 16)
        parts.append(_svg_line(lx, ly, lx + 20, ly, color, 2))
        parts.append(_svg_text(lx + 25, ly + 4, label, font_size=10, anchor="start"))

    parts.append(_close_svg())
    return "\n".join(parts)


def plot_temperature_timeseries(
    steps: list[float],
    temperatures: list[float],
    target_temp: float | None = None,
    *,
    width: int = 800,
    height: int = 300,
) -> str:
    """SVG plot of temperature vs. simulation step."""
    axes, x_px, y_px = _axes_svg(width, height, "Temperature vs. Time", "Step", "Temperature (K)", steps, temperatures)
    parts = [axes]

    points = list(zip(x_px, y_px))
    parts.append(_svg_polyline(points, "#F44336"))

    if target_temp is not None:
        pw = width - _ML - _MR
        ph = height - _MT - _MB
        tgt_y = _scale_to_viewport([target_temp], float(_MT + ph), float(_MT))[0]
        parts.append(_svg_line(float(_ML), tgt_y, float(_ML + pw), tgt_y, "#4CAF50", 1, "5,5"))
        parts.append(_svg_text(float(_ML + pw - 5), tgt_y - 5, f"target={target_temp:.1f}K", font_size=9, anchor="end"))

    parts.append(_close_svg())
    return "\n".join(parts)


def plot_rmsd_timeseries(
    steps: list[float],
    rmsd_values: list[float],
    *,
    width: int = 800,
    height: int = 300,
) -> str:
    """SVG plot of RMSD vs. simulation step."""
    axes, x_px, y_px = _axes_svg(width, height, "RMSD vs. Time", "Step", "RMSD (nm)", steps, rmsd_values)
    parts = [axes]
    parts.append(_svg_polyline(list(zip(x_px, y_px)), "#2196F3"))
    parts.append(_close_svg())
    return "\n".join(parts)


def plot_rdf(
    r_values: list[float],
    g_r_values: list[float],
    *,
    width: int = 800,
    height: int = 400,
) -> str:
    """SVG plot of radial distribution function g(r)."""
    axes, x_px, y_px = _axes_svg(width, height, "Radial Distribution Function", "r (nm)", "g(r)", r_values, g_r_values)
    parts = [axes]
    parts.append(_svg_polyline(list(zip(x_px, y_px)), "#2196F3"))

    # Ideal gas reference line at g(r) = 1
    pw = width - _ML - _MR
    ph = height - _MT - _MB
    ref_y = _scale_to_viewport([1.0], float(_MT + ph), float(_MT))[0]
    parts.append(_svg_line(float(_ML), ref_y, float(_ML + pw), ref_y, "#9E9E9E", 1, "4,4"))

    parts.append(_close_svg())
    return "\n".join(parts)


def plot_pmf(
    cv_values: list[float],
    pmf_values: list[float],
    *,
    transition_states: list[TransitionState] | None = None,
    width: int = 800,
    height: int = 400,
) -> str:
    """SVG plot of the potential of mean force."""
    axes, x_px, y_px = _axes_svg(width, height, "Potential of Mean Force", "CV", "PMF (kJ/mol)", cv_values, pmf_values)
    parts = [axes]
    parts.append(_svg_polyline(list(zip(x_px, y_px)), "#2196F3"))

    pw = width - _ML - _MR
    ph = height - _MT - _MB

    # Mark basin minima (green dots)
    # Find local minima
    for i in range(1, len(pmf_values) - 1):
        if pmf_values[i] <= pmf_values[i - 1] and pmf_values[i] <= pmf_values[i + 1]:
            parts.append(_svg_circle(x_px[i], y_px[i], 4.0, "#4CAF50"))

    # Mark transition states (red dots)
    if transition_states:
        for ts in transition_states:
            ts_x = _scale_to_viewport([ts.cv_value_at_transition], float(_ML), float(_ML + pw))[0]
            ts_y = _scale_to_viewport(
                [ts.energy_barrier + min(pmf_values)], float(_MT + ph), float(_MT),
            )[0]
            parts.append(_svg_circle(ts_x, ts_y, 5.0, "#F44336"))
            # Dashed line showing barrier height
            base_y = _scale_to_viewport([min(pmf_values)], float(_MT + ph), float(_MT))[0]
            parts.append(_svg_line(ts_x, ts_y, ts_x, base_y, "#F44336", 1, "3,3"))

    parts.append(_close_svg())
    return "\n".join(parts)


def plot_reaction_coordinate(
    rc: ReactionCoordinate,
    *,
    width: int = 800,
    height: int = 400,
) -> str:
    """SVG plot of the free-energy surface along the reaction coordinate."""
    cv = list(rc.cv_values)
    fe = list(rc.free_energy)
    axes, x_px, y_px = _axes_svg(width, height, "Reaction Coordinate", "CV", "Free Energy (kJ/mol)", cv, fe)
    parts = [axes]
    parts.append(_svg_polyline(list(zip(x_px, y_px)), "#2196F3", 2))

    pw = width - _ML - _MR
    ph = height - _MT - _MB

    # Annotate basins
    if rc.transition_states:
        for ts in rc.transition_states:
            # Reactant basin label
            r_mid = 0.5 * (ts.reactant_basin[0] + ts.reactant_basin[1])
            r_x = _scale_to_viewport([r_mid], float(_ML), float(_ML + pw))[0]
            parts.append(_svg_text(r_x, float(_MT + ph + 35), "Reactant", font_size=10))

            # Product basin label
            p_mid = 0.5 * (ts.product_basin[0] + ts.product_basin[1])
            p_x = _scale_to_viewport([p_mid], float(_ML), float(_ML + pw))[0]
            parts.append(_svg_text(p_x, float(_MT + ph + 35), "Product", font_size=10))

            # Transition state marker
            ts_x = _scale_to_viewport([ts.cv_value_at_transition], float(_ML), float(_ML + pw))[0]
            ts_fe = ts.energy_barrier + rc.reactant_free_energy
            ts_y = _scale_to_viewport([ts_fe], float(_MT + ph), float(_MT))[0]
            parts.append(_svg_circle(ts_x, ts_y, 5.0, "#F44336"))
            # Arrow-like marker above
            parts.append(_svg_text(ts_x, ts_y - 10, "TS", font_size=10))

            # Barrier height annotation
            r_fe_y = _scale_to_viewport([rc.reactant_free_energy], float(_MT + ph), float(_MT))[0]
            parts.append(_svg_line(ts_x + 15, r_fe_y, ts_x + 15, ts_y, "#FF9800", 1, "3,3"))
            mid_y = 0.5 * (r_fe_y + ts_y)
            parts.append(_svg_text(ts_x + 20, mid_y, f"dG++={rc.barrier_height:.1f}", font_size=9, anchor="start"))

            # dG annotation
            p_fe_y = _scale_to_viewport([rc.product_free_energy], float(_MT + ph), float(_MT))[0]
            ann_x = float(_ML + pw - 30)
            parts.append(_svg_line(ann_x, r_fe_y, ann_x, p_fe_y, "#9C27B0", 1, "3,3"))
            dg_mid = 0.5 * (r_fe_y + p_fe_y)
            parts.append(_svg_text(ann_x + 5, dg_mid, f"dG={rc.reaction_free_energy:.1f}", font_size=9, anchor="start"))
            break  # annotate first TS pair only

    parts.append(_close_svg())
    return "\n".join(parts)


def plot_free_energy_landscape_2d(
    cv1_values: list[float],
    cv2_values: list[float],
    energies: list[float],
    *,
    width: int = 600,
    height: int = 600,
) -> str:
    """SVG heatmap of a 2-D free-energy landscape."""
    n = len(cv1_values)
    if n != len(cv2_values) or n != len(energies):
        raise ContractValidationError("cv1_values, cv2_values, and energies must have equal length.")
    if not n:
        raise ContractValidationError("Input arrays must be non-empty.")

    parts: list[str] = []
    parts.append(_svg_header(width, height, "Free Energy Landscape"))
    parts.append(_svg_rect(0, 0, float(width), float(height), "white"))
    parts.append(_svg_text(width / 2.0, 20.0, "Free Energy Landscape", font_size=14))

    pw = width - _ML - _MR
    ph = height - _MT - _MB

    # Build 2-D grid via binning
    n_grid = int(sqrt(n)) if n >= 4 else max(1, int(sqrt(n)))
    n_grid = max(2, min(n_grid, 50))

    cv1_min, cv1_max = min(cv1_values), max(cv1_values)
    cv2_min, cv2_max = min(cv2_values), max(cv2_values)
    if cv1_max == cv1_min:
        cv1_max = cv1_min + 1.0
    if cv2_max == cv2_min:
        cv2_max = cv2_min + 1.0

    bw1 = (cv1_max - cv1_min) / n_grid
    bw2 = (cv2_max - cv2_min) / n_grid

    grid: list[list[list[float]]] = [[[] for _ in range(n_grid)] for _ in range(n_grid)]
    for k in range(n):
        i = min(int((cv1_values[k] - cv1_min) / bw1), n_grid - 1)
        j = min(int((cv2_values[k] - cv2_min) / bw2), n_grid - 1)
        grid[i][j].append(energies[k])

    e_min_global = min(energies)
    e_max_global = max(energies)
    e_range = e_max_global - e_min_global if e_max_global != e_min_global else 1.0

    cell_w = pw / n_grid
    cell_h = ph / n_grid

    for i in range(n_grid):
        for j in range(n_grid):
            if grid[i][j]:
                val = sum(grid[i][j]) / len(grid[i][j])
            else:
                val = e_max_global
            frac = (val - e_min_global) / e_range
            # Blue (low) -> Red (high)
            r = int(min(255, max(0, frac * 255)))
            b = int(min(255, max(0, (1 - frac) * 255)))
            g = int(min(255, max(0, (1 - 2 * abs(frac - 0.5)) * 200)))
            color = f"#{r:02x}{g:02x}{b:02x}"
            x = _ML + i * cell_w
            # Flip j so CV2 increases upward
            y = _MT + (n_grid - 1 - j) * cell_h
            parts.append(_svg_rect(x, y, cell_w + 0.5, cell_h + 0.5, color))

    # Axis labels
    parts.append(_svg_text(width / 2.0, float(height - 5), "CV1", font_size=11))
    parts.append(
        f'<text x="15" y="{height / 2}" font-size="11" text-anchor="middle"'
        f' font-family="monospace" transform="rotate(-90 15 {height / 2})">CV2</text>'
    )

    # Color bar
    cb_x = float(_ML + pw + 5)
    cb_w = 12.0
    for k in range(20):
        frac = k / 19.0
        r = int(frac * 255)
        b = int((1 - frac) * 255)
        g = int((1 - 2 * abs(frac - 0.5)) * 200)
        color = f"#{r:02x}{g:02x}{b:02x}"
        cy = _MT + (19 - k) / 20.0 * ph
        parts.append(_svg_rect(cb_x, cy, cb_w, ph / 20.0 + 0.5, color))

    parts.append(_close_svg())
    return "\n".join(parts)


def plot_rmsf_per_residue(
    particle_indices: list[int],
    rmsf_values: list[float],
    *,
    width: int = 800,
    height: int = 300,
) -> str:
    """SVG bar chart of RMSF per particle/residue."""
    n = len(particle_indices)
    if n != len(rmsf_values):
        raise ContractValidationError("particle_indices and rmsf_values must have equal length.")
    if not n:
        raise ContractValidationError("Input arrays must be non-empty.")

    parts: list[str] = []
    parts.append(_svg_header(width, height, "RMSF per Residue"))
    parts.append(_svg_rect(0, 0, float(width), float(height), "white"))
    parts.append(_svg_text(width / 2.0, 20.0, "RMSF per Residue", font_size=14))

    pw = width - _ML - _MR
    ph = height - _MT - _MB

    max_rmsf = max(rmsf_values) if rmsf_values else 1.0
    if max_rmsf == 0.0:
        max_rmsf = 1.0

    bar_w = pw / n

    # Y axis
    parts.append(_svg_line(float(_ML), float(_MT), float(_ML), float(_MT + ph), "black"))
    parts.append(_svg_line(float(_ML), float(_MT + ph), float(_ML + pw), float(_MT + ph), "black"))

    for i in range(n):
        bar_h = (rmsf_values[i] / max_rmsf) * ph
        x = _ML + i * bar_w
        y = _MT + ph - bar_h
        parts.append(_svg_rect(x, y, max(bar_w - 1, 1), bar_h, "#42A5F5"))

    # X label
    parts.append(_svg_text(width / 2.0, float(height - 5), "Residue Index", font_size=11))
    parts.append(
        f'<text x="15" y="{height / 2}" font-size="11" text-anchor="middle"'
        f' font-family="monospace" transform="rotate(-90 15 {height / 2})">RMSF (nm)</text>'
    )

    # Y ticks
    y_ticks = _nice_ticks(0.0, max_rmsf)
    for tv in y_ticks:
        py = _MT + ph - (tv / max_rmsf) * ph
        parts.append(_svg_line(float(_ML - 5), py, float(_ML), py, "black"))
        parts.append(_svg_text(float(_ML - 8), py + 3, f"{tv:.3g}", font_size=9, anchor="end"))

    parts.append(_close_svg())
    return "\n".join(parts)


def plot_binding_energy_decomposition(
    components: dict[str, float],
    *,
    width: int = 600,
    height: int = 400,
) -> str:
    """SVG stacked bar chart of binding energy components."""
    if not components:
        raise ContractValidationError("components dict must be non-empty.")

    parts: list[str] = []
    parts.append(_svg_header(width, height, "Binding Energy Decomposition"))
    parts.append(_svg_rect(0, 0, float(width), float(height), "white"))
    parts.append(_svg_text(width / 2.0, 20.0, "Binding Energy Decomposition", font_size=14))

    pw = width - _ML - _MR
    ph = height - _MT - _MB

    labels = list(components.keys())
    values = list(components.values())
    n = len(labels)

    abs_max = max(abs(v) for v in values) if values else 1.0
    if abs_max == 0.0:
        abs_max = 1.0

    bar_w = pw / max(n, 1)
    zero_y = _MT + ph / 2.0  # zero line in middle

    colors = ["#42A5F5", "#66BB6A", "#EF5350", "#FFA726", "#AB47BC", "#26C6DA", "#EC407A"]

    # Axes
    parts.append(_svg_line(float(_ML), float(_MT), float(_ML), float(_MT + ph), "black"))
    parts.append(_svg_line(float(_ML), zero_y, float(_ML + pw), zero_y, "#999999", 1, "3,3"))

    for i in range(n):
        v = values[i]
        bar_h = abs(v) / abs_max * (ph / 2.0)
        x = _ML + i * bar_w + 2
        if v >= 0:
            y = zero_y - bar_h
        else:
            y = zero_y
        color = colors[i % len(colors)]
        parts.append(_svg_rect(x, y, max(bar_w - 4, 4), bar_h, color))
        # Label
        label_y = float(_MT + ph + 18)
        label_x = x + (bar_w - 4) / 2.0
        parts.append(_svg_text(label_x, label_y, labels[i], font_size=8))
        # Value on bar
        val_y = y - 5 if v >= 0 else y + bar_h + 12
        parts.append(_svg_text(label_x, val_y, f"{v:.2f}", font_size=8))

    parts.append(_svg_text(width / 2.0, float(height - 5), "Component", font_size=11))
    parts.append(
        f'<text x="15" y="{height / 2}" font-size="11" text-anchor="middle"'
        f' font-family="monospace" transform="rotate(-90 15 {height / 2})">Energy (kJ/mol)</text>'
    )

    parts.append(_close_svg())
    return "\n".join(parts)


# ===================================================================
# Full HTML Report Generator
# ===================================================================


def generate_full_html_report(
    analysis_report: object,
    *,
    energy_tracker: object | None = None,
) -> str:
    """Generate a self-contained HTML page with inline SVG plots.

    Parameters
    ----------
    analysis_report:
        An ``AnalysisReport`` instance (from ``adaptive_analysis.py``).
    energy_tracker:
        Optional object carrying ``ke_history``, ``pe_history``,
        ``total_history``, and ``temperature_history`` lists of
        ``(step, value)`` tuples.
    """
    sections: list[str] = []

    # --- header ---
    sections.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    sections.append("<title>NeuroCGMD Analysis Report</title>")
    sections.append("<style>")
    sections.append("body{font-family:monospace;margin:20px;background:#fafafa}")
    sections.append("h1{color:#333}h2{color:#555;border-bottom:1px solid #ddd;padding-bottom:4px}")
    sections.append("table{border-collapse:collapse;margin:10px 0}td,th{border:1px solid #ccc;padding:6px 12px}")
    sections.append("th{background:#eee}.section{margin-bottom:30px}")
    sections.append("</style></head><body>")
    sections.append("<h1>NeuroCGMD Adaptive Analysis Report</h1>")

    # --- summary table ---
    sections.append('<div class="section"><h2>Summary</h2><table>')
    steps = getattr(analysis_report, "steps_analyzed", 0)
    convergence = getattr(analysis_report, "convergence_metrics", {})
    converged = "Yes" if hasattr(analysis_report, "is_converged") and analysis_report.is_converged() else "No"
    sections.append(f"<tr><th>Steps Analyzed</th><td>{steps}</td></tr>")
    sections.append(f"<tr><th>Converged</th><td>{converged}</td></tr>")

    rmsd_summary = getattr(analysis_report, "rmsd_summary", {})
    if rmsd_summary:
        mean_rmsd = rmsd_summary.get("mean_rmsd", "N/A")
        std_rmsd = rmsd_summary.get("std_rmsd", "N/A")
        sections.append(f"<tr><th>Mean RMSD</th><td>{mean_rmsd}</td></tr>")
        sections.append(f"<tr><th>Std RMSD</th><td>{std_rmsd}</td></tr>")

    for key, val in convergence.items():
        sections.append(f"<tr><th>{key}</th><td>{val}</td></tr>")

    sections.append("</table></div>")

    # --- energy plot ---
    if energy_tracker is not None:
        ke_hist = getattr(energy_tracker, "ke_history", [])
        pe_hist = getattr(energy_tracker, "pe_history", [])
        total_hist = getattr(energy_tracker, "total_history", [])
        temp_hist = getattr(energy_tracker, "temperature_history", [])

        if ke_hist and pe_hist and total_hist:
            e_steps = [float(s) for s, _ in ke_hist]
            ke_vals = [float(v) for _, v in ke_hist]
            pe_vals = [float(v) for _, v in pe_hist]
            tot_vals = [float(v) for _, v in total_hist]
            sections.append('<div class="section"><h2>Energy</h2>')
            sections.append(plot_energy_timeseries(e_steps, ke_vals, pe_vals, tot_vals))
            sections.append("</div>")

        if temp_hist:
            t_steps = [float(s) for s, _ in temp_hist]
            t_vals = [float(v) for _, v in temp_hist]
            sections.append('<div class="section"><h2>Temperature</h2>')
            sections.append(plot_temperature_timeseries(t_steps, t_vals))
            sections.append("</div>")

    # --- RMSD plot ---
    rmsd_data = getattr(analysis_report, "rmsd_summary", {})
    rmsd_hist = rmsd_data.get("history", [])
    if rmsd_hist:
        r_steps = [float(s) for s, _ in rmsd_hist]
        r_vals = [float(v) for _, v in rmsd_hist]
        sections.append('<div class="section"><h2>RMSD</h2>')
        sections.append(plot_rmsd_timeseries(r_steps, r_vals))
        sections.append("</div>")

    # --- RMSF bar chart ---
    rmsf_data = rmsd_data.get("rmsf", [])
    if rmsf_data:
        indices = list(range(len(rmsf_data)))
        sections.append('<div class="section"><h2>RMSF per Residue</h2>')
        sections.append(plot_rmsf_per_residue(indices, list(rmsf_data)))
        sections.append("</div>")

    # --- RDF plot ---
    rdf_data = getattr(analysis_report, "rdf_data", {})
    r_vals_rdf = rdf_data.get("r_values", [])
    g_vals = rdf_data.get("g_r_values", [])
    if r_vals_rdf and g_vals:
        sections.append('<div class="section"><h2>Radial Distribution Function</h2>')
        sections.append(plot_rdf(list(r_vals_rdf), list(g_vals)))
        sections.append("</div>")

    # --- PMF / Reaction coordinate ---
    pmf_data = getattr(analysis_report, "pmf_data", {})
    pmf_cv = pmf_data.get("cv_values", [])
    pmf_vals = pmf_data.get("pmf_values", [])
    if pmf_cv and pmf_vals:
        sections.append('<div class="section"><h2>Potential of Mean Force</h2>')
        sections.append(plot_pmf(list(pmf_cv), list(pmf_vals)))
        sections.append("</div>")

        # Build reaction coordinate
        try:
            gen = ReactionCoordinateGenerator()
            rc = gen.from_pmf(tuple(pmf_cv), tuple(pmf_vals))
            sections.append('<div class="section"><h2>Reaction Coordinate</h2>')
            sections.append(plot_reaction_coordinate(rc))
            sections.append("</div>")
        except (ContractValidationError, ValueError, ZeroDivisionError):
            pass

    # --- Binding energy decomposition ---
    binding = getattr(analysis_report, "binding_energy", {})
    if binding and isinstance(binding, dict):
        plot_components: dict[str, float] = {}
        for k, v in binding.items():
            if isinstance(v, (int, float)):
                plot_components[k] = float(v)
        if plot_components:
            sections.append('<div class="section"><h2>Binding Energy Decomposition</h2>')
            sections.append(plot_binding_energy_decomposition(plot_components))
            sections.append("</div>")

    # --- Transition state table ---
    pmf_ts: list[TransitionState] = []
    if pmf_cv and pmf_vals:
        try:
            gen = ReactionCoordinateGenerator()
            rc = gen.from_pmf(tuple(pmf_cv), tuple(pmf_vals))
            pmf_ts = list(rc.transition_states)
        except (ContractValidationError, ValueError, ZeroDivisionError):
            pass

    if pmf_ts:
        sections.append('<div class="section"><h2>Detected Transition States</h2><table>')
        sections.append("<tr><th>CV at TS</th><th>Barrier (kJ/mol)</th><th>Fwd Rate</th><th>Rev Rate</th>"
                        "<th>Reactant Basin</th><th>Product Basin</th></tr>")
        for ts in pmf_ts:
            sections.append(
                f"<tr><td>{ts.cv_value_at_transition:.4f}</td>"
                f"<td>{ts.energy_barrier:.2f}</td>"
                f"<td>{ts.forward_rate:.4e}</td>"
                f"<td>{ts.reverse_rate:.4e}</td>"
                f"<td>{ts.reactant_basin[0]:.3f} - {ts.reactant_basin[1]:.3f}</td>"
                f"<td>{ts.product_basin[0]:.3f} - {ts.product_basin[1]:.3f}</td></tr>"
            )
        sections.append("</table></div>")

    sections.append("</body></html>")
    return "\n".join(sections)
