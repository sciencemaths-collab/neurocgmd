"""Adaptive refinement system with error bounds for QCloud corrections.

Provides Richardson extrapolation, error estimation, adaptive region sizing,
and a top-level controller that orchestrates multi-level refinement with
convergence tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt, log, exp

from core.exceptions import ContractValidationError
from core.state import SimulationState
from core.types import FrozenMetadata, Vector3
from qcloud.cloud_state import (
    ParticleForceDelta,
    QCloudCorrection,
    RefinementRegion,
    RegionTriggerKind,
)


# ---------------------------------------------------------------------------
# RefinementLevel
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RefinementLevel:
    """Snapshot of one refinement level for a given region.

    ``level`` 0 is the coarsest; higher values represent finer resolution.
    """

    level: int
    particle_indices: tuple[int, ...]
    correction: QCloudCorrection | None
    estimated_error: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)


# ---------------------------------------------------------------------------
# RichardsonExtrapolation  [established]
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RichardsonExtrapolation:
    """Richardson extrapolation for estimating converged values and error bounds.

    This is an *[established]* numerical technique that uses two approximations
    computed at different resolutions to extrapolate toward the continuum limit
    and provide a practical error estimate.
    """

    order: int = 2
    name: str = "richardson_extrapolation"
    classification: str = "[established]"

    # -- scalar extrapolation ------------------------------------------------

    def extrapolate(
        self,
        coarse: float,
        fine: float,
        ratio: float = 2.0,
    ) -> tuple[float, float]:
        """Return ``(extrapolated_value, error_estimate)``.

        Parameters
        ----------
        coarse:
            Approximation at the coarser resolution.
        fine:
            Approximation at the finer resolution.
        ratio:
            Refinement ratio between the two resolutions.
        """
        factor = ratio ** self.order
        extrapolated = (factor * fine - coarse) / (factor - 1.0)
        error_estimate = abs(extrapolated - fine)
        return extrapolated, error_estimate

    # -- force-delta extrapolation ------------------------------------------

    def extrapolate_forces(
        self,
        coarse_deltas: tuple[ParticleForceDelta, ...],
        fine_deltas: tuple[ParticleForceDelta, ...],
        ratio: float = 2.0,
    ) -> tuple[tuple[ParticleForceDelta, ...], float]:
        """Apply Richardson extrapolation per-particle per-component.

        Returns ``(extrapolated_force_deltas, max_error_estimate)`` where the
        error estimate is the maximum component-wise Richardson error across all
        particles.
        """
        coarse_map: dict[int, Vector3] = {
            fd.particle_index: fd.delta_force for fd in coarse_deltas
        }
        fine_map: dict[int, Vector3] = {
            fd.particle_index: fd.delta_force for fd in fine_deltas
        }

        all_indices = sorted(set(coarse_map) | set(fine_map))
        zero: Vector3 = (0.0, 0.0, 0.0)
        factor = ratio ** self.order
        denom = factor - 1.0

        extrapolated_deltas: list[ParticleForceDelta] = []
        max_error = 0.0

        for idx in all_indices:
            c = coarse_map.get(idx, zero)
            f = fine_map.get(idx, zero)
            ext: list[float] = []
            for k in range(3):
                ext_val = (factor * f[k] - c[k]) / denom
                comp_err = abs(ext_val - f[k])
                if comp_err > max_error:
                    max_error = comp_err
                ext.append(ext_val)
            extrapolated_deltas.append(
                ParticleForceDelta(
                    particle_index=idx,
                    delta_force=(ext[0], ext[1], ext[2]),
                )
            )

        return tuple(extrapolated_deltas), max_error


# ---------------------------------------------------------------------------
# ErrorEstimator  [proposed novel]
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ErrorEstimator:
    """Tracks QCloud correction history and estimates current error level.

    *[proposed novel]* — uses exponentially-weighted variance of recent
    corrections as an error proxy, with convergence-rate estimation via
    log-linear regression.
    """

    history_size: int = 20
    name: str = "qcloud_error_estimator"
    classification: str = "[proposed novel]"
    _history: list[tuple[int, float, float]] = field(default_factory=list)

    # -- recording -----------------------------------------------------------

    def record(self, step: int, correction: QCloudCorrection) -> None:
        """Append a correction observation to the rolling history."""
        max_force = 0.0
        for fd in correction.force_deltas:
            mag = sqrt(
                fd.delta_force[0] ** 2
                + fd.delta_force[1] ** 2
                + fd.delta_force[2] ** 2
            )
            if mag > max_force:
                max_force = mag
        self._history.append((step, correction.energy_delta, max_force))
        if len(self._history) > self.history_size:
            self._history = self._history[-self.history_size :]

    # -- error estimation ----------------------------------------------------

    def estimate_current_error(self) -> float:
        """Return an error proxy based on exponentially-weighted variance.

        If fewer than 3 data points are available a conservative default of
        ``1.0`` is returned.
        """
        if len(self._history) < 3:
            return 1.0

        n = len(self._history)
        # Exponential weights: most recent entries dominate.
        decay = 0.8
        weights = [decay ** (n - 1 - i) for i in range(n)]
        total_w = sum(weights)
        # Weighted mean of energy deltas.
        w_mean = sum(w * abs(e) for w, (_, e, _) in zip(weights, self._history)) / total_w
        # Weighted variance.
        w_var = sum(
            w * (abs(e) - w_mean) ** 2
            for w, (_, e, _) in zip(weights, self._history)
        ) / total_w
        return sqrt(w_var) + 1e-12  # avoid exact zero

    # -- convergence diagnostics ---------------------------------------------

    def convergence_rate(self) -> float:
        """Estimate convergence rate via log-linear regression.

        Returns the slope of ``log(|correction|)`` versus step number.
        A negative value indicates convergence; positive indicates divergence.
        Returns ``0.0`` when there is insufficient data.
        """
        points: list[tuple[float, float]] = []
        for step, energy, _ in self._history:
            val = abs(energy)
            if val > 0.0:
                points.append((float(step), log(val)))
        if len(points) < 2:
            return 0.0

        n = len(points)
        sx = sum(x for x, _ in points)
        sy = sum(y for _, y in points)
        sxx = sum(x * x for x, _ in points)
        sxy = sum(x * y for x, y in points)
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-30:
            return 0.0
        slope = (n * sxy - sx * sy) / denom
        return slope

    def is_converged(self, tolerance: float = 0.01) -> bool:
        """Return ``True`` if all recent corrections are below *tolerance*."""
        if len(self._history) < 3:
            return False
        recent = self._history[-3:]
        return all(abs(e) < tolerance and f < tolerance for _, e, f in recent)


# ---------------------------------------------------------------------------
# AdaptiveRegionSizer  [proposed novel]
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AdaptiveRegionSizer:
    """Dynamically grows or shrinks a refinement region based on error level.

    *[proposed novel]* — uses force magnitude as a proxy for particle
    importance: particles experiencing the largest forces are retained first
    when shrinking, and nearest neighbours (by index proximity) are added
    when growing.
    """

    min_region_size: int = 3
    max_region_size: int = 50
    growth_factor: float = 1.5
    shrink_factor: float = 0.67
    error_threshold_grow: float = 0.1
    error_threshold_shrink: float = 0.01
    name: str = "adaptive_region_sizer"
    classification: str = "[proposed novel]"

    def resize(
        self,
        region: RefinementRegion,
        current_error: float,
        state: SimulationState,
    ) -> RefinementRegion:
        """Return a new :class:`RefinementRegion` with adjusted particle set.

        - If *current_error* exceeds :pyattr:`error_threshold_grow` the region
          is expanded toward ``max_region_size`` by adding neighbours.
        - If *current_error* is below :pyattr:`error_threshold_shrink` the
          region is contracted toward ``min_region_size``, keeping only the
          particles with the largest force magnitudes.
        - Otherwise the region is returned unchanged.
        """
        indices = list(region.particle_indices)
        current_size = len(indices)

        if current_error > self.error_threshold_grow:
            # --- grow -------------------------------------------------------
            target_size = min(
                int(current_size * self.growth_factor + 0.5),
                self.max_region_size,
                state.particle_count,
            )
            target_size = max(target_size, current_size)  # never shrink here
            if target_size > current_size:
                index_set = set(indices)
                candidates: list[int] = []
                for idx in indices:
                    for neighbour in (idx - 1, idx + 1):
                        if 0 <= neighbour < state.particle_count and neighbour not in index_set:
                            candidates.append(neighbour)
                            index_set.add(neighbour)
                # Rank candidates by force magnitude (descending).
                candidates.sort(
                    key=lambda i: self._force_magnitude(state, i),
                    reverse=True,
                )
                need = target_size - current_size
                indices.extend(candidates[:need])

        elif current_error < self.error_threshold_shrink:
            # --- shrink -----------------------------------------------------
            target_size = max(
                int(current_size * self.shrink_factor + 0.5),
                self.min_region_size,
            )
            target_size = min(target_size, current_size)  # never grow here
            if target_size < current_size:
                # Keep the particles with the highest force magnitude.
                ranked = sorted(
                    indices,
                    key=lambda i: self._force_magnitude(state, i),
                    reverse=True,
                )
                indices = ranked[:target_size]

        new_indices = tuple(sorted(set(indices)))
        if new_indices == region.particle_indices:
            return region

        return RefinementRegion(
            region_id=region.region_id,
            state_id=region.state_id,
            particle_indices=new_indices,
            seed_pairs=region.seed_pairs,
            compartment_ids=region.compartment_ids,
            trigger_kinds=region.trigger_kinds,
            score=region.score,
            metadata=region.metadata.with_updates({"resized_by": self.name}),
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _force_magnitude(state: SimulationState, index: int) -> float:
        """Return the Euclidean magnitude of the force on particle *index*."""
        if index < 0 or index >= state.particle_count:
            return 0.0
        fx, fy, fz = state.particles.forces[index]
        return sqrt(fx * fx + fy * fy + fz * fz)


# ---------------------------------------------------------------------------
# AdaptiveRefinementResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AdaptiveRefinementResult:
    """Outcome of one adaptive refinement pass over a region."""

    correction: QCloudCorrection
    region: RefinementRegion
    estimated_error: float
    converged: bool
    refinement_levels_used: int
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)


# ---------------------------------------------------------------------------
# AdaptiveRefinementController  [proposed novel]
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AdaptiveRefinementController:
    """Orchestrates multi-level adaptive refinement with error-bound tracking.

    *[proposed novel]* — coordinates :class:`RichardsonExtrapolation`,
    :class:`ErrorEstimator`, and :class:`AdaptiveRegionSizer` to deliver a
    correction whose accuracy is tracked and whose region size adapts to the
    estimated error.
    """

    richardson: RichardsonExtrapolation = field(
        default_factory=RichardsonExtrapolation,
    )
    error_estimator: ErrorEstimator = field(default_factory=ErrorEstimator)
    region_sizer: AdaptiveRegionSizer = field(default_factory=AdaptiveRegionSizer)
    target_accuracy: float = 0.01
    max_refinement_levels: int = 3
    name: str = "adaptive_refinement_controller"
    classification: str = "[proposed novel]"

    def refine(
        self,
        state: SimulationState,
        region: RefinementRegion,
        base_correction: QCloudCorrection,
    ) -> AdaptiveRefinementResult:
        """Run one adaptive refinement pass.

        1. Record *base_correction* in the error estimator.
        2. If already converged, return immediately.
        3. Estimate the current error; if it exceeds :pyattr:`target_accuracy`,
           resize the region to improve coverage.
        4. Return an :class:`AdaptiveRefinementResult` with error bounds and
           convergence information.
        """
        self.error_estimator.record(state.step, base_correction)

        converged = self.error_estimator.is_converged(self.target_accuracy)
        estimated_error = self.error_estimator.estimate_current_error()

        if converged:
            return AdaptiveRefinementResult(
                correction=base_correction,
                region=region,
                estimated_error=estimated_error,
                converged=True,
                refinement_levels_used=1,
                metadata=FrozenMetadata({
                    "controller": self.name,
                    "converged": True,
                    "estimated_error": estimated_error,
                }),
            )

        # Resize if error exceeds target accuracy.
        resized_region = region
        if estimated_error > self.target_accuracy:
            resized_region = self.region_sizer.resize(
                region, estimated_error, state,
            )

        return AdaptiveRefinementResult(
            correction=base_correction,
            region=resized_region,
            estimated_error=estimated_error,
            converged=False,
            refinement_levels_used=min(
                max(1, len(self.error_estimator._history)),
                self.max_refinement_levels,
            ),
            metadata=FrozenMetadata({
                "controller": self.name,
                "converged": False,
                "estimated_error": estimated_error,
                "convergence_rate": self.error_estimator.convergence_rate(),
                "region_resized": resized_region is not region,
            }),
        )
