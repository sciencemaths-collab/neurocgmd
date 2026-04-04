"""QCloud-informed structural event detection and adaptive feedback.

Tracks per-particle and per-region correction magnitudes over time.
Detects bond breaking/forming events when corrections spike, and feeds
correction history back to inform future region selection priorities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import sqrt
from typing import Sequence

from core.exceptions import ContractValidationError
from core.types import FrozenMetadata, Vector3


class StructuralEventKind(StrEnum):
    """Types of structural events detected from QCloud correction patterns."""

    BOND_FORMING = "bond_forming"
    BOND_BREAKING = "bond_breaking"
    CONFORMATIONAL_SHIFT = "conformational_shift"
    INTERFACE_REARRANGEMENT = "interface_rearrangement"


@dataclass(frozen=True, slots=True)
class StructuralEvent:
    """A detected structural event inferred from QCloud correction patterns."""

    kind: StructuralEventKind
    step: int
    time: float
    particle_indices: tuple[int, ...]
    correction_magnitude: float
    baseline_magnitude: float
    confidence: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)


@dataclass(frozen=True, slots=True)
class ParticleCorrectionRecord:
    """Rolling correction statistics for one particle."""

    total_corrections: int = 0
    magnitude_sum: float = 0.0
    magnitude_sq_sum: float = 0.0
    max_magnitude: float = 0.0
    last_step: int = 0

    @property
    def mean_magnitude(self) -> float:
        return self.magnitude_sum / self.total_corrections if self.total_corrections > 0 else 0.0

    @property
    def std_magnitude(self) -> float:
        if self.total_corrections < 2:
            return 0.0
        mean = self.mean_magnitude
        var = self.magnitude_sq_sum / self.total_corrections - mean * mean
        return sqrt(max(0.0, var))


@dataclass(frozen=True, slots=True)
class RegionCorrectionFeedback:
    """Feedback for a region to inform future selection priority."""

    particle_indices: tuple[int, ...]
    mean_correction_magnitude: float
    max_correction_magnitude: float
    correction_count: int
    priority_boost: float


@dataclass(slots=True)
class QCloudEventAnalyzer:
    """Analyzes QCloud correction patterns to detect structural events
    and provide adaptive feedback to the region selector.

    Maintains per-particle rolling statistics of correction magnitudes.
    When a correction is significantly larger than the particle's baseline,
    it flags a structural event (bond breaking/forming/conformational shift).
    """

    spike_threshold_sigma: float = 2.5
    min_corrections_for_baseline: int = 3
    correction_decay: float = 0.95
    name: str = "qcloud_event_analyzer"
    classification: str = "[proposed novel]"

    _particle_stats: dict[int, ParticleCorrectionRecord] = field(default_factory=dict)
    _detected_events: list[StructuralEvent] = field(default_factory=list)
    _correction_history: list[dict] = field(default_factory=list)
    _region_correction_totals: dict[tuple[int, ...], float] = field(default_factory=dict)
    _region_correction_counts: dict[tuple[int, ...], int] = field(default_factory=dict)

    def record_qcloud_result(
        self,
        qcloud_result,
        step: int,
        time: float,
    ) -> tuple[StructuralEvent, ...]:
        """Process a QCloudCouplingResult and return any detected structural events.

        Call this after each production cycle where QCloud was applied.
        """
        if qcloud_result is None:
            return ()

        new_events: list[StructuralEvent] = []
        cycle_record = {
            "step": step,
            "time": time,
            "regions": [],
        }

        for correction in qcloud_result.applied_corrections:
            region_particles = correction.affected_particles()
            region_key = tuple(sorted(region_particles))

            # Track per-region totals for feedback
            region_mag = 0.0
            particle_magnitudes: dict[int, float] = {}

            for fd in correction.force_deltas:
                dx, dy, dz = fd.delta_force
                mag = sqrt(dx * dx + dy * dy + dz * dz)
                particle_magnitudes[fd.particle_index] = mag
                region_mag += mag

                # Update particle rolling stats
                old = self._particle_stats.get(fd.particle_index, ParticleCorrectionRecord())
                self._particle_stats[fd.particle_index] = ParticleCorrectionRecord(
                    total_corrections=old.total_corrections + 1,
                    magnitude_sum=old.magnitude_sum * self.correction_decay + mag,
                    magnitude_sq_sum=old.magnitude_sq_sum * self.correction_decay + mag * mag,
                    max_magnitude=max(old.max_magnitude, mag),
                    last_step=step,
                )

                # Check for spike → structural event
                if old.total_corrections >= self.min_corrections_for_baseline:
                    baseline = old.mean_magnitude
                    std = old.std_magnitude
                    threshold = baseline + self.spike_threshold_sigma * max(std, baseline * 0.1)
                    if mag > threshold and mag > 0.01:
                        event_kind = self._classify_event(
                            fd.particle_index, mag, baseline, correction)
                        event = StructuralEvent(
                            kind=event_kind,
                            step=step,
                            time=time,
                            particle_indices=region_key,
                            correction_magnitude=mag,
                            baseline_magnitude=baseline,
                            confidence=correction.confidence,
                            metadata=FrozenMetadata({
                                "particle_index": fd.particle_index,
                                "sigma_excess": (mag - baseline) / max(std, 1e-9),
                                "region_id": str(correction.region_id),
                                "energy_delta": correction.energy_delta,
                            }),
                        )
                        new_events.append(event)
                        self._detected_events.append(event)

            # Update region correction totals
            prev_total = self._region_correction_totals.get(region_key, 0.0)
            prev_count = self._region_correction_counts.get(region_key, 0)
            self._region_correction_totals[region_key] = prev_total * self.correction_decay + region_mag
            self._region_correction_counts[region_key] = prev_count + 1

            cycle_record["regions"].append({
                "particles": list(region_key),
                "total_magnitude": region_mag,
                "energy_delta": correction.energy_delta,
                "confidence": correction.confidence,
                "particle_magnitudes": particle_magnitudes,
            })

        self._correction_history.append(cycle_record)
        return tuple(new_events)

    def _classify_event(
        self,
        particle_index: int,
        magnitude: float,
        baseline: float,
        correction,
    ) -> StructuralEventKind:
        """Classify a correction spike as a specific structural event type."""
        ratio = magnitude / max(baseline, 1e-9)
        energy_delta = abs(correction.energy_delta)

        # Large energy correction + force spike → bond event
        if energy_delta > 1.0 and ratio > 5.0:
            if correction.energy_delta > 0:
                return StructuralEventKind.BOND_BREAKING
            else:
                return StructuralEventKind.BOND_FORMING

        # Multiple particles in region spiking → interface rearrangement
        spike_count = sum(
            1 for fd in correction.force_deltas
            if sqrt(sum(c * c for c in fd.delta_force)) > baseline * 2.0
        )
        if spike_count >= 2:
            return StructuralEventKind.INTERFACE_REARRANGEMENT

        return StructuralEventKind.CONFORMATIONAL_SHIFT

    def get_region_priority_feedback(self) -> tuple[RegionCorrectionFeedback, ...]:
        """Return priority boosts for regions based on correction history.

        Regions with consistently large corrections get higher priority
        for future QCloud evaluation — they represent areas where the
        CG model needs the most help.
        """
        feedback: list[RegionCorrectionFeedback] = []
        for region_key, total_mag in self._region_correction_totals.items():
            count = self._region_correction_counts.get(region_key, 1)
            mean_mag = total_mag / count if count > 0 else 0.0

            # Priority boost scales with correction magnitude
            # Normalize: typical small correction ~0.01, large ~1.0
            boost = min(1.0, mean_mag * 2.0)

            feedback.append(RegionCorrectionFeedback(
                particle_indices=region_key,
                mean_correction_magnitude=mean_mag,
                max_correction_magnitude=max(
                    (self._particle_stats[pi].max_magnitude
                     for pi in region_key if pi in self._particle_stats),
                    default=0.0,
                ),
                correction_count=count,
                priority_boost=boost,
            ))

        return tuple(sorted(feedback, key=lambda f: -f.priority_boost))

    def get_particle_priority_scores(self) -> dict[int, float]:
        """Return per-particle priority scores based on correction history.

        Higher score = this particle has received larger QCloud corrections
        = the CG model is less accurate here = more compute should be focused here.
        """
        scores: dict[int, float] = {}
        if not self._particle_stats:
            return scores
        max_mean = max(
            (s.mean_magnitude for s in self._particle_stats.values()),
            default=1.0,
        )
        if max_mean < 1e-12:
            max_mean = 1.0
        for pi, stats in self._particle_stats.items():
            scores[pi] = stats.mean_magnitude / max_mean
        return scores

    def detected_events(self) -> tuple[StructuralEvent, ...]:
        return tuple(self._detected_events)

    def correction_history(self) -> list[dict]:
        return list(self._correction_history)

    def summary(self) -> dict:
        """Summary statistics for the analysis pipeline."""
        events = self._detected_events
        return {
            "total_events_detected": len(events),
            "bond_forming_events": sum(1 for e in events if e.kind == StructuralEventKind.BOND_FORMING),
            "bond_breaking_events": sum(1 for e in events if e.kind == StructuralEventKind.BOND_BREAKING),
            "conformational_shifts": sum(1 for e in events if e.kind == StructuralEventKind.CONFORMATIONAL_SHIFT),
            "interface_rearrangements": sum(1 for e in events if e.kind == StructuralEventKind.INTERFACE_REARRANGEMENT),
            "particles_with_corrections": len(self._particle_stats),
            "regions_tracked": len(self._region_correction_totals),
            "total_correction_cycles": len(self._correction_history),
        }
