"""Foundational pair-trace memory used by plasticity rules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.interfaces import ValidatableComponent
from core.types import FrozenMetadata
from graph.adjacency_utils import build_edge_lookup
from graph.graph_manager import ConnectivityGraph


def normalize_pair(pair: tuple[int, int]) -> tuple[int, int]:
    """Return a stable undirected pair key."""

    source_index, target_index = sorted(pair)
    return (source_index, target_index)


@dataclass(frozen=True, slots=True)
class PairTraceState(ValidatableComponent):
    """Minimal pair-level trace before the dedicated memory subsystem exists."""

    source_index: int
    target_index: int
    activity_level: float
    coactivity_level: float
    persistence: int
    last_seen_step: int
    seen_count: int
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        source_index, target_index = normalize_pair((self.source_index, self.target_index))
        object.__setattr__(self, "source_index", source_index)
        object.__setattr__(self, "target_index", target_index)
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def normalized_pair(self) -> tuple[int, int]:
        return (self.source_index, self.target_index)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.source_index < 0 or self.target_index < 0:
            issues.append("PairTraceState indices must be non-negative.")
        if self.source_index == self.target_index:
            issues.append("PairTraceState endpoints must be distinct.")
        for name, value in (
            ("activity_level", self.activity_level),
            ("coactivity_level", self.coactivity_level),
        ):
            if not (0.0 <= value <= 1.0):
                issues.append(f"{name} must lie in the interval [0, 1].")
        if self.persistence < 0:
            issues.append("persistence must be non-negative.")
        if self.last_seen_step < 0 or self.seen_count < 0:
            issues.append("last_seen_step and seen_count must be non-negative.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_index": self.source_index,
            "target_index": self.target_index,
            "activity_level": self.activity_level,
            "coactivity_level": self.coactivity_level,
            "persistence": self.persistence,
            "last_seen_step": self.last_seen_step,
            "seen_count": self.seen_count,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PairTraceState":
        return cls(
            source_index=int(data["source_index"]),
            target_index=int(data["target_index"]),
            activity_level=float(data["activity_level"]),
            coactivity_level=float(data["coactivity_level"]),
            persistence=int(data["persistence"]),
            last_seen_step=int(data["last_seen_step"]),
            seen_count=int(data["seen_count"]),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


def build_trace_lookup(
    traces: tuple[PairTraceState, ...] | list[PairTraceState],
) -> dict[tuple[int, int], PairTraceState]:
    """Return a lookup from normalized pair to trace state."""

    lookup: dict[tuple[int, int], PairTraceState] = {}
    for trace in traces:
        pair = trace.normalized_pair()
        if pair in lookup:
            raise ContractValidationError(f"Duplicate plasticity trace detected for pair {pair}.")
        lookup[pair] = trace
    return lookup


def update_pair_traces(
    *,
    current_step: int,
    graph: ConnectivityGraph,
    previous_traces: tuple[PairTraceState, ...] = (),
    activity_signals: Mapping[tuple[int, int], float] | None = None,
    decay: float = 0.75,
    retention_floor: float = 0.01,
) -> tuple[PairTraceState, ...]:
    """Update pair-level traces from activity signals and the active graph snapshot."""

    if not (0.0 <= decay <= 1.0):
        raise ContractValidationError("decay must lie in the interval [0, 1].")
    if not (0.0 <= retention_floor <= 1.0):
        raise ContractValidationError("retention_floor must lie in the interval [0, 1].")

    previous_lookup = build_trace_lookup(previous_traces)
    signal_lookup = {
        normalize_pair(pair): max(0.0, min(1.0, float(value)))
        for pair, value in (activity_signals or {}).items()
    }
    active_lookup = build_edge_lookup(graph.active_edges())

    candidate_pairs = set(previous_lookup) | set(signal_lookup) | set(active_lookup)
    updated_traces: list[PairTraceState] = []
    for pair in sorted(candidate_pairs):
        previous = previous_lookup.get(pair)
        activity_signal = signal_lookup.get(pair, 0.0)
        currently_linked = pair in active_lookup
        previous_activity = previous.activity_level if previous else 0.0
        previous_coactivity = previous.coactivity_level if previous else 0.0
        new_activity = decay * previous_activity + (1.0 - decay) * activity_signal
        coactivity_signal = activity_signal if currently_linked else 0.5 * activity_signal
        new_coactivity = decay * previous_coactivity + (1.0 - decay) * coactivity_signal
        previous_persistence = previous.persistence if previous else 0
        persistence = previous_persistence + 1 if activity_signal > 0.0 else max(0, previous_persistence - 1)
        seen_count = (previous.seen_count if previous else 0) + int(activity_signal > 0.0 or currently_linked)
        last_seen_step = (
            current_step
            if activity_signal > 0.0 or currently_linked
            else (previous.last_seen_step if previous else current_step)
        )

        if currently_linked or new_activity >= retention_floor or new_coactivity >= retention_floor:
            updated_traces.append(
                PairTraceState(
                    source_index=pair[0],
                    target_index=pair[1],
                    activity_level=new_activity,
                    coactivity_level=new_coactivity,
                    persistence=persistence,
                    last_seen_step=last_seen_step,
                    seen_count=seen_count,
                    metadata={
                        "linked": currently_linked,
                        "signal": activity_signal,
                    },
                )
            )

    return tuple(updated_traces)

