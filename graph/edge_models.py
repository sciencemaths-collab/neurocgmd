"""Dynamic edge models for the adaptive connectivity graph layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from core.exceptions import ContractValidationError
from core.interfaces import ValidatableComponent
from core.types import FrozenMetadata


class DynamicEdgeKind(StrEnum):
    """Kinds of graph edges layered on top of fixed topology."""

    STRUCTURAL_LOCAL = "structural_local"
    ADAPTIVE_LOCAL = "adaptive_local"
    ADAPTIVE_LONG_RANGE = "adaptive_long_range"


@dataclass(frozen=True, slots=True)
class DynamicEdgeState(ValidatableComponent):
    """One undirected adaptive-graph edge between two particle indices."""

    source_index: int
    target_index: int
    kind: DynamicEdgeKind
    weight: float
    distance: float
    created_step: int
    last_updated_step: int
    active: bool = True
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        source_index, target_index = sorted((self.source_index, self.target_index))
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
            issues.append("Dynamic edge indices must be non-negative.")
        if self.source_index == self.target_index:
            issues.append("Dynamic edge endpoints must be distinct.")
        if not (0.0 < self.weight <= 1.0):
            issues.append("Dynamic edge weight must lie in the interval (0, 1].")
        if self.distance < 0.0:
            issues.append("Dynamic edge distance must be non-negative.")
        if self.created_step < 0 or self.last_updated_step < 0:
            issues.append("Dynamic edge step indices must be non-negative.")
        if self.last_updated_step < self.created_step:
            issues.append("last_updated_step cannot be less than created_step.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_index": self.source_index,
            "target_index": self.target_index,
            "kind": self.kind.value,
            "weight": self.weight,
            "distance": self.distance,
            "created_step": self.created_step,
            "last_updated_step": self.last_updated_step,
            "active": self.active,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "DynamicEdgeState":
        return cls(
            source_index=int(data["source_index"]),
            target_index=int(data["target_index"]),
            kind=DynamicEdgeKind(data["kind"]),
            weight=float(data["weight"]),
            distance=float(data["distance"]),
            created_step=int(data["created_step"]),
            last_updated_step=int(data["last_updated_step"]),
            active=bool(data.get("active", True)),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )

