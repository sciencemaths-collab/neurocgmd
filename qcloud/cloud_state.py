"""Bounded data carriers for local quantum-cloud refinement regions and corrections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from core.exceptions import ContractValidationError
from core.interfaces import ValidatableComponent
from core.types import FrozenMetadata, RegionId, StateId, Vector3, coerce_scalar, coerce_vector3


def _normalize_region_pairs(pairs: Sequence[tuple[int, int]] | None) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for source_index, target_index in pairs or ():
        pair = tuple(sorted((int(source_index), int(target_index))))
        if pair[0] == pair[1]:
            raise ContractValidationError("seed_pairs must contain distinct endpoints.")
        if pair not in seen:
            normalized.append(pair)
            seen.add(pair)
    return tuple(normalized)


def _normalize_string_tuple(values: Sequence[str] | None, *, field_name: str) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values or ():
        value = str(raw_value).strip()
        if not value:
            raise ContractValidationError(f"{field_name} must contain non-empty strings.")
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return tuple(normalized)


class RegionTriggerKind(StrEnum):
    """Deterministic trigger labels used to justify local qcloud refinement."""

    ADAPTIVE_EDGE = "adaptive_edge"
    INTER_COMPARTMENT = "inter_compartment"
    MEMORY_PRIORITY = "memory_priority"
    COMPARTMENT_FOCUS = "compartment_focus"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class RefinementRegion(ValidatableComponent):
    """One bounded local region selected for possible quantum-informed correction."""

    region_id: RegionId
    state_id: StateId
    particle_indices: tuple[int, ...]
    seed_pairs: tuple[tuple[int, int], ...] = ()
    compartment_ids: tuple[str, ...] = ()
    trigger_kinds: tuple[RegionTriggerKind, ...] = ()
    score: float = 0.0
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", RegionId(str(self.region_id)))
        object.__setattr__(self, "state_id", StateId(str(self.state_id)))
        object.__setattr__(self, "particle_indices", tuple(sorted(set(int(index) for index in self.particle_indices))))
        object.__setattr__(self, "seed_pairs", _normalize_region_pairs(self.seed_pairs))
        object.__setattr__(
            self,
            "compartment_ids",
            _normalize_string_tuple(self.compartment_ids, field_name="compartment_ids"),
        )
        object.__setattr__(
            self,
            "trigger_kinds",
            tuple(dict.fromkeys(RegionTriggerKind(trigger) for trigger in self.trigger_kinds)),
        )
        object.__setattr__(self, "score", coerce_scalar(self.score, "score"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    @property
    def size(self) -> int:
        return len(self.particle_indices)

    def contains_particle(self, particle_index: int) -> bool:
        return int(particle_index) in self.particle_indices

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not str(self.region_id).strip():
            issues.append("region_id must be a non-empty string.")
        if not str(self.state_id).strip():
            issues.append("state_id must be a non-empty string.")
        if not self.particle_indices:
            issues.append("particle_indices must contain at least one particle.")
        if any(index < 0 for index in self.particle_indices):
            issues.append("particle_indices must be non-negative.")
        for source_index, target_index in self.seed_pairs:
            if source_index < 0 or target_index < 0:
                issues.append("seed_pairs must be non-negative.")
        if not self.trigger_kinds:
            issues.append("trigger_kinds must contain at least one trigger.")
        if self.score < 0.0:
            issues.append("score must be non-negative.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "region_id": str(self.region_id),
            "state_id": str(self.state_id),
            "particle_indices": list(self.particle_indices),
            "seed_pairs": [list(pair) for pair in self.seed_pairs],
            "compartment_ids": list(self.compartment_ids),
            "trigger_kinds": [trigger.value for trigger in self.trigger_kinds],
            "score": self.score,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RefinementRegion":
        return cls(
            region_id=RegionId(str(data["region_id"])),
            state_id=StateId(str(data["state_id"])),
            particle_indices=tuple(int(index) for index in data.get("particle_indices", ())),
            seed_pairs=tuple(tuple(pair) for pair in data.get("seed_pairs", ())),
            compartment_ids=tuple(str(identifier) for identifier in data.get("compartment_ids", ())),
            trigger_kinds=tuple(RegionTriggerKind(str(trigger)) for trigger in data.get("trigger_kinds", ())),
            score=float(data.get("score", 0.0)),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ParticleForceDelta(ValidatableComponent):
    """Local force correction for one particle contributed by a qcloud region."""

    particle_index: int
    delta_force: Vector3
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "particle_index", int(self.particle_index))
        object.__setattr__(self, "delta_force", coerce_vector3(self.delta_force, "delta_force"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        return ("particle_index must be non-negative.",) if self.particle_index < 0 else ()

    def to_dict(self) -> dict[str, object]:
        return {
            "particle_index": self.particle_index,
            "delta_force": list(self.delta_force),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ParticleForceDelta":
        return cls(
            particle_index=int(data["particle_index"]),
            delta_force=data["delta_force"],
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class QCloudCorrection(ValidatableComponent):
    """Bounded local correction payload produced for one refinement region."""

    region_id: RegionId
    method_label: str
    energy_delta: float
    force_deltas: tuple[ParticleForceDelta, ...] = ()
    confidence: float = 1.0
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", RegionId(str(self.region_id)))
        object.__setattr__(self, "energy_delta", coerce_scalar(self.energy_delta, "energy_delta"))
        object.__setattr__(self, "force_deltas", tuple(self.force_deltas))
        object.__setattr__(self, "confidence", coerce_scalar(self.confidence, "confidence"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def affected_particles(self) -> tuple[int, ...]:
        return tuple(force_delta.particle_index for force_delta in self.force_deltas)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not str(self.region_id).strip():
            issues.append("region_id must be a non-empty string.")
        if not self.method_label.strip():
            issues.append("method_label must be a non-empty string.")
        if not (0.0 <= self.confidence <= 1.0):
            issues.append("confidence must lie in the interval [0, 1].")
        affected_particles = self.affected_particles()
        if len(affected_particles) != len(set(affected_particles)):
            issues.append("force_deltas must not contain duplicate particle_index values.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "region_id": str(self.region_id),
            "method_label": self.method_label,
            "energy_delta": self.energy_delta,
            "force_deltas": [force_delta.to_dict() for force_delta in self.force_deltas],
            "confidence": self.confidence,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "QCloudCorrection":
        return cls(
            region_id=RegionId(str(data["region_id"])),
            method_label=str(data["method_label"]),
            energy_delta=float(data["energy_delta"]),
            force_deltas=tuple(
                ParticleForceDelta.from_dict(force_delta)
                for force_delta in data.get("force_deltas", ())
            ),
            confidence=float(data.get("confidence", 1.0)),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )
