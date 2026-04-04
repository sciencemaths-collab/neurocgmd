"""Uncertainty estimation for learned residual predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, StateId, coerce_scalar
from memory.replay_buffer import ReplayItem
from memory.trace_store import TraceRecord
from ml.live_features import LiveFeatureVector
from ml.residual_model import ResidualPrediction

_PRIORITY_TAGS = frozenset({"priority", "instability", "qcloud", "refine"})


@dataclass(frozen=True, slots=True)
class UncertaintyEstimate(ValidatableComponent):
    """Structured uncertainty estimate for one residual prediction."""

    state_id: StateId
    energy_uncertainty: float
    force_uncertainty: float
    total_uncertainty: float
    trigger_qcloud: bool
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", StateId(str(self.state_id)))
        object.__setattr__(self, "energy_uncertainty", coerce_scalar(self.energy_uncertainty, "energy_uncertainty"))
        object.__setattr__(self, "force_uncertainty", coerce_scalar(self.force_uncertainty, "force_uncertainty"))
        object.__setattr__(self, "total_uncertainty", coerce_scalar(self.total_uncertainty, "total_uncertainty"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for field_name in ("energy_uncertainty", "force_uncertainty", "total_uncertainty"):
            value = getattr(self, field_name)
            if not (0.0 <= value <= 1.0):
                issues.append(f"{field_name} must lie in the interval [0, 1].")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "state_id": str(self.state_id),
            "energy_uncertainty": self.energy_uncertainty,
            "force_uncertainty": self.force_uncertainty,
            "total_uncertainty": self.total_uncertainty,
            "trigger_qcloud": self.trigger_qcloud,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "UncertaintyEstimate":
        return cls(
            state_id=StateId(str(data["state_id"])),
            energy_uncertainty=float(data["energy_uncertainty"]),
            force_uncertainty=float(data["force_uncertainty"]),
            total_uncertainty=float(data["total_uncertainty"]),
            trigger_qcloud=bool(data["trigger_qcloud"]),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@runtime_checkable
class UncertaintyModel(Protocol):
    """Protocol for uncertainty estimators layered above residual predictions."""

    name: str
    classification: str

    def estimate(
        self,
        prediction: ResidualPrediction,
        *,
        trace_record: TraceRecord | None = None,
        replay_item: ReplayItem | None = None,
        live_features: LiveFeatureVector | None = None,
    ) -> UncertaintyEstimate:
        """Estimate uncertainty for a residual prediction."""


@dataclass(slots=True)
class HeuristicUncertaintyModel(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Deterministic uncertainty estimator used before calibrated ML uncertainty exists."""

    base_uncertainty: float = 0.1
    low_confidence_scale: float = 0.7
    force_delta_scale: float = 0.05
    priority_tag_bonus: float = 0.15
    high_score_threshold: float = 1.0
    high_score_bonus: float = 0.1
    chemistry_mismatch_bonus: float = 0.22
    chemistry_flexibility_scale: float = 0.18
    structure_drift_scale: float = 0.20
    shadow_regression_bonus: float = 0.16
    trigger_threshold: float = 0.6
    name: str = "heuristic_uncertainty_model"
    classification: str = "[adapted]"

    def describe_role(self) -> str:
        return (
            "Provides a deterministic uncertainty estimate for learned residual predictions "
            "before calibrated uncertainty models are introduced."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "ml/residual_model.py",
            "memory/replay_buffer.py",
            "memory/trace_store.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/online_ml_residual_learning.md",
            "docs/sections/section_11_online_ml_residual_learning.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for field_name in (
            "base_uncertainty",
            "low_confidence_scale",
            "force_delta_scale",
            "priority_tag_bonus",
            "high_score_bonus",
            "chemistry_mismatch_bonus",
            "chemistry_flexibility_scale",
            "structure_drift_scale",
            "shadow_regression_bonus",
            "trigger_threshold",
        ):
            value = getattr(self, field_name)
            if value < 0.0:
                issues.append(f"{field_name} must be non-negative.")
        if self.high_score_threshold < 0.0:
            issues.append("high_score_threshold must be non-negative.")
        if self.trigger_threshold > 1.0:
            issues.append("trigger_threshold must not exceed 1.0.")
        return tuple(issues)

    def estimate(
        self,
        prediction: ResidualPrediction,
        *,
        trace_record: TraceRecord | None = None,
        replay_item: ReplayItem | None = None,
        live_features: LiveFeatureVector | None = None,
    ) -> UncertaintyEstimate:
        energy_uncertainty = min(
            1.0,
            self.base_uncertainty + (1.0 - prediction.confidence) * self.low_confidence_scale,
        )
        force_uncertainty = min(
            1.0,
            energy_uncertainty + len(prediction.force_deltas) * self.force_delta_scale,
        )
        total_uncertainty = max(energy_uncertainty, force_uncertainty)

        priority_hits: set[str] = set()
        if trace_record is not None:
            priority_hits |= _PRIORITY_TAGS & set(trace_record.tags)
        if replay_item is not None:
            priority_hits |= _PRIORITY_TAGS & set(replay_item.tags)
            if replay_item.score >= self.high_score_threshold:
                total_uncertainty += self.high_score_bonus

        if priority_hits:
            total_uncertainty += self.priority_tag_bonus

        chemistry_bonus = 0.0
        structure_bonus = 0.0
        shadow_bonus = 0.0
        if live_features is not None:
            chemistry_mean_pair_score = live_features.value("chemistry_mean_pair_score", 0.5)
            chemistry_favorable_pair_fraction = live_features.value("chemistry_favorable_pair_fraction", 0.5)
            chemistry_flexibility_pressure = live_features.value("chemistry_flexibility_pressure", 0.0)
            structure_rmsd_normalized = live_features.value("structure_rmsd_normalized", 0.0)
            shadow_force_regression = live_features.value("shadow_force_regression", 0.0)
            shadow_energy_regression = live_features.value("shadow_energy_regression", 0.0)

            chemistry_bonus = max(0.0, 0.60 - chemistry_mean_pair_score) * self.chemistry_mismatch_bonus
            chemistry_bonus += max(0.0, 0.50 - chemistry_favorable_pair_fraction) * (self.chemistry_mismatch_bonus * 0.8)
            structure_bonus = structure_rmsd_normalized * self.structure_drift_scale
            shadow_bonus = max(shadow_force_regression, shadow_energy_regression) * self.shadow_regression_bonus

            energy_uncertainty = min(1.0, energy_uncertainty + chemistry_bonus * 0.5 + structure_bonus)
            force_uncertainty = min(
                1.0,
                force_uncertainty
                + chemistry_bonus
                + chemistry_flexibility_pressure * self.chemistry_flexibility_scale
                + shadow_bonus,
            )
            total_uncertainty = max(total_uncertainty, energy_uncertainty, force_uncertainty)

        total_uncertainty = min(1.0, total_uncertainty)
        return UncertaintyEstimate(
            state_id=prediction.state_id,
            energy_uncertainty=min(1.0, energy_uncertainty),
            force_uncertainty=min(1.0, force_uncertainty),
            total_uncertainty=total_uncertainty,
            trigger_qcloud=total_uncertainty >= self.trigger_threshold,
            metadata=FrozenMetadata(
                {
                    "prediction_confidence": prediction.confidence,
                    "priority_hits": tuple(sorted(priority_hits)),
                    "chemistry_bonus": chemistry_bonus,
                    "structure_bonus": structure_bonus,
                    "shadow_bonus": shadow_bonus,
                }
            ),
        )
