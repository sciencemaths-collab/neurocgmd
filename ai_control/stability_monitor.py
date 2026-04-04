"""Stability scoring and escalation assessment for executive control."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from chemistry.interface_logic import ChemistryInterfaceReport
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata, StateId, coerce_scalar
from graph.graph_manager import ConnectivityGraph
from memory.episode_registry import EpisodeKind, EpisodeRegistry, EpisodeStatus
from memory.trace_store import TraceRecord
from ml.uncertainty_model import UncertaintyEstimate
from qcloud.qcloud_coupling import QCloudCouplingResult

_PRIORITY_TAGS = frozenset({"priority", "instability", "qcloud", "refine"})


def _normalize_compartment_ids(compartment_ids: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_identifier in compartment_ids:
        identifier = str(raw_identifier).strip()
        if not identifier:
            continue
        if identifier not in seen:
            normalized.append(identifier)
            seen.add(identifier)
    return tuple(normalized)


class StabilityLevel(StrEnum):
    """Discrete stability levels for the executive controller."""

    STABLE = "stable"
    WATCH = "watch"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class StabilitySignal(ValidatableComponent):
    """One explicit signal that contributed to the overall stability assessment."""

    source: str
    level: StabilityLevel
    score: float
    message: str
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", coerce_scalar(self.score, "score"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.source.strip():
            issues.append("source must be a non-empty string.")
        if not (0.0 <= self.score <= 1.0):
            issues.append("score must lie in the interval [0, 1].")
        if not self.message.strip():
            issues.append("message must be a non-empty string.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class StabilityAssessment(ValidatableComponent):
    """Composite stability summary consumed by the resource allocator and controller."""

    state_id: StateId
    step: int
    level: StabilityLevel
    normalized_risk: float
    trigger_qcloud: bool
    recommended_focus_compartments: tuple[str, ...] = ()
    signals: tuple[StabilitySignal, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", StateId(str(self.state_id)))
        object.__setattr__(self, "normalized_risk", coerce_scalar(self.normalized_risk, "normalized_risk"))
        object.__setattr__(
            self,
            "recommended_focus_compartments",
            _normalize_compartment_ids(self.recommended_focus_compartments),
        )
        object.__setattr__(self, "signals", tuple(self.signals))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not str(self.state_id).strip():
            issues.append("state_id must be a non-empty string.")
        if self.step < 0:
            issues.append("step must be non-negative.")
        if not (0.0 <= self.normalized_risk <= 1.0):
            issues.append("normalized_risk must lie in the interval [0, 1].")
        return tuple(issues)


@dataclass(slots=True)
class StabilityMonitor(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Heuristic stability monitor that converts subsystem signals into a bounded risk level."""

    uncertainty_weight: float = 0.7
    adaptive_edge_ratio_weight: float = 0.35
    memory_priority_bonus: float = 0.15
    open_instability_bonus: float = 0.2
    qcloud_relief: float = 0.1
    chemistry_weight: float = 0.4
    watch_threshold: float = 0.35
    warning_threshold: float = 0.6
    critical_threshold: float = 0.8
    name: str = "stability_monitor"
    classification: str = "[hybrid]"

    def describe_role(self) -> str:
        return (
            "Aggregates graph, memory, qcloud, and ML signals into an explicit "
            "bounded stability assessment for executive control."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "graph/graph_manager.py",
            "memory/trace_store.py",
            "memory/episode_registry.py",
            "ml/uncertainty_model.py",
            "qcloud/qcloud_coupling.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/ai_executive_control.md",
            "docs/sections/section_12_ai_executive_control_layer.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for field_name in (
            "uncertainty_weight",
            "adaptive_edge_ratio_weight",
            "memory_priority_bonus",
            "open_instability_bonus",
            "qcloud_relief",
            "chemistry_weight",
            "watch_threshold",
            "warning_threshold",
            "critical_threshold",
        ):
            value = getattr(self, field_name)
            if value < 0.0:
                issues.append(f"{field_name} must be non-negative.")
        if not (self.watch_threshold <= self.warning_threshold <= self.critical_threshold <= 1.0):
            issues.append(
                "watch_threshold, warning_threshold, and critical_threshold must satisfy "
                "watch <= warning <= critical <= 1.0."
            )
        return tuple(issues)

    def _level_for_score(self, score: float) -> StabilityLevel:
        if score >= self.critical_threshold:
            return StabilityLevel.CRITICAL
        if score >= self.warning_threshold:
            return StabilityLevel.WARNING
        if score >= self.watch_threshold:
            return StabilityLevel.WATCH
        return StabilityLevel.STABLE

    def assess(
        self,
        state: SimulationState,
        graph: ConnectivityGraph,
        *,
        trace_record: TraceRecord | None = None,
        uncertainty_estimate: UncertaintyEstimate | None = None,
        episode_registry: EpisodeRegistry | None = None,
        qcloud_result: QCloudCouplingResult | None = None,
        chemistry_report: ChemistryInterfaceReport | None = None,
    ) -> StabilityAssessment:
        if graph.particle_count != state.particle_count:
            raise ContractValidationError("ConnectivityGraph particle_count must match the SimulationState particle count.")
        if graph.step != state.step:
            raise ContractValidationError("ConnectivityGraph step must match the SimulationState step.")
        if trace_record is not None and trace_record.state_id != state.provenance.state_id:
            raise ContractValidationError("trace_record.state_id must match the current SimulationState state_id.")
        if uncertainty_estimate is not None and uncertainty_estimate.state_id != state.provenance.state_id:
            raise ContractValidationError(
                "uncertainty_estimate.state_id must match the current SimulationState state_id."
            )

        active_edge_count = len(graph.active_edges())
        adaptive_edge_ratio = (
            len(graph.adaptive_edges()) / active_edge_count if active_edge_count else 0.0
        )

        risk = 0.0
        signals: list[StabilitySignal] = []
        priority_compartments: list[str] = []

        if uncertainty_estimate is not None:
            uncertainty_score = min(1.0, uncertainty_estimate.total_uncertainty * self.uncertainty_weight)
            risk += uncertainty_score
            signals.append(
                StabilitySignal(
                    source="ml_uncertainty",
                    level=self._level_for_score(uncertainty_estimate.total_uncertainty),
                    score=uncertainty_estimate.total_uncertainty,
                    message="Residual uncertainty contributed to the stability assessment.",
                )
            )

        if adaptive_edge_ratio > 0.0:
            graph_score = min(1.0, adaptive_edge_ratio * self.adaptive_edge_ratio_weight)
            risk += graph_score
            signals.append(
                StabilitySignal(
                    source="graph_adaptivity",
                    level=self._level_for_score(min(1.0, adaptive_edge_ratio)),
                    score=min(1.0, adaptive_edge_ratio),
                    message="Adaptive graph density contributed to the stability assessment.",
                    metadata={"active_edge_count": active_edge_count},
                )
            )

        if chemistry_report is not None:
            chemistry_risk = min(
                1.0,
                (
                    (1.0 - chemistry_report.mean_pair_score) * 0.5
                    + (1.0 - chemistry_report.favorable_pair_fraction) * 0.25
                    + chemistry_report.flexibility_pressure * 0.25
                )
                * self.chemistry_weight,
            )
            risk += chemistry_risk
            priority_compartments.extend(chemistry_report.compartment_ids)
            signals.append(
                StabilitySignal(
                    source="chemistry_interface",
                    level=self._level_for_score(min(1.0, chemistry_risk / max(self.chemistry_weight, 1.0e-12))),
                    score=min(1.0, chemistry_risk / max(self.chemistry_weight, 1.0e-12)),
                    message="Chemistry-interface plausibility contributed to the stability assessment.",
                    metadata={
                        "mean_pair_score": chemistry_report.mean_pair_score,
                        "favorable_pair_fraction": chemistry_report.favorable_pair_fraction,
                        "flexibility_pressure": chemistry_report.flexibility_pressure,
                    },
                )
            )

        if trace_record is not None and (_PRIORITY_TAGS & set(trace_record.tags)):
            risk += self.memory_priority_bonus
            priority_compartments.extend(trace_record.compartment_ids)
            signals.append(
                StabilitySignal(
                    source="memory_priority",
                    level=StabilityLevel.WARNING,
                    score=min(1.0, self.memory_priority_bonus),
                    message="Priority replay or instability tags raised the stability score.",
                    metadata={"tags": trace_record.tags},
                )
            )

        open_instability_exists = False
        if episode_registry is not None:
            for episode in episode_registry.episodes_for_state(state.provenance.state_id):
                if episode.kind == EpisodeKind.INSTABILITY and episode.status == EpisodeStatus.OPEN:
                    open_instability_exists = True
                    break
        if open_instability_exists:
            risk += self.open_instability_bonus
            signals.append(
                StabilitySignal(
                    source="episode_registry",
                    level=StabilityLevel.CRITICAL,
                    score=min(1.0, self.open_instability_bonus),
                    message="An open instability episode is active for this state.",
                )
            )

        if qcloud_result is not None and qcloud_result.selected_regions:
            risk = max(0.0, risk - self.qcloud_relief)
            priority_compartments.extend(
                compartment_id
                for region in qcloud_result.selected_regions
                for compartment_id in region.compartment_ids
            )
            signals.append(
                StabilitySignal(
                    source="qcloud_feedback",
                    level=StabilityLevel.WATCH,
                    score=min(1.0, self.qcloud_relief),
                    message="Applied qcloud regions reduced the immediate stability risk estimate.",
                    metadata={"selected_region_count": len(qcloud_result.selected_regions)},
                )
            )

        normalized_risk = min(1.0, max(0.0, risk))
        level = self._level_for_score(normalized_risk)
        trigger_qcloud = bool(
            (uncertainty_estimate and uncertainty_estimate.trigger_qcloud)
            or level in {StabilityLevel.WARNING, StabilityLevel.CRITICAL}
        )
        return StabilityAssessment(
            state_id=state.provenance.state_id,
            step=state.step,
            level=level,
            normalized_risk=normalized_risk,
            trigger_qcloud=trigger_qcloud,
            recommended_focus_compartments=tuple(priority_compartments),
            signals=tuple(signals),
            metadata=FrozenMetadata(
                {
                    "adaptive_edge_ratio": adaptive_edge_ratio,
                    "active_edge_count": active_edge_count,
                    "open_instability_exists": open_instability_exists,
                }
            ),
        )
