"""Deterministic decision policies for the executive control layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ai_control.chemistry_governor import ChemistryControlGuidance
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, StateId
from memory.episode_registry import EpisodeKind, EpisodeRegistry, EpisodeStatus

from ai_control.resource_allocator import ResourceAllocation
from ai_control.stability_monitor import StabilityAssessment, StabilityLevel


class ControllerActionKind(StrEnum):
    """Minimal executive-action vocabulary for the foundation controller."""

    HOLD_STEADY = "hold_steady"
    REQUEST_QCLOUD_REFINEMENT = "request_qcloud_refinement"
    ESCALATE_MONITORING = "escalate_monitoring"
    OPEN_INSTABILITY_EPISODE = "open_instability_episode"
    REQUEST_ML_UPDATE = "request_ml_update"
    REVIEW_COMPARTMENT_FOCUS = "review_compartment_focus"
    REVIEW_CHEMISTRY_ALIGNMENT = "review_chemistry_alignment"


@dataclass(frozen=True, slots=True)
class ControllerAction(ValidatableComponent):
    """One explicit recommended action with a deterministic priority."""

    kind: ControllerActionKind
    state_id: StateId
    priority: int
    summary: str
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", StateId(str(self.state_id)))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not str(self.state_id).strip():
            issues.append("state_id must be a non-empty string.")
        if self.priority < 0:
            issues.append("priority must be non-negative.")
        if not self.summary.strip():
            issues.append("summary must be a non-empty string.")
        return tuple(issues)


@dataclass(slots=True)
class DeterministicExecutivePolicy(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Translate assessments and allocations into a deterministic action list."""

    name: str = "deterministic_executive_policy"
    classification: str = "[proposed novel]"

    def describe_role(self) -> str:
        return (
            "Converts explicit stability and resource signals into a transparent "
            "priority-ordered action list for the executive controller."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "ai_control/stability_monitor.py",
            "ai_control/resource_allocator.py",
            "memory/episode_registry.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/ai_executive_control.md",
            "docs/sections/section_12_ai_executive_control_layer.md",
        )

    def validate(self) -> tuple[str, ...]:
        return ()

    def _open_instability_exists(
        self,
        state_id: StateId,
        episode_registry: EpisodeRegistry | None,
    ) -> bool:
        if episode_registry is None:
            return False
        return any(
            episode.kind == EpisodeKind.INSTABILITY and episode.status == EpisodeStatus.OPEN
            for episode in episode_registry.episodes_for_state(state_id)
        )

    def build_actions(
        self,
        state_id: StateId,
        assessment: StabilityAssessment,
        allocation: ResourceAllocation,
        *,
        episode_registry: EpisodeRegistry | None = None,
        chemistry_guidance: ChemistryControlGuidance | None = None,
    ) -> tuple[ControllerAction, ...]:
        actions: list[ControllerAction] = []

        if assessment.trigger_qcloud and allocation.qcloud_region_budget > 0:
            actions.append(
                ControllerAction(
                    kind=ControllerActionKind.REQUEST_QCLOUD_REFINEMENT,
                    state_id=state_id,
                    priority=100 if assessment.level in {StabilityLevel.WARNING, StabilityLevel.CRITICAL} else 70,
                    summary="Request additional qcloud refinement for the current high-risk state.",
                    metadata={"qcloud_region_budget": allocation.qcloud_region_budget},
                )
            )

        if assessment.level in {StabilityLevel.WARNING, StabilityLevel.CRITICAL}:
            actions.append(
                ControllerAction(
                    kind=ControllerActionKind.ESCALATE_MONITORING,
                    state_id=state_id,
                    priority=90,
                    summary="Escalate monitoring intensity for the current state.",
                    metadata={"monitoring_intensity": allocation.monitoring_intensity.value},
                )
            )

        if (
            assessment.level in {StabilityLevel.WARNING, StabilityLevel.CRITICAL}
            and not self._open_instability_exists(state_id, episode_registry)
        ):
            actions.append(
                ControllerAction(
                    kind=ControllerActionKind.OPEN_INSTABILITY_EPISODE,
                    state_id=state_id,
                    priority=80,
                    summary="Open an instability episode to track the current escalation window.",
                )
            )

        if allocation.ml_example_budget > 0:
            actions.append(
                ControllerAction(
                    kind=ControllerActionKind.REQUEST_ML_UPDATE,
                    state_id=state_id,
                    priority=50 if assessment.level == StabilityLevel.STABLE else 60,
                    summary="Run a replay-driven ML update using the current residual memory budget.",
                    metadata={"ml_example_budget": allocation.ml_example_budget},
                )
            )

        if allocation.focus_compartments:
            actions.append(
                ControllerAction(
                    kind=ControllerActionKind.REVIEW_COMPARTMENT_FOCUS,
                    state_id=state_id,
                    priority=40,
                    summary="Review compartment-focused routing and refinement priorities.",
                    metadata={"focus_compartments": allocation.focus_compartments},
                )
            )

        if chemistry_guidance is not None and chemistry_guidance.review_required:
            actions.append(
                ControllerAction(
                    kind=ControllerActionKind.REVIEW_CHEMISTRY_ALIGNMENT,
                    state_id=state_id,
                    priority=75 if chemistry_guidance.chemistry_risk >= 0.55 else 45,
                    summary="Review interface-chemistry alignment before trusting the current assembly geometry.",
                    metadata={
                        "chemistry_risk": chemistry_guidance.chemistry_risk,
                        "focus_compartments": chemistry_guidance.focus_compartments,
                    },
                )
            )

        if not actions:
            actions.append(
                ControllerAction(
                    kind=ControllerActionKind.HOLD_STEADY,
                    state_id=state_id,
                    priority=10,
                    summary="Hold steady; no escalation or additional subsystem work is currently recommended.",
                )
            )

        return tuple(sorted(actions, key=lambda action: (-action.priority, action.kind.value)))
