"""Deterministic resource allocation for executive-control recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ai_control.chemistry_governor import ChemistryControlGuidance
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, StateId
from memory.replay_buffer import ReplayBuffer
from qcloud.qcloud_coupling import QCloudCouplingResult

from ai_control.stability_monitor import StabilityAssessment, StabilityLevel


class MonitoringIntensity(StrEnum):
    """Recommended monitoring intensity level for the current state."""

    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class ExecutionBudget(ValidatableComponent):
    """Simple bounded execution budget handed to the allocator."""

    max_qcloud_regions: int = 2
    max_ml_examples: int = 4
    max_focus_compartments: int = 3

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.max_qcloud_regions < 0:
            issues.append("max_qcloud_regions must be non-negative.")
        if self.max_ml_examples < 0:
            issues.append("max_ml_examples must be non-negative.")
        if self.max_focus_compartments < 0:
            issues.append("max_focus_compartments must be non-negative.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ResourceAllocation(ValidatableComponent):
    """Explicit resource recommendation derived from a stability assessment."""

    state_id: StateId
    level: StabilityLevel
    qcloud_region_budget: int
    ml_example_budget: int
    monitoring_intensity: MonitoringIntensity
    focus_compartments: tuple[str, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", StateId(str(self.state_id)))
        object.__setattr__(
            self,
            "focus_compartments",
            tuple(
                dict.fromkeys(
                    identifier
                    for identifier in (str(value).strip() for value in self.focus_compartments)
                    if identifier
                )
            ),
        )
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not str(self.state_id).strip():
            issues.append("state_id must be a non-empty string.")
        if self.qcloud_region_budget < 0:
            issues.append("qcloud_region_budget must be non-negative.")
        if self.ml_example_budget < 0:
            issues.append("ml_example_budget must be non-negative.")
        return tuple(issues)


@dataclass(slots=True)
class ResourceAllocator(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Translate stability assessments into bounded subsystem resource recommendations."""

    budget: ExecutionBudget = field(default_factory=ExecutionBudget)
    name: str = "resource_allocator"
    classification: str = "[hybrid]"

    def describe_role(self) -> str:
        return (
            "Converts explicit stability assessments into bounded qcloud, ML, and "
            "monitoring budgets without mutating those downstream subsystems."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "ai_control/stability_monitor.py",
            "memory/replay_buffer.py",
            "qcloud/qcloud_coupling.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/ai_executive_control.md",
            "docs/sections/section_12_ai_executive_control_layer.md",
        )

    def validate(self) -> tuple[str, ...]:
        return self.budget.validate()

    def allocate(
        self,
        assessment: StabilityAssessment,
        *,
        replay_buffer: ReplayBuffer | None = None,
        qcloud_result: QCloudCouplingResult | None = None,
        chemistry_guidance: ChemistryControlGuidance | None = None,
    ) -> ResourceAllocation:
        replay_available = len(replay_buffer) if replay_buffer is not None else 0
        qcloud_already_used = len(qcloud_result.selected_regions) if qcloud_result is not None else 0

        if assessment.level == StabilityLevel.CRITICAL:
            base_qcloud_budget = self.budget.max_qcloud_regions
            base_ml_budget = self.budget.max_ml_examples
            monitoring_intensity = MonitoringIntensity.CRITICAL
        elif assessment.level == StabilityLevel.WARNING:
            base_qcloud_budget = min(2, self.budget.max_qcloud_regions)
            base_ml_budget = min(3, self.budget.max_ml_examples)
            monitoring_intensity = MonitoringIntensity.HIGH
        elif assessment.level == StabilityLevel.WATCH:
            base_qcloud_budget = 1 if assessment.trigger_qcloud else 0
            base_ml_budget = min(2, self.budget.max_ml_examples)
            monitoring_intensity = MonitoringIntensity.ELEVATED
        else:
            base_qcloud_budget = 0
            base_ml_budget = 1 if replay_available else 0
            monitoring_intensity = MonitoringIntensity.LOW

        if chemistry_guidance is not None:
            if chemistry_guidance.recommend_qcloud_boost:
                base_qcloud_budget = min(self.budget.max_qcloud_regions, base_qcloud_budget + 1)
            if chemistry_guidance.recommend_ml_boost:
                base_ml_budget = min(self.budget.max_ml_examples, base_ml_budget + 1)

        qcloud_region_budget = max(0, base_qcloud_budget - qcloud_already_used)
        ml_example_budget = min(base_ml_budget, replay_available) if replay_available else 0
        focus_candidates = list(assessment.recommended_focus_compartments)
        if chemistry_guidance is not None:
            for compartment_id in chemistry_guidance.focus_compartments:
                if compartment_id not in focus_candidates:
                    focus_candidates.append(compartment_id)
        focus_compartments = tuple(focus_candidates[: self.budget.max_focus_compartments])

        return ResourceAllocation(
            state_id=assessment.state_id,
            level=assessment.level,
            qcloud_region_budget=qcloud_region_budget,
            ml_example_budget=ml_example_budget,
            monitoring_intensity=monitoring_intensity,
            focus_compartments=focus_compartments,
            metadata=FrozenMetadata(
                {
                    "normalized_risk": assessment.normalized_risk,
                    "replay_available": replay_available,
                    "qcloud_already_used": qcloud_already_used,
                    "chemistry_risk": chemistry_guidance.chemistry_risk if chemistry_guidance is not None else None,
                }
            ),
        )
