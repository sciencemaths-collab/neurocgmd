"""Executive controller that orchestrates monitoring, allocation, and policy outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from chemistry.interface_logic import ChemistryInterfaceReport
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata, StateId
from graph.graph_manager import ConnectivityGraph
from memory.episode_registry import EpisodeRegistry
from memory.replay_buffer import ReplayBuffer
from memory.trace_store import TraceRecord
from ml.uncertainty_model import UncertaintyEstimate
from qcloud.qcloud_coupling import QCloudCouplingResult

from ai_control.chemistry_governor import ChemistryAwareGovernor
from ai_control.policies import ControllerAction, DeterministicExecutivePolicy
from ai_control.resource_allocator import ResourceAllocation, ResourceAllocator
from ai_control.stability_monitor import StabilityAssessment, StabilityMonitor

if TYPE_CHECKING:
    from ml.live_features import LiveFeatureVector


@dataclass(frozen=True, slots=True)
class ControllerDecision(ValidatableComponent):
    """Final controller output for one state and decision cycle."""

    state_id: StateId
    step: int
    assessment: StabilityAssessment
    allocation: ResourceAllocation
    actions: tuple[ControllerAction, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", StateId(str(self.state_id)))
        object.__setattr__(self, "actions", tuple(self.actions))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def highest_priority_action(self) -> ControllerAction:
        return self.actions[0]

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not str(self.state_id).strip():
            issues.append("state_id must be a non-empty string.")
        if self.step < 0:
            issues.append("step must be non-negative.")
        if self.assessment.state_id != self.state_id:
            issues.append("assessment.state_id must match decision state_id.")
        if self.allocation.state_id != self.state_id:
            issues.append("allocation.state_id must match decision state_id.")
        if not self.actions:
            issues.append("actions must contain at least one controller action.")
        if any(action.state_id != self.state_id for action in self.actions):
            issues.append("all controller actions must target the decision state_id.")
        return tuple(issues)


@dataclass(slots=True)
class ExecutiveController(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Compose monitoring and allocation outputs into prioritized controller actions."""

    monitor: StabilityMonitor = field(default_factory=StabilityMonitor)
    chemistry_governor: ChemistryAwareGovernor = field(default_factory=ChemistryAwareGovernor)
    allocator: ResourceAllocator = field(default_factory=ResourceAllocator)
    policy: DeterministicExecutivePolicy = field(default_factory=DeterministicExecutivePolicy)
    name: str = "executive_controller"
    classification: str = "[proposed novel]"

    def describe_role(self) -> str:
        return (
            "Consumes graph, memory, qcloud, and ML signals and produces explicit "
            "prioritized recommendations without taking ownership of those subsystems."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "ai_control/chemistry_governor.py",
            "ai_control/stability_monitor.py",
            "ai_control/resource_allocator.py",
            "ai_control/policies.py",
            "memory/episode_registry.py",
            "qcloud/qcloud_coupling.py",
            "ml/uncertainty_model.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/ai_executive_control.md",
            "docs/sections/section_12_ai_executive_control_layer.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        issues.extend(self.monitor.validate())
        issues.extend(self.chemistry_governor.validate())
        issues.extend(self.allocator.validate())
        issues.extend(self.policy.validate())
        return tuple(issues)

    def decide(
        self,
        state: SimulationState,
        graph: ConnectivityGraph,
        *,
        trace_record: TraceRecord | None = None,
        uncertainty_estimate: UncertaintyEstimate | None = None,
        episode_registry: EpisodeRegistry | None = None,
        replay_buffer: ReplayBuffer | None = None,
        qcloud_result: QCloudCouplingResult | None = None,
        chemistry_report: ChemistryInterfaceReport | None = None,
        live_features: LiveFeatureVector | None = None,
    ) -> ControllerDecision:
        assessment = self.monitor.assess(
            state,
            graph,
            trace_record=trace_record,
            uncertainty_estimate=uncertainty_estimate,
            episode_registry=episode_registry,
            qcloud_result=qcloud_result,
            chemistry_report=chemistry_report,
        )
        chemistry_guidance = self.chemistry_governor.guide(
            state.provenance.state_id,
            chemistry_report,
            live_features=live_features,
        )
        allocation = self.allocator.allocate(
            assessment,
            replay_buffer=replay_buffer,
            qcloud_result=qcloud_result,
            chemistry_guidance=chemistry_guidance,
        )

        state_id = state.provenance.state_id
        ordered_actions = self.policy.build_actions(
            state_id,
            assessment,
            allocation,
            episode_registry=episode_registry,
            chemistry_guidance=chemistry_guidance,
        )
        return ControllerDecision(
            state_id=state_id,
            step=state.step,
            assessment=assessment,
            allocation=allocation,
            actions=ordered_actions,
            metadata=FrozenMetadata(
                {
                    "action_count": len(ordered_actions),
                    "highest_priority_action": ordered_actions[0].kind.value,
                    "policy_name": self.policy.name,
                    "chemistry_risk": chemistry_guidance.chemistry_risk if chemistry_guidance is not None else None,
                }
            ),
        )
