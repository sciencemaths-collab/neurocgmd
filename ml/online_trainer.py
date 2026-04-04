"""Replay-driven online training hooks for learned residual models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state_registry import SimulationStateRegistry
from core.types import FrozenMetadata, StateId, coerce_scalar
from forcefields.base_forcefield import BaseForceField
from memory.replay_buffer import ReplayBuffer, ReplayItem
from memory.trace_store import TraceRecord, TraceStore
from ml.residual_model import ResidualModel, ResidualTarget, StateAwareResidualModel
from topology.system_topology import SystemTopology


class ReplaySamplingMode(StrEnum):
    """Deterministic replay selection modes for the Section 11 trainer."""

    HIGHEST_SCORE = "highest_score"
    LATEST = "latest"


@dataclass(frozen=True, slots=True)
class ReplayTrainingExample(ValidatableComponent):
    """One replay-backed training example for a residual model update."""

    replay_item: ReplayItem
    trace_record: TraceRecord
    residual_target: ResidualTarget
    sample_weight: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_weight", coerce_scalar(self.sample_weight, "sample_weight"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.sample_weight <= 0.0:
            issues.append("sample_weight must be strictly positive.")
        if self.replay_item.state_id != self.trace_record.state_id:
            issues.append("Replay item and trace record must reference the same state_id.")
        if self.replay_item.state_id != self.residual_target.state_id:
            issues.append("Replay item and residual target must reference the same state_id.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class OnlineTrainingReport:
    """Summary of one replay-driven online update pass."""

    processed_items: int
    consumed_examples: int
    skipped_items: int
    updated_state_ids: tuple[StateId, ...]
    mean_sample_weight: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)


@dataclass(slots=True)
class ReplayDrivenOnlineTrainer(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Build replay-backed examples and update a residual model deterministically."""

    max_examples_per_update: int = 8
    sampling_mode: ReplaySamplingMode = ReplaySamplingMode.HIGHEST_SCORE
    minimum_replay_score: float = 0.0
    name: str = "replay_driven_online_trainer"
    classification: str = "[hybrid]"

    def describe_role(self) -> str:
        return (
            "Consumes replay-buffer items and aligned residual targets to update "
            "learned residual models without taking ownership of memory storage."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "memory/replay_buffer.py",
            "memory/trace_store.py",
            "ml/residual_model.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/online_ml_residual_learning.md",
            "docs/sections/section_11_online_ml_residual_learning.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.max_examples_per_update <= 0:
            issues.append("max_examples_per_update must be strictly positive.")
        if self.minimum_replay_score < 0.0:
            issues.append("minimum_replay_score must be non-negative.")
        return tuple(issues)

    def _select_items(
        self,
        replay_buffer: ReplayBuffer,
        *,
        limit: int | None = None,
    ) -> tuple[ReplayItem, ...]:
        selection_limit = self.max_examples_per_update if limit is None else min(limit, self.max_examples_per_update)
        if selection_limit < 0:
            raise ContractValidationError("limit must be non-negative when provided.")
        if self.sampling_mode == ReplaySamplingMode.HIGHEST_SCORE:
            return replay_buffer.highest_score(selection_limit)
        return replay_buffer.latest(selection_limit)

    def build_examples(
        self,
        replay_buffer: ReplayBuffer,
        trace_store: TraceStore,
        residual_targets: Mapping[StateId, ResidualTarget],
        *,
        limit: int | None = None,
    ) -> tuple[ReplayTrainingExample, ...]:
        examples: list[ReplayTrainingExample] = []
        selected_items = self._select_items(replay_buffer, limit=limit)
        available_state_ids = set(trace_store.state_ids())
        for replay_item in selected_items:
            if replay_item.score < self.minimum_replay_score:
                continue
            if replay_item.state_id not in available_state_ids:
                continue
            residual_target = residual_targets.get(replay_item.state_id)
            if residual_target is None:
                continue
            trace_record = trace_store.get_record(replay_item.state_id)
            examples.append(
                ReplayTrainingExample(
                    replay_item=replay_item,
                    trace_record=trace_record,
                    residual_target=residual_target,
                    sample_weight=max(replay_item.score, 1e-6),
                    metadata=FrozenMetadata(
                        {
                            "sampling_mode": self.sampling_mode.value,
                            "source_record_id": replay_item.source_record_id,
                        }
                    ),
                )
            )
        return tuple(examples)

    def update_from_replay(
        self,
        model: ResidualModel,
        replay_buffer: ReplayBuffer,
        trace_store: TraceStore,
        residual_targets: Mapping[StateId, ResidualTarget],
        *,
        limit: int | None = None,
    ) -> OnlineTrainingReport:
        selected_items = self._select_items(replay_buffer, limit=limit)
        examples = self.build_examples(
            replay_buffer,
            trace_store,
            residual_targets,
            limit=limit,
        )
        for example in examples:
            model.observe(example.residual_target, sample_weight=example.sample_weight)
        mean_sample_weight = (
            sum(example.sample_weight for example in examples) / len(examples)
            if examples
            else 0.0
        )
        return OnlineTrainingReport(
            processed_items=len(selected_items),
            consumed_examples=len(examples),
            skipped_items=len(selected_items) - len(examples),
            updated_state_ids=tuple(example.residual_target.state_id for example in examples),
            mean_sample_weight=mean_sample_weight,
            metadata=FrozenMetadata(
                {
                    "sampling_mode": self.sampling_mode.value,
                    "trained_state_count": model.trained_state_count(),
                }
            ),
        )

    def update_from_replay_with_states(
        self,
        model: ResidualModel,
        replay_buffer: ReplayBuffer,
        trace_store: TraceStore,
        residual_targets: Mapping[StateId, ResidualTarget],
        *,
        state_registry: SimulationStateRegistry,
        topology: SystemTopology,
        forcefield: BaseForceField,
        force_evaluator,
        limit: int | None = None,
    ) -> OnlineTrainingReport:
        """Update a residual model using explicit state context when supported.

        Models that implement `StateAwareResidualModel` receive the full state and
        baseline force evaluation so local/neighborhood-aware training can happen
        without weakening the base residual protocol. Other models fall back to
        the original replay-driven target-only update path.
        """
        if not isinstance(model, StateAwareResidualModel):
            return self.update_from_replay(
                model,
                replay_buffer,
                trace_store,
                residual_targets,
                limit=limit,
            )

        selected_items = self._select_items(replay_buffer, limit=limit)
        examples = self.build_examples(
            replay_buffer,
            trace_store,
            residual_targets,
            limit=limit,
        )
        for example in examples:
            state = state_registry.get_state(example.residual_target.state_id)
            base_evaluation = force_evaluator.evaluate(state, topology, forcefield)
            model.observe_state(
                state,
                base_evaluation,
                example.residual_target,
                sample_weight=example.sample_weight,
            )
        mean_sample_weight = (
            sum(example.sample_weight for example in examples) / len(examples)
            if examples
            else 0.0
        )
        return OnlineTrainingReport(
            processed_items=len(selected_items),
            consumed_examples=len(examples),
            skipped_items=len(selected_items) - len(examples),
            updated_state_ids=tuple(example.residual_target.state_id for example in examples),
            mean_sample_weight=mean_sample_weight,
            metadata=FrozenMetadata(
                {
                    "sampling_mode": self.sampling_mode.value,
                    "trained_state_count": model.trained_state_count(),
                    "state_aware": True,
                }
            ),
        )
