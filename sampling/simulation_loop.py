"""Primary simulation-loop orchestration for the early MD foundation."""

from __future__ import annotations

from dataclasses import dataclass

from core.exceptions import ContractValidationError
from core.state import SimulationState
from core.state_registry import LifecycleStage, SimulationStateRegistry
from core.types import FrozenMetadata, StateId
from forcefields.base_forcefield import BaseForceField
from integrators.base import ForceEvaluator, StateIntegrator
from topology.system_topology import SystemTopology


@dataclass(frozen=True, slots=True)
class SimulationRunResult:
    """Structured result for a completed simulation-loop run."""

    initial_state: SimulationState
    final_state: SimulationState
    produced_state_ids: tuple[StateId, ...]
    steps_completed: int


@dataclass(slots=True)
class SimulationLoop:
    """Run an integrator repeatedly while preserving registry lineage."""

    topology: SystemTopology
    forcefield: BaseForceField
    integrator: StateIntegrator
    force_evaluator: ForceEvaluator
    registry: SimulationStateRegistry
    stage: LifecycleStage | str = LifecycleStage.INTEGRATION

    def run(
        self,
        steps: int,
        *,
        notes: str = "",
        metadata: FrozenMetadata | dict[str, object] | None = None,
    ) -> SimulationRunResult:
        if steps < 0:
            raise ContractValidationError("steps must be non-negative.")
        if len(self.registry) == 0:
            raise ContractValidationError(
                "SimulationLoop requires a registry with an initial state already registered."
            )

        initial_state = self.registry.latest_state()
        current_state = initial_state
        produced_state_ids: list[StateId] = []
        user_metadata = metadata if isinstance(metadata, FrozenMetadata) else FrozenMetadata(metadata)

        for _ in range(steps):
            step_result = self.integrator.step(
                current_state,
                self.topology,
                self.forcefield,
                self.force_evaluator,
            )
            merged_metadata = user_metadata.to_dict()
            merged_metadata.update(step_result.metadata.to_dict())
            merged_metadata.update(
                {
                    "integrator": self.integrator.name,
                    "force_evaluator": self.force_evaluator.name,
                }
            )
            current_state = self.registry.derive_state(
                current_state,
                particles=step_result.particles,
                time=step_result.time,
                step=step_result.step,
                potential_energy=step_result.potential_energy,
                observables=step_result.observables,
                stage=self.stage,
                notes=notes,
                metadata=FrozenMetadata(merged_metadata),
            )
            produced_state_ids.append(current_state.provenance.state_id)

        return SimulationRunResult(
            initial_state=initial_state,
            final_state=current_state,
            produced_state_ids=tuple(produced_state_ids),
            steps_completed=steps,
        )

