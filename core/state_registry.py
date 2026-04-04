"""Lifecycle helpers and lineage tracking for immutable simulation states."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4

from core.exceptions import ContractValidationError
from core.state import (
    ParticleState,
    SimulationCell,
    SimulationState,
    StateProvenance,
    ThermodynamicState,
    UnitSystem,
)
from core.types import FrozenMetadata, SimulationId, StateId

_UNCHANGED = object()


class LifecycleStage(StrEnum):
    """Common lifecycle stage labels for state provenance."""

    INITIALIZATION = "initialization"
    INTEGRATION = "integration"
    FORCE_EVALUATION = "force_evaluation"
    CORRECTION = "correction"
    CHECKPOINT = "checkpoint"
    ANALYSIS = "analysis"
    CUSTOM = "custom"


@dataclass(slots=True)
class IdentifierMint:
    """Monotonic identifier generator scoped to one local session."""

    session_token: str = field(default_factory=lambda: uuid4().hex[:8])
    simulation_counter: int = 0
    state_counter: int = 0

    def new_simulation_id(self) -> SimulationId:
        self.simulation_counter += 1
        return SimulationId(f"sim-{self.session_token}-{self.simulation_counter:06d}")

    def new_state_id(self, step: int | None = None) -> StateId:
        self.state_counter += 1
        step_fragment = f"-step{step:08d}" if step is not None else ""
        return StateId(f"state-{self.session_token}-{self.state_counter:08d}{step_fragment}")


@dataclass(frozen=True, slots=True)
class StateSnapshotSummary:
    """Compact summary for checkpoint, diagnostics, and audit trails."""

    state_id: StateId
    parent_state_id: StateId | None
    stage: str
    step: int
    time: float
    particle_count: int
    kinetic_energy: float
    potential_energy: float | None

    def total_energy(self) -> float | None:
        if self.potential_energy is None:
            return None
        return self.kinetic_energy + self.potential_energy


@dataclass(slots=True)
class SimulationStateRegistry:
    """Registry that creates, validates, and indexes simulation states."""

    created_by: str
    identifier_mint: IdentifierMint = field(default_factory=IdentifierMint)
    simulation_id: SimulationId | None = None
    _states: dict[StateId, SimulationState] = field(default_factory=dict, init=False, repr=False)
    _state_order: list[StateId] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.created_by.strip():
            raise ContractValidationError("created_by must be a non-empty string.")
        if self.simulation_id is None:
            self.simulation_id = self.identifier_mint.new_simulation_id()

    def __len__(self) -> int:
        return len(self._state_order)

    def new_provenance(
        self,
        stage: LifecycleStage | str,
        *,
        step: int | None = None,
        parent_state_id: StateId | None = None,
        created_by: str | None = None,
        notes: str = "",
        metadata: FrozenMetadata | dict[str, object] | None = None,
    ) -> StateProvenance:
        """Create a provenance record tied to this registry's simulation id."""

        stage_label = stage.value if isinstance(stage, LifecycleStage) else str(stage).strip()
        if not stage_label:
            raise ContractValidationError("stage must be a non-empty string.")
        return StateProvenance(
            simulation_id=self.require_simulation_id(),
            state_id=self.identifier_mint.new_state_id(step),
            parent_state_id=parent_state_id,
            created_by=created_by or self.created_by,
            stage=stage_label,
            notes=notes,
            metadata=metadata if isinstance(metadata, FrozenMetadata) else FrozenMetadata(metadata),
        )

    def require_simulation_id(self) -> SimulationId:
        if self.simulation_id is None:
            raise ContractValidationError("simulation_id has not been initialized.")
        return self.simulation_id

    def create_initial_state(
        self,
        *,
        particles: ParticleState,
        units: UnitSystem | None = None,
        thermodynamics: ThermodynamicState | None = None,
        cell: SimulationCell | None = None,
        time: float = 0.0,
        step: int = 0,
        potential_energy: float | None = None,
        observables: FrozenMetadata | dict[str, object] | None = None,
        notes: str = "",
        metadata: FrozenMetadata | dict[str, object] | None = None,
        stage: LifecycleStage | str = LifecycleStage.INITIALIZATION,
    ) -> SimulationState:
        """Create and register the root state for a simulation lineage."""

        if self._state_order:
            raise ContractValidationError("Initial state already exists for this registry.")
        provenance = self.new_provenance(
            stage,
            step=step,
            created_by=self.created_by,
            notes=notes,
            metadata=metadata,
        )
        state = SimulationState(
            units=units or UnitSystem.md_nano(),
            particles=particles,
            thermodynamics=thermodynamics or ThermodynamicState(),
            provenance=provenance,
            cell=cell,
            time=time,
            step=step,
            potential_energy=potential_energy,
            observables=observables if observables is not None else FrozenMetadata(),
        )
        self.register_state(state)
        return state

    def derive_state(
        self,
        parent: SimulationState,
        *,
        particles: ParticleState | None = None,
        units: UnitSystem | None = None,
        thermodynamics: ThermodynamicState | None = None,
        cell: SimulationCell | None | object = _UNCHANGED,
        time: float | None = None,
        step: int | None = None,
        potential_energy: float | None | object = _UNCHANGED,
        observables: FrozenMetadata | dict[str, object] | None = None,
        stage: LifecycleStage | str = LifecycleStage.INTEGRATION,
        notes: str = "",
        metadata: FrozenMetadata | dict[str, object] | None = None,
        created_by: str | None = None,
    ) -> SimulationState:
        """Create and register a new state derived from an existing parent."""

        parent_state_id = parent.provenance.state_id
        if parent_state_id not in self._states:
            raise ContractValidationError(
                f"Parent state {parent_state_id} is not registered in this registry."
            )

        next_step = parent.step + 1 if step is None else step
        provenance = self.new_provenance(
            stage,
            step=next_step,
            parent_state_id=parent_state_id,
            created_by=created_by or self.created_by,
            notes=notes,
            metadata=metadata,
        )
        state = SimulationState(
            units=units or parent.units,
            particles=particles or parent.particles,
            thermodynamics=thermodynamics or parent.thermodynamics,
            provenance=provenance,
            cell=parent.cell if cell is _UNCHANGED else cell,
            time=parent.time if time is None else time,
            step=next_step,
            potential_energy=(
                parent.potential_energy if potential_energy is _UNCHANGED else potential_energy
            ),
            observables=parent.observables if observables is None else observables,
        )
        self.register_state(state)
        return state

    def register_state(self, state: SimulationState) -> None:
        """Insert an existing state after validating lineage consistency."""

        state_id = state.provenance.state_id
        if state_id in self._states:
            raise ContractValidationError(f"State {state_id} is already registered.")
        if state.provenance.simulation_id != self.require_simulation_id():
            raise ContractValidationError(
                "State simulation_id does not match the registry simulation_id."
            )

        parent_state_id = state.provenance.parent_state_id
        if parent_state_id is not None:
            if parent_state_id not in self._states:
                raise ContractValidationError(
                    f"Parent state {parent_state_id} must be registered before its child."
                )
            parent = self._states[parent_state_id]
            if state.step < parent.step:
                raise ContractValidationError("Child state step cannot be less than parent step.")
            if state.time < parent.time:
                raise ContractValidationError("Child state time cannot be less than parent time.")

        self._states[state_id] = state
        self._state_order.append(state_id)

    def latest_state(self) -> SimulationState:
        if not self._state_order:
            raise ContractValidationError("No states have been registered yet.")
        return self._states[self._state_order[-1]]

    def get_state(self, state_id: StateId) -> SimulationState:
        return self._states[state_id]

    def state_ids(self) -> tuple[StateId, ...]:
        return tuple(self._state_order)

    def lineage_for(self, state_id: StateId) -> tuple[StateId, ...]:
        """Return the root-to-leaf lineage ending at the requested state."""

        lineage: list[StateId] = []
        current = self.get_state(state_id)
        while True:
            lineage.append(current.provenance.state_id)
            parent_state_id = current.provenance.parent_state_id
            if parent_state_id is None:
                break
            current = self.get_state(parent_state_id)
        lineage.reverse()
        return tuple(lineage)

    def summaries(self) -> tuple[StateSnapshotSummary, ...]:
        """Return compact summaries in registration order."""

        return tuple(
            StateSnapshotSummary(
                state_id=state.provenance.state_id,
                parent_state_id=state.provenance.parent_state_id,
                stage=state.provenance.stage,
                step=state.step,
                time=state.time,
                particle_count=state.particle_count,
                kinetic_energy=state.kinetic_energy(),
                potential_energy=state.potential_energy,
            )
            for state in (self._states[state_id] for state_id in self._state_order)
        )

