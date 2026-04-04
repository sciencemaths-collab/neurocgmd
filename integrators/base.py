"""Integrator contracts and step-result containers for the simulation loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from core.state import ParticleState, SimulationState
from core.types import FrozenMetadata
from forcefields.base_forcefield import BaseForceField
from physics.forces.composite import ForceEvaluation
from topology.system_topology import SystemTopology


@runtime_checkable
class ForceEvaluator(Protocol):
    """Protocol for anything that can evaluate forces and potential energy."""

    name: str
    classification: str

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
    ) -> ForceEvaluation:
        """Return the force block and potential energy for the supplied state."""


@dataclass(frozen=True, slots=True)
class IntegratorStepResult:
    """Single-step integrator output before registry insertion."""

    particles: ParticleState
    time: float
    step: int
    potential_energy: float | None
    observables: FrozenMetadata = field(default_factory=FrozenMetadata)
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)


@runtime_checkable
class StateIntegrator(Protocol):
    """Protocol implemented by time integrators that advance immutable state."""

    name: str
    classification: str
    time_step: float

    def step(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
        force_evaluator: ForceEvaluator,
    ) -> IntegratorStepResult:
        """Advance the supplied state by one integrator step."""

