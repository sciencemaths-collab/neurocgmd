"""Foundational architectural contracts and state models for NeuroCGMD."""

from core.project_manifest import ProjectManifest, SectionDefinition, build_default_manifest
from core.state import (
    EnsembleKind,
    ParticleState,
    SimulationCell,
    SimulationState,
    StateProvenance,
    ThermodynamicState,
    UnitSystem,
)
from core.state_registry import (
    IdentifierMint,
    LifecycleStage,
    SimulationStateRegistry,
    StateSnapshotSummary,
)
from core.types import (
    BeadId,
    CompartmentId,
    FrozenMetadata,
    ModelId,
    RegionId,
    SimulationId,
    StateId,
)
from core.units import (
    BOLTZMANN_CONSTANT,
    COULOMB_CONSTANT,
    ONE_4PI_EPS0,
    UnitConverter,
    UnitValidator,
)

__all__ = [
    "BeadId",
    "CompartmentId",
    "EnsembleKind",
    "FrozenMetadata",
    "IdentifierMint",
    "LifecycleStage",
    "ModelId",
    "ParticleState",
    "ProjectManifest",
    "RegionId",
    "SectionDefinition",
    "SimulationCell",
    "SimulationId",
    "SimulationState",
    "SimulationStateRegistry",
    "StateId",
    "StateProvenance",
    "StateSnapshotSummary",
    "ThermodynamicState",
    "UnitSystem",
    "BOLTZMANN_CONSTANT",
    "COULOMB_CONSTANT",
    "ONE_4PI_EPS0",
    "UnitConverter",
    "UnitValidator",
    "build_default_manifest",
]

