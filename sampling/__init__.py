"""Sampling loops, ensembles, and exploration policies."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sampling.enhanced_sampling import (
        CollectiveVariable,
        DistanceCV,
        GaussianHill,
        MetadynamicsEngine,
        RadiusOfGyrationCV,
        ReplicaExchangeManager,
        ReplicaState,
    )
    from sampling.production_engine import HybridProductionEngine, ProductionCycleReport
    from sampling.production_loop import (
        EnergyTracker,
        ProductionRunResult,
        ProductionSimulationLoop,
        TemperatureSchedule,
    )
    from sampling.simulation_loop import SimulationLoop, SimulationRunResult

__all__ = [
    "CollectiveVariable",
    "DistanceCV",
    "EnergyTracker",
    "GaussianHill",
    "MetadynamicsEngine",
    "HybridProductionEngine",
    "ProductionRunResult",
    "ProductionSimulationLoop",
    "RadiusOfGyrationCV",
    "ReplicaExchangeManager",
    "ReplicaState",
    "ProductionCycleReport",
    "SimulationLoop",
    "SimulationRunResult",
    "TemperatureSchedule",
]

_LAZY_EXPORTS = {
    "CollectiveVariable": ("sampling.enhanced_sampling", "CollectiveVariable"),
    "DistanceCV": ("sampling.enhanced_sampling", "DistanceCV"),
    "EnergyTracker": ("sampling.production_loop", "EnergyTracker"),
    "GaussianHill": ("sampling.enhanced_sampling", "GaussianHill"),
    "HybridProductionEngine": ("sampling.production_engine", "HybridProductionEngine"),
    "MetadynamicsEngine": ("sampling.enhanced_sampling", "MetadynamicsEngine"),
    "ProductionCycleReport": ("sampling.production_engine", "ProductionCycleReport"),
    "ProductionRunResult": ("sampling.production_loop", "ProductionRunResult"),
    "ProductionSimulationLoop": ("sampling.production_loop", "ProductionSimulationLoop"),
    "RadiusOfGyrationCV": ("sampling.enhanced_sampling", "RadiusOfGyrationCV"),
    "ReplicaExchangeManager": ("sampling.enhanced_sampling", "ReplicaExchangeManager"),
    "ReplicaState": ("sampling.enhanced_sampling", "ReplicaState"),
    "SimulationLoop": ("sampling.simulation_loop", "SimulationLoop"),
    "SimulationRunResult": ("sampling.simulation_loop", "SimulationRunResult"),
    "TemperatureSchedule": ("sampling.production_loop", "TemperatureSchedule"),
}


def __getattr__(name: str) -> object:
    """Resolve package exports lazily to avoid circular import traps."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
