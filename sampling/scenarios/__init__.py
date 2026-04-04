"""Scenario definitions for concrete simulation objectives and demos."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sampling.scenarios.barnase_barstar import BarnaseBarstarScenario
    from sampling.scenarios.complex_assembly import (
        ComplexAssemblyProgress,
        ComplexAssemblySetup,
        EncounterComplexScenario,
    )
    from sampling.scenarios.contracts import DashboardScenario
    from sampling.scenarios.imported_protein import (
        ImportedProteinComplexScenario,
        ImportedProteinScenarioSpec,
    )
    from sampling.scenarios.spike_ace2 import SpikeAce2Scenario

__all__ = [
    "BarnaseBarstarScenario",
    "ComplexAssemblyProgress",
    "ComplexAssemblySetup",
    "DashboardScenario",
    "EncounterComplexScenario",
    "ImportedProteinComplexScenario",
    "ImportedProteinScenarioSpec",
    "SpikeAce2Scenario",
]

_LAZY_EXPORTS = {
    "BarnaseBarstarScenario": ("sampling.scenarios.barnase_barstar", "BarnaseBarstarScenario"),
    "ComplexAssemblyProgress": ("sampling.scenarios.complex_assembly", "ComplexAssemblyProgress"),
    "ComplexAssemblySetup": ("sampling.scenarios.complex_assembly", "ComplexAssemblySetup"),
    "DashboardScenario": ("sampling.scenarios.contracts", "DashboardScenario"),
    "EncounterComplexScenario": ("sampling.scenarios.complex_assembly", "EncounterComplexScenario"),
    "ImportedProteinComplexScenario": (
        "sampling.scenarios.imported_protein",
        "ImportedProteinComplexScenario",
    ),
    "ImportedProteinScenarioSpec": (
        "sampling.scenarios.imported_protein",
        "ImportedProteinScenarioSpec",
    ),
    "SpikeAce2Scenario": ("sampling.scenarios.spike_ace2", "SpikeAce2Scenario"),
}


def __getattr__(name: str) -> object:
    """Resolve scenario exports lazily to avoid package-level import cycles."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
