"""Time integration schemes and simulation stepping hooks."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from integrators.baoab import BAOABIntegrator
    from integrators.base import ForceEvaluator, IntegratorStepResult, StateIntegrator
    from integrators.langevin import LangevinIntegrator

__all__ = [
    "BAOABIntegrator",
    "ForceEvaluator",
    "IntegratorStepResult",
    "LangevinIntegrator",
    "StateIntegrator",
]

_LAZY_EXPORTS = {
    "BAOABIntegrator": ("integrators.baoab", "BAOABIntegrator"),
    "ForceEvaluator": ("integrators.base", "ForceEvaluator"),
    "IntegratorStepResult": ("integrators.base", "IntegratorStepResult"),
    "LangevinIntegrator": ("integrators.langevin", "LangevinIntegrator"),
    "StateIntegrator": ("integrators.base", "StateIntegrator"),
}


def __getattr__(name: str) -> object:
    """Resolve integrator exports lazily to avoid circular package import chains."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
