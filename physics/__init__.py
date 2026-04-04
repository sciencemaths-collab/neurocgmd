"""Physics kernels, energy terms, and force computations."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physics.constraints import (
        ConstrainedIntegratorWrapper,
        ConstraintResult,
        DistanceConstraint,
        LINCSolver,
        SHAKESolver,
    )
    from physics.electrostatics import (
        COULOMB_CONSTANT,
        ChargeSet,
        EwaldParameters,
        EwaldResult,
        EwaldSummation,
        ReactionFieldElectrostatics,
    )
    from physics.neighbor_list import (
        AcceleratedNonbondedForceModel,
        CellList,
        NeighborList,
        NeighborListBuilder,
    )
    from physics.periodic_boundary import (
        PBCHandler,
        TriclinicPBCHandler,
        minimum_image_displacement,
        minimum_image_distance,
        wrap_positions,
    )
    from physics.production_evaluator import ProductionForceEvaluator

__all__: tuple[str, ...] = (
    "AcceleratedNonbondedForceModel",
    "COULOMB_CONSTANT",
    "CellList",
    "ChargeSet",
    "ConstrainedIntegratorWrapper",
    "ConstraintResult",
    "DistanceConstraint",
    "EwaldParameters",
    "EwaldResult",
    "EwaldSummation",
    "LINCSolver",
    "NeighborList",
    "NeighborListBuilder",
    "PBCHandler",
    "ProductionForceEvaluator",
    "ReactionFieldElectrostatics",
    "SHAKESolver",
    "TriclinicPBCHandler",
    "minimum_image_displacement",
    "minimum_image_distance",
    "wrap_positions",
)

_LAZY_EXPORTS = {
    "AcceleratedNonbondedForceModel": ("physics.neighbor_list", "AcceleratedNonbondedForceModel"),
    "COULOMB_CONSTANT": ("physics.electrostatics", "COULOMB_CONSTANT"),
    "CellList": ("physics.neighbor_list", "CellList"),
    "ChargeSet": ("physics.electrostatics", "ChargeSet"),
    "ConstrainedIntegratorWrapper": ("physics.constraints", "ConstrainedIntegratorWrapper"),
    "ConstraintResult": ("physics.constraints", "ConstraintResult"),
    "DistanceConstraint": ("physics.constraints", "DistanceConstraint"),
    "EwaldParameters": ("physics.electrostatics", "EwaldParameters"),
    "EwaldResult": ("physics.electrostatics", "EwaldResult"),
    "EwaldSummation": ("physics.electrostatics", "EwaldSummation"),
    "LINCSolver": ("physics.constraints", "LINCSolver"),
    "NeighborList": ("physics.neighbor_list", "NeighborList"),
    "NeighborListBuilder": ("physics.neighbor_list", "NeighborListBuilder"),
    "PBCHandler": ("physics.periodic_boundary", "PBCHandler"),
    "ProductionForceEvaluator": ("physics.production_evaluator", "ProductionForceEvaluator"),
    "ReactionFieldElectrostatics": ("physics.electrostatics", "ReactionFieldElectrostatics"),
    "SHAKESolver": ("physics.constraints", "SHAKESolver"),
    "TriclinicPBCHandler": ("physics.periodic_boundary", "TriclinicPBCHandler"),
    "minimum_image_displacement": ("physics.periodic_boundary", "minimum_image_displacement"),
    "minimum_image_distance": ("physics.periodic_boundary", "minimum_image_distance"),
    "wrap_positions": ("physics.periodic_boundary", "wrap_positions"),
}


def __getattr__(name: str) -> object:
    """Resolve physics exports lazily to avoid package-level circular imports."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
