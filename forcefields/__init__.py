"""Force field definitions and parameterization boundaries."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from forcefields.base_forcefield import (
    BaseForceField,
    BondParameter,
    NonbondedParameter,
)
from forcefields.bonded_potentials import (
    AngleForceModel,
    AngleParameter,
    DihedralForceModel,
    DihedralParameter,
    ImproperDihedralParameter,
)
from forcefields.nonbonded_potentials import (
    COULOMB_CONSTANT,
    CoulombPotential,
    ShiftedForceLJ,
    SwitchFunction,
    WCAPotential,
)
from forcefields.protein_import_forcefield import ImportedProteinForceFieldBuilder
from forcefields.protein_shadow_profiles import (
    ProteinBeadFamily,
    ProteinBeadFamilyAssignment,
    ProteinShadowProfileFactory,
)
from forcefields.spatial_semantic_profiles import (
    ProteinSpatialProfileFactory,
    SpatialSemanticParameterSet,
    SpatialSemanticProfile,
)
from forcefields.trusted_sources import (
    TrustedNonbondedProfile,
    TrustedParameterSet,
    TrustedScienceSource,
)

if TYPE_CHECKING:
    from forcefields.hybrid_engine import (
        HybridClassicalKernelPolicy,
        HybridForceEngine,
        HybridForceEnginePolicy,
        HybridForceResult,
    )

__all__ = [
    "AngleForceModel",
    "AngleParameter",
    "BaseForceField",
    "BondParameter",
    "COULOMB_CONSTANT",
    "CoulombPotential",
    "DihedralForceModel",
    "DihedralParameter",
    "HybridClassicalKernelPolicy",
    "HybridForceEngine",
    "HybridForceEnginePolicy",
    "HybridForceResult",
    "ImproperDihedralParameter",
    "ImportedProteinForceFieldBuilder",
    "NonbondedParameter",
    "ProteinBeadFamily",
    "ProteinBeadFamilyAssignment",
    "ProteinShadowProfileFactory",
    "ProteinSpatialProfileFactory",
    "ShiftedForceLJ",
    "SpatialSemanticParameterSet",
    "SpatialSemanticProfile",
    "SwitchFunction",
    "TrustedNonbondedProfile",
    "TrustedParameterSet",
    "TrustedScienceSource",
    "WCAPotential",
]

_LAZY_EXPORTS = {
    "HybridClassicalKernelPolicy": ("forcefields.hybrid_engine", "HybridClassicalKernelPolicy"),
    "HybridForceEngine": ("forcefields.hybrid_engine", "HybridForceEngine"),
    "HybridForceEnginePolicy": ("forcefields.hybrid_engine", "HybridForceEnginePolicy"),
    "HybridForceResult": ("forcefields.hybrid_engine", "HybridForceResult"),
}


def __getattr__(name: str) -> object:
    """Resolve high-level forcefield exports lazily to avoid circular imports."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
