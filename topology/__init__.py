"""Static topology models for beads, bonds, and system composition."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from topology.beads import Bead, BeadRole, BeadType
    from topology.bonds import Bond, BondKind, build_neighbor_map, connected_components
    from topology.protein_coarse_mapping import ProteinCoarseMapper
    from topology.protein_import_models import (
        ImportedBeadBlock,
        ImportedProteinSystem,
        ImportedResidueRecord,
    )
    from topology.system_topology import SystemTopology

__all__ = [
    "Bead",
    "BeadRole",
    "BeadType",
    "Bond",
    "BondKind",
    "ImportedBeadBlock",
    "ImportedProteinSystem",
    "ImportedResidueRecord",
    "ProteinCoarseMapper",
    "SystemTopology",
    "build_neighbor_map",
    "connected_components",
]

_LAZY_EXPORTS = {
    "Bead": ("topology.beads", "Bead"),
    "BeadRole": ("topology.beads", "BeadRole"),
    "BeadType": ("topology.beads", "BeadType"),
    "Bond": ("topology.bonds", "Bond"),
    "BondKind": ("topology.bonds", "BondKind"),
    "ImportedBeadBlock": ("topology.protein_import_models", "ImportedBeadBlock"),
    "ImportedProteinSystem": ("topology.protein_import_models", "ImportedProteinSystem"),
    "ImportedResidueRecord": ("topology.protein_import_models", "ImportedResidueRecord"),
    "ProteinCoarseMapper": ("topology.protein_coarse_mapping", "ProteinCoarseMapper"),
    "SystemTopology": ("topology.system_topology", "SystemTopology"),
    "build_neighbor_map": ("topology.bonds", "build_neighbor_map"),
    "connected_components": ("topology.bonds", "connected_components"),
}


def __getattr__(name: str) -> object:
    """Resolve topology exports lazily to avoid package-level import cycles."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
