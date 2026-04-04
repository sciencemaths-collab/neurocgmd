"""Baseline parameter contracts for the established coarse-grained force-field layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.types import FrozenMetadata
from topology.bonds import Bond, BondKind
from topology.system_topology import SystemTopology


def _normalized_type_pair(type_a: str, type_b: str) -> tuple[str, str]:
    return tuple(sorted((type_a, type_b)))


@dataclass(frozen=True, slots=True)
class BondParameter:
    """Established harmonic-bond parameter for one bead-type pair."""

    bead_type_a: str
    bead_type_b: str
    equilibrium_distance: float
    stiffness: float
    kind: BondKind = BondKind.STRUCTURAL
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.bead_type_a.strip() or not self.bead_type_b.strip():
            issues.append("BondParameter bead types must be non-empty strings.")
        if self.equilibrium_distance <= 0.0:
            issues.append("BondParameter equilibrium_distance must be positive.")
        if self.stiffness <= 0.0:
            issues.append("BondParameter stiffness must be positive.")
        return tuple(issues)

    def parameter_key(self) -> tuple[BondKind, tuple[str, str]]:
        return (self.kind, _normalized_type_pair(self.bead_type_a, self.bead_type_b))

    def to_dict(self) -> dict[str, object]:
        return {
            "bead_type_a": self.bead_type_a,
            "bead_type_b": self.bead_type_b,
            "equilibrium_distance": self.equilibrium_distance,
            "stiffness": self.stiffness,
            "kind": self.kind.value,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "BondParameter":
        return cls(
            bead_type_a=str(data["bead_type_a"]),
            bead_type_b=str(data["bead_type_b"]),
            equilibrium_distance=float(data["equilibrium_distance"]),
            stiffness=float(data["stiffness"]),
            kind=BondKind(data.get("kind", BondKind.STRUCTURAL.value)),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class NonbondedParameter:
    """Established Lennard-Jones parameter for one bead-type pair."""

    bead_type_a: str
    bead_type_b: str
    sigma: float
    epsilon: float
    cutoff: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.bead_type_a.strip() or not self.bead_type_b.strip():
            issues.append("NonbondedParameter bead types must be non-empty strings.")
        if self.sigma <= 0.0:
            issues.append("NonbondedParameter sigma must be positive.")
        if self.epsilon < 0.0:
            issues.append("NonbondedParameter epsilon must be non-negative.")
        if self.cutoff <= 0.0:
            issues.append("NonbondedParameter cutoff must be positive.")
        return tuple(issues)

    def parameter_key(self) -> tuple[str, str]:
        return _normalized_type_pair(self.bead_type_a, self.bead_type_b)

    def to_dict(self) -> dict[str, object]:
        return {
            "bead_type_a": self.bead_type_a,
            "bead_type_b": self.bead_type_b,
            "sigma": self.sigma,
            "epsilon": self.epsilon,
            "cutoff": self.cutoff,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "NonbondedParameter":
        return cls(
            bead_type_a=str(data["bead_type_a"]),
            bead_type_b=str(data["bead_type_b"]),
            sigma=float(data["sigma"]),
            epsilon=float(data["epsilon"]),
            cutoff=float(data["cutoff"]),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class BaseForceField:
    """Static force-field parameter store for the early coarse-grained substrate."""

    name: str
    bond_parameters: tuple[BondParameter, ...] = ()
    nonbonded_parameters: tuple[NonbondedParameter, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bond_parameters", tuple(self.bond_parameters))
        object.__setattr__(self, "nonbonded_parameters", tuple(self.nonbonded_parameters))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name.strip():
            issues.append("BaseForceField name must be a non-empty string.")
        bond_keys = [parameter.parameter_key() for parameter in self.bond_parameters]
        if len(bond_keys) != len(set(bond_keys)):
            issues.append("Bond parameters must not contain duplicate kind/type keys.")
        nonbonded_keys = [parameter.parameter_key() for parameter in self.nonbonded_parameters]
        if len(nonbonded_keys) != len(set(nonbonded_keys)):
            issues.append("Nonbonded parameters must not contain duplicate type-pair keys.")
        return tuple(issues)

    def bond_parameter_for_bead_types(
        self,
        bead_type_a: str,
        bead_type_b: str,
        *,
        kind: BondKind = BondKind.STRUCTURAL,
    ) -> BondParameter:
        key = (kind, _normalized_type_pair(bead_type_a, bead_type_b))
        for parameter in self.bond_parameters:
            if parameter.parameter_key() == key:
                return parameter
        raise KeyError(key)

    def bond_parameter_for(self, topology: SystemTopology, bond: Bond) -> BondParameter:
        bead_a = topology.bead_for_particle(bond.particle_index_a)
        bead_b = topology.bead_for_particle(bond.particle_index_b)
        return self.bond_parameter_for_bead_types(
            bead_a.bead_type,
            bead_b.bead_type,
            kind=bond.kind,
        )

    def nonbonded_parameter_for_bead_types(
        self, bead_type_a: str, bead_type_b: str
    ) -> NonbondedParameter:
        key = _normalized_type_pair(bead_type_a, bead_type_b)
        for parameter in self.nonbonded_parameters:
            if parameter.parameter_key() == key:
                return parameter
        raise KeyError(key)

    def nonbonded_parameter_for_pair(
        self, topology: SystemTopology, particle_index_a: int, particle_index_b: int
    ) -> NonbondedParameter:
        bead_a = topology.bead_for_particle(particle_index_a)
        bead_b = topology.bead_for_particle(particle_index_b)
        return self.nonbonded_parameter_for_bead_types(bead_a.bead_type, bead_b.bead_type)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "bond_parameters": [parameter.to_dict() for parameter in self.bond_parameters],
            "nonbonded_parameters": [
                parameter.to_dict() for parameter in self.nonbonded_parameters
            ],
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "BaseForceField":
        return cls(
            name=str(data["name"]),
            bond_parameters=tuple(
                BondParameter.from_dict(item) for item in data.get("bond_parameters", ())
            ),
            nonbonded_parameters=tuple(
                NonbondedParameter.from_dict(item)
                for item in data.get("nonbonded_parameters", ())
            ),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )

