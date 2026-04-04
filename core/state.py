"""Units-aware immutable state models for the simulation substrate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from math import fsum
from typing import ClassVar, Self

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import (
    BoolVector3,
    FrozenMetadata,
    Matrix3x3,
    ScalarTuple,
    SimulationId,
    StateId,
    Vector3,
    VectorTuple,
    coerce_bool_vector3,
    coerce_scalar,
    coerce_scalar_tuple,
    coerce_vector3,
    coerce_vector_block,
)

ZERO_VECTOR: Vector3 = (0.0, 0.0, 0.0)


class EnsembleKind(StrEnum):
    """Supported thermodynamic ensemble tags for the early engine scaffold."""

    NVE = "NVE"
    NVT = "NVT"
    NPT = "NPT"
    CUSTOM = "CUSTOM"


def _zero_vectors(count: int) -> VectorTuple:
    return tuple(ZERO_VECTOR for _ in range(count))


def _determinant(matrix: Matrix3x3) -> float:
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = matrix
    return (
        ax * (by * cz - bz * cy)
        - ay * (bx * cz - bz * cx)
        + az * (bx * cy - by * cx)
    )


@dataclass(frozen=True, slots=True)
class UnitSystem(ValidatableComponent):
    """Named unit system attached to every simulation state."""

    name: str = "md_nano"
    length_unit: str = "nm"
    time_unit: str = "ps"
    mass_unit: str = "amu"
    energy_unit: str = "kJ/mol"
    temperature_unit: str = "K"
    pressure_unit: str = "bar"
    force_unit: str = "kJ/(mol*nm)"
    charge_unit: str = "e"

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for field_name, value in self.to_dict().items():
            if not isinstance(value, str) or not value.strip():
                issues.append(f"{field_name} must be a non-empty string.")
        return tuple(issues)

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "length_unit": self.length_unit,
            "time_unit": self.time_unit,
            "mass_unit": self.mass_unit,
            "energy_unit": self.energy_unit,
            "temperature_unit": self.temperature_unit,
            "pressure_unit": self.pressure_unit,
            "force_unit": self.force_unit,
            "charge_unit": self.charge_unit,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "UnitSystem":
        return cls(**dict(data))

    @classmethod
    def md_nano(cls) -> "UnitSystem":
        """Return the default MD-friendly nanometer/picosecond unit convention."""

        return cls()


@dataclass(frozen=True, slots=True)
class SimulationCell(ValidatableComponent):
    """Periodic cell geometry and boundary activation flags."""

    box_vectors: Matrix3x3
    periodic_axes: BoolVector3 = (True, True, True)
    origin: Vector3 = ZERO_VECTOR

    def __post_init__(self) -> None:
        object.__setattr__(self, "box_vectors", coerce_vector_block(self.box_vectors, "box_vectors"))
        object.__setattr__(self, "periodic_axes", coerce_bool_vector3(self.periodic_axes, "periodic_axes"))
        object.__setattr__(self, "origin", coerce_vector3(self.origin, "origin"))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        volume = self.volume()
        if volume <= 0.0:
            issues.append("SimulationCell must have a positive non-zero volume.")
        return tuple(issues)

    def volume(self) -> float:
        """Return the absolute cell volume."""

        return abs(_determinant(self.box_vectors))

    def to_dict(self) -> dict[str, object]:
        return {
            "box_vectors": [list(vector) for vector in self.box_vectors],
            "periodic_axes": list(self.periodic_axes),
            "origin": list(self.origin),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SimulationCell":
        return cls(
            box_vectors=data["box_vectors"],
            periodic_axes=data.get("periodic_axes", (True, True, True)),
            origin=data.get("origin", ZERO_VECTOR),
        )


@dataclass(frozen=True, slots=True)
class ThermodynamicState(ValidatableComponent):
    """Thermodynamic controls associated with a simulation state."""

    ensemble: EnsembleKind = EnsembleKind.NVE
    target_temperature: float | None = None
    target_pressure: float | None = None
    friction_coefficient: float | None = None

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.target_temperature is not None and self.target_temperature <= 0.0:
            issues.append("target_temperature must be positive when provided.")
        if self.target_pressure is not None and self.target_pressure <= 0.0:
            issues.append("target_pressure must be positive when provided.")
        if self.friction_coefficient is not None and self.friction_coefficient < 0.0:
            issues.append("friction_coefficient must be non-negative when provided.")
        if self.ensemble in {EnsembleKind.NVT, EnsembleKind.NPT} and self.target_temperature is None:
            issues.append(f"{self.ensemble.value} requires target_temperature.")
        if self.ensemble == EnsembleKind.NPT and self.target_pressure is None:
            issues.append("NPT requires target_pressure.")
        if self.ensemble == EnsembleKind.NVE and (
            self.target_temperature is not None or self.target_pressure is not None
        ):
            issues.append("NVE should not define target_temperature or target_pressure.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "ensemble": self.ensemble.value,
            "target_temperature": self.target_temperature,
            "target_pressure": self.target_pressure,
            "friction_coefficient": self.friction_coefficient,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ThermodynamicState":
        return cls(
            ensemble=EnsembleKind(data.get("ensemble", EnsembleKind.NVE.value)),
            target_temperature=data.get("target_temperature"),
            target_pressure=data.get("target_pressure"),
            friction_coefficient=data.get("friction_coefficient"),
        )


@dataclass(frozen=True, slots=True)
class ParticleState(ValidatableComponent):
    """Particle-resolved dynamic state independent of topology semantics."""

    positions: VectorTuple
    masses: ScalarTuple
    velocities: VectorTuple = ()
    forces: VectorTuple = ()
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        positions = coerce_vector_block(self.positions, "positions")
        masses = coerce_scalar_tuple(self.masses, "masses")
        particle_count = len(positions)
        velocities = (
            coerce_vector_block(self.velocities, "velocities")
            if self.velocities
            else _zero_vectors(particle_count)
        )
        forces = (
            coerce_vector_block(self.forces, "forces")
            if self.forces
            else _zero_vectors(particle_count)
        )
        labels = tuple(self.labels) if self.labels is not None else None

        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "forces", forces)
        object.__setattr__(self, "labels", labels)

        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    @property
    def particle_count(self) -> int:
        return len(self.positions)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.particle_count == 0:
            issues.append("ParticleState must contain at least one particle.")
        if len(self.masses) != self.particle_count:
            issues.append("masses length must match positions length.")
        if len(self.velocities) != self.particle_count:
            issues.append("velocities length must match positions length.")
        if len(self.forces) != self.particle_count:
            issues.append("forces length must match positions length.")
        if any(mass <= 0.0 for mass in self.masses):
            issues.append("All particle masses must be strictly positive.")
        if self.labels is not None:
            if len(self.labels) != self.particle_count:
                issues.append("labels length must match positions length.")
            if any(not label.strip() for label in self.labels):
                issues.append("Particle labels must be non-empty strings.")
        return tuple(issues)

    def total_mass(self) -> float:
        return fsum(self.masses)

    def kinetic_energy(self) -> float:
        """Return classical kinetic energy in the current unit system."""

        return fsum(
            0.5 * mass * (vx * vx + vy * vy + vz * vz)
            for mass, (vx, vy, vz) in zip(self.masses, self.velocities, strict=True)
        )

    def center_of_mass(self) -> Vector3:
        total_mass = self.total_mass()
        return (
            fsum(mass * position[0] for mass, position in zip(self.masses, self.positions, strict=True))
            / total_mass,
            fsum(mass * position[1] for mass, position in zip(self.masses, self.positions, strict=True))
            / total_mass,
            fsum(mass * position[2] for mass, position in zip(self.masses, self.positions, strict=True))
            / total_mass,
        )

    def with_positions(self, positions: Sequence[Sequence[int | float]]) -> "ParticleState":
        return ParticleState(
            positions=positions,
            masses=self.masses,
            velocities=self.velocities,
            forces=self.forces,
            labels=self.labels,
        )

    def with_velocities(self, velocities: Sequence[Sequence[int | float]]) -> "ParticleState":
        return ParticleState(
            positions=self.positions,
            masses=self.masses,
            velocities=velocities,
            forces=self.forces,
            labels=self.labels,
        )

    def with_forces(self, forces: Sequence[Sequence[int | float]]) -> "ParticleState":
        return ParticleState(
            positions=self.positions,
            masses=self.masses,
            velocities=self.velocities,
            forces=forces,
            labels=self.labels,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "positions": [list(vector) for vector in self.positions],
            "masses": list(self.masses),
            "velocities": [list(vector) for vector in self.velocities],
            "forces": [list(vector) for vector in self.forces],
            "labels": list(self.labels) if self.labels is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ParticleState":
        labels = data.get("labels")
        return cls(
            positions=data["positions"],
            masses=data["masses"],
            velocities=data.get("velocities", ()),
            forces=data.get("forces", ()),
            labels=tuple(labels) if labels is not None else None,
        )


@dataclass(frozen=True, slots=True)
class StateProvenance(ValidatableComponent):
    """Immutable provenance record for a simulation snapshot."""

    simulation_id: SimulationId
    state_id: StateId
    parent_state_id: StateId | None
    created_by: str
    stage: str
    notes: str = ""
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "simulation_id", SimulationId(str(self.simulation_id)))
        object.__setattr__(self, "state_id", StateId(str(self.state_id)))
        if self.parent_state_id is not None:
            object.__setattr__(self, "parent_state_id", StateId(str(self.parent_state_id)))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not str(self.simulation_id).strip():
            issues.append("simulation_id must be a non-empty string.")
        if not str(self.state_id).strip():
            issues.append("state_id must be a non-empty string.")
        if self.parent_state_id is not None and str(self.parent_state_id) == str(self.state_id):
            issues.append("parent_state_id cannot equal state_id.")
        if not self.created_by.strip():
            issues.append("created_by must be a non-empty string.")
        if not self.stage.strip():
            issues.append("stage must be a non-empty string.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "simulation_id": str(self.simulation_id),
            "state_id": str(self.state_id),
            "parent_state_id": str(self.parent_state_id) if self.parent_state_id else None,
            "created_by": self.created_by,
            "stage": self.stage,
            "notes": self.notes,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "StateProvenance":
        parent_state_id = data.get("parent_state_id")
        return cls(
            simulation_id=SimulationId(str(data["simulation_id"])),
            state_id=StateId(str(data["state_id"])),
            parent_state_id=StateId(str(parent_state_id)) if parent_state_id else None,
            created_by=str(data["created_by"]),
            stage=str(data["stage"]),
            notes=str(data.get("notes", "")),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class SimulationState(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Canonical simulation snapshot consumed by every downstream subsystem."""

    name: ClassVar[str] = "simulation_state"
    classification: ClassVar[str] = "[adapted]"

    units: UnitSystem
    particles: ParticleState
    thermodynamics: ThermodynamicState
    provenance: StateProvenance
    cell: SimulationCell | None = None
    time: float = 0.0
    step: int = 0
    potential_energy: float | None = None
    observables: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", coerce_scalar(self.time, "time"))
        if self.potential_energy is not None:
            object.__setattr__(
                self,
                "potential_energy",
                coerce_scalar(self.potential_energy, "potential_energy"),
            )
        if not isinstance(self.observables, FrozenMetadata):
            object.__setattr__(self, "observables", FrozenMetadata(self.observables))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    @property
    def particle_count(self) -> int:
        return self.particles.particle_count

    def kinetic_energy(self) -> float:
        return self.particles.kinetic_energy()

    def total_energy(self) -> float | None:
        if self.potential_energy is None:
            return None
        return self.kinetic_energy() + self.potential_energy

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        issues.extend(self.units.validate())
        issues.extend(self.particles.validate())
        issues.extend(self.thermodynamics.validate())
        issues.extend(self.provenance.validate())
        if self.cell is not None:
            issues.extend(self.cell.validate())
        if self.time < 0.0:
            issues.append("time must be non-negative.")
        if self.step < 0:
            issues.append("step must be non-negative.")
        return tuple(issues)

    def describe_role(self) -> str:
        return (
            "Carries the particle-resolved physical snapshot, thermodynamic controls, "
            "units, and provenance required by all downstream subsystems."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return ("core/types.py", "core/interfaces.py")

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/state_model.md",
            "docs/sections/section_02_core_state.md",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "units": self.units.to_dict(),
            "particles": self.particles.to_dict(),
            "thermodynamics": self.thermodynamics.to_dict(),
            "provenance": self.provenance.to_dict(),
            "cell": self.cell.to_dict() if self.cell is not None else None,
            "time": self.time,
            "step": self.step,
            "potential_energy": self.potential_energy,
            "observables": self.observables.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SimulationState":
        cell_data = data.get("cell")
        return cls(
            units=UnitSystem.from_dict(data["units"]),
            particles=ParticleState.from_dict(data["particles"]),
            thermodynamics=ThermodynamicState.from_dict(data["thermodynamics"]),
            provenance=StateProvenance.from_dict(data["provenance"]),
            cell=SimulationCell.from_dict(cell_data) if cell_data is not None else None,
            time=data.get("time", 0.0),
            step=int(data.get("step", 0)),
            potential_energy=data.get("potential_energy"),
            observables=FrozenMetadata(data.get("observables", {})),
        )

