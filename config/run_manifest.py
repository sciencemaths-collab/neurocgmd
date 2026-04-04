"""TOML-backed run manifest contracts for the user-facing MD workflow.

[hybrid]

The manifest borrows familiar production-MD configuration ideas from
established packages while keeping NeuroCGMD's novel architecture explicit.
It is the single control plane for prepare/run/analyze workflows.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from config.protein_mapping import ProteinEntityGroup, ProteinMappingConfig
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import EnsembleKind
from core.types import FrozenMetadata, coerce_scalar


def _coerce_metadata(value: FrozenMetadata | Mapping[str, object] | None) -> FrozenMetadata:
    return value if isinstance(value, FrozenMetadata) else FrozenMetadata(value)


class SolventMode(StrEnum):
    """Preparation-time solvent modes."""

    VACUUM = "vacuum"
    IMPLICIT = "implicit"
    EXPLICIT = "explicit"


class ProtonationMode(StrEnum):
    """Preparation-time protonation strategies."""

    AUTO = "auto"
    PRESERVE = "preserve"
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class SystemConfig(ValidatableComponent):
    """System-level inputs for a run manifest."""

    name: str
    structure: str
    description: str = ""
    entity_groups: tuple[ProteinEntityGroup, ...] = ()
    mapping_config: ProteinMappingConfig = field(default_factory=ProteinMappingConfig)
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_groups", tuple(self.entity_groups))
        if not isinstance(self.mapping_config, ProteinMappingConfig):
            object.__setattr__(self, "mapping_config", ProteinMappingConfig.from_dict(self.mapping_config))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name.strip():
            issues.append("system.name must be a non-empty string.")
        if not self.structure.strip():
            issues.append("system.structure must be a non-empty string.")
        issues.extend(self.mapping_config.validate())
        for group in self.entity_groups:
            issues.extend(group.validate())
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "structure": self.structure,
            "description": self.description,
            "entity_groups": [group.to_dict() for group in self.entity_groups],
            "mapping_config": self.mapping_config.to_dict(),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SystemConfig":
        return cls(
            name=str(data["name"]),
            structure=str(data["structure"]),
            description=str(data.get("description", "")),
            entity_groups=tuple(
                ProteinEntityGroup.from_dict(item)
                for item in data.get("entity_groups", ())
            ),
            mapping_config=ProteinMappingConfig.from_dict(data.get("mapping_config", {})),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class PrepareConfig(ValidatableComponent):
    """Preparation controls for hydrogens, solvent, and ions."""

    add_hydrogens: bool = True
    ph: float = 7.4
    protonation: ProtonationMode = ProtonationMode.AUTO
    fix_missing_atoms: bool = False
    fix_missing_residues: bool = False
    solvent_mode: SolventMode = SolventMode.EXPLICIT
    water_model: str = "tip3p"
    box_type: str = "dodecahedron"
    padding_nm: float = 1.0
    neutralize: bool = True
    salt: str = "NaCl"
    ionic_strength_molar: float = 0.15
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ph", coerce_scalar(self.ph, "ph"))
        object.__setattr__(self, "padding_nm", coerce_scalar(self.padding_nm, "padding_nm"))
        object.__setattr__(
            self,
            "ionic_strength_molar",
            coerce_scalar(self.ionic_strength_molar, "ionic_strength_molar"),
        )
        object.__setattr__(self, "protonation", ProtonationMode(self.protonation))
        object.__setattr__(self, "solvent_mode", SolventMode(self.solvent_mode))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.ph <= 0.0:
            issues.append("prepare.ph must be strictly positive.")
        if self.padding_nm < 0.0:
            issues.append("prepare.padding_nm must be non-negative.")
        if not self.water_model.strip():
            issues.append("prepare.water_model must be a non-empty string.")
        if not self.box_type.strip():
            issues.append("prepare.box_type must be a non-empty string.")
        if not self.salt.strip():
            issues.append("prepare.salt must be a non-empty string.")
        if self.ionic_strength_molar < 0.0:
            issues.append("prepare.ionic_strength_molar must be non-negative.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "add_hydrogens": self.add_hydrogens,
            "ph": self.ph,
            "protonation": self.protonation.value,
            "fix_missing_atoms": self.fix_missing_atoms,
            "fix_missing_residues": self.fix_missing_residues,
            "solvent_mode": self.solvent_mode.value,
            "water_model": self.water_model,
            "box_type": self.box_type,
            "padding_nm": self.padding_nm,
            "neutralize": self.neutralize,
            "salt": self.salt,
            "ionic_strength_molar": self.ionic_strength_molar,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PrepareConfig":
        return cls(
            add_hydrogens=bool(data.get("add_hydrogens", True)),
            ph=float(data.get("ph", 7.4)),
            protonation=ProtonationMode(str(data.get("protonation", ProtonationMode.AUTO.value))),
            fix_missing_atoms=bool(data.get("fix_missing_atoms", False)),
            fix_missing_residues=bool(data.get("fix_missing_residues", False)),
            solvent_mode=SolventMode(str(data.get("solvent_mode", SolventMode.EXPLICIT.value))),
            water_model=str(data.get("water_model", "tip3p")),
            box_type=str(data.get("box_type", "dodecahedron")),
            padding_nm=float(data.get("padding_nm", 1.0)),
            neutralize=bool(data.get("neutralize", True)),
            salt=str(data.get("salt", "NaCl")),
            ionic_strength_molar=float(data.get("ionic_strength_molar", 0.15)),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ForcefieldConfig(ValidatableComponent):
    """Named force-field and constraint choices."""

    protein: str = "hybrid_protein_v1"
    water: str = "tip3p"
    ions: str = "joung_cheatham"
    nonbonded: str = "lj"
    electrostatics: str = "screened_coulomb"
    constraints: str = "h_bonds"
    constraint_algorithm: str = "lincs"
    lincs_order: int = 4
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for field_name in (
            "protein",
            "water",
            "ions",
            "nonbonded",
            "electrostatics",
            "constraints",
            "constraint_algorithm",
        ):
            if not getattr(self, field_name).strip():
                issues.append(f"forcefield.{field_name} must be a non-empty string.")
        if self.lincs_order <= 0:
            issues.append("forcefield.lincs_order must be strictly positive.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "protein": self.protein,
            "water": self.water,
            "ions": self.ions,
            "nonbonded": self.nonbonded,
            "electrostatics": self.electrostatics,
            "constraints": self.constraints,
            "constraint_algorithm": self.constraint_algorithm,
            "lincs_order": self.lincs_order,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ForcefieldConfig":
        return cls(
            protein=str(data.get("protein", "hybrid_protein_v1")),
            water=str(data.get("water", "tip3p")),
            ions=str(data.get("ions", "joung_cheatham")),
            nonbonded=str(data.get("nonbonded", "lj")),
            electrostatics=str(data.get("electrostatics", "screened_coulomb")),
            constraints=str(data.get("constraints", "h_bonds")),
            constraint_algorithm=str(data.get("constraint_algorithm", "lincs")),
            lincs_order=int(data.get("lincs_order", 4)),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class NeighborListConfig(ValidatableComponent):
    """Neighbor-search and cutoff controls."""

    cutoff_scheme: str = "Verlet"
    vdw_cutoff_nm: float = 1.2
    coulomb_cutoff_nm: float = 1.2
    neighbor_skin_nm: float = 0.3
    update_stride: int = 10
    pme: bool = False
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "vdw_cutoff_nm", coerce_scalar(self.vdw_cutoff_nm, "vdw_cutoff_nm"))
        object.__setattr__(
            self,
            "coulomb_cutoff_nm",
            coerce_scalar(self.coulomb_cutoff_nm, "coulomb_cutoff_nm"),
        )
        object.__setattr__(self, "neighbor_skin_nm", coerce_scalar(self.neighbor_skin_nm, "neighbor_skin_nm"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.cutoff_scheme.strip():
            issues.append("neighbor_list.cutoff_scheme must be a non-empty string.")
        if self.vdw_cutoff_nm <= 0.0:
            issues.append("neighbor_list.vdw_cutoff_nm must be strictly positive.")
        if self.coulomb_cutoff_nm <= 0.0:
            issues.append("neighbor_list.coulomb_cutoff_nm must be strictly positive.")
        if self.neighbor_skin_nm < 0.0:
            issues.append("neighbor_list.neighbor_skin_nm must be non-negative.")
        if self.update_stride <= 0:
            issues.append("neighbor_list.update_stride must be strictly positive.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "cutoff_scheme": self.cutoff_scheme,
            "vdw_cutoff_nm": self.vdw_cutoff_nm,
            "coulomb_cutoff_nm": self.coulomb_cutoff_nm,
            "neighbor_skin_nm": self.neighbor_skin_nm,
            "update_stride": self.update_stride,
            "pme": self.pme,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "NeighborListConfig":
        return cls(
            cutoff_scheme=str(data.get("cutoff_scheme", "Verlet")),
            vdw_cutoff_nm=float(data.get("vdw_cutoff_nm", 1.2)),
            coulomb_cutoff_nm=float(data.get("coulomb_cutoff_nm", 1.2)),
            neighbor_skin_nm=float(data.get("neighbor_skin_nm", 0.3)),
            update_stride=int(data.get("update_stride", 10)),
            pme=bool(data.get("pme", False)),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class MinimizationStageConfig(ValidatableComponent):
    """Energy-minimization stage controls."""

    enabled: bool = True
    algorithm: str = "steepest_descent"
    max_steps: int = 5000
    tolerance: float = 100.0
    step_size: float = 0.001
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tolerance", coerce_scalar(self.tolerance, "tolerance"))
        object.__setattr__(self, "step_size", coerce_scalar(self.step_size, "step_size"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.algorithm.strip():
            issues.append("stages.em.algorithm must be a non-empty string.")
        if self.max_steps < 0:
            issues.append("stages.em.max_steps must be non-negative.")
        if self.tolerance <= 0.0:
            issues.append("stages.em.tolerance must be strictly positive.")
        if self.step_size <= 0.0:
            issues.append("stages.em.step_size must be strictly positive.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "algorithm": self.algorithm,
            "max_steps": self.max_steps,
            "tolerance": self.tolerance,
            "step_size": self.step_size,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MinimizationStageConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            algorithm=str(data.get("algorithm", "steepest_descent")),
            max_steps=int(data.get("max_steps", 5000)),
            tolerance=float(data.get("tolerance", 100.0)),
            step_size=float(data.get("step_size", 0.001)),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class DynamicsStageConfig(ValidatableComponent):
    """NVT/NPT/production dynamics stage controls."""

    enabled: bool = True
    ensemble: EnsembleKind = EnsembleKind.NVT
    dt: float = 0.002
    nsteps: int = 1000
    temperature: float = 300.0
    pressure: float | None = None
    friction_coefficient: float = 1.0
    trajectory_stride: int = 100
    energy_stride: int = 100
    checkpoint_stride: int = 500
    eval_stride: int = 10
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ensemble", EnsembleKind(self.ensemble))
        object.__setattr__(self, "dt", coerce_scalar(self.dt, "dt"))
        object.__setattr__(self, "temperature", coerce_scalar(self.temperature, "temperature"))
        if self.pressure is not None:
            object.__setattr__(self, "pressure", coerce_scalar(self.pressure, "pressure"))
        object.__setattr__(
            self,
            "friction_coefficient",
            coerce_scalar(self.friction_coefficient, "friction_coefficient"),
        )
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.dt <= 0.0:
            issues.append("stage.dt must be strictly positive.")
        if self.nsteps < 0:
            issues.append("stage.nsteps must be non-negative.")
        if self.temperature <= 0.0:
            issues.append("stage.temperature must be strictly positive.")
        if self.ensemble == EnsembleKind.NPT and (self.pressure is None or self.pressure <= 0.0):
            issues.append("NPT stages require a strictly positive pressure.")
        if self.pressure is not None and self.pressure <= 0.0:
            issues.append("stage.pressure must be strictly positive when provided.")
        if self.friction_coefficient < 0.0:
            issues.append("stage.friction_coefficient must be non-negative.")
        for field_name in ("trajectory_stride", "energy_stride", "checkpoint_stride", "eval_stride"):
            if getattr(self, field_name) <= 0:
                issues.append(f"stage.{field_name} must be strictly positive.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "ensemble": self.ensemble.value,
            "dt": self.dt,
            "nsteps": self.nsteps,
            "temperature": self.temperature,
            "pressure": self.pressure,
            "friction_coefficient": self.friction_coefficient,
            "trajectory_stride": self.trajectory_stride,
            "energy_stride": self.energy_stride,
            "checkpoint_stride": self.checkpoint_stride,
            "eval_stride": self.eval_stride,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object], *, default_ensemble: EnsembleKind) -> "DynamicsStageConfig":
        pressure = data.get("pressure")
        if pressure is None and default_ensemble == EnsembleKind.NPT:
            pressure = 1.0
        return cls(
            enabled=bool(data.get("enabled", True)),
            ensemble=EnsembleKind(str(data.get("ensemble", default_ensemble.value))),
            dt=float(data.get("dt", 0.002)),
            nsteps=int(data.get("nsteps", 1000)),
            temperature=float(data.get("temperature", 300.0)),
            pressure=float(pressure) if pressure is not None else None,
            friction_coefficient=float(data.get("friction_coefficient", 1.0)),
            trajectory_stride=int(data.get("trajectory_stride", 100)),
            energy_stride=int(data.get("energy_stride", 100)),
            checkpoint_stride=int(data.get("checkpoint_stride", 500)),
            eval_stride=int(data.get("eval_stride", 1)),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class QCloudConfig(ValidatableComponent):
    """Qcloud refinement controls."""

    enabled: bool = True
    trigger_threshold: float = 0.45
    max_regions: int = 2
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "trigger_threshold",
            coerce_scalar(self.trigger_threshold, "trigger_threshold"),
        )
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not 0.0 <= self.trigger_threshold <= 1.0:
            issues.append("hybrid.qcloud.trigger_threshold must lie in [0, 1].")
        if self.max_regions <= 0:
            issues.append("hybrid.qcloud.max_regions must be strictly positive.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "trigger_threshold": self.trigger_threshold,
            "max_regions": self.max_regions,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "QCloudConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            trigger_threshold=float(data.get("trigger_threshold", 0.45)),
            max_regions=int(data.get("max_regions", 2)),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class MLConfig(ValidatableComponent):
    """ML-residual controls."""

    enabled: bool = True
    model: str = "scalable_residual"
    online_training: bool = True
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        if not self.model.strip():
            return ("hybrid.ml.model must be a non-empty string.",)
        return ()

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "online_training": self.online_training,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MLConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            model=str(data.get("model", "scalable_residual")),
            online_training=bool(data.get("online_training", True)),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ControlConfig(ValidatableComponent):
    """Executive-control runtime settings."""

    enabled: bool = True
    mode: str = "conservative"
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        if not self.mode.strip():
            return ("hybrid.control.mode must be a non-empty string.",)
        return ()

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ControlConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            mode=str(data.get("mode", "conservative")),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class HybridConfig(ValidatableComponent):
    """Novel runtime feature toggles layered over the classical spine."""

    graph_enabled: bool = True
    shadow_enabled: bool = True
    memory_enabled: bool = True
    qcloud: QCloudConfig = field(default_factory=QCloudConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.qcloud, QCloudConfig):
            object.__setattr__(self, "qcloud", QCloudConfig.from_dict(self.qcloud))
        if not isinstance(self.ml, MLConfig):
            object.__setattr__(self, "ml", MLConfig.from_dict(self.ml))
        if not isinstance(self.control, ControlConfig):
            object.__setattr__(self, "control", ControlConfig.from_dict(self.control))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues = list(self.qcloud.validate())
        issues.extend(self.ml.validate())
        issues.extend(self.control.validate())
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "graph_enabled": self.graph_enabled,
            "shadow_enabled": self.shadow_enabled,
            "memory_enabled": self.memory_enabled,
            "qcloud": self.qcloud.to_dict(),
            "ml": self.ml.to_dict(),
            "control": self.control.to_dict(),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "HybridConfig":
        return cls(
            graph_enabled=bool(data.get("graph_enabled", True)),
            shadow_enabled=bool(data.get("shadow_enabled", True)),
            memory_enabled=bool(data.get("memory_enabled", True)),
            qcloud=QCloudConfig.from_dict(data.get("qcloud", {})),
            ml=MLConfig.from_dict(data.get("ml", {})),
            control=ControlConfig.from_dict(data.get("control", {})),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class OutputConfig(ValidatableComponent):
    """File outputs emitted by prepare/run/analyze commands."""

    output_dir: str = "outputs/default_run"
    prepared_bundle: str = "prepared_bundle.json"
    trajectory: str = "trajectory.jsonl"
    energy: str = "energies.csv"
    checkpoint: str = "checkpoint.json"
    run_summary: str = "run_summary.json"
    analysis_json: str = "analysis.json"
    analysis_html: str = "analysis.html"
    log: str = "run.log"
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for field_name in (
            "output_dir",
            "prepared_bundle",
            "trajectory",
            "energy",
            "checkpoint",
            "run_summary",
            "analysis_json",
            "analysis_html",
            "log",
        ):
            value = getattr(self, field_name)
            if not str(value).strip():
                issues.append(f"outputs.{field_name} must be a non-empty string.")
        return tuple(issues)

    def resolve(self, repo_root: str | Path, relative_path: str) -> Path:
        root = Path(repo_root).expanduser().resolve()
        return (root / self.output_dir / relative_path).resolve()

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": self.output_dir,
            "prepared_bundle": self.prepared_bundle,
            "trajectory": self.trajectory,
            "energy": self.energy,
            "checkpoint": self.checkpoint,
            "run_summary": self.run_summary,
            "analysis_json": self.analysis_json,
            "analysis_html": self.analysis_html,
            "log": self.log,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "OutputConfig":
        return cls(
            output_dir=str(data.get("output_dir", "outputs/default_run")),
            prepared_bundle=str(data.get("prepared_bundle", "prepared_bundle.json")),
            trajectory=str(data.get("trajectory", "trajectory.jsonl")),
            energy=str(data.get("energy", "energies.csv")),
            checkpoint=str(data.get("checkpoint", "checkpoint.json")),
            run_summary=str(data.get("run_summary", "run_summary.json")),
            analysis_json=str(data.get("analysis_json", "analysis.json")),
            analysis_html=str(data.get("analysis_html", "analysis.html")),
            log=str(data.get("log", "run.log")),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class AnalysisConfig(ValidatableComponent):
    """Observer-side analysis configuration."""

    benchmark_repeats: int = 1
    include_structure_report: bool = True
    include_fidelity_report: bool = True
    include_chemistry_report: bool = True
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        if self.benchmark_repeats < 0:
            return ("analysis.benchmark_repeats must be non-negative.",)
        return ()

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark_repeats": self.benchmark_repeats,
            "include_structure_report": self.include_structure_report,
            "include_fidelity_report": self.include_fidelity_report,
            "include_chemistry_report": self.include_chemistry_report,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "AnalysisConfig":
        return cls(
            benchmark_repeats=int(data.get("benchmark_repeats", 1)),
            include_structure_report=bool(data.get("include_structure_report", True)),
            include_fidelity_report=bool(data.get("include_fidelity_report", True)),
            include_chemistry_report=bool(data.get("include_chemistry_report", True)),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class RunManifest(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """[hybrid] User-facing prepare/run/analyze manifest."""

    system: SystemConfig
    prepare: PrepareConfig = field(default_factory=PrepareConfig)
    forcefield: ForcefieldConfig = field(default_factory=ForcefieldConfig)
    neighbor_list: NeighborListConfig = field(default_factory=NeighborListConfig)
    em: MinimizationStageConfig = field(default_factory=MinimizationStageConfig)
    nvt: DynamicsStageConfig = field(
        default_factory=lambda: DynamicsStageConfig(ensemble=EnsembleKind.NVT)
    )
    npt: DynamicsStageConfig = field(
        default_factory=lambda: DynamicsStageConfig(
            ensemble=EnsembleKind.NPT,
            pressure=1.0,
        )
    )
    production: DynamicsStageConfig = field(
        default_factory=lambda: DynamicsStageConfig(ensemble=EnsembleKind.NVT, nsteps=5000)
    )
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    source_path: str = ""
    classification: str = "[hybrid]"
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.system, SystemConfig):
            object.__setattr__(self, "system", SystemConfig.from_dict(self.system))
        if not isinstance(self.prepare, PrepareConfig):
            object.__setattr__(self, "prepare", PrepareConfig.from_dict(self.prepare))
        if not isinstance(self.forcefield, ForcefieldConfig):
            object.__setattr__(self, "forcefield", ForcefieldConfig.from_dict(self.forcefield))
        if not isinstance(self.neighbor_list, NeighborListConfig):
            object.__setattr__(self, "neighbor_list", NeighborListConfig.from_dict(self.neighbor_list))
        if not isinstance(self.em, MinimizationStageConfig):
            object.__setattr__(self, "em", MinimizationStageConfig.from_dict(self.em))
        if not isinstance(self.nvt, DynamicsStageConfig):
            object.__setattr__(
                self,
                "nvt",
                DynamicsStageConfig.from_dict(self.nvt, default_ensemble=EnsembleKind.NVT),
            )
        if not isinstance(self.npt, DynamicsStageConfig):
            object.__setattr__(
                self,
                "npt",
                DynamicsStageConfig.from_dict(self.npt, default_ensemble=EnsembleKind.NPT),
            )
        if not isinstance(self.production, DynamicsStageConfig):
            object.__setattr__(
                self,
                "production",
                DynamicsStageConfig.from_dict(self.production, default_ensemble=EnsembleKind.NVT),
            )
        if not isinstance(self.hybrid, HybridConfig):
            object.__setattr__(self, "hybrid", HybridConfig.from_dict(self.hybrid))
        if not isinstance(self.outputs, OutputConfig):
            object.__setattr__(self, "outputs", OutputConfig.from_dict(self.outputs))
        if not isinstance(self.analysis, AnalysisConfig):
            object.__setattr__(self, "analysis", AnalysisConfig.from_dict(self.analysis))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Stores the full user-facing simulation workflow, including protein import, "
            "preparation planning, staged dynamics, hybrid feature toggles, and outputs."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "config/protein_mapping.py",
            "prepare/pipeline.py",
            "scripts/neurocgmd.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/manifest_driven_md_workflow.md",)

    def validate(self) -> tuple[str, ...]:
        issues = list(self.system.validate())
        issues.extend(self.prepare.validate())
        issues.extend(self.forcefield.validate())
        issues.extend(self.neighbor_list.validate())
        issues.extend(self.em.validate())
        issues.extend(self.nvt.validate())
        issues.extend(self.npt.validate())
        issues.extend(self.production.validate())
        issues.extend(self.hybrid.validate())
        issues.extend(self.outputs.validate())
        issues.extend(self.analysis.validate())
        if self.source_path and not str(self.source_path).strip():
            issues.append("source_path must be empty or a non-empty string.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "system": self.system.to_dict(),
            "prepare": self.prepare.to_dict(),
            "forcefield": self.forcefield.to_dict(),
            "neighbor_list": self.neighbor_list.to_dict(),
            "stages": {
                "em": self.em.to_dict(),
                "nvt": self.nvt.to_dict(),
                "npt": self.npt.to_dict(),
                "production": self.production.to_dict(),
            },
            "hybrid": self.hybrid.to_dict(),
            "outputs": self.outputs.to_dict(),
            "analysis": self.analysis.to_dict(),
            "source_path": self.source_path,
            "classification": self.classification,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, object],
        *,
        source_path: str = "",
    ) -> "RunManifest":
        stages = data.get("stages", {})
        prepare_data = dict(data.get("prepare", {}))
        prepare_data.update(data.get("solvent", {}))
        prepare_data.update(data.get("ions", {}))
        forcefield_data = dict(data.get("forcefield", {}))
        forcefield_data.update(data.get("constraints", {}))
        neighbor_data = dict(data.get("neighbor_list", {}))
        neighbor_data.update(data.get("nonbonded", {}))
        return cls(
            system=SystemConfig.from_dict(data["system"]),
            prepare=PrepareConfig.from_dict(prepare_data),
            forcefield=ForcefieldConfig.from_dict(forcefield_data),
            neighbor_list=NeighborListConfig.from_dict(neighbor_data),
            em=MinimizationStageConfig.from_dict(stages.get("em", {})),
            nvt=DynamicsStageConfig.from_dict(stages.get("nvt", {}), default_ensemble=EnsembleKind.NVT),
            npt=DynamicsStageConfig.from_dict(stages.get("npt", {}), default_ensemble=EnsembleKind.NPT),
            production=DynamicsStageConfig.from_dict(
                stages.get("production", {}),
                default_ensemble=EnsembleKind.NVT,
            ),
            hybrid=HybridConfig.from_dict(data.get("hybrid", {})),
            outputs=OutputConfig.from_dict(data.get("outputs", {})),
            analysis=AnalysisConfig.from_dict(data.get("analysis", {})),
            source_path=source_path,
            classification=str(data.get("classification", "[hybrid]")),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


def load_run_manifest(path: str | Path) -> RunManifest:
    """Load and validate one run manifest from TOML."""

    resolved_path = Path(path).expanduser().resolve()
    payload = tomllib.loads(resolved_path.read_text(encoding="utf-8"))
    return RunManifest.from_dict(payload, source_path=str(resolved_path))
