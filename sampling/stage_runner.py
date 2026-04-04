"""Manifest-driven staged execution on top of the integrated production engine."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config import DynamicsStageConfig, MinimizationStageConfig, RunManifest
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import ParticleState, SimulationState, ThermodynamicState
from core.state_registry import LifecycleStage
from core.types import FrozenMetadata, StateId
from integrators.langevin import LangevinIntegrator

if TYPE_CHECKING:
    from sampling.production_engine import ProductionCycleReport


def _vector_norm(vector: tuple[float, float, float]) -> float:
    return sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])


@dataclass(frozen=True, slots=True)
class StageRecord(ValidatableComponent):
    """One completed workflow stage."""

    stage_name: str
    requested_steps: int
    executed_steps: int
    final_state_id: StateId
    final_step: int
    final_time: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.stage_name.strip():
            issues.append("stage_name must be a non-empty string.")
        if self.requested_steps < 0 or self.executed_steps < 0:
            issues.append("requested_steps and executed_steps must be non-negative.")
        if self.final_step < 0:
            issues.append("final_step must be non-negative.")
        if self.final_time < 0.0:
            issues.append("final_time must be non-negative.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_name": self.stage_name,
            "requested_steps": self.requested_steps,
            "executed_steps": self.executed_steps,
            "final_state_id": str(self.final_state_id),
            "final_step": self.final_step,
            "final_time": self.final_time,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ProductionRunSummary(ValidatableComponent):
    """Top-level result of a manifest-driven run."""

    system_name: str
    manifest_path: str
    prepared_bundle_path: str
    stage_records: tuple[StageRecord, ...]
    final_state: SimulationState
    final_cycle_metadata: FrozenMetadata
    output_artifacts: FrozenMetadata
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_records", tuple(self.stage_records))
        if not isinstance(self.final_cycle_metadata, FrozenMetadata):
            object.__setattr__(self, "final_cycle_metadata", FrozenMetadata(self.final_cycle_metadata))
        if not isinstance(self.output_artifacts, FrozenMetadata):
            object.__setattr__(self, "output_artifacts", FrozenMetadata(self.output_artifacts))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.system_name.strip():
            issues.append("system_name must be a non-empty string.")
        if not self.manifest_path.strip():
            issues.append("manifest_path must be a non-empty string.")
        if not self.prepared_bundle_path.strip():
            issues.append("prepared_bundle_path must be a non-empty string.")
        if not self.stage_records:
            issues.append("stage_records must contain at least one completed stage.")
        issues.extend(self.final_state.validate())
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "system_name": self.system_name,
            "manifest_path": self.manifest_path,
            "prepared_bundle_path": self.prepared_bundle_path,
            "stage_records": [record.to_dict() for record in self.stage_records],
            "final_state": self.final_state.to_dict(),
            "final_cycle_metadata": self.final_cycle_metadata.to_dict(),
            "output_artifacts": self.output_artifacts.to_dict(),
            "metadata": self.metadata.to_dict(),
        }


@dataclass(slots=True)
class ProductionStageRunner(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """[hybrid] Execute EM/NVT/NPT/production stages on the production engine."""

    name: str = "production_stage_runner"
    classification: str = "[hybrid]"

    def describe_role(self) -> str:
        return (
            "Runs user-facing staged MD workflows while keeping all force, qcloud, ML, "
            "graph, memory, and control logic inside the shared production engine."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "sampling/production_engine.py",
            "config/run_manifest.py",
            "io/trajectory_writer.py",
            "io/checkpoint_writer.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/manifest_driven_md_workflow.md",)

    def validate(self) -> tuple[str, ...]:
        return ()

    def run(
        self,
        *,
        context: Any,
        manifest: RunManifest,
        prepared_bundle_path: str,
        trajectory_writer: Any,
        checkpoint_writer: Any,
        energy_path: str | Path,
        benchmark_repeats: int = 0,
    ) -> ProductionRunSummary:
        """Execute a staged run and emit trajectory, energy, and checkpoint artifacts."""

        energy_destination = Path(energy_path).expanduser().resolve()
        energy_destination.parent.mkdir(parents=True, exist_ok=True)
        stage_records: list[StageRecord] = []
        self._write_energy_header(energy_destination)

        initial_state = context.registry.latest_state()
        trajectory_writer.append_state(
            initial_state,
            metadata={"stage": "initial", "stage_step": 0},
        )
        checkpoint_writer.write(
            initial_state,
            metadata={"stage": "initial", "system_name": manifest.system.name},
        )
        self._append_energy_sample(energy_destination, initial_state, stage_name="initial", stage_step=0)

        if manifest.em.enabled and manifest.em.max_steps > 0:
            stage_records.append(
                self._run_minimization_stage(
                    context=context,
                    config=manifest.em,
                    trajectory_writer=trajectory_writer,
                    checkpoint_writer=checkpoint_writer,
                    energy_path=energy_destination,
                )
            )
        for stage_name, config in (
            ("nvt", manifest.nvt),
            ("npt", manifest.npt),
            ("production", manifest.production),
        ):
            if config.enabled and config.nsteps > 0:
                stage_records.append(
                    self._run_dynamics_stage(
                        context=context,
                        stage_name=stage_name,
                        config=config,
                        trajectory_writer=trajectory_writer,
                        checkpoint_writer=checkpoint_writer,
                        energy_path=energy_destination,
                    )
                )

        final_cycle = context.engine.collect_cycle(benchmark_repeats=benchmark_repeats)
        final_state = context.registry.latest_state()
        checkpoint_writer.write(
            final_state,
            metadata={
                "stage": "final",
                "system_name": manifest.system.name,
                "final_action": final_cycle.final_decision.highest_priority_action().kind.value,
            },
        )
        return ProductionRunSummary(
            system_name=manifest.system.name,
            manifest_path=manifest.source_path,
            prepared_bundle_path=prepared_bundle_path,
            stage_records=tuple(stage_records),
            final_state=final_state,
            final_cycle_metadata=self._cycle_metadata(final_cycle),
            output_artifacts=FrozenMetadata(
                {
                    "trajectory_path": str(Path(trajectory_writer.path).expanduser().resolve()),
                    "energy_path": str(energy_destination),
                    "checkpoint_path": str(Path(checkpoint_writer.path).expanduser().resolve()),
                }
            ),
            metadata={
                "analysis_benchmark_repeats": benchmark_repeats,
                "stage_count": len(stage_records),
                "qcloud_event_analysis": context.engine.qcloud_event_analyzer.summary(),
                "qcloud_detected_events": [
                    {
                        "kind": e.kind.value,
                        "step": e.step,
                        "time": e.time,
                        "particles": list(e.particle_indices),
                        "correction_magnitude": e.correction_magnitude,
                        "baseline_magnitude": e.baseline_magnitude,
                        "confidence": e.confidence,
                    }
                    for e in context.engine.qcloud_event_analyzer.detected_events()
                ],
            },
        )

    def _run_minimization_stage(
        self,
        *,
        context: Any,
        config: MinimizationStageConfig,
        trajectory_writer: Any,
        checkpoint_writer: Any,
        energy_path: Path,
    ) -> StageRecord:
        state = context.registry.latest_state()
        executed_steps = 0
        sample_stride = max(1, min(25, config.max_steps))
        for local_step in range(1, config.max_steps + 1):
            evaluation = context.engine.preview_force_evaluation(state)
            max_force = max(_vector_norm(force) for force in evaluation.forces)
            self._append_energy_sample(energy_path, state, stage_name="em", stage_step=local_step)
            if max_force <= config.tolerance:
                break
            positions = tuple(
                (
                    position[0] + config.step_size * force[0] / mass,
                    position[1] + config.step_size * force[1] / mass,
                    position[2] + config.step_size * force[2] / mass,
                )
                for position, force, mass in zip(
                    state.particles.positions,
                    evaluation.forces,
                    state.particles.masses,
                    strict=True,
                )
            )
            zero_velocities = tuple((0.0, 0.0, 0.0) for _ in positions)
            state = context.registry.derive_state(
                state,
                particles=ParticleState(
                    positions=positions,
                    masses=state.particles.masses,
                    velocities=zero_velocities,
                    forces=evaluation.forces,
                    labels=state.particles.labels,
                ),
                potential_energy=evaluation.potential_energy,
                stage=LifecycleStage.CORRECTION,
                notes="steepest-descent minimization step",
                metadata={
                    "stage": "em",
                    "max_force": max_force,
                    "algorithm": config.algorithm,
                },
                created_by=self.name,
            )
            executed_steps = local_step
            if local_step % sample_stride == 0 or local_step == config.max_steps:
                trajectory_writer.append_state(
                    state,
                    metadata={"stage": "em", "stage_step": local_step},
                )
                checkpoint_writer.write(
                    state,
                    metadata={"stage": "em", "stage_step": local_step},
                )
        final_state = context.registry.latest_state()
        return StageRecord(
            stage_name="em",
            requested_steps=config.max_steps,
            executed_steps=executed_steps,
            final_state_id=final_state.provenance.state_id,
            final_step=final_state.step,
            final_time=final_state.time,
            metadata={"tolerance": config.tolerance},
        )

    def _run_dynamics_stage(
        self,
        *,
        context: Any,
        stage_name: str,
        config: DynamicsStageConfig,
        trajectory_writer: Any,
        checkpoint_writer: Any,
        energy_path: Path,
    ) -> StageRecord:
        self._configure_runtime_stage(context, stage_name=stage_name, config=config)
        eval_stride = config.eval_stride
        executed_steps = 0
        for local_step in range(1, config.nsteps + 1):
            is_eval_step = (eval_stride <= 1) or (local_step % eval_stride == 0) or (local_step == config.nsteps)
            context.engine.advance(steps=1, record_final_state=False, full_eval=is_eval_step)
            state = context.registry.latest_state()
            executed_steps = local_step
            if local_step % config.trajectory_stride == 0 or local_step == config.nsteps:
                trajectory_writer.append_state(
                    state,
                    metadata={"stage": stage_name, "stage_step": local_step},
                )
            if local_step % config.energy_stride == 0 or local_step == config.nsteps:
                self._append_energy_sample(energy_path, state, stage_name=stage_name, stage_step=local_step)
            if local_step % config.checkpoint_stride == 0 or local_step == config.nsteps:
                checkpoint_writer.write(
                    state,
                    metadata={"stage": stage_name, "stage_step": local_step},
                )
        final_state = context.registry.latest_state()
        return StageRecord(
            stage_name=stage_name,
            requested_steps=config.nsteps,
            executed_steps=executed_steps,
            final_state_id=final_state.provenance.state_id,
            final_step=final_state.step,
            final_time=final_state.time,
            metadata={
                "ensemble": config.ensemble.value,
                "dt": config.dt,
                "temperature": config.temperature,
                "pressure": config.pressure,
            },
        )

    def _configure_runtime_stage(self, context: Any, *, stage_name: str, config: DynamicsStageConfig) -> None:
        integrator = LangevinIntegrator(
            time_step=config.dt,
            friction_coefficient=config.friction_coefficient,
        )
        context.loop.integrator = integrator
        context.engine.integrator = integrator
        latest_state = context.registry.latest_state()
        context.registry.derive_state(
            latest_state,
            thermodynamics=ThermodynamicState(
                ensemble=config.ensemble,
                target_temperature=config.temperature,
                target_pressure=config.pressure,
                friction_coefficient=config.friction_coefficient,
            ),
            step=latest_state.step,
            time=latest_state.time,
            stage=LifecycleStage.CHECKPOINT,
            notes=f"{stage_name} stage configuration",
            metadata={"stage": stage_name, "dt": config.dt},
            created_by=self.name,
        )

    def _write_energy_header(self, path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "stage",
                    "stage_step",
                    "state_id",
                    "simulation_step",
                    "time",
                    "potential_energy",
                    "kinetic_energy",
                    "total_energy",
                ]
            )

    def _append_energy_sample(
        self,
        path: Path,
        state: SimulationState,
        *,
        stage_name: str,
        stage_step: int,
    ) -> None:
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    stage_name,
                    stage_step,
                    str(state.provenance.state_id),
                    state.step,
                    state.time,
                    state.potential_energy,
                    state.kinetic_energy(),
                    state.total_energy(),
                ]
            )

    def _cycle_metadata(self, cycle: "ProductionCycleReport") -> FrozenMetadata:
        return FrozenMetadata(
            {
                "state_id": str(cycle.state_id),
                "assembly_score": cycle.progress.assembly_score,
                "stage_label": cycle.progress.stage_label,
                "final_action": cycle.final_decision.highest_priority_action().kind.value,
                "qcloud_applied": cycle.metadata.get("qcloud_applied", False),
                "selected_region_count": cycle.metadata.get("selected_region_count", 0),
                "trace_record_count": cycle.metadata.get("trace_record_count", 0),
            }
        )
