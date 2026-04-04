"""Trajectory-, controller-, and dashboard-oriented diagnostic view models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape

from ai_control.controller import ControllerDecision
from benchmarks.baseline_suite import BenchmarkReport
from compartments.registry import CompartmentRegistry
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata, StateId, Vector3, coerce_scalar, coerce_vector3
from memory.trace_store import TraceRecord
from ml.uncertainty_model import UncertaintyEstimate
from qcloud.qcloud_coupling import QCloudCouplingResult
from validation.drift_checks import DriftCheckReport
from validation.sanity_checks import SanityCheckReport
from visualization.graph_views import GraphSnapshotView
from visualization.problem_views import ProblemStatementView


def _normalize_identifiers(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_value in values or ():
        value = str(raw_value).strip()
        if not value:
            continue
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class ParticleView(ValidatableComponent):
    """Serializable particle-level diagnostic view."""

    particle_index: int
    label: str
    position: Vector3
    velocity: Vector3
    force: Vector3
    compartments: tuple[str, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", coerce_vector3(self.position, "position"))
        object.__setattr__(self, "velocity", coerce_vector3(self.velocity, "velocity"))
        object.__setattr__(self, "force", coerce_vector3(self.force, "force"))
        object.__setattr__(self, "compartments", _normalize_identifiers(self.compartments))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.particle_index < 0:
            issues.append("particle_index must be non-negative.")
        if not self.label.strip():
            issues.append("label must be a non-empty string.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "particle_index": self.particle_index,
            "label": self.label,
            "position": list(self.position),
            "velocity": list(self.velocity),
            "force": list(self.force),
            "compartments": list(self.compartments),
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ControllerActionView(ValidatableComponent):
    """Serializable action recommendation view."""

    kind: str
    priority: int
    summary: str
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.kind.strip():
            issues.append("kind must be a non-empty string.")
        if self.priority < 0:
            issues.append("priority must be non-negative.")
        if not self.summary.strip():
            issues.append("summary must be a non-empty string.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "priority": self.priority,
            "summary": self.summary,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ValidationSummaryView(ValidatableComponent):
    """Serializable validation summary for dashboard consumers."""

    sanity_passed: bool | None = None
    sanity_failed_checks: tuple[str, ...] = ()
    drift_passed: bool | None = None
    drift_violations: tuple[str, ...] = ()
    max_energy_drift: float | None = None
    max_position_displacement: float | None = None
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sanity_failed_checks", _normalize_identifiers(self.sanity_failed_checks))
        object.__setattr__(self, "drift_violations", tuple(str(value) for value in self.drift_violations))
        if self.max_energy_drift is not None:
            object.__setattr__(self, "max_energy_drift", coerce_scalar(self.max_energy_drift, "max_energy_drift"))
        if self.max_position_displacement is not None:
            object.__setattr__(
                self,
                "max_position_displacement",
                coerce_scalar(self.max_position_displacement, "max_position_displacement"),
            )
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        return ()

    def to_dict(self) -> dict[str, object]:
        return {
            "sanity_passed": self.sanity_passed,
            "sanity_failed_checks": list(self.sanity_failed_checks),
            "drift_passed": self.drift_passed,
            "drift_violations": list(self.drift_violations),
            "max_energy_drift": self.max_energy_drift,
            "max_position_displacement": self.max_position_displacement,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_reports(
        cls,
        *,
        sanity_report: SanityCheckReport | None = None,
        drift_report: DriftCheckReport | None = None,
    ) -> "ValidationSummaryView":
        return cls(
            sanity_passed=sanity_report.passed() if sanity_report is not None else None,
            sanity_failed_checks=(
                tuple(check.name for check in sanity_report.failed_checks()) if sanity_report is not None else ()
            ),
            drift_passed=drift_report.passed() if drift_report is not None else None,
            drift_violations=drift_report.violations if drift_report is not None else (),
            max_energy_drift=drift_report.max_energy_drift if drift_report is not None else None,
            max_position_displacement=(
                drift_report.max_position_displacement if drift_report is not None else None
            ),
            metadata=FrozenMetadata(
                {
                    "has_sanity_report": sanity_report is not None,
                    "has_drift_report": drift_report is not None,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkCaseView(ValidatableComponent):
    """Serializable benchmark-case summary."""

    name: str
    iterations: int
    elapsed_seconds: float
    average_seconds_per_iteration: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "elapsed_seconds", coerce_scalar(self.elapsed_seconds, "elapsed_seconds"))
        object.__setattr__(
            self,
            "average_seconds_per_iteration",
            coerce_scalar(self.average_seconds_per_iteration, "average_seconds_per_iteration"),
        )
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name.strip():
            issues.append("name must be a non-empty string.")
        if self.iterations <= 0:
            issues.append("iterations must be strictly positive.")
        if self.elapsed_seconds < 0.0 or self.average_seconds_per_iteration < 0.0:
            issues.append("benchmark timings must be non-negative.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "elapsed_seconds": self.elapsed_seconds,
            "average_seconds_per_iteration": self.average_seconds_per_iteration,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkSummaryView(ValidatableComponent):
    """Serializable benchmark summary view."""

    total_elapsed_seconds: float = 0.0
    cases: tuple[BenchmarkCaseView, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "total_elapsed_seconds", coerce_scalar(self.total_elapsed_seconds, "total_elapsed_seconds"))
        object.__setattr__(self, "cases", tuple(self.cases))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        if self.total_elapsed_seconds < 0.0:
            return ("total_elapsed_seconds must be non-negative.",)
        return ()

    def to_dict(self) -> dict[str, object]:
        return {
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "cases": [case.to_dict() for case in self.cases],
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_report(cls, report: BenchmarkReport | None) -> "BenchmarkSummaryView | None":
        if report is None:
            return None
        return cls(
            total_elapsed_seconds=report.total_elapsed_seconds(),
            cases=tuple(
                BenchmarkCaseView(
                    name=case.name,
                    iterations=case.iterations,
                    elapsed_seconds=case.elapsed_seconds,
                    average_seconds_per_iteration=case.average_seconds_per_iteration(),
                    metadata=case.metadata,
                )
                for case in report.cases
            ),
            metadata=report.metadata,
        )


@dataclass(frozen=True, slots=True)
class TrajectoryFrameView(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Serializable state-aligned trajectory frame summary."""

    name: str = "trajectory_frame_view"
    classification: str = "[adapted]"
    simulation_id: str = ""
    state_id: StateId = StateId("")
    step: int = 0
    time: float = 0.0
    particle_count: int = 0
    kinetic_energy: float = 0.0
    potential_energy: float | None = None
    total_energy: float | None = None
    units: FrozenMetadata = field(default_factory=FrozenMetadata)
    particles: tuple[ParticleView, ...] = ()
    controller_actions: tuple[ControllerActionView, ...] = ()
    trace_tags: tuple[str, ...] = ()
    uncertainty_total: float | None = None
    qcloud_region_count: int = 0
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", StateId(str(self.state_id)))
        object.__setattr__(self, "time", coerce_scalar(self.time, "time"))
        object.__setattr__(self, "kinetic_energy", coerce_scalar(self.kinetic_energy, "kinetic_energy"))
        if self.potential_energy is not None:
            object.__setattr__(self, "potential_energy", coerce_scalar(self.potential_energy, "potential_energy"))
        if self.total_energy is not None:
            object.__setattr__(self, "total_energy", coerce_scalar(self.total_energy, "total_energy"))
        if self.uncertainty_total is not None:
            object.__setattr__(self, "uncertainty_total", coerce_scalar(self.uncertainty_total, "uncertainty_total"))
        object.__setattr__(self, "particles", tuple(self.particles))
        object.__setattr__(self, "controller_actions", tuple(self.controller_actions))
        object.__setattr__(self, "trace_tags", _normalize_identifiers(self.trace_tags))
        if not isinstance(self.units, FrozenMetadata):
            object.__setattr__(self, "units", FrozenMetadata(self.units))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Adapts the current simulation state and attached diagnostics into a "
            "serializable trajectory frame for local visualization."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "core/state.py",
            "ai_control/controller.py",
            "validation/sanity_checks.py",
            "validation/drift_checks.py",
            "benchmarks/baseline_suite.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/visualization_and_diagnostics.md",
            "docs/sections/section_14_visualization_and_diagnostics.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.simulation_id.strip():
            issues.append("simulation_id must be a non-empty string.")
        if not str(self.state_id).strip():
            issues.append("state_id must be a non-empty string.")
        if self.step < 0:
            issues.append("step must be non-negative.")
        if self.time < 0.0:
            issues.append("time must be non-negative.")
        if self.particle_count <= 0:
            issues.append("particle_count must be positive.")
        if len(self.particles) != self.particle_count:
            issues.append("particles length must match particle_count.")
        if self.qcloud_region_count < 0:
            issues.append("qcloud_region_count must be non-negative.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "classification": self.classification,
            "simulation_id": self.simulation_id,
            "state_id": str(self.state_id),
            "step": self.step,
            "time": self.time,
            "particle_count": self.particle_count,
            "kinetic_energy": self.kinetic_energy,
            "potential_energy": self.potential_energy,
            "total_energy": self.total_energy,
            "units": self.units.to_dict(),
            "particles": [particle.to_dict() for particle in self.particles],
            "controller_actions": [action.to_dict() for action in self.controller_actions],
            "trace_tags": list(self.trace_tags),
            "uncertainty_total": self.uncertainty_total,
            "qcloud_region_count": self.qcloud_region_count,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_state(
        cls,
        state: SimulationState,
        *,
        compartments: CompartmentRegistry | None = None,
        trace_record: TraceRecord | None = None,
        controller_decision: ControllerDecision | None = None,
        uncertainty_estimate: UncertaintyEstimate | None = None,
        qcloud_result: QCloudCouplingResult | None = None,
    ) -> "TrajectoryFrameView":
        if compartments is not None and compartments.particle_count != state.particle_count:
            raise ContractValidationError("compartments.particle_count must match the SimulationState particle count.")
        if trace_record is not None and trace_record.state_id != state.provenance.state_id:
            raise ContractValidationError("trace_record.state_id must match the SimulationState state_id.")
        if controller_decision is not None and controller_decision.state_id != state.provenance.state_id:
            raise ContractValidationError("controller_decision.state_id must match the SimulationState state_id.")
        if uncertainty_estimate is not None and uncertainty_estimate.state_id != state.provenance.state_id:
            raise ContractValidationError("uncertainty_estimate.state_id must match the SimulationState state_id.")

        membership_map = compartments.membership_map() if compartments is not None else {}
        labels = state.particles.labels or tuple(f"P{index}" for index in range(state.particle_count))
        particles = tuple(
            ParticleView(
                particle_index=index,
                label=labels[index],
                position=state.particles.positions[index],
                velocity=state.particles.velocities[index],
                force=state.particles.forces[index],
                compartments=tuple(str(identifier) for identifier in membership_map.get(index, ())),
            )
            for index in range(state.particle_count)
        )
        actions = tuple(
            ControllerActionView(
                kind=action.kind.value,
                priority=action.priority,
                summary=action.summary,
                metadata=action.metadata,
            )
            for action in (controller_decision.actions if controller_decision is not None else ())
        )
        return cls(
            simulation_id=str(state.provenance.simulation_id),
            state_id=state.provenance.state_id,
            step=state.step,
            time=state.time,
            particle_count=state.particle_count,
            kinetic_energy=state.kinetic_energy(),
            potential_energy=state.potential_energy,
            total_energy=state.total_energy(),
            units=FrozenMetadata(state.units.to_dict()),
            particles=particles,
            controller_actions=actions,
            trace_tags=trace_record.tags if trace_record is not None else (),
            uncertainty_total=(
                uncertainty_estimate.total_uncertainty if uncertainty_estimate is not None else None
            ),
            qcloud_region_count=len(qcloud_result.selected_regions) if qcloud_result is not None else 0,
            metadata=FrozenMetadata(
                {
                    "stage": state.provenance.stage,
                    "controller_action_count": len(actions),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class DashboardSnapshotView(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Composite dashboard snapshot rendered to JSON and HTML."""

    name: str = "dashboard_snapshot_view"
    classification: str = "[adapted]"
    title: str = "NeuroCGMD Dashboard"
    generated_at: str = ""
    problem: ProblemStatementView | None = None
    trajectory: TrajectoryFrameView | None = None
    graph: GraphSnapshotView | None = None
    validation: ValidationSummaryView | None = None
    benchmarks: BenchmarkSummaryView | None = None
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not self.generated_at:
            object.__setattr__(self, "generated_at", datetime.now(timezone.utc).isoformat())
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Combines trajectory, graph, validation, and benchmark summaries into a "
            "single serializable local dashboard snapshot."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "visualization/trajectory_views.py",
            "visualization/graph_views.py",
            "validation/sanity_checks.py",
            "validation/drift_checks.py",
            "benchmarks/baseline_suite.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/visualization_and_diagnostics.md",
            "docs/sections/section_14_visualization_and_diagnostics.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.title.strip():
            issues.append("title must be a non-empty string.")
        if self.trajectory is None:
            issues.append("trajectory must be provided.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "classification": self.classification,
            "title": self.title,
            "generated_at": self.generated_at,
            "problem": self.problem.to_dict() if self.problem is not None else None,
            "trajectory": self.trajectory.to_dict() if self.trajectory is not None else None,
            "graph": self.graph.to_dict() if self.graph is not None else None,
            "validation": self.validation.to_dict() if self.validation is not None else None,
            "benchmarks": self.benchmarks.to_dict() if self.benchmarks is not None else None,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_components(
        cls,
        *,
        title: str,
        problem: ProblemStatementView | None = None,
        state: SimulationState,
        graph: GraphSnapshotView | None = None,
        compartments: CompartmentRegistry | None = None,
        trace_record: TraceRecord | None = None,
        controller_decision: ControllerDecision | None = None,
        uncertainty_estimate: UncertaintyEstimate | None = None,
        qcloud_result: QCloudCouplingResult | None = None,
        sanity_report: SanityCheckReport | None = None,
        drift_report: DriftCheckReport | None = None,
        benchmark_report: BenchmarkReport | None = None,
        metadata: FrozenMetadata | dict[str, object] | None = None,
    ) -> "DashboardSnapshotView":
        trajectory = TrajectoryFrameView.from_state(
            state,
            compartments=compartments,
            trace_record=trace_record,
            controller_decision=controller_decision,
            uncertainty_estimate=uncertainty_estimate,
            qcloud_result=qcloud_result,
        )
        return cls(
            title=title,
            problem=problem,
            trajectory=trajectory,
            graph=graph,
            validation=ValidationSummaryView.from_reports(
                sanity_report=sanity_report,
                drift_report=drift_report,
            ),
            benchmarks=BenchmarkSummaryView.from_report(benchmark_report),
            metadata=metadata if isinstance(metadata, FrozenMetadata) else FrozenMetadata(metadata),
        )

    def render_html(self, *, json_endpoint: str = "dashboard.json", refresh_ms: int = 1000) -> str:
        title = escape(self.title)
        endpoint = escape(json_endpoint)
        refresh_ms = max(250, int(refresh_ms))
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: #fffaf3;
      --ink: #18222d;
      --muted: #6c7a86;
      --accent: #c75c2a;
      --accent-soft: #f1c9b1;
      --ok: #1f7a4c;
      --warn: #b45d00;
      --bad: #a62323;
      --line: #e0d4c6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff8ef 0%, transparent 35%),
        linear-gradient(180deg, #efe6d9 0%, var(--bg) 100%);
    }}
    .shell {{ padding: 24px; max-width: 1400px; margin: 0 auto; }}
    .hero {{
      background: linear-gradient(135deg, rgba(199,92,42,0.14), rgba(24,34,45,0.08));
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 24px;
      margin-bottom: 20px;
      box-shadow: 0 14px 40px rgba(24,34,45,0.08);
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 2rem; }}
    .hero p {{ margin: 0; color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(24,34,45,0.05);
    }}
    .panel h2 {{ margin: 0 0 14px; font-size: 1rem; letter-spacing: 0.03em; text-transform: uppercase; }}
    .metrics {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .metric {{
      padding: 12px;
      border-radius: 14px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(224,212,198,0.85);
    }}
    .metric strong {{ display: block; font-size: 1.35rem; }}
    .metric span {{ color: var(--muted); font-size: 0.85rem; }}
    .status {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }}
    .status.ok {{ color: white; background: var(--ok); }}
    .status.warn {{ color: white; background: var(--warn); }}
    .status.bad {{ color: white; background: var(--bad); }}
    .status.active {{ color: white; background: var(--accent); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    th, td {{
      text-align: left;
      padding: 9px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 700; }}
    .split {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 16px;
    }}
    .tiny {{ color: var(--muted); font-size: 0.82rem; }}
    .tag {{
      display: inline-block;
      margin: 0 6px 6px 0;
      padding: 5px 9px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #6e2d10;
      font-size: 0.78rem;
      font-weight: 700;
    }}
    svg {{
      width: 100%;
      min-height: 260px;
      border-radius: 16px;
      background: linear-gradient(180deg, #fffdf9 0%, #f7efe3 100%);
      border: 1px solid var(--line);
    }}
    @media (max-width: 900px) {{
      .split {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>{title}</h1>
      <p id="hero-summary">Waiting for problem summary...</p>
      <p class="tiny" id="timestamp">Waiting for snapshot...</p>
    </section>

    <div class="grid">
      <section class="panel">
        <h2>Problem</h2>
        <div id="problem-summary"></div>
      </section>
      <section class="panel">
        <h2>Simulation</h2>
        <div class="metrics" id="simulation-metrics"></div>
      </section>
      <section class="panel">
        <h2>Validation</h2>
        <div id="validation-summary"></div>
      </section>
      <section class="panel">
        <h2>Benchmarks</h2>
        <div id="benchmark-summary"></div>
      </section>
      <section class="panel">
        <h2>Tags</h2>
        <div id="trace-tags"></div>
      </section>
    </div>

    <div class="split">
      <section class="panel">
        <h2>Graph View</h2>
        <svg id="graph-canvas" viewBox="0 0 640 320" preserveAspectRatio="xMidYMid meet"></svg>
        <p class="tiny" id="graph-meta"></p>
      </section>
      <section class="panel">
        <h2>Controller Actions</h2>
        <table>
          <thead><tr><th>Kind</th><th>Priority</th><th>Summary</th></tr></thead>
          <tbody id="controller-actions"></tbody>
        </table>
      </section>
    </div>

    <div class="split" style="margin-top:16px;">
      <section class="panel">
        <h2>Particles</h2>
        <table>
          <thead><tr><th>Particle</th><th>Position</th><th>Velocity</th><th>Force</th><th>Compartments</th></tr></thead>
          <tbody id="particle-table"></tbody>
        </table>
      </section>
      <section class="panel">
        <h2>Graph Edges</h2>
        <table>
          <thead><tr><th>Pair</th><th>Kind</th><th>Weight</th><th>Distance</th><th>Route</th></tr></thead>
          <tbody id="edge-table"></tbody>
        </table>
      </section>
    </div>
  </div>

  <script>
    const jsonEndpoint = {endpoint!r};
    const refreshMs = {refresh_ms};

    function statusChip(kind, label) {{
      return `<span class="status ${{kind}}">${{label}}</span>`;
    }}

    function formatVector(vector) {{
      return `[${{vector.map((value) => Number(value).toFixed(2)).join(", ")}}]`;
    }}

    function renderMetrics(trajectory) {{
      const metrics = [
        ["Step", trajectory.step],
        ["Time", Number(trajectory.time).toFixed(3)],
        ["Particles", trajectory.particle_count],
        ["Kinetic", Number(trajectory.kinetic_energy).toFixed(3)],
        ["Potential", trajectory.potential_energy === null ? "n/a" : Number(trajectory.potential_energy).toFixed(3)],
        ["Total", trajectory.total_energy === null ? "n/a" : Number(trajectory.total_energy).toFixed(3)],
        ["Uncertainty", trajectory.uncertainty_total === null ? "n/a" : Number(trajectory.uncertainty_total).toFixed(3)],
        ["QCloud Regions", trajectory.qcloud_region_count],
      ];
      document.getElementById("simulation-metrics").innerHTML = metrics.map(([label, value]) =>
        `<div class="metric"><strong>${{value}}</strong><span>${{label}}</span></div>`
      ).join("");
    }}

    function renderProblem(problem) {{
      const heroSummary = document.getElementById("hero-summary");
      if (!problem) {{
        heroSummary.textContent = "Auto-refreshing local dashboard for trajectory, graph, validation, and benchmark diagnostics.";
        document.getElementById("problem-summary").innerHTML = '<p class="tiny">No concrete problem statement attached.</p>';
        return;
      }}
      heroSummary.textContent = problem.summary;
      const bound = Boolean(problem.metadata && problem.metadata.bound);
      const stageKind = bound ? "ok" : "active";
      const metrics = (problem.metrics || []).map((metric) => `
        <div class="metric">
          <strong>${{metric.value}}</strong>
          <span>${{metric.label}}</span>
          ${{metric.detail ? `<div class="tiny" style="margin-top:8px;">${{metric.detail}}</div>` : ""}}
        </div>
      `).join("");
      const referenceMetrics = (problem.reference_metrics || []).map((metric) => `
        <div class="metric">
          <strong>${{metric.value}}</strong>
          <span>${{metric.label}}</span>
          ${{metric.detail ? `<div class="tiny" style="margin-top:8px;">${{metric.detail}}</div>` : ""}}
        </div>
      `).join("");
      const structureMetrics = (problem.structure_metrics || []).map((metric) => `
        <div class="metric">
          <strong>${{metric.value}}</strong>
          <span>${{metric.label}}</span>
          ${{metric.detail ? `<div class="tiny" style="margin-top:8px;">${{metric.detail}}</div>` : ""}}
        </div>
      `).join("");
      const fidelityMetrics = (problem.fidelity_metrics || []).map((metric) => `
        <div class="metric">
          <strong>${{metric.value}}</strong>
          <span>${{metric.label}}</span>
          ${{metric.detail ? `<div class="tiny" style="margin-top:8px;">${{metric.detail}}</div>` : ""}}
        </div>
      `).join("");
      const chemistryMetrics = (problem.chemistry_metrics || []).map((metric) => `
        <div class="metric">
          <strong>${{metric.value}}</strong>
          <span>${{metric.label}}</span>
          ${{metric.detail ? `<div class="tiny" style="margin-top:8px;">${{metric.detail}}</div>` : ""}}
        </div>
      `).join("");
      const referenceBlock = referenceMetrics
        ? `
          <div style="margin-top:14px; padding-top:14px; border-top:1px solid var(--line);">
            <div style="margin-bottom:8px;">${{statusChip("warn", problem.reference_title || "Reference")}}</div>
            ${{problem.reference_summary ? `<div class="tiny"><strong>Known answer set:</strong> ${{problem.reference_summary}}</div>` : ""}}
            <div class="metrics" style="margin-top:12px;">${{referenceMetrics}}</div>
          </div>
        `
        : "";
      const structureBlock = structureMetrics
        ? `
          <div style="margin-top:14px; padding-top:14px; border-top:1px solid var(--line);">
            <div style="margin-bottom:8px;">${{statusChip("active", problem.structure_title || "Atomistic Alignment")}}</div>
            ${{problem.structure_summary ? `<div class="tiny"><strong>Geometry check:</strong> ${{problem.structure_summary}}</div>` : ""}}
            <div class="metrics" style="margin-top:12px;">${{structureMetrics}}</div>
          </div>
        `
        : "";
      const chemistryBlock = chemistryMetrics
        ? `
          <div style="margin-top:14px; padding-top:14px; border-top:1px solid var(--line);">
            <div style="margin-bottom:8px;">${{statusChip("active", problem.chemistry_title || "Interface Chemistry")}}</div>
            ${{problem.chemistry_summary ? `<div class="tiny"><strong>Chemistry check:</strong> ${{problem.chemistry_summary}}</div>` : ""}}
            <div class="metrics" style="margin-top:12px;">${{chemistryMetrics}}</div>
          </div>
        `
        : "";
      const fidelityBlock = fidelityMetrics
        ? `
          <div style="margin-top:14px; padding-top:14px; border-top:1px solid var(--line);">
            <div style="margin-bottom:8px;">${{statusChip("active", problem.fidelity_title || "Shadow Fidelity")}}</div>
            ${{problem.fidelity_summary ? `<div class="tiny"><strong>Shadow delta:</strong> ${{problem.fidelity_summary}}</div>` : ""}}
            <div class="metrics" style="margin-top:12px;">${{fidelityMetrics}}</div>
          </div>
        `
        : "";
      document.getElementById("problem-summary").innerHTML = `
        <div style="margin-bottom:10px;">${{statusChip(stageKind, problem.stage)}}</div>
        <div class="tiny"><strong>Objective:</strong> ${{problem.objective}}</div>
        <div class="metrics" style="margin-top:12px;">${{metrics || '<span class="tiny">No objective metrics attached.</span>'}}</div>
        ${{referenceBlock}}
        ${{structureBlock}}
        ${{chemistryBlock}}
        ${{fidelityBlock}}
      `;
    }}

    function renderValidation(validation) {{
      if (!validation) {{
        document.getElementById("validation-summary").innerHTML = '<p class="tiny">No validation report attached.</p>';
        return;
      }}
      const sanity = validation.sanity_passed === null
        ? '<p class="tiny">No sanity report attached.</p>'
        : `${{statusChip(validation.sanity_passed ? "ok" : "bad", validation.sanity_passed ? "Sanity Pass" : "Sanity Fail")}}`;
      const drift = validation.drift_passed === null
        ? '<p class="tiny">No drift report attached.</p>'
        : `${{statusChip(validation.drift_passed ? "ok" : "warn", validation.drift_passed ? "Drift Pass" : "Drift Alert")}}`;
      const failed = (validation.sanity_failed_checks || []).map((item) => `<span class="tag">${{item}}</span>`).join("");
      const violations = (validation.drift_violations || []).map((item) => `<div class="tiny">${{item}}</div>`).join("");
      document.getElementById("validation-summary").innerHTML = `
        <div style="margin-bottom:10px;">${{sanity}} ${{drift}}</div>
        <div class="tiny">Max energy drift: ${{validation.max_energy_drift === null ? "n/a" : Number(validation.max_energy_drift).toFixed(3)}}</div>
        <div class="tiny">Max position displacement: ${{validation.max_position_displacement === null ? "n/a" : Number(validation.max_position_displacement).toFixed(3)}}</div>
        <div style="margin-top:10px;">${{failed || violations || '<span class="tiny">No active validation findings.</span>'}}</div>
      `;
    }}

    function renderBenchmarks(benchmarks) {{
      if (!benchmarks) {{
        document.getElementById("benchmark-summary").innerHTML = '<p class="tiny">No benchmark report attached.</p>';
        return;
      }}
      const cases = benchmarks.cases.map((item) =>
        `<div class="tiny"><strong>${{item.name}}</strong>: ${{Number(item.average_seconds_per_iteration).toFixed(6)}} s/iter</div>`
      ).join("");
      document.getElementById("benchmark-summary").innerHTML = `
        <div class="tiny">Total elapsed: ${{Number(benchmarks.total_elapsed_seconds).toFixed(6)}} s</div>
        <div style="margin-top:10px;">${{cases}}</div>
      `;
    }}

    function renderTags(trajectory) {{
      const tags = (trajectory.trace_tags || []).map((tag) => `<span class="tag">${{tag}}</span>`).join("");
      document.getElementById("trace-tags").innerHTML = tags || '<span class="tiny">No trace tags attached.</span>';
    }}

    function renderActions(trajectory) {{
      const rows = (trajectory.controller_actions || []).map((action) => `
        <tr>
          <td>${{action.kind}}</td>
          <td>${{action.priority}}</td>
          <td>${{action.summary}}</td>
        </tr>
      `).join("");
      document.getElementById("controller-actions").innerHTML = rows || '<tr><td colspan="3" class="tiny">No controller actions attached.</td></tr>';
    }}

    function renderParticles(trajectory) {{
      document.getElementById("particle-table").innerHTML = trajectory.particles.map((particle) => `
        <tr>
          <td>${{particle.label}} (#${{particle.particle_index}})</td>
          <td>${{formatVector(particle.position)}}</td>
          <td>${{formatVector(particle.velocity)}}</td>
          <td>${{formatVector(particle.force)}}</td>
          <td>${{(particle.compartments || []).join(", ") || "none"}}</td>
        </tr>
      `).join("");
    }}

    function renderEdges(graph) {{
      if (!graph) {{
        document.getElementById("edge-table").innerHTML = '<tr><td colspan="5" class="tiny">No graph snapshot attached.</td></tr>';
        document.getElementById("graph-meta").textContent = 'No graph snapshot attached.';
        document.getElementById("graph-canvas").innerHTML = '';
        return;
      }}
      document.getElementById("graph-meta").textContent =
        `step=${{graph.step}}, active=${{graph.active_edge_count}}, structural=${{graph.structural_edge_count}}, adaptive=${{graph.adaptive_edge_count}}`;
      document.getElementById("edge-table").innerHTML = graph.edges.map((edge) => `
        <tr>
          <td>${{edge.source_index}}-${{edge.target_index}}</td>
          <td>${{edge.kind}}</td>
          <td>${{Number(edge.weight).toFixed(3)}}</td>
          <td>${{Number(edge.distance).toFixed(3)}}</td>
          <td>${{edge.route_label}}</td>
        </tr>
      `).join("");

      const svg = document.getElementById("graph-canvas");
      const nodes = graph.nodes || [];
      if (!nodes.length) {{
        svg.innerHTML = '';
        return;
      }}
      const xs = nodes.map((node) => Number(node.position[0]));
      const ys = nodes.map((node) => Number(node.position[1]));
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const scaleX = (value) => 60 + ((value - minX) / ((maxX - minX) || 1)) * 520;
      const scaleY = (value) => 260 - ((value - minY) / ((maxY - minY) || 1)) * 180;
      const nodeMap = new Map(nodes.map((node) => [node.particle_index, {{x: scaleX(Number(node.position[0])), y: scaleY(Number(node.position[1])), label: node.label}}]));
      const edgeMarkup = graph.edges.map((edge) => {{
        const a = nodeMap.get(edge.source_index);
        const b = nodeMap.get(edge.target_index);
        const stroke = edge.kind.includes('structural') ? '#18222d' : '#c75c2a';
        return `<line x1="${{a.x}}" y1="${{a.y}}" x2="${{b.x}}" y2="${{b.y}}" stroke="${{stroke}}" stroke-width="${{1 + Number(edge.weight) * 2}}" opacity="0.7" />`;
      }}).join('');
      const nodeMarkup = nodes.map((node) => {{
        const point = nodeMap.get(node.particle_index);
        const fill = (node.compartments || []).length ? '#c75c2a' : '#18222d';
        return `
          <circle cx="${{point.x}}" cy="${{point.y}}" r="10" fill="${{fill}}" />
          <text x="${{point.x}}" y="${{point.y - 16}}" text-anchor="middle" font-size="12" fill="#18222d">${{node.label}}</text>
        `;
      }}).join('');
      svg.innerHTML = edgeMarkup + nodeMarkup;
    }}

    async function refresh() {{
      try {{
        const response = await fetch(`${{jsonEndpoint}}?t=${{Date.now()}}`, {{ cache: 'no-store' }});
        const data = await response.json();
        document.getElementById("timestamp").textContent = `Generated at ${{data.generated_at}}`;
        renderProblem(data.problem);
        renderMetrics(data.trajectory);
        renderValidation(data.validation);
        renderBenchmarks(data.benchmarks);
        renderTags(data.trajectory);
        renderActions(data.trajectory);
        renderParticles(data.trajectory);
        renderEdges(data.graph);
      }} catch (error) {{
        document.getElementById("timestamp").textContent = `Waiting for snapshot... (${{error}})`;
      }}
    }}

    refresh();
    setInterval(refresh, refreshMs);
  </script>
</body>
</html>
"""
