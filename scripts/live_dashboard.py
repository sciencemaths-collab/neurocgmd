"""Local live-dashboard harness for the current NeuroCGMD foundation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_control import ExecutiveController
from benchmarks import BaselineBenchmarkSuite
from chemistry import ChemistryInterfaceAnalyzer, ProteinChemistryModel
from compartments import CompartmentRegistry
from core.state import SimulationState
from core.state_registry import LifecycleStage, SimulationStateRegistry
from forcefields import HybridForceEngine
from forcefields.base_forcefield import BaseForceField
from graph import ConnectivityGraph, ConnectivityGraphManager
from integrators.langevin import LangevinIntegrator
from memory import ReplayBuffer, TraceStore
from ml import (
    HeuristicUncertaintyModel,
    LiveFeatureEncoder,
    ResidualModel,
    ScalableResidualModel,
)
from physics.forces.composite import BaselineForceEvaluator, ForceEvaluation
from qcloud import (
    LocalRegionSelector,
    NullQCloudCorrectionModel,
    QCloudForceCoupler,
    RegionSelectionPolicy,
)
from sampling.scenarios import (
    BarnaseBarstarScenario,
    ComplexAssemblyProgress,
    ComplexAssemblySetup,
    DashboardScenario,
    EncounterComplexScenario,
    ImportedProteinComplexScenario,
    ImportedProteinScenarioSpec,
    SpikeAce2Scenario,
)
from sampling import HybridProductionEngine, ProductionCycleReport
from sampling.simulation_loop import SimulationLoop
from topology import SystemTopology
from validation import FoundationSanityChecker, TrajectoryDriftChecker
from visualization import (
    DashboardSnapshotView,
    GraphSnapshotView,
    ObjectiveMetricView,
    ProblemStatementView,
)


def _load_export_registry() -> Any:
    module_path = REPO_ROOT / "io" / "export_registry.py"
    spec = importlib.util.spec_from_file_location("project_export_registry", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load io/export_registry.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("project_export_registry", module)
    spec.loader.exec_module(module)
    return module


@dataclass(slots=True)
class LiveDashboardContext:
    """Mutable runtime context for the local live dashboard."""

    scenario: DashboardScenario
    scenario_setup: ComplexAssemblySetup
    engine: HybridProductionEngine
    registry: SimulationStateRegistry
    topology: SystemTopology
    forcefield: BaseForceField
    loop: SimulationLoop
    graph_manager: ConnectivityGraphManager
    compartments: CompartmentRegistry
    trace_store: TraceStore
    replay_buffer: ReplayBuffer
    residual_model: ResidualModel
    uncertainty_model: HeuristicUncertaintyModel
    chemistry_model: ProteinChemistryModel
    chemistry_analyzer: ChemistryInterfaceAnalyzer
    live_feature_encoder: LiveFeatureEncoder
    controller: ExecutiveController
    qcloud_coupler: QCloudForceCoupler
    qcloud_selector: LocalRegionSelector
    benchmark_suite: BaselineBenchmarkSuite
    sanity_checker: FoundationSanityChecker
    drift_checker: TrajectoryDriftChecker
    force_evaluator: BaselineForceEvaluator = field(default_factory=BaselineForceEvaluator)
    qcloud_correction_model: NullQCloudCorrectionModel = field(default_factory=NullQCloudCorrectionModel)
    graph: ConnectivityGraph | None = None


@dataclass(frozen=True, slots=True)
class DashboardStepData:
    """Raw per-step diagnostics shared by the live dashboard and batch validation."""

    state: Any
    graph: ConnectivityGraph
    progress: ComplexAssemblyProgress
    base_evaluation: ForceEvaluation
    qcloud_result: Any
    structure_report: Any
    fidelity_report: Any
    chemistry_report: Any
    trace_record: Any
    residual_prediction: Any
    live_features: Any
    uncertainty: Any
    decision: Any
    sanity_report: Any
    drift_report: Any
    benchmark_report: Any


def _build_context_for_scenario(
    scenario: DashboardScenario,
    *,
    initial_state_override: SimulationState | None = None,
) -> LiveDashboardContext:
    """Build a live dashboard context for one concrete scenario."""

    setup = scenario.build_setup()
    registry = SimulationStateRegistry(created_by="live-dashboard")
    chemistry_model = ProteinChemistryModel()
    if initial_state_override is None:
        registry.create_initial_state(
            particles=setup.initial_particles,
            thermodynamics=setup.thermodynamics,
        )
    else:
        registry.create_initial_state(
            particles=initial_state_override.particles,
            units=initial_state_override.units,
            thermodynamics=initial_state_override.thermodynamics,
            cell=initial_state_override.cell,
            time=initial_state_override.time,
            step=initial_state_override.step,
            potential_energy=initial_state_override.potential_energy,
            observables=initial_state_override.observables,
            stage=LifecycleStage.CHECKPOINT,
            notes="context initial-state override",
            metadata={
                "override_source_state_id": str(initial_state_override.provenance.state_id),
                "override_source_stage": initial_state_override.provenance.stage,
            },
        )
    force_evaluator = BaselineForceEvaluator()
    loop = SimulationLoop(
        topology=setup.topology,
        forcefield=setup.forcefield,
        integrator=LangevinIntegrator(
            time_step=setup.integrator_time_step,
            friction_coefficient=setup.integrator_friction,
        ),
        force_evaluator=force_evaluator,
        registry=registry,
    )
    residual_model: ResidualModel = ScalableResidualModel(
        hidden_dim=24,
        interaction_cutoff=2.4,
        learning_rate=0.003,
        force_loss_weight=0.35,
        confidence_growth_rate=0.025,
    )
    graph_manager = ConnectivityGraphManager()
    trace_store = TraceStore(simulation_id=registry.require_simulation_id())
    replay_buffer = ReplayBuffer(capacity=64)
    uncertainty_model = HeuristicUncertaintyModel(
        base_uncertainty=0.12,
        low_confidence_scale=0.55,
        priority_tag_bonus=0.1,
        trigger_threshold=0.45,
    )
    chemistry_analyzer = ChemistryInterfaceAnalyzer(chemistry_model=chemistry_model)
    live_feature_encoder = LiveFeatureEncoder()
    controller = ExecutiveController()
    qcloud_selector = LocalRegionSelector(
        policy=RegionSelectionPolicy(max_regions=2, max_region_size=4, min_region_score=0.1)
    )
    qcloud_correction_model = scenario.build_qcloud_correction_model() or NullQCloudCorrectionModel()
    hybrid_force_engine = HybridForceEngine()
    benchmark_suite = BaselineBenchmarkSuite(default_repeats=2)
    sanity_checker = FoundationSanityChecker()
    drift_checker = TrajectoryDriftChecker()
    engine = HybridProductionEngine(
        scenario=scenario,
        setup=setup,
        registry=registry,
        integrator=loop.integrator,
        compartments=setup.compartments,
        hybrid_engine=hybrid_force_engine,
        graph_manager=graph_manager,
        trace_store=trace_store,
        replay_buffer=replay_buffer,
        residual_model=residual_model,
        uncertainty_model=uncertainty_model,
        chemistry_analyzer=chemistry_analyzer,
        live_feature_encoder=live_feature_encoder,
        controller=controller,
        qcloud_selector=qcloud_selector,
        qcloud_correction_model=qcloud_correction_model,
        benchmark_suite=benchmark_suite,
        sanity_checker=sanity_checker,
        drift_checker=drift_checker,
        reference_force_evaluator=force_evaluator,
        default_benchmark_repeats=1,
    )
    return LiveDashboardContext(
        scenario=scenario,
        scenario_setup=setup,
        engine=engine,
        registry=registry,
        topology=setup.topology,
        forcefield=setup.forcefield,
        loop=loop,
        graph_manager=graph_manager,
        compartments=setup.compartments,
        trace_store=trace_store,
        replay_buffer=replay_buffer,
        residual_model=residual_model,
        uncertainty_model=uncertainty_model,
        chemistry_model=chemistry_model,
        chemistry_analyzer=chemistry_analyzer,
        live_feature_encoder=live_feature_encoder,
        controller=controller,
        qcloud_coupler=hybrid_force_engine.qcloud_coupler,
        qcloud_selector=qcloud_selector,
        benchmark_suite=benchmark_suite,
        sanity_checker=sanity_checker,
        drift_checker=drift_checker,
        force_evaluator=force_evaluator,
        qcloud_correction_model=qcloud_correction_model,
    )


def build_dashboard_context_for_scenario(
    scenario: DashboardScenario,
    *,
    initial_state_override: SimulationState | None = None,
) -> LiveDashboardContext:
    """Public helper for building a dashboard context from an explicit scenario object."""

    return _build_context_for_scenario(scenario, initial_state_override=initial_state_override)


def build_complex_assembly_context() -> LiveDashboardContext:
    """Build the encounter-complex demo context."""

    return _build_context_for_scenario(EncounterComplexScenario())


def build_barnase_barstar_context() -> LiveDashboardContext:
    """Build the barnase-barstar reference benchmark context."""

    return _build_context_for_scenario(BarnaseBarstarScenario())


def build_spike_ace2_context() -> LiveDashboardContext:
    """Build the harder ACE2-spike reference benchmark context."""

    return _build_context_for_scenario(SpikeAce2Scenario())


def _load_imported_scenario_spec(path_text: str | Path) -> ImportedProteinScenarioSpec:
    resolved_path = Path(path_text).expanduser().resolve()
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    spec_payload = payload["scenario_spec"] if isinstance(payload, dict) and "scenario_spec" in payload else payload
    return ImportedProteinScenarioSpec.from_dict(spec_payload)


def build_imported_protein_context(spec: ImportedProteinScenarioSpec) -> LiveDashboardContext:
    """Build a dashboard context from a generic imported-protein scenario spec."""

    return _build_context_for_scenario(ImportedProteinComplexScenario(spec=spec))


def build_live_dashboard_context(
    scenario_name: str = "spike_ace2",
    *,
    imported_spec: ImportedProteinScenarioSpec | None = None,
) -> LiveDashboardContext:
    """Resolve and build the requested live-dashboard scenario."""

    if imported_spec is not None:
        return build_imported_protein_context(imported_spec)
    if scenario_name == "spike_ace2":
        return build_spike_ace2_context()
    if scenario_name == "barnase_barstar":
        return build_barnase_barstar_context()
    if scenario_name == "encounter_complex":
        return build_complex_assembly_context()
    raise ValueError(f"Unknown scenario: {scenario_name!r}")


def build_demo_context() -> LiveDashboardContext:
    """Backward-compatible alias for the scenario-driven live dashboard context."""

    return build_live_dashboard_context()


def _ensure_graph(context: LiveDashboardContext) -> ConnectivityGraph:
    context.graph = context.engine.current_graph()
    return context.graph


def _build_chemistry_report(
    context: LiveDashboardContext,
    state,
    progress: ComplexAssemblyProgress,
):
    focus_compartments = tuple(str(value) for value in context.scenario_setup.focus_compartments)
    if len(focus_compartments) < 2:
        return None
    distance_cutoff = max(progress.capture_distance, progress.contact_distance + 0.25)
    return context.chemistry_analyzer.assess(
        state,
        context.topology,
        context.compartments,
        compartment_ids=(focus_compartments[0], focus_compartments[1]),
        distance_cutoff=distance_cutoff,
    )


def _build_problem_view(
    context: LiveDashboardContext,
    state,
    progress: ComplexAssemblyProgress,
    *,
    structure_report=None,
    chemistry_report=None,
    fidelity_report=None,
) -> ProblemStatementView:
    reference_title = ""
    reference_summary = ""
    reference_metrics: tuple[ObjectiveMetricView, ...] = ()
    reference_report = context.scenario.build_reference_report(progress)
    if reference_report is not None:
        reference_title = reference_report.title
        reference_summary = reference_report.summary
        reference_metrics = tuple(
            ObjectiveMetricView(
                label=metric.label,
                value=metric.current_value,
                detail=f"Target: {metric.target_value}. {metric.detail}".strip(),
                status=(
                    "good"
                    if metric.status == "good"
                    else "warn"
                    if metric.status == "warn"
                    else "active"
                ),
            )
            for metric in reference_report.metrics
        )
    structure_title = ""
    structure_summary = ""
    structure_metrics: tuple[ObjectiveMetricView, ...] = ()
    if structure_report is not None:
        structure_title = structure_report.title
        structure_summary = structure_report.summary
        structure_metrics = tuple(
            ObjectiveMetricView(
                label=metric.label,
                value=metric.value,
                detail=metric.detail,
                status=metric.status,
            )
            for metric in structure_report.metrics
        )
    chemistry_title = ""
    chemistry_summary = ""
    chemistry_metrics: tuple[ObjectiveMetricView, ...] = ()
    if chemistry_report is not None:
        chemistry_title = chemistry_report.title
        chemistry_summary = (
            f"Evaluated {chemistry_report.evaluated_pair_count} active interface pairs across "
            f"{chemistry_report.compartment_ids[0]} and {chemistry_report.compartment_ids[1]}."
        )
        chemistry_metrics = (
            ObjectiveMetricView(
                label="Mean Chemistry Score",
                value=f"{chemistry_report.mean_pair_score:.3f}",
                detail="Higher means the observed interface better matches bounded chemistry expectations.",
                status="good" if chemistry_report.mean_pair_score >= 0.65 else "warn" if chemistry_report.mean_pair_score < 0.45 else "active",
            ),
            ObjectiveMetricView(
                label="Favorable Pair Fraction",
                value=f"{chemistry_report.favorable_pair_fraction:.1%}",
                detail="Fraction of active interface pairs whose chemistry score clears the favorable threshold.",
                status="good" if chemistry_report.favorable_pair_fraction >= 0.6 else "warn" if chemistry_report.favorable_pair_fraction < 0.35 else "active",
            ),
            ObjectiveMetricView(
                label="Charge Complementarity",
                value=f"{chemistry_report.charge_complementarity:.3f}",
                detail="Opposite-charge or neutral-compatible pairing quality across the active interface.",
                status="good" if chemistry_report.charge_complementarity >= 0.65 else "active",
            ),
            ObjectiveMetricView(
                label="Flexibility Pressure",
                value=f"{chemistry_report.flexibility_pressure:.3f}",
                detail="Higher means the current interface is dominated by flexible or poorly anchored chemistry.",
                status="warn" if chemistry_report.flexibility_pressure >= 0.45 else "active",
            ),
        )
    fidelity_title = ""
    fidelity_summary = ""
    fidelity_metrics: tuple[ObjectiveMetricView, ...] = ()
    if fidelity_report is not None:
        fidelity_title = fidelity_report.title
        fidelity_summary = (
            f"Target: {fidelity_report.target_label}. "
            f"{len(fidelity_report.improved_metrics())}/{len(fidelity_report.metrics)} metrics improved."
        )
        fidelity_metrics = tuple(
            ObjectiveMetricView(
                label=metric.label.replace("_", " ").title(),
                value=f"{metric.baseline_error:.3f} -> {metric.corrected_error:.3f}",
                detail="Shadow-corrected error relative to the trusted target should move downward." if metric.improved else "Current shadow correction has not yet improved this metric.",
                status="good" if metric.improved else "warn",
            )
            for metric in fidelity_report.metrics
        )
    return ProblemStatementView(
        title=context.scenario_setup.title.replace("NeuroCGMD Live Dashboard | ", ""),
        summary=context.scenario_setup.summary,
        objective=context.scenario_setup.objective,
        stage=progress.stage_label,
        metrics=(
            ObjectiveMetricView(
                label="Interface Gap",
                value=f"{progress.interface_distance:.3f}",
                detail=f"Initial {progress.initial_interface_distance:.3f}, bound <= {progress.bound_distance:.3f}",
                status="good" if progress.bound else "active",
            ),
            ObjectiveMetricView(
                label="Cross Contacts",
                value=f"{progress.cross_contact_count}/{progress.target_contact_count}",
                detail=f"Contacts counted at <= {progress.contact_distance:.2f} reduced units",
                status="good" if progress.cross_contact_count >= progress.target_contact_count else "active",
            ),
            ObjectiveMetricView(
                label="Graph Bridges",
                value=f"{progress.graph_bridge_count}",
                detail="Inter-complex edges currently linking compartments A and B",
                status="active" if progress.graph_bridge_count else "neutral",
            ),
            ObjectiveMetricView(
                label="Assembly Score",
                value=f"{progress.assembly_score:.3f}",
                detail="Weighted blend of distance closure, physical contacts, and graph bridging",
                status="good" if progress.assembly_score >= 0.8 else "warn" if progress.assembly_score < 0.4 else "active",
            ),
        ),
        reference_title=reference_title,
        reference_summary=reference_summary,
        reference_metrics=reference_metrics,
        structure_title=structure_title,
        structure_summary=structure_summary,
        structure_metrics=structure_metrics,
        chemistry_title=chemistry_title,
        chemistry_summary=chemistry_summary,
        chemistry_metrics=chemistry_metrics,
        fidelity_title=fidelity_title,
        fidelity_summary=fidelity_summary,
        fidelity_metrics=fidelity_metrics,
        metadata={
            "scenario": context.scenario.name,
            "bound": progress.bound,
            "classification": context.scenario.classification,
        },
    )


def collect_dashboard_step_data(
    context: LiveDashboardContext,
    *,
    benchmark_repeats: int = 1,
) -> DashboardStepData:
    """Collect one reusable dashboard/validation diagnostic slice from the current context."""

    if benchmark_repeats <= 0:
        raise ValueError("benchmark_repeats must be strictly positive.")

    cycle_report: ProductionCycleReport = context.engine.collect_cycle(
        benchmark_repeats=benchmark_repeats,
    )
    context.graph = cycle_report.graph
    return DashboardStepData(
        state=cycle_report.state,
        graph=cycle_report.graph,
        progress=cycle_report.progress,
        base_evaluation=cycle_report.classical_evaluation,
        qcloud_result=cycle_report.qcloud_result,
        structure_report=cycle_report.structure_report,
        fidelity_report=cycle_report.fidelity_report,
        chemistry_report=cycle_report.chemistry_report,
        trace_record=cycle_report.trace_record,
        residual_prediction=cycle_report.residual_prediction,
        live_features=cycle_report.live_features,
        uncertainty=cycle_report.final_uncertainty,
        decision=cycle_report.final_decision,
        sanity_report=cycle_report.sanity_report,
        drift_report=cycle_report.drift_report,
        benchmark_report=cycle_report.benchmark_report,
    )


def build_dashboard_snapshot(
    context: LiveDashboardContext,
    *,
    title: str | None = None,
) -> DashboardSnapshotView:
    """Build one dashboard snapshot from the current runtime context."""

    step_data = collect_dashboard_step_data(context)
    state = step_data.state
    graph = step_data.graph
    graph_view = GraphSnapshotView.from_state_graph(
        state,
        graph,
        compartments=context.compartments,
    )
    problem_view = _build_problem_view(
        context,
        state,
        step_data.progress,
        structure_report=step_data.structure_report,
        chemistry_report=step_data.chemistry_report,
        fidelity_report=step_data.fidelity_report,
    )
    return DashboardSnapshotView.from_components(
        title=title or context.scenario_setup.title,
        problem=problem_view,
        state=state,
        graph=graph_view,
        compartments=context.compartments,
        trace_record=step_data.trace_record,
        controller_decision=step_data.decision,
        uncertainty_estimate=step_data.uncertainty,
        qcloud_result=step_data.qcloud_result,
        sanity_report=step_data.sanity_report,
        drift_report=step_data.drift_report,
        benchmark_report=step_data.benchmark_report,
        metadata={
            "source": "live_dashboard",
            "scenario": context.scenario.name,
            "assembly_score": step_data.progress.assembly_score,
            "chemistry_mean_pair_score": (
                step_data.chemistry_report.mean_pair_score if step_data.chemistry_report is not None else None
            ),
        },
    )


def write_dashboard_snapshot(
    context: LiveDashboardContext,
    output_dir: str | Path,
    *,
    title: str | None = None,
    refresh_ms: int = 1000,
) -> Any:
    """Build and export one dashboard snapshot bundle."""

    snapshot = build_dashboard_snapshot(context, title=title)
    export_registry = _load_export_registry()
    return export_registry.export_dashboard_snapshot(
        snapshot,
        output_dir,
        refresh_ms=refresh_ms,
    )


def advance_demo(context: LiveDashboardContext, *, steps: int = 1) -> None:
    """Advance the demo simulation context forward by the requested number of steps."""

    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if steps == 0:
        return
    context.engine.integrator = context.loop.integrator
    context.engine.advance(steps, record_final_state=False)
    _ensure_graph(context)


def _start_server(output_dir: Path, port: int) -> ThreadingHTTPServer:
    handler = partial(SimpleHTTPRequestHandler, directory=str(output_dir))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local NeuroCGMD live dashboard demo.")
    parser.add_argument("--output-dir", default="/tmp/neurocgmd_dashboard", help="Directory for dashboard artifacts.")
    parser.add_argument("--steps", type=int, default=12, help="Number of streamed dashboard updates to emit.")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between dashboard updates.")
    parser.add_argument("--port", type=int, default=8765, help="Port to use when serving the dashboard.")
    parser.add_argument(
        "--scenario",
        default="spike_ace2",
        help="Scientific scenario to stream in the live dashboard.",
    )
    parser.add_argument(
        "--import-spec",
        default=None,
        help="Optional JSON import summary/spec file created by import_protein_system.py.",
    )
    parser.add_argument("--serve", action="store_true", help="Serve the dashboard directory over HTTP while streaming.")
    parser.add_argument("--refresh-ms", type=int, default=1000, help="Client-side polling interval for the dashboard HTML.")
    parser.add_argument("--title", default=None, help="Optional dashboard title override.")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve()
    imported_spec = _load_imported_scenario_spec(args.import_spec) if args.import_spec else None
    context = build_live_dashboard_context(args.scenario, imported_spec=imported_spec)
    server: ThreadingHTTPServer | None = None

    if args.serve:
        server = _start_server(output_dir, args.port)
        print(f"Serving dashboard at http://127.0.0.1:{args.port}/index.html")

    try:
        for index in range(max(1, args.steps)):
            if index > 0:
                advance_demo(context, steps=1)
            bundle = write_dashboard_snapshot(
                context,
                output_dir,
                title=args.title,
                refresh_ms=args.refresh_ms,
            )
            print(f"[step {context.registry.latest_state().step}] wrote dashboard to {bundle.output_dir}")
            if index < max(1, args.steps) - 1:
                time.sleep(max(0.0, args.interval))
    finally:
        if server is not None:
            server.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
