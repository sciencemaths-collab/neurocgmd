"""Manifest-driven NeuroCGMD prepare/run/analyze CLI."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import RunManifest, load_run_manifest
from core.state import SimulationState
from core.state_registry import SimulationStateRegistry
from ml import ResidualMemoryModel, ScalableResidualModel
from prepare import PreparationPipeline
from qcloud import LocalRegionSelector, NullQCloudCorrectionModel, RegionSelectionPolicy
from sampling.scenarios import ImportedProteinComplexScenario
from sampling.stage_runner import ProductionStageRunner
from scripts.live_dashboard import build_dashboard_context_for_scenario


def _load_module(module_name: str, relative_path: str) -> Any:
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(module_name, module)
    spec.loader.exec_module(module)
    return module


def _load_trajectory_writer() -> Any:
    return _load_module("project_trajectory_writer", "io/trajectory_writer.py")


def _load_checkpoint_writer() -> Any:
    return _load_module("project_checkpoint_writer", "io/checkpoint_writer.py")


def _runtime_seed_to_state(bundle) -> SimulationState:
    registry = SimulationStateRegistry(created_by="neurocgmd-bootstrap")
    return registry.create_initial_state(
        particles=bundle.runtime_seed.particles,
        units=bundle.runtime_seed.units,
        thermodynamics=bundle.runtime_seed.thermodynamics,
        cell=bundle.runtime_seed.cell,
        notes="prepared runtime seed bootstrap",
    )


def _resolve_output_path(manifest: RunManifest, relative_path: str) -> Path:
    return manifest.outputs.resolve(REPO_ROOT, relative_path)


def _load_prepared_bundle(path: Path):
    pipeline = PreparationPipeline()
    return pipeline.load_bundle(path)


def _configure_context_from_manifest(context: Any, manifest: RunManifest) -> None:
    qcloud_policy = RegionSelectionPolicy(
        max_regions=manifest.hybrid.qcloud.max_regions,
        max_region_size=context.qcloud_selector.policy.max_region_size,
        bonded_neighbor_hops=context.qcloud_selector.policy.bonded_neighbor_hops,
        min_region_score=min(
            context.qcloud_selector.policy.min_region_score,
            manifest.hybrid.qcloud.trigger_threshold,
        ),
        inter_compartment_bonus=context.qcloud_selector.policy.inter_compartment_bonus,
        long_range_bonus=context.qcloud_selector.policy.long_range_bonus,
        memory_priority_bonus=context.qcloud_selector.policy.memory_priority_bonus,
        focus_compartment_bonus=context.qcloud_selector.policy.focus_compartment_bonus,
    )
    selector = LocalRegionSelector(policy=qcloud_policy)
    context.qcloud_selector = selector
    context.engine.qcloud_selector = selector

    if not manifest.hybrid.qcloud.enabled:
        null_model = NullQCloudCorrectionModel()
        context.qcloud_correction_model = null_model
        context.engine.qcloud_correction_model = null_model

    if not manifest.hybrid.ml.enabled:
        residual_model = ResidualMemoryModel()
    elif manifest.hybrid.ml.model == "scalable_residual":
        residual_model = ScalableResidualModel(
            hidden_dim=24,
            interaction_cutoff=max(2.0, manifest.neighbor_list.vdw_cutoff_nm + manifest.neighbor_list.neighbor_skin_nm),
            learning_rate=0.003,
            force_loss_weight=0.35,
            confidence_growth_rate=0.025,
        )
    else:
        residual_model = ResidualMemoryModel()
    context.residual_model = residual_model
    context.engine.residual_model = residual_model


def _prepare_bundle(manifest: RunManifest) -> Path:
    pipeline = PreparationPipeline()
    bundle = pipeline.prepare(manifest)
    destination = _resolve_output_path(manifest, manifest.outputs.prepared_bundle)
    pipeline.write_bundle(bundle, destination)
    return destination


def _build_run_context(manifest: RunManifest, prepared_bundle_path: Path) -> tuple[Any, Any]:
    bundle = _load_prepared_bundle(prepared_bundle_path)
    scenario = ImportedProteinComplexScenario(spec=bundle.scenario_spec)
    seed_state = _runtime_seed_to_state(bundle)
    context = build_dashboard_context_for_scenario(scenario, initial_state_override=seed_state)
    _configure_context_from_manifest(context, manifest)
    return context, bundle


def _run_prepare(args: argparse.Namespace) -> int:
    manifest = load_run_manifest(args.config)
    destination = _prepare_bundle(manifest)
    print(f"Wrote prepared bundle to {destination}")
    return 0


def _run_run(args: argparse.Namespace) -> int:
    manifest = load_run_manifest(args.config)
    prepared_bundle_path = (
        Path(args.prepared_bundle).expanduser().resolve()
        if args.prepared_bundle
        else _resolve_output_path(manifest, manifest.outputs.prepared_bundle)
    )
    if not prepared_bundle_path.exists():
        prepared_bundle_path = _prepare_bundle(manifest)

    context, bundle = _build_run_context(manifest, prepared_bundle_path)
    trajectory_writer_module = _load_trajectory_writer()
    checkpoint_writer_module = _load_checkpoint_writer()
    trajectory_path = _resolve_output_path(manifest, manifest.outputs.trajectory)
    checkpoint_path = _resolve_output_path(manifest, manifest.outputs.checkpoint)
    energy_path = _resolve_output_path(manifest, manifest.outputs.energy)
    run_summary_path = _resolve_output_path(manifest, manifest.outputs.run_summary)
    log_path = _resolve_output_path(manifest, manifest.outputs.log)

    runner = ProductionStageRunner()
    run_summary = runner.run(
        context=context,
        manifest=manifest,
        prepared_bundle_path=str(prepared_bundle_path),
        trajectory_writer=trajectory_writer_module.TrajectoryWriter(trajectory_path),
        checkpoint_writer=checkpoint_writer_module.CheckpointWriter(checkpoint_path),
        energy_path=energy_path,
        benchmark_repeats=max(0, args.benchmark_repeats),
    )
    run_summary_path.parent.mkdir(parents=True, exist_ok=True)
    run_summary_path.write_text(json.dumps(run_summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    log_path.write_text(
        "\n".join(
            [
                f"system={manifest.system.name}",
                f"prepared_bundle={prepared_bundle_path}",
                f"trajectory={trajectory_path}",
                f"energy={energy_path}",
                f"checkpoint={checkpoint_path}",
                f"final_state_id={run_summary.final_state.provenance.state_id}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote run summary to {run_summary_path}")
    print(f"Wrote trajectory to {trajectory_path}")
    print(f"Wrote checkpoint to {checkpoint_path}")
    print(f"Wrote energy log to {energy_path}")
    print(
        f"Completed {manifest.system.name} with "
        f"{len(run_summary.stage_records)} stages and final step {run_summary.final_state.step}."
    )

    # ---- Auto-run analysis + plotting ----
    print("\n" + "=" * 60)
    print("Running post-simulation analysis and plotting...")
    print("=" * 60)
    try:
        _run_analyze(args)
    except Exception as exc:
        print(f"Warning: HTML analysis failed: {exc}")

    try:
        from scripts.plot_analysis_png import main as plot_main
        output_dir = str((REPO_ROOT / manifest.outputs.output_dir).resolve())
        plot_main(output_dir)
    except Exception as exc:
        print(f"Warning: PNG plotting failed: {exc}")

    return 0


def _render_analysis_html(report: dict[str, object]) -> str:
    metrics_html = "".join(
        f"<li><strong>{label}</strong>: {value}</li>"
        for label, value in report["headline_metrics"].items()
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{report['title']}</title>"
        "<style>body{font-family:Georgia,serif;margin:2rem;line-height:1.5}code{font-family:monospace}"
        "section{margin-bottom:1.5rem}ul{padding-left:1.2rem}</style></head><body>"
        f"<h1>{report['title']}</h1>"
        f"<section><p>{report['summary']}</p></section>"
        "<section><h2>Headline Metrics</h2><ul>"
        f"{metrics_html}</ul></section>"
        f"<section><h2>Cycle Metadata</h2><pre>{json.dumps(report['cycle_metadata'], indent=2, sort_keys=True)}</pre></section>"
        "</body></html>"
    )


def _load_trajectory_states(manifest: RunManifest) -> list[SimulationState]:
    """Load all frames from the trajectory JSONL file."""
    traj_path = _resolve_output_path(manifest, manifest.outputs.trajectory)
    states: list[SimulationState] = []
    if traj_path.exists():
        for line in traj_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                frame = json.loads(line)
                states.append(SimulationState.from_dict(frame["state"]))
    return states


def _load_energy_histories(manifest: RunManifest):
    """Parse energies.csv into (step, pe), (step, ke), (step, total) histories."""
    energy_path = _resolve_output_path(manifest, manifest.outputs.energy)
    pe_hist: list[tuple[float, float]] = []
    ke_hist: list[tuple[float, float]] = []
    total_hist: list[tuple[float, float]] = []
    if energy_path.exists():
        for line in energy_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("stage"):
                continue
            parts = line.split(",")
            if len(parts) >= 8:
                try:
                    step = float(parts[3])
                    pe = float(parts[5])
                    ke = float(parts[6])
                    total = float(parts[7])
                except ValueError:
                    continue
                pe_hist.append((step, pe))
                ke_hist.append((step, ke))
                total_hist.append((step, total))
    return pe_hist, ke_hist, total_hist


def _load_qcloud_event_data(manifest: RunManifest) -> dict:
    """Load QCloud event analysis from run_summary.json if available."""
    run_summary_path = _resolve_output_path(manifest, manifest.outputs.run_summary)
    if run_summary_path.exists():
        try:
            data = json.loads(run_summary_path.read_text(encoding="utf-8"))
            metadata = data.get("metadata", {})
            return {
                "summary": metadata.get("qcloud_event_analysis", {}),
                "detected_events": metadata.get("qcloud_detected_events", []),
            }
        except (json.JSONDecodeError, KeyError):
            pass
    return {"summary": {}, "detected_events": []}


def _run_analyze(args: argparse.Namespace) -> int:
    from validation.adaptive_analysis import RDFCalculator, RMSDTracker, UmbrellaSampler, UmbrellaSamplingWindow
    from validation.molecular_observables import (
        MolecularObservableCollector,
        SASACalculator,
        RadiusOfGyration,
    )
    from validation.transition_analysis import (
        TransitionStateDetector,
        ReactionCoordinateGenerator,
        generate_full_html_report,
    )

    manifest = load_run_manifest(args.config)
    prepared_bundle_path = (
        Path(args.prepared_bundle).expanduser().resolve()
        if args.prepared_bundle
        else _resolve_output_path(manifest, manifest.outputs.prepared_bundle)
    )
    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else _resolve_output_path(manifest, manifest.outputs.checkpoint)
    )
    bundle = _load_prepared_bundle(prepared_bundle_path)
    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint_state = SimulationState.from_dict(checkpoint_payload["state"])
    scenario = ImportedProteinComplexScenario(spec=bundle.scenario_spec)
    context = build_dashboard_context_for_scenario(scenario, initial_state_override=checkpoint_state)
    _configure_context_from_manifest(context, manifest)
    cycle = context.engine.collect_cycle(benchmark_repeats=manifest.analysis.benchmark_repeats)

    # ---- Load trajectory frames ----
    print("Loading trajectory frames...")
    traj_states = _load_trajectory_states(manifest)
    print(f"  Loaded {len(traj_states)} frames.")

    # ---- Load energy histories ----
    pe_hist, ke_hist, total_hist = _load_energy_histories(manifest)

    # ---- RMSD / RMSF ----
    print("Computing RMSD / RMSF...")
    rmsd_tracker = RMSDTracker(reference_positions=traj_states[0].particles.positions) if traj_states else None
    if rmsd_tracker is not None:
        for state in traj_states:
            rmsd_tracker.record(state.step, state)
    rmsd_summary = rmsd_tracker.summary() if rmsd_tracker else {}

    # ---- RDF ----
    print("Computing RDF...")
    rdf_calc = RDFCalculator(cutoff=2.0, n_bins=100)
    # Sample up to 200 frames for RDF to keep it fast
    rdf_stride = max(1, len(traj_states) // 200)
    for i in range(0, len(traj_states), rdf_stride):
        rdf_calc.accumulate(traj_states[i])
    r_vals, g_vals = rdf_calc.compute_rdf()
    rdf_summary = rdf_calc.summary()

    # ---- Molecular observables (SASA, Rg, H-bonds, contacts, secondary structure, E2E) ----
    print("Computing molecular observables (SASA, Rg, H-bonds, contacts, secondary structure)...")
    mol_collector = MolecularObservableCollector()
    topology = context.engine.setup.topology
    obs_stride = max(1, len(traj_states) // 100)
    for i in range(0, len(traj_states), obs_stride):
        mol_collector.record(traj_states[i].step, traj_states[i], topology)
    mol_summary = mol_collector.summary()

    # ---- Umbrella sampling / PMF via Boltzmann inversion on inter-entity distance ----
    print("Computing PMF from trajectory (Boltzmann inversion)...")
    from validation.adaptive_analysis import _distance
    from math import log as _log

    # Use inter-entity distance as CV (barnase-barstar center-of-mass distance)
    # Get bead counts from import_summary in the prepared bundle
    entities_info = bundle.import_summary.entities if hasattr(bundle, "import_summary") else ()

    group_a_indices: list[int] = []
    group_b_indices: list[int] = []
    if len(entities_info) >= 2:
        offset = 0
        for idx, ent in enumerate(entities_info):
            bc = getattr(ent, "bead_count", 0)
            indices = list(range(offset, offset + bc))
            if idx == 0:
                group_a_indices = indices
            elif idx == 1:
                group_b_indices = indices
            offset += bc
    elif traj_states:
        # Fallback: split particles in half
        n = traj_states[0].particle_count
        group_a_indices = list(range(n // 2))
        group_b_indices = list(range(n // 2, n))

    cv_history: list[float] = []
    pmf_cv: list[float] = []
    pmf_vals: list[float] = []
    if group_a_indices and group_b_indices:
        for state in traj_states:
            pos = state.particles.positions
            masses = state.particles.masses
            # Center of mass for each group
            def _com(indices):
                total_m = sum(masses[i] for i in indices)
                if total_m == 0:
                    total_m = 1.0
                cx = sum(masses[i] * pos[i][0] for i in indices) / total_m
                cy = sum(masses[i] * pos[i][1] for i in indices) / total_m
                cz = sum(masses[i] * pos[i][2] for i in indices) / total_m
                return (cx, cy, cz)
            com_a = _com(group_a_indices)
            com_b = _com(group_b_indices)
            cv_history.append(_distance(com_a, com_b))

        # Boltzmann inversion PMF: F(r) = -kT * ln(P(r)) + const
        if cv_history:
            n_bins = 50
            cv_min = min(cv_history)
            cv_max = max(cv_history)
            if cv_max > cv_min:
                bin_width = (cv_max - cv_min) / n_bins
                histogram = [0] * n_bins
                for cv in cv_history:
                    b = min(int((cv - cv_min) / bin_width), n_bins - 1)
                    histogram[b] += 1
                max_count = max(histogram)
                kbt = 1.0  # reduced units
                raw_pmf = []
                for i in range(n_bins):
                    r = cv_min + (i + 0.5) * bin_width
                    pmf_cv.append(r)
                    if histogram[i] > 0:
                        raw_pmf.append(-kbt * _log(histogram[i] / max_count))
                    else:
                        raw_pmf.append(float("inf"))
                finite = [v for v in raw_pmf if v != float("inf")]
                pmf_min = min(finite) if finite else 0.0
                pmf_vals = [v - pmf_min if v != float("inf") else float("inf") for v in raw_pmf]

    # ---- Transition state detection ----
    print("Detecting transition states...")
    transition_detector = TransitionStateDetector()
    if cv_history:
        pe_by_step = {int(s): e for s, e in pe_hist} if pe_hist else {}
        for i, state in enumerate(traj_states):
            cv_val = cv_history[i]
            energy = pe_by_step.get(state.step, state.potential_energy or 0.0)
            transition_detector.record(state.step, cv_val, energy)
    detected_transitions = transition_detector.detect_transitions()

    # ---- Binding energy from entity interaction ----
    print("Computing binding energy decomposition...")
    binding_energy: dict[str, float] = {}
    if group_a_indices and group_b_indices and traj_states:
        # Estimate from last 20% of trajectory
        n_tail = max(1, len(traj_states) // 5)
        tail_states = traj_states[-n_tail:]
        pe_values = [s.potential_energy for s in tail_states if s.potential_energy is not None]
        if pe_values:
            binding_energy["mean_potential_energy"] = sum(pe_values) / len(pe_values)
            binding_energy["n_frames_averaged"] = float(len(pe_values))

    # ---- Build analysis report object for generate_full_html_report ----
    class _AnalysisReport:
        pass

    report_obj = _AnalysisReport()
    report_obj.steps_analyzed = len(traj_states)
    report_obj.convergence_metrics = {
        "rdf_peak_position_nm": f"{rdf_summary.get('peak_position', 0.0):.3f}",
        "rdf_coordination_number": f"{rdf_summary.get('coordination_number', 0.0):.2f}",
        "sasa_mean_nm2": f"{mol_summary.get('sasa_mean_nm2', 0.0):.3f}",
        "rg_mean_nm": f"{mol_summary.get('rg_mean_nm', 0.0):.3f}",
        "hbonds_mean": f"{mol_summary.get('hbonds_mean', 0.0):.1f}",
        "contacts_mean": f"{mol_summary.get('contacts_mean', 0.0):.1f}",
        "e2e_mean_nm": f"{mol_summary.get('e2e_mean_nm', 0.0):.3f}",
    }
    report_obj.rmsd_summary = {
        **rmsd_summary,
        "history": list(rmsd_tracker._history) if rmsd_tracker else [],
        "rmsf": list(rmsd_summary.get("rmsf_per_particle", ())),
    }
    report_obj.rdf_data = {"r_values": list(r_vals), "g_r_values": list(g_vals)}
    report_obj.pmf_data = {"cv_values": pmf_cv, "pmf_values": pmf_vals}
    report_obj.binding_energy = binding_energy

    def _is_converged():
        if rmsd_summary and rmsd_summary.get("std_rmsd", 1.0) < 0.5 * rmsd_summary.get("mean_rmsd", 1.0):
            return True
        return False
    report_obj.is_converged = _is_converged

    class _EnergyTracker:
        pass
    energy_tracker = _EnergyTracker()
    energy_tracker.ke_history = ke_hist
    energy_tracker.pe_history = pe_hist
    energy_tracker.total_history = total_hist
    energy_tracker.temperature_history = []  # CG reduced units, no explicit temperature

    # ---- Generate full HTML report with all SVG plots ----
    print("Generating HTML report with SVG plots...")
    html_content = generate_full_html_report(report_obj, energy_tracker=energy_tracker)

    # ---- Also build JSON summary ----
    analysis_json_path = _resolve_output_path(manifest, manifest.outputs.analysis_json)
    analysis_html_path = _resolve_output_path(manifest, manifest.outputs.analysis_html)

    structure_metrics = (
        {metric.label: metric.value for metric in cycle.structure_report.metrics}
        if cycle.structure_report is not None
        else {}
    )
    fidelity_metrics = (
        {
            metric.label: {
                "baseline_error": metric.baseline_error,
                "corrected_error": metric.corrected_error,
            }
            for metric in cycle.fidelity_report.metrics
        }
        if cycle.fidelity_report is not None
        else {}
    )
    chemistry_metrics = (
        {
            "title": cycle.chemistry_report.title,
            "evaluated_pair_count": cycle.chemistry_report.evaluated_pair_count,
            "favorable_pair_fraction": cycle.chemistry_report.favorable_pair_fraction,
            "mean_pair_score": cycle.chemistry_report.mean_pair_score,
            "charge_complementarity": cycle.chemistry_report.charge_complementarity,
            "hydropathy_alignment": cycle.chemistry_report.hydropathy_alignment,
            "flexibility_pressure": cycle.chemistry_report.flexibility_pressure,
            "hotspot_pair_fraction": cycle.chemistry_report.hotspot_pair_fraction,
        }
        if cycle.chemistry_report is not None
        else {}
    )
    json_report = {
        "title": f"NeuroCGMD Analysis | {manifest.system.name}",
        "summary": "Full trajectory analysis with molecular observables and SVG plots.",
        "headline_metrics": {
            "stage_label": cycle.progress.stage_label,
            "assembly_score": cycle.progress.assembly_score,
            "final_action": cycle.final_decision.highest_priority_action().kind.value,
            "state_id": str(cycle.state_id),
        },
        "structure_metrics": structure_metrics,
        "fidelity_metrics": fidelity_metrics,
        "chemistry_metrics": chemistry_metrics,
        "molecular_observables": mol_summary,
        "rmsd_summary": {k: v for k, v in rmsd_summary.items() if k != "rmsf_per_particle"},
        "rdf_summary": rdf_summary,
        "pmf_computed": len(pmf_vals) > 0,
        "transition_states_detected": len(detected_transitions),
        "binding_energy": binding_energy,
        "qcloud_event_analysis": _load_qcloud_event_data(manifest),
        "cycle_metadata": cycle.metadata.to_dict(),
        "checkpoint_metadata": checkpoint_payload.get("metadata", {}),
    }
    analysis_json_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_json_path.write_text(json.dumps(json_report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    analysis_html_path.write_text(html_content, encoding="utf-8")
    print(f"Wrote analysis report to {analysis_html_path}")
    print(f"Wrote analysis payload to {analysis_json_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manifest-driven NeuroCGMD workflow CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Build a prepared bundle from a TOML manifest.")
    prepare_parser.add_argument("--config", required=True, help="Path to the TOML run manifest.")
    prepare_parser.set_defaults(func=_run_prepare)

    run_parser = subparsers.add_parser("run", help="Run staged MD from a TOML manifest.")
    run_parser.add_argument("--config", required=True, help="Path to the TOML run manifest.")
    run_parser.add_argument(
        "--prepared-bundle",
        help="Optional explicit prepared bundle path. Defaults to the manifest output location.",
    )
    run_parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=0,
        help="Observer-side foundation benchmark repeats to run after the production trajectory.",
    )
    run_parser.set_defaults(func=_run_run)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a completed checkpoint.")
    analyze_parser.add_argument("--config", required=True, help="Path to the TOML run manifest.")
    analyze_parser.add_argument("--prepared-bundle", help="Optional explicit prepared bundle path.")
    analyze_parser.add_argument("--checkpoint", help="Optional explicit checkpoint path.")
    analyze_parser.set_defaults(func=_run_analyze)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
