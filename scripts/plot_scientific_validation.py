"""Generate plotted scientific-validation reports for live protein benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.state import EnsembleKind, ThermodynamicState
from core.state_registry import LifecycleStage
from integrators.langevin import LangevinIntegrator
from scripts.live_dashboard import (
    LiveDashboardContext,
    advance_demo,
    build_live_dashboard_context,
    collect_dashboard_step_data,
)
from validation import ScientificValidationRunner, ScientificValidationSample
from visualization import render_scientific_validation_report


def _configure_stochastic_validation_context(
    context: LiveDashboardContext,
    *,
    random_seed: int,
    target_temperature: float,
) -> LiveDashboardContext:
    """Promote one live-dashboard context into a reduced-unit stochastic validation run."""

    current_integrator = context.loop.integrator
    context.loop.integrator = LangevinIntegrator(
        time_step=current_integrator.time_step,
        friction_coefficient=current_integrator.friction_coefficient,
        stochastic=True,
        assume_reduced_units=True,
        thermal_energy_scale=1.0,
        random_seed=random_seed,
    )
    initial_state = context.registry.latest_state()
    if initial_state.thermodynamics.target_temperature is None:
        context.registry.derive_state(
            initial_state,
            thermodynamics=ThermodynamicState(
                ensemble=EnsembleKind.NVT,
                target_temperature=target_temperature,
                friction_coefficient=context.scenario_setup.integrator_friction,
            ),
            stage=LifecycleStage.CHECKPOINT,
            notes="scientific validation thermostat setup",
            metadata={"validation_seed": random_seed},
        )
    return context


def _sample_dashboard_context(
    context: LiveDashboardContext,
    *,
    replicate_index: int,
    sample_step: int,
    benchmark_repeats: int,
) -> ScientificValidationSample:
    """Collect one scientific-validation sample from the live benchmark flow."""

    step_data = collect_dashboard_step_data(context, benchmark_repeats=benchmark_repeats)
    structure_report = context.scenario.build_structure_report(
        step_data.state,
        progress=step_data.progress,
    )
    fidelity_report = context.scenario.build_fidelity_report(
        step_data.state,
        baseline_evaluation=step_data.base_evaluation,
        corrected_evaluation=step_data.qcloud_result.force_evaluation,
        progress=step_data.progress,
    )
    if structure_report is None or fidelity_report is None:
        raise RuntimeError(
            "The scientific validation plot currently requires both structure and fidelity reports."
        )
    return ScientificValidationSample.from_reports(
        replicate_index=replicate_index,
        sample_step=sample_step,
        state_step=step_data.state.step,
        time=step_data.state.time,
        progress=step_data.progress,
        structure_report=structure_report,
        fidelity_report=fidelity_report,
        benchmark_report=step_data.benchmark_report,
        metadata={
            "scenario": context.scenario.name,
            "state_id": str(step_data.state.provenance.state_id),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot repeated scientific validation for a live protein benchmark."
    )
    parser.add_argument(
        "--scenario",
        default="spike_ace2",
        choices=("spike_ace2", "barnase_barstar"),
        help="Live benchmark scenario to validate.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/neurocgmd_scientific_validation",
        help="Directory where the report HTML and JSON will be written.",
    )
    parser.add_argument("--replicates", type=int, default=3, help="Number of repeated trajectories to sample.")
    parser.add_argument("--steps", type=int, default=80, help="Simulation steps per replicate.")
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=4,
        help="Number of simulation steps between sampled validation points.",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=1,
        help="Foundation-benchmark repeat count used for each sampled architecture slice.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=4100,
        help="Base random seed used when enabling stochastic replicate sampling.",
    )
    parser.add_argument(
        "--target-temperature",
        type=float,
        default=1.0,
        help="Reduced-unit target temperature used by the stochastic validation thermostat.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable the stochastic validation thermostat and run deterministic replicate copies instead.",
    )
    parser.add_argument("--title", default="Scientific Validation Report", help="Optional report title override.")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = ScientificValidationRunner(
        replicates=args.replicates,
        steps_per_replicate=args.steps,
        sample_interval=args.sample_interval,
    )

    def _context_factory(replicate_index: int) -> LiveDashboardContext:
        context = build_live_dashboard_context(args.scenario)
        if args.deterministic:
            return context
        return _configure_stochastic_validation_context(
            context,
            random_seed=args.seed_base + replicate_index,
            target_temperature=args.target_temperature,
        )

    report = runner.run(
        scenario_name=args.scenario,
        context_factory=_context_factory,
        advance_context=lambda context, steps: advance_demo(context, steps=steps),
        sample_context=lambda context, replicate_index, sample_step: _sample_dashboard_context(
            context,
            replicate_index=replicate_index,
            sample_step=sample_step,
            benchmark_repeats=args.benchmark_repeats,
        ),
        metadata={
            "scenario": args.scenario,
            "deterministic": args.deterministic,
            "seed_base": args.seed_base,
            "target_temperature": args.target_temperature,
            "benchmark_repeats": args.benchmark_repeats,
        },
        title=args.title,
    )

    json_path = output_dir / "validation.json"
    html_path = output_dir / "index.html"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(render_scientific_validation_report(report), encoding="utf-8")

    print(f"Wrote scientific validation report to {html_path}")
    print(f"Wrote scientific validation payload to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
