"""Run shared spatial-prior transfer tuning across multiple protein benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qcloud import ProteinShadowTuningPreset
from sampling.scenarios import ImportedProteinScenarioSpec
from validation.protein_transfer_tuning import (
    ProteinTransferTuningRunner,
    build_spatial_transfer_candidate_grid,
)


def _parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one numeric value is required")
    return values


def _parse_str_list(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def _load_imported_spec(path_text: str) -> ImportedProteinScenarioSpec:
    resolved_path = Path(path_text).expanduser().resolve()
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    spec_payload = payload["scenario_spec"] if isinstance(payload, dict) and "scenario_spec" in payload else payload
    return ImportedProteinScenarioSpec.from_dict(spec_payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tune shared spatial shadow priors across spike_ace2 and barnase_barstar."
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/neurocgmd_spatial_transfer_tuning",
        help="Directory where the JSON tuning report will be written.",
    )
    parser.add_argument(
        "--scenarios",
        type=_parse_str_list,
        default=("spike_ace2", "barnase_barstar"),
        help="Comma-separated benchmark scenarios to include in the transfer panel.",
    )
    parser.add_argument("--replicates", type=int, default=2, help="Replicates per scenario during tuning.")
    parser.add_argument("--steps", type=int, default=12, help="Simulation steps per replicate.")
    parser.add_argument("--sample-interval", type=int, default=4, help="Sampling interval per replicate.")
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=1,
        help="Foundation benchmark repeats used for each sampled state.",
    )
    parser.add_argument(
        "--preferred-distance-scales",
        type=_parse_float_list,
        default=(1.0,),
        help="Comma-separated spatial profile preferred-distance multipliers.",
    )
    parser.add_argument(
        "--distance-tolerance-scales",
        type=_parse_float_list,
        default=(1.0,),
        help="Comma-separated spatial profile distance-tolerance multipliers.",
    )
    parser.add_argument(
        "--attraction-scales",
        type=_parse_float_list,
        default=(0.95, 1.0, 1.05),
        help="Comma-separated spatial profile attraction scales.",
    )
    parser.add_argument(
        "--repulsion-scales",
        type=_parse_float_list,
        default=(0.95, 1.0, 1.05),
        help="Comma-separated spatial profile repulsion scales.",
    )
    parser.add_argument(
        "--directional-scales",
        type=_parse_float_list,
        default=(1.0,),
        help="Comma-separated spatial profile directional-emphasis scales.",
    )
    parser.add_argument(
        "--chemistry-scales",
        type=_parse_float_list,
        default=(0.90, 1.0, 1.10),
        help="Comma-separated spatial profile chemistry scales.",
    )
    parser.add_argument(
        "--local-field-energy-scales",
        type=_parse_float_list,
        default=(1.0,),
        help="Comma-separated multipliers for the local spatial-field energy scale.",
    )
    parser.add_argument(
        "--local-field-repulsion-scales",
        type=_parse_float_list,
        default=(1.0,),
        help="Comma-separated multipliers for the local spatial-field repulsion scale.",
    )
    parser.add_argument(
        "--alignment-floors",
        type=_parse_float_list,
        default=(0.24, 0.28, 0.32),
        help="Comma-separated spatial alignment floors.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top candidates to summarize in stdout.",
    )
    parser.add_argument(
        "--exclude-baseline",
        action="store_true",
        help="Do not include the unmodified preset as an explicit baseline candidate.",
    )
    parser.add_argument(
        "--import-spec",
        action="append",
        default=[],
        help="Optional JSON import summary/spec files created by import_protein_system.py.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_spatial_transfer_candidate_grid(
        ProteinShadowTuningPreset(),
        include_baseline=not args.exclude_baseline,
        preferred_distance_scales=args.preferred_distance_scales,
        distance_tolerance_scales=args.distance_tolerance_scales,
        attraction_scales=args.attraction_scales,
        repulsion_scales=args.repulsion_scales,
        directional_scales=args.directional_scales,
        chemistry_scales=args.chemistry_scales,
        local_field_energy_scales=args.local_field_energy_scales,
        local_field_repulsion_scales=args.local_field_repulsion_scales,
        alignment_floors=args.alignment_floors,
    )
    imported_specs = tuple(_load_imported_spec(path_text) for path_text in args.import_spec)
    runner = ProteinTransferTuningRunner(
        replicates=args.replicates,
        steps_per_replicate=args.steps,
        sample_interval=args.sample_interval,
        benchmark_repeats=args.benchmark_repeats,
        scenario_names=args.scenarios,
        imported_scenario_specs=imported_specs,
    )
    report = runner.tune(
        candidates,
        metadata={
            "scenarios": args.scenarios,
            "imported_scenarios": tuple(spec.name for spec in imported_specs),
            "preferred_distance_scales": args.preferred_distance_scales,
            "distance_tolerance_scales": args.distance_tolerance_scales,
            "attraction_scales": args.attraction_scales,
            "repulsion_scales": args.repulsion_scales,
            "directional_scales": args.directional_scales,
            "chemistry_scales": args.chemistry_scales,
            "local_field_energy_scales": args.local_field_energy_scales,
            "local_field_repulsion_scales": args.local_field_repulsion_scales,
            "alignment_floors": args.alignment_floors,
        },
    )

    payload_path = output_dir / "transfer_tuning.json"
    payload_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    best = report.best_candidate()
    baseline = report.baseline_candidate()

    print(f"Wrote transfer-tuning report to {payload_path}")
    print(
        "Best candidate:",
        best.candidate.name,
        f"(mean_score={best.mean_score:.4f})",
    )
    if baseline is not None:
        print(
            "Baseline candidate:",
            baseline.candidate.name,
            f"(mean_score={baseline.mean_score:.4f}, delta={report.score_margin_over_baseline():+.4f})",
        )
    for scenario_score in best.scenario_scores:
        print(
            f"  {scenario_score.scenario_name}: score={scenario_score.combined_score:.4f}, "
            f"force_rms={scenario_score.summary.force_rms_improvement_rate:.3f}, "
            f"contact={scenario_score.summary.final_contact_recovery_mean:.3f}, "
            f"rmsd={scenario_score.summary.final_atomistic_centroid_rmsd_mean:.3f}",
        )
    print("Top candidates:")
    for candidate in report.top_candidates(limit=args.top_k):
        print(
            f"  {candidate.candidate.name}: mean_score={candidate.mean_score:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
