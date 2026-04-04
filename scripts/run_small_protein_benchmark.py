"""Run a fast benchmark on a small real protein and write a JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.small_protein import SmallProteinBenchmarkRunner, SmallProteinBenchmarkSpec
from config import ProteinEntityGroup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a fast small-protein benchmark on a real local PDB chain.",
    )
    parser.add_argument(
        "--pdb",
        default=str((REPO_ROOT / "benchmarks" / "reference_cases" / "data" / "1BRS.pdb").resolve()),
        help="Path to the local protein PDB file.",
    )
    parser.add_argument("--chain", default="A", help="Chain identifier to benchmark.")
    parser.add_argument("--entity-id", default="barnase", help="Semantic entity identifier for the chain.")
    parser.add_argument("--name", default="barnase_small_protein_benchmark", help="Benchmark name.")
    parser.add_argument("--residues-per-bead", type=int, default=10, help="Residues per coarse bead.")
    parser.add_argument("--repeats", type=int, default=20, help="Timing repeats for each case.")
    parser.add_argument("--rollout-steps", type=int, default=4, help="Reserved short-rollout depth in the benchmark spec.")
    parser.add_argument("--warmup-training-passes", type=int, default=6, help="How many qcloud-derived warmup updates to apply to the ML residual.")
    parser.add_argument("--manual-region-size", type=int, default=6, help="How many center beads to include in the shadow benchmark region.")
    parser.add_argument("--output", required=True, help="Path to the JSON benchmark report.")
    args = parser.parse_args(argv)

    spec = SmallProteinBenchmarkSpec(
        name=args.name,
        pdb_path=str(Path(args.pdb).expanduser().resolve()),
        entity_group=ProteinEntityGroup(
            entity_id=args.entity_id,
            chain_ids=(args.chain,),
            description="CLI small-protein benchmark entity.",
        ),
        residues_per_bead=args.residues_per_bead,
        repeats=args.repeats,
        rollout_steps=args.rollout_steps,
        warmup_training_passes=args.warmup_training_passes,
        manual_region_size=args.manual_region_size,
    )
    report = SmallProteinBenchmarkRunner(spec=spec).run()

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote small-protein benchmark report to {output_path}")
    print(
        f"Benchmarked {report.entity_id} from {report.structure_id} with "
        f"{report.residue_count} residues mapped into {report.bead_count} coarse beads."
    )
    engine_modes = report.engine_mode_summary()
    print("Engine modes:")
    print(
        f"  classical_only: "
        f"{engine_modes['classical_only']['average_seconds_per_iteration']:.6f} s/iter"
    )
    print(
        f"  hybrid_production (production_hybrid_engine): "
        f"{engine_modes['hybrid_production']['average_seconds_per_iteration']:.6f} s/iter"
    )
    print("Diagnostics:")
    for case in report.benchmark_report.cases:
        print(
            f"  {case.name}: {case.average_seconds_per_iteration():.6f} s/iter "
            f"over {case.iterations} repeats"
        )
    print(
        f"Backend parity passed: {report.parity_report.all_passed()} | "
        f"backend={report.execution_plan.selection.selected_backend} | "
        f"mode={report.execution_plan.execution_mode}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
