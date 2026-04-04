"""Generate a plotted small-protein benchmark report with JSON and HTML outputs."""

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
from visualization import render_small_protein_benchmark_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run and plot the fast small-protein benchmark.",
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
    parser.add_argument("--rollout-steps", type=int, default=4, help="Short hybrid rollout depth.")
    parser.add_argument("--warmup-training-passes", type=int, default=6, help="Warmup updates for the ML residual.")
    parser.add_argument("--manual-region-size", type=int, default=6, help="How many center beads to include in the shadow benchmark region.")
    parser.add_argument(
        "--output-dir",
        default="/tmp/neurocgmd_small_protein_benchmark_plot",
        help="Directory where benchmark JSON and HTML artifacts will be written.",
    )
    args = parser.parse_args(argv)

    spec = SmallProteinBenchmarkSpec(
        name=args.name,
        pdb_path=str(Path(args.pdb).expanduser().resolve()),
        entity_group=ProteinEntityGroup(
            entity_id=args.entity_id,
            chain_ids=(args.chain,),
            description="Plotted small-protein benchmark entity.",
        ),
        residues_per_bead=args.residues_per_bead,
        repeats=args.repeats,
        rollout_steps=args.rollout_steps,
        warmup_training_passes=args.warmup_training_passes,
        manual_region_size=args.manual_region_size,
    )
    report = SmallProteinBenchmarkRunner(spec=spec).run()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark.json"
    html_path = output_dir / "index.html"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(render_small_protein_benchmark_report(report), encoding="utf-8")

    print(f"Wrote small-protein benchmark JSON to {json_path}")
    print(f"Wrote small-protein benchmark HTML to {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
