"""Import an arbitrary local protein complex into the coarse repository substrate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import ProteinEntityGroup, ProteinMappingConfig
from sampling.scenarios import ImportedProteinScenarioSpec
from topology import ProteinCoarseMapper


def _parse_entity_group(text: str) -> ProteinEntityGroup:
    try:
        entity_id, chains_text = text.split(":", maxsplit=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "entity groups must look like entity_id:CHAIN[,CHAIN]"
        ) from exc
    chain_ids = tuple(chain_id.strip() for chain_id in chains_text.split(",") if chain_id.strip())
    if not entity_id.strip() or not chain_ids:
        raise argparse.ArgumentTypeError(
            "entity groups must look like entity_id:CHAIN[,CHAIN]"
        )
    return ProteinEntityGroup(entity_id=entity_id.strip(), chain_ids=chain_ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import an arbitrary local protein complex and write a reusable coarse summary.",
    )
    parser.add_argument("--pdb", required=True, help="Path to the local PDB file.")
    parser.add_argument(
        "--entity",
        type=_parse_entity_group,
        action="append",
        required=True,
        help="Repeatable entity definition: entity_id:CHAIN[,CHAIN]",
    )
    parser.add_argument("--name", required=True, help="Scenario/import name.")
    parser.add_argument(
        "--residues-per-bead",
        type=int,
        default=8,
        help="How many residues to group into one coarse bead block.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where the JSON import summary should be written.",
    )
    args = parser.parse_args(argv)

    mapping_config = ProteinMappingConfig(residues_per_bead=args.residues_per_bead)
    mapper = ProteinCoarseMapper(mapping_config=mapping_config)
    imported_system = mapper.import_from_pdb(
        pdb_path=args.pdb,
        entity_groups=tuple(args.entity),
        structure_id=args.name,
    )
    scenario_spec = ImportedProteinScenarioSpec(
        name=args.name,
        pdb_path=str(Path(args.pdb).expanduser().resolve()),
        entity_groups=tuple(args.entity),
        mapping_config=mapping_config,
    )
    payload = {
        "scenario_spec": scenario_spec.to_dict(),
        "imported_system": imported_system.to_dict(),
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote imported protein summary to {output_path}")
    print(
        f"Imported {imported_system.structure_id} with "
        f"{len(imported_system.entity_groups)} entities, "
        f"{len(imported_system.residues)} residues, and "
        f"{len(imported_system.bead_blocks)} coarse beads."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
