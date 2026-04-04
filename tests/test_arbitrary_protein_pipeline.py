"""Tests for the arbitrary-protein import pipeline and fresh-process CLI path."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from config import ProteinEntityGroup
from sampling.scenarios import ImportedProteinComplexScenario, ImportedProteinScenarioSpec
from topology import ProteinCoarseMapper


class ArbitraryProteinPipelineTests(unittest.TestCase):
    """Verify generic protein import works beyond hand-authored scenarios."""

    def test_mapper_imports_barnase_barstar_reference_complex(self) -> None:
        mapper = ProteinCoarseMapper()

        imported = mapper.import_from_pdb(
            pdb_path=Path("benchmarks/reference_cases/data/1BRS.pdb"),
            entity_groups=(
                ProteinEntityGroup(entity_id="barnase", chain_ids=("A",)),
                ProteinEntityGroup(entity_id="barstar", chain_ids=("D",)),
            ),
            structure_id="barnase_barstar_imported_test",
        )

        self.assertEqual(imported.entity_ids(), ("barnase", "barstar"))
        self.assertEqual(imported.particles.particle_count, len(imported.reference_target.landmarks))
        self.assertGreaterEqual(len(imported.bead_indices_for_entity("barnase")), 2)
        self.assertGreaterEqual(len(imported.bead_indices_for_entity("barstar")), 2)
        self.assertTrue(imported.reference_target.interface_contacts)

    def test_imported_protein_scenario_builds_setup_from_local_pdb(self) -> None:
        scenario = ImportedProteinComplexScenario(
            spec=ImportedProteinScenarioSpec(
                name="barnase_barstar_imported_scenario",
                pdb_path=str(Path("benchmarks/reference_cases/data/1BRS.pdb").resolve()),
                entity_groups=(
                    ProteinEntityGroup(entity_id="barnase", chain_ids=("A",)),
                    ProteinEntityGroup(entity_id="barstar", chain_ids=("D",)),
                ),
            )
        )

        setup = scenario.build_setup()

        self.assertEqual(setup.topology.system_id, "barnase_barstar_imported_scenario")
        self.assertEqual(setup.focus_compartments, ("barnase", "barstar"))
        self.assertGreater(setup.initial_particles.particle_count, 0)

    def test_import_cli_runs_in_fresh_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "imported.json"
            command = [
                sys.executable,
                "scripts/import_protein_system.py",
                "--pdb",
                "benchmarks/reference_cases/data/1BRS.pdb",
                "--entity",
                "barnase:A",
                "--entity",
                "barstar:D",
                "--name",
                "barnase_barstar_imported_cli",
                "--output",
                str(output_path),
            ]

            result = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("scenario_spec", payload)
            self.assertIn("imported_system", payload)
            self.assertEqual(payload["scenario_spec"]["name"], "barnase_barstar_imported_cli")


if __name__ == "__main__":
    unittest.main()
