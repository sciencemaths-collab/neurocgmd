"""Tests for the TOML-backed user-facing run manifest."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import RunManifest, SolventMode, load_run_manifest
from core.state import EnsembleKind


class RunManifestTests(unittest.TestCase):
    """Keep the user-facing TOML control surface honest."""

    def test_load_manifest_supports_mature_md_style_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "run.toml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "[system]",
                        'name = "barnase_barstar_cli"',
                        'structure = "benchmarks/reference_cases/data/1BRS.pdb"',
                        "",
                        "[[system.entity_groups]]",
                        'entity_id = "barnase"',
                        'chain_ids = ["A"]',
                        "",
                        "[[system.entity_groups]]",
                        'entity_id = "barstar"',
                        'chain_ids = ["D"]',
                        "",
                        "[solvent]",
                        'solvent_mode = "explicit"',
                        'water_model = "tip3p"',
                        'box_type = "dodecahedron"',
                        "padding_nm = 1.2",
                        "",
                        "[ions]",
                        "neutralize = true",
                        'salt = "NaCl"',
                        "ionic_strength_molar = 0.2",
                        "",
                        "[constraints]",
                        'constraints = "h_bonds"',
                        'constraint_algorithm = "lincs"',
                        "lincs_order = 4",
                        "",
                        "[nonbonded]",
                        'cutoff_scheme = "Verlet"',
                        "vdw_cutoff_nm = 1.25",
                        "coulomb_cutoff_nm = 1.25",
                        "neighbor_skin_nm = 0.35",
                        "",
                        "[stages.nvt]",
                        "enabled = true",
                        'ensemble = "NVT"',
                        "dt = 0.004",
                        "nsteps = 20",
                        "temperature = 300.0",
                        "friction_coefficient = 0.8",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = load_run_manifest(manifest_path)

        self.assertIsInstance(manifest, RunManifest)
        self.assertEqual(manifest.system.name, "barnase_barstar_cli")
        self.assertEqual(manifest.prepare.solvent_mode, SolventMode.EXPLICIT)
        self.assertAlmostEqual(manifest.prepare.ionic_strength_molar, 0.2)
        self.assertEqual(manifest.forcefield.constraint_algorithm, "lincs")
        self.assertAlmostEqual(manifest.neighbor_list.vdw_cutoff_nm, 1.25)
        self.assertEqual(manifest.nvt.ensemble, EnsembleKind.NVT)
        self.assertEqual(len(manifest.system.entity_groups), 2)


if __name__ == "__main__":
    unittest.main()
