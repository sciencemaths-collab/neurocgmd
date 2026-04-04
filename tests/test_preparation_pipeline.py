"""Tests for manifest-driven preparation and single-entity support."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import RunManifest
from core.state_registry import SimulationStateRegistry
from prepare import PreparationPipeline, infer_entity_groups
from sampling.scenarios import ImportedProteinComplexScenario, ImportedProteinScenarioSpec
from config import ProteinEntityGroup


class PreparationPipelineTests(unittest.TestCase):
    """Verify prepare-layer bundles connect cleanly to imported proteins."""

    def test_infer_entity_groups_uses_chain_ids_from_local_pdb(self) -> None:
        groups = infer_entity_groups(Path("benchmarks/reference_cases/data/1BRS.pdb"))

        self.assertGreaterEqual(len(groups), 2)
        self.assertEqual(groups[0].chain_ids, ("A",))

    def test_pipeline_builds_prepared_bundle_with_solvation_and_ions(self) -> None:
        manifest = RunManifest.from_dict(
            {
                "system": {
                    "name": "barnase_barstar_prepare",
                    "structure": "benchmarks/reference_cases/data/1BRS.pdb",
                    "entity_groups": [
                        {"entity_id": "barnase", "chain_ids": ["A"]},
                        {"entity_id": "barstar", "chain_ids": ["D"]},
                    ],
                },
                "solvent": {
                    "solvent_mode": "explicit",
                    "water_model": "tip3p",
                    "padding_nm": 1.0,
                },
                "ions": {
                    "neutralize": True,
                    "salt": "NaCl",
                    "ionic_strength_molar": 0.15,
                },
            }
        )
        pipeline = PreparationPipeline()

        bundle = pipeline.prepare(manifest)

        self.assertEqual(bundle.system_name, "barnase_barstar_prepare")
        self.assertGreater(bundle.import_summary.residue_count, 0)
        self.assertGreater(bundle.solvation_plan.estimated_water_molecules, 0)
        self.assertGreaterEqual(bundle.ion_plan.estimated_total_ions, 0)
        self.assertEqual(bundle.runtime_seed.cell.volume() > 0.0, True)

    def test_single_entity_imported_protein_scenario_now_builds(self) -> None:
        scenario = ImportedProteinComplexScenario(
            spec=ImportedProteinScenarioSpec(
                name="barnase_single_entity",
                pdb_path=str(Path("benchmarks/reference_cases/data/1BRS.pdb").resolve()),
                entity_groups=(
                    ProteinEntityGroup(entity_id="barnase", chain_ids=("A",)),
                ),
            )
        )

        setup = scenario.build_setup()
        registry = SimulationStateRegistry(created_by="single-entity-test")
        state = registry.create_initial_state(
            particles=setup.initial_particles,
            thermodynamics=setup.thermodynamics,
        )
        progress = scenario.measure_progress(state)
        self.assertEqual(setup.focus_compartments, ("barnase",))
        self.assertGreaterEqual(progress.assembly_score, 0.0)

    def test_prepared_bundle_round_trips_to_json(self) -> None:
        manifest = RunManifest.from_dict(
            {
                "system": {
                    "name": "roundtrip_prepare",
                    "structure": "benchmarks/reference_cases/data/1BRS.pdb",
                    "entity_groups": [
                        {"entity_id": "barnase", "chain_ids": ["A"]},
                        {"entity_id": "barstar", "chain_ids": ["D"]},
                    ],
                }
            }
        )
        pipeline = PreparationPipeline()
        bundle = pipeline.prepare(manifest)

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "prepared_bundle.json"
            pipeline.write_bundle(bundle, destination)
            loaded = pipeline.load_bundle(destination)

        self.assertEqual(loaded.system_name, bundle.system_name)
        self.assertEqual(loaded.import_summary.bead_count, bundle.import_summary.bead_count)
        self.assertEqual(loaded.runtime_seed.particles.particle_count, bundle.runtime_seed.particles.particle_count)


if __name__ == "__main__":
    unittest.main()
