"""Manifest-driven preparation pipeline for imported protein systems."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from benchmarks.reference_cases.structure_targets import load_local_pdb_loader
from config import ProteinEntityGroup, RunManifest
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import EnsembleKind, ThermodynamicState, UnitSystem
from prepare.ions import IonPlacementPlanner
from prepare.models import (
    ImportSummary,
    PreparationEntitySummary,
    PreparedRuntimeSeed,
    PreparedSystemBundle,
)
from prepare.protonation import ProtonationPlanner
from prepare.solvation import SolvationPlanner
from sampling.scenarios import ImportedProteinScenarioSpec
from topology import ProteinCoarseMapper
from topology.protein_import_models import ImportedProteinSystem


def infer_entity_groups(pdb_path: str | Path) -> tuple[ProteinEntityGroup, ...]:
    """Infer entity groups from the ATOM chain identifiers of a local PDB file."""

    pdb_loader = load_local_pdb_loader()
    structure = pdb_loader.load_pdb_file(pdb_path, structure_id=Path(pdb_path).stem)
    chain_ids = tuple(
        sorted(
            {
                atom.chain_id
                for atom in structure.atoms
                if atom.record_type == "ATOM" and atom.chain_id.strip()
            }
        )
    )
    if len(chain_ids) <= 1:
        return (
            ProteinEntityGroup(
                entity_id="protein",
                chain_ids=chain_ids or ("A",),
                description="Auto-inferred single protein entity.",
            ),
        )
    return tuple(
        ProteinEntityGroup(
            entity_id=f"chain_{chain_id.lower()}",
            chain_ids=(chain_id,),
            description=f"Auto-inferred entity for chain {chain_id}.",
        )
        for chain_id in chain_ids
    )


@dataclass(slots=True)
class PreparationPipeline(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """[hybrid] Build preparation bundles from user-facing run manifests."""

    mapper: ProteinCoarseMapper = field(default_factory=ProteinCoarseMapper)
    protonation_planner: ProtonationPlanner = field(default_factory=ProtonationPlanner)
    solvation_planner: SolvationPlanner = field(default_factory=SolvationPlanner)
    ion_planner: IonPlacementPlanner = field(default_factory=IonPlacementPlanner)
    name: str = "preparation_pipeline"
    classification: str = "[hybrid]"

    def describe_role(self) -> str:
        return (
            "Turns a TOML manifest into one explicit prepared-system bundle that can be "
            "handed directly into run and analyze workflows."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "config/run_manifest.py",
            "prepare/protonation.py",
            "prepare/solvation.py",
            "prepare/ions.py",
            "topology/protein_coarse_mapping.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/manifest_driven_md_workflow.md",)

    def validate(self) -> tuple[str, ...]:
        issues = list(self.mapper.validate())
        issues.extend(self.protonation_planner.validate())
        return tuple(issues)

    def prepare(self, manifest: RunManifest) -> PreparedSystemBundle:
        """Build one prepared-system bundle from a run manifest."""

        entity_groups = manifest.system.entity_groups or infer_entity_groups(manifest.system.structure)
        imported = self.mapper.import_from_pdb(
            pdb_path=manifest.system.structure,
            entity_groups=entity_groups,
            structure_id=manifest.system.name,
        )
        protonation_plan = self.protonation_planner.plan(
            imported.residues,
            ph=manifest.prepare.ph,
            add_hydrogens=manifest.prepare.add_hydrogens,
            present_hydrogen_counts={},
        )
        solvation_plan = self.solvation_planner.plan(
            imported,
            mode=manifest.prepare.solvent_mode.value,
            water_model=manifest.prepare.water_model,
            box_type=manifest.prepare.box_type,
            padding_nm=manifest.prepare.padding_nm,
        )
        ion_plan = self.ion_planner.plan(
            net_charge=protonation_plan.estimated_net_charge,
            box_volume_nm3=solvation_plan.box_volume_nm3,
            neutralize=manifest.prepare.neutralize,
            salt=manifest.prepare.salt,
            ionic_strength_molar=manifest.prepare.ionic_strength_molar,
        )
        runtime_seed = PreparedRuntimeSeed(
            units=UnitSystem.md_nano(),
            particles=imported.particles,
            thermodynamics=self._initial_thermodynamics(manifest),
            cell=solvation_plan.cell,
            metadata={
                "prepare_mode": manifest.prepare.solvent_mode.value,
                "forcefield": manifest.forcefield.protein,
            },
        )
        return PreparedSystemBundle(
            system_name=manifest.system.name,
            manifest_path=manifest.source_path or manifest.system.structure,
            structure_path=str(Path(manifest.system.structure).expanduser().resolve()),
            entity_groups=entity_groups,
            mapping_config=manifest.system.mapping_config,
            scenario_spec=ImportedProteinScenarioSpec(
                name=manifest.system.name,
                pdb_path=str(Path(manifest.system.structure).expanduser().resolve()),
                entity_groups=entity_groups,
                mapping_config=manifest.system.mapping_config,
            ),
            import_summary=self._import_summary(imported),
            protonation_plan=protonation_plan,
            solvation_plan=solvation_plan,
            ion_plan=ion_plan,
            runtime_seed=runtime_seed,
            metadata={
                "manifest_description": manifest.system.description,
                "solvent_mode": manifest.prepare.solvent_mode.value,
            },
        )

    def write_bundle(self, bundle: PreparedSystemBundle, path: str | Path) -> Path:
        """Serialize one prepared bundle to JSON."""

        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(bundle.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return destination

    def load_bundle(self, path: str | Path) -> PreparedSystemBundle:
        """Load one previously serialized prepared bundle."""

        resolved = Path(path).expanduser().resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        return PreparedSystemBundle.from_dict(payload)

    def _initial_thermodynamics(self, manifest: RunManifest) -> ThermodynamicState:
        for stage in (manifest.nvt, manifest.npt, manifest.production):
            if stage.enabled:
                return ThermodynamicState(
                    ensemble=stage.ensemble,
                    target_temperature=stage.temperature,
                    target_pressure=stage.pressure,
                    friction_coefficient=stage.friction_coefficient,
                )
        return ThermodynamicState(ensemble=EnsembleKind.NVE)

    def _import_summary(self, imported: ImportedProteinSystem) -> ImportSummary:
        entities = tuple(
            PreparationEntitySummary(
                entity_id=group.entity_id,
                chain_ids=group.chain_ids,
                residue_count=sum(
                    1 for residue in imported.residues if residue.metadata.get("entity_id") == group.entity_id
                ),
                bead_count=sum(1 for block in imported.bead_blocks if block.entity_id == group.entity_id),
            )
            for group in imported.entity_groups
        )
        return ImportSummary(
            structure_id=imported.structure_id,
            source_path=imported.source_path,
            residue_count=len(imported.residues),
            bead_count=len(imported.bead_blocks),
            particle_count=imported.particles.particle_count,
            entities=entities,
            metadata=imported.metadata,
        )
