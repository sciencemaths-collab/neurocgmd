"""Heuristic protonation planning for manifest-driven protein preparation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import cos, pi, sin, sqrt

from chemistry import ProteinChemistryModel
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from prepare.models import ExplicitAtomPlacement, ProtonationPlan, ProtonationSitePlan
from topology.protein_import_models import ImportedResidueRecord

_ESTIMATED_HYDROGEN_TOTALS: dict[str, int] = {
    "ALA": 5,
    "ARG": 10,
    "ASN": 4,
    "ASP": 3,
    "CYS": 3,
    "GLN": 6,
    "GLU": 5,
    "GLY": 3,
    "HIS": 5,
    "ILE": 11,
    "LEU": 11,
    "LYS": 12,
    "MET": 9,
    "PHE": 9,
    "PRO": 7,
    "SER": 3,
    "THR": 5,
    "TRP": 10,
    "TYR": 9,
    "VAL": 9,
}

_HYDROGEN_RADIUS_NM = 0.11


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
    if norm <= 1.0e-12:
        return (1.0, 0.0, 0.0)
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _hydrogen_direction(index: int, total: int) -> tuple[float, float, float]:
    polar_fraction = (index + 0.5) / max(1, total)
    polar = pi * polar_fraction
    azimuth = (2.0 * pi * (index + 1)) / max(1, total + 1)
    return _normalize(
        (
            sin(polar) * cos(azimuth),
            sin(polar) * sin(azimuth),
            cos(polar),
        )
    )


@dataclass(slots=True)
class ProtonationPlanner(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """[adapted] Residue-semantics-based protonation and hydrogen planner."""

    chemistry_model: ProteinChemistryModel = field(default_factory=ProteinChemistryModel)
    name: str = "protonation_planner"
    classification: str = "[adapted]"

    def describe_role(self) -> str:
        return (
            "Creates an explicit protonation and hydrogen-addition plan from residue semantics "
            "without pretending to be a full atomistic protonation backend."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return ("chemistry/residue_semantics.py", "prepare/models.py")

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/manifest_driven_md_workflow.md",)

    def validate(self) -> tuple[str, ...]:
        return self.chemistry_model.validate()

    def present_hydrogen_counts(self, structure: object) -> dict[tuple[str, int], int]:
        """Count hydrogen atoms already present in a parsed structure."""

        counts: defaultdict[tuple[str, int], int] = defaultdict(int)
        for atom in structure.atoms:
            atom_name = atom.atom_name.strip().upper()
            element = atom.element.strip().upper()
            if atom.record_type != "ATOM":
                continue
            if element == "H" or atom_name.startswith("H"):
                counts[(atom.chain_id, atom.residue_sequence)] += 1
        return dict(counts)

    def plan(
        self,
        residues: tuple[ImportedResidueRecord, ...],
        *,
        ph: float,
        add_hydrogens: bool,
        present_hydrogen_counts: dict[tuple[str, int], int] | None = None,
    ) -> ProtonationPlan:
        """Create one bounded protonation plan from imported residues."""

        hydrogen_counts = defaultdict(int, present_hydrogen_counts or {})
        site_plans: list[ProtonationSitePlan] = []
        net_charge = 0.0
        total_present = 0
        total_to_add = 0
        for residue in residues:
            descriptor = self.chemistry_model.descriptor_for_residue_name(
                residue.residue_name,
                label=f"{residue.chain_id}:{residue.residue_sequence}",
                metadata={"chain_id": residue.chain_id},
            )
            protonation_state = self._protonation_state_for(residue.residue_name, ph, descriptor.formal_charge)
            present_count = int(hydrogen_counts[(residue.chain_id, residue.residue_sequence)])
            estimated_total = _ESTIMATED_HYDROGEN_TOTALS.get(residue.residue_name.upper(), 6)
            estimated_to_add = max(0, estimated_total - present_count) if add_hydrogens else 0
            site_plans.append(
                ProtonationSitePlan(
                    chain_id=residue.chain_id,
                    residue_sequence=residue.residue_sequence,
                    residue_name=residue.residue_name,
                    protonation_state=protonation_state,
                    formal_charge=descriptor.formal_charge,
                    present_hydrogen_count=present_count,
                    estimated_total_hydrogen_count=estimated_total,
                    estimated_hydrogens_to_add=estimated_to_add,
                    metadata=descriptor.metadata.with_updates(
                        {
                            "descriptor_source": descriptor.descriptor_source,
                            "hydropathy": descriptor.hydropathy,
                        }
                    ),
                )
            )
            total_present += present_count
            total_to_add += estimated_to_add
            net_charge += descriptor.formal_charge
        return ProtonationPlan(
            method="heuristic_residue_semantics",
            ph=ph,
            add_hydrogens=add_hydrogens,
            estimated_net_charge=net_charge,
            total_present_hydrogen_atoms=total_present,
            total_estimated_hydrogens_to_add=total_to_add,
            sites=tuple(site_plans),
            metadata={"residue_count": len(residues)},
        )

    def build_explicit_hydrogens(
        self,
        residues: tuple[ImportedResidueRecord, ...],
        *,
        structure: object,
        protonation_plan: ProtonationPlan,
        coordinate_scale: float,
    ) -> tuple[ExplicitAtomPlacement, ...]:
        """Build explicit missing hydrogen coordinates in nanometer units."""

        residue_lookup = {
            (residue.chain_id, residue.residue_sequence): residue
            for residue in residues
        }
        heavy_atoms_by_residue: dict[tuple[str, int], list[object]] = defaultdict(list)
        for atom in structure.atoms:
            if atom.record_type != "ATOM":
                continue
            atom_name = atom.atom_name.strip().upper()
            element = atom.element.strip().upper()
            if element == "H" or atom_name.startswith("H"):
                continue
            heavy_atoms_by_residue[(atom.chain_id, atom.residue_sequence)].append(atom)

        placements: list[ExplicitAtomPlacement] = []
        for site in protonation_plan.sites:
            residue_key = (site.chain_id, site.residue_sequence)
            missing_count = site.estimated_hydrogens_to_add
            if missing_count <= 0:
                continue
            heavy_atoms = heavy_atoms_by_residue.get(residue_key, [])
            residue = residue_lookup[residue_key]
            if heavy_atoms:
                centroid = (
                    sum(atom.coordinates[0] for atom in heavy_atoms) / len(heavy_atoms),
                    sum(atom.coordinates[1] for atom in heavy_atoms) / len(heavy_atoms),
                    sum(atom.coordinates[2] for atom in heavy_atoms) / len(heavy_atoms),
                )
                anchor = (
                    centroid[0] * coordinate_scale,
                    centroid[1] * coordinate_scale,
                    centroid[2] * coordinate_scale,
                )
            else:
                anchor = residue.centroid
            for hydrogen_index in range(missing_count):
                direction = _hydrogen_direction(hydrogen_index, missing_count)
                coordinates = (
                    anchor[0] + _HYDROGEN_RADIUS_NM * direction[0],
                    anchor[1] + _HYDROGEN_RADIUS_NM * direction[1],
                    anchor[2] + _HYDROGEN_RADIUS_NM * direction[2],
                )
                placements.append(
                    ExplicitAtomPlacement(
                        atom_name=f"H{hydrogen_index + 1:02d}",
                        element="H",
                        residue_name=site.residue_name,
                        chain_id=site.chain_id,
                        residue_sequence=site.residue_sequence,
                        coordinates=coordinates,
                        record_type="ATOM",
                        metadata={
                            "source": "protonation_planner",
                            "protonation_state": site.protonation_state,
                            "formal_charge": site.formal_charge,
                        },
                    )
                )
        return tuple(placements)

    def _protonation_state_for(self, residue_name: str, ph: float, formal_charge: float) -> str:
        residue_key = residue_name.upper()
        if residue_key in {"ASP", "GLU"}:
            return "deprotonated" if ph >= 4.5 else "protonated"
        if residue_key in {"LYS", "ARG"}:
            return "protonated" if ph <= 10.5 else "deprotonated"
        if residue_key == "HIS":
            return "partially_protonated" if 5.5 <= ph <= 7.5 else ("protonated" if ph < 5.5 else "neutral")
        if residue_key == "CYS":
            return "thiol" if ph < 8.5 else "thiolate"
        if formal_charge > 0.25:
            return "cationic"
        if formal_charge < -0.25:
            return "anionic"
        return "neutral"
