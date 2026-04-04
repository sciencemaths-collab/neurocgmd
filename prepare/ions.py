"""Preparation-time ion planning and explicit coordinate placement."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import Vector3
from prepare.models import ExplicitAtomPlacement, ExplicitMoleculePlacement, IonPlacementPlan

_AVOGADRO = 6.02214076e23
_NM3_TO_LITERS = 1.0e-24


def _split_salt_name(salt: str) -> tuple[str, str]:
    normalized = salt.strip()
    if normalized.upper() == "NACL":
        return ("Na+", "Cl-")
    if normalized.upper() == "KCL":
        return ("K+", "Cl-")
    if normalized.upper() == "MGCL2":
        return ("Mg2+", "Cl-")
    return ("cation", "anion")


def _canonical_ion_identity(species: str) -> tuple[str, str, str]:
    normalized = species.strip()
    upper = normalized.upper()
    if upper == "NA+":
        return ("NA", "NA", "Na")
    if upper == "CL-":
        return ("CL", "CL", "Cl")
    if upper == "K+":
        return ("K", "K", "K")
    if upper == "MG2+":
        return ("MG", "MG", "Mg")
    symbol = "".join(character for character in normalized if character.isalpha()).upper() or "ION"
    return (symbol[:3], symbol[:4], symbol[:2].title())


@dataclass(slots=True)
class IonPlacementPlanner(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """[adapted] Estimate neutralization and bulk-salt ion counts."""

    name: str = "ion_placement_planner"
    classification: str = "[adapted]"

    def describe_role(self) -> str:
        return (
            "Converts preparation-time net charge and box volume into explicit ion-count "
            "estimates for neutralization and background salt."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return ("prepare/models.py",)

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/manifest_driven_md_workflow.md",)

    def validate(self) -> tuple[str, ...]:
        return ()

    def plan(
        self,
        *,
        net_charge: float,
        box_volume_nm3: float,
        neutralize: bool,
        salt: str,
        ionic_strength_molar: float,
    ) -> IonPlacementPlan:
        """Estimate ion counts for a prepared run."""

        cation_name, anion_name = _split_salt_name(salt)
        cations = 0
        anions = 0
        if neutralize:
            rounded_charge = int(round(net_charge))
            if rounded_charge > 0:
                anions += rounded_charge
            elif rounded_charge < 0:
                cations += abs(rounded_charge)

        salt_pairs = 0
        if ionic_strength_molar > 0.0 and box_volume_nm3 > 0.0:
            salt_pairs = int(
                ceil(ionic_strength_molar * box_volume_nm3 * _NM3_TO_LITERS * _AVOGADRO)
            )
        cations += salt_pairs
        anions += salt_pairs
        return IonPlacementPlan(
            neutralize=neutralize,
            salt=salt,
            ionic_strength_molar=ionic_strength_molar,
            cation_name=cation_name,
            anion_name=anion_name,
            estimated_cation_count=cations,
            estimated_anion_count=anions,
            estimated_total_ions=cations + anions,
            metadata={
                "net_charge": net_charge,
                "salt_pairs": salt_pairs,
                "box_volume_nm3": box_volume_nm3,
            },
        )

    def build_explicit_ions(
        self,
        plan: IonPlacementPlan,
        *,
        candidate_sites: tuple[Vector3, ...],
    ) -> tuple[tuple[ExplicitMoleculePlacement, ...], frozenset[int]]:
        """Build explicit ion placements on a deterministic subset of solvent sites."""

        total_requested = min(plan.estimated_total_ions, len(candidate_sites))
        if total_requested <= 0:
            return (), frozenset()
        selected_indices = self._spread_site_indices(total_requested, len(candidate_sites))
        species_sequence = self._species_sequence(plan, total_requested)
        molecules: list[ExplicitMoleculePlacement] = []
        occupied_indices: set[int] = set()
        for residue_sequence, (species, site_index) in enumerate(
            zip(species_sequence, selected_indices, strict=False),
            start=1,
        ):
            residue_name, atom_name, element = _canonical_ion_identity(species)
            coordinates = candidate_sites[site_index]
            occupied_indices.add(site_index)
            molecules.append(
                ExplicitMoleculePlacement(
                    molecule_id=f"ion_{residue_sequence}",
                    residue_name=residue_name,
                    chain_id="I",
                    residue_sequence=residue_sequence,
                    atoms=(
                        ExplicitAtomPlacement(
                            atom_name=atom_name,
                            element=element,
                            residue_name=residue_name,
                            chain_id="I",
                            residue_sequence=residue_sequence,
                            coordinates=coordinates,
                            record_type="HETATM",
                            metadata={
                                "source": "ion_placement_planner",
                                "species": species,
                            },
                        ),
                    ),
                    metadata={
                        "species": species,
                        "source": "ion_placement_planner",
                    },
                )
            )
        return tuple(molecules), frozenset(occupied_indices)

    def _species_sequence(
        self,
        plan: IonPlacementPlan,
        total_requested: int,
    ) -> tuple[str, ...]:
        species: list[str] = []
        cations_remaining = min(plan.estimated_cation_count, total_requested)
        anions_remaining = min(plan.estimated_anion_count, max(0, total_requested - cations_remaining))
        while len(species) < total_requested:
            if cations_remaining > 0:
                species.append(plan.cation_name)
                cations_remaining -= 1
                if len(species) >= total_requested:
                    break
            if anions_remaining > 0:
                species.append(plan.anion_name)
                anions_remaining -= 1
        while len(species) < total_requested:
            species.append(plan.cation_name if plan.estimated_cation_count >= plan.estimated_anion_count else plan.anion_name)
        return tuple(species)

    def _spread_site_indices(self, requested: int, total_sites: int) -> tuple[int, ...]:
        if requested <= 0 or total_sites <= 0:
            return ()
        if requested >= total_sites:
            return tuple(range(total_sites))
        selected: list[int] = []
        used: set[int] = set()
        for index in range(requested):
            target = int(round(((index + 0.5) * total_sites) / requested - 0.5))
            candidate = min(max(target, 0), total_sites - 1)
            while candidate in used and candidate < total_sites - 1:
                candidate += 1
            while candidate in used and candidate > 0:
                candidate -= 1
            used.add(candidate)
            selected.append(candidate)
        return tuple(selected)
