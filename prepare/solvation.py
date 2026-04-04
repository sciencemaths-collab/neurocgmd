"""Preparation-time solvent and box planning."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, prod, sin, sqrt

from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationCell
from core.types import Vector3
from prepare.models import ExplicitAtomPlacement, ExplicitMoleculePlacement, SolvationPlan
from topology.protein_import_models import ImportedProteinSystem

_WATER_NUMBER_DENSITY_PER_NM3 = 33.4
_WATER_SITE_SPACING_NM = 0.31
_SOLUTE_EXCLUSION_RADIUS_NM = 0.24
_WATER_OH_DISTANCE_NM = 0.09572
_WATER_HOH_ANGLE_RAD = 1.824218134


@dataclass(slots=True)
class SolvationPlanner(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """[adapted] Build a bounded solvation and periodic-cell plan."""

    name: str = "solvation_planner"
    classification: str = "[adapted]"

    def describe_role(self) -> str:
        return (
            "Estimates the solvent box and water count needed for a prepared run while "
            "keeping the result explicitly separate from the later production dynamics."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return ("prepare/models.py", "core/state.py")

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/manifest_driven_md_workflow.md",)

    def validate(self) -> tuple[str, ...]:
        return ()

    def build_candidate_sites(
        self,
        plan: SolvationPlan,
        *,
        solute_positions: tuple[Vector3, ...],
        spacing_nm: float = _WATER_SITE_SPACING_NM,
        exclusion_radius_nm: float = _SOLUTE_EXCLUSION_RADIUS_NM,
    ) -> tuple[Vector3, ...]:
        """Build explicit solvent lattice sites inside the prepared box."""

        cell = plan.cell
        lengths = (
            cell.box_vectors[0][0],
            cell.box_vectors[1][1],
            cell.box_vectors[2][2],
        )
        counts = tuple(max(1, int(length / spacing_nm)) for length in lengths)
        exclusion_sq = exclusion_radius_nm * exclusion_radius_nm
        center = tuple(cell.origin[axis] + 0.5 * lengths[axis] for axis in range(3))
        candidates: list[tuple[float, Vector3]] = []
        for ix in range(counts[0]):
            for iy in range(counts[1]):
                for iz in range(counts[2]):
                    point = (
                        cell.origin[0] + (ix + 0.5) * spacing_nm,
                        cell.origin[1] + (iy + 0.5) * spacing_nm,
                        cell.origin[2] + (iz + 0.5) * spacing_nm,
                    )
                    if any(point[axis] >= cell.origin[axis] + lengths[axis] for axis in range(3)):
                        continue
                    if self._is_too_close(point, solute_positions, exclusion_sq):
                        continue
                    center_distance_sq = (
                        (point[0] - center[0]) ** 2
                        + (point[1] - center[1]) ** 2
                        + (point[2] - center[2]) ** 2
                    )
                    candidates.append((center_distance_sq, point))
        candidates.sort(key=lambda item: item[0])
        return tuple(point for _, point in candidates)

    def build_explicit_waters(
        self,
        plan: SolvationPlan,
        *,
        candidate_sites: tuple[Vector3, ...],
        occupied_indices: set[int] | None = None,
        requested_count: int | None = None,
    ) -> tuple[ExplicitMoleculePlacement, ...]:
        """Build explicit water coordinates from candidate solvent sites."""

        occupied = occupied_indices or set()
        waters_to_place = plan.estimated_water_molecules if requested_count is None else max(0, requested_count)
        molecules: list[ExplicitMoleculePlacement] = []
        residue_sequence = 1
        for site_index, oxygen in enumerate(candidate_sites):
            if site_index in occupied:
                continue
            if len(molecules) >= waters_to_place:
                break
            h1, h2 = self._water_hydrogens(oxygen, len(molecules))
            molecules.append(
                ExplicitMoleculePlacement(
                    molecule_id=f"water_{residue_sequence}",
                    residue_name="HOH",
                    chain_id="W",
                    residue_sequence=residue_sequence,
                    atoms=(
                        ExplicitAtomPlacement(
                            atom_name="O",
                            element="O",
                            residue_name="HOH",
                            chain_id="W",
                            residue_sequence=residue_sequence,
                            coordinates=oxygen,
                            record_type="HETATM",
                            metadata={"source": "solvation_planner"},
                        ),
                        ExplicitAtomPlacement(
                            atom_name="H1",
                            element="H",
                            residue_name="HOH",
                            chain_id="W",
                            residue_sequence=residue_sequence,
                            coordinates=h1,
                            record_type="HETATM",
                            metadata={"source": "solvation_planner"},
                        ),
                        ExplicitAtomPlacement(
                            atom_name="H2",
                            element="H",
                            residue_name="HOH",
                            chain_id="W",
                            residue_sequence=residue_sequence,
                            coordinates=h2,
                            record_type="HETATM",
                            metadata={"source": "solvation_planner"},
                        ),
                    ),
                    metadata={"water_model": plan.water_model},
                )
            )
            residue_sequence += 1
        return tuple(molecules)

    def plan(
        self,
        imported_system: ImportedProteinSystem,
        *,
        mode: str,
        water_model: str,
        box_type: str,
        padding_nm: float,
    ) -> SolvationPlan:
        """Estimate one solvent and cell plan from the imported coarse coordinates."""

        positions = imported_system.particles.positions
        mins = tuple(min(position[axis] for position in positions) for axis in range(3))
        maxs = tuple(max(position[axis] for position in positions) for axis in range(3))
        lengths = tuple(max(0.2, maxs[axis] - mins[axis] + 2.0 * padding_nm) for axis in range(3))
        origin = tuple(mins[axis] - padding_nm for axis in range(3))
        cell = SimulationCell(
            box_vectors=(
                (lengths[0], 0.0, 0.0),
                (0.0, lengths[1], 0.0),
                (0.0, 0.0, lengths[2]),
            ),
            origin=origin,
        )
        box_volume_nm3 = prod(lengths)
        estimated_water = 0
        if mode == "explicit":
            estimated_water = max(0, int(round(box_volume_nm3 * _WATER_NUMBER_DENSITY_PER_NM3)))
        return SolvationPlan(
            mode=mode,
            water_model=water_model,
            box_type=box_type,
            padding_nm=padding_nm,
            cell=cell,
            box_volume_nm3=box_volume_nm3,
            estimated_water_molecules=estimated_water,
            metadata={
                "structure_id": imported_system.structure_id,
                "entity_count": len(imported_system.entity_groups),
            },
        )

    def _is_too_close(
        self,
        point: Vector3,
        solute_positions: tuple[Vector3, ...],
        exclusion_sq: float,
    ) -> bool:
        for solute in solute_positions:
            dx = point[0] - solute[0]
            dy = point[1] - solute[1]
            dz = point[2] - solute[2]
            if dx * dx + dy * dy + dz * dz < exclusion_sq:
                return True
        return False

    def _water_hydrogens(
        self,
        oxygen: Vector3,
        index: int,
    ) -> tuple[Vector3, Vector3]:
        azimuth = (index * 0.61803398875) % 1.0 * 2.0 * 3.141592653589793
        half_angle = 0.5 * _WATER_HOH_ANGLE_RAD
        h1 = (
            oxygen[0] + _WATER_OH_DISTANCE_NM * cos(azimuth + half_angle),
            oxygen[1] + _WATER_OH_DISTANCE_NM * sin(azimuth + half_angle),
            oxygen[2],
        )
        h2 = (
            oxygen[0] + _WATER_OH_DISTANCE_NM * cos(azimuth - half_angle),
            oxygen[1] + _WATER_OH_DISTANCE_NM * sin(azimuth - half_angle),
            oxygen[2] + 0.5 * _WATER_OH_DISTANCE_NM,
        )
        return h1, h2
