"""O(N) cell-list neighbor finding for the coarse-grained molecular dynamics engine."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, sqrt

from core.exceptions import ContractValidationError
from core.state import SimulationState
from core.types import FrozenMetadata, Vector3, VectorTuple
from forcefields.base_forcefield import BaseForceField
from physics.forces.nonbonded_forces import NonbondedForceReport
from topology.system_topology import SystemTopology


# ---------------------------------------------------------------------------
# Cell list: spatial hash of particles into uniform grid cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CellList:
    """Spatial hash that maps particles into uniform grid cells."""

    cell_dimensions: tuple[int, int, int]
    cell_size: float
    particle_cells: dict[int, int]
    cell_particles: dict[int, tuple[int, ...]]


def _build_cell_list(
    positions: VectorTuple,
    particle_count: int,
    cell_size: float,
) -> CellList:
    """Partition *particle_count* positions into a uniform grid of side *cell_size*.

    The grid is built from the axis-aligned bounding box of the positions,
    expanded by a small epsilon so that particles sitting exactly on the upper
    boundary are included.
    """
    if particle_count == 0:
        return CellList(
            cell_dimensions=(0, 0, 0),
            cell_size=cell_size,
            particle_cells={},
            cell_particles={},
        )

    # Compute bounding box.
    min_coords = [positions[0][axis] for axis in range(3)]
    max_coords = [positions[0][axis] for axis in range(3)]
    for idx in range(1, particle_count):
        pos = positions[idx]
        for axis in range(3):
            if pos[axis] < min_coords[axis]:
                min_coords[axis] = pos[axis]
            if pos[axis] > max_coords[axis]:
                max_coords[axis] = pos[axis]

    # Grid dimensions – at least 1 cell per axis.
    dims: list[int] = []
    for axis in range(3):
        extent = max_coords[axis] - min_coords[axis]
        n = max(1, int(floor(extent / cell_size)) + 1)
        dims.append(n)

    nx, ny, nz = dims[0], dims[1], dims[2]

    particle_cells: dict[int, int] = {}
    cell_particles_map: dict[int, list[int]] = {}

    for idx in range(particle_count):
        pos = positions[idx]
        cx = min(int(floor((pos[0] - min_coords[0]) / cell_size)), nx - 1)
        cy = min(int(floor((pos[1] - min_coords[1]) / cell_size)), ny - 1)
        cz = min(int(floor((pos[2] - min_coords[2]) / cell_size)), nz - 1)
        cell_index = cx * ny * nz + cy * nz + cz
        particle_cells[idx] = cell_index
        if cell_index not in cell_particles_map:
            cell_particles_map[cell_index] = []
        cell_particles_map[cell_index].append(idx)

    cell_particles: dict[int, tuple[int, ...]] = {
        k: tuple(v) for k, v in cell_particles_map.items()
    }

    return CellList(
        cell_dimensions=(nx, ny, nz),
        cell_size=cell_size,
        particle_cells=particle_cells,
        cell_particles=cell_particles,
    )


# ---------------------------------------------------------------------------
# Neighbor list: pairs within cutoff + skin distance
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NeighborList:
    """Immutable set of particle pairs within the interaction cutoff."""

    pairs: tuple[tuple[int, int], ...]
    distances: tuple[float, ...]
    cutoff: float
    metadata: FrozenMetadata
    skin: float


# ---------------------------------------------------------------------------
# Builder – constructs the cell list and extracts neighbor pairs
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class NeighborListBuilder:
    """Build a cell-list based neighbor list in O(N) time."""

    cutoff: float
    skin: float = 0.3
    name: str = "cell_list_neighbor_builder"
    classification: str = "[established]"

    def build(
        self,
        positions: VectorTuple,
        particle_count: int,
        *,
        excluded_pairs: frozenset[tuple[int, int]] | None = None,
    ) -> NeighborList:
        """Return a :class:`NeighborList` for the given positions."""
        effective_cutoff = self.cutoff + self.skin
        cell_size = effective_cutoff

        cell_list = _build_cell_list(positions, particle_count, cell_size)

        nx, ny, nz = cell_list.cell_dimensions
        cutoff_sq = effective_cutoff * effective_cutoff
        excluded = excluded_pairs if excluded_pairs is not None else frozenset()

        pairs: list[tuple[int, int]] = []
        distances: list[float] = []

        # Stencil offsets for the 26 neighbours plus the cell itself.
        stencil = _NEIGHBOR_STENCIL

        visited_cell_pairs: set[tuple[int, int]] = set()

        for cell_index, particles_a in cell_list.cell_particles.items():
            # Decode cell indices.
            cx = cell_index // (ny * nz)
            remainder = cell_index % (ny * nz)
            cy = remainder // nz
            cz = remainder % nz

            for dx, dy, dz in stencil:
                ncx = cx + dx
                ncy = cy + dy
                ncz = cz + dz

                # Non-periodic: skip out-of-range neighbours.
                if ncx < 0 or ncx >= nx or ncy < 0 or ncy >= ny or ncz < 0 or ncz >= nz:
                    continue

                neighbor_index = ncx * ny * nz + ncy * nz + ncz
                particles_b = cell_list.cell_particles.get(neighbor_index)
                if particles_b is None:
                    continue

                # Avoid double-counting cell pairs (except self-pairs handled below).
                if neighbor_index != cell_index:
                    pair_key = (min(cell_index, neighbor_index), max(cell_index, neighbor_index))
                    if pair_key in visited_cell_pairs:
                        continue
                    visited_cell_pairs.add(pair_key)

                same_cell = neighbor_index == cell_index

                for idx_a in particles_a:
                    pos_a = positions[idx_a]
                    for idx_b in particles_b:
                        if same_cell and idx_b <= idx_a:
                            continue
                        if not same_cell and idx_b < idx_a:
                            # Canonical ordering: smaller index first.
                            a, b = idx_b, idx_a
                        else:
                            a, b = idx_a, idx_b

                        if (a, b) in excluded:
                            continue

                        pos_b = positions[idx_b]
                        dx2 = pos_b[0] - pos_a[0]
                        dy2 = pos_b[1] - pos_a[1]
                        dz2 = pos_b[2] - pos_a[2]
                        dist_sq = dx2 * dx2 + dy2 * dy2 + dz2 * dz2

                        if dist_sq <= cutoff_sq and dist_sq > 0.0:
                            pairs.append((a, b))
                            distances.append(sqrt(dist_sq))

        return NeighborList(
            pairs=tuple(pairs),
            distances=tuple(distances),
            cutoff=self.cutoff,
            metadata=FrozenMetadata(
                {
                    "builder": self.name,
                    "effective_cutoff": self.cutoff + self.skin,
                    "pair_count": len(pairs),
                }
            ),
            skin=self.skin,
        )

    def needs_rebuild(
        self,
        positions: VectorTuple,
        last_positions: VectorTuple,
    ) -> bool:
        """Return ``True`` if any particle has moved more than *skin / 2*."""
        threshold_sq = (self.skin / 2.0) ** 2
        for pos, last in zip(positions, last_positions):
            dx = pos[0] - last[0]
            dy = pos[1] - last[1]
            dz = pos[2] - last[2]
            if dx * dx + dy * dy + dz * dz > threshold_sq:
                return True
        return False


# 27-cell stencil (self + 26 neighbours).
_NEIGHBOR_STENCIL: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
)


# ---------------------------------------------------------------------------
# Accelerated nonbonded force model using the neighbor list
# ---------------------------------------------------------------------------


class AcceleratedNonbondedForceModel:
    """Lennard-Jones nonbonded force evaluation accelerated by cell-list neighbor finding.

    Wraps the same LJ pair-force formula as :class:`LennardJonesNonbondedForceModel`
    but iterates only over neighbor-list pairs instead of all O(N^2) pairs.
    """

    name = "accelerated_nonbonded_force"
    classification = "[adapted]"

    def __init__(
        self,
        builder: NeighborListBuilder,
        *,
        exclude_bonded_pairs: bool = True,
    ) -> None:
        self.builder = builder
        self.exclude_bonded_pairs = exclude_bonded_pairs
        self._cached_neighbor_list: NeighborList | None = None
        self._cached_positions: VectorTuple | None = None

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
    ) -> NonbondedForceReport:
        """Evaluate LJ nonbonded forces using the cell-list neighbor list."""
        issues = topology.validate_against_particle_state(state.particles)
        if issues:
            raise ContractValidationError("; ".join(issues))

        positions = state.particles.positions
        particle_count = state.particle_count

        excluded_pairs: frozenset[tuple[int, int]] = (
            frozenset(bond.normalized_pair() for bond in topology.bonds)
            if self.exclude_bonded_pairs
            else frozenset()
        )

        # Decide whether the cached neighbor list can be reused.
        rebuild = True
        if (
            self._cached_neighbor_list is not None
            and self._cached_positions is not None
        ):
            rebuild = self.builder.needs_rebuild(positions, self._cached_positions)

        if rebuild:
            neighbor_list = self.builder.build(
                positions,
                particle_count,
                excluded_pairs=excluded_pairs,
            )
            self._cached_neighbor_list = neighbor_list
            self._cached_positions = positions
        else:
            neighbor_list = self._cached_neighbor_list  # type: ignore[assignment]

        forces = [[0.0, 0.0, 0.0] for _ in range(particle_count)]
        evaluated_pairs: list[tuple[int, int]] = []

        for pair_idx in range(len(neighbor_list.pairs)):
            index_a, index_b = neighbor_list.pairs[pair_idx]
            parameter = forcefield.nonbonded_parameter_for_pair(
                topology, index_a, index_b
            )

            pos_a = positions[index_a]
            pos_b = positions[index_b]

            dx = pos_b[0] - pos_a[0]
            dy = pos_b[1] - pos_a[1]
            dz = pos_b[2] - pos_a[2]
            dist_sq = dx * dx + dy * dy + dz * dz

            if dist_sq == 0.0:
                raise ContractValidationError(
                    "Nonbonded pair has zero separation; Lennard-Jones force is undefined."
                )

            distance = sqrt(dist_sq)
            if distance > parameter.cutoff:
                continue

            sd = parameter.sigma / distance
            sd6 = sd ** 6
            sd12 = sd6 * sd6
            coefficient = 24.0 * parameter.epsilon * (2.0 * sd12 - sd6) / dist_sq

            fx = coefficient * dx
            fy = coefficient * dy
            fz = coefficient * dz

            forces[index_a][0] += fx
            forces[index_a][1] += fy
            forces[index_a][2] += fz
            forces[index_b][0] -= fx
            forces[index_b][1] -= fy
            forces[index_b][2] -= fz

            evaluated_pairs.append((index_a, index_b))

        return NonbondedForceReport(
            forces=tuple(tuple(v) for v in forces),
            evaluated_pairs=tuple(evaluated_pairs),
        )
