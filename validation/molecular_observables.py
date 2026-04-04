"""Molecular observable calculators for coarse-grained MD trajectory analysis.

Provides SASA, radius of gyration, hydrogen bond analysis, per-residue energy
decomposition, contact maps, secondary structure estimation, end-to-end distance,
and a unified collector that wires everything together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt, pi, acos, cos, sin

from core.exceptions import ContractValidationError
from core.state import SimulationState, SimulationCell
from core.types import FrozenMetadata, Vector3, VectorTuple
from topology.system_topology import SystemTopology

# ---------------------------------------------------------------------------
# Default CG bead radii (nm) for SASA calculations
# ---------------------------------------------------------------------------

DEFAULT_CG_RADII: dict[str, float] = {
    "BB": 0.24,
    "P5": 0.26,
    "Nda": 0.24,
    "C1": 0.26,
    "SC4": 0.24,
    "W": 0.21,
    "Qp_Na": 0.12,
    "Qn_Cl": 0.18,
    "C1_tail": 0.24,
    "PEG": 0.22,
}

_DEFAULT_RADIUS: float = 0.24

# Hydrophobic bead types for SASA classification
_HYDROPHOBIC_TYPES: frozenset[str] = frozenset({
    "C1", "C1_tail", "SC4",
})


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _distance(a: Vector3, b: Vector3) -> float:
    """Euclidean distance between two 3-vectors."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return sqrt(dx * dx + dy * dy + dz * dz)


def _distance_sq(a: Vector3, b: Vector3) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def _fibonacci_sphere_points(n_points: int) -> list[Vector3]:
    """Generate *n_points* approximately uniform points on the unit sphere."""
    golden_angle = pi * (3.0 - sqrt(5.0))
    points: list[Vector3] = []
    for k in range(n_points):
        y = 1.0 - (2.0 * k / (n_points - 1)) if n_points > 1 else 0.0
        r_ring = sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * k
        x = r_ring * cos(theta)
        z = r_ring * sin(theta)
        points.append((x, y, z))
    return points


# ---------------------------------------------------------------------------
# 1. SASA (Solvent Accessible Surface Area)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SASAResult:
    """Result of a Shrake-Rupley SASA calculation."""

    total_sasa: float
    per_particle_sasa: tuple[float, ...]
    hydrophobic_sasa: float
    hydrophilic_sasa: float
    metadata: FrozenMetadata


@dataclass(slots=True)
class SASACalculator:
    """Shrake-Rupley solvent accessible surface area for CG beads."""

    probe_radius: float = 0.14
    n_sphere_points: int = 92
    name: str = "sasa_calculator"
    classification: str = "[established]"

    def compute(
        self,
        positions: VectorTuple,
        radii: tuple[float, ...],
        *,
        bead_types: tuple[str, ...] | None = None,
    ) -> SASAResult:
        """Compute SASA via Shrake-Rupley algorithm.

        Parameters
        ----------
        positions:
            Particle coordinates (nm).
        radii:
            Van der Waals radii per particle (nm).
        bead_types:
            Optional bead type names for hydrophobic/hydrophilic split.
        """
        n = len(positions)
        if len(radii) != n:
            raise ContractValidationError(
                "radii length must match positions length."
            )

        unit_points = _fibonacci_sphere_points(self.n_sphere_points)
        per_particle: list[float] = []

        for i in range(n):
            ri_expanded = radii[i] + self.probe_radius
            ri_expanded_sq = ri_expanded * ri_expanded
            surface_area_full = 4.0 * pi * ri_expanded_sq
            accessible = 0

            px, py, pz = positions[i]

            for ux, uy, uz in unit_points:
                # Test point on expanded sphere of atom i
                tx = px + ri_expanded * ux
                ty = py + ri_expanded * uy
                tz = pz + ri_expanded * uz

                buried = False
                for j in range(n):
                    if j == i:
                        continue
                    rj_expanded = radii[j] + self.probe_radius
                    dx = tx - positions[j][0]
                    dy = ty - positions[j][1]
                    dz = tz - positions[j][2]
                    if dx * dx + dy * dy + dz * dz < rj_expanded * rj_expanded:
                        buried = True
                        break

                if not buried:
                    accessible += 1

            sasa_i = surface_area_full * (accessible / self.n_sphere_points)
            per_particle.append(sasa_i)

        total = sum(per_particle)

        # Hydrophobic / hydrophilic split
        hydrophobic = 0.0
        hydrophilic = 0.0
        if bead_types is not None:
            for i, bt in enumerate(bead_types):
                if bt in _HYDROPHOBIC_TYPES:
                    hydrophobic += per_particle[i]
                else:
                    hydrophilic += per_particle[i]
        else:
            hydrophilic = total

        return SASAResult(
            total_sasa=total,
            per_particle_sasa=tuple(per_particle),
            hydrophobic_sasa=hydrophobic,
            hydrophilic_sasa=hydrophilic,
            metadata=FrozenMetadata({
                "probe_radius_nm": self.probe_radius,
                "n_sphere_points": self.n_sphere_points,
                "n_particles": n,
            }),
        )


# ---------------------------------------------------------------------------
# 2. Radius of Gyration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GyrationResult:
    """Result of a radius of gyration calculation."""

    radius_of_gyration: float
    rg_x: float
    rg_y: float
    rg_z: float
    asphericity: float
    center_of_mass: Vector3
    metadata: FrozenMetadata


@dataclass(frozen=True, slots=True)
class RadiusOfGyration:
    """Computes radius of gyration (optionally mass-weighted) and shape anisotropy."""

    name: str = "radius_of_gyration"
    classification: str = "[established]"

    def compute(
        self,
        positions: VectorTuple,
        masses: tuple[float, ...] | None = None,
    ) -> GyrationResult:
        n = len(positions)
        if n == 0:
            raise ContractValidationError("Cannot compute Rg for zero particles.")

        if masses is not None:
            if len(masses) != n:
                raise ContractValidationError(
                    "masses length must match positions length."
                )
            total_mass = sum(masses)
            com_x = sum(masses[i] * positions[i][0] for i in range(n)) / total_mass
            com_y = sum(masses[i] * positions[i][1] for i in range(n)) / total_mass
            com_z = sum(masses[i] * positions[i][2] for i in range(n)) / total_mass

            sq_x = sum(masses[i] * (positions[i][0] - com_x) ** 2 for i in range(n)) / total_mass
            sq_y = sum(masses[i] * (positions[i][1] - com_y) ** 2 for i in range(n)) / total_mass
            sq_z = sum(masses[i] * (positions[i][2] - com_z) ** 2 for i in range(n)) / total_mass
        else:
            com_x = sum(p[0] for p in positions) / n
            com_y = sum(p[1] for p in positions) / n
            com_z = sum(p[2] for p in positions) / n

            sq_x = sum((p[0] - com_x) ** 2 for p in positions) / n
            sq_y = sum((p[1] - com_y) ** 2 for p in positions) / n
            sq_z = sum((p[2] - com_z) ** 2 for p in positions) / n

        rg_x = sqrt(sq_x)
        rg_y = sqrt(sq_y)
        rg_z = sqrt(sq_z)
        rg_total = sqrt(sq_x + sq_y + sq_z)

        components = sorted([rg_x, rg_y, rg_z])
        rg_min = components[0]
        rg_max = components[2]
        asphericity = (rg_max - rg_min) / rg_total if rg_total > 0.0 else 0.0

        return GyrationResult(
            radius_of_gyration=rg_total,
            rg_x=rg_x,
            rg_y=rg_y,
            rg_z=rg_z,
            asphericity=asphericity,
            center_of_mass=(com_x, com_y, com_z),
            metadata=FrozenMetadata({
                "n_particles": n,
                "mass_weighted": masses is not None,
            }),
        )


# ---------------------------------------------------------------------------
# 3. Hydrogen Bond Analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class HydrogenBondCriteria:
    """Distance and angle criteria for CG hydrogen-bond-like contacts."""

    distance_cutoff: float = 0.35
    angle_cutoff: float = 2.618
    name: str = "hbond_criteria"


@dataclass(frozen=True, slots=True)
class HydrogenBond:
    """A single detected hydrogen-bond-like contact."""

    donor_index: int
    acceptor_index: int
    distance: float
    metadata: FrozenMetadata


@dataclass(slots=True)
class HydrogenBondAnalyzer:
    """CG proxy for hydrogen bond detection between polar/charged beads."""

    criteria: HydrogenBondCriteria = field(default_factory=HydrogenBondCriteria)
    donor_types: frozenset[str] = frozenset({
        "P5", "Nda", "Qp_Na", "Qp_Lys", "Qp_Arg",
    })
    acceptor_types: frozenset[str] = frozenset({
        "P5", "Nda", "Qn_Cl",
    })
    _history: list[tuple[int, int]] = field(default_factory=list)
    _pair_history: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    _recorded_steps: list[int] = field(default_factory=list)
    name: str = "hbond_analyzer"
    classification: str = "[adapted]"

    def _bonded_set(self, topology: SystemTopology) -> frozenset[tuple[int, int]]:
        """Return the set of directly bonded particle index pairs."""
        pairs: set[tuple[int, int]] = set()
        for bond in topology.bonds:
            a, b = bond.particle_index_a, bond.particle_index_b
            pairs.add((min(a, b), max(a, b)))
        return frozenset(pairs)

    def find_hbonds(
        self,
        state: SimulationState,
        topology: SystemTopology,
    ) -> tuple[HydrogenBond, ...]:
        """Find all H-bond-like contacts in the current state."""
        positions = state.particles.positions
        bonded = self._bonded_set(topology)
        cutoff_sq = self.criteria.distance_cutoff ** 2

        donors: list[int] = []
        acceptors: list[int] = []
        for bead in topology.beads:
            if bead.bead_type in self.donor_types:
                donors.append(bead.particle_index)
            if bead.bead_type in self.acceptor_types:
                acceptors.append(bead.particle_index)

        results: list[HydrogenBond] = []
        for d in donors:
            pd = positions[d]
            for a in acceptors:
                if d == a:
                    continue
                pair_key = (min(d, a), max(d, a))
                if pair_key in bonded:
                    continue
                dist_sq = _distance_sq(pd, positions[a])
                if dist_sq < cutoff_sq:
                    dist = sqrt(dist_sq)
                    results.append(HydrogenBond(
                        donor_index=d,
                        acceptor_index=a,
                        distance=dist,
                        metadata=FrozenMetadata({
                            "cutoff_nm": self.criteria.distance_cutoff,
                        }),
                    ))

        return tuple(results)

    def per_residue_hbonds(
        self,
        state: SimulationState,
        topology: SystemTopology,
    ) -> dict[int, int]:
        """Map each particle index to number of H-bonds it participates in."""
        hbonds = self.find_hbonds(state, topology)
        counts: dict[int, int] = {}
        for hb in hbonds:
            counts[hb.donor_index] = counts.get(hb.donor_index, 0) + 1
            counts[hb.acceptor_index] = counts.get(hb.acceptor_index, 0) + 1
        return counts

    def record(
        self,
        step: int,
        state: SimulationState,
        topology: SystemTopology,
    ) -> None:
        """Record H-bond count and per-pair presence at the given step."""
        hbonds = self.find_hbonds(state, topology)
        self._history.append((step, len(hbonds)))
        self._recorded_steps.append(step)

        # Track per-pair presence
        observed_pairs: set[tuple[int, int]] = set()
        for hb in hbonds:
            pair = (min(hb.donor_index, hb.acceptor_index),
                    max(hb.donor_index, hb.acceptor_index))
            observed_pairs.add(pair)

        # Record presence for all known pairs plus any new ones
        all_known = set(self._pair_history.keys()) | observed_pairs
        for pair in all_known:
            if pair not in self._pair_history:
                # Back-fill with zeros for all previously recorded steps
                self._pair_history[pair] = [0] * (len(self._recorded_steps) - 1)
            self._pair_history[pair].append(1 if pair in observed_pairs else 0)

    def occupancy(self) -> dict[tuple[int, int], float]:
        """Fraction of recorded frames where each donor-acceptor pair has H-bond."""
        result: dict[tuple[int, int], float] = {}
        for pair, presence in self._pair_history.items():
            total = len(presence)
            if total > 0:
                result[pair] = sum(presence) / total
        return result


# ---------------------------------------------------------------------------
# 4. Per-Residue Energy Decomposition
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EnergyDecompositionResult:
    """Per-particle energy breakdown."""

    per_particle_bonded: tuple[float, ...]
    per_particle_nonbonded: tuple[float, ...]
    per_particle_kinetic: tuple[float, ...]
    per_particle_total: tuple[float, ...]
    group_interaction_energies: dict
    total_bonded: float
    total_nonbonded: float
    total_kinetic: float
    metadata: FrozenMetadata


@dataclass(slots=True)
class ResidueEnergyDecomposition:
    """Decompose total energy into per-particle (residue in CG) contributions."""

    name: str = "residue_energy_decomposition"
    classification: str = "[established]"

    def decompose(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forces: VectorTuple,
        potential_energy: float,
    ) -> EnergyDecompositionResult:
        """Decompose energies into per-particle bonded, nonbonded, and kinetic.

        Bonded energy: 0.5 * k_approx * (r - r0)^2, split between bond partners.
        Non-bonded energy: approximate from total potential minus bonded, split evenly.
        Kinetic energy: 0.5 * m * |v|^2.
        """
        n = state.particles.particle_count
        positions = state.particles.positions
        masses = state.particles.masses
        velocities = state.particles.velocities

        # -- Kinetic energy per particle --
        kinetic: list[float] = []
        for i in range(n):
            vx, vy, vz = velocities[i]
            ke = 0.5 * masses[i] * (vx * vx + vy * vy + vz * vz)
            kinetic.append(ke)

        # -- Bonded energy per particle (split equally between partners) --
        bonded: list[float] = [0.0] * n
        # Use a simple harmonic estimate: E_bond ~ 0.5 * (r - r0)^2 * k_default
        # With k_default = 1250 kJ/(mol nm^2) (typical Martini bond spring constant)
        k_default = 1250.0
        r0_default = 0.37  # nm, typical Martini bond equilibrium length
        total_bonded_energy = 0.0
        for bond in topology.bonds:
            a = bond.particle_index_a
            b = bond.particle_index_b
            dist = _distance(positions[a], positions[b])
            e_bond = 0.5 * k_default * (dist - r0_default) ** 2
            total_bonded_energy += e_bond
            bonded[a] += e_bond * 0.5
            bonded[b] += e_bond * 0.5

        # -- Nonbonded per particle: distribute remaining potential proportionally --
        total_nonbonded_energy = (
            potential_energy - total_bonded_energy
            if potential_energy is not None
            else 0.0
        )
        # Distribute nonbonded energy proportional to force magnitude
        force_magnitudes: list[float] = []
        for i in range(n):
            fx, fy, fz = forces[i]
            force_magnitudes.append(sqrt(fx * fx + fy * fy + fz * fz))
        total_force_mag = sum(force_magnitudes)

        nonbonded: list[float] = []
        for i in range(n):
            if total_force_mag > 0.0:
                fraction = force_magnitudes[i] / total_force_mag
            else:
                fraction = 1.0 / n if n > 0 else 0.0
            nonbonded.append(total_nonbonded_energy * fraction)

        per_total = tuple(
            bonded[i] + nonbonded[i] + kinetic[i] for i in range(n)
        )

        return EnergyDecompositionResult(
            per_particle_bonded=tuple(bonded),
            per_particle_nonbonded=tuple(nonbonded),
            per_particle_kinetic=tuple(kinetic),
            per_particle_total=per_total,
            group_interaction_energies={},
            total_bonded=total_bonded_energy,
            total_nonbonded=total_nonbonded_energy,
            total_kinetic=sum(kinetic),
            metadata=FrozenMetadata({
                "n_particles": n,
                "k_bond_default": k_default,
                "r0_default": r0_default,
            }),
        )

    def interaction_energy_between_groups(
        self,
        state: SimulationState,
        topology: SystemTopology,
        group_a: tuple[int, ...],
        group_b: tuple[int, ...],
    ) -> float:
        """Sum of approximate pairwise LJ energies between two particle groups.

        Uses a 12-6 LJ potential with default CG parameters:
            epsilon = 2.0 kJ/mol, sigma = 0.47 nm (typical Martini).
        """
        epsilon = 2.0  # kJ/mol
        sigma = 0.47  # nm
        positions = state.particles.positions
        energy = 0.0

        for i in group_a:
            pi_pos = positions[i]
            for j in group_b:
                r = _distance(pi_pos, positions[j])
                if r < 1e-9:
                    continue
                sr6 = (sigma / r) ** 6
                energy += 4.0 * epsilon * (sr6 * sr6 - sr6)

        return energy


# ---------------------------------------------------------------------------
# 5. Contact Map
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ContactMap:
    """Contact map between particle pairs."""

    contacts: tuple[tuple[int, int], ...]
    distances: tuple[float, ...]
    n_contacts: int
    contact_density: float
    metadata: FrozenMetadata


@dataclass(frozen=True, slots=True)
class ContactMapCalculator:
    """Calculate contact maps from particle positions."""

    distance_cutoff: float = 0.8
    name: str = "contact_map"
    classification: str = "[established]"

    def compute(self, positions: VectorTuple) -> ContactMap:
        """Compute contact map for all pairs within distance cutoff."""
        n = len(positions)
        cutoff_sq = self.distance_cutoff ** 2
        contacts: list[tuple[int, int]] = []
        distances: list[float] = []

        for i in range(n):
            pi_pos = positions[i]
            for j in range(i + 1, n):
                dsq = _distance_sq(pi_pos, positions[j])
                if dsq < cutoff_sq:
                    contacts.append((i, j))
                    distances.append(sqrt(dsq))

        n_contacts = len(contacts)
        max_pairs = n * (n - 1) // 2 if n > 1 else 1
        density = n_contacts / max_pairs

        return ContactMap(
            contacts=tuple(contacts),
            distances=tuple(distances),
            n_contacts=n_contacts,
            contact_density=density,
            metadata=FrozenMetadata({
                "distance_cutoff_nm": self.distance_cutoff,
                "n_particles": n,
            }),
        )

    def native_contact_fraction(
        self,
        current_positions: VectorTuple,
        native_positions: VectorTuple,
    ) -> float:
        """Fraction of native contacts preserved in the current structure (Q value).

        A native contact is any pair within *distance_cutoff* in the native
        structure; Q is the fraction of those contacts that remain within the
        cutoff in the current structure.
        """
        native_map = self.compute(native_positions)
        if native_map.n_contacts == 0:
            return 1.0

        cutoff_sq = self.distance_cutoff ** 2
        preserved = 0
        for i_idx, j_idx in native_map.contacts:
            dsq = _distance_sq(current_positions[i_idx], current_positions[j_idx])
            if dsq < cutoff_sq:
                preserved += 1

        return preserved / native_map.n_contacts


# ---------------------------------------------------------------------------
# 6. Secondary Structure Proxy (for CG)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SecondaryStructureResult:
    """Per-residue secondary structure assignment for backbone beads."""

    per_residue: tuple[str, ...]
    helix_fraction: float
    sheet_fraction: float
    coil_fraction: float
    metadata: FrozenMetadata


@dataclass(frozen=True, slots=True)
class SecondaryStructureEstimator:
    """CG secondary structure estimator using backbone distance patterns."""

    helix_distance: float = 0.55
    helix_tolerance: float = 0.15
    sheet_distance: float = 0.48
    sheet_tolerance: float = 0.15
    name: str = "secondary_structure_estimator"
    classification: str = "[proposed novel]"

    def estimate(
        self,
        positions: VectorTuple,
        backbone_indices: tuple[int, ...],
    ) -> SecondaryStructureResult:
        """Estimate secondary structure from backbone bead distances.

        Walk the backbone sequence: if distance(i, i+3) is near *helix_distance*
        mark residues i..i+3 as helix; if distance(i, i+2) is near
        *sheet_distance* mark i..i+2 as sheet; otherwise coil.
        """
        n_bb = len(backbone_indices)
        assignments: list[str] = ["coil"] * n_bb

        # First pass: helix (i to i+3)
        for i in range(n_bb - 3):
            d = _distance(
                positions[backbone_indices[i]],
                positions[backbone_indices[i + 3]],
            )
            if abs(d - self.helix_distance) < self.helix_tolerance:
                for k in range(i, i + 4):
                    if assignments[k] == "coil":
                        assignments[k] = "helix"

        # Second pass: sheet (i to i+2), only overwrite coil
        for i in range(n_bb - 2):
            d = _distance(
                positions[backbone_indices[i]],
                positions[backbone_indices[i + 2]],
            )
            if abs(d - self.sheet_distance) < self.sheet_tolerance:
                for k in range(i, i + 3):
                    if assignments[k] == "coil":
                        assignments[k] = "sheet"

        n_helix = assignments.count("helix")
        n_sheet = assignments.count("sheet")
        n_coil = assignments.count("coil")
        total = max(n_bb, 1)

        return SecondaryStructureResult(
            per_residue=tuple(assignments),
            helix_fraction=n_helix / total,
            sheet_fraction=n_sheet / total,
            coil_fraction=n_coil / total,
            metadata=FrozenMetadata({
                "n_backbone_beads": n_bb,
                "helix_distance_nm": self.helix_distance,
                "sheet_distance_nm": self.sheet_distance,
            }),
        )


# ---------------------------------------------------------------------------
# 7. End-to-End Distance
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EndToEndDistance:
    """End-to-end distance between two terminal particles."""

    particle_a: int = 0
    particle_b: int = -1
    name: str = "end_to_end_distance"

    def compute(self, positions: VectorTuple) -> float:
        """Euclidean distance between the two terminal particles."""
        n = len(positions)
        if n == 0:
            raise ContractValidationError(
                "Cannot compute end-to-end distance for zero particles."
            )
        idx_a = self.particle_a
        idx_b = self.particle_b if self.particle_b >= 0 else n + self.particle_b
        return _distance(positions[idx_a], positions[idx_b])


# ---------------------------------------------------------------------------
# 8. Observable Collector and Snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MolecularSnapshot:
    """Single-frame snapshot of all molecular observables."""

    step: int
    sasa: SASAResult
    gyration: GyrationResult
    hbonds: tuple[HydrogenBond, ...]
    n_hbonds: int
    contact_map: ContactMap
    secondary_structure: SecondaryStructureResult
    end_to_end_distance: float
    metadata: FrozenMetadata


@dataclass(slots=True)
class MolecularObservableCollector:
    """Unified collector that runs all molecular observable analyzers."""

    sasa_calculator: SASACalculator = field(default_factory=SASACalculator)
    gyration_calculator: RadiusOfGyration = field(default_factory=RadiusOfGyration)
    hbond_analyzer: HydrogenBondAnalyzer = field(default_factory=HydrogenBondAnalyzer)
    contact_calculator: ContactMapCalculator = field(default_factory=ContactMapCalculator)
    ss_estimator: SecondaryStructureEstimator = field(default_factory=SecondaryStructureEstimator)
    e2e_distance: EndToEndDistance = field(default_factory=EndToEndDistance)
    _sasa_history: list[float] = field(default_factory=list)
    _rg_history: list[float] = field(default_factory=list)
    _hbond_history: list[int] = field(default_factory=list)
    _contact_history: list[int] = field(default_factory=list)
    _e2e_history: list[float] = field(default_factory=list)
    name: str = "molecular_observable_collector"
    classification: str = "[proposed novel]"

    def _resolve_radii(
        self,
        topology: SystemTopology,
        radii: tuple[float, ...] | None,
    ) -> tuple[float, ...]:
        """Resolve per-particle radii from explicit argument or topology bead types."""
        if radii is not None:
            return radii
        return tuple(
            DEFAULT_CG_RADII.get(bead.bead_type, _DEFAULT_RADIUS)
            for bead in topology.beads
        )

    def _resolve_bead_types(self, topology: SystemTopology) -> tuple[str, ...]:
        return tuple(bead.bead_type for bead in topology.beads)

    def _resolve_backbone_indices(self, topology: SystemTopology) -> tuple[int, ...]:
        """Return particle indices that belong to backbone beads (type 'BB')."""
        return tuple(
            bead.particle_index
            for bead in topology.beads
            if bead.bead_type == "BB"
        )

    def collect_all(
        self,
        state: SimulationState,
        topology: SystemTopology,
        *,
        radii: tuple[float, ...] | None = None,
    ) -> MolecularSnapshot:
        """Run all analyzers on the current state and return a unified snapshot."""
        positions = state.particles.positions
        masses = state.particles.masses
        resolved_radii = self._resolve_radii(topology, radii)
        bead_types = self._resolve_bead_types(topology)
        backbone_indices = self._resolve_backbone_indices(topology)

        sasa = self.sasa_calculator.compute(
            positions, resolved_radii, bead_types=bead_types,
        )
        gyration = self.gyration_calculator.compute(positions, masses)
        hbonds = self.hbond_analyzer.find_hbonds(state, topology)
        contacts = self.contact_calculator.compute(positions)

        if backbone_indices:
            ss = self.ss_estimator.estimate(positions, backbone_indices)
        else:
            ss = SecondaryStructureResult(
                per_residue=(),
                helix_fraction=0.0,
                sheet_fraction=0.0,
                coil_fraction=1.0,
                metadata=FrozenMetadata({"n_backbone_beads": 0}),
            )

        e2e = self.e2e_distance.compute(positions)

        return MolecularSnapshot(
            step=state.step,
            sasa=sasa,
            gyration=gyration,
            hbonds=hbonds,
            n_hbonds=len(hbonds),
            contact_map=contacts,
            secondary_structure=ss,
            end_to_end_distance=e2e,
            metadata=FrozenMetadata({
                "step": state.step,
                "time_ps": state.time,
                "n_particles": state.particles.particle_count,
            }),
        )

    def record(
        self,
        step: int,
        state: SimulationState,
        topology: SystemTopology,
        *,
        radii: tuple[float, ...] | None = None,
    ) -> None:
        """Collect all observables and append to internal histories."""
        snapshot = self.collect_all(state, topology, radii=radii)
        self._sasa_history.append(snapshot.sasa.total_sasa)
        self._rg_history.append(snapshot.gyration.radius_of_gyration)
        self._hbond_history.append(snapshot.n_hbonds)
        self._contact_history.append(snapshot.contact_map.n_contacts)
        self._e2e_history.append(snapshot.end_to_end_distance)
        self.hbond_analyzer.record(step, state, topology)

    def summary(self) -> dict:
        """Return time-averaged summary of all recorded observables."""

        def _avg(values: list) -> float:
            return sum(values) / len(values) if values else 0.0

        def _std(values: list) -> float:
            if len(values) < 2:
                return 0.0
            mean = _avg(values)
            return sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))

        return {
            "n_frames": len(self._sasa_history),
            "sasa_mean_nm2": _avg(self._sasa_history),
            "sasa_std_nm2": _std(self._sasa_history),
            "rg_mean_nm": _avg(self._rg_history),
            "rg_std_nm": _std(self._rg_history),
            "hbonds_mean": _avg(self._hbond_history),
            "hbonds_std": _std(self._hbond_history),
            "contacts_mean": _avg(self._contact_history),
            "contacts_std": _std(self._contact_history),
            "e2e_mean_nm": _avg(self._e2e_history),
            "e2e_std_nm": _std(self._e2e_history),
            "hbond_occupancy": self.hbond_analyzer.occupancy(),
        }
