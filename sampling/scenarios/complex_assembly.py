"""Concrete encounter-complex scenario used by the live dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from benchmarks.reference_cases import ReferenceComparisonReport
from compartments import CompartmentDomain, CompartmentRegistry
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import ParticleState, SimulationState, ThermodynamicState
from core.types import BeadId, FrozenMetadata, Vector3
from forcefields.base_forcefield import BaseForceField, BondParameter, NonbondedParameter
from graph.graph_manager import ConnectivityGraph
from physics.forces.composite import ForceEvaluation
from qcloud.qcloud_coupling import QCloudCorrectionModel
from topology import Bead, BeadRole, BeadType, Bond, SystemTopology
from validation import FidelityComparisonReport, StructureComparisonReport


def _distance(a: Vector3, b: Vector3) -> float:
    return sqrt(sum((a[axis] - b[axis]) ** 2 for axis in range(3)))


def _normalized_progress(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 1.0
    clamped = min(max(value, lower), upper)
    return (clamped - lower) / (upper - lower)


@dataclass(frozen=True, slots=True)
class ComplexAssemblyProgress(ValidatableComponent):
    """Deterministic summary of encounter-complex assembly progress."""

    name: str = "complex_assembly_progress"
    classification: str = "[adapted]"
    interface_pair: tuple[int, int] = (1, 5)
    initial_interface_distance: float = 0.0
    interface_distance: float = 0.0
    capture_distance: float = 1.8
    bound_distance: float = 1.25
    contact_distance: float = 1.8
    cross_contact_count: int = 0
    graph_bridge_count: int = 0
    target_contact_count: int = 4
    target_graph_bridge_count: int = 3
    stage_label: str = "Separated Search"
    assembly_score: float = 0.0
    bound: bool = False
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.initial_interface_distance <= 0.0:
            issues.append("initial_interface_distance must be positive.")
        if self.interface_distance < 0.0:
            issues.append("interface_distance must be non-negative.")
        if self.capture_distance <= 0.0 or self.bound_distance <= 0.0:
            issues.append("capture_distance and bound_distance must be positive.")
        if self.capture_distance <= self.bound_distance:
            issues.append("capture_distance must exceed bound_distance.")
        if self.contact_distance <= 0.0:
            issues.append("contact_distance must be positive.")
        if self.cross_contact_count < 0 or self.graph_bridge_count < 0:
            issues.append("contact and graph counts must be non-negative.")
        if self.target_contact_count <= 0 or self.target_graph_bridge_count <= 0:
            issues.append("target counts must be strictly positive.")
        if not self.stage_label.strip():
            issues.append("stage_label must be a non-empty string.")
        if not (0.0 <= self.assembly_score <= 1.0):
            issues.append("assembly_score must lie in [0, 1].")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "classification": self.classification,
            "interface_pair": list(self.interface_pair),
            "initial_interface_distance": self.initial_interface_distance,
            "interface_distance": self.interface_distance,
            "capture_distance": self.capture_distance,
            "bound_distance": self.bound_distance,
            "contact_distance": self.contact_distance,
            "cross_contact_count": self.cross_contact_count,
            "graph_bridge_count": self.graph_bridge_count,
            "target_contact_count": self.target_contact_count,
            "target_graph_bridge_count": self.target_graph_bridge_count,
            "stage_label": self.stage_label,
            "assembly_score": self.assembly_score,
            "bound": self.bound,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ComplexAssemblySetup(ValidatableComponent):
    """Materialized initial conditions and metadata for the encounter complex."""

    title: str
    summary: str
    objective: str
    topology: SystemTopology
    forcefield: BaseForceField
    initial_particles: ParticleState
    thermodynamics: ThermodynamicState
    compartments: CompartmentRegistry
    focus_compartments: tuple[str, ...]
    integrator_time_step: float
    integrator_friction: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "focus_compartments", tuple(self.focus_compartments))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.title.strip():
            issues.append("title must be a non-empty string.")
        if not self.summary.strip():
            issues.append("summary must be a non-empty string.")
        if not self.objective.strip():
            issues.append("objective must be a non-empty string.")
        if self.topology.validate_against_particle_state(self.initial_particles):
            issues.append("topology must align with initial_particles.")
        if self.compartments.particle_count != self.initial_particles.particle_count:
            issues.append("compartments must align with initial particle count.")
        if not self.focus_compartments:
            issues.append("focus_compartments must be non-empty.")
        if self.integrator_time_step <= 0.0:
            issues.append("integrator_time_step must be positive.")
        if self.integrator_friction < 0.0:
            issues.append("integrator_friction must be non-negative.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class EncounterComplexScenario(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """[adapted] Coarse-grained two-trimer encounter-complex assembly problem."""

    name: str = "encounter_complex_assembly"
    classification: str = "[adapted]"
    complex_a_indices: tuple[int, ...] = (0, 1, 2)
    complex_b_indices: tuple[int, ...] = (3, 4, 5)
    interface_pair: tuple[int, int] = (1, 5)
    initial_offset: float = 3.8
    contact_distance: float = 1.8
    capture_distance: float = 1.8
    bound_distance: float = 1.25
    target_contact_count: int = 4
    target_graph_bridge_count: int = 3
    recommended_time_step: float = 0.03
    recommended_friction: float = 0.6
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Provides a concrete coarse-grained encounter-complex assembly target "
            "for the live dashboard, tests, and future validation runs."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "core/state.py",
            "topology/system_topology.py",
            "forcefields/base_forcefield.py",
            "compartments/registry.py",
            "graph/graph_manager.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/use_cases/encounter_complex_dashboard.md",
            "docs/architecture/visualization_and_diagnostics.md",
        )

    def reference_case(self) -> None:
        return None

    def build_reference_report(self, progress: ComplexAssemblyProgress) -> ReferenceComparisonReport | None:
        del progress
        return None

    def build_structure_report(
        self,
        state: SimulationState,
        *,
        progress: ComplexAssemblyProgress | None = None,
    ) -> StructureComparisonReport | None:
        del state, progress
        return None

    def build_qcloud_correction_model(self) -> QCloudCorrectionModel | None:
        return None

    def build_fidelity_report(
        self,
        state: SimulationState,
        *,
        baseline_evaluation: ForceEvaluation,
        corrected_evaluation: ForceEvaluation,
        progress: ComplexAssemblyProgress | None = None,
    ) -> FidelityComparisonReport | None:
        del state, baseline_evaluation, corrected_evaluation, progress
        return None

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if len(self.complex_a_indices) < 2 or len(self.complex_b_indices) < 2:
            issues.append("Both complexes must contain at least two particles.")
        if len(set(self.complex_a_indices + self.complex_b_indices)) != len(
            self.complex_a_indices + self.complex_b_indices
        ):
            issues.append("complex_a_indices and complex_b_indices must be disjoint.")
        if self.initial_offset <= 0.0:
            issues.append("initial_offset must be positive.")
        if self.contact_distance <= 0.0 or self.capture_distance <= 0.0 or self.bound_distance <= 0.0:
            issues.append("distance thresholds must be positive.")
        if self.capture_distance <= self.bound_distance:
            issues.append("capture_distance must exceed bound_distance.")
        if self.target_contact_count <= 0 or self.target_graph_bridge_count <= 0:
            issues.append("target counts must be strictly positive.")
        if self.recommended_time_step <= 0.0:
            issues.append("recommended_time_step must be positive.")
        if self.recommended_friction < 0.0:
            issues.append("recommended_friction must be non-negative.")
        return tuple(issues)

    def initial_positions(self) -> tuple[Vector3, ...]:
        offset = self.initial_offset
        return (
            (0.0, 0.0, 0.0),
            (0.9, 0.7, 0.0),
            (0.9, -0.7, 0.0),
            (offset, 0.0, 0.0),
            (offset - 0.9, 0.7, 0.0),
            (offset - 0.9, -0.7, 0.0),
        )

    def initial_velocities(self) -> tuple[Vector3, ...]:
        return (
            (0.02, 0.0, 0.0),
            (0.02, -0.01, 0.0),
            (0.04, 0.0, 0.0),
            (-0.24, 0.0, 0.0),
            (-0.24, 0.0, 0.0),
            (-0.24, 0.01, 0.0),
        )

    def initial_interface_distance(self) -> float:
        positions = self.initial_positions()
        return _distance(positions[self.interface_pair[0]], positions[self.interface_pair[1]])

    def build_setup(self) -> ComplexAssemblySetup:
        particles = ParticleState(
            positions=self.initial_positions(),
            masses=(1.25, 1.0, 1.0, 1.25, 1.0, 1.0),
            velocities=self.initial_velocities(),
            labels=(
                "A_core",
                "A_iface",
                "A_support",
                "B_core",
                "B_support",
                "B_iface",
            ),
        )
        topology = SystemTopology(
            system_id=self.name,
            bead_types=(
                BeadType(name="core", role=BeadRole.STRUCTURAL),
                BeadType(name="iface_a", role=BeadRole.FUNCTIONAL),
                BeadType(name="iface_b", role=BeadRole.FUNCTIONAL),
                BeadType(name="support", role=BeadRole.LINKER),
            ),
            beads=(
                Bead(bead_id=BeadId("a0"), particle_index=0, bead_type="core", label="A_core", compartment_hint="A"),
                Bead(
                    bead_id=BeadId("a1"),
                    particle_index=1,
                    bead_type="iface_a",
                    label="A_iface",
                    compartment_hint="A",
                ),
                Bead(
                    bead_id=BeadId("a2"),
                    particle_index=2,
                    bead_type="support",
                    label="A_support",
                    compartment_hint="A",
                ),
                Bead(bead_id=BeadId("b0"), particle_index=3, bead_type="core", label="B_core", compartment_hint="B"),
                Bead(
                    bead_id=BeadId("b1"),
                    particle_index=4,
                    bead_type="support",
                    label="B_support",
                    compartment_hint="B",
                ),
                Bead(
                    bead_id=BeadId("b2"),
                    particle_index=5,
                    bead_type="iface_b",
                    label="B_iface",
                    compartment_hint="B",
                ),
            ),
            bonds=(Bond(0, 1), Bond(0, 2), Bond(1, 2), Bond(3, 4), Bond(3, 5), Bond(4, 5)),
            metadata={
                "scenario": self.name,
                "classification": self.classification,
            },
        )
        forcefield = BaseForceField(
            name=f"{self.name}_forcefield",
            bond_parameters=(
                BondParameter("core", "iface_a", equilibrium_distance=1.12, stiffness=85.0),
                BondParameter("core", "iface_b", equilibrium_distance=1.12, stiffness=85.0),
                BondParameter("core", "support", equilibrium_distance=1.12, stiffness=85.0),
                BondParameter("iface_a", "support", equilibrium_distance=1.4, stiffness=85.0),
                BondParameter("iface_b", "support", equilibrium_distance=1.4, stiffness=85.0),
            ),
            nonbonded_parameters=(
                NonbondedParameter("core", "core", sigma=1.0, epsilon=0.10, cutoff=4.2),
                NonbondedParameter("core", "iface_a", sigma=1.0, epsilon=0.12, cutoff=4.2),
                NonbondedParameter("core", "iface_b", sigma=1.0, epsilon=0.12, cutoff=4.2),
                NonbondedParameter("core", "support", sigma=1.0, epsilon=0.12, cutoff=4.2),
                NonbondedParameter("iface_a", "iface_a", sigma=1.0, epsilon=0.08, cutoff=4.2),
                NonbondedParameter("iface_b", "iface_b", sigma=1.0, epsilon=0.08, cutoff=4.2),
                NonbondedParameter("support", "support", sigma=1.0, epsilon=0.06, cutoff=4.2),
                NonbondedParameter("iface_a", "iface_b", sigma=1.05, epsilon=2.4, cutoff=4.2),
                NonbondedParameter("iface_a", "support", sigma=1.0, epsilon=0.08, cutoff=4.2),
                NonbondedParameter("iface_b", "support", sigma=1.0, epsilon=0.08, cutoff=4.2),
            ),
            metadata={
                "scenario": self.name,
                "dominant_interface_pair": list(self.interface_pair),
            },
        )
        compartments = CompartmentRegistry(
            particle_count=particles.particle_count,
            domains=(
                CompartmentDomain.from_members("A", "complex-a", self.complex_a_indices),
                CompartmentDomain.from_members("B", "complex-b", self.complex_b_indices),
            ),
        )
        return ComplexAssemblySetup(
            title="NeuroCGMD Live Dashboard | Encounter Complex",
            summary=(
                "Two coarse-grained trimers begin in a separated encounter state and "
                "must assemble into a stable cross-interface complex."
            ),
            objective=(
                "Reduce the dominant interface gap from the initial encounter geometry "
                "to the bound window while auxiliary cross-complex contacts accumulate."
            ),
            topology=topology,
            forcefield=forcefield,
            initial_particles=particles,
            thermodynamics=ThermodynamicState(),
            compartments=compartments,
            focus_compartments=("A", "B"),
            integrator_time_step=self.recommended_time_step,
            integrator_friction=self.recommended_friction,
            metadata={
                "scenario": self.name,
                "classification": self.classification,
                "initial_interface_distance": self.initial_interface_distance(),
            },
        )

    def measure_progress(
        self,
        state: SimulationState,
        *,
        graph: ConnectivityGraph | None = None,
    ) -> ComplexAssemblyProgress:
        if state.particle_count != len(self.complex_a_indices) + len(self.complex_b_indices):
            raise ContractValidationError("state particle_count does not match this scenario definition.")

        interface_distance = _distance(
            state.particles.positions[self.interface_pair[0]],
            state.particles.positions[self.interface_pair[1]],
        )
        cross_contact_count = self._count_cross_contacts(state)
        graph_bridge_count = self._count_graph_bridges(graph)
        distance_score = 1.0 - _normalized_progress(
            interface_distance,
            self.bound_distance,
            self.initial_interface_distance(),
        )
        contact_score = min(1.0, cross_contact_count / self.target_contact_count)
        graph_score = min(1.0, graph_bridge_count / self.target_graph_bridge_count)
        assembly_score = max(0.0, min(1.0, 0.55 * distance_score + 0.30 * contact_score + 0.15 * graph_score))
        bound = interface_distance <= self.bound_distance and cross_contact_count >= self.target_contact_count
        return ComplexAssemblyProgress(
            interface_pair=self.interface_pair,
            initial_interface_distance=self.initial_interface_distance(),
            interface_distance=interface_distance,
            capture_distance=self.capture_distance,
            bound_distance=self.bound_distance,
            contact_distance=self.contact_distance,
            cross_contact_count=cross_contact_count,
            graph_bridge_count=graph_bridge_count,
            target_contact_count=self.target_contact_count,
            target_graph_bridge_count=self.target_graph_bridge_count,
            stage_label=self._stage_label(interface_distance, cross_contact_count, bound),
            assembly_score=round(assembly_score, 6),
            bound=bound,
            metadata={
                "simulation_id": str(state.provenance.simulation_id),
                "state_id": str(state.provenance.state_id),
            },
        )

    def _count_cross_contacts(self, state: SimulationState) -> int:
        count = 0
        for source in self.complex_a_indices:
            source_position = state.particles.positions[source]
            for target in self.complex_b_indices:
                target_position = state.particles.positions[target]
                if _distance(source_position, target_position) <= self.contact_distance:
                    count += 1
        return count

    def _count_graph_bridges(self, graph: ConnectivityGraph | None) -> int:
        if graph is None:
            return 0
        complex_a = set(self.complex_a_indices)
        complex_b = set(self.complex_b_indices)
        return sum(
            1
            for edge in graph.active_edges()
            if (
                edge.source_index in complex_a
                and edge.target_index in complex_b
                or edge.source_index in complex_b
                and edge.target_index in complex_a
            )
        )

    def _stage_label(self, interface_distance: float, contact_count: int, bound: bool) -> str:
        if bound:
            return "Locked Complex"
        if interface_distance <= self.capture_distance and contact_count >= 3:
            return "Docking Transition"
        if interface_distance <= self.initial_interface_distance() - 0.25 or contact_count > 0:
            return "Encounter Alignment"
        return "Separated Search"
