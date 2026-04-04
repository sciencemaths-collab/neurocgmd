"""Barnase-barstar coarse-grained proxy scenario anchored to real data."""

from __future__ import annotations

from dataclasses import dataclass, field

from benchmarks import barnase_barstar_reference_case, build_barnase_barstar_proxy_report
from benchmarks.reference_cases import ReferenceComparisonReport
from benchmarks.reference_cases.models import ExperimentalReferenceCase
from benchmarks.reference_cases.barnase_barstar_structure_targets import barnase_barstar_structure_target
from compartments import CompartmentDomain, CompartmentRegistry
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import ParticleState, SimulationState, ThermodynamicState
from core.types import BeadId, FrozenMetadata, Vector3
from forcefields.base_forcefield import BaseForceField, BondParameter, NonbondedParameter
from graph.graph_manager import ConnectivityGraph
from physics.forces.composite import ForceEvaluation
from qcloud import ProteinShadowRuntimeBundle, ProteinShadowTuner
from qcloud.qcloud_coupling import QCloudCorrectionModel
from sampling.scenarios.complex_assembly import (
    ComplexAssemblyProgress,
    ComplexAssemblySetup,
    _distance,
    _normalized_progress,
)
from topology import Bead, BeadRole, BeadType, Bond, SystemTopology
from validation import (
    FidelityComparisonReport,
    LandmarkObservation,
    ReferenceForceTarget,
    ShadowFidelityAssessor,
    StructureComparisonReport,
    compare_landmark_observations,
)


_BARNASE_BARSTAR_LANDMARK_LABELS: tuple[str, ...] = (
    "Barnase_core",
    "Barnase_basic_patch",
    "Barnase_recognition_loop",
    "Barstar_core",
    "Barstar_helix_face",
    "Barstar_acidic_patch",
)


@dataclass(frozen=True, slots=True)
class BarnaseBarstarScenario(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """[hybrid] Coarse-grained barnase-barstar benchmark proxy."""

    name: str = "barnase_barstar_proxy"
    classification: str = "[hybrid]"
    barnase_indices: tuple[int, ...] = (0, 1, 2)
    barstar_indices: tuple[int, ...] = (3, 4, 5)
    interface_pair: tuple[int, int] = (1, 5)
    initial_offset: float = 3.8
    contact_distance: float = 1.8
    capture_distance: float = 1.9
    bound_distance: float = 1.25
    target_contact_count: int = 5
    target_graph_bridge_count: int = 3
    recommended_time_step: float = 0.03
    recommended_friction: float = 0.6
    shadow_tuner: ProteinShadowTuner = field(default_factory=ProteinShadowTuner)
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Provides a coarse-grained live benchmark proxy for the real barnase-barstar "
            "association problem while keeping the experimental reference case explicit."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "benchmarks/reference_cases/barnase_barstar.py",
            "core/state.py",
            "topology/system_topology.py",
            "forcefields/base_forcefield.py",
            "graph/graph_manager.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/use_cases/barnase_barstar_reference_case.md",
            "docs/architecture/visualization_and_diagnostics.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if len(self.barnase_indices) < 2 or len(self.barstar_indices) < 2:
            issues.append("Both protein proxies must contain at least two particles.")
        if len(set(self.barnase_indices + self.barstar_indices)) != len(self.barnase_indices + self.barstar_indices):
            issues.append("barnase_indices and barstar_indices must be disjoint.")
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
        issues.extend(self.shadow_tuner.validate())
        return tuple(issues)

    def reference_case(self) -> ExperimentalReferenceCase:
        return barnase_barstar_reference_case()

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
        reference_case = self.reference_case()
        particles = ParticleState(
            positions=self.initial_positions(),
            masses=(1.25, 1.0, 1.0, 1.25, 1.0, 1.0),
            velocities=self.initial_velocities(),
            labels=(
                "Barnase_core",
                "Barnase_basic_patch",
                "Barnase_recognition_loop",
                "Barstar_core",
                "Barstar_helix_face",
                "Barstar_acidic_patch",
            ),
        )
        topology = SystemTopology(
            system_id=self.name,
            bead_types=(
                BeadType(name="core", role=BeadRole.STRUCTURAL),
                BeadType(name="basic", role=BeadRole.FUNCTIONAL),
                BeadType(name="loop", role=BeadRole.LINKER),
                BeadType(name="helix", role=BeadRole.FUNCTIONAL),
                BeadType(name="acidic", role=BeadRole.FUNCTIONAL),
            ),
            beads=(
                Bead(bead_id=BeadId("bn0"), particle_index=0, bead_type="core", label="Barnase_core", compartment_hint="barnase"),
                Bead(
                    bead_id=BeadId("bn1"),
                    particle_index=1,
                    bead_type="basic",
                    label="Barnase_basic_patch",
                    compartment_hint="barnase",
                ),
                Bead(
                    bead_id=BeadId("bn2"),
                    particle_index=2,
                    bead_type="loop",
                    label="Barnase_recognition_loop",
                    compartment_hint="barnase",
                ),
                Bead(bead_id=BeadId("bs0"), particle_index=3, bead_type="core", label="Barstar_core", compartment_hint="barstar"),
                Bead(
                    bead_id=BeadId("bs1"),
                    particle_index=4,
                    bead_type="helix",
                    label="Barstar_helix_face",
                    compartment_hint="barstar",
                ),
                Bead(
                    bead_id=BeadId("bs2"),
                    particle_index=5,
                    bead_type="acidic",
                    label="Barstar_acidic_patch",
                    compartment_hint="barstar",
                ),
            ),
            bonds=(Bond(0, 1), Bond(0, 2), Bond(1, 2), Bond(3, 4), Bond(3, 5), Bond(4, 5)),
            metadata={
                "scenario": self.name,
                "classification": self.classification,
                "reference_case": reference_case.name,
            },
        )
        forcefield = BaseForceField(
            name=f"{self.name}_forcefield",
            bond_parameters=(
                BondParameter("core", "basic", equilibrium_distance=1.12, stiffness=85.0),
                BondParameter("core", "loop", equilibrium_distance=1.12, stiffness=85.0),
                BondParameter("basic", "loop", equilibrium_distance=1.40, stiffness=85.0),
                BondParameter("core", "helix", equilibrium_distance=1.12, stiffness=85.0),
                BondParameter("core", "acidic", equilibrium_distance=1.12, stiffness=85.0),
                BondParameter("helix", "acidic", equilibrium_distance=1.40, stiffness=85.0),
            ),
            nonbonded_parameters=(
                NonbondedParameter("core", "core", sigma=1.0, epsilon=0.10, cutoff=4.2),
                NonbondedParameter("core", "basic", sigma=1.0, epsilon=0.12, cutoff=4.2),
                NonbondedParameter("core", "loop", sigma=1.0, epsilon=0.10, cutoff=4.2),
                NonbondedParameter("core", "helix", sigma=1.0, epsilon=0.12, cutoff=4.2),
                NonbondedParameter("core", "acidic", sigma=1.0, epsilon=0.12, cutoff=4.2),
                NonbondedParameter("basic", "basic", sigma=1.0, epsilon=0.08, cutoff=4.2),
                NonbondedParameter("basic", "loop", sigma=1.0, epsilon=0.10, cutoff=4.2),
                NonbondedParameter("basic", "helix", sigma=1.0, epsilon=0.18, cutoff=4.2),
                NonbondedParameter("basic", "acidic", sigma=1.05, epsilon=2.4, cutoff=4.2),
                NonbondedParameter("loop", "loop", sigma=1.0, epsilon=0.08, cutoff=4.2),
                NonbondedParameter("loop", "helix", sigma=1.0, epsilon=1.1, cutoff=4.2),
                NonbondedParameter("loop", "acidic", sigma=1.0, epsilon=0.12, cutoff=4.2),
                NonbondedParameter("helix", "helix", sigma=1.0, epsilon=0.08, cutoff=4.2),
                NonbondedParameter("helix", "acidic", sigma=1.0, epsilon=0.10, cutoff=4.2),
                NonbondedParameter("acidic", "acidic", sigma=1.0, epsilon=0.08, cutoff=4.2),
            ),
            metadata={
                "scenario": self.name,
                "reference_case": reference_case.name,
                "dominant_interface_pair": list(self.interface_pair),
            },
        )
        compartments = CompartmentRegistry(
            particle_count=particles.particle_count,
            domains=(
                CompartmentDomain.from_members("barnase", "barnase", self.barnase_indices),
                CompartmentDomain.from_members("barstar", "barstar", self.barstar_indices),
            ),
        )
        shadow_bundle = self._shadow_runtime_bundle(topology)
        return ComplexAssemblySetup(
            title="NeuroCGMD Live Dashboard | Barnase-Barstar",
            summary=(
                "Coarse-grained proxy of the real barnase-barstar association benchmark, "
                "anchored to known bound structure and measured kinetics."
            ),
            objective=(
                "Form a barnase-barstar encounter complex that closes the dominant basic-acidic "
                "interface while the recognition-loop and helix-face contacts recover a native-like docked state."
            ),
            topology=topology,
            forcefield=forcefield,
            initial_particles=particles,
            thermodynamics=ThermodynamicState(),
            compartments=compartments,
            focus_compartments=("barnase", "barstar"),
            integrator_time_step=shadow_bundle.dynamics_recommendation.time_step,
            integrator_friction=shadow_bundle.dynamics_recommendation.friction_coefficient,
            metadata={
                "scenario": self.name,
                "classification": self.classification,
                "reference_case": reference_case.name,
                "bound_complex_pdb_id": reference_case.structural_reference.bound_complex_pdb_id,
                "initial_interface_distance": self.initial_interface_distance(),
                "shadow_tuning_preset": shadow_bundle.dynamics_recommendation.metadata["preset_name"],
                "shadow_time_step_multiplier": shadow_bundle.dynamics_recommendation.metadata["time_step_multiplier"],
                "shadow_friction_multiplier": shadow_bundle.dynamics_recommendation.metadata["friction_multiplier"],
            },
        )

    def measure_progress(
        self,
        state: SimulationState,
        *,
        graph: ConnectivityGraph | None = None,
    ) -> ComplexAssemblyProgress:
        if state.particle_count != len(self.barnase_indices) + len(self.barstar_indices):
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
        assembly_score = max(0.0, min(1.0, 0.60 * distance_score + 0.25 * contact_score + 0.15 * graph_score))
        bound = interface_distance <= self.bound_distance and cross_contact_count >= self.target_contact_count
        return ComplexAssemblyProgress(
            name="barnase_barstar_progress",
            classification=self.classification,
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
                "reference_case": self.reference_case().name,
            },
        )

    def build_reference_report(self, progress: ComplexAssemblyProgress) -> ReferenceComparisonReport:
        return build_barnase_barstar_proxy_report(
            reference_case=self.reference_case(),
            current_stage=progress.stage_label,
            interface_distance=progress.interface_distance,
            bound_distance=progress.bound_distance,
            cross_contact_count=progress.cross_contact_count,
            target_contact_count=progress.target_contact_count,
            bound=progress.bound,
        )

    def build_structure_report(
        self,
        state: SimulationState,
        *,
        progress: ComplexAssemblyProgress | None = None,
    ) -> StructureComparisonReport:
        del progress
        observations = self._landmark_observations(state)
        return compare_landmark_observations(
            state_id=state.provenance.state_id,
            target=barnase_barstar_structure_target(),
            observations=observations,
        )

    def build_qcloud_correction_model(self) -> QCloudCorrectionModel | None:
        return self._shadow_runtime_bundle(self.build_setup().topology).build_correction_model()

    def build_fidelity_report(
        self,
        state: SimulationState,
        *,
        baseline_evaluation: ForceEvaluation,
        corrected_evaluation: ForceEvaluation,
        progress: ComplexAssemblyProgress | None = None,
    ) -> FidelityComparisonReport:
        del progress
        return ShadowFidelityAssessor().assess(
            target=self._shadow_reference_force_target(state),
            baseline=baseline_evaluation,
            corrected=corrected_evaluation,
            title="Shadow Fidelity",
        )

    def _landmark_observations(self, state: SimulationState) -> tuple[LandmarkObservation, ...]:
        return tuple(
            LandmarkObservation(
                label=label,
                position=state.particles.positions[index],
                metadata={"particle_index": index},
            )
            for index, label in enumerate(_BARNASE_BARSTAR_LANDMARK_LABELS)
        )

    def _shadow_runtime_bundle(self, topology: SystemTopology) -> ProteinShadowRuntimeBundle:
        return self._protein_shadow_tuner().build_runtime_bundle(
            topology=topology,
            scenario_label=self.name,
            base_time_step=self.recommended_time_step,
            base_friction=self.recommended_friction,
            reference_label=self.reference_case().name,
            metadata={
                "reference_case": self.reference_case().name,
                "bound_complex_pdb_id": self.reference_case().structural_reference.bound_complex_pdb_id,
            },
        )

    def _protein_shadow_tuner(self) -> ProteinShadowTuner:
        return self.shadow_tuner

    def _shadow_reference_force_target(self, state: SimulationState) -> ReferenceForceTarget:
        if state.particle_count != len(_BARNASE_BARSTAR_LANDMARK_LABELS):
            raise ContractValidationError("state particle_count does not match barnase_barstar landmark mapping.")

        reference_target = barnase_barstar_structure_target()
        label_to_index = {label: index for index, label in enumerate(_BARNASE_BARSTAR_LANDMARK_LABELS)}
        accumulated_forces = [[0.0, 0.0, 0.0] for _ in range(state.particle_count)]
        total_energy = 0.0

        for contact in reference_target.interface_contacts:
            source_index = label_to_index[contact.source_label]
            target_index = label_to_index[contact.target_label]
            source_position = state.particles.positions[source_index]
            target_position = state.particles.positions[target_index]
            displacement = tuple(target_position[axis] - source_position[axis] for axis in range(3))
            distance = max(_distance(source_position, target_position), 1.0e-8)
            target_distance = float(contact.metadata["target_distance"])
            stiffness = self._contact_target_stiffness(contact.source_label, contact.target_label)
            delta = distance - target_distance
            total_energy += -0.5 * stiffness * delta * delta
            magnitude = stiffness * delta / distance
            for axis in range(3):
                force_component = magnitude * displacement[axis]
                accumulated_forces[source_index][axis] += force_component
                accumulated_forces[target_index][axis] -= force_component

        return ReferenceForceTarget(
            label="1BRS Shadow Contact Target",
            potential_energy=total_energy,
            forces=tuple(tuple(vector) for vector in accumulated_forces),
            metadata={
                "reference_case": self.reference_case().name,
                "contact_count": len(reference_target.interface_contacts),
                "representation": "structure_informed_contact_force_target",
            },
        )

    def _contact_target_stiffness(self, source_label: str, target_label: str) -> float:
        labels = {source_label, target_label}
        if labels == {"Barnase_basic_patch", "Barstar_acidic_patch"}:
            return 1.70
        if labels == {"Barnase_recognition_loop", "Barstar_helix_face"}:
            return 1.30
        if labels == {"Barnase_basic_patch", "Barstar_helix_face"}:
            return 1.10
        if labels == {"Barnase_recognition_loop", "Barstar_acidic_patch"}:
            return 0.95
        return 0.75

    def _count_cross_contacts(self, state: SimulationState) -> int:
        count = 0
        for source in self.barnase_indices:
            source_position = state.particles.positions[source]
            for target in self.barstar_indices:
                target_position = state.particles.positions[target]
                if _distance(source_position, target_position) <= self.contact_distance:
                    count += 1
        return count

    def _count_graph_bridges(self, graph: ConnectivityGraph | None) -> int:
        if graph is None:
            return 0
        barnase = set(self.barnase_indices)
        barstar = set(self.barstar_indices)
        return sum(
            1
            for edge in graph.active_edges()
            if (
                edge.source_index in barnase
                and edge.target_index in barstar
                or edge.source_index in barstar
                and edge.target_index in barnase
            )
        )

    def _stage_label(self, interface_distance: float, contact_count: int, bound: bool) -> str:
        if bound:
            return "Native-Like Docking"
        if interface_distance <= self.capture_distance and contact_count >= 3:
            return "Encounter Complex"
        if interface_distance <= self.initial_interface_distance() - 0.25 or contact_count > 0:
            return "Electrostatic Steering"
        return "Diffusive Search"
