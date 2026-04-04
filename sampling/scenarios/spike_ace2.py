"""Harder ACE2-spike RBD coarse-grained proxy scenario anchored to real data."""

from __future__ import annotations

from dataclasses import dataclass, field

from benchmarks import build_spike_ace2_proxy_report, spike_ace2_reference_case
from benchmarks.reference_cases import ReferenceComparisonReport
from benchmarks.reference_cases.models import ExperimentalReferenceCase
from benchmarks.reference_cases.spike_ace2_structure_targets import spike_ace2_structure_target
from compartments import CompartmentDomain, CompartmentRegistry
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import ParticleState, SimulationState, ThermodynamicState
from core.types import BeadId, FrozenMetadata, Vector3
from forcefields import (
    BaseForceField,
    BondParameter,
    NonbondedParameter,
)
from graph.graph_manager import ConnectivityGraph
from physics.forces.composite import ForceEvaluation
from qcloud import (
    ProteinShadowRuntimeBundle,
    ProteinShadowTuner,
)
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

_SPIKE_ACE2_LANDMARK_LABELS: tuple[str, ...] = (
    "ACE2_core",
    "ACE2_alpha1_support",
    "ACE2_alpha1_hotspot",
    "ACE2_beta_hotspot",
    "ACE2_support",
    "ACE2_glycan_side",
    "RBD_core",
    "RBD_ridge_support",
    "RBD_ridge_hotspot",
    "RBD_loop_hotspot",
    "RBD_support",
    "RBD_shielded_side",
)


@dataclass(frozen=True, slots=True)
class SpikeAce2Scenario(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """[hybrid] Coarse-grained ACE2-spike RBD benchmark proxy."""

    name: str = "spike_ace2_proxy"
    classification: str = "[hybrid]"
    ace2_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    spike_indices: tuple[int, ...] = (6, 7, 8, 9, 10, 11)
    interface_pair: tuple[int, int] = (2, 8)
    initial_offset: float = 2.9
    contact_distance: float = 2.10
    capture_distance: float = 2.35
    bound_distance: float = 1.25
    target_contact_count: int = 7
    target_graph_bridge_count: int = 4
    recommended_time_step: float = 0.022
    recommended_friction: float = 0.75
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
            "Provides a harder coarse-grained live benchmark proxy for the SARS-CoV-2 "
            "spike RBD - ACE2 recognition problem while keeping the experimental target explicit."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "benchmarks/reference_cases/spike_ace2.py",
            "core/state.py",
            "topology/system_topology.py",
            "forcefields/base_forcefield.py",
            "graph/graph_manager.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/use_cases/spike_ace2_reference_case.md",
            "docs/use_cases/spike_ace2_live_proxy.md",
            "docs/architecture/visualization_and_diagnostics.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if len(self.ace2_indices) < 3 or len(self.spike_indices) < 3:
            issues.append("Both ACE2 and spike proxy domains must contain at least three particles.")
        if len(set(self.ace2_indices + self.spike_indices)) != len(self.ace2_indices + self.spike_indices):
            issues.append("ace2_indices and spike_indices must be disjoint.")
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
        return spike_ace2_reference_case()

    def initial_positions(self) -> tuple[Vector3, ...]:
        offset = self.initial_offset
        return (
            (-1.9, 0.0, 0.0),
            (-1.0, 1.0, 0.1),
            (-0.2, 0.45, 0.0),
            (-0.15, -0.35, 0.0),
            (-1.0, -1.0, -0.1),
            (-2.0, 0.85, 0.0),
            (offset + 1.0, 0.0, 0.0),
            (offset + 0.45, 1.05, 0.1),
            (offset - 0.2, 0.45, 0.0),
            (offset - 0.05, -0.35, 0.0),
            (offset + 0.55, -1.0, -0.1),
            (offset + 1.15, 0.85, 0.0),
        )

    def initial_velocities(self) -> tuple[Vector3, ...]:
        return (
            (0.03, 0.0, 0.0),
            (0.04, -0.005, 0.0),
            (0.045, -0.01, 0.0),
            (0.045, 0.01, 0.0),
            (0.04, 0.0, 0.0),
            (0.03, 0.0, 0.0),
            (-0.48, 0.0, 0.0),
            (-0.44, -0.01, 0.0),
            (-0.46, 0.0, 0.0),
            (-0.44, 0.01, 0.0),
            (-0.43, 0.0, 0.0),
            (-0.41, 0.0, 0.0),
        )

    def initial_interface_distance(self) -> float:
        positions = self.initial_positions()
        return _distance(positions[self.interface_pair[0]], positions[self.interface_pair[1]])

    def build_setup(self) -> ComplexAssemblySetup:
        reference_case = self.reference_case()
        particles = ParticleState(
            positions=self.initial_positions(),
            masses=(1.6, 1.15, 1.0, 1.0, 1.1, 1.2, 1.5, 1.1, 1.0, 1.0, 1.1, 1.2),
            velocities=self.initial_velocities(),
            labels=(
                "ACE2_core",
                "ACE2_alpha1_support",
                "ACE2_alpha1_hotspot",
                "ACE2_beta_hotspot",
                "ACE2_support",
                "ACE2_glycan_side",
                "RBD_core",
                "RBD_ridge_support",
                "RBD_ridge_hotspot",
                "RBD_loop_hotspot",
                "RBD_support",
                "RBD_shielded_side",
            ),
        )
        topology = SystemTopology(
            system_id=self.name,
            bead_types=(
                BeadType(name="core", role=BeadRole.STRUCTURAL),
                BeadType(name="helix", role=BeadRole.FUNCTIONAL),
                BeadType(name="hotspot", role=BeadRole.ANCHOR),
                BeadType(name="support", role=BeadRole.STRUCTURAL),
                BeadType(name="shield", role=BeadRole.LINKER),
            ),
            beads=(
                Bead(BeadId("a0"), 0, "core", "ACE2_core", compartment_hint="ace2"),
                Bead(BeadId("a1"), 1, "helix", "ACE2_alpha1_support", compartment_hint="ace2"),
                Bead(BeadId("a2"), 2, "hotspot", "ACE2_alpha1_hotspot", compartment_hint="ace2"),
                Bead(BeadId("a3"), 3, "hotspot", "ACE2_beta_hotspot", compartment_hint="ace2"),
                Bead(BeadId("a4"), 4, "support", "ACE2_support", compartment_hint="ace2"),
                Bead(BeadId("a5"), 5, "shield", "ACE2_glycan_side", compartment_hint="ace2"),
                Bead(BeadId("s0"), 6, "core", "RBD_core", compartment_hint="spike_rbd"),
                Bead(BeadId("s1"), 7, "support", "RBD_ridge_support", compartment_hint="spike_rbd"),
                Bead(BeadId("s2"), 8, "hotspot", "RBD_ridge_hotspot", compartment_hint="spike_rbd"),
                Bead(BeadId("s3"), 9, "hotspot", "RBD_loop_hotspot", compartment_hint="spike_rbd"),
                Bead(BeadId("s4"), 10, "support", "RBD_support", compartment_hint="spike_rbd"),
                Bead(BeadId("s5"), 11, "shield", "RBD_shielded_side", compartment_hint="spike_rbd"),
            ),
            bonds=(
                Bond(0, 1),
                Bond(1, 2),
                Bond(2, 3),
                Bond(3, 4),
                Bond(0, 4),
                Bond(1, 5),
                Bond(0, 5),
                Bond(6, 7),
                Bond(7, 8),
                Bond(8, 9),
                Bond(9, 10),
                Bond(6, 10),
                Bond(6, 11),
                Bond(7, 11),
            ),
            metadata={
                "scenario": self.name,
                "classification": self.classification,
                "reference_case": reference_case.name,
            },
        )
        forcefield = BaseForceField(
            name=f"{self.name}_forcefield",
            bond_parameters=(
                BondParameter("core", "helix", equilibrium_distance=1.18, stiffness=95.0),
                BondParameter("helix", "hotspot", equilibrium_distance=1.05, stiffness=95.0),
                BondParameter("hotspot", "hotspot", equilibrium_distance=0.92, stiffness=95.0),
                BondParameter("hotspot", "support", equilibrium_distance=1.10, stiffness=95.0),
                BondParameter("core", "support", equilibrium_distance=1.35, stiffness=95.0),
                BondParameter("helix", "shield", equilibrium_distance=1.10, stiffness=95.0),
                BondParameter("core", "shield", equilibrium_distance=1.18, stiffness=95.0),
                BondParameter("support", "shield", equilibrium_distance=1.18, stiffness=95.0),
            ),
            nonbonded_parameters=(
                NonbondedParameter("core", "core", sigma=1.0, epsilon=0.08, cutoff=5.5),
                NonbondedParameter("core", "helix", sigma=1.0, epsilon=0.10, cutoff=5.5),
                NonbondedParameter("core", "hotspot", sigma=1.0, epsilon=0.12, cutoff=5.5),
                NonbondedParameter("core", "support", sigma=1.0, epsilon=0.10, cutoff=5.5),
                NonbondedParameter("core", "shield", sigma=1.05, epsilon=0.04, cutoff=5.5),
                NonbondedParameter("helix", "helix", sigma=1.0, epsilon=0.10, cutoff=5.5),
                NonbondedParameter("helix", "hotspot", sigma=1.0, epsilon=2.40, cutoff=5.5),
                NonbondedParameter("helix", "support", sigma=1.0, epsilon=0.12, cutoff=5.5),
                NonbondedParameter("helix", "shield", sigma=1.05, epsilon=0.05, cutoff=5.5),
                NonbondedParameter("hotspot", "hotspot", sigma=1.0, epsilon=2.80, cutoff=5.5),
                NonbondedParameter("hotspot", "support", sigma=1.0, epsilon=1.50, cutoff=5.5),
                NonbondedParameter("hotspot", "shield", sigma=1.05, epsilon=0.06, cutoff=5.5),
                NonbondedParameter("support", "support", sigma=1.0, epsilon=0.10, cutoff=5.5),
                NonbondedParameter("support", "shield", sigma=1.05, epsilon=0.06, cutoff=5.5),
                NonbondedParameter("shield", "shield", sigma=1.08, epsilon=0.02, cutoff=5.5),
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
                CompartmentDomain.from_members("ace2", "ace2", self.ace2_indices),
                CompartmentDomain.from_members("spike_rbd", "spike_rbd", self.spike_indices),
            ),
        )
        shadow_bundle = self._shadow_runtime_bundle(topology)
        return ComplexAssemblySetup(
            title="NeuroCGMD Live Dashboard | Spike-ACE2",
            summary=(
                "Harder coarse-grained proxy of SARS-CoV-2 spike RBD recognition of human ACE2, "
                "anchored to the known bound complex and reported receptor-binding affinity."
            ),
            objective=(
                "Form a distributed spike-ACE2 hotspot network that closes the ACE2 alpha1 / RBD ridge "
                "interface while preserving a larger, more constrained recognition geometry than the "
                "early barnase-barstar benchmark."
            ),
            topology=topology,
            forcefield=forcefield,
            initial_particles=particles,
            thermodynamics=ThermodynamicState(),
            compartments=compartments,
            focus_compartments=("ace2", "spike_rbd"),
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
        if state.particle_count != len(self.ace2_indices) + len(self.spike_indices):
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
            name="spike_ace2_progress",
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
        return build_spike_ace2_proxy_report(
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
            target=spike_ace2_structure_target(),
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
    ) -> FidelityComparisonReport | None:
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
            for index, label in enumerate(_SPIKE_ACE2_LANDMARK_LABELS)
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
        if state.particle_count != len(_SPIKE_ACE2_LANDMARK_LABELS):
            raise ContractValidationError("state particle_count does not match spike_ace2 landmark mapping.")

        reference_target = spike_ace2_structure_target()
        label_to_index = {label: index for index, label in enumerate(_SPIKE_ACE2_LANDMARK_LABELS)}
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
            label="6M0J Shadow Contact Target",
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
        if labels == {"ACE2_alpha1_hotspot", "RBD_ridge_hotspot"}:
            return 1.65
        if labels == {"ACE2_beta_hotspot", "RBD_loop_hotspot"}:
            return 1.35
        if "support" in source_label.lower() or "support" in target_label.lower():
            return 0.95
        if "glycan" in source_label.lower() or "shielded" in target_label.lower():
            return 0.45
        return 0.8

    def _count_cross_contacts(self, state: SimulationState) -> int:
        count = 0
        for source in self.ace2_indices:
            source_position = state.particles.positions[source]
            for target in self.spike_indices:
                target_position = state.particles.positions[target]
                if _distance(source_position, target_position) <= self.contact_distance:
                    count += 1
        return count

    def _count_graph_bridges(self, graph: ConnectivityGraph | None) -> int:
        if graph is None:
            return 0
        ace2 = set(self.ace2_indices)
        spike = set(self.spike_indices)
        return sum(
            1
            for edge in graph.active_edges()
            if (
                edge.source_index in ace2
                and edge.target_index in spike
                or edge.source_index in spike
                and edge.target_index in ace2
            )
        )

    def _stage_label(self, interface_distance: float, contact_count: int, bound: bool) -> str:
        if bound:
            return "Native-Like Recognition"
        if interface_distance <= self.capture_distance and contact_count >= max(4, self.target_contact_count // 2):
            return "Hotspot Capture"
        if interface_distance <= self.initial_interface_distance() - 0.50 or contact_count >= 2:
            return "Long-Range Steering"
        return "Diffusive Search"
