"""Generic imported-protein complex scenario built from local PDB input."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from benchmarks.reference_cases import ReferenceComparisonReport
from benchmarks.reference_cases.models import ExperimentalReferenceCase
from config.protein_mapping import ProteinEntityGroup, ProteinMappingConfig
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import ParticleState, SimulationState, ThermodynamicState
from core.types import FrozenMetadata, StateId, Vector3
from forcefields import BaseForceField, ImportedProteinForceFieldBuilder
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
from topology.protein_coarse_mapping import ProteinCoarseMapper
from topology.protein_import_models import ImportedProteinSystem
from validation import (
    FidelityComparisonReport,
    LandmarkObservation,
    ReferenceForceTarget,
    ShadowFidelityAssessor,
    StructureComparisonReport,
    compare_landmark_observations,
)


@dataclass(frozen=True, slots=True)
class ImportedProteinScenarioSpec(ValidatableComponent):
    """Serializable spec for turning a local PDB into one live imported-protein scenario."""

    name: str
    pdb_path: str
    entity_groups: tuple[ProteinEntityGroup, ...]
    mapping_config: ProteinMappingConfig = field(default_factory=ProteinMappingConfig)
    initial_separation_offset_nm: float = 2.4
    approach_velocity_magnitude: float = 0.18
    contact_distance_nm: float = 1.6
    capture_distance_nm: float = 1.9
    bound_distance_nm: float = 1.3
    recommended_time_step: float = 0.028
    recommended_friction: float = 0.7
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_groups", tuple(self.entity_groups))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name.strip():
            issues.append("name must be a non-empty string.")
        if not self.pdb_path.strip():
            issues.append("pdb_path must be a non-empty string.")
        if len(self.entity_groups) < 1:
            issues.append("Imported protein scenarios require at least one entity group.")
        for group in self.entity_groups:
            issues.extend(group.validate())
        issues.extend(self.mapping_config.validate())
        if self.initial_separation_offset_nm <= 0.0:
            issues.append("initial_separation_offset_nm must be strictly positive.")
        if self.approach_velocity_magnitude <= 0.0:
            issues.append("approach_velocity_magnitude must be strictly positive.")
        if self.contact_distance_nm <= 0.0:
            issues.append("contact_distance_nm must be strictly positive.")
        if self.capture_distance_nm <= self.bound_distance_nm:
            issues.append("capture_distance_nm must exceed bound_distance_nm.")
        if self.bound_distance_nm <= 0.0:
            issues.append("bound_distance_nm must be strictly positive.")
        if self.recommended_time_step <= 0.0:
            issues.append("recommended_time_step must be strictly positive.")
        if self.recommended_friction < 0.0:
            issues.append("recommended_friction must be non-negative.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "pdb_path": self.pdb_path,
            "entity_groups": [group.to_dict() for group in self.entity_groups],
            "mapping_config": self.mapping_config.to_dict(),
            "initial_separation_offset_nm": self.initial_separation_offset_nm,
            "approach_velocity_magnitude": self.approach_velocity_magnitude,
            "contact_distance_nm": self.contact_distance_nm,
            "capture_distance_nm": self.capture_distance_nm,
            "bound_distance_nm": self.bound_distance_nm,
            "recommended_time_step": self.recommended_time_step,
            "recommended_friction": self.recommended_friction,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ImportedProteinScenarioSpec":
        return cls(
            name=str(data["name"]),
            pdb_path=str(data["pdb_path"]),
            entity_groups=tuple(
                ProteinEntityGroup.from_dict(item)
                for item in data.get("entity_groups", ())
            ),
            mapping_config=ProteinMappingConfig.from_dict(data.get("mapping_config", {})),
            initial_separation_offset_nm=float(data.get("initial_separation_offset_nm", 2.4)),
            approach_velocity_magnitude=float(data.get("approach_velocity_magnitude", 0.18)),
            contact_distance_nm=float(data.get("contact_distance_nm", 1.6)),
            capture_distance_nm=float(data.get("capture_distance_nm", 1.9)),
            bound_distance_nm=float(data.get("bound_distance_nm", 1.3)),
            recommended_time_step=float(data.get("recommended_time_step", 0.028)),
            recommended_friction=float(data.get("recommended_friction", 0.7)),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


def _centroid(points: tuple[Vector3, ...]) -> Vector3:
    count = len(points)
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


@dataclass(frozen=True, slots=True)
class ImportedProteinComplexScenario(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """[hybrid] Arbitrary complex scenario built from a local imported protein structure."""

    spec: ImportedProteinScenarioSpec
    shadow_tuner: ProteinShadowTuner = field(default_factory=ProteinShadowTuner)
    name: str = "imported_protein_complex"
    classification: str = "[hybrid]"
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", self.spec.name)
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Turns arbitrary local protein complexes into live scenarios without hand-authored "
            "benchmark code, so new proteins can enter validation and transfer tuning through "
            "the same architecture as the curated demos."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "topology/protein_coarse_mapping.py",
            "forcefields/protein_import_forcefield.py",
            "qcloud/protein_shadow_tuning.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/arbitrary_protein_input_pipeline.md",)

    def validate(self) -> tuple[str, ...]:
        issues = list(self.spec.validate())
        issues.extend(self.shadow_tuner.validate())
        return tuple(issues)

    def reference_case(self) -> ExperimentalReferenceCase | None:
        return None

    def build_reference_report(self, progress: ComplexAssemblyProgress) -> ReferenceComparisonReport | None:
        del progress
        return None

    def build_setup(self) -> ComplexAssemblySetup:
        imported = self._imported_system()
        forcefield = self._forcefield_builder().build(
            imported.topology,
            scenario_label=self.spec.name,
            reference_label=imported.reference_target.source_pdb_id,
            metadata={
                "source_path": imported.source_path,
                "imported": True,
            },
        )
        shadow_bundle = self._shadow_runtime_bundle(imported.topology)
        initial_particles = self._build_initial_particles(imported)
        entity_ids = imported.entity_ids()
        if len(entity_ids) >= 2:
            objective = (
                f"Rebind the imported {entity_ids[0]}-{entity_ids[1]} complex "
                "from a displaced coarse start while tracking recovery against the local reference scaffold."
            )
            summary = (
                "Arbitrary imported-protein complex built directly from a local PDB and mapped "
                "into the repository's coarse, shadow-aware, benchmarkable substrate."
            )
        else:
            objective = (
                f"Stabilize the imported {entity_ids[0]} fold under hybrid production dynamics "
                "while tracking structural drift against the local reference scaffold."
            )
            summary = (
                "Arbitrary imported single-protein system built directly from a local PDB and "
                "mapped into the repository's coarse, shadow-aware, benchmarkable substrate."
            )
        return ComplexAssemblySetup(
            title=f"NeuroCGMD Live Dashboard | {self.spec.name}",
            summary=summary,
            objective=objective,
            topology=imported.topology,
            forcefield=forcefield,
            initial_particles=initial_particles,
            thermodynamics=ThermodynamicState(),
            compartments=imported.compartments,
            focus_compartments=entity_ids[:2] or entity_ids,
            integrator_time_step=shadow_bundle.dynamics_recommendation.time_step,
            integrator_friction=shadow_bundle.dynamics_recommendation.friction_coefficient,
            metadata={
                "imported_scenario": True,
                "source_path": imported.source_path,
                "entity_ids": imported.entity_ids(),
                "shadow_tuning_preset": shadow_bundle.dynamics_recommendation.metadata["preset_name"],
            },
        )

    def measure_progress(
        self,
        state: SimulationState,
        *,
        graph: ConnectivityGraph | None = None,
    ) -> ComplexAssemblyProgress:
        imported = self._imported_system()
        if len(imported.entity_ids()) == 1:
            return self._measure_single_entity_progress(imported, state, graph=graph)
        interface_pair = self._interface_pair(imported)
        first_entity, second_entity = imported.entity_ids()[:2]
        first_indices = imported.bead_indices_for_entity(first_entity)
        second_indices = imported.bead_indices_for_entity(second_entity)
        interface_distance = min(
            _distance(state.particles.positions[left], state.particles.positions[right])
            for left in first_indices
            for right in second_indices
        )
        cross_contact_count = self._count_cross_contacts(state, first_indices, second_indices)
        graph_bridge_count = self._count_graph_bridges(graph, set(first_indices), set(second_indices))
        target_contact_count = max(1, len(imported.reference_target.interface_contacts))
        target_graph_bridge_count = max(1, min(target_contact_count, 4))
        closure_progress = 1.0 - _normalized_progress(
            interface_distance,
            self.spec.bound_distance_nm,
            self.spec.initial_separation_offset_nm + self.spec.capture_distance_nm,
        )
        contact_progress = min(1.0, cross_contact_count / target_contact_count)
        bridge_progress = min(1.0, graph_bridge_count / target_graph_bridge_count)
        assembly_score = min(1.0, 0.5 * closure_progress + 0.35 * contact_progress + 0.15 * bridge_progress)
        if interface_distance <= self.spec.bound_distance_nm:
            stage_label = "Imported Bound"
        elif interface_distance <= self.spec.capture_distance_nm:
            stage_label = "Imported Capture"
        else:
            stage_label = "Imported Search"
        return ComplexAssemblyProgress(
            interface_pair=interface_pair,
            initial_interface_distance=self._initial_interface_distance(imported, interface_pair),
            interface_distance=interface_distance,
            capture_distance=self.spec.capture_distance_nm,
            bound_distance=self.spec.bound_distance_nm,
            contact_distance=self.spec.contact_distance_nm,
            cross_contact_count=cross_contact_count,
            graph_bridge_count=graph_bridge_count,
            target_contact_count=target_contact_count,
            target_graph_bridge_count=target_graph_bridge_count,
            stage_label=stage_label,
            assembly_score=assembly_score,
            bound=interface_distance <= self.spec.bound_distance_nm,
            metadata={
                "scenario_name": self.spec.name,
                "source_pdb_id": imported.reference_target.source_pdb_id,
            },
        )

    def _measure_single_entity_progress(
        self,
        imported: ImportedProteinSystem,
        state: SimulationState,
        *,
        graph: ConnectivityGraph | None = None,
    ) -> ComplexAssemblyProgress:
        entity_id = imported.entity_ids()[0]
        indices = imported.bead_indices_for_entity(entity_id)
        reference_positions = imported.particles.positions
        current_positions = state.particles.positions
        rmsd = (
            sum(
                (
                    (current_positions[index][0] - reference_positions[index][0]) ** 2
                    + (current_positions[index][1] - reference_positions[index][1]) ** 2
                    + (current_positions[index][2] - reference_positions[index][2]) ** 2
                )
                for index in indices
            )
            / len(indices)
        ) ** 0.5
        initial_rmsd = self._initial_single_entity_rmsd(imported, indices)
        target_contacts = self._single_entity_reference_contacts(imported, indices)
        recovered_contacts = sum(
            1
            for left, right, max_distance in target_contacts
            if _distance(current_positions[left], current_positions[right]) <= max_distance
        )
        graph_bridge_count = self._count_graph_bridges(graph, set(indices), set(indices))
        target_graph_bridge_count = max(1, min(len(target_contacts), len(indices)))
        closure_progress = 1.0 - _normalized_progress(
            rmsd,
            0.18,
            max(initial_rmsd, 0.35),
        )
        contact_progress = min(1.0, recovered_contacts / max(1, len(target_contacts)))
        bridge_progress = min(1.0, graph_bridge_count / target_graph_bridge_count)
        stability_score = min(1.0, 0.55 * closure_progress + 0.30 * contact_progress + 0.15 * bridge_progress)
        stage_label = "Native Stability" if rmsd <= 0.35 else ("Soft Drift" if rmsd <= 0.75 else "Large Drift")
        interface_pair = (indices[0], indices[-1]) if len(indices) >= 2 else (indices[0], indices[0])
        return ComplexAssemblyProgress(
            interface_pair=interface_pair,
            initial_interface_distance=max(initial_rmsd, 0.18),
            interface_distance=rmsd,
            capture_distance=max(0.5, initial_rmsd),
            bound_distance=0.18,
            contact_distance=self.spec.contact_distance_nm,
            cross_contact_count=recovered_contacts,
            graph_bridge_count=graph_bridge_count,
            target_contact_count=max(1, len(target_contacts)),
            target_graph_bridge_count=target_graph_bridge_count,
            stage_label=stage_label,
            assembly_score=stability_score,
            bound=rmsd <= 0.35,
            metadata={
                "scenario_name": self.spec.name,
                "mode": "single_entity_stability",
                "entity_id": entity_id,
            },
        )

    def build_structure_report(
        self,
        state: SimulationState,
        *,
        progress: ComplexAssemblyProgress | None = None,
    ) -> StructureComparisonReport:
        del progress
        imported = self._imported_system()
        observations = tuple(
            LandmarkObservation(
                label=block.label,
                position=state.particles.positions[index],
                metadata={"particle_index": index, "entity_id": block.entity_id},
            )
            for index, block in enumerate(imported.bead_blocks)
        )
        return compare_landmark_observations(
            state_id=state.provenance.state_id,
            target=imported.reference_target,
            observations=observations,
        )

    def build_qcloud_correction_model(self) -> QCloudCorrectionModel | None:
        return self._shadow_runtime_bundle(self._imported_system().topology).build_correction_model()

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

    def _imported_system(self) -> ImportedProteinSystem:
        return ProteinCoarseMapper(mapping_config=self.spec.mapping_config).import_from_pdb(
            pdb_path=self.spec.pdb_path,
            entity_groups=self.spec.entity_groups,
            structure_id=self.spec.name,
        )

    def _forcefield_builder(self) -> ImportedProteinForceFieldBuilder:
        return ImportedProteinForceFieldBuilder()

    def _shadow_runtime_bundle(self, topology) -> ProteinShadowRuntimeBundle:
        return self.shadow_tuner.build_runtime_bundle(
            topology=topology,
            scenario_label=self.spec.name,
            base_time_step=self.spec.recommended_time_step,
            base_friction=self.spec.recommended_friction,
            reference_label=self.spec.name,
            metadata={
                "source_path": str(Path(self.spec.pdb_path).expanduser().resolve()),
                "imported": True,
            },
        )

    def _build_initial_particles(self, imported: ImportedProteinSystem) -> ParticleState:
        positions = self._displaced_positions(imported)
        velocities = self._initial_velocities(imported)
        return ParticleState(
            positions=positions,
            masses=imported.particles.masses,
            velocities=velocities,
            labels=imported.particles.labels,
        )

    def _displaced_positions(self, imported: ImportedProteinSystem) -> tuple[Vector3, ...]:
        entity_ids = imported.entity_ids()
        entity_centers = {
            entity_id: _centroid(
                tuple(
                    imported.particles.positions[index]
                    for index in imported.bead_indices_for_entity(entity_id)
                )
            )
            for entity_id in entity_ids
        }
        base_offsets = {}
        if len(entity_ids) == 1:
            entity_id = entity_ids[0]
            base_offsets[entity_id] = (0.0, 0.0, 0.0)
        elif len(entity_ids) == 2:
            half_offset = self.spec.initial_separation_offset_nm / 2.0
            base_offsets[entity_ids[0]] = (-half_offset, 0.0, 0.0)
            base_offsets[entity_ids[1]] = (half_offset, 0.0, 0.0)
        else:
            for entity_index, entity_id in enumerate(entity_ids):
                base_offsets[entity_id] = (
                    (entity_index - (len(entity_ids) - 1) / 2.0) * self.spec.initial_separation_offset_nm,
                    0.0,
                    0.0,
                )
        displaced_positions: list[Vector3] = []
        for index, block in enumerate(imported.bead_blocks):
            reference_position = imported.particles.positions[index]
            entity_center = entity_centers[block.entity_id]
            relative = (
                reference_position[0] - entity_center[0],
                reference_position[1] - entity_center[1],
                reference_position[2] - entity_center[2],
            )
            offset = base_offsets[block.entity_id]
            if len(entity_ids) == 1:
                relative = (
                    relative[0] * 1.08,
                    relative[1] * 1.08,
                    relative[2] * 1.08,
                )
            displaced_positions.append(
                (
                    offset[0] + relative[0],
                    offset[1] + relative[1],
                    offset[2] + relative[2],
                )
            )
        return tuple(displaced_positions)

    def _initial_velocities(self, imported: ImportedProteinSystem) -> tuple[Vector3, ...]:
        entity_ids = imported.entity_ids()
        if len(entity_ids) < 2:
            return tuple((0.0, 0.0, 0.0) for _ in imported.bead_blocks)
        direction_map = {
            entity_ids[0]: 1.0,
            entity_ids[1]: -1.0,
        }
        for entity_id in entity_ids[2:]:
            direction_map[entity_id] = -1.0
        return tuple(
            (
                direction_map.get(block.entity_id, 0.0) * self.spec.approach_velocity_magnitude,
                0.0,
                0.0,
            )
            for block in imported.bead_blocks
        )

    def _interface_pair(self, imported: ImportedProteinSystem) -> tuple[int, int]:
        dominant_pair = tuple(imported.reference_target.metadata.get("dominant_interface_pair", ()))
        label_to_index = {
            block.label: index
            for index, block in enumerate(imported.bead_blocks)
        }
        if len(dominant_pair) == 2 and dominant_pair[0] in label_to_index and dominant_pair[1] in label_to_index:
            return (label_to_index[dominant_pair[0]], label_to_index[dominant_pair[1]])

        first_entity, second_entity = imported.entity_ids()[:2]
        return min(
            (
                (left, right)
                for left in imported.bead_indices_for_entity(first_entity)
                for right in imported.bead_indices_for_entity(second_entity)
            ),
            key=lambda pair: _distance(
                imported.particles.positions[pair[0]],
                imported.particles.positions[pair[1]],
            ),
        )

    def _initial_interface_distance(
        self,
        imported: ImportedProteinSystem,
        interface_pair: tuple[int, int],
    ) -> float:
        displaced_positions = self._displaced_positions(imported)
        return _distance(displaced_positions[interface_pair[0]], displaced_positions[interface_pair[1]])

    def _count_cross_contacts(
        self,
        state: SimulationState,
        first_indices: tuple[int, ...],
        second_indices: tuple[int, ...],
    ) -> int:
        return sum(
            1
            for left in first_indices
            for right in second_indices
            if _distance(state.particles.positions[left], state.particles.positions[right]) <= self.spec.contact_distance_nm
        )

    def _count_graph_bridges(
        self,
        graph: ConnectivityGraph | None,
        first_indices: set[int],
        second_indices: set[int],
    ) -> int:
        if graph is None:
            return 0
        return sum(
            1
            for edge in graph.active_edges()
            if (
                edge.source_index in first_indices
                and edge.target_index in second_indices
                and edge.source_index != edge.target_index
                or edge.source_index in second_indices
                and edge.target_index in first_indices
                and edge.source_index != edge.target_index
            )
        )

    def _single_entity_reference_contacts(
        self,
        imported: ImportedProteinSystem,
        indices: tuple[int, ...],
    ) -> tuple[tuple[int, int, float], ...]:
        contacts: list[tuple[float, int, int, float]] = []
        positions = imported.particles.positions
        for left_offset, left_index in enumerate(indices):
            for right_index in indices[left_offset + 2 :]:
                target_distance = _distance(positions[left_index], positions[right_index])
                if target_distance <= self.spec.capture_distance_nm:
                    contacts.append(
                        (
                            target_distance,
                            left_index,
                            right_index,
                            target_distance + self.spec.contact_distance_nm * 0.15,
                        )
                    )
        contacts.sort(key=lambda item: item[0])
        return tuple((left_index, right_index, max_distance) for _, left_index, right_index, max_distance in contacts[:12])

    def _initial_single_entity_rmsd(
        self,
        imported: ImportedProteinSystem,
        indices: tuple[int, ...],
    ) -> float:
        displaced_positions = self._displaced_positions(imported)
        reference_positions = imported.particles.positions
        return (
            sum(
                (
                    (displaced_positions[index][0] - reference_positions[index][0]) ** 2
                    + (displaced_positions[index][1] - reference_positions[index][1]) ** 2
                    + (displaced_positions[index][2] - reference_positions[index][2]) ** 2
                )
                for index in indices
            )
            / len(indices)
        ) ** 0.5

    def _shadow_reference_force_target(self, state: SimulationState) -> ReferenceForceTarget:
        imported = self._imported_system()
        label_to_index = {
            block.label: index
            for index, block in enumerate(imported.bead_blocks)
        }
        accumulated_forces = [[0.0, 0.0, 0.0] for _ in range(state.particle_count)]
        total_energy = 0.0
        for contact in imported.reference_target.interface_contacts:
            source_index = label_to_index[contact.source_label]
            target_index = label_to_index[contact.target_label]
            source_position = state.particles.positions[source_index]
            target_position = state.particles.positions[target_index]
            displacement = tuple(target_position[axis] - source_position[axis] for axis in range(3))
            current_distance = max(_distance(source_position, target_position), 1.0e-8)
            target_distance = float(contact.metadata.get("target_distance", contact.max_distance))
            stiffness = self._contact_target_stiffness(imported, contact.source_label, contact.target_label)
            delta = current_distance - target_distance
            total_energy += -0.5 * stiffness * delta * delta
            magnitude = stiffness * delta / current_distance
            for axis in range(3):
                force_component = magnitude * displacement[axis]
                accumulated_forces[source_index][axis] += force_component
                accumulated_forces[target_index][axis] -= force_component
        return ReferenceForceTarget(
            label=f"{imported.reference_target.source_pdb_id} Imported Shadow Contact Target",
            potential_energy=total_energy,
            forces=tuple(tuple(vector) for vector in accumulated_forces),
            metadata={
                "imported": True,
                "source_path": imported.source_path,
                "contact_count": len(imported.reference_target.interface_contacts),
            },
        )

    def _contact_target_stiffness(
        self,
        imported: ImportedProteinSystem,
        source_label: str,
        target_label: str,
    ) -> float:
        label_to_block = {block.label: block for block in imported.bead_blocks}
        source_type = label_to_block[source_label].bead_type
        target_type = label_to_block[target_label].bead_type
        types = {source_type, target_type}
        if "hotspot" in types:
            return 1.45
        if types == {"basic", "acidic"}:
            return 1.55
        if "core" in types:
            return 1.10
        if "shield" in types or "loop" in types:
            return 0.65
        return 0.95
