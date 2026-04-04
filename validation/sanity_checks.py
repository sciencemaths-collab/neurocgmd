"""Observer-side sanity checks for cross-section output alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from ai_control.controller import ControllerDecision
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata, StateId
from graph.graph_manager import ConnectivityGraph
from ml.residual_model import ResidualPrediction
from physics.forces.composite import ForceEvaluation
from qcloud.qcloud_coupling import QCloudCouplingResult
from topology.system_topology import SystemTopology


@dataclass(frozen=True, slots=True)
class SanityCheckResult(ValidatableComponent):
    """One named validation outcome emitted by the foundation sanity checker."""

    name: str
    passed: bool
    message: str
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name.strip():
            issues.append("name must be a non-empty string.")
        if not self.message.strip():
            issues.append("message must be a non-empty string.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class SanityCheckReport(ValidatableComponent):
    """Aggregated sanity-check report for one state-aligned observer pass."""

    state_id: StateId
    checks: tuple[SanityCheckResult, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", StateId(str(self.state_id)))
        object.__setattr__(self, "checks", tuple(self.checks))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def failed_checks(self) -> tuple[SanityCheckResult, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not str(self.state_id).strip():
            issues.append("state_id must be a non-empty string.")
        if not self.checks:
            issues.append("checks must contain at least one sanity-check result.")
        return tuple(issues)


@dataclass(slots=True)
class FoundationSanityChecker(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Run explicit alignment checks across established and adaptive outputs."""

    energy_tolerance: float = 1e-9
    name: str = "foundation_sanity_checker"
    classification: str = "[established]"

    def describe_role(self) -> str:
        return (
            "Runs observer-side alignment and accounting checks across state, topology, "
            "force, graph, qcloud, ML, and controller outputs."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "core/state.py",
            "topology/system_topology.py",
            "physics/forces/composite.py",
            "graph/graph_manager.py",
            "qcloud/qcloud_coupling.py",
            "ml/residual_model.py",
            "ai_control/controller.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/validation_and_benchmarking.md",
            "docs/sections/section_13_validation_and_benchmarking_suite.md",
        )

    def validate(self) -> tuple[str, ...]:
        if self.energy_tolerance < 0.0:
            return ("energy_tolerance must be non-negative.",)
        return ()

    def run(
        self,
        state: SimulationState,
        *,
        topology: SystemTopology | None = None,
        force_evaluation: ForceEvaluation | None = None,
        graph: ConnectivityGraph | None = None,
        qcloud_result: QCloudCouplingResult | None = None,
        residual_prediction: ResidualPrediction | None = None,
        controller_decision: ControllerDecision | None = None,
    ) -> SanityCheckReport:
        checks: list[SanityCheckResult] = [
            SanityCheckResult(
                name="state_basic_validity",
                passed=state.particle_count > 0 and state.step >= 0 and isfinite(state.time),
                message="State carries a positive particle count, non-negative step, and finite time.",
                metadata={"particle_count": state.particle_count, "step": state.step, "time": state.time},
            )
        ]

        if topology is not None:
            checks.append(
                SanityCheckResult(
                    name="state_topology_alignment",
                    passed=topology.particle_count == state.particle_count,
                    message="Topology particle count matches the observed simulation state.",
                    metadata={"topology_particle_count": topology.particle_count},
                )
            )

        if force_evaluation is not None:
            force_block_ok = len(force_evaluation.forces) == state.particle_count and all(
                len(vector) == 3 and all(isfinite(component) for component in vector)
                for vector in force_evaluation.forces
            )
            checks.append(
                SanityCheckResult(
                    name="force_block_alignment",
                    passed=force_block_ok,
                    message="Force block matches the state particle count and contains finite 3D vectors.",
                    metadata={"force_count": len(force_evaluation.forces)},
                )
            )

            if force_evaluation.component_energies:
                component_sum = sum(float(value) for value in force_evaluation.component_energies.values())
                energy_ok = isfinite(force_evaluation.potential_energy) and abs(
                    component_sum - force_evaluation.potential_energy
                ) <= self.energy_tolerance
                energy_message = "Component-energy accounting matches the reported potential energy."
            else:
                component_sum = 0.0
                energy_ok = isfinite(force_evaluation.potential_energy)
                energy_message = "Potential energy is finite; no component-energy accounting was supplied."

            checks.append(
                SanityCheckResult(
                    name="force_energy_accounting",
                    passed=energy_ok,
                    message=energy_message,
                    metadata={
                        "potential_energy": force_evaluation.potential_energy,
                        "component_energy_sum": component_sum,
                    },
                )
            )

        if graph is not None:
            checks.append(
                SanityCheckResult(
                    name="graph_alignment",
                    passed=graph.particle_count == state.particle_count and graph.step == state.step,
                    message="Connectivity graph particle count and step align with the simulation state.",
                    metadata={
                        "graph_particle_count": graph.particle_count,
                        "graph_step": graph.step,
                        "active_edge_count": len(graph.active_edges()),
                    },
                )
            )

        if qcloud_result is not None:
            region_state_ok = all(region.state_id == state.provenance.state_id for region in qcloud_result.selected_regions)
            correction_region_ids = {correction.region_id for correction in qcloud_result.applied_corrections}
            selected_region_ids = {region.region_id for region in qcloud_result.selected_regions}
            checks.append(
                SanityCheckResult(
                    name="qcloud_alignment",
                    passed=(
                        region_state_ok
                        and correction_region_ids <= selected_region_ids
                        and len(qcloud_result.force_evaluation.forces) == state.particle_count
                    ),
                    message="QCloud result remains aligned to the current state, regions, and force block.",
                    metadata={
                        "selected_region_count": len(qcloud_result.selected_regions),
                        "applied_correction_count": len(qcloud_result.applied_corrections),
                    },
                )
            )

        if residual_prediction is not None:
            checks.append(
                SanityCheckResult(
                    name="residual_alignment",
                    passed=residual_prediction.state_id == state.provenance.state_id,
                    message="Residual prediction targets the current state identifier.",
                    metadata={"prediction_confidence": residual_prediction.confidence},
                )
            )

        if controller_decision is not None:
            checks.append(
                SanityCheckResult(
                    name="controller_alignment",
                    passed=(
                        controller_decision.state_id == state.provenance.state_id
                        and controller_decision.step == state.step
                        and bool(controller_decision.actions)
                    ),
                    message="Controller decision aligns to the current state and exposes at least one action.",
                    metadata={
                        "decision_step": controller_decision.step,
                        "action_count": len(controller_decision.actions),
                    },
                )
            )

        return SanityCheckReport(
            state_id=state.provenance.state_id,
            checks=tuple(checks),
            metadata=FrozenMetadata(
                {
                    "check_count": len(checks),
                    "failed_check_count": sum(0 if check.passed else 1 for check in checks),
                }
            ),
        )
