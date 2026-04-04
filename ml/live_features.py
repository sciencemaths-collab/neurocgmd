"""Live feature encoding for uncertainty, control, and adaptive runtime decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from chemistry.interface_logic import ChemistryInterfaceReport
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata, StateId, coerce_scalar
from graph.graph_manager import ConnectivityGraph

if TYPE_CHECKING:
    from validation.fidelity_checks import FidelityComparisonReport
    from validation.structure_metrics import StructureComparisonReport


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _parse_float(text: str, *, label: str) -> float:
    try:
        return float(text.strip())
    except ValueError as error:
        raise ContractValidationError(f"Unable to parse {label} as a float: {text!r}") from error


def _parse_fraction(text: str, *, label: str) -> float:
    numerator_text, separator, denominator_text = text.partition("/")
    if separator != "/":
        raise ContractValidationError(f"Unable to parse {label} as a fraction: {text!r}")
    numerator = _parse_float(numerator_text, label=label)
    denominator = _parse_float(denominator_text, label=label)
    if denominator <= 0.0:
        raise ContractValidationError(f"{label} denominator must be positive.")
    return numerator / denominator


def _structure_metric_map(report: StructureComparisonReport | None) -> dict[str, str]:
    if report is None:
        return {}
    return {metric.label: metric.value for metric in report.metrics}


@dataclass(frozen=True, slots=True)
class LiveFeatureVector(ValidatableComponent):
    """Bounded runtime feature vector for one state."""

    state_id: StateId
    values: FrozenMetadata = field(default_factory=FrozenMetadata)
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", StateId(str(self.state_id)))
        if not isinstance(self.values, FrozenMetadata):
            object.__setattr__(self, "values", FrozenMetadata(self.values))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def value(self, key: str, default: float = 0.0) -> float:
        if key not in self.values:
            return default
        return float(self.values[key])

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for key, value in self.values.to_dict().items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                issues.append(f"values[{key!r}] must be numeric.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "state_id": str(self.state_id),
            "values": self.values.to_dict(),
            "metadata": self.metadata.to_dict(),
        }


@dataclass(slots=True)
class LiveFeatureEncoder(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """[hybrid] Encode live scientific/runtime signals into a bounded feature vector."""

    rmsd_scale: float = 12.0
    dominant_pair_scale: float = 2.5
    name: str = "live_feature_encoder"
    classification: str = "[hybrid]"

    def describe_role(self) -> str:
        return (
            "Encodes graph, structure, fidelity, and chemistry signals into a stable "
            "feature vector for uncertainty models and executive control."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "graph/graph_manager.py",
            "validation/structure_metrics.py",
            "validation/fidelity_checks.py",
            "chemistry/interface_logic.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/chemistry_semantics_and_live_control.md",)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.rmsd_scale <= 0.0:
            issues.append("rmsd_scale must be positive.")
        if self.dominant_pair_scale <= 0.0:
            issues.append("dominant_pair_scale must be positive.")
        return tuple(issues)

    def encode(
        self,
        state: SimulationState,
        graph: ConnectivityGraph,
        *,
        progress: object,
        chemistry_report: ChemistryInterfaceReport | None = None,
        structure_report: StructureComparisonReport | None = None,
        fidelity_report: FidelityComparisonReport | None = None,
    ) -> LiveFeatureVector:
        active_edge_count = len(graph.active_edges())
        adaptive_edge_ratio = len(graph.adaptive_edges()) / active_edge_count if active_edge_count else 0.0
        progress_target_contacts = max(1, int(progress.target_contact_count))
        graph_target_bridges = max(1, int(progress.target_graph_bridge_count))

        structure_metrics = _structure_metric_map(structure_report)
        rmsd = (
            _parse_float(structure_metrics["Atomistic Centroid RMSD"], label="Atomistic Centroid RMSD")
            if "Atomistic Centroid RMSD" in structure_metrics
            else 0.0
        )
        contact_recovery = (
            _parse_fraction(structure_metrics["Contact Recovery"], label="Contact Recovery")
            if "Contact Recovery" in structure_metrics
            else 0.0
        )
        dominant_pair_error = (
            _parse_float(structure_metrics["Dominant Pair Error"], label="Dominant Pair Error")
            if "Dominant Pair Error" in structure_metrics
            else 0.0
        )

        shadow_force_regression = 0.0
        shadow_energy_regression = 0.0
        shadow_max_regression = 0.0
        if fidelity_report is not None:
            shadow_energy_regression = float(
                fidelity_report.metric_for("energy_absolute_error").corrected_error
                > fidelity_report.metric_for("energy_absolute_error").baseline_error
            )
            shadow_force_regression = float(
                fidelity_report.metric_for("force_rms_error").corrected_error
                > fidelity_report.metric_for("force_rms_error").baseline_error
            )
            shadow_max_regression = float(
                fidelity_report.metric_for("max_force_component_error").corrected_error
                > fidelity_report.metric_for("max_force_component_error").baseline_error
            )

        chemistry_values = {
            "chemistry_mean_pair_score": chemistry_report.mean_pair_score if chemistry_report is not None else 0.5,
            "chemistry_favorable_pair_fraction": chemistry_report.favorable_pair_fraction if chemistry_report is not None else 0.5,
            "chemistry_charge_complementarity": chemistry_report.charge_complementarity if chemistry_report is not None else 0.5,
            "chemistry_flexibility_pressure": chemistry_report.flexibility_pressure if chemistry_report is not None else 0.0,
            "chemistry_hotspot_pair_fraction": chemistry_report.hotspot_pair_fraction if chemistry_report is not None else 0.0,
        }

        values = {
            "assembly_score": float(progress.assembly_score),
            "interface_gap": float(progress.interface_distance),
            "cross_contact_fraction": float(progress.cross_contact_count) / progress_target_contacts,
            "graph_bridge_fraction": float(progress.graph_bridge_count) / graph_target_bridges,
            "adaptive_edge_ratio": adaptive_edge_ratio,
            "structure_rmsd_normalized": _clamp(rmsd / self.rmsd_scale, 0.0, 1.0),
            "structure_contact_recovery": contact_recovery,
            "dominant_pair_error_normalized": _clamp(dominant_pair_error / self.dominant_pair_scale, 0.0, 1.0),
            "shadow_energy_regression": shadow_energy_regression,
            "shadow_force_regression": shadow_force_regression,
            "shadow_max_force_regression": shadow_max_regression,
            **chemistry_values,
        }
        return LiveFeatureVector(
            state_id=state.provenance.state_id,
            values=FrozenMetadata(values),
            metadata=FrozenMetadata(
                {
                    "feature_count": len(values),
                    "has_structure_report": structure_report is not None,
                    "has_fidelity_report": fidelity_report is not None,
                    "has_chemistry_report": chemistry_report is not None,
                }
            ),
        )
