"""Chemistry-aware executive guidance layered above stability assessment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from chemistry.interface_logic import ChemistryInterfaceReport
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, StateId, coerce_scalar

if TYPE_CHECKING:
    from ml.live_features import LiveFeatureVector


@dataclass(frozen=True, slots=True)
class ChemistryControlGuidance(ValidatableComponent):
    """One explicit chemistry-derived guidance payload for executive control."""

    state_id: StateId
    chemistry_risk: float
    recommend_qcloud_boost: bool
    recommend_ml_boost: bool
    review_required: bool
    focus_compartments: tuple[str, ...] = ()
    summary: str = ""
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", StateId(str(self.state_id)))
        object.__setattr__(self, "chemistry_risk", coerce_scalar(self.chemistry_risk, "chemistry_risk"))
        object.__setattr__(
            self,
            "focus_compartments",
            tuple(
                dict.fromkeys(
                    identifier
                    for identifier in (str(value).strip() for value in self.focus_compartments)
                    if identifier
                )
            ),
        )
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not 0.0 <= self.chemistry_risk <= 1.0:
            issues.append("chemistry_risk must lie in the interval [0, 1].")
        if not self.summary.strip():
            issues.append("summary must be a non-empty string.")
        return tuple(issues)


@dataclass(slots=True)
class ChemistryAwareGovernor(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """[proposed novel] Turn chemistry-interface quality into control guidance."""

    qcloud_boost_threshold: float = 0.55
    ml_boost_threshold: float = 0.45
    review_threshold: float = 0.35
    name: str = "chemistry_aware_governor"
    classification: str = "[proposed novel]"

    def describe_role(self) -> str:
        return (
            "Interprets chemistry-interface quality as a control signal so qcloud, ML, "
            "and review actions can react to chemical implausibility explicitly."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return ("chemistry/interface_logic.py", "ml/live_features.py")

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/chemistry_semantics_and_live_control.md",)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for field_name in ("qcloud_boost_threshold", "ml_boost_threshold", "review_threshold"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                issues.append(f"{field_name} must lie in the interval [0, 1].")
        return tuple(issues)

    def guide(
        self,
        state_id: StateId,
        chemistry_report: ChemistryInterfaceReport | None,
        *,
        live_features: LiveFeatureVector | None = None,
    ) -> ChemistryControlGuidance | None:
        if chemistry_report is None:
            return None

        structure_penalty = live_features.value("structure_rmsd_normalized", 0.0) if live_features is not None else 0.0
        shadow_penalty = live_features.value("shadow_force_regression", 0.0) if live_features is not None else 0.0
        chemistry_risk = min(
            1.0,
            (1.0 - chemistry_report.mean_pair_score) * 0.45
            + (1.0 - chemistry_report.favorable_pair_fraction) * 0.25
            + chemistry_report.flexibility_pressure * 0.20
            + structure_penalty * 0.10
            + shadow_penalty * 0.08,
        )
        return ChemistryControlGuidance(
            state_id=state_id,
            chemistry_risk=chemistry_risk,
            recommend_qcloud_boost=chemistry_risk >= self.qcloud_boost_threshold,
            recommend_ml_boost=chemistry_risk >= self.ml_boost_threshold,
            review_required=chemistry_risk >= self.review_threshold,
            focus_compartments=chemistry_report.compartment_ids,
            summary=(
                "Chemistry-aware control guidance derived from interface plausibility, "
                "flexibility pressure, and current structure/fidelity context."
            ),
            metadata={
                "chemistry_mean_pair_score": chemistry_report.mean_pair_score,
                "chemistry_favorable_pair_fraction": chemistry_report.favorable_pair_fraction,
                "chemistry_flexibility_pressure": chemistry_report.flexibility_pressure,
            },
        )
