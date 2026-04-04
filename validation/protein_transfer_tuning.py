"""Multi-protein transfer tuning for shared spatial-semantic shadow priors."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from statistics import fmean
from typing import TYPE_CHECKING

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, coerce_scalar
from qcloud import ProteinShadowTuner, ProteinShadowTuningPreset
from scripts.live_dashboard import (
    advance_demo,
    build_dashboard_context_for_scenario,
    collect_dashboard_step_data,
)
from sampling.scenarios import ImportedProteinScenarioSpec
from validation.scientific_validation import (
    ScientificValidationReport,
    ScientificValidationRunner,
    ScientificValidationSample,
    ScientificValidationSummary,
)

if TYPE_CHECKING:
    from sampling.scenarios import DashboardScenario


def _preset_metadata(preset: ProteinShadowTuningPreset) -> dict[str, object]:
    return {
        "name": preset.name,
        "minimum_site_distance": preset.minimum_site_distance,
        "max_interaction_distance": preset.max_interaction_distance,
        "energy_scale": preset.energy_scale,
        "electrostatic_scale": preset.electrostatic_scale,
        "spatial_profile_preferred_distance_scale": preset.spatial_profile_preferred_distance_scale,
        "spatial_profile_distance_tolerance_scale": preset.spatial_profile_distance_tolerance_scale,
        "spatial_profile_attraction_scale": preset.spatial_profile_attraction_scale,
        "spatial_profile_repulsion_scale": preset.spatial_profile_repulsion_scale,
        "spatial_profile_directional_scale": preset.spatial_profile_directional_scale,
        "spatial_profile_chemistry_scale": preset.spatial_profile_chemistry_scale,
        "spatial_energy_scale": preset.spatial_energy_scale,
        "spatial_repulsion_scale": preset.spatial_repulsion_scale,
        "spatial_alignment_floor": preset.spatial_alignment_floor,
        "spatial_max_pair_energy_magnitude": preset.spatial_max_pair_energy_magnitude,
        "spatial_max_pair_force_magnitude": preset.spatial_max_pair_force_magnitude,
        "time_step_multiplier": preset.time_step_multiplier,
        "friction_multiplier": preset.friction_multiplier,
        "metadata": preset.metadata.to_dict(),
    }


def _normalized_scale_grid(values: tuple[float, ...], *, label: str) -> tuple[float, ...]:
    normalized = tuple(coerce_scalar(value, label) for value in values)
    if not normalized:
        raise ContractValidationError(f"{label} must contain at least one value.")
    if any(value <= 0.0 for value in normalized):
        raise ContractValidationError(f"{label} values must all be strictly positive.")
    return normalized


def _build_validation_sample_from_context(
    context: object,
    *,
    replicate_index: int,
    sample_step: int,
    benchmark_repeats: int,
) -> ScientificValidationSample:
    step_data = collect_dashboard_step_data(context, benchmark_repeats=benchmark_repeats)
    scenario = context.scenario
    structure_report = scenario.build_structure_report(
        step_data.state,
        progress=step_data.progress,
    )
    fidelity_report = scenario.build_fidelity_report(
        step_data.state,
        baseline_evaluation=step_data.base_evaluation,
        corrected_evaluation=step_data.qcloud_result.force_evaluation,
        progress=step_data.progress,
    )
    if structure_report is None or fidelity_report is None:
        raise ContractValidationError(
            f"Scenario {scenario.name!r} must provide structure and fidelity reports for transfer tuning."
        )
    return ScientificValidationSample.from_reports(
        replicate_index=replicate_index,
        sample_step=sample_step,
        state_step=step_data.state.step,
        time=step_data.state.time,
        progress=step_data.progress,
        structure_report=structure_report,
        fidelity_report=fidelity_report,
        benchmark_report=step_data.benchmark_report,
        metadata={
            "scenario": scenario.name,
            "state_id": str(step_data.state.provenance.state_id),
            "transfer_tuning": True,
        },
    )


@dataclass(frozen=True, slots=True)
class SpatialTransferCandidate(ValidatableComponent):
    """One candidate shared preset for multi-protein transfer tuning."""

    name: str
    preset: ProteinShadowTuningPreset
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
        issues.extend(self.preset.validate())
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "preset": _preset_metadata(self.preset),
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ScenarioTransferScore(ValidatableComponent):
    """Per-scenario transfer summary for one shared preset candidate."""

    scenario_name: str
    summary: ScientificValidationSummary
    combined_score: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "combined_score", coerce_scalar(self.combined_score, "combined_score"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.scenario_name.strip():
            issues.append("scenario_name must be a non-empty string.")
        issues.extend(self.summary.validate())
        if not 0.0 <= self.combined_score <= 1.0:
            issues.append("combined_score must be between 0 and 1.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_name": self.scenario_name,
            "summary": self.summary.to_dict(),
            "combined_score": self.combined_score,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ProteinTransferCandidateResult(ValidatableComponent):
    """Aggregated multi-protein result for one candidate preset."""

    candidate: SpatialTransferCandidate
    scenario_scores: tuple[ScenarioTransferScore, ...]
    mean_score: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_scores", tuple(self.scenario_scores))
        object.__setattr__(self, "mean_score", coerce_scalar(self.mean_score, "mean_score"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        issues.extend(self.candidate.validate())
        if not self.scenario_scores:
            issues.append("scenario_scores must contain at least one scenario result.")
        if not 0.0 <= self.mean_score <= 1.0:
            issues.append("mean_score must be between 0 and 1.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_dict(),
            "scenario_scores": [score.to_dict() for score in self.scenario_scores],
            "mean_score": self.mean_score,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ProteinTransferTuningReport(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Transfer-tuning report covering a shared preset across multiple proteins."""

    title: str
    candidates: tuple[ProteinTransferCandidateResult, ...]
    scenario_names: tuple[str, ...]
    classification: str = "[hybrid]"
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "scenario_names", tuple(self.scenario_names))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Compares shared spatial-semantic presets across multiple protein benchmarks so "
            "transfer improvements are explicit instead of hidden inside one hand-tuned case."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "validation/scientific_validation.py",
            "sampling/scenarios/spike_ace2.py",
            "sampling/scenarios/barnase_barstar.py",
            "qcloud/protein_shadow_tuning.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/multi_protein_spatial_transfer_tuning.md",)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.title.strip():
            issues.append("title must be a non-empty string.")
        if not self.candidates:
            issues.append("candidates must contain at least one result.")
        if not self.scenario_names:
            issues.append("scenario_names must contain at least one scenario.")
        return tuple(issues)

    def best_candidate(self) -> ProteinTransferCandidateResult:
        return max(self.candidates, key=lambda result: result.mean_score)

    def baseline_candidate(self) -> ProteinTransferCandidateResult | None:
        for candidate in self.candidates:
            if candidate.candidate.metadata.get("is_baseline") is True:
                return candidate
        for candidate in self.candidates:
            if candidate.candidate.name == "baseline":
                return candidate
        return None

    def score_margin_over_baseline(self) -> float | None:
        baseline = self.baseline_candidate()
        if baseline is None:
            return None
        return self.best_candidate().mean_score - baseline.mean_score

    def top_candidates(self, *, limit: int = 5) -> tuple[ProteinTransferCandidateResult, ...]:
        if limit <= 0:
            raise ContractValidationError("limit must be strictly positive.")
        return tuple(sorted(self.candidates, key=lambda result: result.mean_score, reverse=True)[:limit])

    def to_dict(self) -> dict[str, object]:
        best = self.best_candidate()
        baseline = self.baseline_candidate()
        return {
            "title": self.title,
            "classification": self.classification,
            "scenario_names": list(self.scenario_names),
            "best_candidate_name": best.candidate.name,
            "best_mean_score": best.mean_score,
            "baseline_candidate_name": baseline.candidate.name if baseline is not None else None,
            "baseline_mean_score": baseline.mean_score if baseline is not None else None,
            "best_margin_over_baseline": self.score_margin_over_baseline(),
            "candidate_count": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metadata": self.metadata.to_dict(),
        }


@dataclass(slots=True)
class ProteinTransferTuningRunner(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Run deterministic transfer tuning of shared spatial priors across protein benchmarks."""

    replicates: int = 2
    steps_per_replicate: int = 12
    sample_interval: int = 4
    benchmark_repeats: int = 1
    scenario_names: tuple[str, ...] = ("spike_ace2", "barnase_barstar")
    imported_scenario_specs: tuple[ImportedProteinScenarioSpec, ...] = ()
    rmsd_ceiling: float = 15.0
    runtime_reference_seconds: float = 0.005
    name: str = "protein_transfer_tuning_runner"
    classification: str = "[hybrid]"

    def __post_init__(self) -> None:
        self.scenario_names = tuple(self.scenario_names)
        self.imported_scenario_specs = tuple(self.imported_scenario_specs)

    def describe_role(self) -> str:
        return (
            "Tunes shared spatial-semantic shadow priors across multiple protein benchmarks "
            "using the same scientific-validation contracts that drive the live dashboard."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "validation/scientific_validation.py",
            "scripts/live_dashboard.py",
            "qcloud/protein_shadow_tuning.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/multi_protein_spatial_transfer_tuning.md",)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.replicates <= 0:
            issues.append("replicates must be strictly positive.")
        if self.steps_per_replicate < 0:
            issues.append("steps_per_replicate must be non-negative.")
        if self.sample_interval <= 0:
            issues.append("sample_interval must be strictly positive.")
        if self.benchmark_repeats <= 0:
            issues.append("benchmark_repeats must be strictly positive.")
        if not self.scenario_names and not self.imported_scenario_specs:
            issues.append("At least one built-in or imported scenario must be configured.")
        if len(self.scenario_names) != len(set(self.scenario_names)):
            issues.append("scenario_names must be unique.")
        imported_names = tuple(spec.name for spec in self.imported_scenario_specs)
        if len(imported_names) != len(set(imported_names)):
            issues.append("imported_scenario_specs must have unique names.")
        if set(imported_names) & set(self.scenario_names):
            issues.append("Imported scenario names must not collide with built-in scenario_names.")
        for spec in self.imported_scenario_specs:
            issues.extend(spec.validate())
        if self.rmsd_ceiling <= 0.0:
            issues.append("rmsd_ceiling must be strictly positive.")
        if self.runtime_reference_seconds <= 0.0:
            issues.append("runtime_reference_seconds must be strictly positive.")
        return tuple(issues)

    def tune(
        self,
        candidates: tuple[SpatialTransferCandidate, ...],
        *,
        title: str = "Protein Transfer Spatial Tuning Report",
        metadata: FrozenMetadata | dict[str, object] | None = None,
    ) -> ProteinTransferTuningReport:
        issues = list(self.validate())
        if not candidates:
            issues.append("candidates must contain at least one spatial transfer candidate.")
        for candidate in candidates:
            issues.extend(candidate.validate())
        if issues:
            raise ContractValidationError("; ".join(issues))

        runner = ScientificValidationRunner(
            replicates=self.replicates,
            steps_per_replicate=self.steps_per_replicate,
            sample_interval=self.sample_interval,
        )

        candidate_results: list[ProteinTransferCandidateResult] = []
        scenario_entries = self._scenario_entries()
        for candidate in candidates:
            scenario_scores: list[ScenarioTransferScore] = []
            for scenario_name, scenario in (
                (scenario_name, self._build_scenario(entry, candidate.preset))
                for scenario_name, entry in scenario_entries
            ):
                report = runner.run(
                    scenario_name=scenario_name,
                    context_factory=lambda _replicate_index, tuned_scenario=scenario: build_dashboard_context_for_scenario(
                        tuned_scenario
                    ),
                    advance_context=lambda context, steps: advance_demo(context, steps=steps),
                    sample_context=lambda context, replicate_index, sample_step: _build_validation_sample_from_context(
                        context,
                        replicate_index=replicate_index,
                        sample_step=sample_step,
                        benchmark_repeats=self.benchmark_repeats,
                    ),
                    metadata={
                        "scenario": scenario_name,
                        "transfer_candidate": candidate.name,
                    },
                    title=f"{scenario_name} transfer validation",
                )
                scenario_scores.append(
                    ScenarioTransferScore(
                        scenario_name=scenario_name,
                        summary=report.summary,
                        combined_score=self._score_report(report),
                        metadata={
                            "candidate": candidate.name,
                            "sample_count": len(report.samples),
                        },
                    )
                )
            candidate_results.append(
                ProteinTransferCandidateResult(
                    candidate=candidate,
                    scenario_scores=tuple(scenario_scores),
                    mean_score=fmean(score.combined_score for score in scenario_scores),
                    metadata={"scenario_count": len(scenario_scores)},
                )
            )

        return ProteinTransferTuningReport(
            title=title,
            candidates=tuple(sorted(candidate_results, key=lambda result: result.mean_score, reverse=True)),
            scenario_names=tuple(scenario_name for scenario_name, _entry in scenario_entries),
            metadata=FrozenMetadata(metadata),
        )

    def _scenario_entries(self) -> tuple[tuple[str, object], ...]:
        return tuple((scenario_name, scenario_name) for scenario_name in self.scenario_names) + tuple(
            (spec.name, spec) for spec in self.imported_scenario_specs
        )

    def _build_scenario(
        self,
        scenario_entry: object,
        preset: ProteinShadowTuningPreset,
    ) -> "DashboardScenario":
        from sampling.scenarios import (
            BarnaseBarstarScenario,
            ImportedProteinComplexScenario,
            SpikeAce2Scenario,
        )

        shadow_tuner = ProteinShadowTuner(preset=preset)
        if isinstance(scenario_entry, ImportedProteinScenarioSpec):
            return ImportedProteinComplexScenario(spec=scenario_entry, shadow_tuner=shadow_tuner)
        if scenario_entry == "spike_ace2":
            return SpikeAce2Scenario(shadow_tuner=shadow_tuner)
        if scenario_entry == "barnase_barstar":
            return BarnaseBarstarScenario(shadow_tuner=shadow_tuner)
        raise ContractValidationError(f"Unsupported transfer-tuning scenario: {scenario_entry!r}")

    def _score_report(self, report: ScientificValidationReport) -> float:
        summary = report.summary
        structural_score = max(
            0.0,
            min(1.0, 1.0 - summary.final_atomistic_centroid_rmsd_mean / self.rmsd_ceiling),
        )
        runtime_score = 1.0 / (
            1.0 + summary.mean_benchmark_total_seconds / self.runtime_reference_seconds
        )
        score = (
            0.28 * summary.force_rms_improvement_rate
            + 0.18 * summary.max_force_component_improvement_rate
            + 0.18 * summary.full_shadow_improvement_rate
            + 0.12 * summary.energy_improvement_rate
            + 0.12 * summary.final_contact_recovery_mean
            + 0.07 * summary.final_assembly_score_mean
            + 0.03 * structural_score
            + 0.02 * runtime_score
        )
        return max(0.0, min(1.0, score))


def build_spatial_transfer_candidate_grid(
    base_preset: ProteinShadowTuningPreset | None = None,
    *,
    include_baseline: bool = True,
    preferred_distance_scales: tuple[float, ...] = (1.0,),
    distance_tolerance_scales: tuple[float, ...] = (1.0,),
    attraction_scales: tuple[float, ...] = (0.95, 1.0, 1.05),
    repulsion_scales: tuple[float, ...] = (0.95, 1.0, 1.05),
    directional_scales: tuple[float, ...] = (1.0,),
    chemistry_scales: tuple[float, ...] = (0.90, 1.0, 1.10),
    local_field_energy_scales: tuple[float, ...] = (1.0,),
    local_field_repulsion_scales: tuple[float, ...] = (1.0,),
    alignment_floors: tuple[float, ...] = (0.24, 0.28, 0.32),
) -> tuple[SpatialTransferCandidate, ...]:
    """Build a deterministic candidate grid around the shared spatial shadow preset."""

    preset = base_preset or ProteinShadowTuningPreset()
    preferred_distance_scales = _normalized_scale_grid(preferred_distance_scales, label="preferred_distance_scales")
    distance_tolerance_scales = _normalized_scale_grid(distance_tolerance_scales, label="distance_tolerance_scales")
    attraction_scales = _normalized_scale_grid(attraction_scales, label="attraction_scales")
    repulsion_scales = _normalized_scale_grid(repulsion_scales, label="repulsion_scales")
    directional_scales = _normalized_scale_grid(directional_scales, label="directional_scales")
    chemistry_scales = _normalized_scale_grid(chemistry_scales, label="chemistry_scales")
    local_field_energy_scales = _normalized_scale_grid(local_field_energy_scales, label="local_field_energy_scales")
    local_field_repulsion_scales = _normalized_scale_grid(
        local_field_repulsion_scales,
        label="local_field_repulsion_scales",
    )
    alignment_floors = tuple(coerce_scalar(value, "alignment_floor") for value in alignment_floors)
    if not alignment_floors:
        raise ContractValidationError("alignment_floors must contain at least one value.")

    candidates: list[SpatialTransferCandidate] = []
    if include_baseline:
        candidates.append(
            SpatialTransferCandidate(
                name="baseline",
                preset=replace(preset, name="baseline"),
                metadata={
                    "is_baseline": True,
                    "preferred_distance_scale": 1.0,
                    "distance_tolerance_scale": 1.0,
                    "attraction_scale": 1.0,
                    "repulsion_scale": 1.0,
                    "directional_scale": 1.0,
                    "chemistry_scale": 1.0,
                    "local_field_energy_scale": 1.0,
                    "local_field_repulsion_scale": 1.0,
                    "alignment_floor": preset.spatial_alignment_floor,
                },
            )
        )
    for preferred_distance_scale in preferred_distance_scales:
        for distance_tolerance_scale in distance_tolerance_scales:
            for attraction_scale in attraction_scales:
                for repulsion_scale in repulsion_scales:
                    for directional_scale in directional_scales:
                        for chemistry_scale in chemistry_scales:
                            for local_field_energy_scale in local_field_energy_scales:
                                for local_field_repulsion_scale in local_field_repulsion_scales:
                                    for alignment_floor in alignment_floors:
                                        is_baseline_equivalent = (
                                            preferred_distance_scale == 1.0
                                            and distance_tolerance_scale == 1.0
                                            and attraction_scale == 1.0
                                            and repulsion_scale == 1.0
                                            and directional_scale == 1.0
                                            and chemistry_scale == 1.0
                                            and local_field_energy_scale == 1.0
                                            and local_field_repulsion_scale == 1.0
                                            and alignment_floor == preset.spatial_alignment_floor
                                        )
                                        if include_baseline and is_baseline_equivalent:
                                            continue
                                        candidate_preset = replace(
                                            preset,
                                            name=(
                                                "protein_transfer"
                                                f"_pd{preferred_distance_scale:.2f}"
                                                f"_dt{distance_tolerance_scale:.2f}"
                                                f"_a{attraction_scale:.2f}"
                                                f"_r{repulsion_scale:.2f}"
                                                f"_d{directional_scale:.2f}"
                                                f"_c{chemistry_scale:.2f}"
                                                f"_le{local_field_energy_scale:.2f}"
                                                f"_lr{local_field_repulsion_scale:.2f}"
                                                f"_f{alignment_floor:.2f}"
                                            ),
                                            spatial_profile_preferred_distance_scale=preferred_distance_scale,
                                            spatial_profile_distance_tolerance_scale=distance_tolerance_scale,
                                            spatial_profile_attraction_scale=attraction_scale,
                                            spatial_profile_repulsion_scale=repulsion_scale,
                                            spatial_profile_directional_scale=directional_scale,
                                            spatial_profile_chemistry_scale=chemistry_scale,
                                            spatial_energy_scale=preset.spatial_energy_scale * local_field_energy_scale,
                                            spatial_repulsion_scale=(
                                                preset.spatial_repulsion_scale * local_field_repulsion_scale
                                            ),
                                            spatial_alignment_floor=alignment_floor,
                                        )
                                        candidates.append(
                                            SpatialTransferCandidate(
                                                name=candidate_preset.name,
                                                preset=candidate_preset,
                                                metadata={
                                                    "preferred_distance_scale": preferred_distance_scale,
                                                    "distance_tolerance_scale": distance_tolerance_scale,
                                                    "attraction_scale": attraction_scale,
                                                    "repulsion_scale": repulsion_scale,
                                                    "directional_scale": directional_scale,
                                                    "chemistry_scale": chemistry_scale,
                                                    "local_field_energy_scale": local_field_energy_scale,
                                                    "local_field_repulsion_scale": local_field_repulsion_scale,
                                                    "alignment_floor": alignment_floor,
                                                    "absolute_local_field_energy_scale": candidate_preset.spatial_energy_scale,
                                                    "absolute_local_field_repulsion_scale": candidate_preset.spatial_repulsion_scale,
                                                },
                                            )
                                        )
    return tuple(candidates)


__all__ = [
    "ProteinTransferCandidateResult",
    "ProteinTransferTuningReport",
    "ProteinTransferTuningRunner",
    "ScenarioTransferScore",
    "SpatialTransferCandidate",
    "build_spatial_transfer_candidate_grid",
]
