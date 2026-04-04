"""Canonical section order and architectural registry for the project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from core.constants import (
    CLASSIFICATION_LABELS,
    MANDATORY_PROGRESS_FILES,
    TOP_LEVEL_DIRECTORIES,
)


@dataclass(frozen=True, slots=True)
class SectionDefinition:
    """Metadata for one planned build section."""

    number: int
    name: str
    objective: str
    classification: str
    primary_folders: tuple[str, ...]
    planned_scripts: tuple[str, ...]
    validation_focus: str

    def identifier(self) -> str:
        """Return a stable human-readable identifier for the section."""

        return f"Section {self.number}. {self.name}"


DEFAULT_SECTIONS: Final[tuple[SectionDefinition, ...]] = (
    SectionDefinition(
        number=1,
        name="Project scaffold and core architecture",
        objective="Create the repository skeleton, continuity files, and baseline contracts.",
        classification="[adapted]",
        primary_folders=("progress_info", "docs", "config", "core", "tests", "scripts"),
        planned_scripts=(
            "core/constants.py",
            "core/interfaces.py",
            "core/project_manifest.py",
            "scripts/validate_scaffold.py",
        ),
        validation_focus="Structural completeness, manifest integrity, and continuity readiness.",
    ),
    SectionDefinition(
        number=2,
        name="Core data models and simulation state",
        objective="Define units-aware state containers, provenance, and immutable identifiers.",
        classification="[adapted]",
        primary_folders=("core", "config", "tests", "docs", "progress_info"),
        planned_scripts=("core/types.py", "core/state.py", "core/state_registry.py"),
        validation_focus="State shape correctness, typing, serialization boundaries, and invariants.",
    ),
    SectionDefinition(
        number=3,
        name="Topology and bead system",
        objective="Represent beads, bonded structure, neighborhood metadata, and system composition.",
        classification="[adapted]",
        primary_folders=("topology", "core", "tests", "docs", "progress_info"),
        planned_scripts=(
            "topology/beads.py",
            "topology/bonds.py",
            "topology/system_topology.py",
        ),
        validation_focus="Topological consistency, index safety, and graph connectivity invariants.",
    ),
    SectionDefinition(
        number=4,
        name="Force field foundation",
        objective="Define baseline coarse-grained energy terms and force interfaces.",
        classification="[established]",
        primary_folders=("physics", "forcefields", "tests", "docs", "progress_info"),
        planned_scripts=(
            "physics/energies/bonded.py",
            "physics/energies/nonbonded.py",
            "physics/forces/nonbonded_forces.py",
            "forcefields/base_forcefield.py",
        ),
        validation_focus="Energy-force consistency, parameter validation, and baseline benchmarks.",
    ),
    SectionDefinition(
        number=5,
        name="Integrators and simulation loop",
        objective="Build time integration, ensemble hooks, and the primary execution loop.",
        classification="[established]",
        primary_folders=("integrators", "sampling", "core", "tests", "docs", "progress_info"),
        planned_scripts=(
            "integrators/base.py",
            "integrators/langevin.py",
            "sampling/simulation_loop.py",
        ),
        validation_focus="Stability, reproducibility, drift monitoring, and step orchestration.",
    ),
    SectionDefinition(
        number=6,
        name="Dynamic graph connectivity layer",
        objective="Introduce adaptive interaction graphs on top of the physical substrate.",
        classification="[proposed novel]",
        primary_folders=("graph", "topology", "tests", "docs", "progress_info"),
        planned_scripts=(
            "graph/edge_models.py",
            "graph/graph_manager.py",
            "graph/connectivity_rules.py",
            "graph/adjacency_utils.py",
        ),
        validation_focus="Connectivity updates, local/long-range balance, and adjacency correctness.",
    ),
    SectionDefinition(
        number=7,
        name="Plasticity and rewiring rules",
        objective="Implement reinforcement, pruning, growth, and memory-trace adaptation.",
        classification="[proposed novel]",
        primary_folders=("plasticity", "graph", "memory", "tests", "docs", "progress_info"),
        planned_scripts=(
            "plasticity/hebbian.py",
            "plasticity/pruning.py",
            "plasticity/reinforcement.py",
            "plasticity/traces.py",
            "plasticity/engine.py",
        ),
        validation_focus="Rule stability, rewiring boundedness, and failure mode identification.",
    ),
    SectionDefinition(
        number=8,
        name="Compartment system",
        objective="Represent modular regions with specialized update and routing behavior.",
        classification="[hybrid]",
        primary_folders=("compartments", "graph", "tests", "docs", "progress_info"),
        planned_scripts=(
            "compartments/registry.py",
            "compartments/domain_models.py",
            "compartments/routing.py",
        ),
        validation_focus="Compartment membership, routing correctness, and state synchronization.",
    ),
    SectionDefinition(
        number=9,
        name="Memory and replay system",
        objective="Track trajectory history, instability episodes, and reusable experiences.",
        classification="[hybrid]",
        primary_folders=("memory", "ml", "tests", "docs", "progress_info"),
        planned_scripts=(
            "memory/trace_store.py",
            "memory/replay_buffer.py",
            "memory/episode_registry.py",
        ),
        validation_focus="Replay fidelity, storage bounds, and trace retrieval correctness.",
    ),
    SectionDefinition(
        number=10,
        name="Quantum-cloud framework",
        objective="Select local refinement regions and couple quantum-informed corrections.",
        classification="[hybrid]",
        primary_folders=("qcloud", "physics", "tests", "docs", "progress_info"),
        planned_scripts=(
            "qcloud/cloud_state.py",
            "qcloud/region_selector.py",
            "qcloud/qcloud_coupling.py",
        ),
        validation_focus="Region-selection correctness, coupling stability, and uncertainty triggers.",
    ),
    SectionDefinition(
        number=11,
        name="Online ML residual learning",
        objective="Add residual models, uncertainty estimation, and online update hooks.",
        classification="[hybrid]",
        primary_folders=("ml", "memory", "tests", "docs", "progress_info"),
        planned_scripts=(
            "ml/residual_model.py",
            "ml/uncertainty_model.py",
            "ml/online_trainer.py",
        ),
        validation_focus="Residual accuracy, uncertainty calibration, and training loop safety.",
    ),
    SectionDefinition(
        number=12,
        name="AI executive control layer",
        objective="Allocate compute, detect instability, and coordinate adaptive actions.",
        classification="[proposed novel]",
        primary_folders=("ai_control", "ml", "qcloud", "tests", "docs", "progress_info"),
        planned_scripts=(
            "ai_control/controller.py",
            "ai_control/policies.py",
            "ai_control/resource_allocator.py",
            "ai_control/stability_monitor.py",
        ),
        validation_focus="Policy correctness, control-loop latency, and fallback behavior.",
    ),
    SectionDefinition(
        number=13,
        name="Validation and benchmarking suite",
        objective="Create physical sanity checks, ablations, and baseline comparison pipelines.",
        classification="[established]",
        primary_folders=("validation", "benchmarks", "tests", "docs", "progress_info"),
        planned_scripts=(
            "validation/sanity_checks.py",
            "validation/drift_checks.py",
            "benchmarks/baseline_suite.py",
        ),
        validation_focus="Coverage breadth, reproducible metrics, and benchmark comparability.",
    ),
    SectionDefinition(
        number=14,
        name="Visualization and diagnostics",
        objective="Expose interpretable diagnostics for state, graphs, plasticity, and control.",
        classification="[adapted]",
        primary_folders=("visualization", "io", "scripts", "docs", "tests", "progress_info"),
        planned_scripts=(
            "visualization/trajectory_views.py",
            "visualization/graph_views.py",
            "io/export_registry.py",
            "scripts/live_dashboard.py",
        ),
        validation_focus="Diagnostic correctness, export completeness, and human interpretability.",
    ),
    SectionDefinition(
        number=15,
        name="Performance optimization and scaling hooks",
        objective="Prepare acceleration, profiling, and distributed execution interfaces.",
        classification="[adapted]",
        primary_folders=("optimization", "integrators", "ml", "docs", "tests", "progress_info"),
        planned_scripts=(
            "optimization/profiling.py",
            "optimization/backend_registry.py",
            "optimization/scaling_hooks.py",
        ),
        validation_focus="Performance regressions, backend parity, and scalability assumptions.",
    ),
)


@dataclass(frozen=True, slots=True)
class ProjectManifest:
    """Canonical repository manifest used for continuity and validation."""

    top_level_directories: tuple[str, ...] = TOP_LEVEL_DIRECTORIES
    mandatory_progress_files: tuple[str, ...] = MANDATORY_PROGRESS_FILES
    sections: tuple[SectionDefinition, ...] = DEFAULT_SECTIONS

    def section_numbers(self) -> tuple[int, ...]:
        """Return the section numbers in declared order."""

        return tuple(section.number for section in self.sections)

    def section_by_number(self, number: int) -> SectionDefinition:
        """Return a section definition by number."""

        for section in self.sections:
            if section.number == number:
                return section
        raise KeyError(f"Unknown section number: {number}")

    def invalid_classification_sections(self) -> tuple[str, ...]:
        """Return section identifiers that use unsupported classification labels."""

        return tuple(
            section.identifier()
            for section in self.sections
            if section.classification not in CLASSIFICATION_LABELS
        )


def build_default_manifest() -> ProjectManifest:
    """Return the canonical manifest used across docs, tests, and scripts."""

    return ProjectManifest()
