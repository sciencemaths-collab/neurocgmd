"""Project-wide constants used by the Section 1 scaffold."""

from typing import Final

PROJECT_NAME: Final[str] = "NeuroCGMD"
WORKING_NAMESPACE: Final[str] = "neurocgmd"
PROJECT_TAGLINE: Final[str] = (
    "Adaptive coarse-grained molecular dynamics with graph plasticity, "
    "quantum-cloud refinement, memory, and AI control."
)

CLASSIFICATION_LABELS: Final[tuple[str, ...]] = (
    "[established]",
    "[adapted]",
    "[hybrid]",
    "[proposed novel]",
)

BUILD_STATUS_LABELS: Final[tuple[str, ...]] = (
    "NOT STARTED",
    "IN PROGRESS",
    "COMPLETE",
    "BLOCKED",
    "EXPERIMENTAL",
)

TOP_LEVEL_DIRECTORIES: Final[tuple[str, ...]] = (
    "progress_info",
    "docs",
    "config",
    "core",
    "physics",
    "topology",
    "forcefields",
    "graph",
    "plasticity",
    "compartments",
    "qcloud",
    "ml",
    "ai_control",
    "integrators",
    "sampling",
    "memory",
    "optimization",
    "io",
    "visualization",
    "validation",
    "benchmarks",
    "tests",
    "scripts",
)

MANDATORY_PROGRESS_FILES: Final[tuple[str, ...]] = (
    "project_map.md",
    "build_status.md",
    "section_registry.md",
    "module_dependency_map.md",
    "api_contracts.md",
    "decision_log.md",
    "next_steps.md",
    "change_log.md",
    "continuation_brief.md",
)

