"""Helpers for discovering and validating the repository layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.constants import MANDATORY_PROGRESS_FILES, TOP_LEVEL_DIRECTORIES
from core.exceptions import ProjectScaffoldError


@dataclass(frozen=True, slots=True)
class RepositoryLayout:
    """Concrete paths for the scaffold's required directories and progress files."""

    repo_root: Path
    top_level_directories: tuple[Path, ...]
    progress_files: tuple[Path, ...]

    def missing_directories(self) -> tuple[Path, ...]:
        """Return top-level directories that do not currently exist."""

        return tuple(path for path in self.top_level_directories if not path.is_dir())

    def missing_progress_files(self) -> tuple[Path, ...]:
        """Return progress tracking files that do not currently exist."""

        return tuple(path for path in self.progress_files if not path.is_file())

    def missing_paths(self) -> tuple[Path, ...]:
        """Return all missing scaffold paths."""

        return self.missing_directories() + self.missing_progress_files()


def infer_repo_root(start: Path | None = None) -> Path:
    """Walk upward until the Section 1 scaffold root is found."""

    candidate = (start or Path(__file__).resolve()).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for current in (candidate, *candidate.parents):
        if (current / "progress_info").is_dir() and (current / "core").is_dir():
            return current

    raise ProjectScaffoldError(
        "Could not infer the repository root. Expected both 'progress_info/' and 'core/' "
        "to exist in the project root or one of its parents."
    )


def build_repository_layout(repo_root: Path | None = None) -> RepositoryLayout:
    """Build the canonical path layout for scaffold validation."""

    root = infer_repo_root(repo_root)
    directories = tuple(root / directory for directory in TOP_LEVEL_DIRECTORIES)
    progress_files = tuple(
        root / "progress_info" / filename for filename in MANDATORY_PROGRESS_FILES
    )
    return RepositoryLayout(
        repo_root=root,
        top_level_directories=directories,
        progress_files=progress_files,
    )

