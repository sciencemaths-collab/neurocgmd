"""Validate the Section 1 repository scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.runtime import build_repository_layout
from core.project_manifest import build_default_manifest


@dataclass(frozen=True, slots=True)
class ScaffoldValidationResult:
    """Structured result for repository scaffold validation."""

    missing_directories: tuple[str, ...]
    missing_progress_files: tuple[str, ...]
    manifest_gaps: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return True when the scaffold is structurally complete."""

        return not (
            self.missing_directories or self.missing_progress_files or self.manifest_gaps
        )


def validate_repository_scaffold(repo_root: Path | None = None) -> ScaffoldValidationResult:
    """Validate required directories, progress files, and manifest consistency."""

    layout = build_repository_layout(repo_root)
    manifest = build_default_manifest()

    missing_directories = tuple(
        str(path.relative_to(layout.repo_root))
        for path in layout.missing_directories()
    )
    missing_progress_files = tuple(
        str(path.relative_to(layout.repo_root))
        for path in layout.missing_progress_files()
    )

    manifest_gaps: list[str] = []
    expected_numbers = tuple(range(1, len(manifest.sections) + 1))
    if manifest.section_numbers() != expected_numbers:
        manifest_gaps.append("Section numbering is not contiguous from 1..N.")

    invalid_classifications = manifest.invalid_classification_sections()
    if invalid_classifications:
        manifest_gaps.extend(
            f"{identifier} uses an unsupported classification label."
            for identifier in invalid_classifications
        )

    return ScaffoldValidationResult(
        missing_directories=missing_directories,
        missing_progress_files=missing_progress_files,
        manifest_gaps=tuple(manifest_gaps),
    )


def main() -> int:
    """Run validation and print a concise status report."""

    result = validate_repository_scaffold()
    if result.ok:
        print("Scaffold validation passed.")
        return 0

    print("Scaffold validation failed.")
    if result.missing_directories:
        print("Missing directories:")
        for entry in result.missing_directories:
            print(f"  - {entry}")
    if result.missing_progress_files:
        print("Missing progress files:")
        for entry in result.missing_progress_files:
            print(f"  - {entry}")
    if result.manifest_gaps:
        print("Manifest issues:")
        for entry in result.manifest_gaps:
            print(f"  - {entry}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
