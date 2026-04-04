"""Tests for the Section 1 repository scaffold."""

from __future__ import annotations

from pathlib import Path
import unittest

from config.runtime import build_repository_layout
from scripts.validate_scaffold import validate_repository_scaffold


class ScaffoldTests(unittest.TestCase):
    """Verify required scaffold files and directories remain present."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]

    def test_repository_layout_has_no_missing_paths(self) -> None:
        layout = build_repository_layout(self.repo_root)
        self.assertEqual(layout.missing_paths(), ())

    def test_validation_script_passes(self) -> None:
        result = validate_repository_scaffold(self.repo_root)
        self.assertTrue(
            result.ok,
            msg=(
                f"Missing directories: {result.missing_directories}; "
                f"missing progress files: {result.missing_progress_files}; "
                f"manifest gaps: {result.manifest_gaps}"
            ),
        )

    def test_io_directory_is_documented_without_shadowing_stdlib(self) -> None:
        self.assertTrue((self.repo_root / "io" / "README.md").is_file())
        self.assertFalse((self.repo_root / "io" / "__init__.py").exists())


if __name__ == "__main__":
    unittest.main()
