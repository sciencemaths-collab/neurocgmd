"""Tests for the canonical project manifest."""

from __future__ import annotations

import unittest

from core.constants import CLASSIFICATION_LABELS, TOP_LEVEL_DIRECTORIES
from core.project_manifest import build_default_manifest


class ProjectManifestTests(unittest.TestCase):
    """Verify the manifest remains aligned with the scaffold contract."""

    def test_section_numbers_are_contiguous(self) -> None:
        manifest = build_default_manifest()
        expected = tuple(range(1, len(manifest.sections) + 1))
        self.assertEqual(manifest.section_numbers(), expected)

    def test_section_classifications_are_supported(self) -> None:
        manifest = build_default_manifest()
        for section in manifest.sections:
            self.assertIn(section.classification, CLASSIFICATION_LABELS)

    def test_manifest_folders_match_known_top_level_directories(self) -> None:
        manifest = build_default_manifest()
        known_directories = set(TOP_LEVEL_DIRECTORIES)
        for section in manifest.sections:
            for folder in section.primary_folders:
                self.assertIn(folder, known_directories)


if __name__ == "__main__":
    unittest.main()

