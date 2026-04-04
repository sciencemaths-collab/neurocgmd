"""Tests for Section 2 core type helpers and immutable metadata."""

from __future__ import annotations

import unittest

from core.exceptions import ContractValidationError
from core.types import FrozenMetadata, coerce_vector3


class CoreTypesTests(unittest.TestCase):
    """Verify identifier-adjacent helpers behave deterministically."""

    def test_frozen_metadata_roundtrip(self) -> None:
        metadata = FrozenMetadata(
            {
                "stage": "initialization",
                "scores": [1, 2, 3],
                "nested": {"active": True, "label": "alpha"},
            }
        )

        self.assertEqual(
            metadata.to_dict(),
            {
                "nested": {"active": True, "label": "alpha"},
                "scores": [1, 2, 3],
                "stage": "initialization",
            },
        )
        self.assertEqual(metadata["stage"], "initialization")

    def test_frozen_metadata_rejects_unsupported_values(self) -> None:
        with self.assertRaises(ContractValidationError):
            FrozenMetadata({"bad": object()})

    def test_coerce_vector3_enforces_three_dimensions(self) -> None:
        self.assertEqual(coerce_vector3([1, 2, 3]), (1.0, 2.0, 3.0))
        with self.assertRaises(ContractValidationError):
            coerce_vector3([1, 2], "short_vector")


if __name__ == "__main__":
    unittest.main()

