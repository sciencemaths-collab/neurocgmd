"""Protocol-level interfaces shared across future project sections."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class ArchitecturalComponent(Protocol):
    """Minimal contract for any major subsystem in the platform."""

    name: str
    classification: str

    def describe_role(self) -> str:
        """Return the subsystem's scientific and software role."""

    def declared_dependencies(self) -> Sequence[str]:
        """Return named upstream modules or subsystems."""


@runtime_checkable
class ValidatableComponent(Protocol):
    """Contract for subsystems that can perform internal validation."""

    def validate(self) -> Sequence[str]:
        """Return validation issues; an empty sequence means the check passed."""


@runtime_checkable
class DocumentedComponent(Protocol):
    """Contract for components that expose primary documentation locations."""

    def documentation_paths(self) -> Sequence[str]:
        """Return documentation paths that define the component's behavior."""

