"""Path-loaded checkpoint writer for manifest-driven runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.exceptions import ContractValidationError
from core.interfaces import ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata


@dataclass(frozen=True, slots=True)
class CheckpointArtifact(ValidatableComponent):
    """One serialized checkpoint payload."""

    state: SimulationState
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        return self.state.validate()

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.to_dict(),
            "metadata": self.metadata.to_dict(),
        }


@dataclass(slots=True)
class CheckpointWriter:
    """Whole-file JSON checkpoint writer."""

    path: str | Path

    def write(
        self,
        state: SimulationState,
        *,
        metadata: FrozenMetadata | dict[str, object] | None = None,
    ) -> Path:
        destination = Path(self.path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        artifact = CheckpointArtifact(
            state=state,
            metadata=metadata if isinstance(metadata, FrozenMetadata) else FrozenMetadata(metadata),
        )
        destination.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return destination
