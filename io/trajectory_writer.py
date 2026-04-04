"""Path-loaded trajectory writer for manifest-driven runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.exceptions import ContractValidationError
from core.interfaces import ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata


@dataclass(frozen=True, slots=True)
class TrajectoryFrame(ValidatableComponent):
    """One serialized trajectory frame."""

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
class TrajectoryWriter:
    """Append-only JSONL trajectory writer."""

    path: str | Path
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)
    _initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            self.metadata = FrozenMetadata(self.metadata)

    def append_state(
        self,
        state: SimulationState,
        *,
        metadata: FrozenMetadata | dict[str, object] | None = None,
    ) -> None:
        destination = Path(self.path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame = TrajectoryFrame(
            state=state,
            metadata=self.metadata.with_updates(metadata or {}),
        )
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(frame.to_dict(), sort_keys=True))
            handle.write("\n")
        self._initialized = True

