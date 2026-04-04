"""Export-safe dashboard serialization utilities loaded by path, not package import."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.exceptions import ContractValidationError
from core.interfaces import ValidatableComponent
from core.types import FrozenMetadata
from visualization.trajectory_views import DashboardSnapshotView


@dataclass(frozen=True, slots=True)
class ExportArtifact(ValidatableComponent):
    """One concrete exported artifact on disk."""

    label: str
    relative_path: str
    media_type: str
    byte_size: int
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.label.strip():
            issues.append("label must be a non-empty string.")
        if not self.relative_path.strip():
            issues.append("relative_path must be a non-empty string.")
        if not self.media_type.strip():
            issues.append("media_type must be a non-empty string.")
        if self.byte_size < 0:
            issues.append("byte_size must be non-negative.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "relative_path": self.relative_path,
            "media_type": self.media_type,
            "byte_size": self.byte_size,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DashboardExportBundle(ValidatableComponent):
    """Structured result for one dashboard export operation."""

    output_dir: str
    html_artifact: ExportArtifact
    json_artifact: ExportArtifact
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.output_dir.strip():
            issues.append("output_dir must be a non-empty string.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": self.output_dir,
            "html_artifact": self.html_artifact.to_dict(),
            "json_artifact": self.json_artifact.to_dict(),
            "metadata": self.metadata.to_dict(),
        }


def export_dashboard_snapshot(
    snapshot: DashboardSnapshotView,
    output_dir: str | Path,
    *,
    html_filename: str = "index.html",
    json_filename: str = "dashboard.json",
    refresh_ms: int = 1000,
) -> DashboardExportBundle:
    """Write a dashboard snapshot to HTML and JSON artifacts."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    json_path = destination / json_filename
    html_path = destination / html_filename
    payload = snapshot.to_dict()
    json_text = json.dumps(payload, indent=2, sort_keys=True)
    html_text = snapshot.render_html(json_endpoint=json_filename, refresh_ms=refresh_ms)

    json_path.write_text(json_text, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")

    return DashboardExportBundle(
        output_dir=str(destination),
        html_artifact=ExportArtifact(
            label="dashboard_html",
            relative_path=html_filename,
            media_type="text/html",
            byte_size=len(html_text.encode("utf-8")),
            metadata=FrozenMetadata({"title": snapshot.title}),
        ),
        json_artifact=ExportArtifact(
            label="dashboard_json",
            relative_path=json_filename,
            media_type="application/json",
            byte_size=len(json_text.encode("utf-8")),
            metadata=FrozenMetadata({"state_id": str(snapshot.trajectory.state_id)}),
        ),
        metadata=FrozenMetadata({"refresh_ms": refresh_ms}),
    )


def load_dashboard_payload(path: str | Path) -> dict[str, object]:
    """Load a previously exported dashboard JSON payload."""

    resolved_path = Path(path).expanduser().resolve()
    return json.loads(resolved_path.read_text(encoding="utf-8"))
