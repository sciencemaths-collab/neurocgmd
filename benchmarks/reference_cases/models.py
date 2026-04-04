"""Reference-case models for experimentally grounded benchmark targets."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, coerce_scalar


@dataclass(frozen=True, slots=True)
class ReferenceSource(ValidatableComponent):
    """One primary source supporting a benchmark target."""

    label: str
    url: str
    source_type: str
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
        if not self.url.startswith("http"):
            issues.append("url must be an http(s) link.")
        if not self.source_type.strip():
            issues.append("source_type must be a non-empty string.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "url": self.url,
            "source_type": self.source_type,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ReferenceObservable(ValidatableComponent):
    """One experimentally anchored target observable."""

    name: str
    expected_value: float
    units: str
    description: str
    source_label: str
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected_value", coerce_scalar(self.expected_value, "expected_value"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name.strip():
            issues.append("name must be a non-empty string.")
        if not self.units.strip():
            issues.append("units must be a non-empty string.")
        if not self.description.strip():
            issues.append("description must be a non-empty string.")
        if not self.source_label.strip():
            issues.append("source_label must be a non-empty string.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "expected_value": self.expected_value,
            "units": self.units,
            "description": self.description,
            "source_label": self.source_label,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class StructuralReference(ValidatableComponent):
    """Structural truth bundle for one benchmark target."""

    bound_complex_pdb_id: str
    unbound_partner_pdb_ids: tuple[str, ...]
    description: str
    source_label: str
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "unbound_partner_pdb_ids", tuple(self.unbound_partner_pdb_ids))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.bound_complex_pdb_id.strip():
            issues.append("bound_complex_pdb_id must be a non-empty string.")
        if len(self.unbound_partner_pdb_ids) < 2:
            issues.append("unbound_partner_pdb_ids must contain at least two structures.")
        if not self.description.strip():
            issues.append("description must be a non-empty string.")
        if not self.source_label.strip():
            issues.append("source_label must be a non-empty string.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "bound_complex_pdb_id": self.bound_complex_pdb_id,
            "unbound_partner_pdb_ids": list(self.unbound_partner_pdb_ids),
            "description": self.description,
            "source_label": self.source_label,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ExperimentalReferenceCase(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Experimentally grounded benchmark case with known answer(s)."""

    name: str
    classification: str
    title: str
    problem_type: str
    summary: str
    structural_reference: StructuralReference
    observables: tuple[ReferenceObservable, ...]
    sources: tuple[ReferenceSource, ...]
    recommended_comparisons: tuple[str, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "observables", tuple(self.observables))
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "recommended_comparisons", tuple(self.recommended_comparisons))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Stores a real-world molecular benchmark target with known structural "
            "and/or kinetic answers so later simulations can be compared honestly."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "benchmarks/reference_cases/models.py",
            "docs/use_cases/barnase_barstar_reference_case.md",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/validation_and_benchmarking.md",
            "docs/use_cases/barnase_barstar_reference_case.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name.strip():
            issues.append("name must be a non-empty string.")
        if not self.classification.strip():
            issues.append("classification must be a non-empty string.")
        if not self.title.strip():
            issues.append("title must be a non-empty string.")
        if not self.problem_type.strip():
            issues.append("problem_type must be a non-empty string.")
        if not self.summary.strip():
            issues.append("summary must be a non-empty string.")
        if not self.observables:
            issues.append("observables must contain at least one reference observable.")
        if len(self.observable_names()) != len(set(self.observable_names())):
            issues.append("observable names must be unique.")
        if not self.sources:
            issues.append("sources must contain at least one primary source.")
        if not self.recommended_comparisons:
            issues.append("recommended_comparisons must be non-empty.")
        return tuple(issues)

    def observable_names(self) -> tuple[str, ...]:
        return tuple(observable.name for observable in self.observables)

    def observable_for(self, name: str) -> ReferenceObservable:
        for observable in self.observables:
            if observable.name == name:
                return observable
        raise KeyError(name)

    def source_for(self, label: str) -> ReferenceSource:
        for source in self.sources:
            if source.label == label:
                return source
        raise KeyError(label)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "classification": self.classification,
            "title": self.title,
            "problem_type": self.problem_type,
            "summary": self.summary,
            "structural_reference": self.structural_reference.to_dict(),
            "observables": [observable.to_dict() for observable in self.observables],
            "sources": [source.to_dict() for source in self.sources],
            "recommended_comparisons": list(self.recommended_comparisons),
            "metadata": self.metadata.to_dict(),
        }
