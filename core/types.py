"""Core identifiers, metadata containers, and backend-agnostic shape aliases."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import NewType, TypeAlias

from core.exceptions import ContractValidationError

SimulationId = NewType("SimulationId", str)
StateId = NewType("StateId", str)
BeadId = NewType("BeadId", str)
CompartmentId = NewType("CompartmentId", str)
RegionId = NewType("RegionId", str)
ModelId = NewType("ModelId", str)

Scalar: TypeAlias = float
Vector3: TypeAlias = tuple[Scalar, Scalar, Scalar]
Matrix3x3: TypeAlias = tuple[Vector3, Vector3, Vector3]
ScalarTuple: TypeAlias = tuple[Scalar, ...]
VectorTuple: TypeAlias = tuple[Vector3, ...]
BoolVector3: TypeAlias = tuple[bool, bool, bool]

SPACE_DIMENSIONS = 3


def coerce_scalar(value: int | float, name: str = "value") -> float:
    """Return a numeric value as float while rejecting invalid scalar inputs."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{name} must be numeric; received {value!r}.")
    return float(value)


def coerce_scalar_tuple(values: Iterable[int | float], name: str) -> ScalarTuple:
    """Return an immutable tuple of floats."""

    return tuple(coerce_scalar(value, f"{name}[{index}]") for index, value in enumerate(values))


def coerce_vector3(values: Sequence[int | float], name: str = "vector") -> Vector3:
    """Return an immutable 3D vector with float entries."""

    data = tuple(coerce_scalar(value, f"{name}[{index}]") for index, value in enumerate(values))
    if len(data) != SPACE_DIMENSIONS:
        raise ContractValidationError(
            f"{name} must contain exactly {SPACE_DIMENSIONS} elements; received {len(data)}."
        )
    return data


def coerce_vector_block(rows: Iterable[Sequence[int | float]], name: str) -> VectorTuple:
    """Return an immutable block of 3D vectors."""

    return tuple(coerce_vector3(row, f"{name}[{index}]") for index, row in enumerate(rows))


def coerce_bool_vector3(values: Sequence[bool], name: str) -> BoolVector3:
    """Return an immutable 3D boolean tuple."""

    data = tuple(values)
    if len(data) != SPACE_DIMENSIONS or any(not isinstance(entry, bool) for entry in data):
        raise ContractValidationError(
            f"{name} must contain exactly {SPACE_DIMENSIONS} boolean values."
        )
    return data


def freeze_metadata_value(value: object, path: str = "metadata") -> object:
    """Recursively convert a metadata value into an immutable representation."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, FrozenMetadata):
        return value
    if isinstance(value, Mapping):
        return FrozenMetadata(value)
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_metadata_value(entry, f"{path}[{index}]")
            for index, entry in enumerate(value)
        )
    raise ContractValidationError(
        f"{path} contains unsupported metadata value {value!r} of type {type(value).__name__}."
    )


def thaw_metadata_value(value: object) -> object:
    """Convert an immutable metadata value back into JSON-friendly containers."""

    if isinstance(value, FrozenMetadata):
        return value.to_dict()
    if isinstance(value, tuple):
        return [thaw_metadata_value(entry) for entry in value]
    return value


@dataclass(frozen=True, slots=True, init=False)
class FrozenMetadata(Mapping[str, object]):
    """Deterministic immutable mapping for state metadata and observables."""

    _items: tuple[tuple[str, object], ...]

    def __init__(self, mapping: Mapping[str, object] | None = None) -> None:
        normalized_items: list[tuple[str, object]] = []
        for key, value in (mapping or {}).items():
            if not isinstance(key, str) or not key.strip():
                raise ContractValidationError(
                    f"Metadata keys must be non-empty strings; received {key!r}."
                )
            normalized_items.append((key, freeze_metadata_value(value, key)))
        normalized_items.sort(key=lambda item: item[0])
        object.__setattr__(self, "_items", tuple(normalized_items))

    def __getitem__(self, key: str) -> object:
        for stored_key, value in self._items:
            if stored_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly dictionary view of the metadata."""

        return {key: thaw_metadata_value(value) for key, value in self._items}

    def with_updates(
        self, updates: Mapping[str, object] | None = None, /, **kwargs: object
    ) -> "FrozenMetadata":
        """Return a new metadata object with updated entries."""

        merged = self.to_dict()
        if updates:
            merged.update(dict(updates))
        if kwargs:
            merged.update(kwargs)
        return FrozenMetadata(merged)

