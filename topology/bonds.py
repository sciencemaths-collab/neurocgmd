"""Static bonded connectivity primitives for the system topology."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from core.exceptions import ContractValidationError
from core.interfaces import ValidatableComponent
from core.types import FrozenMetadata


class BondKind(StrEnum):
    """High-level bond category for topology and later force-field dispatch."""

    STRUCTURAL = "structural"
    ELASTIC = "elastic"
    CROSSLINK = "crosslink"
    CONSTRAINT = "constraint"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class Bond(ValidatableComponent):
    """Static undirected bond between two particle indices."""

    particle_index_a: int
    particle_index_b: int
    kind: BondKind = BondKind.STRUCTURAL
    bond_id: str | None = None
    equilibrium_distance: float | None = None
    stiffness: float | None = None
    order: int = 1
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.particle_index_a < 0 or self.particle_index_b < 0:
            issues.append("Bond particle indices must be non-negative.")
        if self.particle_index_a == self.particle_index_b:
            issues.append("Bond endpoints must reference two different particle indices.")
        if self.order <= 0:
            issues.append("Bond order must be a positive integer.")
        if self.bond_id is not None and not self.bond_id.strip():
            issues.append("bond_id must be non-empty when provided.")
        if self.equilibrium_distance is not None and self.equilibrium_distance <= 0.0:
            issues.append("equilibrium_distance must be positive when provided.")
        if self.stiffness is not None and self.stiffness <= 0.0:
            issues.append("stiffness must be positive when provided.")
        return tuple(issues)

    def normalized_pair(self) -> tuple[int, int]:
        """Return the undirected endpoint pair in sorted order."""

        return tuple(sorted((self.particle_index_a, self.particle_index_b)))

    def to_dict(self) -> dict[str, object]:
        return {
            "particle_index_a": self.particle_index_a,
            "particle_index_b": self.particle_index_b,
            "kind": self.kind.value,
            "bond_id": self.bond_id,
            "equilibrium_distance": self.equilibrium_distance,
            "stiffness": self.stiffness,
            "order": self.order,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Bond":
        return cls(
            particle_index_a=int(data["particle_index_a"]),
            particle_index_b=int(data["particle_index_b"]),
            kind=BondKind(data.get("kind", BondKind.STRUCTURAL.value)),
            bond_id=str(data["bond_id"]) if data.get("bond_id") else None,
            equilibrium_distance=data.get("equilibrium_distance"),
            stiffness=data.get("stiffness"),
            order=int(data.get("order", 1)),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


def build_neighbor_map(particle_count: int, bonds: Iterable[Bond]) -> dict[int, tuple[int, ...]]:
    """Build a deterministic undirected neighbor map for the bonded topology."""

    if particle_count < 0:
        raise ContractValidationError("particle_count must be non-negative.")

    adjacency: dict[int, set[int]] = {index: set() for index in range(particle_count)}
    for bond in bonds:
        a, b = bond.normalized_pair()
        if a >= particle_count or b >= particle_count:
            raise ContractValidationError(
                f"Bond {bond.normalized_pair()} exceeds particle_count={particle_count}."
            )
        adjacency[a].add(b)
        adjacency[b].add(a)
    return {index: tuple(sorted(neighbors)) for index, neighbors in adjacency.items()}


def connected_components(
    particle_count: int, bonds: Iterable[Bond]
) -> tuple[tuple[int, ...], ...]:
    """Return connected components induced by the bonded topology."""

    adjacency = build_neighbor_map(particle_count, bonds)
    remaining = set(range(particle_count))
    components: list[tuple[int, ...]] = []

    while remaining:
        root = min(remaining)
        queue: deque[int] = deque([root])
        component: list[int] = []
        remaining.remove(root)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append(tuple(sorted(component)))

    components.sort(key=lambda component: component[0] if component else -1)
    return tuple(components)

