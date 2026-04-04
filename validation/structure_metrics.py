"""Observer-side structural-comparison metrics for benchmarked live scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from benchmarks.reference_cases.structure_targets import ReferenceStructureTarget
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, StateId, Vector3, coerce_vector3


def _subtract(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(vector: Vector3, factor: float) -> Vector3:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _distance(a: Vector3, b: Vector3) -> float:
    delta = _subtract(a, b)
    return sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2])


def _centroid(points: tuple[Vector3, ...]) -> Vector3:
    count = len(points)
    if count == 0:
        raise ContractValidationError("At least one point is required to compute a centroid.")
    factor = 1.0 / count
    summed = (0.0, 0.0, 0.0)
    for point in points:
        summed = _add(summed, point)
    return _scale(summed, factor)


def _normalize_quaternion(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = sqrt(sum(value * value for value in quaternion))
    if norm == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(value / norm for value in quaternion)


def _largest_eigenvector_symmetric4(matrix: tuple[tuple[float, ...], ...]) -> tuple[float, float, float, float]:
    vector = (1.0, 0.0, 0.0, 0.0)
    for _ in range(64):
        next_vector = tuple(sum(matrix[row][column] * vector[column] for column in range(4)) for row in range(4))
        vector = _normalize_quaternion(next_vector)
    return vector


def _rotation_matrix_from_quaternion(quaternion: tuple[float, float, float, float]) -> tuple[Vector3, Vector3, Vector3]:
    w, x, y, z = _normalize_quaternion(quaternion)
    return (
        (
            w * w + x * x - y * y - z * z,
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ),
        (
            2.0 * (x * y + w * z),
            w * w - x * x + y * y - z * z,
            2.0 * (y * z - w * x),
        ),
        (
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            w * w - x * x - y * y + z * z,
        ),
    )


def _apply_rotation(matrix: tuple[Vector3, Vector3, Vector3], point: Vector3) -> Vector3:
    return (
        matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2],
        matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2],
        matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2],
    )


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def best_fit_rmsd(
    observed_points: tuple[Vector3, ...],
    target_points: tuple[Vector3, ...],
) -> float:
    """Return best-fit RMSD between two matched 3D landmark sets."""

    if len(observed_points) != len(target_points):
        raise ContractValidationError("observed_points and target_points must have the same length.")
    if len(observed_points) < 2:
        raise ContractValidationError("At least two matched points are required for structural comparison.")

    observed_centroid = _centroid(observed_points)
    target_centroid = _centroid(target_points)
    centered_observed = tuple(_subtract(point, observed_centroid) for point in observed_points)
    centered_target = tuple(_subtract(point, target_centroid) for point in target_points)

    sxx = sxy = sxz = 0.0
    syx = syy = syz = 0.0
    szx = szy = szz = 0.0
    for observed, target in zip(centered_observed, centered_target):
        ox, oy, oz = observed
        tx, ty, tz = target
        sxx += ox * tx
        sxy += ox * ty
        sxz += ox * tz
        syx += oy * tx
        syy += oy * ty
        syz += oy * tz
        szx += oz * tx
        szy += oz * ty
        szz += oz * tz

    horn_matrix = (
        (sxx + syy + szz, syz - szy, szx - sxz, sxy - syx),
        (syz - szy, sxx - syy - szz, sxy + syx, szx + sxz),
        (szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy),
        (sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz),
    )
    rotation = _rotation_matrix_from_quaternion(_largest_eigenvector_symmetric4(horn_matrix))
    squared_error = 0.0
    for observed, target in zip(centered_observed, centered_target):
        aligned = _apply_rotation(rotation, observed)
        delta = _subtract(aligned, target)
        squared_error += delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
    return sqrt(squared_error / len(observed_points))


def best_fit_similarity_alignment(
    observed_points: tuple[Vector3, ...],
    target_points: tuple[Vector3, ...],
) -> tuple[float, float, tuple[Vector3, ...]]:
    """Return best-fit similarity RMSD, scale, and transformed observed points."""

    if len(observed_points) != len(target_points):
        raise ContractValidationError("observed_points and target_points must have the same length.")
    if len(observed_points) < 2:
        raise ContractValidationError("At least two matched points are required for structural comparison.")

    observed_centroid = _centroid(observed_points)
    target_centroid = _centroid(target_points)
    centered_observed = tuple(_subtract(point, observed_centroid) for point in observed_points)
    centered_target = tuple(_subtract(point, target_centroid) for point in target_points)

    sxx = sxy = sxz = 0.0
    syx = syy = syz = 0.0
    szx = szy = szz = 0.0
    for observed, target in zip(centered_observed, centered_target):
        ox, oy, oz = observed
        tx, ty, tz = target
        sxx += ox * tx
        sxy += ox * ty
        sxz += ox * tz
        syx += oy * tx
        syy += oy * ty
        syz += oy * tz
        szx += oz * tx
        szy += oz * ty
        szz += oz * tz

    horn_matrix = (
        (sxx + syy + szz, syz - szy, szx - sxz, sxy - syx),
        (syz - szy, sxx - syy - szz, sxy + syx, szx + sxz),
        (szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy),
        (sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz),
    )
    rotation = _rotation_matrix_from_quaternion(_largest_eigenvector_symmetric4(horn_matrix))
    rotated_observed = tuple(_apply_rotation(rotation, point) for point in centered_observed)
    denominator = sum(_dot(point, point) for point in rotated_observed)
    if denominator == 0.0:
        scale = 1.0
    else:
        scale = sum(_dot(target, observed) for target, observed in zip(centered_target, rotated_observed)) / denominator

    transformed = tuple(
        _add(_scale(point, scale), target_centroid)
        for point in rotated_observed
    )
    squared_error = 0.0
    for transformed_point, target_point in zip(transformed, target_points):
        delta = _subtract(transformed_point, target_point)
        squared_error += _dot(delta, delta)
    return sqrt(squared_error / len(observed_points)), scale, transformed


@dataclass(frozen=True, slots=True)
class LandmarkObservation(ValidatableComponent):
    """One observed landmark location derived from the live state."""

    label: str
    position: Vector3
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", coerce_vector3(self.position, "position"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        if not self.label.strip():
            return ("label must be a non-empty string.",)
        return ()


@dataclass(frozen=True, slots=True)
class StructureMetric(ValidatableComponent):
    """One structural-comparison metric for dashboard consumption."""

    label: str
    value: str
    detail: str = ""
    status: str = "neutral"
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
        if not self.value.strip():
            issues.append("value must be a non-empty string.")
        if self.status not in {"neutral", "active", "good", "warn"}:
            issues.append("status must be one of: neutral, active, good, warn.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class StructureComparisonReport(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Observer-side report comparing proxy landmarks to a reference scaffold."""

    name: str = "structure_comparison_report"
    classification: str = "[adapted]"
    title: str = ""
    summary: str = ""
    metrics: tuple[StructureMetric, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Keeps proxy-vs-reference structural benchmarking explicit in the live "
            "dashboard without pretending the current simulation is fully atomistic."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "benchmarks/reference_cases/structure_targets.py",
            "docs/use_cases/spike_ace2_reference_case.md",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/use_cases/spike_ace2_reference_case.md",
            "docs/use_cases/spike_ace2_live_proxy.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.title.strip():
            issues.append("title must be a non-empty string.")
        if not self.summary.strip():
            issues.append("summary must be a non-empty string.")
        return tuple(issues)


def compare_landmark_observations(
    *,
    state_id: StateId | str,
    target: ReferenceStructureTarget,
    observations: tuple[LandmarkObservation, ...],
) -> StructureComparisonReport:
    """Compare one set of observed proxy landmarks against a local PDB-derived scaffold."""

    normalized_state_id = StateId(str(state_id))
    observation_map = {observation.label: observation for observation in observations}
    matched_targets = tuple(landmark for landmark in target.landmarks if landmark.label in observation_map)
    matched_observations = tuple(observation_map[landmark.label] for landmark in matched_targets)
    if len(matched_targets) < 2:
        raise ContractValidationError("At least two matched landmarks are required for structural comparison.")

    target_points = tuple(landmark.target_position for landmark in matched_targets)
    observed_points = tuple(observation.position for observation in matched_observations)
    rmsd, scale, transformed_points = best_fit_similarity_alignment(observed_points, target_points)
    transformed_map = {
        landmark.label: transformed_points[index]
        for index, landmark in enumerate(matched_targets)
    }

    contact_hits = 0
    for contact in target.interface_contacts:
        if contact.source_label not in transformed_map or contact.target_label not in transformed_map:
            continue
        if _distance(
            transformed_map[contact.source_label],
            transformed_map[contact.target_label],
        ) <= contact.max_distance:
            contact_hits += 1
    contact_total = len(target.interface_contacts)
    contact_fraction = contact_hits / contact_total if contact_total else 0.0

    dominant_pair = tuple(target.metadata.get("dominant_interface_pair", ()))
    dominant_error = None
    dominant_observed = None
    dominant_target = None
    if len(dominant_pair) == 2 and dominant_pair[0] in transformed_map and dominant_pair[1] in transformed_map:
        dominant_target = _distance(
            target.landmark_for(dominant_pair[0]).target_position,
            target.landmark_for(dominant_pair[1]).target_position,
        )
        dominant_observed = _distance(
            transformed_map[dominant_pair[0]],
            transformed_map[dominant_pair[1]],
        )
        dominant_error = abs(dominant_observed - dominant_target)

    metrics = [
        StructureMetric(
            label="Atomistic Centroid RMSD",
            value=f"{rmsd:.3f}",
            detail=(
                f"Similarity-fitted RMSD over {len(matched_targets)}/{len(target.landmarks)} residue-group centroids "
                f"derived from local {target.source_pdb_id} atom records."
            ),
            status="good" if rmsd <= 2.0 else "active" if rmsd <= 4.0 else "warn",
        ),
        StructureMetric(
            label="Matched Residue Groups",
            value=f"{len(matched_targets)}/{len(target.landmarks)}",
            detail="Number of residue-group centroids available for atomistic-reference comparison.",
            status="good" if len(matched_targets) == len(target.landmarks) else "warn",
        ),
        StructureMetric(
            label="Contact Recovery",
            value=f"{contact_hits}/{contact_total}",
            detail="Recovered hotspot-family contacts after similarity-fitting the coarse proxy into the local atomistic reference frame.",
            status="good" if contact_fraction >= 0.8 else "active" if contact_fraction >= 0.4 else "warn",
        ),
    ]
    metrics.append(
        StructureMetric(
            label="Fit Scale",
            value=f"{scale:.3f}",
            detail="Uniform scale factor mapping reduced coarse-grained units into the local atomistic reference frame.",
            status="active",
        )
    )
    if dominant_error is not None and dominant_observed is not None and dominant_target is not None:
        metrics.append(
            StructureMetric(
                label="Dominant Pair Error",
                value=f"{dominant_error:.3f}",
                detail=(
                    f"Observed dominant hotspot distance {dominant_observed:.3f} versus target "
                    f"{dominant_target:.3f} in the local atomistic reference frame."
                ),
                status="good" if dominant_error <= 1.0 else "active" if dominant_error <= 2.5 else "warn",
            )
        )

    return StructureComparisonReport(
        title="Atomistic Alignment",
        summary=(
            f"Similarity-fitted comparison against local atomistic centroids derived from {target.source_pdb_id}. "
            "These metrics are grounded in real atom coordinates, but the live simulation itself remains coarse-grained."
        ),
        metrics=tuple(metrics),
        metadata={
            "state_id": str(normalized_state_id),
            "source_pdb_id": target.source_pdb_id,
            "matched_landmark_count": len(matched_targets),
            "target_landmark_count": len(target.landmarks),
            "contact_hits": contact_hits,
            "contact_total": contact_total,
            "representation": target.metadata.get("representation", "unknown"),
            "fit_scale": scale,
        },
    )
