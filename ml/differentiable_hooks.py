"""Parameter-gradient-ready hooks for ML modules without locking into one autodiff stack."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, coerce_scalar
from ml.scalable_residual import ScalableResidualModel


@dataclass(frozen=True, slots=True)
class DifferentiableParameter(ValidatableComponent):
    """One trainable parameter block exposed through a stable differentiable hook."""

    name: str
    parameter_group: str
    shape: tuple[int, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(value) for value in self.shape))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name.strip():
            issues.append("name must be a non-empty string.")
        if not self.parameter_group.strip():
            issues.append("parameter_group must be a non-empty string.")
        if any(dimension <= 0 for dimension in self.shape):
            issues.append("shape dimensions must be strictly positive.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ParameterGradient(ValidatableComponent):
    """Finite-difference or external gradient estimate for one parameter block."""

    parameter: DifferentiableParameter
    values: tuple[float, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(coerce_scalar(value, "gradient_value") for value in self.values))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        expected_size = 1
        for dimension in self.parameter.shape:
            expected_size *= dimension
        if len(self.values) != expected_size:
            issues.append("gradient vector length must match the parameter shape.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ParameterSnapshot(ValidatableComponent):
    """Serializable parameter snapshot for reversible updates."""

    parameter: DifferentiableParameter
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(coerce_scalar(value, "parameter_value") for value in self.values))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        expected_size = 1
        for dimension in self.parameter.shape:
            expected_size *= dimension
        if len(self.values) != expected_size:
            issues.append("snapshot vector length must match the parameter shape.")
        return tuple(issues)


class DifferentiableModule(Protocol):
    """Stable differentiable hook independent of any one autodiff runtime."""

    def list_parameters(self) -> tuple[DifferentiableParameter, ...]:
        """Return the exposed differentiable parameter blocks."""

    def snapshot_parameters(self) -> tuple[ParameterSnapshot, ...]:
        """Return the current parameter values."""

    def restore_parameters(self, snapshots: Sequence[ParameterSnapshot]) -> None:
        """Restore previously captured parameter values."""

    def parameter_values(self, parameter_name: str) -> tuple[float, ...]:
        """Read one parameter block."""

    def assign_parameter_values(self, parameter_name: str, values: Sequence[float]) -> None:
        """Write one parameter block."""


@dataclass(slots=True)
class ScalableResidualDifferentiableAdapter(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Expose the trainable head of `ScalableResidualModel` through a stable hook."""

    model: ScalableResidualModel
    name: str = "scalable_residual_differentiable_adapter"
    classification: str = "[adapted]"

    def describe_role(self) -> str:
        return (
            "Exposes trainable parameter blocks from the scalable piece-local residual model "
            "so future autodiff or finite-difference workflows can plug in cleanly."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return ("ml/scalable_residual.py",)

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/backend_compute_spine.md",)

    def validate(self) -> tuple[str, ...]:
        return ()

    def list_parameters(self) -> tuple[DifferentiableParameter, ...]:
        return (
            DifferentiableParameter(
                name="energy_head_weights",
                parameter_group="energy_head",
                shape=(len(self.model._corrector._energy_weights),),
            ),
            DifferentiableParameter(name="energy_head_bias", parameter_group="energy_head", shape=(1,)),
            DifferentiableParameter(
                name="force_head_weights",
                parameter_group="force_head",
                shape=(len(self.model._corrector._force_weights),),
            ),
            DifferentiableParameter(name="force_head_bias", parameter_group="force_head", shape=(1,)),
        )

    def snapshot_parameters(self) -> tuple[ParameterSnapshot, ...]:
        return tuple(
            ParameterSnapshot(parameter=parameter, values=self.parameter_values(parameter.name))
            for parameter in self.list_parameters()
        )

    def restore_parameters(self, snapshots: Sequence[ParameterSnapshot]) -> None:
        for snapshot in snapshots:
            self.assign_parameter_values(snapshot.parameter.name, snapshot.values)

    def parameter_values(self, parameter_name: str) -> tuple[float, ...]:
        if parameter_name == "energy_head_weights":
            return tuple(self.model._corrector._energy_weights)
        if parameter_name == "energy_head_bias":
            return (self.model._corrector._energy_bias,)
        if parameter_name == "force_head_weights":
            return tuple(self.model._corrector._force_weights)
        if parameter_name == "force_head_bias":
            return (self.model._corrector._force_bias,)
        raise KeyError(parameter_name)

    def assign_parameter_values(self, parameter_name: str, values: Sequence[float]) -> None:
        values_tuple = tuple(coerce_scalar(value, "parameter_value") for value in values)
        if parameter_name == "energy_head_weights":
            if len(values_tuple) != len(self.model._corrector._energy_weights):
                raise ContractValidationError("energy_head_weights length does not match the model.")
            self.model._corrector._energy_weights = list(values_tuple)
            return
        if parameter_name == "energy_head_bias":
            if len(values_tuple) != 1:
                raise ContractValidationError("energy_head_bias expects exactly one value.")
            self.model._corrector._energy_bias = values_tuple[0]
            return
        if parameter_name == "force_head_weights":
            if len(values_tuple) != len(self.model._corrector._force_weights):
                raise ContractValidationError("force_head_weights length does not match the model.")
            self.model._corrector._force_weights = list(values_tuple)
            return
        if parameter_name == "force_head_bias":
            if len(values_tuple) != 1:
                raise ContractValidationError("force_head_bias expects exactly one value.")
            self.model._corrector._force_bias = values_tuple[0]
            return
        raise KeyError(parameter_name)


@dataclass(slots=True)
class FiniteDifferenceGradientEstimator(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Finite-difference gradient estimator over a differentiable hook."""

    epsilon: float = 1e-5
    name: str = "finite_difference_gradient_estimator"
    classification: str = "[adapted]"

    def describe_role(self) -> str:
        return "Approximates parameter gradients through a stable hook without assuming one autodiff backend."

    def declared_dependencies(self) -> tuple[str, ...]:
        return ("ml/differentiable_hooks.py",)

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/backend_compute_spine.md",)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.epsilon <= 0.0:
            issues.append("epsilon must be strictly positive.")
        return tuple(issues)

    def estimate(
        self,
        module: DifferentiableModule,
        objective: Callable[[DifferentiableModule], float],
    ) -> tuple[ParameterGradient, ...]:
        base_snapshots = module.snapshot_parameters()
        gradients: list[ParameterGradient] = []
        for parameter in module.list_parameters():
            original_values = list(module.parameter_values(parameter.name))
            partials: list[float] = []
            for index in range(len(original_values)):
                plus = list(original_values)
                minus = list(original_values)
                plus[index] += self.epsilon
                minus[index] -= self.epsilon
                module.assign_parameter_values(parameter.name, plus)
                plus_value = objective(module)
                module.assign_parameter_values(parameter.name, minus)
                minus_value = objective(module)
                partials.append((plus_value - minus_value) / (2.0 * self.epsilon))
                module.assign_parameter_values(parameter.name, original_values)
            gradients.append(
                ParameterGradient(
                    parameter=parameter,
                    values=tuple(partials),
                    metadata=FrozenMetadata({"estimator": self.name}),
                )
            )
        module.restore_parameters(base_snapshots)
        return tuple(gradients)
