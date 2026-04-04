"""Custom exceptions for architectural and scaffold-level failures."""


class ProjectScaffoldError(RuntimeError):
    """Raised when the repository scaffold is missing required structure."""


class ContractValidationError(ValueError):
    """Raised when a component violates an explicit architectural contract."""

