"""Physical constants and unit conversion utilities for MD simulations.

All constants use the GROMACS-compatible MD unit system:
nanometers (nm), picoseconds (ps), atomic mass units (amu),
kilojoules per mole (kJ/mol), elementary charges (e), and kelvin (K).

Classification: [established]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

from core.exceptions import ContractValidationError

# ---------------------------------------------------------------------------
# Physical constants in MD units (nm, ps, amu, kJ/mol, e, K)
# ---------------------------------------------------------------------------

BOLTZMANN_CONSTANT: float = 0.00831446
"""Boltzmann constant in kJ/(mol*K) -- the GROMACS convention."""

AVOGADRO: float = 6.02214076e23
"""Avogadro's number (mol^{-1})."""

ELEMENTARY_CHARGE: float = 1.602176634e-19
"""Elementary charge in coulombs (C)."""

VACUUM_PERMITTIVITY: float = 8.8541878128e-12
"""Vacuum permittivity in F/m."""

COULOMB_CONSTANT: float = 138.935458
"""Coulomb's constant in MD units: kJ*nm/(mol*e^2)."""

PLANCK: float = 6.62607015e-34
"""Planck constant in J*s."""

SPEED_OF_LIGHT: float = 299792.458
"""Speed of light in nm/ps."""

ONE_4PI_EPS0: float = 138.935458
"""1/(4*pi*eps0) in MD units: kJ*nm/(mol*e^2).  Same as COULOMB_CONSTANT."""


# ---------------------------------------------------------------------------
# Unit converter
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnitConverter:
    """Stateless converter between common MD unit conventions.

    All methods are pure functions with no side effects.
    """

    # -- energy -------------------------------------------------------------

    def kcal_to_kj(self, value: float) -> float:
        """Convert kilocalories per mole to kilojoules per mole."""
        return value * 4.184

    def kj_to_kcal(self, value: float) -> float:
        """Convert kilojoules per mole to kilocalories per mole."""
        return value / 4.184

    # -- length -------------------------------------------------------------

    def angstrom_to_nm(self, value: float) -> float:
        """Convert angstroms to nanometers."""
        return value * 0.1

    def nm_to_angstrom(self, value: float) -> float:
        """Convert nanometers to angstroms."""
        return value * 10.0

    # -- angle --------------------------------------------------------------

    def degree_to_radian(self, value: float) -> float:
        """Convert degrees to radians."""
        return value * pi / 180.0

    def radian_to_degree(self, value: float) -> float:
        """Convert radians to degrees."""
        return value * 180.0 / pi

    # -- pressure -----------------------------------------------------------

    def bar_to_kpa(self, value: float) -> float:
        """Convert bar to kilopascals."""
        return value * 100.0

    # -- time ---------------------------------------------------------------

    def fs_to_ps(self, value: float) -> float:
        """Convert femtoseconds to picoseconds."""
        return value * 0.001

    # -- mass ---------------------------------------------------------------

    def dalton_to_kg(self, value: float) -> float:
        """Convert daltons (amu) to kilograms."""
        return value * 1.66054e-27

    # -- thermodynamic convenience ------------------------------------------

    def thermal_energy(self, temperature: float) -> float:
        """Return kB * T in kJ/mol."""
        return BOLTZMANN_CONSTANT * temperature

    def beta(self, temperature: float) -> float:
        """Return 1 / (kB * T) in mol/kJ."""
        kt = BOLTZMANN_CONSTANT * temperature
        if kt == 0.0:
            raise ContractValidationError(
                "Cannot compute beta at zero thermal energy (temperature=0)."
            )
        return 1.0 / kt


# ---------------------------------------------------------------------------
# Unit validator
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnitValidator:
    """Validate physical quantities against domain constraints.

    Each ``validate_*`` method returns a tuple of human-readable issue
    strings.  An empty tuple signals a valid input.
    """

    def validate_length(
        self, value: float, name: str = "length",
    ) -> tuple[str, ...]:
        """Length must be strictly positive."""
        issues: list[str] = []
        if not isinstance(value, (int, float)):
            issues.append(f"{name} must be numeric; received {value!r}.")
        elif value <= 0.0:
            issues.append(f"{name} must be positive; received {value}.")
        return tuple(issues)

    def validate_energy(
        self, value: float, name: str = "energy",
    ) -> tuple[str, ...]:
        """Energy can be any finite float."""
        issues: list[str] = []
        if not isinstance(value, (int, float)):
            issues.append(f"{name} must be numeric; received {value!r}.")
        return tuple(issues)

    def validate_temperature(
        self, value: float, name: str = "temperature",
    ) -> tuple[str, ...]:
        """Temperature must be strictly positive."""
        issues: list[str] = []
        if not isinstance(value, (int, float)):
            issues.append(f"{name} must be numeric; received {value!r}.")
        elif value <= 0.0:
            issues.append(f"{name} must be positive; received {value}.")
        return tuple(issues)

    def validate_mass(
        self, value: float, name: str = "mass",
    ) -> tuple[str, ...]:
        """Mass must be strictly positive."""
        issues: list[str] = []
        if not isinstance(value, (int, float)):
            issues.append(f"{name} must be numeric; received {value!r}.")
        elif value <= 0.0:
            issues.append(f"{name} must be positive; received {value}.")
        return tuple(issues)

    def validate_charge(
        self, value: float, name: str = "charge",
    ) -> tuple[str, ...]:
        """Charge can be any finite float."""
        issues: list[str] = []
        if not isinstance(value, (int, float)):
            issues.append(f"{name} must be numeric; received {value!r}.")
        return tuple(issues)

    def validate_time_step(
        self,
        value: float,
        max_dt: float = 0.005,
        name: str = "time_step",
    ) -> tuple[str, ...]:
        """Time step must be in the half-open interval (0, max_dt]."""
        issues: list[str] = []
        if not isinstance(value, (int, float)):
            issues.append(f"{name} must be numeric; received {value!r}.")
        elif value <= 0.0:
            issues.append(f"{name} must be positive; received {value}.")
        elif value > max_dt:
            issues.append(
                f"{name} must be <= {max_dt} ps; received {value}."
            )
        return tuple(issues)
