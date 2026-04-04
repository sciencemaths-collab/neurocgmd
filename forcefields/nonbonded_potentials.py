"""Production nonbonded interaction terms: WCA, shifted-force LJ, switch function, Coulomb."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from core.exceptions import ContractValidationError


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Coulomb constant in kJ*nm/(mol*e^2) for use with CG distance units (nm).
COULOMB_CONSTANT: float = 138.935458

# LJ minimum location factor: 2^(1/6).
_TWO_ONE_SIXTH: float = 2.0 ** (1.0 / 6.0)


# ---------------------------------------------------------------------------
# WCA (Weeks-Chandler-Andersen) purely repulsive potential
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class WCAPotential:
    """Weeks-Chandler-Andersen purely repulsive potential.

    The LJ potential is truncated and shifted at its minimum r_c = 2^(1/6)*sigma
    so that only the repulsive branch survives.

    E_WCA(r) = 4*eps*((sigma/r)^12 - (sigma/r)^6) + eps   for r < r_c
             = 0                                             for r >= r_c

    Classification: [established]
    """

    sigma: float
    epsilon: float
    name: str = "wca_potential"
    classification: str = "[established]"

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.sigma <= 0.0:
            issues.append("WCAPotential sigma must be positive.")
        if self.epsilon < 0.0:
            issues.append("WCAPotential epsilon must be non-negative.")
        return tuple(issues)

    @property
    def cutoff(self) -> float:
        """Return the WCA cutoff distance r_c = 2^(1/6) * sigma."""
        return _TWO_ONE_SIXTH * self.sigma

    def energy(self, r: float) -> float:
        """Return the WCA potential energy at distance *r*."""
        if r >= self.cutoff:
            return 0.0
        sr6 = (self.sigma / r) ** 6
        return 4.0 * self.epsilon * (sr6 * sr6 - sr6) + self.epsilon

    def force_magnitude(self, r: float) -> float:
        """Return the magnitude of the WCA force at distance *r*.

        The force is purely repulsive (positive = repulsion along r-hat).
        """
        if r >= self.cutoff:
            return 0.0
        sr6 = (self.sigma / r) ** 6
        return 24.0 * self.epsilon * (2.0 * sr6 * sr6 - sr6) / r


# ---------------------------------------------------------------------------
# Shifted-force Lennard-Jones
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ShiftedForceLJ:
    """Lennard-Jones potential with a linear force-shift at the cutoff.

    Both the energy and the force go continuously to zero at *cutoff*:

        E_sf(r) = E_LJ(r) - E_LJ(r_c) - (r - r_c) * dE_LJ/dr|_{r_c}

    Classification: [established]
    """

    sigma: float
    epsilon: float
    cutoff: float
    name: str = "shifted_force_lj"
    classification: str = "[established]"

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.sigma <= 0.0:
            issues.append("ShiftedForceLJ sigma must be positive.")
        if self.epsilon < 0.0:
            issues.append("ShiftedForceLJ epsilon must be non-negative.")
        if self.cutoff <= 0.0:
            issues.append("ShiftedForceLJ cutoff must be positive.")
        return tuple(issues)

    # ---- LJ helpers (un-shifted) ------------------------------------------

    def _lj_energy(self, r: float) -> float:
        sr6 = (self.sigma / r) ** 6
        return 4.0 * self.epsilon * (sr6 * sr6 - sr6)

    def _lj_force(self, r: float) -> float:
        """Return -dE_LJ/dr (positive means repulsive)."""
        sr6 = (self.sigma / r) ** 6
        return 24.0 * self.epsilon * (2.0 * sr6 * sr6 - sr6) / r

    def _lj_dE_dr(self, r: float) -> float:
        """Return dE_LJ/dr (the derivative, not the force)."""
        return -self._lj_force(r)

    # ---- public API -------------------------------------------------------

    def energy(self, r: float) -> float:
        """Return the shifted-force LJ energy at distance *r*."""
        if r >= self.cutoff:
            return 0.0
        rc = self.cutoff
        return self._lj_energy(r) - self._lj_energy(rc) - (r - rc) * self._lj_dE_dr(rc)

    def force_magnitude(self, r: float) -> float:
        """Return the magnitude of the shifted-force LJ force at distance *r*.

        Positive values denote repulsion.
        """
        if r >= self.cutoff:
            return 0.0
        rc = self.cutoff
        # F_sf(r) = F_LJ(r) - F_LJ(r_c)  (force shift so F(rc) = 0)
        return self._lj_force(r) - self._lj_force(rc)


# ---------------------------------------------------------------------------
# Switch function
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SwitchFunction:
    """Smooth switching function that transitions from 1 to 0 between two cutoffs.

    S(r) = 1                                                        for r < r_inner
    S(r) = (r_outer^2 - r^2)^2 * (r_outer^2 + 2*r^2 - 3*r_inner^2)
           / (r_outer^2 - r_inner^2)^3                              for r_inner <= r <= r_outer
    S(r) = 0                                                        for r > r_outer
    """

    inner_cutoff: float
    outer_cutoff: float

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.inner_cutoff < 0.0:
            issues.append("SwitchFunction inner_cutoff must be non-negative.")
        if self.outer_cutoff <= self.inner_cutoff:
            issues.append("SwitchFunction outer_cutoff must be greater than inner_cutoff.")
        return tuple(issues)

    def evaluate(self, r: float) -> tuple[float, float]:
        """Return ``(switch_value, switch_derivative)`` at distance *r*."""
        if r < self.inner_cutoff:
            return 1.0, 0.0
        if r > self.outer_cutoff:
            return 0.0, 0.0

        r2 = r * r
        ro2 = self.outer_cutoff * self.outer_cutoff
        ri2 = self.inner_cutoff * self.inner_cutoff
        diff2 = ro2 - ri2
        denominator = diff2 * diff2 * diff2  # (r_outer^2 - r_inner^2)^3

        a = ro2 - r2  # (r_outer^2 - r^2)
        b = ro2 + 2.0 * r2 - 3.0 * ri2  # (r_outer^2 + 2*r^2 - 3*r_inner^2)

        switch_value = (a * a * b) / denominator

        # Derivative dS/dr via product rule:
        # dS/dr = [2*a*(-2r)*b + a^2*(4r)] / denominator
        #       = 2*r*a*(-2*b + 2*a) / denominator   -- simplified --
        #       but let's keep the clean product-rule form:
        # d(a)/dr = -2*r,  d(b)/dr = 4*r
        da_dr = -2.0 * r
        db_dr = 4.0 * r
        switch_derivative = (2.0 * a * da_dr * b + a * a * db_dr) / denominator

        return switch_value, switch_derivative


# ---------------------------------------------------------------------------
# Coulomb potential
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CoulombPotential:
    """Point-charge Coulomb interaction in CG distance/energy units.

    E(r) = COULOMB_CONSTANT * qi * qj / r
    F(r) = COULOMB_CONSTANT * qi * qj / r^2  (magnitude along r-hat)

    COULOMB_CONSTANT = 138.935458 kJ*nm/(mol*e^2)
    Classification: [established]
    """

    name: str = "coulomb_potential"
    classification: str = "[established]"

    @staticmethod
    def energy(qi: float, qj: float, r: float) -> float:
        """Return the Coulomb energy between charges *qi* and *qj* at distance *r*."""
        return COULOMB_CONSTANT * qi * qj / r

    @staticmethod
    def force(qi: float, qj: float, r: float) -> float:
        """Return the magnitude of the Coulomb force along the pair axis."""
        return COULOMB_CONSTANT * qi * qj / (r * r)
