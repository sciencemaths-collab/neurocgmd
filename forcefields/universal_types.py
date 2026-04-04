"""Universal bead-type taxonomy and interaction matrix for the NeuroCGMD forcefield.

Inspired by the MARTINI coarse-grained force field interaction matrix but
generalised to handle *all* material classes relevant to multiscale brain
simulations: proteins, lipids, water, ions, surfaces, polymers, and ligands.

The module is self-contained.  It depends only on ``core.types.FrozenMetadata``
and ``core.exceptions.ContractValidationError``; it has no dependencies on other
forcefield modules.

Key design decisions
--------------------
* **ChemicalCharacter** encodes the MARTINI Qp/Qn/P/N/C/S taxonomy as a
  ``StrEnum`` so that it can be used as dictionary keys, serialised to JSON, and
  compared ergonomically.
* **INTERACTION_MATRIX** is a symmetric dict keyed by a *frozenset* pair of
  ``ChemicalCharacter`` values.  A helper builds the symmetric pairs from an
  upper-triangle specification so the matrix is defined only once.
* **compute_pair_parameters** supports three mixing rules --
  Lorentz--Berthelot, geometric, and the native MARTINI matrix look-up -- and
  returns ``(sigma, epsilon)`` ready for a Lennard-Jones potential.
* **STANDARD_BEADS** provides a curated library of ~30 bead types covering
  protein backbone/side-chain, water, common ions, lipid head/tail groups,
  polymer repeat units, inorganic/organic surfaces, and drug-like ligand
  fragments.
* **AMINO_ACID_BEAD_MAP** maps every standard 3-letter amino-acid code to an
  ordered list of bead names from STANDARD_BEADS, following the MARTINI protein
  mapping convention (~1-4 beads per residue).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import sqrt
from typing import Dict, List, Tuple

from core.exceptions import ContractValidationError
from core.types import FrozenMetadata

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class MaterialClass(StrEnum):
    """Broad material class to which a coarse-grained bead belongs."""

    PROTEIN = "PROTEIN"
    PEPTIDE = "PEPTIDE"
    WATER = "WATER"
    ION = "ION"
    LIPID = "LIPID"
    LIGAND = "LIGAND"
    POLYMER = "POLYMER"
    ORGANIC_SURFACE = "ORGANIC_SURFACE"
    INORGANIC_SURFACE = "INORGANIC_SURFACE"
    CUSTOM = "CUSTOM"


class ChemicalCharacter(StrEnum):
    """Chemical nature of a bead -- the MARTINI Q/P/N/C/S categories extended
    with explicit sign for charged beads."""

    CHARGED_POSITIVE = "Qp"  # e.g. Arg, Lys, Na+, K+
    CHARGED_NEGATIVE = "Qn"  # e.g. Asp, Glu, Cl-
    POLAR = "P"              # e.g. Ser, Thr, water, lipid headgroup, hydroxyl
    INTERMEDIATE = "N"       # e.g. His, Tyr, amide, mixed character
    APOLAR = "C"             # e.g. Leu, Ile, Val, lipid tail, hydrocarbon
    SPECIAL = "S"            # e.g. Cys (disulfide), Pro (ring), backbone


class InteractionStrength(StrEnum):
    """Seven interaction-strength levels from the MARTINI force field, ordered
    from most attractive (O) to most repulsive (VI)."""

    O = "O"      # supra-attractive
    I = "I"      # attractive
    II = "II"    # semi-attractive
    III = "III"  # intermediate
    IV = "IV"    # semi-repulsive
    V = "V"      # repulsive
    VI = "VI"    # supra-repulsive


class MixingRule(StrEnum):
    """Mixing rules for computing pair parameters between two bead types."""

    LORENTZ_BERTHELOT = "LORENTZ_BERTHELOT"
    GEOMETRIC = "GEOMETRIC"
    MARTINI_MATRIX = "MARTINI_MATRIX"


# ---------------------------------------------------------------------------
# UniversalBeadType dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UniversalBeadType:
    """A single coarse-grained bead type in the NeuroCGMD universal taxonomy.

    Parameters
    ----------
    name:
        Unique string identifier (e.g. ``"P5"``, ``"C1"``, ``"W"``).
    material_class:
        The broad material family this bead belongs to.
    chemical_character:
        The MARTINI-style chemical character (Qp, Qn, P, N, C, S).
    subtype:
        Fine-grained level within a character class (0--5).  Subtypes allow
        discrimination between, e.g., strongly polar (0) and weakly polar (5)
        beads sharing the same ``chemical_character``.
    sigma:
        Lennard-Jones sigma in nm.  The MARTINI default is 0.47 nm.
    mass:
        Mass in atomic mass units.  MARTINI maps ~4 heavy atoms per bead,
        giving a default of ~72 amu.
    charge:
        Partial or formal charge in elementary charges (*e*).
    description:
        Human-readable description of the bead type.
    metadata:
        Arbitrary immutable key-value metadata.
    """

    name: str
    material_class: MaterialClass
    chemical_character: ChemicalCharacter
    subtype: int = 0
    sigma: float = 0.47
    mass: float = 72.0
    charge: float = 0.0
    description: str = ""
    metadata: FrozenMetadata = field(default_factory=lambda: FrozenMetadata({}))

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ContractValidationError(
                "UniversalBeadType.name must be a non-empty string."
            )
        if not 0 <= self.subtype <= 5:
            raise ContractValidationError(
                f"UniversalBeadType.subtype must be in [0, 5]; got {self.subtype}."
            )
        if self.sigma <= 0.0:
            raise ContractValidationError(
                f"UniversalBeadType.sigma must be positive; got {self.sigma}."
            )
        if self.mass <= 0.0:
            raise ContractValidationError(
                f"UniversalBeadType.mass must be positive; got {self.mass}."
            )


# ---------------------------------------------------------------------------
# Interaction matrix
# ---------------------------------------------------------------------------

# Shorthand aliases for readability.
_Qp = ChemicalCharacter.CHARGED_POSITIVE
_Qn = ChemicalCharacter.CHARGED_NEGATIVE
_P = ChemicalCharacter.POLAR
_N = ChemicalCharacter.INTERMEDIATE
_C = ChemicalCharacter.APOLAR
_S = ChemicalCharacter.SPECIAL

_O = InteractionStrength.O
_I = InteractionStrength.I
_II = InteractionStrength.II
_III = InteractionStrength.III
_IV = InteractionStrength.IV
_V = InteractionStrength.V
_VI = InteractionStrength.VI

# Upper-triangle definition (including diagonal).  The helper below mirrors it
# to produce the full symmetric matrix.
_INTERACTION_UPPER: list[tuple[ChemicalCharacter, ChemicalCharacter, InteractionStrength]] = [
    # Qp row
    (_Qp, _Qp, _O),   (_Qp, _Qn, _O),  (_Qp, _P, _I),
    (_Qp, _N, _I),     (_Qp, _C, _IV),   (_Qp, _S, _I),
    # Qn row (only entries where col > row in enum order)
    (_Qn, _Qn, _O),    (_Qn, _P, _I),
    (_Qn, _N, _I),     (_Qn, _C, _IV),   (_Qn, _S, _I),
    # P row
    (_P, _P, _I),      (_P, _N, _II),
    (_P, _C, _IV),     (_P, _S, _II),
    # N row
    (_N, _N, _III),    (_N, _C, _IV),    (_N, _S, _III),
    # C row
    (_C, _C, _II),     (_C, _S, _III),
    # S row
    (_S, _S, _III),
]


def _build_interaction_matrix(
    upper: list[tuple[ChemicalCharacter, ChemicalCharacter, InteractionStrength]],
) -> Dict[Tuple[ChemicalCharacter, ChemicalCharacter], InteractionStrength]:
    """Build a symmetric interaction matrix from an upper-triangle spec.

    The matrix is keyed by ``(row, col)`` tuples.  Both ``(a, b)`` and
    ``(b, a)`` map to the same ``InteractionStrength``.
    """
    matrix: Dict[Tuple[ChemicalCharacter, ChemicalCharacter], InteractionStrength] = {}
    for a, b, level in upper:
        matrix[(a, b)] = level
        matrix[(b, a)] = level
    return matrix


INTERACTION_MATRIX: Dict[
    Tuple[ChemicalCharacter, ChemicalCharacter], InteractionStrength
] = _build_interaction_matrix(_INTERACTION_UPPER)
"""Symmetric interaction matrix mapping a pair of :class:`ChemicalCharacter`
values to an :class:`InteractionStrength` level.

Usage::

    level = INTERACTION_MATRIX[(ChemicalCharacter.POLAR, ChemicalCharacter.APOLAR)]
    # -> InteractionStrength.IV
"""

# ---------------------------------------------------------------------------
# Epsilon look-up table
# ---------------------------------------------------------------------------

EPSILON_FOR_LEVEL: Dict[InteractionStrength, float] = {
    InteractionStrength.O: 5.6,    # supra-attractive    (kJ/mol)
    InteractionStrength.I: 5.0,    # attractive
    InteractionStrength.II: 4.5,   # semi-attractive
    InteractionStrength.III: 4.0,  # intermediate
    InteractionStrength.IV: 3.5,   # semi-repulsive
    InteractionStrength.V: 3.1,    # repulsive
    InteractionStrength.VI: 2.7,   # supra-repulsive
}
"""Maps each :class:`InteractionStrength` level to its Lennard-Jones well depth
*epsilon* in kJ/mol."""

# ---------------------------------------------------------------------------
# Pair-parameter computation
# ---------------------------------------------------------------------------


def compute_pair_parameters(
    bead_a: UniversalBeadType,
    bead_b: UniversalBeadType,
    mixing_rule: MixingRule = MixingRule.MARTINI_MATRIX,
) -> Tuple[float, float]:
    """Compute combined Lennard-Jones pair parameters for two bead types.

    Parameters
    ----------
    bead_a, bead_b:
        The two bead types whose interaction parameters are requested.
    mixing_rule:
        One of :class:`MixingRule`.  Defaults to ``MARTINI_MATRIX``.

    Returns
    -------
    (sigma, epsilon):
        Combined sigma (nm) and epsilon (kJ/mol).

    Raises
    ------
    ContractValidationError
        If the pair is not found in the interaction matrix when using the
        ``MARTINI_MATRIX`` rule, or if an unknown mixing rule is supplied.
    """
    if mixing_rule == MixingRule.MARTINI_MATRIX:
        key = (bead_a.chemical_character, bead_b.chemical_character)
        level = INTERACTION_MATRIX.get(key)
        if level is None:
            raise ContractValidationError(
                f"No interaction matrix entry for pair "
                f"({bead_a.chemical_character}, {bead_b.chemical_character})."
            )
        epsilon = EPSILON_FOR_LEVEL[level]
        sigma = (bead_a.sigma + bead_b.sigma) / 2.0
        return (sigma, epsilon)

    if mixing_rule == MixingRule.LORENTZ_BERTHELOT:
        sigma = (bead_a.sigma + bead_b.sigma) / 2.0
        epsilon = sqrt(bead_a.sigma * bead_b.sigma)  # placeholder per-bead eps
        # NOTE: Lorentz-Berthelot normally combines per-bead epsilons.  Since
        # UniversalBeadType does not carry a per-bead epsilon, we fall back to
        # the matrix epsilon and combine via geometric mean.
        key = (bead_a.chemical_character, bead_b.chemical_character)
        level_a = INTERACTION_MATRIX.get(
            (bead_a.chemical_character, bead_a.chemical_character)
        )
        level_b = INTERACTION_MATRIX.get(
            (bead_b.chemical_character, bead_b.chemical_character)
        )
        if level_a is None or level_b is None:
            raise ContractValidationError(
                "Cannot determine per-bead epsilon for Lorentz-Berthelot mixing; "
                "self-interaction entry missing in matrix."
            )
        eps_a = EPSILON_FOR_LEVEL[level_a]
        eps_b = EPSILON_FOR_LEVEL[level_b]
        epsilon = sqrt(eps_a * eps_b)
        return (sigma, epsilon)

    if mixing_rule == MixingRule.GEOMETRIC:
        sigma = sqrt(bead_a.sigma * bead_b.sigma)
        level_a = INTERACTION_MATRIX.get(
            (bead_a.chemical_character, bead_a.chemical_character)
        )
        level_b = INTERACTION_MATRIX.get(
            (bead_b.chemical_character, bead_b.chemical_character)
        )
        if level_a is None or level_b is None:
            raise ContractValidationError(
                "Cannot determine per-bead epsilon for geometric mixing; "
                "self-interaction entry missing in matrix."
            )
        eps_a = EPSILON_FOR_LEVEL[level_a]
        eps_b = EPSILON_FOR_LEVEL[level_b]
        epsilon = sqrt(eps_a * eps_b)
        return (sigma, epsilon)

    raise ContractValidationError(f"Unknown mixing rule: {mixing_rule!r}")


# ---------------------------------------------------------------------------
# Standard bead library
# ---------------------------------------------------------------------------

def _bead(
    name: str,
    material: MaterialClass,
    char: ChemicalCharacter,
    *,
    subtype: int = 0,
    sigma: float = 0.47,
    mass: float = 72.0,
    charge: float = 0.0,
    description: str = "",
) -> UniversalBeadType:
    """Convenience factory for building library beads with empty metadata."""
    return UniversalBeadType(
        name=name,
        material_class=material,
        chemical_character=char,
        subtype=subtype,
        sigma=sigma,
        mass=mass,
        charge=charge,
        description=description,
        metadata=FrozenMetadata({}),
    )


# ---- Protein beads --------------------------------------------------------

_BB = _bead(
    "BB", MaterialClass.PROTEIN, ChemicalCharacter.SPECIAL,
    description="Protein backbone bead (~4 heavy atoms: N-CA-C-O).",
)
_P5 = _bead(
    "P5", MaterialClass.PROTEIN, ChemicalCharacter.POLAR,
    description="Strongly polar side chain (Asp/Glu/Lys/Arg sidechain charge group).",
)
_Nda = _bead(
    "Nda", MaterialClass.PROTEIN, ChemicalCharacter.INTERMEDIATE,
    description="H-bond donor/acceptor side chain (Ser, Thr, Asn, Gln).",
)
_C1 = _bead(
    "C1", MaterialClass.PROTEIN, ChemicalCharacter.APOLAR,
    description="Apolar side chain (Leu, Ile, Val, Met, Ala, Cys, Pro).",
)
_SC4 = _bead(
    "SC4", MaterialClass.PROTEIN, ChemicalCharacter.APOLAR,
    subtype=4, sigma=0.43,
    description="Aromatic ring bead (Phe, Trp, Tyr ring fragments).",
)
_Qp_Lys = _bead(
    "Qp_Lys", MaterialClass.PROTEIN, ChemicalCharacter.CHARGED_POSITIVE,
    charge=1.0,
    description="Lysine terminal ammonium (+NH3).",
)
_Qp_Arg = _bead(
    "Qp_Arg", MaterialClass.PROTEIN, ChemicalCharacter.CHARGED_POSITIVE,
    charge=1.0,
    description="Arginine guanidinium group.",
)

# ---- Water beads -----------------------------------------------------------

_W = _bead(
    "W", MaterialClass.WATER, ChemicalCharacter.POLAR,
    description="Standard MARTINI water bead (4 water molecules).",
)
_WF = _bead(
    "WF", MaterialClass.WATER, ChemicalCharacter.POLAR,
    description="Antifreeze water bead (prevents ice-like ordering artifacts).",
)

# ---- Ion beads -------------------------------------------------------------

_Qp_Na = _bead(
    "Qp_Na", MaterialClass.ION, ChemicalCharacter.CHARGED_POSITIVE,
    mass=23.0, charge=1.0,
    description="Sodium ion (Na+).",
)
_Qp_K = _bead(
    "Qp_K", MaterialClass.ION, ChemicalCharacter.CHARGED_POSITIVE,
    mass=39.0, charge=1.0,
    description="Potassium ion (K+).",
)
_Qp_Ca = _bead(
    "Qp_Ca", MaterialClass.ION, ChemicalCharacter.CHARGED_POSITIVE,
    mass=40.0, charge=2.0,
    description="Calcium ion (Ca2+).",
)
_Qn_Cl = _bead(
    "Qn_Cl", MaterialClass.ION, ChemicalCharacter.CHARGED_NEGATIVE,
    mass=35.5, charge=-1.0,
    description="Chloride ion (Cl-).",
)

# ---- Lipid beads -----------------------------------------------------------

_Qo_PO4 = _bead(
    "Qo_PO4", MaterialClass.LIPID, ChemicalCharacter.CHARGED_NEGATIVE,
    charge=-1.0,
    description="Phosphate headgroup (PO4-).",
)
_P1_NH3 = _bead(
    "P1_NH3", MaterialClass.LIPID, ChemicalCharacter.POLAR,
    description="Amine headgroup (NH3/ethanolamine).",
)
_C1_tail = _bead(
    "C1_tail", MaterialClass.LIPID, ChemicalCharacter.APOLAR,
    mass=56.0,
    description="Saturated lipid tail segment (~4 CH2 groups).",
)
_C3_tail = _bead(
    "C3_tail", MaterialClass.LIPID, ChemicalCharacter.APOLAR,
    subtype=3, mass=56.0,
    description="Unsaturated lipid tail segment (contains C=C).",
)

# ---- Polymer beads ---------------------------------------------------------

_PEG = _bead(
    "PEG", MaterialClass.POLYMER, ChemicalCharacter.INTERMEDIATE,
    sigma=0.43, mass=44.0,
    description="Polyethylene glycol repeat unit (-CH2-O-CH2-).",
)
_PS = _bead(
    "PS", MaterialClass.POLYMER, ChemicalCharacter.APOLAR,
    mass=104.0,
    description="Polystyrene repeat unit.",
)

# ---- Surface beads ---------------------------------------------------------

_Au_surf = _bead(
    "Au_surf", MaterialClass.INORGANIC_SURFACE, ChemicalCharacter.APOLAR,
    mass=197.0,
    description="Gold surface atom bead.",
)
_SiO2_surf = _bead(
    "SiO2_surf", MaterialClass.INORGANIC_SURFACE, ChemicalCharacter.POLAR,
    mass=60.0,
    description="Silica surface bead (SiO2).",
)
_C_graph = _bead(
    "C_graph", MaterialClass.ORGANIC_SURFACE, ChemicalCharacter.APOLAR,
    sigma=0.34, mass=12.0,
    description="Graphene carbon bead.",
)
_SAM_CH3 = _bead(
    "SAM_CH3", MaterialClass.ORGANIC_SURFACE, ChemicalCharacter.APOLAR,
    description="Self-assembled monolayer methyl terminus (-CH3).",
)
_SAM_OH = _bead(
    "SAM_OH", MaterialClass.ORGANIC_SURFACE, ChemicalCharacter.POLAR,
    description="Self-assembled monolayer hydroxyl terminus (-OH).",
)

# ---- Ligand beads ----------------------------------------------------------

_LIG_donor = _bead(
    "LIG_donor", MaterialClass.LIGAND, ChemicalCharacter.POLAR,
    sigma=0.43,
    description="Ligand H-bond donor fragment.",
)
_LIG_acceptor = _bead(
    "LIG_acceptor", MaterialClass.LIGAND, ChemicalCharacter.POLAR,
    sigma=0.43,
    description="Ligand H-bond acceptor fragment.",
)
_LIG_hydrophobic = _bead(
    "LIG_hydrophobic", MaterialClass.LIGAND, ChemicalCharacter.APOLAR,
    sigma=0.43,
    description="Ligand hydrophobic group.",
)
_LIG_aromatic = _bead(
    "LIG_aromatic", MaterialClass.LIGAND, ChemicalCharacter.APOLAR,
    subtype=4, sigma=0.36,
    description="Ligand aromatic ring fragment.",
)

# ---- Assembled library -----------------------------------------------------

STANDARD_BEADS: Dict[str, UniversalBeadType] = {
    # Protein
    "BB": _BB,
    "P5": _P5,
    "Nda": _Nda,
    "C1": _C1,
    "SC4": _SC4,
    "Qp_Lys": _Qp_Lys,
    "Qp_Arg": _Qp_Arg,
    # Water
    "W": _W,
    "WF": _WF,
    # Ions
    "Qp_Na": _Qp_Na,
    "Qp_K": _Qp_K,
    "Qp_Ca": _Qp_Ca,
    "Qn_Cl": _Qn_Cl,
    # Lipids
    "Qo_PO4": _Qo_PO4,
    "P1_NH3": _P1_NH3,
    "C1_tail": _C1_tail,
    "C3_tail": _C3_tail,
    # Polymers
    "PEG": _PEG,
    "PS": _PS,
    # Surfaces
    "Au_surf": _Au_surf,
    "SiO2_surf": _SiO2_surf,
    "C_graph": _C_graph,
    "SAM_CH3": _SAM_CH3,
    "SAM_OH": _SAM_OH,
    # Ligands
    "LIG_donor": _LIG_donor,
    "LIG_acceptor": _LIG_acceptor,
    "LIG_hydrophobic": _LIG_hydrophobic,
    "LIG_aromatic": _LIG_aromatic,
}
"""Module-level dictionary mapping bead name to :class:`UniversalBeadType`.

Contains ~30 standard bead types spanning proteins, water, ions, lipids,
polymers, inorganic/organic surfaces, and ligand fragments.
"""

# ---------------------------------------------------------------------------
# Amino-acid to CG bead mapping
# ---------------------------------------------------------------------------

AMINO_ACID_BEAD_MAP: Dict[str, List[str]] = {
    "ALA": ["BB", "C1"],
    "GLY": ["BB"],
    "VAL": ["BB", "C1"],
    "LEU": ["BB", "C1"],
    "ILE": ["BB", "C1"],
    "PRO": ["BB", "C1"],
    "PHE": ["BB", "SC4", "SC4"],
    "TRP": ["BB", "SC4", "SC4", "Nda"],
    "MET": ["BB", "C1"],
    "SER": ["BB", "Nda"],
    "THR": ["BB", "Nda"],
    "CYS": ["BB", "C1"],
    "TYR": ["BB", "SC4", "Nda"],
    "ASN": ["BB", "Nda"],
    "GLN": ["BB", "Nda"],
    "ASP": ["BB", "P5"],
    "GLU": ["BB", "P5"],
    "LYS": ["BB", "P5", "Qp_Lys"],
    "ARG": ["BB", "P5", "Qp_Arg"],
    "HIS": ["BB", "Nda", "Nda"],
}
"""Maps standard 3-letter amino-acid codes to an ordered list of bead names
from :data:`STANDARD_BEADS`, following the MARTINI protein coarse-graining
convention (1--4 beads per residue)."""
