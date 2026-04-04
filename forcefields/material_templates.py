"""Pre-built coarse-grained forcefield parameter templates for all 14 NeuroCGMD system types.

Each template encapsulates MARTINI-compatible (or domain-adapted) bonded
parameters for a single material class.  The ``TEMPLATE_REGISTRY`` maps
human-readable names to ``MaterialTemplate`` instances, and
``SYSTEM_CONFIGS`` defines how templates are combined for each supported
simulation system type.

Design notes
------------
* All bead types are referenced by *string* identifiers so that this module
  stays standalone — no imports from ``universal_types.py`` or sibling
  forcefield modules.  The ``universal_forcefield.py`` layer is responsible
  for wiring these string references to concrete bead-type objects.
* Dataclasses are frozen + slotted for immutability and memory efficiency.
* Energy units follow GROMACS / MARTINI conventions throughout:
  - distances in **nm**
  - angles in **radians**
  - energies in **kJ/mol**
  - force constants in **kJ/(mol nm^2)** or **kJ/(mol rad^2)** as appropriate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

from core.exceptions import ContractValidationError
from core.types import FrozenMetadata

# ---------------------------------------------------------------------------
# Component dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BondTemplate:
    """Harmonic bond between two CG bead types."""

    bead_type_a: str
    bead_type_b: str
    equilibrium_distance: float  # nm
    force_constant: float  # kJ/(mol*nm^2)
    description: str = ""


@dataclass(frozen=True, slots=True)
class AngleTemplate:
    """Harmonic angle among three CG bead types (b is the central bead)."""

    bead_type_a: str
    bead_type_b: str  # central
    bead_type_c: str
    equilibrium_angle: float  # radians
    force_constant: float  # kJ/(mol*rad^2)
    description: str = ""


@dataclass(frozen=True, slots=True)
class DihedralTemplate:
    """Periodic (Ryckaert-Bellemans style) dihedral among four CG bead types."""

    bead_types: tuple[str, str, str, str]
    force_constant: float  # kJ/mol
    multiplicity: int = 1
    phase: float = 0.0  # radians
    description: str = ""


@dataclass(frozen=True, slots=True)
class MaterialTemplate:
    """Complete bonded-parameter set for a single material class."""

    name: str
    description: str
    classification: str  # "[established]", "[adapted]", "[hybrid]", "[proposed novel]"
    bonds: tuple[BondTemplate, ...] = ()
    angles: tuple[AngleTemplate, ...] = ()
    dihedrals: tuple[DihedralTemplate, ...] = ()
    special_parameters: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        _valid = ("[established]", "[adapted]", "[hybrid]", "[proposed novel]")
        if self.classification not in _valid:
            raise ContractValidationError(
                f"MaterialTemplate classification must be one of {_valid}; "
                f"got {self.classification!r}."
            )


# ---------------------------------------------------------------------------
# Material templates
# ---------------------------------------------------------------------------

# -- 1. Protein (MARTINI-derived) ------------------------------------------

PROTEIN_TEMPLATE = MaterialTemplate(
    name="protein",
    description="Coarse-grained protein model (MARTINI-derived backbone/sidechain).",
    classification="[established]",
    bonds=(
        BondTemplate("BB", "BB", 0.35, 1250.0, "backbone-backbone bond"),
        BondTemplate("BB", "SC", 0.32, 5000.0, "backbone-sidechain bond"),
    ),
    angles=(
        AngleTemplate("BB", "BB", "BB", 2.09, 25.0, "backbone angle (~120 deg)"),
        AngleTemplate("BB", "BB", "SC", 2.62, 25.0, "backbone-sidechain branch (~150 deg)"),
    ),
    dihedrals=(
        DihedralTemplate(("BB", "BB", "BB", "BB"), 1.0, 1, pi, "backbone torsion"),
    ),
)

# -- 2. Peptide -------------------------------------------------------------

PEPTIDE_TEMPLATE = MaterialTemplate(
    name="peptide",
    description="Short-chain peptide model with cap beads and softer backbone angles.",
    classification="[established]",
    bonds=(
        BondTemplate("BB", "BB", 0.35, 1250.0, "backbone-backbone bond"),
        BondTemplate("BB", "SC", 0.32, 5000.0, "backbone-sidechain bond"),
        BondTemplate("CAP", "BB", 0.35, 1250.0, "terminal cap-backbone bond"),
    ),
    angles=(
        AngleTemplate("BB", "BB", "BB", 2.09, 20.0, "backbone angle (~120 deg, softer)"),
        AngleTemplate("BB", "BB", "SC", 2.62, 20.0, "backbone-sidechain branch (~150 deg, softer)"),
        AngleTemplate("CAP", "BB", "BB", 2.09, 20.0, "cap-backbone angle"),
    ),
    dihedrals=(
        DihedralTemplate(("BB", "BB", "BB", "BB"), 1.0, 1, pi, "backbone torsion"),
    ),
)

# -- 3. Water (single-site CG) ---------------------------------------------

WATER_TEMPLATE = MaterialTemplate(
    name="water",
    description="Single-site coarse-grained water (MARTINI W model, 4:1 mapping).",
    classification="[established]",
    special_parameters=FrozenMetadata({
        "model": "MARTINI_W",
        "self_interaction_epsilon": 5.0,
    }),
)

# -- 4. Ion -----------------------------------------------------------------

ION_TEMPLATE = MaterialTemplate(
    name="ion",
    description="Monovalent ions (Na+, Cl-) for implicit/explicit electrostatics.",
    classification="[established]",
    special_parameters=FrozenMetadata({
        "hydration_radius": 0.35,
        "debye_length_factor": 1.0,
    }),
)

# -- 5. Lipid DPPC ----------------------------------------------------------

LIPID_DPPC_TEMPLATE = MaterialTemplate(
    name="lipid_dppc",
    description="Dipalmitoylphosphatidylcholine (DPPC) — saturated two-tail lipid.",
    classification="[established]",
    bonds=(
        BondTemplate("PO4", "NH3", 0.47, 1250.0, "headgroup PO4-NH3"),
        BondTemplate("NH3", "GL", 0.37, 1250.0, "headgroup NH3-glycerol"),
        BondTemplate("GL", "C1_tail", 0.47, 1250.0, "glycerol to tail start"),
        BondTemplate("C1_tail", "C1_tail", 0.47, 1250.0, "tail chain C1-C1"),
    ),
    angles=(
        AngleTemplate("PO4", "NH3", "GL", 2.09, 25.0, "headgroup bend (~120 deg)"),
        AngleTemplate("NH3", "GL", "C1_tail", 3.14, 25.0, "glycerol-tail junction (~180 deg)"),
        AngleTemplate("GL", "C1_tail", "C1_tail", 3.14, 45.0, "tail entry angle (~180 deg)"),
        AngleTemplate("C1_tail", "C1_tail", "C1_tail", 3.14, 45.0, "tail chain angle (~180 deg)"),
    ),
)

# -- 6. Lipid DOPC (unsaturated) -------------------------------------------

LIPID_DOPC_TEMPLATE = MaterialTemplate(
    name="lipid_dopc",
    description="Dioleoylphosphatidylcholine (DOPC) — unsaturated two-tail lipid.",
    classification="[established]",
    bonds=(
        BondTemplate("PO4", "NH3", 0.47, 1250.0, "headgroup PO4-NH3"),
        BondTemplate("NH3", "GL", 0.37, 1250.0, "headgroup NH3-glycerol"),
        BondTemplate("GL", "C1_tail", 0.47, 1250.0, "glycerol to tail start"),
        BondTemplate("C1_tail", "C3_tail", 0.47, 1250.0, "saturated-to-unsaturated segment"),
        BondTemplate("C3_tail", "C3_tail", 0.47, 1250.0, "unsaturated tail chain"),
        BondTemplate("C3_tail", "C1_tail", 0.47, 1250.0, "unsaturated-to-saturated segment"),
    ),
    angles=(
        AngleTemplate("PO4", "NH3", "GL", 2.09, 25.0, "headgroup bend (~120 deg)"),
        AngleTemplate("NH3", "GL", "C1_tail", 3.14, 25.0, "glycerol-tail junction (~180 deg)"),
        AngleTemplate("GL", "C1_tail", "C3_tail", 3.14, 45.0, "tail entry angle (~180 deg)"),
        AngleTemplate("C1_tail", "C3_tail", "C3_tail", 2.09, 45.0, "double-bond kink (~120 deg)"),
        AngleTemplate("C3_tail", "C3_tail", "C1_tail", 2.09, 45.0, "unsaturated segment angle (~120 deg)"),
        AngleTemplate("C1_tail", "C1_tail", "C1_tail", 3.14, 45.0, "saturated tail chain angle (~180 deg)"),
    ),
)

# -- 7. Polymer PEG --------------------------------------------------------

POLYMER_PEG_TEMPLATE = MaterialTemplate(
    name="polymer_peg",
    description="Poly(ethylene glycol) coarse-grained chain.",
    classification="[adapted]",
    bonds=(
        BondTemplate("PEG", "PEG", 0.33, 7000.0, "PEG repeat-unit bond"),
    ),
    angles=(
        AngleTemplate("PEG", "PEG", "PEG", 2.44, 50.0, "PEG chain angle (~140 deg)"),
    ),
)

# -- 8. Gold surface --------------------------------------------------------

GOLD_SURFACE_TEMPLATE = MaterialTemplate(
    name="gold_surface",
    description="CG gold surface (FCC lattice, very stiff bonds).",
    classification="[adapted]",
    bonds=(
        BondTemplate("Au", "Au", 0.288, 50000.0, "Au-Au lattice bond"),
    ),
    special_parameters=FrozenMetadata({
        "lattice_type": "fcc",
        "lattice_constant": 0.408,
    }),
)

# -- 9. Silica surface ------------------------------------------------------

SILICA_SURFACE_TEMPLATE = MaterialTemplate(
    name="silica_surface",
    description="Amorphous silica surface with hydroxyl groups.",
    classification="[adapted]",
    bonds=(
        BondTemplate("SiO2", "SiO2", 0.31, 40000.0, "SiO2 network bond"),
    ),
    angles=(
        AngleTemplate("SiO2", "SiO2", "SiO2", 2.44, 500.0, "SiO2 network angle (~140 deg)"),
    ),
    special_parameters=FrozenMetadata({
        "lattice_type": "amorphous",
        "hydroxyl_density": 4.6,
    }),
)

# -- 10. Graphene -----------------------------------------------------------

GRAPHENE_TEMPLATE = MaterialTemplate(
    name="graphene",
    description="CG graphene sheet (hexagonal lattice, very stiff).",
    classification="[established]",
    bonds=(
        BondTemplate("C", "C", 0.142, 100000.0, "C-C graphene bond"),
    ),
    angles=(
        AngleTemplate("C", "C", "C", 2.094, 1000.0, "C-C-C graphene angle (~120 deg)"),
    ),
    special_parameters=FrozenMetadata({
        "lattice_type": "hexagonal",
        "layer_spacing": 0.335,
    }),
)

# -- 11. Self-assembled monolayer (SAM) -------------------------------------

SAM_TEMPLATE = MaterialTemplate(
    name="sam",
    description="Self-assembled monolayer (thiol anchor — alkyl chain — functional head).",
    classification="[adapted]",
    bonds=(
        BondTemplate("SAM_anchor", "SAM_chain", 0.47, 5000.0, "anchor-chain bond"),
        BondTemplate("SAM_chain", "SAM_chain", 0.47, 5000.0, "chain-chain bond"),
        BondTemplate("SAM_chain", "SAM_head", 0.47, 5000.0, "chain-head bond"),
    ),
)

# -- 12. Generic ligand -----------------------------------------------------

LIGAND_GENERIC_TEMPLATE = MaterialTemplate(
    name="ligand_generic",
    description="Generic small-molecule ligand placeholder (to be refined per system).",
    classification="[hybrid]",
    bonds=(
        BondTemplate("LIG", "LIG", 0.30, 8000.0, "generic ligand internal bond"),
    ),
    angles=(
        AngleTemplate("LIG", "LIG", "LIG", 2.09, 50.0, "generic ligand angle (~120 deg)"),
    ),
)


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

TEMPLATE_REGISTRY: dict[str, MaterialTemplate] = {
    "protein": PROTEIN_TEMPLATE,
    "peptide": PEPTIDE_TEMPLATE,
    "water": WATER_TEMPLATE,
    "ion": ION_TEMPLATE,
    "lipid_dppc": LIPID_DPPC_TEMPLATE,
    "lipid_dopc": LIPID_DOPC_TEMPLATE,
    "polymer_peg": POLYMER_PEG_TEMPLATE,
    "gold_surface": GOLD_SURFACE_TEMPLATE,
    "silica_surface": SILICA_SURFACE_TEMPLATE,
    "graphene": GRAPHENE_TEMPLATE,
    "sam": SAM_TEMPLATE,
    "ligand_generic": LIGAND_GENERIC_TEMPLATE,
}


# ---------------------------------------------------------------------------
# System-type configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SystemTypeConfig:
    """Describes which material templates are combined for a simulation system type."""

    name: str
    description: str
    material_templates: tuple[str, ...]
    cross_interaction_notes: str
    recommended_cutoff: float = 1.2  # nm
    recommended_time_step: float = 0.020  # ps  (MARTINI standard: 20 fs)
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        for tname in self.material_templates:
            if tname not in TEMPLATE_REGISTRY:
                raise ContractValidationError(
                    f"SystemTypeConfig '{self.name}' references unknown template "
                    f"'{tname}'. Available: {sorted(TEMPLATE_REGISTRY)}."
                )


SYSTEM_CONFIGS: dict[str, SystemTypeConfig] = {
    # -- 1. Protein in solution ---------------------------------------------
    "protein": SystemTypeConfig(
        name="protein",
        description="Solvated protein system with counterions.",
        material_templates=("protein", "water", "ion"),
        cross_interaction_notes="Standard MARTINI mixing rules for protein-water LJ.",
    ),

    # -- 2. Peptide in solution ---------------------------------------------
    "peptide": SystemTypeConfig(
        name="peptide",
        description="Short peptide in explicit CG solvent.",
        material_templates=("peptide", "water", "ion"),
        cross_interaction_notes="Same as protein; softer backbone angles for short chains.",
    ),

    # -- 3. Membrane --------------------------------------------------------
    "membrane": SystemTypeConfig(
        name="membrane",
        description="Pure lipid bilayer (DPPC) with water and ions.",
        material_templates=("lipid_dppc", "water", "ion"),
        cross_interaction_notes="Lipid-water interactions use MARTINI level-IV parameters.",
    ),

    # -- 4. Protein–membrane ------------------------------------------------
    "protein_membrane": SystemTypeConfig(
        name="protein_membrane",
        description="Transmembrane or peripheral protein embedded in a lipid bilayer.",
        material_templates=("protein", "lipid_dppc", "water", "ion"),
        cross_interaction_notes=(
            "Protein-lipid cross interactions from MARTINI; "
            "hydrophobic mismatch handled via elastic network."
        ),
    ),

    # -- 5. Protein–ligand --------------------------------------------------
    "protein_ligand": SystemTypeConfig(
        name="protein_ligand",
        description="Protein with a bound or proximal small-molecule ligand.",
        material_templates=("protein", "ligand_generic", "water", "ion"),
        cross_interaction_notes="Ligand parameters are placeholders; refine per system.",
    ),

    # -- 6. Protein on gold surface -----------------------------------------
    "protein_surface_gold": SystemTypeConfig(
        name="protein_surface_gold",
        description="Protein adsorbed on a gold surface.",
        material_templates=("protein", "gold_surface", "water", "ion"),
        cross_interaction_notes=(
            "Protein-Au interactions use adapted LJ well depths "
            "from polarizable gold models."
        ),
    ),

    # -- 7. Protein on silica surface ---------------------------------------
    "protein_surface_silica": SystemTypeConfig(
        name="protein_surface_silica",
        description="Protein adsorbed on a silica surface.",
        material_templates=("protein", "silica_surface", "water", "ion"),
        cross_interaction_notes=(
            "Silica hydroxyl-protein hydrogen bonding approximated "
            "via shifted LJ potentials."
        ),
    ),

    # -- 8. Peptide on surface ----------------------------------------------
    "peptide_surface": SystemTypeConfig(
        name="peptide_surface",
        description="Short peptide interacting with a gold surface.",
        material_templates=("peptide", "gold_surface", "water", "ion"),
        cross_interaction_notes="Same gold interaction rules as protein_surface_gold.",
    ),

    # -- 9. Self-assembly ---------------------------------------------------
    "self_assembly": SystemTypeConfig(
        name="self_assembly",
        description="Multi-lipid self-assembly (vesicle/micelle formation).",
        material_templates=("lipid_dppc", "lipid_dopc", "water", "ion"),
        cross_interaction_notes="concentration-dependent",
    ),

    # -- 10. Polymer solution -----------------------------------------------
    "polymer_solution": SystemTypeConfig(
        name="polymer_solution",
        description="PEG polymer chains in CG water.",
        material_templates=("polymer_peg", "water", "ion"),
        cross_interaction_notes="PEG-water interactions calibrated to match Rg data.",
    ),

    # -- 11. Conformational transition --------------------------------------
    "conformational_transition": SystemTypeConfig(
        name="conformational_transition",
        description="Protein conformational-change simulation (dual-basin Go model).",
        material_templates=("protein", "water", "ion"),
        cross_interaction_notes="Go-model contacts",
    ),

    # -- 12. Protein–peptide ------------------------------------------------
    "protein_peptide": SystemTypeConfig(
        name="protein_peptide",
        description="Protein interacting with short peptide fragments.",
        material_templates=("protein", "peptide", "water", "ion"),
        cross_interaction_notes="Standard MARTINI cross-type mixing rules.",
    ),

    # -- 13. Surface-functionalized -----------------------------------------
    "surface_functionalized": SystemTypeConfig(
        name="surface_functionalized",
        description="Gold surface functionalized with a self-assembled monolayer.",
        material_templates=("sam", "gold_surface", "water", "ion"),
        cross_interaction_notes="SAM anchor-Au bonding via position restraints.",
    ),

    # -- 14. Mixed polymer–protein ------------------------------------------
    "mixed_polymer_protein": SystemTypeConfig(
        name="mixed_polymer_protein",
        description="PEG-protein conjugate or PEG crowding around a protein.",
        material_templates=("protein", "polymer_peg", "water", "ion"),
        cross_interaction_notes="PEG-protein cross interactions use standard mixing rules.",
    ),
}
