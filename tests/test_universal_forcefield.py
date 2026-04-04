"""Comprehensive tests for the universal forcefield framework.

Covers universal bead types, interaction matrix, mixing rules, material
templates, the universal forcefield assembler, Go-model contact generation,
self-assembly modulation, and adsorption potential generation across all 14
supported NeuroCGMD system types.
"""

from __future__ import annotations

import unittest
from math import pi, sqrt

from forcefields.universal_types import (
    MaterialClass, ChemicalCharacter, InteractionStrength, MixingRule,
    UniversalBeadType, STANDARD_BEADS, INTERACTION_MATRIX,
    EPSILON_FOR_LEVEL, compute_pair_parameters, AMINO_ACID_BEAD_MAP,
)
from forcefields.material_templates import (
    MaterialTemplate, BondTemplate, AngleTemplate, DihedralTemplate,
    SystemTypeConfig, TEMPLATE_REGISTRY, SYSTEM_CONFIGS,
)
from forcefields.universal_forcefield import (
    UniversalForceField, AssembledForceField, SystemComposition,
    GoModelContactGenerator, SelfAssemblyModulator,
    AdsorptionPotentialGenerator, quick_build,
)
from forcefields.base_forcefield import BondParameter, NonbondedParameter
from forcefields.bonded_potentials import AngleParameter, DihedralParameter
from core.types import FrozenMetadata
from core.exceptions import ContractValidationError


# ============================================================================
# 1. TestUniversalBeadTypes
# ============================================================================


class TestUniversalBeadTypes(unittest.TestCase):
    """Validate the standard bead library and bead type properties."""

    def test_standard_beads_exist(self) -> None:
        """STANDARD_BEADS must contain entries for all core bead names."""
        required = ("BB", "W", "Qp_Na", "Qn_Cl", "C1_tail", "Au_surf", "PEG")
        for name in required:
            self.assertIn(name, STANDARD_BEADS, f"Missing standard bead: {name}")

    def test_standard_beads_has_reasonable_count(self) -> None:
        """Library should contain at least 20 bead types."""
        self.assertGreaterEqual(len(STANDARD_BEADS), 20)

    def test_bead_material_class_protein(self) -> None:
        """BB backbone bead must belong to the PROTEIN material class."""
        self.assertEqual(STANDARD_BEADS["BB"].material_class, MaterialClass.PROTEIN)

    def test_bead_material_class_water(self) -> None:
        """W water bead must belong to the WATER material class."""
        self.assertEqual(STANDARD_BEADS["W"].material_class, MaterialClass.WATER)

    def test_bead_material_class_inorganic_surface(self) -> None:
        """Au_surf bead must belong to INORGANIC_SURFACE."""
        self.assertEqual(
            STANDARD_BEADS["Au_surf"].material_class,
            MaterialClass.INORGANIC_SURFACE,
        )

    def test_bead_material_class_lipid(self) -> None:
        """C1_tail bead must belong to LIPID."""
        self.assertEqual(STANDARD_BEADS["C1_tail"].material_class, MaterialClass.LIPID)

    def test_bead_material_class_polymer(self) -> None:
        """PEG bead must belong to POLYMER."""
        self.assertEqual(STANDARD_BEADS["PEG"].material_class, MaterialClass.POLYMER)

    def test_bead_material_class_ion(self) -> None:
        """Ion beads must belong to ION."""
        self.assertEqual(STANDARD_BEADS["Qp_Na"].material_class, MaterialClass.ION)
        self.assertEqual(STANDARD_BEADS["Qn_Cl"].material_class, MaterialClass.ION)

    def test_bead_charges_positive_ion(self) -> None:
        """Qp_Na (sodium) must carry charge +1."""
        self.assertEqual(STANDARD_BEADS["Qp_Na"].charge, 1.0)

    def test_bead_charges_negative_ion(self) -> None:
        """Qn_Cl (chloride) must carry charge -1."""
        self.assertEqual(STANDARD_BEADS["Qn_Cl"].charge, -1.0)

    def test_bead_charges_neutral_water(self) -> None:
        """W (water) must be charge-neutral."""
        self.assertEqual(STANDARD_BEADS["W"].charge, 0.0)

    def test_bead_charges_neutral_backbone(self) -> None:
        """BB (backbone) must be charge-neutral."""
        self.assertEqual(STANDARD_BEADS["BB"].charge, 0.0)

    def test_bead_sigma_positive(self) -> None:
        """Every bead must have positive sigma."""
        for name, bead in STANDARD_BEADS.items():
            self.assertGreater(bead.sigma, 0.0, f"{name} sigma is not positive")

    def test_bead_mass_positive(self) -> None:
        """Every bead must have positive mass."""
        for name, bead in STANDARD_BEADS.items():
            self.assertGreater(bead.mass, 0.0, f"{name} mass is not positive")

    def test_bead_name_matches_key(self) -> None:
        """Each STANDARD_BEADS key must match the bead's .name attribute."""
        for key, bead in STANDARD_BEADS.items():
            self.assertEqual(key, bead.name)

    def test_amino_acid_map_complete(self) -> None:
        """All 20 standard amino acids must be present in AMINO_ACID_BEAD_MAP."""
        standard_20 = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
            "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
            "THR", "TRP", "TYR", "VAL",
        }
        for aa in standard_20:
            self.assertIn(aa, AMINO_ACID_BEAD_MAP, f"{aa} missing from map")

    def test_amino_acid_map_all_start_with_bb(self) -> None:
        """Every amino acid mapping should start with a backbone (BB) bead."""
        for aa, beads in AMINO_ACID_BEAD_MAP.items():
            self.assertTrue(
                len(beads) >= 1 and beads[0] == "BB",
                f"{aa} mapping does not start with BB",
            )

    def test_amino_acid_map_beads_in_standard(self) -> None:
        """All bead names referenced by amino acid map must exist in STANDARD_BEADS."""
        for aa, beads in AMINO_ACID_BEAD_MAP.items():
            for bead_name in beads:
                self.assertIn(
                    bead_name, STANDARD_BEADS,
                    f"Amino acid {aa} references unknown bead {bead_name}",
                )

    def test_bead_type_construction_validation(self) -> None:
        """UniversalBeadType must reject invalid sigma / mass / subtype."""
        with self.assertRaises(ContractValidationError):
            UniversalBeadType(
                name="bad", material_class=MaterialClass.PROTEIN,
                chemical_character=ChemicalCharacter.POLAR, sigma=-0.1,
            )
        with self.assertRaises(ContractValidationError):
            UniversalBeadType(
                name="bad", material_class=MaterialClass.PROTEIN,
                chemical_character=ChemicalCharacter.POLAR, mass=-1.0,
            )
        with self.assertRaises(ContractValidationError):
            UniversalBeadType(
                name="bad", material_class=MaterialClass.PROTEIN,
                chemical_character=ChemicalCharacter.POLAR, subtype=9,
            )

    def test_bead_type_empty_name_rejected(self) -> None:
        """UniversalBeadType must reject an empty name."""
        with self.assertRaises(ContractValidationError):
            UniversalBeadType(
                name="", material_class=MaterialClass.PROTEIN,
                chemical_character=ChemicalCharacter.POLAR,
            )


# ============================================================================
# 2. TestInteractionMatrix
# ============================================================================


class TestInteractionMatrix(unittest.TestCase):
    """Validate the symmetric interaction matrix and epsilon ordering."""

    def test_matrix_is_symmetric(self) -> None:
        """INTERACTION_MATRIX[(A,B)] must equal INTERACTION_MATRIX[(B,A)]."""
        for (a, b), level in INTERACTION_MATRIX.items():
            reverse = INTERACTION_MATRIX.get((b, a))
            self.assertIsNotNone(reverse, f"Reverse entry ({b},{a}) missing")
            self.assertEqual(
                level, reverse,
                f"Asymmetric: ({a},{b})={level} vs ({b},{a})={reverse}",
            )

    def test_charged_pairs_supra_attractive(self) -> None:
        """Qp + Qn (opposite charges) must map to level O (supra-attractive)."""
        level = INTERACTION_MATRIX[(
            ChemicalCharacter.CHARGED_POSITIVE,
            ChemicalCharacter.CHARGED_NEGATIVE,
        )]
        self.assertEqual(level, InteractionStrength.O)

    def test_apolar_apolar_semi_attractive(self) -> None:
        """C + C (apolar-apolar) must map to level II (semi-attractive)."""
        level = INTERACTION_MATRIX[(
            ChemicalCharacter.APOLAR,
            ChemicalCharacter.APOLAR,
        )]
        self.assertEqual(level, InteractionStrength.II)

    def test_polar_apolar_semi_repulsive(self) -> None:
        """P + C (polar-apolar) must map to level IV (semi-repulsive)."""
        level = INTERACTION_MATRIX[(
            ChemicalCharacter.POLAR,
            ChemicalCharacter.APOLAR,
        )]
        self.assertEqual(level, InteractionStrength.IV)

    def test_polar_polar_attractive(self) -> None:
        """P + P (polar-polar) must be level I or II (attractive)."""
        level = INTERACTION_MATRIX[(
            ChemicalCharacter.POLAR,
            ChemicalCharacter.POLAR,
        )]
        self.assertIn(level, (InteractionStrength.I, InteractionStrength.II))

    def test_intermediate_intermediate(self) -> None:
        """N + N (intermediate-intermediate) must be level III."""
        level = INTERACTION_MATRIX[(
            ChemicalCharacter.INTERMEDIATE,
            ChemicalCharacter.INTERMEDIATE,
        )]
        self.assertEqual(level, InteractionStrength.III)

    def test_epsilon_ordering(self) -> None:
        """Epsilon must strictly decrease from level O through VI."""
        ordered_levels = [
            InteractionStrength.O,
            InteractionStrength.I,
            InteractionStrength.II,
            InteractionStrength.III,
            InteractionStrength.IV,
            InteractionStrength.V,
            InteractionStrength.VI,
        ]
        for i in range(len(ordered_levels) - 1):
            stronger = EPSILON_FOR_LEVEL[ordered_levels[i]]
            weaker = EPSILON_FOR_LEVEL[ordered_levels[i + 1]]
            self.assertGreater(
                stronger, weaker,
                f"Epsilon for {ordered_levels[i]} ({stronger}) must exceed "
                f"epsilon for {ordered_levels[i + 1]} ({weaker})",
            )

    def test_epsilon_values_positive(self) -> None:
        """All epsilon values must be positive."""
        for level, eps in EPSILON_FOR_LEVEL.items():
            self.assertGreater(eps, 0.0, f"Epsilon for level {level} not positive")

    def test_epsilon_covers_all_levels(self) -> None:
        """EPSILON_FOR_LEVEL must contain entries for at least levels O through VI."""
        required = [
            InteractionStrength.O, InteractionStrength.I,
            InteractionStrength.II, InteractionStrength.III,
            InteractionStrength.IV, InteractionStrength.V,
            InteractionStrength.VI,
        ]
        for lev in required:
            self.assertIn(lev, EPSILON_FOR_LEVEL)

    def test_all_character_self_pairs_defined(self) -> None:
        """The diagonal (X, X) must be defined for all ChemicalCharacter values."""
        for char in ChemicalCharacter:
            self.assertIn(
                (char, char), INTERACTION_MATRIX,
                f"Self-pair ({char}, {char}) missing from matrix",
            )


# ============================================================================
# 3. TestMixingRules
# ============================================================================


class TestMixingRules(unittest.TestCase):
    """Validate compute_pair_parameters under all mixing rules."""

    def test_compute_pair_martini_known_characters(self) -> None:
        """Two beads with known characters must produce the expected epsilon."""
        bb = STANDARD_BEADS["BB"]   # SPECIAL character
        w = STANDARD_BEADS["W"]     # POLAR character
        sigma, epsilon = compute_pair_parameters(bb, w)
        # Sigma should be arithmetic mean
        expected_sigma = (bb.sigma + w.sigma) / 2.0
        self.assertAlmostEqual(sigma, expected_sigma, places=8)
        # Epsilon should come from the INTERACTION_MATRIX
        expected_level = INTERACTION_MATRIX[(bb.chemical_character, w.chemical_character)]
        expected_eps = EPSILON_FOR_LEVEL[expected_level]
        self.assertAlmostEqual(epsilon, expected_eps, places=8)

    def test_compute_pair_lorentz_berthelot_sigma(self) -> None:
        """Lorentz-Berthelot sigma must be the arithmetic mean."""
        bb = STANDARD_BEADS["BB"]
        au = STANDARD_BEADS["Au_surf"]
        sigma, _eps = compute_pair_parameters(
            bb, au, mixing_rule=MixingRule.LORENTZ_BERTHELOT,
        )
        expected = (bb.sigma + au.sigma) / 2.0
        self.assertAlmostEqual(sigma, expected, places=8)

    def test_compute_pair_lorentz_berthelot_epsilon(self) -> None:
        """Lorentz-Berthelot epsilon must be the geometric mean of self-epsilons."""
        bb = STANDARD_BEADS["BB"]
        w = STANDARD_BEADS["W"]
        _sigma, epsilon = compute_pair_parameters(
            bb, w, mixing_rule=MixingRule.LORENTZ_BERTHELOT,
        )
        # Per-bead epsilon is looked up from self-interaction matrix entry
        level_bb = INTERACTION_MATRIX[(bb.chemical_character, bb.chemical_character)]
        level_w = INTERACTION_MATRIX[(w.chemical_character, w.chemical_character)]
        expected = sqrt(EPSILON_FOR_LEVEL[level_bb] * EPSILON_FOR_LEVEL[level_w])
        self.assertAlmostEqual(epsilon, expected, places=8)

    def test_compute_pair_geometric_sigma(self) -> None:
        """Geometric mixing sigma must be the geometric mean."""
        bb = STANDARD_BEADS["BB"]
        peg = STANDARD_BEADS["PEG"]
        sigma, _eps = compute_pair_parameters(
            bb, peg, mixing_rule=MixingRule.GEOMETRIC,
        )
        expected = sqrt(bb.sigma * peg.sigma)
        self.assertAlmostEqual(sigma, expected, places=8)

    def test_compute_pair_geometric_epsilon(self) -> None:
        """Geometric mixing epsilon must be the geometric mean of self-epsilons."""
        bb = STANDARD_BEADS["BB"]
        peg = STANDARD_BEADS["PEG"]
        _sigma, epsilon = compute_pair_parameters(
            bb, peg, mixing_rule=MixingRule.GEOMETRIC,
        )
        level_bb = INTERACTION_MATRIX[(bb.chemical_character, bb.chemical_character)]
        level_peg = INTERACTION_MATRIX[(peg.chemical_character, peg.chemical_character)]
        expected = sqrt(EPSILON_FOR_LEVEL[level_bb] * EPSILON_FOR_LEVEL[level_peg])
        self.assertAlmostEqual(epsilon, expected, places=8)

    def test_cross_material_protein_water(self) -> None:
        """protein BB + water W must return valid positive (sigma, epsilon)."""
        bb = STANDARD_BEADS["BB"]
        w = STANDARD_BEADS["W"]
        sigma, epsilon = compute_pair_parameters(bb, w)
        self.assertGreater(sigma, 0.0)
        self.assertGreater(epsilon, 0.0)

    def test_cross_material_protein_surface(self) -> None:
        """protein BB + Au_surf must return valid positive (sigma, epsilon)."""
        bb = STANDARD_BEADS["BB"]
        au = STANDARD_BEADS["Au_surf"]
        sigma, epsilon = compute_pair_parameters(bb, au)
        self.assertGreater(sigma, 0.0)
        self.assertGreater(epsilon, 0.0)

    def test_cross_material_ion_water(self) -> None:
        """Ion + water must return valid parameters."""
        na = STANDARD_BEADS["Qp_Na"]
        w = STANDARD_BEADS["W"]
        sigma, epsilon = compute_pair_parameters(na, w)
        self.assertGreater(sigma, 0.0)
        self.assertGreater(epsilon, 0.0)

    def test_self_interaction_consistent(self) -> None:
        """Self-interaction parameters should match the matrix diagonal."""
        for name in ("BB", "W", "C1_tail", "Au_surf"):
            bead = STANDARD_BEADS[name]
            sigma, epsilon = compute_pair_parameters(bead, bead)
            self.assertAlmostEqual(sigma, bead.sigma, places=8)
            level = INTERACTION_MATRIX[(bead.chemical_character, bead.chemical_character)]
            self.assertAlmostEqual(epsilon, EPSILON_FOR_LEVEL[level], places=8)

    def test_martini_mixing_is_default(self) -> None:
        """Default mixing rule must be MARTINI_MATRIX."""
        bb = STANDARD_BEADS["BB"]
        w = STANDARD_BEADS["W"]
        # Without explicit rule
        s1, e1 = compute_pair_parameters(bb, w)
        # With explicit MARTINI_MATRIX
        s2, e2 = compute_pair_parameters(bb, w, mixing_rule=MixingRule.MARTINI_MATRIX)
        self.assertAlmostEqual(s1, s2)
        self.assertAlmostEqual(e1, e2)


# ============================================================================
# 4. TestMaterialTemplates
# ============================================================================


class TestMaterialTemplates(unittest.TestCase):
    """Validate material templates and system type configurations."""

    def test_template_registry_complete(self) -> None:
        """TEMPLATE_REGISTRY must contain all expected material entries."""
        expected = (
            "protein", "peptide", "water", "ion", "lipid_dppc",
            "lipid_dopc", "polymer_peg", "gold_surface",
            "silica_surface", "graphene", "sam", "ligand_generic",
        )
        for name in expected:
            self.assertIn(name, TEMPLATE_REGISTRY, f"Missing template: {name}")

    def test_template_instances_are_material_template(self) -> None:
        """Every entry in TEMPLATE_REGISTRY must be a MaterialTemplate."""
        for name, tpl in TEMPLATE_REGISTRY.items():
            self.assertIsInstance(tpl, MaterialTemplate, f"{name} is not a MaterialTemplate")

    def test_protein_template_has_backbone_bond(self) -> None:
        """Protein template must include a BB-BB backbone bond near 0.35 nm."""
        protein = TEMPLATE_REGISTRY["protein"]
        bb_bonds = [
            b for b in protein.bonds
            if b.bead_type_a == "BB" and b.bead_type_b == "BB"
        ]
        self.assertGreater(len(bb_bonds), 0, "No BB-BB bond in protein template")
        self.assertAlmostEqual(bb_bonds[0].equilibrium_distance, 0.35, places=2)

    def test_protein_template_has_backbone_angle(self) -> None:
        """Protein template must have at least one backbone angle."""
        protein = TEMPLATE_REGISTRY["protein"]
        bb_angles = [
            a for a in protein.angles
            if a.bead_type_a == "BB" and a.bead_type_b == "BB" and a.bead_type_c == "BB"
        ]
        self.assertGreater(len(bb_angles), 0, "No BB-BB-BB angle in protein template")

    def test_protein_template_has_dihedral(self) -> None:
        """Protein template must have at least one dihedral."""
        protein = TEMPLATE_REGISTRY["protein"]
        self.assertGreater(len(protein.dihedrals), 0, "No dihedrals in protein template")

    def test_lipid_dppc_template_has_angles(self) -> None:
        """Lipid DPPC template must have at least one angle definition."""
        lipid = TEMPLATE_REGISTRY["lipid_dppc"]
        self.assertGreater(len(lipid.angles), 0, "No angles in lipid_dppc template")

    def test_lipid_dppc_template_has_bonds(self) -> None:
        """Lipid DPPC template must have bond definitions."""
        lipid = TEMPLATE_REGISTRY["lipid_dppc"]
        self.assertGreater(len(lipid.bonds), 0, "No bonds in lipid_dppc template")

    def test_water_template_has_no_bonds(self) -> None:
        """Water (single-site CG) should have no bonded interactions."""
        water = TEMPLATE_REGISTRY["water"]
        self.assertEqual(len(water.bonds), 0)

    def test_ion_template_has_no_bonds(self) -> None:
        """Monatomic ion template should have no bonded interactions."""
        ion = TEMPLATE_REGISTRY["ion"]
        self.assertEqual(len(ion.bonds), 0)

    def test_system_configs_cover_all_14_types(self) -> None:
        """SYSTEM_CONFIGS must contain all 14 documented system types."""
        expected_types = {
            "protein", "peptide", "membrane", "protein_membrane",
            "protein_ligand", "protein_surface_gold", "protein_surface_silica",
            "peptide_surface", "self_assembly", "polymer_solution",
            "conformational_transition", "protein_peptide",
            "surface_functionalized", "mixed_polymer_protein",
        }
        for stype in expected_types:
            self.assertIn(stype, SYSTEM_CONFIGS, f"Missing system config: {stype}")
        self.assertEqual(len(expected_types), 14)

    def test_system_config_references_valid_templates(self) -> None:
        """Every template name in every SystemTypeConfig must exist in TEMPLATE_REGISTRY."""
        for name, config in SYSTEM_CONFIGS.items():
            for tpl_name in config.material_templates:
                self.assertIn(
                    tpl_name, TEMPLATE_REGISTRY,
                    f"Config '{name}' references unknown template '{tpl_name}'",
                )

    def test_system_configs_are_proper_type(self) -> None:
        """Every SYSTEM_CONFIGS entry must be a SystemTypeConfig instance."""
        for name, config in SYSTEM_CONFIGS.items():
            self.assertIsInstance(config, SystemTypeConfig, f"{name}")

    def test_bond_template_distance_positive(self) -> None:
        """All bond template equilibrium distances must be positive."""
        for name, tpl in TEMPLATE_REGISTRY.items():
            for bt in tpl.bonds:
                self.assertGreater(
                    bt.equilibrium_distance, 0.0,
                    f"{name}: {bt.bead_type_a}-{bt.bead_type_b} bond distance <= 0",
                )

    def test_angle_template_in_valid_range(self) -> None:
        """All angle templates must have equilibrium angles in (0, pi]."""
        for name, tpl in TEMPLATE_REGISTRY.items():
            for at in tpl.angles:
                self.assertGreater(at.equilibrium_angle, 0.0)
                self.assertLessEqual(at.equilibrium_angle, pi + 0.01)

    def test_bond_template_construction(self) -> None:
        """BondTemplate can be constructed with valid parameters."""
        bt = BondTemplate("X", "Y", 0.35, 1000.0, "test bond")
        self.assertEqual(bt.bead_type_a, "X")
        self.assertEqual(bt.bead_type_b, "Y")
        self.assertAlmostEqual(bt.equilibrium_distance, 0.35)
        self.assertAlmostEqual(bt.force_constant, 1000.0)

    def test_angle_template_construction(self) -> None:
        """AngleTemplate can be constructed with valid parameters."""
        at = AngleTemplate("A", "B", "C", 2.09, 25.0, "test angle")
        self.assertEqual(at.bead_type_b, "B")
        self.assertAlmostEqual(at.equilibrium_angle, 2.09)

    def test_dihedral_template_construction(self) -> None:
        """DihedralTemplate can be constructed with valid parameters."""
        dt = DihedralTemplate(("A", "B", "C", "D"), 1.0, 1, pi, "test dihedral")
        self.assertEqual(dt.bead_types, ("A", "B", "C", "D"))
        self.assertEqual(dt.multiplicity, 1)


# ============================================================================
# 5. TestUniversalForceField
# ============================================================================


class TestUniversalForceField(unittest.TestCase):
    """Validate the universal forcefield assembler across all system types."""

    def test_build_protein_system(self) -> None:
        """quick_build for a protein system must return an AssembledForceField."""
        result = quick_build("protein", 6, tuple(["BB"] * 6))
        self.assertIsInstance(result, AssembledForceField)
        self.assertEqual(result.system_type, "protein")

    def test_build_protein_system_has_bonds(self) -> None:
        """Protein forcefield must include bond parameters from the protein template."""
        result = quick_build("protein", 6, tuple(["BB"] * 6))
        self.assertGreater(len(result.base_forcefield.bond_parameters), 0)

    def test_build_protein_system_has_nonbonded(self) -> None:
        """Protein forcefield must have nonbonded parameters."""
        result = quick_build("protein", 6, tuple(["BB"] * 6))
        self.assertGreater(len(result.base_forcefield.nonbonded_parameters), 0)

    def test_build_protein_system_has_angles(self) -> None:
        """Protein forcefield must have angle parameters."""
        result = quick_build("protein", 4, tuple(["BB"] * 4))
        self.assertGreater(len(result.angle_parameters), 0)

    def test_build_protein_system_has_dihedrals(self) -> None:
        """Protein forcefield must have dihedral parameters."""
        result = quick_build("protein", 6, tuple(["BB"] * 6))
        self.assertGreater(len(result.dihedral_parameters), 0)

    def test_build_protein_membrane(self) -> None:
        """Protein-membrane system must include both protein and lipid bond types."""
        beads = ("BB", "BB", "BB", "C1_tail", "C1_tail", "W", "Qp_Na", "Qn_Cl")
        result = quick_build("protein_membrane", len(beads), beads)
        self.assertIsInstance(result, AssembledForceField)
        # Must have bond parameters from both protein and lipid templates
        bond_types = {
            (b.bead_type_a, b.bead_type_b)
            for b in result.base_forcefield.bond_parameters
        }
        # At least some protein bonds and lipid bonds should be present
        self.assertGreater(len(bond_types), 0)

    def test_build_protein_surface(self) -> None:
        """Protein-surface system must have cross-type nonbonded parameters."""
        beads = ("BB", "BB", "Au_surf", "Au_surf", "W", "Qp_Na")
        result = quick_build("protein_surface_gold", len(beads), beads)
        self.assertIsInstance(result, AssembledForceField)
        # Nonbonded must cover all unique bead-type pairs
        nb_pairs = {
            tuple(sorted((p.bead_type_a, p.bead_type_b)))
            for p in result.base_forcefield.nonbonded_parameters
        }
        # BB-Au_surf cross interaction must exist
        self.assertIn(("Au_surf", "BB"), nb_pairs)

    def test_all_system_types_build(self) -> None:
        """Every one of the 14 system types must build without errors."""
        for sys_type in SYSTEM_CONFIGS:
            with self.subTest(system_type=sys_type):
                result = quick_build(sys_type, 4)
                self.assertIsInstance(result, AssembledForceField)
                self.assertEqual(result.system_type, sys_type)
                self.assertGreater(
                    len(result.base_forcefield.nonbonded_parameters), 0,
                    f"{sys_type}: no nonbonded parameters generated",
                )

    def test_assembled_has_charges_for_ions(self) -> None:
        """An ion-containing system must have non-zero charges for ion particles."""
        beads = ("BB", "Qp_Na", "Qn_Cl", "W")
        result = quick_build("protein", len(beads), beads)
        self.assertEqual(len(result.charges), 4)
        # Na+ should be positive, Cl- should be negative
        self.assertGreater(result.charges[1], 0.0)
        self.assertLess(result.charges[2], 0.0)

    def test_nonbonded_parameters_cover_all_pairs(self) -> None:
        """For a mixed system, every unique bead-type pair must have nonbonded params."""
        beads = ("BB", "W", "Qp_Na", "Qn_Cl", "C1_tail")
        result = quick_build("protein_membrane", len(beads), beads)
        nb_pairs = {
            tuple(sorted((p.bead_type_a, p.bead_type_b)))
            for p in result.base_forcefield.nonbonded_parameters
        }
        unique_names = sorted(set(beads))
        for i, a in enumerate(unique_names):
            for b in unique_names[i:]:
                pair = tuple(sorted((a, b)))
                self.assertIn(
                    pair, nb_pairs,
                    f"Missing nonbonded parameter for pair {pair}",
                )

    def test_nonbonded_sigma_positive(self) -> None:
        """All generated nonbonded sigma values must be positive."""
        result = quick_build("protein", 4, ("BB", "W", "Qp_Na", "Qn_Cl"))
        for p in result.base_forcefield.nonbonded_parameters:
            self.assertGreater(p.sigma, 0.0)

    def test_nonbonded_epsilon_non_negative(self) -> None:
        """All generated nonbonded epsilon values must be non-negative."""
        result = quick_build("protein", 4, ("BB", "W", "Qp_Na", "Qn_Cl"))
        for p in result.base_forcefield.nonbonded_parameters:
            self.assertGreaterEqual(p.epsilon, 0.0)

    def test_assembled_describe(self) -> None:
        """AssembledForceField.describe() must return a non-empty string."""
        result = quick_build("protein", 4, tuple(["BB"] * 4))
        desc = result.describe()
        self.assertIsInstance(desc, str)
        self.assertIn("protein", desc)

    def test_assembled_particle_charges(self) -> None:
        """AssembledForceField.particle_charges() must return proper charge tuple."""
        result = quick_build("protein", 3, ("BB", "Qp_Na", "Qn_Cl"))
        charges = result.particle_charges()
        self.assertEqual(len(charges), 3)
        self.assertAlmostEqual(charges[0], 0.0)
        self.assertGreater(charges[1], 0.0)
        self.assertLess(charges[2], 0.0)

    def test_direct_build_from_composition(self) -> None:
        """UniversalForceField.build_from_composition must work with explicit composition."""
        comp = SystemComposition(
            system_type="protein",
            bead_names=("BB", "W"),
            particle_bead_assignments=("BB", "BB", "W", "W"),
        )
        uff = UniversalForceField()
        result = uff.build_from_composition(comp)
        self.assertIsInstance(result, AssembledForceField)
        self.assertEqual(result.system_type, "protein")

    def test_build_for_system_type_convenience(self) -> None:
        """UniversalForceField.build_for_system_type must produce valid output."""
        uff = UniversalForceField()
        result = uff.build_for_system_type("membrane", ("C1_tail",) * 4)
        self.assertIsInstance(result, AssembledForceField)

    def test_invalid_system_type_raises(self) -> None:
        """quick_build with an unknown system type must raise ContractValidationError."""
        with self.assertRaises(ContractValidationError):
            quick_build("nonexistent_system", 4)

    def test_validate_method(self) -> None:
        """UniversalForceField.validate() must return no issues for defaults."""
        uff = UniversalForceField()
        issues = uff.validate()
        self.assertEqual(len(issues), 0)

    def test_validate_bad_mixing_rule(self) -> None:
        """UniversalForceField.validate() must flag an unknown mixing rule."""
        uff = UniversalForceField(mixing_rule="invalid")
        issues = uff.validate()
        self.assertGreater(len(issues), 0)

    def test_active_templates_populated(self) -> None:
        """AssembledForceField should record which templates were active."""
        result = quick_build("protein", 4, tuple(["BB"] * 4))
        self.assertGreater(len(result.active_templates), 0)
        # protein system should activate at least the protein template
        self.assertIn("protein", result.active_templates)


# ============================================================================
# 6. TestGoModelContacts
# ============================================================================


class TestGoModelContacts(unittest.TestCase):
    """Validate Go-model native contact generation."""

    def test_generates_native_contacts(self) -> None:
        """Particles within cutoff and |i-j|>3 must produce contacts."""
        # Place 5 particles: 0..3 in a line, particle 4 near particle 0
        positions = (
            (0.0, 0.0, 0.0),
            (0.15, 0.0, 0.0),
            (0.30, 0.0, 0.0),
            (0.45, 0.0, 0.0),
            (0.10, 0.0, 0.0),  # close to particle 0, and |4-0| = 4 > 3
        )
        gen = GoModelContactGenerator(
            native_positions=positions,
            contact_cutoff=0.8,
            contact_strength=2.0,
        )
        contacts = gen.generate_contacts()
        # Pair (0,4): dist = 0.10, |4-0|=4 > 3 -> contact
        self.assertGreater(len(contacts), 0, "Expected at least one contact")
        for c in contacts:
            self.assertIsInstance(c, BondParameter)
            self.assertGreater(c.equilibrium_distance, 0.0)

    def test_respects_sequence_separation(self) -> None:
        """Pairs with |i-j| <= 3 must not produce contacts."""
        # All particles close together but only 4 particles total
        # So max |i-j| = 3 (pairs: 0-1, 0-2, 0-3, 1-2, 1-3, 2-3)
        # None should pass the |i-j| > 3 filter
        positions = (
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.2, 0.0, 0.0),
            (0.3, 0.0, 0.0),
        )
        gen = GoModelContactGenerator(
            native_positions=positions,
            contact_cutoff=0.8,
        )
        contacts = gen.generate_contacts()
        self.assertEqual(len(contacts), 0, "|i-j| <= 3 for all pairs with only 4 particles")

    def test_respects_cutoff_distance(self) -> None:
        """Distant particles must not produce contacts even with |i-j| > 3."""
        positions = (
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.2, 0.0, 0.0),
            (0.3, 0.0, 0.0),
            (10.0, 0.0, 0.0),  # far away; |4-0| > 3 but dist > cutoff
        )
        gen = GoModelContactGenerator(
            native_positions=positions,
            contact_cutoff=0.8,
        )
        contacts = gen.generate_contacts()
        # All pairs with |i-j| > 3: (0,4) at dist=10.0 -> no contact
        self.assertEqual(len(contacts), 0)

    def test_contact_equilibrium_distance_matches_native(self) -> None:
        """The contact's equilibrium distance must equal the native distance."""
        positions = (
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.2, 0.0, 0.0),
            (0.3, 0.0, 0.0),
            (0.5, 0.0, 0.0),  # dist to 0 is 0.5; |4-0|=4
        )
        gen = GoModelContactGenerator(
            native_positions=positions,
            contact_cutoff=0.8,
        )
        contacts = gen.generate_contacts()
        self.assertGreater(len(contacts), 0)
        # The (0,4) contact should have eq_dist = 0.5
        found_04 = [c for c in contacts if "P0" in (c.bead_type_a, c.bead_type_b)
                     and "P4" in (c.bead_type_a, c.bead_type_b)]
        self.assertEqual(len(found_04), 1)
        self.assertAlmostEqual(found_04[0].equilibrium_distance, 0.5, places=6)

    def test_multiple_contacts_generated(self) -> None:
        """Multiple qualifying pairs should each produce a contact."""
        # 6 particles placed close together: pairs (0,4), (0,5), (1,5)
        # all have |i-j| >= 4
        positions = (
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.2, 0.0, 0.0),
            (0.3, 0.0, 0.0),
            (0.05, 0.0, 0.0),
            (0.15, 0.0, 0.0),
        )
        gen = GoModelContactGenerator(
            native_positions=positions,
            contact_cutoff=0.8,
        )
        contacts = gen.generate_contacts()
        self.assertGreaterEqual(len(contacts), 2)


# ============================================================================
# 7. TestSelfAssemblyModulator
# ============================================================================


class TestSelfAssemblyModulator(unittest.TestCase):
    """Validate concentration-dependent epsilon modulation."""

    def _make_test_params(self) -> tuple[NonbondedParameter, ...]:
        """Create a small set of nonbonded parameters for testing."""
        return (
            # Same material class (LIPID-LIPID) -> should be boosted
            NonbondedParameter(
                bead_type_a="C1_tail", bead_type_b="C1_tail",
                sigma=0.47, epsilon=4.5, cutoff=1.2,
            ),
            # Cross material class (LIPID-WATER) -> should NOT be boosted
            NonbondedParameter(
                bead_type_a="C1_tail", bead_type_b="W",
                sigma=0.47, epsilon=3.5, cutoff=1.2,
            ),
        )

    def test_no_boost_below_threshold(self) -> None:
        """At low concentration, parameters must remain unchanged."""
        mod = SelfAssemblyModulator(concentration_threshold=0.5, aggregation_boost=2.0)
        params = self._make_test_params()
        result = mod.modulate_parameters(params, concentration=0.3)
        # Below threshold -> identical parameters returned
        self.assertEqual(len(result), len(params))
        for orig, res in zip(params, result):
            self.assertAlmostEqual(orig.epsilon, res.epsilon)

    def test_boost_at_threshold_boundary(self) -> None:
        """At exactly the threshold, parameters must remain unchanged."""
        mod = SelfAssemblyModulator(concentration_threshold=0.5, aggregation_boost=2.0)
        params = self._make_test_params()
        result = mod.modulate_parameters(params, concentration=0.5)
        # At threshold (<=) -> unchanged
        for orig, res in zip(params, result):
            self.assertAlmostEqual(orig.epsilon, res.epsilon)

    def test_boosts_above_threshold(self) -> None:
        """Above threshold, same-class epsilons must increase."""
        mod = SelfAssemblyModulator(concentration_threshold=0.5, aggregation_boost=2.0)
        params = self._make_test_params()
        result = mod.modulate_parameters(params, concentration=1.0)
        self.assertEqual(len(result), 2)
        # LIPID-LIPID pair: epsilon should be boosted by 2x
        self.assertAlmostEqual(result[0].epsilon, 4.5 * 2.0, places=6)
        # LIPID-WATER pair: should remain unchanged (cross-material)
        self.assertAlmostEqual(result[1].epsilon, 3.5, places=6)

    def test_cross_material_not_boosted(self) -> None:
        """Cross-material pairs must not be boosted regardless of concentration."""
        mod = SelfAssemblyModulator(concentration_threshold=0.1, aggregation_boost=3.0)
        cross_param = NonbondedParameter(
            bead_type_a="BB", bead_type_b="W",
            sigma=0.47, epsilon=4.0, cutoff=1.2,
        )
        result = mod.modulate_parameters((cross_param,), concentration=10.0)
        self.assertAlmostEqual(result[0].epsilon, 4.0)

    def test_boost_factor_applied_correctly(self) -> None:
        """The boost factor must multiply the epsilon exactly."""
        boost = 1.75
        mod = SelfAssemblyModulator(concentration_threshold=0.1, aggregation_boost=boost)
        same_param = NonbondedParameter(
            bead_type_a="BB", bead_type_b="BB",
            sigma=0.47, epsilon=5.0, cutoff=1.2,
        )
        result = mod.modulate_parameters((same_param,), concentration=1.0)
        self.assertAlmostEqual(result[0].epsilon, 5.0 * boost, places=6)


# ============================================================================
# 8. TestAdsorptionPotential
# ============================================================================


class TestAdsorptionPotential(unittest.TestCase):
    """Validate surface-adsorbate cross-interaction generation."""

    def test_generates_cross_interactions(self) -> None:
        """Surface + adsorbate beads must produce NonbondedParameters."""
        gen = AdsorptionPotentialGenerator()
        surface_beads = (STANDARD_BEADS["Au_surf"],)
        adsorbate_beads = (STANDARD_BEADS["BB"],)
        params = gen.generate_surface_interactions(surface_beads, adsorbate_beads)
        self.assertGreater(len(params), 0)
        for p in params:
            self.assertIsInstance(p, NonbondedParameter)
            self.assertGreater(p.sigma, 0.0)
            self.assertGreater(p.epsilon, 0.0)

    def test_polar_surface_polar_adsorbate_stronger(self) -> None:
        """Polar-polar interaction must have higher epsilon than polar-apolar."""
        gen = AdsorptionPotentialGenerator(surface_epsilon=8.0)
        # SiO2 is POLAR surface
        sio2 = STANDARD_BEADS["SiO2_surf"]
        # P5 is POLAR adsorbate, C1 is APOLAR adsorbate
        p5 = STANDARD_BEADS["P5"]
        c1 = STANDARD_BEADS["C1"]

        polar_params = gen.generate_surface_interactions((sio2,), (p5,))
        apolar_params = gen.generate_surface_interactions((sio2,), (c1,))

        self.assertGreater(len(polar_params), 0)
        self.assertGreater(len(apolar_params), 0)
        # Polar-polar: eps_scale = 1.0 -> epsilon = 8.0
        # Polar-apolar: eps_scale = 0.3 -> epsilon = 2.4
        self.assertGreater(polar_params[0].epsilon, apolar_params[0].epsilon)

    def test_apolar_surface_apolar_adsorbate(self) -> None:
        """Apolar-apolar interaction should use 0.8 scaling."""
        gen = AdsorptionPotentialGenerator(surface_epsilon=10.0)
        au = STANDARD_BEADS["Au_surf"]  # APOLAR
        c1 = STANDARD_BEADS["C1"]       # APOLAR
        params = gen.generate_surface_interactions((au,), (c1,))
        self.assertGreater(len(params), 0)
        self.assertAlmostEqual(params[0].epsilon, 10.0 * 0.8, places=6)

    def test_apolar_surface_polar_adsorbate(self) -> None:
        """Apolar surface + polar adsorbate should use 0.4 scaling."""
        gen = AdsorptionPotentialGenerator(surface_epsilon=10.0)
        au = STANDARD_BEADS["Au_surf"]  # APOLAR
        p5 = STANDARD_BEADS["P5"]       # POLAR
        params = gen.generate_surface_interactions((au,), (p5,))
        self.assertGreater(len(params), 0)
        self.assertAlmostEqual(params[0].epsilon, 10.0 * 0.4, places=6)

    def test_multiple_cross_pairs(self) -> None:
        """Multiple surface x adsorbate pairs must all produce parameters."""
        gen = AdsorptionPotentialGenerator()
        surface_beads = (STANDARD_BEADS["Au_surf"], STANDARD_BEADS["SiO2_surf"])
        adsorbate_beads = (STANDARD_BEADS["BB"], STANDARD_BEADS["W"])
        params = gen.generate_surface_interactions(surface_beads, adsorbate_beads)
        # 2 surface * 2 adsorbate = 4 unique pairs
        self.assertEqual(len(params), 4)

    def test_no_duplicate_pairs(self) -> None:
        """Same pair appearing multiple times in input should not yield duplicates."""
        gen = AdsorptionPotentialGenerator()
        au = STANDARD_BEADS["Au_surf"]
        bb = STANDARD_BEADS["BB"]
        # Pass the same pair twice via repeated beads
        params = gen.generate_surface_interactions((au, au), (bb,))
        pair_keys = [tuple(sorted((p.bead_type_a, p.bead_type_b))) for p in params]
        self.assertEqual(len(pair_keys), len(set(pair_keys)), "Duplicate pairs generated")

    def test_cutoff_from_adsorption_range(self) -> None:
        """Generated parameters must use the generator's adsorption_range as cutoff."""
        gen = AdsorptionPotentialGenerator(adsorption_range=1.5)
        params = gen.generate_surface_interactions(
            (STANDARD_BEADS["Au_surf"],), (STANDARD_BEADS["BB"],),
        )
        for p in params:
            self.assertAlmostEqual(p.cutoff, 1.5)


# ============================================================================
# Additional integration tests
# ============================================================================


class TestSystemComposition(unittest.TestCase):
    """Validate the SystemComposition descriptor."""

    def test_valid_composition(self) -> None:
        """A well-formed composition must construct without error."""
        comp = SystemComposition(
            system_type="protein",
            bead_names=("BB", "W"),
            particle_bead_assignments=("BB", "BB", "W"),
        )
        self.assertEqual(comp.system_type, "protein")

    def test_empty_system_type_rejected(self) -> None:
        """Empty system_type must raise ContractValidationError."""
        with self.assertRaises(ContractValidationError):
            SystemComposition(
                system_type="",
                bead_names=("BB",),
                particle_bead_assignments=("BB",),
            )

    def test_empty_bead_names_rejected(self) -> None:
        """Empty bead_names must raise ContractValidationError."""
        with self.assertRaises(ContractValidationError):
            SystemComposition(
                system_type="protein",
                bead_names=(),
                particle_bead_assignments=("BB",),
            )

    def test_unknown_assignment_rejected(self) -> None:
        """Assignments referencing beads not in bead_names must raise."""
        with self.assertRaises(ContractValidationError):
            SystemComposition(
                system_type="protein",
                bead_names=("BB",),
                particle_bead_assignments=("BB", "UNKNOWN"),
            )


class TestMaterialTemplateValidation(unittest.TestCase):
    """Validate MaterialTemplate classification constraint."""

    def test_valid_classifications(self) -> None:
        """All valid classification strings must be accepted."""
        for cls in ("[established]", "[adapted]", "[hybrid]", "[proposed novel]"):
            tpl = MaterialTemplate(name="t", description="d", classification=cls)
            self.assertEqual(tpl.classification, cls)

    def test_invalid_classification_rejected(self) -> None:
        """An invalid classification must raise ContractValidationError."""
        with self.assertRaises(ContractValidationError):
            MaterialTemplate(name="t", description="d", classification="[bad]")


class TestEnumerations(unittest.TestCase):
    """Verify enum membership for MaterialClass, ChemicalCharacter, etc."""

    def test_material_class_members(self) -> None:
        """MaterialClass must include the core material families."""
        required = {"PROTEIN", "WATER", "ION", "LIPID", "POLYMER", "INORGANIC_SURFACE"}
        actual = {m.name for m in MaterialClass}
        self.assertTrue(required.issubset(actual), f"Missing: {required - actual}")

    def test_chemical_character_members(self) -> None:
        """ChemicalCharacter must include Qp, Qn, P, N, C, S."""
        actual_values = {c.value for c in ChemicalCharacter}
        for expected in ("Qp", "Qn", "P", "N", "C", "S"):
            self.assertIn(expected, actual_values, f"Missing character: {expected}")

    def test_interaction_strength_members(self) -> None:
        """InteractionStrength must include levels O through VI."""
        actual_values = {s.value for s in InteractionStrength}
        for expected in ("O", "I", "II", "III", "IV", "V", "VI"):
            self.assertIn(expected, actual_values)

    def test_mixing_rule_members(self) -> None:
        """MixingRule must include the three supported strategies."""
        names = {m.name for m in MixingRule}
        self.assertIn("LORENTZ_BERTHELOT", names)
        self.assertIn("GEOMETRIC", names)
        self.assertIn("MARTINI_MATRIX", names)


if __name__ == "__main__":
    unittest.main()
