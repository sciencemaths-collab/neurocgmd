"""Universal forcefield assembler for NeuroCGMD.

Wires the bead taxonomy (universal_types), interaction matrix, and material
templates into a complete, fully parameterized forcefield for any system type.
This is the capstone module: given a system description it produces an
AssembledForceField ready for simulation, handling all cross-material
interactions through the universal interaction matrix and mixing rules.

Classification: [hybrid] -- combines established MARTINI-style CG
methodology with novel multi-material cross-interaction logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import sqrt, pi

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata
from forcefields.base_forcefield import BaseForceField, BondParameter, NonbondedParameter
from forcefields.bonded_potentials import AngleParameter, DihedralParameter


# ---------------------------------------------------------------------------
# System composition descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SystemComposition:
    """Immutable description of what a simulation system contains.

    This is the input contract for the universal forcefield assembler.
    It declares which bead types are present, how particles map to those
    types, and which system configuration governs template selection.
    """

    system_type: str
    bead_names: tuple[str, ...]
    particle_bead_assignments: tuple[str, ...]
    custom_beads: tuple = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self._validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def _validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.system_type.strip():
            issues.append("SystemComposition system_type must be a non-empty string.")
        if not self.bead_names:
            issues.append("SystemComposition must declare at least one bead name.")
        if not self.particle_bead_assignments:
            issues.append("SystemComposition must have at least one particle assignment.")
        unknown = set(self.particle_bead_assignments) - set(self.bead_names)
        if unknown:
            issues.append(
                f"Particle assignments reference unknown bead names: {sorted(unknown)}"
            )
        return tuple(issues)


# ---------------------------------------------------------------------------
# Assembled forcefield -- the final product
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AssembledForceField:
    """A fully parameterized forcefield ready for simulation.

    Bundles the base forcefield (bonds + nonbonded LJ), angle and dihedral
    parameters, per-particle charges, and metadata about which material
    templates contributed to the assembly.
    """

    base_forcefield: BaseForceField
    angle_parameters: tuple[AngleParameter, ...] = ()
    dihedral_parameters: tuple[DihedralParameter, ...] = ()
    charges: tuple[float, ...] = ()
    system_type: str = ""
    active_templates: tuple[str, ...] = ()
    bead_assignments: tuple[str, ...] = ()
    special_parameters: FrozenMetadata = field(default_factory=FrozenMetadata)
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.special_parameters, FrozenMetadata):
            object.__setattr__(
                self, "special_parameters", FrozenMetadata(self.special_parameters)
            )
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))

    def describe(self) -> str:
        """Return a human-readable summary of the assembled forcefield."""
        lines = [
            f"AssembledForceField for system type: {self.system_type!r}",
            f"  Base forcefield name : {self.base_forcefield.name}",
            f"  Bond parameters      : {len(self.base_forcefield.bond_parameters)}",
            f"  Nonbonded parameters : {len(self.base_forcefield.nonbonded_parameters)}",
            f"  Angle parameters     : {len(self.angle_parameters)}",
            f"  Dihedral parameters  : {len(self.dihedral_parameters)}",
            f"  Particles            : {len(self.bead_assignments)}",
            f"  Unique bead types    : {len(set(self.bead_assignments))}",
            f"  Active templates     : {', '.join(self.active_templates) or 'none'}",
        ]
        if self.charges:
            n_charged = sum(1 for q in self.charges if abs(q) > 1e-8)
            lines.append(f"  Charged particles    : {n_charged}/{len(self.charges)}")
        return "\n".join(lines)

    def particle_charges(self) -> tuple[float, ...]:
        """Return per-particle charges by looking up bead type charge values.

        If explicit charges were already provided at assembly time, those are
        returned directly.  Otherwise charges are resolved from the universal
        bead type registry for each particle assignment.
        """
        if self.charges:
            return self.charges

        from forcefields.universal_types import STANDARD_BEADS

        charges: list[float] = []
        for bead_name in self.bead_assignments:
            if bead_name in STANDARD_BEADS:
                charges.append(STANDARD_BEADS[bead_name].charge)
            else:
                charges.append(0.0)
        return tuple(charges)


# ---------------------------------------------------------------------------
# Universal forcefield assembler
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class UniversalForceField:
    """Capstone assembler that builds a complete forcefield for any system.

    Reads the bead taxonomy from *universal_types*, selects material
    templates from *material_templates* based on the requested system type,
    generates all pairwise nonbonded parameters via the interaction matrix,
    and collects bonded terms from the active templates.

    Classification: [hybrid] -- established MARTINI-lineage CG methodology
    combined with novel multi-material cross-interaction assembly.
    """

    name: str = "universal_forcefield"
    classification: str = "[hybrid]"
    mixing_rule: str = "martini_matrix"
    nonbonded_cutoff: float = 1.2
    electrostatic_method: str = "reaction_field"
    dielectric: float = 15.0

    # -- ArchitecturalComponent ----------------------------------------------

    def describe_role(self) -> str:
        return (
            "Universal forcefield assembler that wires the bead taxonomy, "
            "interaction matrix, and material templates into a complete "
            "forcefield for any system type in NeuroCGMD."
        )

    def declared_dependencies(self) -> Sequence[str]:
        return (
            "forcefields.universal_types",
            "forcefields.material_templates",
            "forcefields.base_forcefield",
            "forcefields.bonded_potentials",
        )

    # -- DocumentedComponent -------------------------------------------------

    def documentation_paths(self) -> Sequence[str]:
        return ("docs/forcefields/universal_forcefield.md",)

    # -- ValidatableComponent ------------------------------------------------

    def validate(self) -> Sequence[str]:
        issues: list[str] = []
        if self.mixing_rule not in ("martini_matrix", "lorentz_berthelot", "geometric"):
            issues.append(
                f"Unknown mixing_rule {self.mixing_rule!r}; expected one of "
                "'martini_matrix', 'lorentz_berthelot', 'geometric'."
            )
        if self.nonbonded_cutoff <= 0.0:
            issues.append("nonbonded_cutoff must be positive.")
        if self.dielectric <= 0.0:
            issues.append("dielectric must be positive.")
        if self.electrostatic_method not in (
            "reaction_field", "pme", "cutoff", "none",
        ):
            issues.append(
                f"Unknown electrostatic_method {self.electrostatic_method!r}."
            )
        return tuple(issues)

    # -- Public build API ----------------------------------------------------

    def build_from_composition(
        self, composition: SystemComposition
    ) -> AssembledForceField:
        """Assemble a complete forcefield from a SystemComposition descriptor.

        Steps:
        1. Look up the SystemTypeConfig for the requested system type.
        2. Collect all MaterialTemplate objects referenced by that config.
        3. Resolve every UniversalBeadType present in the system (standard +
           custom).
        4. Generate all pairwise nonbonded parameters from the interaction
           matrix.
        5. Flatten bond, angle, and dihedral parameters from the templates.
        6. Compute per-particle charges.
        7. Return an AssembledForceField.
        """
        from forcefields.universal_types import (
            STANDARD_BEADS,
            UniversalBeadType,
        )
        from forcefields.material_templates import (
            TEMPLATE_REGISTRY,
            SYSTEM_CONFIGS,
        )

        # 1. System config
        if composition.system_type not in SYSTEM_CONFIGS:
            raise ContractValidationError(
                f"Unknown system_type {composition.system_type!r}; "
                f"available: {sorted(SYSTEM_CONFIGS.keys())}"
            )
        sys_config = SYSTEM_CONFIGS[composition.system_type]

        # 2. Collect active material templates
        active_templates: list = []
        active_template_names: list[str] = []
        for tpl_name in sys_config.material_templates:
            if tpl_name in TEMPLATE_REGISTRY:
                active_templates.append(TEMPLATE_REGISTRY[tpl_name])
                active_template_names.append(tpl_name)

        # 3. Resolve bead types
        bead_type_map: dict[str, UniversalBeadType] = {}
        for bead_name in composition.bead_names:
            if bead_name in STANDARD_BEADS:
                bead_type_map[bead_name] = STANDARD_BEADS[bead_name]
        # Register custom beads
        for custom_bead in composition.custom_beads:
            if isinstance(custom_bead, UniversalBeadType):
                bead_type_map[custom_bead.name] = custom_bead

        # 4. Nonbonded pairs
        bead_list = list(bead_type_map.values())
        nonbonded_params = self._generate_nonbonded_pairs(bead_list)

        # 5. Bonded terms
        bond_params = self._collect_bond_parameters(active_templates)
        angle_params = self._collect_angle_parameters(active_templates)
        dihedral_params = self._collect_dihedral_parameters(active_templates)

        # 6. Per-particle charges
        charges: list[float] = []
        for bead_name in composition.particle_bead_assignments:
            if bead_name in bead_type_map:
                charges.append(bead_type_map[bead_name].charge)
            else:
                charges.append(0.0)

        # 7. Build the base forcefield and assemble
        base_ff = BaseForceField(
            name=f"{self.name}_{composition.system_type}",
            bond_parameters=bond_params,
            nonbonded_parameters=nonbonded_params,
            metadata=FrozenMetadata({
                "mixing_rule": self.mixing_rule,
                "cutoff_nm": self.nonbonded_cutoff,
                "electrostatic_method": self.electrostatic_method,
                "dielectric": self.dielectric,
            }),
        )

        return AssembledForceField(
            base_forcefield=base_ff,
            angle_parameters=angle_params,
            dihedral_parameters=dihedral_params,
            charges=tuple(charges),
            system_type=composition.system_type,
            active_templates=tuple(active_template_names),
            bead_assignments=composition.particle_bead_assignments,
            special_parameters=FrozenMetadata(
                dict(sys_config.special_parameters)
                if hasattr(sys_config, "special_parameters")
                   and sys_config.special_parameters
                else {}
            ),
            metadata=FrozenMetadata({
                "assembler": self.name,
                "classification": self.classification,
                "n_bead_types": len(bead_type_map),
                "n_particles": len(composition.particle_bead_assignments),
            }),
        )

    def build_for_system_type(
        self,
        system_type: str,
        bead_assignments: tuple[str, ...],
    ) -> AssembledForceField:
        """Convenience wrapper: build a forcefield from a system type string.

        Creates a SystemComposition from the bead assignments and delegates
        to build_from_composition.
        """
        unique_beads = tuple(sorted(set(bead_assignments)))
        composition = SystemComposition(
            system_type=system_type,
            bead_names=unique_beads,
            particle_bead_assignments=bead_assignments,
        )
        return self.build_from_composition(composition)

    # -- Internal helpers ----------------------------------------------------

    def _generate_nonbonded_pairs(
        self, bead_types: list
    ) -> tuple[NonbondedParameter, ...]:
        """Generate all unique pairwise LJ parameters for the given bead types.

        For every unique (i, j) pair (including self-interactions) the
        interaction matrix is consulted via compute_pair_parameters to obtain
        sigma and epsilon.  Electrostatics (charges) are handled separately
        at the particle level, not embedded in the LJ parameters.
        """
        from forcefields.universal_types import compute_pair_parameters

        seen_keys: set[tuple[str, str]] = set()
        params: list[NonbondedParameter] = []

        for i, bead_a in enumerate(bead_types):
            for j in range(i, len(bead_types)):
                bead_b = bead_types[j]
                key = tuple(sorted((bead_a.name, bead_b.name)))
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                sigma, epsilon = compute_pair_parameters(bead_a, bead_b)

                # Guard: epsilon can be zero for purely repulsive pairs but
                # sigma must always be positive.
                if sigma <= 0.0:
                    sigma = 0.5 * (bead_a.sigma + bead_b.sigma)
                if sigma <= 0.0:
                    sigma = 0.47  # MARTINI default sigma fallback

                params.append(
                    NonbondedParameter(
                        bead_type_a=bead_a.name,
                        bead_type_b=bead_b.name,
                        sigma=sigma,
                        epsilon=max(epsilon, 0.0),
                        cutoff=self.nonbonded_cutoff,
                        metadata=FrozenMetadata({
                            "mixing_rule": self.mixing_rule,
                            "source": "interaction_matrix",
                        }),
                    )
                )

        return tuple(params)

    @staticmethod
    def _collect_bond_parameters(
        templates: list,
    ) -> tuple[BondParameter, ...]:
        """Flatten BondTemplate entries from all active templates into BondParameters."""
        from forcefields.material_templates import MaterialTemplate

        seen_keys: set = set()
        params: list[BondParameter] = []

        for tpl in templates:
            if not isinstance(tpl, MaterialTemplate):
                continue
            for bt in tpl.bonds:
                bp = BondParameter(
                    bead_type_a=bt.bead_type_a,
                    bead_type_b=bt.bead_type_b,
                    equilibrium_distance=bt.equilibrium_distance,
                    stiffness=bt.force_constant,
                    metadata=FrozenMetadata({
                        "template": tpl.name,
                    }),
                )
                key = bp.parameter_key()
                if key not in seen_keys:
                    seen_keys.add(key)
                    params.append(bp)

        return tuple(params)

    @staticmethod
    def _collect_angle_parameters(
        templates: list,
    ) -> tuple[AngleParameter, ...]:
        """Flatten AngleTemplate entries from all active templates into AngleParameters."""
        from forcefields.material_templates import MaterialTemplate

        seen_keys: set = set()
        params: list[AngleParameter] = []

        for tpl in templates:
            if not isinstance(tpl, MaterialTemplate):
                continue
            for at in tpl.angles:
                ap = AngleParameter(
                    bead_type_a=at.bead_type_a,
                    bead_type_b=at.bead_type_b,
                    bead_type_c=at.bead_type_c,
                    equilibrium_angle=at.equilibrium_angle,
                    force_constant=at.force_constant,
                    metadata=FrozenMetadata({
                        "template": tpl.name,
                    }),
                )
                key = (at.bead_type_a, at.bead_type_b, at.bead_type_c)
                if key not in seen_keys:
                    seen_keys.add(key)
                    params.append(ap)

        return tuple(params)

    @staticmethod
    def _collect_dihedral_parameters(
        templates: list,
    ) -> tuple[DihedralParameter, ...]:
        """Flatten DihedralTemplate entries from all active templates into DihedralParameters."""
        from forcefields.material_templates import MaterialTemplate

        seen_keys: set = set()
        params: list[DihedralParameter] = []

        for tpl in templates:
            if not isinstance(tpl, MaterialTemplate):
                continue
            for dt in tpl.dihedrals:
                bt = dt.bead_types
                dp = DihedralParameter(
                    bead_type_a=bt[0],
                    bead_type_b=bt[1],
                    bead_type_c=bt[2],
                    bead_type_d=bt[3],
                    force_constant=dt.force_constant,
                    multiplicity=dt.multiplicity,
                    phase=dt.phase,
                    metadata=FrozenMetadata({
                        "template": tpl.name,
                    }),
                )
                key = (bt[0], bt[1], bt[2], bt[3], dt.multiplicity)
                if key not in seen_keys:
                    seen_keys.add(key)
                    params.append(dp)

        return tuple(params)


# ---------------------------------------------------------------------------
# Go-model native contact generator
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GoModelContactGenerator:
    """Generate native-contact biasing potentials for Go/SBM models.

    Given a reference native structure, identifies all bead pairs within
    a contact cutoff distance and creates attractive bond parameters that
    bias the system toward the native fold.

    Classification: [adapted] -- Go/structure-based models are established
    in the literature; this adaptation integrates them with the universal
    bead taxonomy.
    """

    native_positions: tuple
    contact_cutoff: float = 0.8
    contact_strength: float = 2.0
    classification: str = "[adapted]"

    def generate_contacts(
        self, positions: tuple | None = None
    ) -> tuple[BondParameter, ...]:
        """Find native contacts and return biasing BondParameters.

        A native contact exists between particles *i* and *j* (|i-j| > 3)
        when their distance in *native_positions* is less than
        *contact_cutoff*.  Each contact becomes an attractive harmonic
        bond with equilibrium distance equal to the native distance and
        stiffness equal to *contact_strength*.
        """
        from topology.bonds import BondKind

        ref = self.native_positions
        n = len(ref)
        contacts: list[BondParameter] = []

        for i in range(n):
            for j in range(i + 4, n):
                dx = ref[i][0] - ref[j][0]
                dy = ref[i][1] - ref[j][1]
                dz = ref[i][2] - ref[j][2]
                dist = sqrt(dx * dx + dy * dy + dz * dz)
                if dist < self.contact_cutoff and dist > 0.0:
                    contacts.append(
                        BondParameter(
                            bead_type_a=f"P{i}",
                            bead_type_b=f"P{j}",
                            equilibrium_distance=dist,
                            stiffness=self.contact_strength,
                            kind=BondKind.STRUCTURAL,
                            metadata=FrozenMetadata({
                                "source": "go_model",
                                "native_contact": True,
                                "particle_i": i,
                                "particle_j": j,
                            }),
                        )
                    )

        return tuple(contacts)


# ---------------------------------------------------------------------------
# Self-assembly modulator
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SelfAssemblyModulator:
    """Modulate nonbonded parameters to capture concentration-dependent
    self-assembly behavior.

    Above a concentration threshold, like-material interactions are boosted
    to drive aggregation.  This is a phenomenological model capturing the
    physics that CG models often under-predict self-assembly kinetics.

    Classification: [proposed novel] -- this concentration-dependent
    parameter modulation is a novel contribution.
    """

    concentration_threshold: float = 0.5
    aggregation_boost: float = 1.5
    classification: str = "[proposed novel]"

    def modulate_parameters(
        self,
        base_params: tuple[NonbondedParameter, ...],
        concentration: float,
    ) -> tuple[NonbondedParameter, ...]:
        """Boost same-material-class attractive interactions above threshold.

        When *concentration* exceeds *concentration_threshold*, epsilon
        values for same-material-class pairs are multiplied by
        *aggregation_boost*.  Cross-material pairs are left unchanged.
        """
        if concentration <= self.concentration_threshold:
            return base_params

        from forcefields.universal_types import STANDARD_BEADS

        modulated: list[NonbondedParameter] = []
        for param in base_params:
            bead_a = STANDARD_BEADS.get(param.bead_type_a)
            bead_b = STANDARD_BEADS.get(param.bead_type_b)

            # Boost only when both beads belong to the same material class
            if (
                bead_a is not None
                and bead_b is not None
                and bead_a.material_class == bead_b.material_class
                and param.epsilon > 0.0
            ):
                boosted_epsilon = param.epsilon * self.aggregation_boost
                modulated.append(
                    NonbondedParameter(
                        bead_type_a=param.bead_type_a,
                        bead_type_b=param.bead_type_b,
                        sigma=param.sigma,
                        epsilon=boosted_epsilon,
                        cutoff=param.cutoff,
                        metadata=FrozenMetadata({
                            "original_epsilon": param.epsilon,
                            "boost_factor": self.aggregation_boost,
                            "concentration": concentration,
                            "modulation": "self_assembly_boost",
                        }),
                    )
                )
            else:
                modulated.append(param)

        return tuple(modulated)


# ---------------------------------------------------------------------------
# Adsorption potential generator
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AdsorptionPotentialGenerator:
    """Generate surface-adsorbate cross-interaction parameters.

    Creates LJ parameters for interactions between surface beads and
    adsorbate beads, with strength modulated by chemical character:
    polar adsorbates on polar surfaces get stronger epsilon, hydrophobic
    adsorbates on polar surfaces get weaker epsilon, etc.

    Classification: [adapted] -- surface adsorption potentials are
    established; this adaptation integrates with the universal bead taxonomy.
    """

    surface_epsilon: float = 8.0
    surface_sigma: float = 0.35
    adsorption_range: float = 1.0
    classification: str = "[adapted]"

    def generate_surface_interactions(
        self,
        surface_beads: tuple,
        adsorbate_beads: tuple,
    ) -> tuple[NonbondedParameter, ...]:
        """Create LJ parameters for all surface-adsorbate cross pairs.

        Chemical compatibility modulates the interaction strength:
        - Polar surface + polar adsorbate   -> full surface_epsilon
        - Polar surface + nonpolar adsorbate -> 0.3 * surface_epsilon
        - Nonpolar surface + nonpolar ads.   -> 0.8 * surface_epsilon
        - Nonpolar surface + polar adsorbate -> 0.4 * surface_epsilon
        """
        from forcefields.universal_types import ChemicalCharacter

        params: list[NonbondedParameter] = []
        seen_keys: set[tuple[str, str]] = set()

        for s_bead in surface_beads:
            for a_bead in adsorbate_beads:
                key = tuple(sorted((s_bead.name, a_bead.name)))
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                # Determine chemical compatibility scaling
                s_polar = s_bead.chemical_character in (
                    ChemicalCharacter.POLAR,
                    ChemicalCharacter.CHARGED_POSITIVE,
                    ChemicalCharacter.CHARGED_NEGATIVE,
                )
                a_polar = a_bead.chemical_character in (
                    ChemicalCharacter.POLAR,
                    ChemicalCharacter.CHARGED_POSITIVE,
                    ChemicalCharacter.CHARGED_NEGATIVE,
                )

                if s_polar and a_polar:
                    eps_scale = 1.0
                elif s_polar and not a_polar:
                    eps_scale = 0.3
                elif not s_polar and not a_polar:
                    eps_scale = 0.8
                else:  # nonpolar surface, polar adsorbate
                    eps_scale = 0.4

                sigma = 0.5 * (self.surface_sigma + a_bead.sigma)
                epsilon = self.surface_epsilon * eps_scale

                params.append(
                    NonbondedParameter(
                        bead_type_a=s_bead.name,
                        bead_type_b=a_bead.name,
                        sigma=sigma,
                        epsilon=epsilon,
                        cutoff=self.adsorption_range,
                        metadata=FrozenMetadata({
                            "source": "adsorption_potential",
                            "eps_scale": eps_scale,
                            "surface_bead": s_bead.name,
                            "adsorbate_bead": a_bead.name,
                        }),
                    )
                )

        return tuple(params)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def quick_build(
    system_type: str,
    n_particles: int,
    bead_assignments: tuple[str, ...] | None = None,
) -> AssembledForceField:
    """Build a forcefield with minimal boilerplate.

    If *bead_assignments* is ``None`` every particle is assigned the first
    bead type listed in the system configuration's default beads.

    Parameters
    ----------
    system_type:
        Key into SYSTEM_CONFIGS (e.g. ``"protein_membrane"``).
    n_particles:
        Total number of particles in the system.
    bead_assignments:
        Optional per-particle bead type mapping.  When omitted, the first
        bead from the system config is replicated for all particles.

    Returns
    -------
    AssembledForceField ready for simulation.
    """
    from forcefields.material_templates import SYSTEM_CONFIGS

    if system_type not in SYSTEM_CONFIGS:
        raise ContractValidationError(
            f"Unknown system_type {system_type!r}; "
            f"available: {sorted(SYSTEM_CONFIGS.keys())}"
        )

    if bead_assignments is None:
        sys_config = SYSTEM_CONFIGS[system_type]
        # Derive a default bead from the first material template
        _first_template_name = sys_config.material_templates[0] if sys_config.material_templates else "protein"
        _default_bead_map = {
            "protein": "BB", "peptide": "BB", "water": "W", "ion": "Qp_Na",
            "lipid_dppc": "C1_tail", "lipid_dopc": "C1_tail",
            "polymer_peg": "PEG", "gold_surface": "Au_surf",
            "silica_surface": "SiO2_surf", "graphene": "C_graph",
            "sam": "SAM_CH3", "ligand_generic": "LIG_hydrophobic",
        }
        default_bead = _default_bead_map.get(_first_template_name, "BB")
        bead_assignments = tuple(default_bead for _ in range(n_particles))

    ff = UniversalForceField()
    return ff.build_for_system_type(system_type, bead_assignments)
