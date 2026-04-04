"""Force-field builder for imported arbitrary-protein coarse systems."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata
from forcefields.base_forcefield import BaseForceField, BondParameter, NonbondedParameter
from forcefields.protein_shadow_profiles import ProteinShadowProfileFactory
from topology.bonds import BondKind
from topology.system_topology import SystemTopology


@dataclass(slots=True)
class ImportedProteinForceFieldBuilder(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """[hybrid] Build baseline coarse force fields for imported protein systems."""

    profile_factory: ProteinShadowProfileFactory = field(default_factory=ProteinShadowProfileFactory)
    name: str = "imported_protein_forcefield_builder"
    classification: str = "[hybrid]"

    def describe_role(self) -> str:
        return (
            "Builds a reusable baseline force field for imported proteins by combining "
            "their structural bond geometry with protein-general nonbonded priors."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "forcefields/base_forcefield.py",
            "forcefields/protein_shadow_profiles.py",
            "topology/protein_coarse_mapping.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/arbitrary_protein_input_pipeline.md",)

    def validate(self) -> tuple[str, ...]:
        return self.profile_factory.validate()

    def build(
        self,
        topology: SystemTopology,
        *,
        scenario_label: str,
        reference_label: str | None = None,
        metadata: FrozenMetadata | dict[str, object] | None = None,
    ) -> BaseForceField:
        """Build a baseline force field from one imported topology."""

        issues = list(self.validate())
        issues.extend(topology.validate())
        if issues:
            raise ContractValidationError("; ".join(issues))

        nonbonded_bundle = self.profile_factory.build_parameter_set(
            topology,
            scenario_label=scenario_label,
            reference_label=reference_label,
        )
        nonbonded_parameters = tuple(
            NonbondedParameter(
                bead_type_a=profile.bead_type_a,
                bead_type_b=profile.bead_type_b,
                sigma=profile.sigma,
                epsilon=profile.epsilon,
                cutoff=profile.cutoff,
                metadata=profile.metadata.with_updates(
                    {
                        "source_label": profile.source_label,
                        "fidelity_label": profile.fidelity_label,
                    }
                ),
            )
            for profile in nonbonded_bundle.nonbonded_profiles
        )

        bonded_measurements: dict[
            tuple[BondKind, tuple[str, str]],
            list[tuple[float, float]],
        ] = defaultdict(list)
        for bond in topology.bonds:
            parameter = topology.bead_for_particle(bond.particle_index_a).bead_type
            other_parameter = topology.bead_for_particle(bond.particle_index_b).bead_type
            key = (bond.kind, tuple(sorted((parameter, other_parameter))))
            bonded_measurements[key].append(
                (
                    bond.equilibrium_distance if bond.equilibrium_distance is not None else 0.9,
                    bond.stiffness if bond.stiffness is not None else 85.0,
                )
            )

        bond_parameters = tuple(
            BondParameter(
                bead_type_a=pair[0],
                bead_type_b=pair[1],
                equilibrium_distance=sum(item[0] for item in measurements) / len(measurements),
                stiffness=sum(item[1] for item in measurements) / len(measurements),
                kind=kind,
                metadata={
                    "measurement_count": len(measurements),
                    "source": "imported_protein_forcefield_builder",
                },
            )
            for (kind, pair), measurements in sorted(bonded_measurements.items(), key=lambda item: item[0][1])
        )

        combined_metadata: dict[str, object] = {}
        if metadata is not None:
            combined_metadata.update(metadata.to_dict() if isinstance(metadata, FrozenMetadata) else metadata)
        combined_metadata.update(
            {
                "scenario": scenario_label,
                "reference_label": reference_label,
                "bead_type_count": len(topology.bead_types),
            }
        )
        return BaseForceField(
            name=f"{scenario_label}_imported_forcefield",
            bond_parameters=bond_parameters,
            nonbonded_parameters=nonbonded_parameters,
            metadata=combined_metadata,
        )
