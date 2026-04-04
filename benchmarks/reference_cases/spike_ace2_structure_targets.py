"""PDB-derived structure targets for the harder spike-ACE2 benchmark."""

from __future__ import annotations

from pathlib import Path

from benchmarks.reference_cases.structure_targets import (
    InterfaceContactTarget,
    ReferenceStructureTarget,
    StructureLandmarkTarget,
    distance,
    load_local_pdb_loader,
    selection_centroid,
)
from core.exceptions import ContractValidationError


def default_spike_ace2_reference_path() -> Path:
    """Return the default local path for the bundled 6M0J structure file."""

    return Path(__file__).resolve().parent / "data" / "6M0J.pdb"


def spike_ace2_structure_target(pdb_path: str | Path | None = None) -> ReferenceStructureTarget:
    """Return a local-PDB-derived landmark scaffold informed by the real 6M0J geometry."""

    resolved_path = Path(pdb_path).expanduser().resolve() if pdb_path is not None else default_spike_ace2_reference_path()
    if not resolved_path.exists():
        raise ContractValidationError(f"Local 6M0J structure file not found at {resolved_path}.")

    pdb_loader = load_local_pdb_loader()
    structure = pdb_loader.load_pdb_file(resolved_path, structure_id="6M0J")

    landmark_specs = (
        ("ACE2_core", "A", (19, 20, 21), ("SER", "THR", "ILE"), "ACE2 structural core anchor."),
        ("ACE2_alpha1_support", "A", (27, 28, 30), ("THR", "PHE", "ASP"), "ACE2 alpha1 support family approaching the binding ridge."),
        ("ACE2_alpha1_hotspot", "A", (31, 35), ("LYS", "GLU"), "ACE2 alpha1 hotspot family around K31/E35."),
        ("ACE2_beta_hotspot", "A", (38, 41), ("ASP", "TYR"), "ACE2 secondary hotspot family around D38/Y41."),
        ("ACE2_support", "A", (353, 355), ("LYS", "ASP"), "ACE2 support hotspot family around K353/D355."),
        ("ACE2_glycan_side", "A", (82, 83), ("MET", "TYR"), "ACE2 side-facing landmark near the M82/Y83 region."),
        ("RBD_core", "E", (438, 439, 440), ("SER", "ASN", "ASN"), "RBD structural core anchor."),
        ("RBD_ridge_support", "E", (449, 453), ("TYR", "TYR"), "RBD ridge support family near Y449/Y453."),
        ("RBD_ridge_hotspot", "E", (493, 498), ("GLN", "GLN"), "RBD ridge hotspot family near Q493/Q498."),
        ("RBD_loop_hotspot", "E", (486, 487), ("PHE", "ASN"), "RBD loop hotspot family around F486/N487."),
        ("RBD_support", "E", (501, 505), ("ASN", "TYR"), "RBD support hotspot family near N501/Y505."),
        ("RBD_shielded_side", "E", (417,), ("LYS",), "RBD shield-side landmark near K417."),
    )

    landmarks = tuple(
        StructureLandmarkTarget(
            label=label,
            chain_id=chain_id,
            residue_ids=residue_ids,
            residue_names=residue_names,
            description=description,
            target_position=selection_centroid(structure, chain_id=chain_id, residue_ids=residue_ids),
            metadata={"source_file": str(resolved_path)},
        )
        for label, chain_id, residue_ids, residue_names, description in landmark_specs
    )
    landmark_map = {landmark.label: landmark for landmark in landmarks}

    contact_specs = (
        ("ACE2_alpha1_hotspot", "RBD_ridge_hotspot", 2.2, "Dominant hotspot-family closure."),
        ("ACE2_beta_hotspot", "RBD_loop_hotspot", 2.2, "Secondary hotspot-family closure."),
        ("ACE2_alpha1_support", "RBD_ridge_support", 2.4, "Support-ridge family alignment."),
        ("ACE2_support", "RBD_support", 2.8, "Distal support contact family."),
        ("ACE2_glycan_side", "RBD_shielded_side", 2.5, "Outer-side orientation family."),
    )
    interface_contacts = tuple(
        InterfaceContactTarget(
            source_label=source_label,
            target_label=target_label,
            max_distance=distance(
                landmark_map[source_label].target_position,
                landmark_map[target_label].target_position,
            )
            + tolerance,
            description=description,
            metadata={
                "target_distance": distance(
                    landmark_map[source_label].target_position,
                    landmark_map[target_label].target_position,
                ),
                "tolerance": tolerance,
            },
        )
        for source_label, target_label, tolerance, description in contact_specs
    )

    return ReferenceStructureTarget(
        name="spike_ace2_landmark_target",
        classification="[adapted]",
        title="Spike-ACE2 Atomistic Centroid Scaffold",
        summary=(
            "A local 6M0J-derived centroid scaffold built from real atom records of "
            "selected interface residue groups. This supports atomistic-reference alignment "
            "of the coarse live proxy without claiming full all-atom simulation fidelity."
        ),
        source_pdb_id="6M0J",
        landmarks=landmarks,
        interface_contacts=interface_contacts,
        metadata={
            "source_pdb_id": "6M0J",
            "source_file": str(resolved_path),
            "representation": "pdb_derived_atomistic_centroids",
            "dominant_interface_pair": ("ACE2_alpha1_hotspot", "RBD_ridge_hotspot"),
            "honesty_note": (
                "Target landmark positions are derived from actual 6M0J atom coordinates, "
                "but the observed live system remains coarse-grained."
            ),
        },
    )
