"""PDB-derived structure targets for the barnase-barstar benchmark."""

from __future__ import annotations

from pathlib import Path

from benchmarks.reference_cases.structure_targets import (
    InterfaceContactTarget,
    ReferenceStructureTarget,
    StructureLandmarkTarget,
    centroid,
    distance,
    load_local_pdb_loader,
    selection_centroid,
)
from core.exceptions import ContractValidationError


def default_barnase_barstar_reference_path() -> Path:
    """Return the default local path for the bundled 1BRS structure file."""

    return Path(__file__).resolve().parent / "data" / "1BRS.pdb"


def barnase_barstar_structure_target(pdb_path: str | Path | None = None) -> ReferenceStructureTarget:
    """Return a local-PDB-derived centroid scaffold informed by the real 1BRS geometry."""

    resolved_path = (
        Path(pdb_path).expanduser().resolve()
        if pdb_path is not None
        else default_barnase_barstar_reference_path()
    )
    if not resolved_path.exists():
        raise ContractValidationError(f"Local 1BRS structure file not found at {resolved_path}.")

    pdb_loader = load_local_pdb_loader()
    structure = pdb_loader.load_pdb_file(resolved_path, structure_id="1BRS")

    landmark_specs = (
        (
            "Barnase_core",
            "A",
            (45, 46, 50, 51, 55, 56),
            ("VAL", "ALA", "SER", "ILE", "ILE", "PHE"),
            "Barnase structural core anchor away from the dominant interface patch.",
        ),
        (
            "Barnase_basic_patch",
            "A",
            (27, 59, 83, 87),
            ("LYS", "ARG", "ARG", "ARG"),
            "Barnase basic steering patch spanning the dominant positively charged recognition surface.",
        ),
        (
            "Barnase_recognition_loop",
            "A",
            (101, 102, 103, 104),
            ("ASP", "HIS", "TYR", "GLN"),
            "Barnase late-docking recognition loop near the bound-state interface rim.",
        ),
        (
            "Barstar_core",
            "D",
            (16, 17, 20, 24, 25, 26),
            ("LEU", "HIS", "LEU", "LEU", "ALA", "LEU"),
            "Barstar structural core anchor supporting the inhibitory interface.",
        ),
        (
            "Barstar_helix_face",
            "D",
            (29, 30, 31, 33, 34),
            ("TYR", "TYR", "GLY", "ASN", "LEU"),
            "Barstar helix-face recognition family that accepts the barnase docking surface.",
        ),
        (
            "Barstar_acidic_patch",
            "D",
            (35, 39),
            ("ASP", "ASP"),
            "Barstar acidic hotspot patch around the dominant electrostatic steering residues.",
        ),
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
        (
            "Barnase_basic_patch",
            "Barstar_acidic_patch",
            1.10,
            "Dominant basic-acidic closure in the bound complex.",
        ),
        (
            "Barnase_basic_patch",
            "Barstar_helix_face",
            1.35,
            "Electrostatically steered approach into the barstar recognition face.",
        ),
        (
            "Barnase_recognition_loop",
            "Barstar_helix_face",
            1.20,
            "Late-stage recognition-loop docking against the barstar helix face.",
        ),
        (
            "Barnase_recognition_loop",
            "Barstar_acidic_patch",
            1.50,
            "Loop stabilization near the acidic rim of the barstar interface.",
        ),
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
        name="barnase_barstar_landmark_target",
        classification="[adapted]",
        title="Barnase-Barstar Atomistic Centroid Scaffold",
        summary=(
            "A local 1BRS-derived centroid scaffold built from real barnase-barstar atom records "
            "for selected interface residue groups. This supports atomistic-reference alignment "
            "of the coarse live proxy without claiming full all-atom simulation fidelity."
        ),
        source_pdb_id="1BRS",
        landmarks=landmarks,
        interface_contacts=interface_contacts,
        metadata={
            "source_pdb_id": "1BRS",
            "source_file": str(resolved_path),
            "representation": "pdb_derived_atomistic_centroids",
            "chain_pair": ("A", "D"),
            "dominant_interface_pair": ("Barnase_basic_patch", "Barstar_acidic_patch"),
            "honesty_note": (
                "Target landmark positions are derived from actual 1BRS atom coordinates, "
                "but the observed live system remains coarse-grained."
            ),
        },
    )
