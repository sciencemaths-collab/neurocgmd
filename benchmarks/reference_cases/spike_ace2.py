"""SARS-CoV-2 spike RBD - ACE2 reference case with known answers."""

from __future__ import annotations

from benchmarks.reference_cases.models import (
    ExperimentalReferenceCase,
    ReferenceObservable,
    ReferenceSource,
    StructuralReference,
)


def spike_ace2_reference_case() -> ExperimentalReferenceCase:
    """Return a harder real-world receptor-binding benchmark target."""

    return ExperimentalReferenceCase(
        name="spike_ace2",
        classification="[established]",
        title="SARS-CoV-2 Spike RBD - ACE2 Binding Benchmark",
        problem_type="viral_receptor_association",
        summary=(
            "The SARS-CoV-2 spike receptor-binding domain recognizes human ACE2 through "
            "a larger, more distributed interface than the early toy docking targets, "
            "making it a harder structural benchmark for later comparison."
        ),
        structural_reference=StructuralReference(
            bound_complex_pdb_id="6M0J",
            unbound_partner_pdb_ids=("1R42", "6VYB"),
            description=(
                "Use 6M0J as the bound ACE2-RBD complex, 1R42 as the free ACE2 receptor "
                "reference, and 6VYB as a prefusion spike reference containing an RBD-up state."
            ),
            source_label="wang_2020_6m0j",
            metadata={
                "alternate_bound_complex_pdb_id": "6LZG",
                "notes": (
                    "The reported affinity depends on construct and assay details, so later "
                    "agreement claims should compare matched experimental conditions."
                ),
            },
        ),
        observables=(
            ReferenceObservable(
                name="apparent_dissociation_constant",
                expected_value=1.47e-8,
                units="M",
                description=(
                    "Apparent dissociation constant reported for the SARS-CoV-2 RBD binding "
                    "human ACE2 in the 2020 structural benchmark literature."
                ),
                source_label="wang_2020_nature",
            ),
            ReferenceObservable(
                name="bound_complex_resolution",
                expected_value=2.45,
                units="angstrom",
                description="Resolution of the bound ACE2-RBD complex structure 6M0J.",
                source_label="wang_2020_6m0j",
            ),
        ),
        sources=(
            ReferenceSource(
                label="wang_2020_6m0j",
                url="https://www.rcsb.org/structure/6M0J",
                source_type="structure",
                metadata={"pdb_id": "6M0J"},
            ),
            ReferenceSource(
                label="ace2_free_1r42",
                url="https://www.rcsb.org/structure/1R42",
                source_type="structure",
                metadata={"pdb_id": "1R42"},
            ),
            ReferenceSource(
                label="spike_prefusion_6vyb",
                url="https://www.rcsb.org/structure/6VYB",
                source_type="structure",
                metadata={"pdb_id": "6VYB"},
            ),
            ReferenceSource(
                label="wang_2020_nature",
                url="https://pubmed.ncbi.nlm.nih.gov/32225176/",
                source_type="structure_and_affinity",
            ),
        ),
        recommended_comparisons=(
            "Compare recovered ACE2 alpha1-helix to RBD ridge orientation against PDB 6M0J.",
            "Compare free ACE2-like and spike-like starting geometries against 1R42 and 6VYB before docking.",
            "Treat the reported apparent Kd as assay-dependent and compare only after calibration to a matched construct.",
            "Track whether the proxy forms a distributed hotspot network instead of a single-point collapse.",
        ),
        metadata={
            "biological_role": "viral receptor recognition initiating host-cell entry",
            "why_this_case_is_harder": (
                "the interface is larger, more distributed, and more conformationally constrained "
                "than the early barnase-barstar benchmark proxy"
            ),
        },
    )
