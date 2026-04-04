"""Barnase-barstar reference case with experimentally known answers."""

from __future__ import annotations

from benchmarks.reference_cases.models import (
    ExperimentalReferenceCase,
    ReferenceObservable,
    ReferenceSource,
    StructuralReference,
)


def barnase_barstar_reference_case() -> ExperimentalReferenceCase:
    """Return the first real-world benchmark target for later comparison."""

    return ExperimentalReferenceCase(
        name="barnase_barstar",
        classification="[established]",
        title="Barnase-Barstar Association Benchmark",
        problem_type="protein_protein_association",
        summary=(
            "A bacterial ribonuclease and its natural inhibitor form one of the best "
            "characterized high-affinity protein-protein complexes, with known bound "
            "structure, free partners, and measured association/dissociation kinetics."
        ),
        structural_reference=StructuralReference(
            bound_complex_pdb_id="1BRS",
            unbound_partner_pdb_ids=("1A2P", "1A19"),
            description=(
                "Use 1BRS as the bound reference complex and 1A2P/1A19 as the free "
                "partner references for later structural comparison."
            ),
            source_label="buckle_1994_1brs",
            metadata={
                "alternate_bound_complex_pdb_id": "1BGS",
                "buried_surface_area_angstrom2": 1630.0,
            },
        ),
        observables=(
            ReferenceObservable(
                name="association_rate_constant",
                expected_value=6.0e8,
                units="M^-1 s^-1",
                description="Association rate constant at pH 8 for wild-type barnase-barstar.",
                source_label="schreiber_fersht_1993",
            ),
            ReferenceObservable(
                name="dissociation_rate_constant",
                expected_value=8.0e-6,
                units="s^-1",
                description="Dissociation rate constant at pH 8 for wild-type barnase-barstar.",
                source_label="schreiber_fersht_1993",
            ),
            ReferenceObservable(
                name="dissociation_constant",
                expected_value=1.3e-14,
                units="M",
                description="Dissociation constant inferred from the measured on/off rates at pH 8.",
                source_label="schreiber_fersht_1993",
            ),
            ReferenceObservable(
                name="basal_association_rate_constant",
                expected_value=1.0e5,
                units="M^-1 s^-1",
                description="Approximate electrostatics-free baseline association rate estimated from screening/mutagenesis analysis.",
                source_label="schreiber_fersht_1996",
            ),
            ReferenceObservable(
                name="electrostatically_assisted_association_ceiling",
                expected_value=5.0e9,
                units="M^-1 s^-1",
                description="Electrostatically assisted upper-end association rate reported for the system.",
                source_label="schreiber_fersht_1996",
            ),
            ReferenceObservable(
                name="bound_complex_resolution",
                expected_value=2.0,
                units="angstrom",
                description="Resolution of the bound barnase-barstar complex structure 1BRS.",
                source_label="buckle_1994_1brs",
            ),
        ),
        sources=(
            ReferenceSource(
                label="buckle_1994_1brs",
                url="https://www.rcsb.org/structure/1BRS",
                source_type="structure",
                metadata={"pdb_id": "1BRS"},
            ),
            ReferenceSource(
                label="hartley_1993_1bgs",
                url="https://www.rcsb.org/structure/1BGS",
                source_type="structure",
                metadata={"pdb_id": "1BGS"},
            ),
            ReferenceSource(
                label="barnase_free_1a2p",
                url="https://www.rcsb.org/structure/1A2P",
                source_type="structure",
                metadata={"pdb_id": "1A2P"},
            ),
            ReferenceSource(
                label="barstar_free_1a19",
                url="https://www.ncbi.nlm.nih.gov/Structure/pdb/1A19",
                source_type="structure",
                metadata={"pdb_id": "1A19"},
            ),
            ReferenceSource(
                label="schreiber_fersht_1993",
                url="https://pubmed.ncbi.nlm.nih.gov/8494892/",
                source_type="kinetics",
            ),
            ReferenceSource(
                label="schreiber_fersht_1996",
                url="https://www.nature.com/articles/nsb0596-427",
                source_type="mechanism",
            ),
        ),
        recommended_comparisons=(
            "Compare recovered bound orientation and interface contact map against PDB 1BRS.",
            "Compare simulated free-partner conformations against 1A2P and 1A19 before binding.",
            "Estimate k_on, k_off, and K_d under matched conditions before claiming agreement.",
            "Test whether long-range electrostatic steering produces an encounter-complex phase before final docking.",
        ),
        metadata={
            "biological_role": "barnase is a secreted RNase and barstar is its intracellular inhibitor",
            "why_this_case_first": "small enough to manage, but rich enough to test structure, kinetics, and electrostatic steering",
        },
    )
