# Chemistry Semantics and Live Control

## Classification

- `chemistry/residue_semantics.py`: `[hybrid]`
- `chemistry/interface_logic.py`: `[hybrid]`
- `ml/live_features.py`: `[hybrid]`
- `ai_control/chemistry_governor.py`: `[proposed novel]`

## Purpose

This phase adds an explicit chemistry-semantic layer between coarse topology and
live decision-making. Before it, the runtime could react to graph density,
memory, qcloud activity, and ML uncertainty, but not to whether the observed
interface was chemically plausible.

The new layer introduces:

- bead-level chemistry descriptors
- cross-interface chemistry scoring
- a live feature encoder spanning chemistry, structure, fidelity, and graph state
- chemistry-aware uncertainty escalation
- chemistry-aware executive guidance
- dashboard-visible interface chemistry diagnostics

## Mathematical Role

Each bead receives a bounded descriptor:

- formal charge `q in [-1, 1]`
- hydropathy `h in [-1, 1]`
- flexibility `f in [0, 1]`
- aromaticity `a in [0, 1]`
- hydrogen-bond capacity `b in [0, 1]`
- hotspot propensity `s in [0, 1]`

For each active cross-interface pair, the analyzer evaluates:

- charge compatibility
- hydropathy alignment
- hydrogen-bond match
- aromatic/hotspot reinforcement
- flexibility penalty
- distance quality

The bounded pair score is:

`pair_score = clamp((w_q C_q + w_h C_h + w_b C_b + w_a C_a + w_d C_d - w_f P_f) / Z, 0, 1)`

where `Z` is the sum of positive weights.

Aggregate interface metrics include:

- favorable pair fraction
- mean pair score
- charge complementarity
- hydropathy alignment
- flexibility pressure
- hotspot-pair fraction

## Architectural Role

The dependency direction is:

- `topology/` and bead metadata feed `chemistry/`
- `chemistry/` feeds `ml/live_features.py`
- `ml/live_features.py` feeds `ml/uncertainty_model.py`
- chemistry reports and live features feed `ai_control/chemistry_governor.py`
- the chemistry governor feeds resource allocation and policy generation
- `scripts/live_dashboard.py` renders the same chemistry report in the GUI

Ownership stays explicit:

- chemistry diagnoses interface plausibility
- ML still owns uncertainty estimation
- AI control still owns actions and budgets
- the dashboard still only renders structured payloads

## Interfaces

Key contracts:

- `ResidueChemistryDescriptor`
- `BeadChemistryAssignment`
- `ProteinChemistrySummary`
- `ProteinChemistryModel`
- `ChemistryPairSignal`
- `ChemistryInterfaceReport`
- `ChemistryInterfaceAnalyzer`
- `LiveFeatureVector`
- `LiveFeatureEncoder`
- `ChemistryControlGuidance`
- `ChemistryAwareGovernor`

## Validation Strategy

Current validation for this phase:

- residue-library and inferred-chemistry unit tests
- cross-interface chemistry-scoring unit tests
- live-feature encoding tests
- chemistry-aware uncertainty tests
- chemistry-aware executive-control tests
- full-suite regression after wiring into the live dashboard path

## Risks and Boundaries

- this is not a full physical chemistry engine
- chemistry descriptors are bounded semantic priors, not a force-field replacement
- chemistry-aware control must remain additive and explicit
- future chemistry tuning must be benchmarked across multiple proteins rather than overfit to one scenario

## Next Work

- tune chemistry priors across additional protein benchmarks
- feed chemistry-quality series into repeated scientific-validation plots
- connect chemistry quality to adaptive shadow tuning
- compare chemistry trajectories against known interfaces beyond `spike_ace2`
