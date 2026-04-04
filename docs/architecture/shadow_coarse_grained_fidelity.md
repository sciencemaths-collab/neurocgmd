# Shadow Coarse-Grained Fidelity Architecture

## Scope

This post-roadmap phase introduces the first explicit `shadow coarse-grained`
fidelity layer. The goal is not to abandon the repository's coarse-grained
substrate, but to surround it with a mirrored correction cloud that can ingest
trusted scientific parameterizations and local high-fidelity signals while
preserving the platform's own architecture.

Primary classification:

- trusted-source provenance: `[adapted]`
- shadow mapping and mirrored cloud construction: `[proposed novel]`
- trusted-shadow local correction coupling: `[hybrid]`

## Core Idea

We retain:

- a fast coarse-grained body for global dynamics

We add:

- a mirrored `shadow` cloud around selected coarse particles
- provenance-aware trusted parameter sources
- local correction generation from those shadow sites
- observer-side fidelity checks against trusted targets

This means the repository can build on established physics, chemistry, and
parameterization practice without flattening the system into a clone of an
existing engine.

## Mathematical Role

Let:

- `S_c` be the coarse-grained simulation state
- `R_k` be a selected local refinement region
- `M_shadow` be the shadow mapping from bead types to mirrored shadow sites
- `H_k = M_shadow(S_c, R_k)` be the shadow cloud snapshot
- `P_trust` be the trusted parameter set with explicit provenance
- `Delta_shadow = g(H_k, P_trust)` be the local shadow correction

Then:

- `F_total = F_base + Delta_shadow^F`
- `E_total = E_base + Delta_shadow^E`

Observer-side fidelity checks compare:

- baseline coarse outputs
- shadow-corrected outputs
- trusted target outputs

without giving the validation layer ownership of execution.

## Foundation Design

- `forcefields/trusted_sources.py`
  - records which trusted science and parameterizations feed the shadow layer
  - keeps source labels, adaptation notes, and trusted interaction profiles explicit
- `qcloud/shadow_mapping.py`
  - defines how each coarse bead type expands into one or more mirrored shadow sites
- `qcloud/shadow_cloud.py`
  - materializes state-aligned shadow-site snapshots for selected particles or regions
- `qcloud/shadow_correction.py`
  - converts shadow-site interactions plus trusted profiles into a bounded local correction payload
- `validation/fidelity_checks.py`
  - measures whether corrected outputs are actually closer to trusted targets than the baseline

## Core Invariants

- the coarse-grained state remains the canonical dynamical substrate
- trusted science is tracked with provenance rather than copied opaquely
- the shadow cloud is a correction structure, not a replacement state model
- qcloud still applies bounded additive corrections rather than silently replacing the baseline force path
- fidelity checks remain observer-side and do not mutate execution
- novelty claims stay in the architecture and control flow, not in pretending standard science was invented here

## Validation Thinking

We test this phase by checking:

- trusted parameter sources resolve explicitly by label and bead-type pair
- shadow mapping builds mirrored shadow sites deterministically from the canonical coarse state
- shadow corrections can be generated for local refinement regions and coupled through the existing qcloud pathway
- observer-side fidelity reports show when shadow corrections move predictions closer to trusted targets

## Immediate Follow-On Work

- connect trusted targets to real benchmark-specific local regions
- run repeated trajectories and compare baseline vs shadow-corrected fidelity over time
- use the Section 15 optimization layer to profile the cost of shadow correction against the baseline force path
- later plug in external all-atom or QM reference engines through explicit parity/adaptor boundaries rather than bypassing the architecture
