# Encounter Complex Dashboard Use Case

## Classification

- scenario definition: `[adapted]`
- dashboard coupling to graph/qcloud/controller diagnostics: `[hybrid]`

## Problem Being Solved

The live dashboard now runs a concrete coarse-grained encounter-complex
assembly problem instead of a generic toy trajectory. Two bonded trimers begin
in a separated encounter state, and the simulation attempts to form a stable
cross-interface complex through:

- one dominant functional interface pair
- auxiliary packing contacts between the two trimers
- adaptive graph bridging that reflects cross-complex connectivity as capture occurs

## Mathematical Picture

This use case monitors a reduced assembly objective:

- primary interface closure
  - `d_if(t) = ||r_i(t) - r_j(t)||`
- cross-complex contact count
  - `C(t) = sum 1[ ||r_a(t) - r_b(t)|| <= r_contact ]`
- graph bridge count
  - `G(t) = number of active graph edges spanning complexes A and B`
- bounded assembly score
  - `S(t) = 0.55 S_d + 0.30 S_c + 0.15 S_g`

where:

- `S_d` rewards interface closure relative to the initial encounter geometry
- `S_c` rewards accumulating physical contacts
- `S_g` rewards explicit graph-level bridging without letting the graph become the only source of truth

## Why This Is A Better Live Problem

- the objective is physically interpretable
- the dashboard can now show a genuine search -> encounter -> docking -> bound story
- memory, qcloud, ML, and controller layers are still visible, but they are now attached to a real objective rather than a free-floating demo

## Current Scope And Honesty Notes

- this is still an early coarse-grained benchmark, not a publication-grade biomolecular model
- the force field is a simple reduced-unit setup intended to make assembly behavior inspectable
- the scenario is appropriate for live development, integration tests, and architectural demos
- future validation can replace this with richer biomolecular targets without changing the dashboard contracts
