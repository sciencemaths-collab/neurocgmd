# Spike-ACE2 Reference Case

## Classification

- experimental target: `[established]`
- live proxy benchmark: `[hybrid]`

## Real-System Target

This benchmark tracks the interaction between the SARS-CoV-2 spike
receptor-binding domain and human ACE2.

Known reference anchors kept explicit in the repository:

- bound complex: `6M0J`
- alternate bound complex: `6LZG`
- free ACE2 reference: `1R42`
- prefusion spike reference with an RBD-up state: `6VYB`
- reported apparent `K_d`: `1.47 x 10^-8 M`
- local atomistic centroid scaffold from `benchmarks/reference_cases/data/6M0J.pdb`

## Why This Is Harder

Compared with the first barnase-barstar benchmark, this target is harder because:

- the interface is larger and more distributed
- recognition is not dominated by one simple complementary hotspot pair
- geometry matters across a broader surface patch
- later comparison work will need to treat construct and assay dependence honestly

## What Later Comparison Should Measure

- recovered bound orientation against `6M0J`
- hotspot-network closure rather than single-point docking alone
- free-state structural plausibility relative to `1R42` and `6VYB`
- affinity estimates only after a construct-matched kinetics workflow exists

## Honesty Boundary

This file records the real benchmark target. It does not claim the current live
dashboard proxy already reproduces the physical spike-ACE2 system faithfully.

The current structural-comparison layer uses a real local `6M0J` PDB file and
computes residue-group centroids from actual atom coordinates. That is more
honest and stronger than the earlier hand-entered scaffold, but it is still not
full all-atom trajectory simulation or full-complex atom-by-atom RMSD.
