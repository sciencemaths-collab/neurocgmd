# Barnase-Barstar Reference Case

## Classification

- biological benchmark target: `[established]`
- use inside this repository: `[adapted]`

## Why This Is The Right First Real Benchmark

Barnase-barstar is a real biological protein-protein association problem with a
known answer set that is unusually strong for simulation benchmarking:

- a solved bound complex structure
- experimentally observed free partner structures
- measured `k_on`, `k_off`, and `K_d`
- a well-established electrostatic steering mechanism

That combination makes it better than a generic toy complex and much safer than
jumping immediately to a giant assembly where the ground truth is fuzzier.

## Ground Truth We Can Compare Against Later

Structural anchors:

- bound complex: `1BRS`
- alternate bound complex: `1BGS`
- free barnase: `1A2P`
- free barstar: `1A19`

Kinetic/thermodynamic anchors at pH 8:

- `k_on = 6.0 x 10^8 M^-1 s^-1`
- `k_off = 8.0 x 10^-6 s^-1`
- `K_d = 1.3 x 10^-14 M`

Mechanistic anchor:

- the system proceeds through a long-range electrostatically steered early encounter phase before precise docking

## What We Should Compare Later

- bound-state structural agreement against `1BRS`
- interface contact recovery
- coarse orientation and docking geometry
- estimated `k_on`, `k_off`, and `K_d`
- whether the model reproduces an encounter-complex regime rather than only instant sticking

## Primary Sources

- Bound complex structure `1BRS`: [RCSB PDB](https://www.rcsb.org/structure/1BRS)
- Alternate bound complex `1BGS`: [RCSB PDB](https://www.rcsb.org/structure/1BGS)
- Free barnase `1A2P`: [RCSB PDB](https://www.rcsb.org/structure/1A2P)
- Free barstar `1A19`: [NCBI Structure](https://www.ncbi.nlm.nih.gov/Structure/pdb/1A19)
- Kinetics and affinity: [Schreiber and Fersht 1993](https://pubmed.ncbi.nlm.nih.gov/8494892/)
- Electrostatic steering mechanism: [Schreiber and Fersht 1996](https://www.nature.com/articles/nsb0596-427)

## Live Proxy Link

The current live dashboard now includes a coarse-grained proxy scenario for this
benchmark in [barnase_barstar_live_proxy.md](/Users/ramonahenry/Documents/1million_lines_brain/docs/use_cases/barnase_barstar_live_proxy.md).
