# Spike-ACE2 Live Proxy Scenario

## Classification

- experimental target: `[established]`
- coarse-grained live proxy: `[hybrid]`

## What This Scenario Is

This is a harder live-dashboard benchmark than the barnase-barstar proxy. It is
still a reduced coarse-grained scenario, but it keeps the real spike-ACE2 answer
set visible while the dashboard streams a larger and more distributed docking
problem.

## Proxy Design

The live model uses two six-bead domain proxies:

- ACE2 proxy
  - core bead
  - alpha1 support bead
  - alpha1 hotspot bead
  - secondary hotspot bead
  - support bead
  - shield-side bead
- spike RBD proxy
  - core bead
  - ridge support bead
  - ridge hotspot bead
  - loop hotspot bead
  - support bead
  - shield-side bead

The dominant proxy interactions are:

- strong `helix-hotspot` attraction to mimic ACE2 alpha1 / RBD ridge recognition
- strong `hotspot-hotspot` attraction to encourage a distributed contact network
- moderate `hotspot-support` attraction to stabilize interface closure
- bounded internal harmonic bonds that preserve each partner's shape

## What The GUI Shows

- current recognition phase
- current proxy interface gap
- current ACE2-spike cross-contact count
- current graph bridge count
- current proxy assembly score
- current atomistic-centroid alignment metrics from local `6M0J`
- current shadow-fidelity deltas comparing baseline vs shadow-corrected force predictions against a benchmark-informed contact-force target
- known experimental answers from the real benchmark:
  - bound complex `6M0J`
  - free references `1R42` and `6VYB`
  - target apparent `K_d`

## Honesty Boundaries

- the live run is a harder proxy benchmark, not a solved spike-ACE2 simulator
- the dashboard keeps the real answer set visible so later work can compare honestly
- the current structure panel uses local atomistic residue-group centroids, not full all-atom trajectory RMSD
- the current shadow-fidelity panel is a benchmark-informed local force/energy comparison, not a claim of full atomistic parity
- structural agreement beyond coarse recognition phase is not claimed yet
- affinity agreement is not claimed yet

## Why This Step Matters

This gives the GUI a harder real-world target now instead of waiting for later
high-fidelity parameterization. From here, future work can add:

- explicit structural comparison to `6M0J`
- hotspot-pair and interface-map comparison
- matched-construct affinity estimation
- better coarse-grained parameterization of the receptor-recognition surface
- later expansion from centroid-group alignment to broader all-atom alignment and richer structural observables
