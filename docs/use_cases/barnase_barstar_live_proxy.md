# Barnase-Barstar Live Proxy Scenario

## Classification

- experimental target: `[established]`
- coarse-grained live proxy: `[hybrid]`

## What This Scenario Is

This is the first live-dashboard scenario tied to a real biological benchmark
with known answers. It is not a fully faithful atomistic barnase-barstar model.
Instead, it is a coarse-grained proxy that keeps the benchmark identity explicit
while giving the GUI a real target to move toward.

## Proxy Design

The live model uses two three-bead protein proxies:

- barnase proxy
  - core bead
  - basic electrostatic patch
  - recognition-loop bead
- barstar proxy
  - core bead
  - helix-face bead
  - acidic electrostatic patch

The dominant proxy interactions are:

- strong `basic-acidic` attraction to mimic electrostatic steering
- moderate `loop-helix` attraction to mimic interface locking
- bounded internal harmonic bonds that preserve each partner as a compact unit

## What The GUI Shows

- current docking phase
- current proxy interface gap
- current cross-protein contact count
- current graph bridge count
- current proxy assembly score
- known experimental answers from the real benchmark:
  - bound complex `1BRS`
  - free structures `1A2P` and `1A19`
  - target `k_on`, `k_off`, and `K_d`

## Honesty Boundaries

- the live run is a proxy benchmark, not a solved barnase-barstar simulator
- the GUI now keeps the real answer set visible so we can compare later without forgetting the target
- kinetic agreement is not claimed yet
- structural agreement beyond coarse docking phase is not claimed yet

## Why This Step Matters

This gives us a real benchmark identity now, instead of waiting until the entire
physics stack is complete. From here, later sections can add:

- explicit structural comparison to `1BRS`
- interface contact-map comparison
- event-counting for association and dissociation
- estimated `k_on`, `k_off`, and `K_d`
- more realistic coarse-grained parameterization
