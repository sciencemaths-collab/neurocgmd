# Force Field Foundation Architecture

## Section 4 Scope

Section 4 introduces the first established physical interaction layer for the
platform. This section is classified as `[established]` because the parameter
structures and baseline interaction forms are standard coarse-grained molecular
simulation components rather than novel architecture.

## Mathematical Role

The Section 4 energy foundation is:

`E_total = E_bonded + E_nonbonded`

With the current baseline terms:

- harmonic bonds
  - `E_bond = sum_(i,j in E_fixed) 0.5 * k_(ij) * (r_(ij) - r0_(ij))^2`
- Lennard-Jones nonbonded pairs
  - `E_LJ = sum_(i<j) 4 * epsilon_(ij) * [ (sigma_(ij)/r_(ij))^12 - (sigma_(ij)/r_(ij))^6 ]`

The corresponding nonbonded force is the gradient of the Lennard-Jones term.

## Architectural Boundaries

### Included in Section 4

- force-field parameter contracts keyed by topology bead types
- bonded energy evaluation
- nonbonded energy evaluation
- nonbonded force evaluation
- energy/force tests with explicit shape checks

### Explicitly Excluded from Section 4

- adaptive graph corrections
- learned residual force terms
- quantum-cloud refinement
- thermostat and integrator logic
- angle, torsion, constraint-group, and advanced solvent terms

## Design Principles

- structural truth lives in `topology/`
- parameter truth lives in `forcefields/`
- energy and force evaluation live in `physics/`
- parameter lookup is symmetric in bead-type pairs
- bond-local overrides are allowed, but only as structural placeholders until richer parameter policies exist

## Validation Thinking

We test Section 4 by checking:

- force-field parameter lookup symmetry
- zero harmonic energy at equilibrium bond length
- positive harmonic energy away from equilibrium
- Lennard-Jones energy evaluation for nonbonded pairs
- equal-and-opposite nonbonded forces and correct output shape

