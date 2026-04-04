#!/usr/bin/env python3
"""
Full NeuroCGMD Pipeline Demo
==============================
Shows how every module feeds into the next in a real simulation.

The pipeline:
  1. BUILD        → core state + topology + forcefield
  2. INTEGRATE    → BAOAB Langevin dynamics (positions + velocities)
  3. FORCES       → neighbor list + LJ + bonds → force vectors
  4. GRAPH        → adaptive connectivity from particle distances
  5. PLASTICITY   → STDP updates edge weights from activity
  6. ML           → neural model learns residual corrections online
  7. UNCERTAINTY  → ensemble disagreement gates expensive operations
  8. QCLOUD       → adaptive refinement on high-uncertainty regions
  9. CONTROL      → AI executive reads everything, recommends actions
  10. VALIDATE    → statistical mechanics checks on the trajectory
  11. METADYNAMICS → bias potential helps cross energy barriers

Each step prints what it receives and what it produces.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math import sqrt
from random import Random

# Core
from core.state import (
    EnsembleKind, ParticleState, SimulationState, StateProvenance,
    ThermodynamicState, UnitSystem,
)
from core.state_registry import SimulationStateRegistry
from core.types import BeadId, FrozenMetadata, SimulationId, StateId

# Topology & Forcefields
from topology.beads import Bead, BeadRole, BeadType
from topology.bonds import Bond
from topology.system_topology import SystemTopology
from forcefields.base_forcefield import BaseForceField, BondParameter, NonbondedParameter

# Physics
from physics.forces.composite import BaselineForceEvaluator
from physics.neighbor_list import NeighborListBuilder, AcceleratedNonbondedForceModel
from physics.forces.nonbonded_forces import LennardJonesNonbondedForceModel

# Integrator
from integrators.baoab import BAOABIntegrator

# Graph
from graph.edge_models import DynamicEdgeKind, DynamicEdgeState
from graph.graph_manager import ConnectivityGraph
from graph.message_passing import MessagePassingGraphUpdater

# Plasticity
from plasticity.engine import PlasticityEngine
from plasticity.stdp import STDPRule, HomeostaticScaling
from plasticity.traces import PairTraceState

# ML
from ml.neural_residual_model import NeuralResidualModel
from ml.residual_model import ResidualTarget, ResidualAugmentedForceEvaluator
from ml.ensemble_uncertainty import EnsembleUncertaintyModel

# QCloud
from qcloud.cloud_state import ParticleForceDelta, QCloudCorrection, RefinementRegion, RegionTriggerKind
from qcloud.adaptive_refinement import AdaptiveRefinementController

# AI Control
from ai_control.stability_monitor import StabilityMonitor
from ai_control.resource_allocator import ResourceAllocator
from ai_control.policies import DeterministicExecutivePolicy
from ai_control.controller import ExecutiveController

# Sampling
from sampling.enhanced_sampling import DistanceCV, MetadynamicsEngine

# Validation
from validation.statistical_mechanics import block_average, compute_autocorrelation, check_equipartition


def main():
    print()
    print("=" * 70)
    print("  NeuroCGMD Full Pipeline — How Everything Connects")
    print("=" * 70)

    # ==================================================================
    # STEP 1: BUILD — core state, topology, forcefield
    # ==================================================================
    print("\n┌─ STEP 1: BUILD ─────────────────────────────────────────────┐")
    print("│  Creates: SimulationState, SystemTopology, BaseForceField   │")
    print("└─────────────────────────────────────────────────────────────┘")

    # A 6-particle protein-like chain: backbone + two "hydrophobic" contacts
    positions = (
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.5, 0.0),
        (3.0, 0.0, 0.0), (4.0, 0.5, 0.0), (5.0, 0.0, 0.0),
    )
    state = SimulationState(
        units=UnitSystem.md_nano(),
        particles=ParticleState(
            positions=positions,
            masses=(1.0,) * 6,
            velocities=((0.05, 0.02, 0.0), (-0.03, 0.01, 0.0),
                        (0.01, -0.04, 0.0), (-0.02, 0.03, 0.0),
                        (0.04, -0.01, 0.0), (-0.01, -0.02, 0.0)),
            labels=("H0", "P1", "P2", "H3", "P4", "H5"),
        ),
        thermodynamics=ThermodynamicState(
            ensemble=EnsembleKind.NVT,
            target_temperature=1.0,
            friction_coefficient=2.0,
        ),
        provenance=StateProvenance(
            simulation_id=SimulationId("demo"),
            state_id=StateId("s0"),
            parent_state_id=None,
            created_by="pipeline_demo",
            stage="initialization",
        ),
    )

    bead_types = (
        BeadType(name="H", role=BeadRole.STRUCTURAL, description="Hydrophobic"),
        BeadType(name="P", role=BeadRole.FUNCTIONAL, description="Polar"),
    )
    beads = (
        Bead(bead_id=BeadId("b0"), particle_index=0, bead_type="H", label="H0"),
        Bead(bead_id=BeadId("b1"), particle_index=1, bead_type="P", label="P1"),
        Bead(bead_id=BeadId("b2"), particle_index=2, bead_type="P", label="P2"),
        Bead(bead_id=BeadId("b3"), particle_index=3, bead_type="H", label="H3"),
        Bead(bead_id=BeadId("b4"), particle_index=4, bead_type="P", label="P4"),
        Bead(bead_id=BeadId("b5"), particle_index=5, bead_type="H", label="H5"),
    )
    bonds = (
        Bond(0, 1, equilibrium_distance=1.0, stiffness=50.0, bond_id="bb01"),
        Bond(1, 2, equilibrium_distance=1.1, stiffness=50.0, bond_id="bb12"),
        Bond(2, 3, equilibrium_distance=1.1, stiffness=50.0, bond_id="bb23"),
        Bond(3, 4, equilibrium_distance=1.1, stiffness=50.0, bond_id="bb34"),
        Bond(4, 5, equilibrium_distance=1.0, stiffness=50.0, bond_id="bb45"),
    )
    topology = SystemTopology(
        system_id="chain6", bead_types=bead_types, beads=beads, bonds=bonds,
    )
    forcefield = BaseForceField(
        name="chain_ff",
        bond_parameters=(
            BondParameter("H", "P", equilibrium_distance=1.0, stiffness=50.0),
            BondParameter("P", "P", equilibrium_distance=1.1, stiffness=50.0),
            BondParameter("H", "H", equilibrium_distance=1.0, stiffness=50.0),
        ),
        nonbonded_parameters=(
            NonbondedParameter("H", "H", sigma=0.8, epsilon=0.3, cutoff=3.0),
            NonbondedParameter("H", "P", sigma=0.6, epsilon=0.1, cutoff=3.0),
            NonbondedParameter("P", "P", sigma=0.5, epsilon=0.05, cutoff=3.0),
        ),
    )

    print(f"  State: {state.particle_count} particles, step={state.step}")
    print(f"  Topology: {len(topology.bonds)} bonds")
    print(f"  Forcefield: {len(forcefield.bond_parameters)} bond params, "
          f"{len(forcefield.nonbonded_parameters)} nonbonded params")
    print(f"  → feeds into: INTEGRATE, FORCES, GRAPH, PLASTICITY, ML, CONTROL")

    # ==================================================================
    # STEP 2: FORCES — evaluate baseline forces
    # ==================================================================
    print("\n┌─ STEP 2: FORCES ────────────────────────────────────────────┐")
    print("│  Input:  SimulationState + Topology + ForceField            │")
    print("│  Output: ForceEvaluation (force vectors + potential energy)  │")
    print("└─────────────────────────────────────────────────────────────┘")

    evaluator = BaselineForceEvaluator()
    fe = evaluator.evaluate(state, topology, forcefield)
    max_force = max(sqrt(sum(c*c for c in f)) for f in fe.forces)
    print(f"  Potential energy: {fe.potential_energy:.4f}")
    print(f"  Component energies: {dict(fe.component_energies)}")
    print(f"  Max force magnitude: {max_force:.4f}")
    print(f"  → feeds into: INTEGRATE (drives particle motion)")
    print(f"  → feeds into: ML (baseline for residual learning)")
    print(f"  → feeds into: QCLOUD (baseline for corrections)")

    # ==================================================================
    # STEP 3: INTEGRATE — BAOAB Langevin dynamics
    # ==================================================================
    print("\n┌─ STEP 3: INTEGRATE (BAOAB) ────────────────────────────────┐")
    print("│  Input:  State + Forces + Thermostat (T, friction)          │")
    print("│  Output: New positions, velocities (proper Boltzmann dist)  │")
    print("└─────────────────────────────────────────────────────────────┘")

    integrator = BAOABIntegrator(
        time_step=0.002, friction_coefficient=2.0,
        assume_reduced_units=True, random_seed=42,
    )

    registry = SimulationStateRegistry(
        created_by="pipeline_demo",
        simulation_id=state.provenance.simulation_id,
    )
    registry.register_state(state)

    energy_history = []
    velocity_history = []
    current = state

    # Run 100 MD steps
    for step_i in range(100):
        result = integrator.step(current, topology, forcefield, evaluator)
        current = registry.derive_state(
            current, particles=result.particles,
            time=result.time, step=result.step,
            potential_energy=result.potential_energy,
            observables=result.observables, stage="integration",
        )
        if result.potential_energy is not None:
            energy_history.append(result.potential_energy)
        velocity_history.append(current.particles.velocities)

    print(f"  Ran 100 BAOAB steps (dt=0.002, T=1.0, gamma=2.0)")
    print(f"  Final step={current.step}, time={current.time:.3f}")
    print(f"  Energy: {energy_history[0]:.4f} → {energy_history[-1]:.4f}")
    print(f"  → feeds into: GRAPH (new positions define connectivity)")
    print(f"  → feeds into: VALIDATE (trajectory for statistical checks)")
    print(f"  → feeds into: PLASTICITY (activity from position changes)")

    # ==================================================================
    # STEP 4: GRAPH — adaptive connectivity from positions
    # ==================================================================
    print("\n┌─ STEP 4: GRAPH (GNN Message Passing) ──────────────────────┐")
    print("│  Input:  Particle positions + Topology                      │")
    print("│  Output: ConnectivityGraph (weighted edges, adaptive)       │")
    print("└─────────────────────────────────────────────────────────────┘")

    # Build initial graph from topology bonds + distance-based adaptive edges
    initial_edges = []
    for bond in topology.bonds:
        d = sqrt(sum(
            (current.particles.positions[bond.particle_index_a][ax]
             - current.particles.positions[bond.particle_index_b][ax]) ** 2
            for ax in range(3)
        ))
        initial_edges.append(DynamicEdgeState(
            source_index=bond.particle_index_a,
            target_index=bond.particle_index_b,
            kind=DynamicEdgeKind.STRUCTURAL_LOCAL,
            weight=1.0, distance=d,
            created_step=0, last_updated_step=current.step,
        ))

    # Add adaptive edges between non-bonded H-H pairs (hydrophobic attraction)
    bonded_pairs = {(b.particle_index_a, b.particle_index_b) for b in topology.bonds}
    bonded_pairs |= {(b, a) for a, b in bonded_pairs}
    h_indices = [0, 3, 5]  # H beads
    for i, a in enumerate(h_indices):
        for b in h_indices[i+1:]:
            if (a, b) not in bonded_pairs:
                d = sqrt(sum(
                    (current.particles.positions[a][ax]
                     - current.particles.positions[b][ax]) ** 2
                    for ax in range(3)
                ))
                initial_edges.append(DynamicEdgeState(
                    source_index=a, target_index=b,
                    kind=DynamicEdgeKind.ADAPTIVE_LONG_RANGE,
                    weight=0.5, distance=d,
                    created_step=0, last_updated_step=current.step,
                ))

    graph = ConnectivityGraph(
        particle_count=6, step=current.step, edges=tuple(initial_edges),
    )

    # Run GNN message passing to update edge weights
    gnn = MessagePassingGraphUpdater(layers=2, weight_update_rate=0.1, message_dim=8)
    graph = gnn.update(current, topology, graph)

    print(f"  Structural edges: {sum(1 for e in graph.active_edges() if e.kind == DynamicEdgeKind.STRUCTURAL_LOCAL)}")
    print(f"  Adaptive edges: {sum(1 for e in graph.active_edges() if e.kind != DynamicEdgeKind.STRUCTURAL_LOCAL)}")
    print(f"  Edge weights: {[round(e.weight, 3) for e in graph.active_edges()]}")
    print(f"  → feeds into: PLASTICITY (graph edges get strengthened/weakened)")
    print(f"  → feeds into: CONTROL (edge count signals stability)")

    # ==================================================================
    # STEP 5: PLASTICITY — STDP updates edge weights
    # ==================================================================
    print("\n┌─ STEP 5: PLASTICITY (STDP + Homeostatic) ──────────────────┐")
    print("│  Input:  ConnectivityGraph + activity signals               │")
    print("│  Output: Updated graph (edges strengthened/weakened/pruned)  │")
    print("└─────────────────────────────────────────────────────────────┘")

    # Activity signals: high for H-H pairs (they're "coactive"), low for others
    activity_signals = {}
    for edge in graph.active_edges():
        pair = edge.normalized_pair()
        a_type = topology.beads[pair[0]].bead_type
        b_type = topology.beads[pair[1]].bead_type
        # H-H pairs are highly active, others moderate
        signal = 0.9 if a_type == "H" and b_type == "H" else 0.3
        activity_signals[pair] = signal

    plasticity = PlasticityEngine(trace_decay=0.8)
    plasticity_result = plasticity.update(
        current, topology, graph,
        activity_signals=activity_signals,
    )
    graph = plasticity_result.graph
    traces = plasticity_result.traces

    # Apply STDP
    stdp = STDPRule()
    graph = stdp.apply(graph, traces, current_step=current.step)

    # Apply homeostatic scaling
    homeo = HomeostaticScaling(target_mean_weight=0.6)
    graph = homeo.apply(graph)

    weights_before = {e.normalized_pair(): round(e.weight, 3) for e in initial_edges}
    weights_after = {e.normalized_pair(): round(e.weight, 3) for e in graph.active_edges()}
    print(f"  Traces: {len(traces)} pair traces computed")
    print(f"  H-H edge (0,3): {weights_before.get((0,3), '?')} → {weights_after.get((0,3), '?')}")
    print(f"  H-H edge (0,5): {weights_before.get((0,5), '?')} → {weights_after.get((0,5), '?')}")
    print(f"  Backbone edge (0,1): {weights_before.get((0,1), '?')} → {weights_after.get((0,1), '?')}")
    print(f"  → feeds into: ML (plasticity-weighted sampling priority)")
    print(f"  → feeds into: CONTROL (connectivity health signal)")

    # ==================================================================
    # STEP 6: ML — neural residual model learns corrections
    # ==================================================================
    print("\n┌─ STEP 6: ML (Neural Residual Model) ───────────────────────┐")
    print("│  Input:  ForceEvaluation + observed energy corrections      │")
    print("│  Output: Predicted residual (additive energy + force delta) │")
    print("└─────────────────────────────────────────────────────────────┘")

    ml_model = NeuralResidualModel(hidden_sizes=(32, 16), learning_rate=0.003, random_seed=42)

    # Simulate "observed" corrections (e.g., from QCloud or higher-level theory)
    rng = Random(42)
    for i in range(40):
        target = ResidualTarget(
            state_id=StateId(f"train-{i}"),
            energy_delta=0.15 + rng.gauss(0, 0.02),  # true correction ~0.15
        )
        ml_model.observe(target)

    # Predict on current state
    fe_current = evaluator.evaluate(current, topology, forcefield)
    prediction = ml_model.predict(current, fe_current)

    print(f"  Trained on 40 observations (target ~0.15 energy correction)")
    print(f"  Predicted energy delta: {prediction.predicted_energy_delta:.4f}")
    print(f"  Confidence: {prediction.confidence:.3f}")
    print(f"  Force corrections: {len(prediction.force_deltas)} particles")
    print(f"  → feeds into: FORCES (augments baseline with learned correction)")
    print(f"  → feeds into: UNCERTAINTY (prediction feeds disagreement check)")
    print(f"  → feeds into: INTEGRATE (corrected forces drive better dynamics)")

    # Use ML-augmented force evaluator for next MD steps
    augmented_evaluator = ResidualAugmentedForceEvaluator(
        base_force_evaluator=evaluator,
        residual_model=ml_model,
    )
    fe_augmented = augmented_evaluator.evaluate(current, topology, forcefield)
    print(f"  Baseline energy: {fe_current.potential_energy:.4f}")
    print(f"  Augmented energy: {fe_augmented.potential_energy:.4f} "
          f"(+{fe_augmented.potential_energy - fe_current.potential_energy:.4f} ML correction)")

    # ==================================================================
    # STEP 7: UNCERTAINTY — ensemble disagreement
    # ==================================================================
    print("\n┌─ STEP 7: UNCERTAINTY (Ensemble Disagreement) ──────────────┐")
    print("│  Input:  ML prediction + optional live features             │")
    print("│  Output: Uncertainty estimate + trigger_qcloud flag         │")
    print("└─────────────────────────────────────────────────────────────┘")

    eum = EnsembleUncertaintyModel(
        ensemble_size=5, trigger_threshold=0.4, random_seed=42,
    )
    # Train ensemble on same targets (with bootstrap diversity)
    rng2 = Random(42)
    for i in range(40):
        target = ResidualTarget(
            state_id=StateId(f"ens-{i}"),
            energy_delta=0.15 + rng2.gauss(0, 0.02),
        )
        eum.observe(target)

    estimate = eum.estimate(prediction)
    print(f"  Energy uncertainty: {estimate.energy_uncertainty:.4f}")
    print(f"  Force uncertainty: {estimate.force_uncertainty:.4f}")
    print(f"  Total uncertainty: {estimate.total_uncertainty:.4f}")
    print(f"  Trigger QCloud? {estimate.trigger_qcloud}")
    print(f"  → feeds into: QCLOUD (if trigger=True, run expensive refinement)")
    print(f"  → feeds into: CONTROL (uncertainty level affects resource allocation)")

    # ==================================================================
    # STEP 8: QCLOUD — adaptive refinement
    # ==================================================================
    print("\n┌─ STEP 8: QCLOUD (Adaptive Refinement) ─────────────────────┐")
    print("│  Input:  High-uncertainty region + base correction          │")
    print("│  Output: Richardson-extrapolated correction + error bounds  │")
    print("└─────────────────────────────────────────────────────────────┘")

    # Define a refinement region around H-H contact
    region = RefinementRegion(
        region_id="hh_contact",
        state_id=current.provenance.state_id,
        particle_indices=(0, 3, 5),
        trigger_kinds=(RegionTriggerKind.ADAPTIVE_EDGE,),
    )
    # Simulate a QCloud correction (normally from quantum calculation)
    correction = QCloudCorrection(
        region_id="hh_contact",
        method_label="shadow_lj_coulomb",
        energy_delta=0.08,
        force_deltas=(
            ParticleForceDelta(particle_index=0, delta_force=(0.01, -0.005, 0.0)),
            ParticleForceDelta(particle_index=3, delta_force=(-0.008, 0.003, 0.0)),
            ParticleForceDelta(particle_index=5, delta_force=(-0.002, 0.002, 0.0)),
        ),
        confidence=0.7,
    )

    arc = AdaptiveRefinementController(target_accuracy=0.01)
    refinement = arc.refine(current, region, correction)

    print(f"  Region: particles {region.particle_indices}")
    print(f"  Base correction: energy_delta={correction.energy_delta:.4f}")
    print(f"  Refined correction: energy_delta={refinement.correction.energy_delta:.4f}")
    print(f"  Estimated error: {refinement.estimated_error:.4f}")
    print(f"  Converged? {refinement.converged}")
    print(f"  → feeds into: ML (QCloud corrections become training targets)")
    print(f"  → feeds into: FORCES (corrections added to baseline)")
    print(f"  → feeds into: CONTROL (convergence status affects decisions)")

    # Feed QCloud correction into ML as training target
    ml_target_from_qcloud = ResidualTarget.from_corrections(
        state_id=current.provenance.state_id,
        corrections=(correction,),
        source_label="qcloud",
    )
    ml_model.observe(ml_target_from_qcloud)
    print(f"  ML model updated with QCloud target (states={ml_model.trained_state_count()})")

    # ==================================================================
    # STEP 9: CONTROL — AI executive reads everything
    # ==================================================================
    print("\n┌─ STEP 9: CONTROL (AI Executive) ───────────────────────────┐")
    print("│  Input:  State + graph + uncertainty + ML + QCloud          │")
    print("│  Output: Prioritized recommendations (non-mutating)        │")
    print("└─────────────────────────────────────────────────────────────┘")

    controller = ExecutiveController()
    decision = controller.decide(
        state=current,
        graph=graph,
        uncertainty_estimate=estimate,
    )

    print(f"  Stability: {decision.assessment.level.value}")
    print(f"  Monitoring intensity: {decision.allocation.monitoring_intensity.value}")
    print(f"  Actions recommended: {len(decision.actions)}")
    for action in decision.actions[:3]:
        print(f"    - [{action.kind.value}] {action.summary} (priority={action.priority})")
    print(f"  → feeds into: INTEGRATE (controls time step, temperature)")
    print(f"  → feeds into: QCLOUD (controls which regions to refine)")
    print(f"  → feeds into: PLASTICITY (controls rewiring aggressiveness)")

    # ==================================================================
    # STEP 10: VALIDATE — statistical mechanics checks
    # ==================================================================
    print("\n┌─ STEP 10: VALIDATE (Statistical Mechanics) ────────────────┐")
    print("│  Input:  Energy trajectory + velocity history               │")
    print("│  Output: Block average, autocorrelation, equipartition      │")
    print("└─────────────────────────────────────────────────────────────┘")

    ba = block_average(tuple(energy_history))
    ac = compute_autocorrelation(tuple(energy_history))
    eq = check_equipartition(
        tuple(velocity_history[-50:]),
        current.particles.masses,
        current.thermodynamics.target_temperature,
    )

    print(f"  Energy: mean={ba.mean:.4f} ± {ba.standard_error:.4f}")
    print(f"  Autocorrelation time: {ac.integrated_autocorrelation_time:.2f} steps")
    print(f"  Effective samples: {ac.effective_sample_size:.1f} / {len(energy_history)}")
    print(f"  Equipartition: chi2={eq.chi_squared:.2f}, p={eq.p_value:.4f}, passed={eq.passed}")
    print(f"  → feeds into: CONTROL (validation failures trigger alerts)")
    print(f"  → feeds into: INTEGRATE (if not equilibrated, adjust parameters)")

    # ==================================================================
    # STEP 11: METADYNAMICS — bias helps cross barriers
    # ==================================================================
    print("\n┌─ STEP 11: METADYNAMICS (Enhanced Sampling) ────────────────┐")
    print("│  Input:  Collective variable (e.g., H0-H3 distance)        │")
    print("│  Output: Bias potential that fills energy minima            │")
    print("└─────────────────────────────────────────────────────────────┘")

    cv = DistanceCV(particle_a=0, particle_b=3, name="H0_H3_distance")
    meta = MetadynamicsEngine(
        collective_variables=(cv,),
        hill_height=0.05, hill_width=0.3,
        well_tempered=True, bias_temperature=5.0,
    )

    # Deposit hills along a short trajectory
    for i in range(5):
        hill = meta.deposit_hill(current)
    bias = meta.compute_bias_energy(current)
    cv_val = cv.compute(current)

    print(f"  CV (H0-H3 distance): {cv_val:.4f}")
    print(f"  Hills deposited: {len(meta._hills)}")
    print(f"  Bias energy at current state: {bias:.4f}")
    print(f"  → feeds into: FORCES (bias forces push system over barriers)")
    print(f"  → feeds into: VALIDATE (reweighting for unbiased observables)")

    # ==================================================================
    # THE FULL LOOP
    # ==================================================================
    print("\n" + "=" * 70)
    print("  HOW THEY ALL CONNECT — THE FEEDBACK LOOPS")
    print("=" * 70)
    print("""
  ┌─────────────┐     forces      ┌──────────────┐
  │   FORCES    │────────────────→│  INTEGRATE   │
  │ (bonds+LJ)  │←───corrections──│  (BAOAB)     │
  └──────┬──────┘                 └──────┬───────┘
         │                               │
         │ baseline                      │ positions
         ▼                               ▼
  ┌──────────────┐   predictions  ┌──────────────┐
  │     ML       │──────────────→│    GRAPH     │
  │ (neural net) │←──targets──┐  │ (GNN msg-pass)│
  └──────┬───────┘            │  └──────┬───────┘
         │                    │         │
         │ prediction         │         │ weighted edges
         ▼                    │         ▼
  ┌──────────────┐            │  ┌──────────────┐
  │ UNCERTAINTY  │            │  │ PLASTICITY   │
  │ (ensemble)   │            │  │ (STDP+homeo) │
  └──────┬───────┘            │  └──────┬───────┘
         │                    │         │
         │ trigger?           │         │ adapted stiffness
         ▼                    │         ▼
  ┌──────────────┐            │  ┌──────────────┐
  │   QCLOUD     │────────────┘  │   CONTROL    │←─── all signals
  │ (refinement) │               │ (AI executive)│
  └──────────────┘               └──────┬───────┘
                                        │
                        recommendations │
                                        ▼
                                 ┌──────────────┐
                                 │  VALIDATE    │
                                 │ (stat mech)  │
                                 └──────────────┘

  ┌──────────────┐
  │ METADYNAMICS │ ←→ FORCES (bias potential to cross barriers)
  └──────────────┘

  Key feedback loops:
    1. FORCES → INTEGRATE → new positions → FORCES (MD loop)
    2. FORCES → ML → corrected FORCES (online learning loop)
    3. UNCERTAINTY → QCLOUD → ML targets (active learning loop)
    4. GRAPH → PLASTICITY → spring stiffness → FORCES (adaptive topology)
    5. VALIDATE → CONTROL → all parameters (self-monitoring loop)
    6. METADYNAMICS → FORCES → new CV values → METADYNAMICS (bias filling)
""")

    print("  Summary of data produced by this demo:")
    print(f"    - 100 MD steps with proper Langevin thermostat")
    print(f"    - {len(graph.active_edges())} adaptive graph edges (GNN-scored)")
    print(f"    - {len(traces)} plasticity traces (STDP-updated)")
    print(f"    - {ml_model.trained_state_count()} ML training observations")
    print(f"    - Ensemble uncertainty: {estimate.total_uncertainty:.4f}")
    print(f"    - QCloud refinement error: {refinement.estimated_error:.4f}")
    print(f"    - {len(decision.actions)} AI control recommendations")
    print(f"    - Energy equilibrium: mean={ba.mean:.2f} ± {ba.standard_error:.4f}")
    print(f"    - {len(meta._hills)} metadynamics hills deposited")
    print()


if __name__ == "__main__":
    main()
