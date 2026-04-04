"""
NeuroCGMD Bridge — direct in-process integration with SPRING.

Provides:
  - spring_to_md()     : convert SPRING pieces/springs → NeuroCGMD state
  - md_to_spring()     : write NeuroCGMD positions back into SPRING pieces
  - MDPoweredSolver    : hybrid solver (BAOAB integration + SPRING local moves)
  - MDSystemAsSPRING   : wrap any NeuroCGMD system as a SPRING Problem
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt, exp
from typing import Any

import numpy as np

from spring.engine import Piece, Spring, Problem, Snapshot, SpringEngine

# NeuroCGMD core
from core.state import (
    EnsembleKind, ParticleState, SimulationState, StateProvenance,
    ThermodynamicState, UnitSystem,
)
from core.state_registry import SimulationStateRegistry
from core.types import BeadId, FrozenMetadata, SimulationId, StateId

# Topology & forcefields
from topology.beads import Bead, BeadRole, BeadType
from topology.bonds import Bond
from topology.system_topology import SystemTopology
from forcefields.base_forcefield import BaseForceField, BondParameter, NonbondedParameter

# Physics
from physics.forces.composite import BaselineForceEvaluator

# Integrators
from integrators.baoab import BAOABIntegrator

# Plasticity
from graph.edge_models import DynamicEdgeKind, DynamicEdgeState
from graph.graph_manager import ConnectivityGraph
from plasticity.engine import PlasticityEngine
from plasticity.stdp import STDPRule, HomeostaticScaling
from plasticity.traces import PairTraceState

# ML
from ml.neural_residual_model import NeuralResidualModel
from ml.residual_model import ResidualTarget, ResidualAugmentedForceEvaluator

# Validation
from validation.statistical_mechanics import block_average, compute_autocorrelation


# ============================================================
# Converters
# ============================================================

def spring_to_md(
    pieces: list[Piece],
    springs: list[Spring],
    *,
    temperature: float = 1.0,
    friction: float = 1.0,
    sim_id: str = "spring-md",
) -> tuple[SimulationState, SystemTopology, BaseForceField]:
    """Convert SPRING pieces/springs into NeuroCGMD objects."""
    n = len(pieces)

    positions = []
    masses = []
    for p in pieces:
        pos = list(p.state.flat) if hasattr(p.state, "flat") else list(p.state)
        while len(pos) < 3:
            pos.append(0.0)
        positions.append(tuple(pos[:3]))
        masses.append(float(p.metadata.get("mass", 1.0)))

    bead_type = BeadType(name="sp", role=BeadRole.GENERIC)
    beads = tuple(
        Bead(bead_id=BeadId(f"p{i}"), particle_index=i, bead_type="sp", label=f"P{i}")
        for i in range(n)
    )
    bonds = tuple(
        Bond(
            s.a, s.b,
            equilibrium_distance=max(float(s.rest_value), 0.05),
            stiffness=float(s.stiffness) * 5.0,
            bond_id=f"s{idx}",
        )
        for idx, s in enumerate(springs)
    )
    topology = SystemTopology(
        system_id=sim_id, bead_types=(bead_type,), beads=beads, bonds=bonds,
    )
    state = SimulationState(
        units=UnitSystem.md_nano(),
        particles=ParticleState(
            positions=tuple(positions),
            masses=tuple(masses),
            velocities=tuple((0.0, 0.0, 0.0) for _ in range(n)),
            labels=tuple(f"P{i}" for i in range(n)),
        ),
        thermodynamics=ThermodynamicState(
            ensemble=EnsembleKind.NVT,
            target_temperature=temperature,
            friction_coefficient=friction,
        ),
        provenance=StateProvenance(
            simulation_id=SimulationId(sim_id),
            state_id=StateId(f"{sim_id}-s0"),
            parent_state_id=None,
            created_by="spring_bridge",
            stage="initialization",
        ),
    )
    forcefield = BaseForceField(
        name="spring_ff",
        bond_parameters=(BondParameter("sp", "sp", equilibrium_distance=1.0, stiffness=20.0),),
        nonbonded_parameters=(NonbondedParameter("sp", "sp", sigma=0.3, epsilon=0.01, cutoff=2.0),),
    )
    return state, topology, forcefield


def md_to_spring(state: SimulationState, pieces: list[Piece], dim: int = 2) -> None:
    """Write NeuroCGMD positions back into SPRING pieces (in-place)."""
    for i, p in enumerate(pieces):
        if i < state.particle_count:
            p.state = np.array(list(state.particles.positions[i][:dim]), dtype=float)


# ============================================================
# MDPoweredSolver
# ============================================================

@dataclass
class MDPoweredSolver:
    """
    Hybrid SPRING solver powered by NeuroCGMD.

    Each iteration:
      1. Convert pieces → NeuroCGMD state
      2. Run BAOAB Langevin MD steps (real thermostat, real forces)
      3. Write positions back to SPRING pieces
      4. Run SPRING local moves (Metropolis discrete optimization)
      5. Train neural ML on energy residuals
      6. Update plasticity graph (STDP + homeostatic)
      7. Feed plasticity weights back into spring stiffness
    """

    max_iterations: int = 100
    md_steps_per_iter: int = 15
    time_step: float = 0.001
    temperature_start: float = 5.0
    temperature_end: float = 0.1
    friction: float = 2.0
    use_ml: bool = True
    use_plasticity: bool = True
    random_seed: int = 42
    verbose: bool = True

    def solve(self, problem: Problem, input_data: Any) -> tuple[Snapshot | None, list[Snapshot]]:
        pieces, springs = problem.shatter(input_data)
        n = len(pieces)
        dim = len(pieces[0].state) if n > 0 else 2

        history: list[Snapshot] = []
        energy_history: list[float] = []

        cooling = (self.temperature_end / max(self.temperature_start, 1e-10)) ** (
            1.0 / max(self.max_iterations, 1)
        )

        # ML model
        ml_model = NeuralResidualModel(
            hidden_sizes=(32, 16), learning_rate=0.002, random_seed=self.random_seed,
        ) if self.use_ml else None

        # Plasticity
        plasticity = PlasticityEngine(trace_decay=0.8) if self.use_plasticity else None
        stdp = STDPRule() if self.use_plasticity else None
        homeo = HomeostaticScaling(target_mean_weight=0.6) if self.use_plasticity else None
        traces: tuple[PairTraceState, ...] = ()
        graph: ConnectivityGraph | None = None

        for iteration in range(self.max_iterations):
            temp = max(self.temperature_start * (cooling ** iteration), 0.01)

            # 1. Convert to MD
            state, topology, forcefield = spring_to_md(
                pieces, springs, temperature=temp, friction=self.friction,
                sim_id=f"spring-iter{iteration}",
            )

            # 2. Build evaluator (with ML augmentation if trained enough)
            base_evaluator = BaselineForceEvaluator()
            evaluator = base_evaluator
            if ml_model and ml_model.trained_state_count() > 10:
                evaluator = ResidualAugmentedForceEvaluator(
                    base_force_evaluator=base_evaluator, residual_model=ml_model,
                )

            # 3. Run BAOAB MD steps
            integrator = BAOABIntegrator(
                time_step=self.time_step, friction_coefficient=self.friction,
                assume_reduced_units=True, random_seed=self.random_seed + iteration,
            )
            registry = SimulationStateRegistry(
                created_by="spring_bridge", simulation_id=state.provenance.simulation_id,
            )
            registry.register_state(state)

            current = state
            md_blew_up = False
            for _ in range(self.md_steps_per_iter):
                result = integrator.step(current, topology, forcefield, evaluator)
                current = registry.derive_state(
                    current, particles=result.particles,
                    time=result.time, step=result.step,
                    potential_energy=result.potential_energy,
                    observables=result.observables, stage="integration",
                )
                pe = result.potential_energy or 0.0
                if abs(pe) > 1e6:
                    md_blew_up = True
                    break
                energy_history.append(pe)

            # 4. Write MD positions back (only if stable)
            if not md_blew_up:
                md_to_spring(current, pieces, dim=dim)

            # 5. SPRING local moves (discrete Metropolis)
            for i, piece in enumerate(pieces):
                moves = problem.local_moves(piece)
                if not moves:
                    continue
                best_state = piece.state.copy()
                best_e = problem.local_energy(piece)
                for s in springs:
                    other = s.b if s.a == i else (s.a if s.b == i else -1)
                    if other >= 0:
                        best_e += problem.coupling_energy(
                            piece if s.a == i else pieces[other],
                            pieces[other] if s.a == i else piece, s,
                        )
                for new_state in moves:
                    old = piece.state.copy()
                    piece.state = new_state
                    new_e = problem.local_energy(piece)
                    for s in springs:
                        other = s.b if s.a == i else (s.a if s.b == i else -1)
                        if other >= 0:
                            new_e += problem.coupling_energy(
                                piece if s.a == i else pieces[other],
                                pieces[other] if s.a == i else piece, s,
                            )
                    if new_e - best_e < 0 or np.random.random() < exp(-(new_e - best_e) / max(temp, 1e-8)):
                        best_state = new_state.copy() if hasattr(new_state, "copy") else new_state
                        best_e = new_e
                    piece.state = old
                pieces[i].state = best_state

            pieces = problem.post_local_solve(pieces)

            # 6. ML training
            if ml_model and iteration > 0 and not md_blew_up:
                md_e = current.potential_energy or 0.0
                spring_e = sum(problem.local_energy(p) for p in pieces) + sum(
                    problem.coupling_energy(pieces[s.a], pieces[s.b], s) for s in springs
                )
                ml_model.observe(ResidualTarget(
                    state_id=StateId(f"iter-{iteration}"),
                    energy_delta=md_e - spring_e,
                ))

            # 7. Plasticity
            if plasticity and n > 1 and not md_blew_up:
                if graph is None:
                    edges = tuple(
                        DynamicEdgeState(
                            s.a, s.b, DynamicEdgeKind.STRUCTURAL_LOCAL,
                            weight=min(1.0, s.stiffness), distance=float(np.linalg.norm(
                                pieces[s.a].state - pieces[s.b].state)),
                            created_step=0, last_updated_step=iteration,
                        )
                        for s in springs if s.a < n and s.b < n
                    )
                    graph = ConnectivityGraph(particle_count=n, step=0, edges=edges)

                activity = {}
                for s in springs:
                    pair = (min(s.a, s.b), max(s.a, s.b))
                    ce = problem.coupling_energy(pieces[s.a], pieces[s.b], s)
                    activity[pair] = min(1.0, max(0.0, 1.0 - ce))

                res = plasticity.update(current, topology, graph,
                                        activity_signals=activity, previous_traces=traces)
                graph, traces = res.graph, res.traces
                if stdp and traces:
                    graph = stdp.apply(graph, traces, current_step=iteration)
                if homeo:
                    graph = homeo.apply(graph)

                # Feed plasticity weights back to springs
                for s in springs:
                    pair = (min(s.a, s.b), max(s.a, s.b))
                    for edge in graph.active_edges():
                        if edge.normalized_pair() == pair:
                            s.stiffness = max(0.01, s.stiffness * (0.9 + 0.2 * edge.weight))
                            break

            # Snapshot
            e_local = sum(problem.local_energy(p) for p in pieces)
            e_coupling = sum(problem.coupling_energy(pieces[s.a], pieces[s.b], s) for s in springs)
            snap = Snapshot(
                pieces=[p.copy() for p in pieces], springs=list(springs),
                energy_local=e_local, energy_coupling=e_coupling,
                energy_total=e_local + e_coupling, iteration=iteration,
            )
            history.append(snap)

            if self.verbose and iteration % max(1, self.max_iterations // 15) == 0:
                ml_info = f"  ml={ml_model.trained_state_count()}" if ml_model else ""
                ginfo = f"  edges={len(graph.active_edges())}" if graph else ""
                blow = "  [!]" if md_blew_up else ""
                print(f"  iter {iteration:4d} | E={snap.energy_total:10.4f}  temp={temp:.3f}"
                      f"{ml_info}{ginfo}{blow}")

        # Validation
        if self.verbose and len(energy_history) >= 20:
            ba = block_average(tuple(energy_history[-200:]))
            print(f"  Validation: mean_E={ba.mean:.4f} SE={ba.standard_error:.4f}")

        best = min(history, key=lambda s: s.energy_total) if history else None
        if self.verbose and best:
            print(f"  Best: E={best.energy_total:.4f} at iteration {best.iteration}")

        return best, history


# ============================================================
# MDSystemAsSPRING — wrap NeuroCGMD system as a SPRING Problem
# ============================================================

class MDSystemAsSPRING(Problem):
    """Wrap an NeuroCGMD molecular system as a SPRING Problem."""

    def __init__(self, dim: int = 3):
        self.dim = dim

    def shatter(self, input_data: dict):
        positions = input_data["positions"]
        bonds_raw = input_data.get("bonds", [])
        masses = input_data.get("masses", [1.0] * len(positions))

        pieces = [
            Piece(
                state=np.array(pos[:self.dim], dtype=float),
                piece_id=i,
                metadata={"mass": masses[i] if i < len(masses) else 1.0},
            )
            for i, pos in enumerate(positions)
        ]
        springs = [
            Spring(
                a=int(b[0]), b=int(b[1]),
                rest_value=b[2] if len(b) > 2 else 1.0,
                stiffness=(b[3] if len(b) > 3 else 100.0) / 100.0,
            )
            for b in bonds_raw
        ]
        return pieces, springs

    def local_energy(self, piece: Piece) -> float:
        return 0.0

    def coupling_energy(self, piece_a: Piece, piece_b: Piece, spring: Spring) -> float:
        dist = np.linalg.norm(piece_a.state - piece_b.state)
        return 0.5 * spring.stiffness * (dist - spring.rest_value) ** 2

    def local_moves(self, piece: Piece) -> list[np.ndarray]:
        return [piece.state + np.random.randn(self.dim) * 0.05 for _ in range(6)]

    def spring_force(self, piece_a: Piece, piece_b: Piece, spring: Spring):
        diff = piece_b.state - piece_a.state
        dist = np.linalg.norm(diff)
        if dist < 1e-8:
            z = np.zeros(self.dim)
            return z, z
        direction = diff / dist
        force_mag = spring.stiffness * (dist - spring.rest_value) * 0.05
        return direction * force_mag, -direction * force_mag
