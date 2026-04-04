"""
SPRING Engine — the universal optimizer.

Any problem that can be described as:
  1. Pieces with internal state
  2. Connections (springs) between pieces
  3. An energy function

...can be solved by this engine.

v2: Added Verlet integration, energy barrier crossing,
    adaptive retokenization triggers, and stiffness matrix tracking.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from typing import Any


@dataclass
class Piece:
    """A single unit in the system — one jigsaw piece / one Rubik's cube."""
    state: np.ndarray                    # internal state matrix
    piece_id: int = 0
    metadata: dict = field(default_factory=dict)

    def copy(self):
        return Piece(
            state=self.state.copy(),
            piece_id=self.piece_id,
            metadata=self.metadata.copy(),
        )


@dataclass
class Spring:
    """Elastic coupling between two pieces."""
    a: int                               # index of piece A
    b: int                               # index of piece B
    stiffness: float = 1.0               # how tightly coupled
    rest_value: float = 0.0              # equilibrium distance
    metadata: dict = field(default_factory=dict)


@dataclass
class Snapshot:
    """State of the system at one point in time."""
    pieces: list[Piece]
    springs: list[Spring]
    energy_local: float
    energy_coupling: float
    energy_total: float
    iteration: int
    action: float = 0.0                  # accumulated action along path
    stiffness_matrix: np.ndarray | None = None


class Problem(ABC):
    """
    Abstract interface — define these 5 methods for ANY problem
    and the SPRING engine solves it.
    """

    @abstractmethod
    def shatter(self, input_data: Any) -> tuple[list[Piece], list[Spring]]:
        """Tokenize: break input into pieces and define springs between them."""
        ...

    @abstractmethod
    def local_energy(self, piece: Piece) -> float:
        """How 'unsolved' is this piece internally? 0 = perfect."""
        ...

    @abstractmethod
    def coupling_energy(self, piece_a: Piece, piece_b: Piece, spring: Spring) -> float:
        """How much tension is in this spring? 0 = perfect fit."""
        ...

    @abstractmethod
    def local_moves(self, piece: Piece) -> list[np.ndarray]:
        """What transformations can this piece try? Returns list of new states."""
        ...

    @abstractmethod
    def spring_force(self, piece_a: Piece, piece_b: Piece, spring: Spring) -> tuple[np.ndarray, np.ndarray]:
        """
        Given spring tension, return nudges to apply to piece_a and piece_b
        to reduce coupling energy. Returns (nudge_a, nudge_b).
        """
        ...

    def post_local_solve(self, pieces: list[Piece]) -> list[Piece]:
        """Optional: repair/validate pieces after local solve step.
        Use this for problems with global constraints (e.g., permutations in TSP)."""
        return pieces

    def adapt_springs(self) -> bool:
        """Whether to use adaptive spring stiffness. Default: True."""
        return True

    def should_retokenize(self, snapshot: Snapshot) -> bool:
        """Optional: should we re-shatter? Default: no."""
        return False

    def retokenize(self, snapshot: Snapshot, input_data: Any) -> tuple[list[Piece], list[Spring]]:
        """Optional: re-shatter with different granularity."""
        return snapshot.pieces, snapshot.springs


class SpringEngine:
    """
    The universal solver.

    Core loop:
      local solve → spring propagation → Verlet integration → energy check → repeat

    Features:
      - Simulated annealing (cross energy barriers)
      - Velocity Verlet integration (least-action paths with momentum)
      - Adaptive spring stiffness (springs learn which connections matter)
      - Stiffness matrix K tracking (full system coupling at each step)
      - Adaptive retokenization (re-shatter if stuck)
      - Multi-scale relaxation (coarse-to-fine temperature schedule)
    """

    def __init__(
        self,
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-6,
        temperature: float = 1.0,
        cooling_rate: float = 0.995,
        momentum: float = 0.0,
        damping: float = 0.01,
        reheat_factor: float = 2.0,
        reheat_patience: int = 50,
        verbose: bool = True,
    ):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.momentum = momentum
        self.damping = damping
        self.reheat_factor = reheat_factor
        self.reheat_patience = reheat_patience
        self.verbose = verbose

    def _build_stiffness_matrix(self, pieces, springs):
        """Build the NxN stiffness matrix K encoding all spring couplings."""
        n = len(pieces)
        K = np.zeros((n, n))
        for s in springs:
            K[s.a, s.b] = s.stiffness
            K[s.b, s.a] = s.stiffness
            K[s.a, s.a] += s.stiffness
            K[s.b, s.b] += s.stiffness
        return K

    def _compute_action(self, prev_states, curr_states, energy):
        """
        Discrete action: S = Σ [½|v|² - E]
        v = state change between steps (kinetic), E = potential
        """
        kinetic = 0.0
        for prev, curr in zip(prev_states, curr_states):
            v = curr - prev
            kinetic += 0.5 * np.sum(v ** 2)
        return kinetic - energy

    def solve(self, problem: Problem, input_data: Any):
        """Run the full SPRING relaxation loop."""

        # === STEP 1: SHATTER ===
        pieces, springs = problem.shatter(input_data)
        history: list[Snapshot] = []
        prev_energy = float("inf")
        best_energy_seen = float("inf")
        stall_counter = 0
        temp = self.temperature
        accumulated_action = 0.0

        # Velocity Verlet state
        velocities = [np.zeros_like(p.state, dtype=float) for p in pieces]
        prev_states = [p.state.copy() for p in pieces]

        for iteration in range(self.max_iterations):

            # === COMPUTE ENERGY ===
            e_local = sum(problem.local_energy(p) for p in pieces)
            e_coupling = sum(
                problem.coupling_energy(pieces[s.a], pieces[s.b], s)
                for s in springs
            )
            e_total = e_local + e_coupling

            # === TRACK ACTION (least-action path) ===
            curr_states = [p.state.copy() for p in pieces]
            step_action = self._compute_action(prev_states, curr_states, e_total)
            accumulated_action += step_action
            prev_states = curr_states

            # === BUILD STIFFNESS MATRIX ===
            K = self._build_stiffness_matrix(pieces, springs)

            snap = Snapshot(
                pieces=[p.copy() for p in pieces],
                springs=list(springs),
                energy_local=e_local,
                energy_coupling=e_coupling,
                energy_total=e_total,
                iteration=iteration,
                action=accumulated_action,
                stiffness_matrix=K,
            )
            history.append(snap)

            if self.verbose and iteration % max(1, self.max_iterations // 20) == 0:
                print(
                    f"  iter {iteration:4d} | "
                    f"E_local={e_local:8.3f}  E_spring={e_coupling:8.3f}  "
                    f"E_total={e_total:8.3f}  action={accumulated_action:8.2f}  "
                    f"temp={temp:.4f}"
                )

            # === CONVERGENCE CHECK ===
            if abs(prev_energy - e_total) < self.convergence_threshold and iteration > 0:
                if self.verbose:
                    print(f"  Converged at iteration {iteration}")
                break
            prev_energy = e_total

            # === STALL DETECTION & REHEAT (barrier crossing) ===
            if e_total < best_energy_seen - self.convergence_threshold:
                best_energy_seen = e_total
                stall_counter = 0
            else:
                stall_counter += 1

            if stall_counter >= self.reheat_patience:
                temp = min(temp * self.reheat_factor, self.temperature)
                stall_counter = 0
                if self.verbose:
                    print(f"  Reheated at iteration {iteration} → temp={temp:.4f}")

            # === RETOKENIZE CHECK ===
            if problem.should_retokenize(snap):
                pieces, springs = problem.retokenize(snap, input_data)
                velocities = [np.zeros_like(p.state, dtype=float) for p in pieces]
                prev_states = [p.state.copy() for p in pieces]
                if self.verbose:
                    print(f"  Re-tokenized at iteration {iteration} → {len(pieces)} pieces")

            # === STEP 2: LOCAL SOLVE (Rubik's phase) ===
            # Build neighbor lookup once for efficiency
            neighbors = {i: [] for i in range(len(pieces))}
            for s in springs:
                neighbors[s.a].append(s)
                neighbors[s.b].append(s)

            for i, piece in enumerate(pieces):
                moves = problem.local_moves(piece)
                if not moves:
                    continue

                best_state = piece.state
                best_energy_i = problem.local_energy(piece)

                neighbor_coupling = 0.0
                for s in neighbors[i]:
                    other = s.b if s.a == i else s.a
                    neighbor_coupling += problem.coupling_energy(
                        piece if s.a == i else pieces[other],
                        pieces[other] if s.a == i else piece,
                        s,
                    )
                best_total_i = best_energy_i + neighbor_coupling

                for new_state in moves:
                    old_state = piece.state
                    piece.state = new_state

                    local_e = problem.local_energy(piece)
                    coup_e = 0.0
                    for s in neighbors[i]:
                        other = s.b if s.a == i else s.a
                        coup_e += problem.coupling_energy(
                            piece if s.a == i else pieces[other],
                            pieces[other] if s.a == i else piece,
                            s,
                        )

                    total_e = local_e + coup_e
                    delta = total_e - best_total_i

                    # Metropolis criterion — allows barrier crossing
                    if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                        best_state = new_state
                        best_total_i = total_e

                    piece.state = old_state

                pieces[i].state = best_state

            # === POST LOCAL SOLVE (repair global constraints) ===
            pieces = problem.post_local_solve(pieces)

            # === STEP 3: SPRING FORCES ===
            nudges = [np.zeros_like(p.state, dtype=float) for p in pieces]
            for s in springs:
                nudge_a, nudge_b = problem.spring_force(pieces[s.a], pieces[s.b], s)
                nudges[s.a] = nudges[s.a] + nudge_a
                nudges[s.b] = nudges[s.b] + nudge_b

            # === STEP 4: VELOCITY VERLET INTEGRATION (least-action) ===
            # v(t+½dt) = (1-damping) * v(t) + F(t)
            # x(t+dt) = x(t) + v(t+½dt)
            for i, piece in enumerate(pieces):
                velocities[i] = (1.0 - self.damping) * self.momentum * velocities[i] + nudges[i]
                piece.state = piece.state + velocities[i]

            # === STEP 5: ADAPTIVE SPRING STIFFNESS ===
            if problem.adapt_springs():
                for s in springs:
                    ce = problem.coupling_energy(pieces[s.a], pieces[s.b], s)
                    if ce < 0.01:
                        s.stiffness *= 0.99
                    elif ce > 1.0:
                        s.stiffness *= 1.01
                    s.stiffness = np.clip(s.stiffness, 0.01, 100.0)

            # === COOL ===
            temp *= self.cooling_rate

        # Return best snapshot
        best = min(history, key=lambda s: s.energy_total)
        if self.verbose:
            print(f"  Best energy: {best.energy_total:.4f} at iteration {best.iteration}")
            print(f"  Final action: {accumulated_action:.4f}")
        return best, history
