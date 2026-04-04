"""
Demo 2: Chain Folding (Protein-like)
=====================================
A chain of N nodes in 2D space must fold to minimize energy.

Each node is a Piece with state = [x, y] position.
Springs connect:
  - Sequential nodes (backbone bonds — stiff)
  - Non-sequential nodes that are "attracted" (like hydrophobic forces)

Energy:
  - Local: each node wants to be near its target region
  - Coupling: backbone springs enforce bond length,
              attraction springs pull compatible nodes together

This proves: SPRING can solve continuous physics-based optimization (like protein folding).
"""

import numpy as np
from spring.engine import Piece, Spring, Problem


class ChainFold(Problem):
    """
    Fold a 2D chain of residues into a compact structure.

    Each residue has a 'type' (H=hydrophobic, P=polar).
    H residues attract each other. P residues are neutral.
    Backbone bonds keep sequential residues connected.
    """

    def __init__(self, bond_length: float = 1.0, bond_stiffness: float = 10.0,
                 attract_stiffness: float = 2.0, attract_range: float = 2.5):
        self.bond_length = bond_length
        self.bond_stiffness = bond_stiffness
        self.attract_stiffness = attract_stiffness
        self.attract_range = attract_range

    def shatter(self, input_data: str):
        """
        Input: sequence like "HHPPHH" (H=hydrophobic, P=polar).
        Each residue becomes a piece at a random position.
        """
        sequence = input_data.upper()
        n = len(sequence)
        self.sequence = sequence

        pieces = []
        for i, residue_type in enumerate(sequence):
            # Start as a straight line with small noise
            pos = np.array([float(i) * self.bond_length, 0.0])
            pos += np.random.randn(2) * 0.1
            pieces.append(Piece(
                state=pos,
                piece_id=i,
                metadata={"type": residue_type, "index": i},
            ))

        springs = []

        # Backbone springs (stiff — keep chain connected)
        for i in range(n - 1):
            springs.append(Spring(
                a=i, b=i + 1,
                stiffness=self.bond_stiffness,
                rest_value=self.bond_length,
                metadata={"type": "backbone"},
            ))

        # Attraction springs between H-H pairs (long range, weaker)
        h_indices = [i for i, r in enumerate(sequence) if r == "H"]
        for idx, i in enumerate(h_indices):
            for j in h_indices[idx + 1:]:
                if abs(i - j) > 2:  # not already backbone-connected neighbors
                    springs.append(Spring(
                        a=i, b=j,
                        stiffness=self.attract_stiffness,
                        rest_value=self.bond_length,  # want them close
                        metadata={"type": "hydrophobic"},
                    ))

        return pieces, springs

    def local_energy(self, piece: Piece) -> float:
        """Slight preference toward origin (compactness)."""
        return 0.01 * np.sum(piece.state ** 2)

    def coupling_energy(self, piece_a: Piece, piece_b: Piece, spring: Spring) -> float:
        """Spring energy: ½k(d - d₀)²"""
        dist = np.linalg.norm(piece_a.state - piece_b.state)
        return 0.5 * spring.stiffness * (dist - spring.rest_value) ** 2

    def local_moves(self, piece: Piece) -> list[np.ndarray]:
        """Random perturbations in 2D — continuous moves."""
        moves = []
        for _ in range(6):
            delta = np.random.randn(2) * 0.3
            moves.append(piece.state + delta)
        return moves

    def spring_force(self, piece_a: Piece, piece_b: Piece, spring: Spring):
        """Hooke's law: F = -k(d - d₀) * direction"""
        diff = piece_b.state - piece_a.state
        dist = np.linalg.norm(diff)
        if dist < 1e-8:
            return np.zeros(2), np.zeros(2)

        direction = diff / dist
        displacement = dist - spring.rest_value
        force_magnitude = spring.stiffness * displacement * 0.05

        nudge_a = direction * force_magnitude
        nudge_b = -direction * force_magnitude
        return nudge_a, nudge_b

    @staticmethod
    def display(pieces: list[Piece]):
        positions = [(p.metadata["type"], p.state[0], p.state[1]) for p in pieces]
        return positions

    @staticmethod
    def compactness(pieces: list[Piece]) -> float:
        """Radius of gyration — how compact is the fold?"""
        positions = np.array([p.state for p in pieces])
        center = positions.mean(axis=0)
        return np.sqrt(np.mean(np.sum((positions - center) ** 2, axis=1)))
