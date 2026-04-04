"""
Demo 1: Tile Puzzle
===================
Arrange N tiles into the correct order.

Each tile is a Piece with state = its current value.
Springs connect adjacent slots — they want neighboring tiles to be sequential.
The SPRING engine solves it by swapping tiles to minimize total energy.

This proves: SPRING can solve discrete combinatorial puzzles.
"""

import numpy as np
from spring.engine import Piece, Spring, Problem


class TilePuzzle(Problem):
    """
    Given a shuffled row of numbered tiles [3, 1, 4, 2, 0],
    find the sorted order [0, 1, 2, 3, 4].
    """

    def __init__(self, target: list[int] | None = None):
        self.target = target  # if None, target is sorted order

    def shatter(self, input_data: list[int]):
        """Each tile becomes a Piece. Springs connect adjacent slots."""
        n = len(input_data)
        self.n = n
        self.target = self.target or list(range(n))

        pieces = []
        for i, val in enumerate(input_data):
            pieces.append(Piece(
                state=np.array([float(val)]),
                piece_id=i,
                metadata={"slot": i},
            ))

        # Springs between adjacent slots
        springs = []
        for i in range(n - 1):
            springs.append(Spring(a=i, b=i + 1, stiffness=1.0))

        return pieces, springs

    def local_energy(self, piece: Piece) -> float:
        """A tile wants to be in its 'home' slot. Energy = displacement from home."""
        val = int(round(piece.state[0]))
        slot = piece.metadata["slot"]
        target_slot = self.target.index(val) if val in self.target else slot
        return abs(slot - target_slot)

    def coupling_energy(self, piece_a: Piece, piece_b: Piece, spring: Spring) -> float:
        """Adjacent tiles want to be sequential."""
        val_a = piece_a.state[0]
        val_b = piece_b.state[0]
        # Ideal: b = a + 1 (for sorted order)
        diff = val_b - val_a
        return spring.stiffness * (diff - 1.0) ** 2

    def local_moves(self, piece: Piece) -> list[np.ndarray]:
        """A tile can swap with any other value."""
        val = int(round(piece.state[0]))
        moves = []
        for other_val in range(self.n):
            if other_val != val:
                moves.append(np.array([float(other_val)]))
        return moves

    def spring_force(self, piece_a: Piece, piece_b: Piece, spring: Spring):
        """Nudge tiles toward sequential ordering."""
        diff = piece_b.state[0] - piece_a.state[0]
        error = diff - 1.0  # want diff = 1
        force = spring.stiffness * error * 0.01
        nudge_a = np.array([force])
        nudge_b = np.array([-force])
        return nudge_a, nudge_b

    @staticmethod
    def display(pieces: list[Piece]):
        vals = [int(round(p.state[0])) for p in sorted(pieces, key=lambda p: p.metadata["slot"])]
        return vals
