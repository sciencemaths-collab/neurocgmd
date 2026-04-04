"""
Demo 3: Graph Coloring
======================
Color a graph with K colors so no two adjacent nodes share a color.

Each node is a Piece with state = one-hot color vector.
Springs connect adjacent nodes — they want DIFFERENT colors.
The SPRING engine solves it by flipping colors to minimize conflicts.

This proves: SPRING can solve constraint satisfaction problems.
"""

import numpy as np
from spring.engine import Piece, Spring, Problem


class GraphColoring(Problem):
    """
    Given a graph (adjacency list) and K colors,
    find a valid coloring with zero conflicts.
    """

    def __init__(self, num_colors: int = 3):
        self.num_colors = num_colors

    def shatter(self, input_data: dict):
        """
        Input: {"nodes": N, "edges": [(i,j), ...]}
        Each node → Piece with state = one-hot color vector.
        Each edge → Spring wanting different colors.
        """
        num_nodes = input_data["nodes"]
        edges = input_data["edges"]
        self.num_nodes = num_nodes

        pieces = []
        for i in range(num_nodes):
            # Random initial color as one-hot
            color = np.zeros(self.num_colors)
            color[np.random.randint(self.num_colors)] = 1.0
            pieces.append(Piece(
                state=color,
                piece_id=i,
                metadata={"node": i},
            ))

        springs = []
        for i, j in edges:
            springs.append(Spring(a=i, b=j, stiffness=1.0))

        return pieces, springs

    def local_energy(self, piece: Piece) -> float:
        """Prefer crisp one-hot colors (not fuzzy blends)."""
        # Entropy-like penalty: want one element = 1, rest = 0
        s = piece.state
        s_clipped = np.clip(s, 1e-10, 1.0)
        return -np.sum(s * np.log(s_clipped)) * 0.1

    def coupling_energy(self, piece_a: Piece, piece_b: Piece, spring: Spring) -> float:
        """Adjacent nodes want DIFFERENT colors. Energy = dot product of color vectors."""
        # If same color: dot product = 1 (high energy)
        # If different: dot product = 0 (zero energy)
        return spring.stiffness * np.dot(piece_a.state, piece_b.state)

    def local_moves(self, piece: Piece) -> list[np.ndarray]:
        """Try switching to each possible color."""
        moves = []
        current_color = np.argmax(piece.state)
        for c in range(self.num_colors):
            if c != current_color:
                new_state = np.zeros(self.num_colors)
                new_state[c] = 1.0
                moves.append(new_state)
        return moves

    def spring_force(self, piece_a: Piece, piece_b: Piece, spring: Spring):
        """Push colors apart — repel matching color dimensions."""
        overlap = piece_a.state * piece_b.state
        force = spring.stiffness * overlap * 0.01
        return -force, -force

    @staticmethod
    def display(pieces: list[Piece]):
        color_names = ["R", "G", "B", "Y", "C", "M", "W", "K"]
        result = {}
        for p in pieces:
            color_idx = int(np.argmax(p.state))
            name = color_names[color_idx] if color_idx < len(color_names) else str(color_idx)
            result[p.metadata["node"]] = name
        return result

    @staticmethod
    def count_conflicts(pieces: list[Piece], springs: list[Spring]) -> int:
        conflicts = 0
        for s in springs:
            if np.argmax(pieces[s.a].state) == np.argmax(pieces[s.b].state):
                conflicts += 1
        return conflicts
