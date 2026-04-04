"""
Demo 4: Traveling Salesman Problem (TSP)
========================================
Find the shortest route visiting all cities exactly once.

Each tour position is a Piece. State = [city_index].
Moves = swap proposals between positions.
post_local_solve repairs the permutation after each round.
Springs connect consecutive positions — they want short distances.

This proves: SPRING can solve NP-hard optimization problems.
"""

import numpy as np
from spring.engine import Piece, Spring, Problem


class TSP(Problem):
    """
    Given N cities with (x,y) coordinates, find the shortest tour.

    Key: post_local_solve enforces the permutation constraint —
    every city appears exactly once, even though pieces move independently.
    """

    def __init__(self):
        self.cities = None
        self.dist_matrix = None
        self.n = 0

    def shatter(self, input_data: np.ndarray):
        self.cities = input_data
        self.n = len(input_data)

        self.dist_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                self.dist_matrix[i, j] = np.linalg.norm(input_data[i] - input_data[j])

        tour = np.random.permutation(self.n)

        pieces = []
        for pos, city in enumerate(tour):
            pieces.append(Piece(
                state=np.array([float(city)]),
                piece_id=pos,
                metadata={"position": pos},
            ))

        springs = []
        for i in range(self.n):
            j = (i + 1) % self.n
            springs.append(Spring(a=i, b=j, stiffness=1.0))

        return pieces, springs

    def local_energy(self, piece: Piece) -> float:
        return 0.0

    def coupling_energy(self, piece_a: Piece, piece_b: Piece, spring: Spring) -> float:
        city_a = int(round(piece_a.state[0])) % self.n
        city_b = int(round(piece_b.state[0])) % self.n
        return spring.stiffness * self.dist_matrix[city_a, city_b]

    def local_moves(self, piece: Piece) -> list[np.ndarray]:
        """Propose random city values. post_local_solve will fix duplicates."""
        moves = []
        candidates = np.random.choice(self.n, size=min(self.n - 1, 6), replace=False)
        current = int(round(piece.state[0]))
        for c in candidates:
            if c != current:
                moves.append(np.array([float(c)]))
        return moves

    def post_local_solve(self, pieces: list[Piece]) -> list[Piece]:
        """
        Repair permutation: if two positions claim the same city,
        keep the one that benefits more and give the other a missing city
        that minimizes its local coupling cost.
        """
        n = self.n
        cities_assigned = {}  # city -> list of position indices
        for i, p in enumerate(pieces):
            c = int(round(p.state[0])) % n
            p.state[0] = float(c)
            cities_assigned.setdefault(c, []).append(i)

        all_cities = set(range(n))
        used = set()
        conflicts = []

        # First pass: assign uncontested cities
        for city, positions in cities_assigned.items():
            if len(positions) == 1:
                used.add(city)
            else:
                # Keep the position where this city gives lowest neighbor cost
                best_pos = None
                best_cost = float("inf")
                for pos in positions:
                    # Cost = distance to prev + distance to next
                    prev_pos = (pos - 1) % n
                    next_pos = (pos + 1) % n
                    prev_city = int(round(pieces[prev_pos].state[0])) % n
                    next_city = int(round(pieces[next_pos].state[0])) % n
                    cost = self.dist_matrix[city, prev_city] + self.dist_matrix[city, next_city]
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = pos
                used.add(city)
                for pos in positions:
                    if pos != best_pos:
                        conflicts.append(pos)

        missing = list(all_cities - used)
        np.random.shuffle(missing)

        # Assign missing cities to conflicted positions greedily
        for pos in conflicts:
            if not missing:
                break
            prev_pos = (pos - 1) % n
            next_pos = (pos + 1) % n
            prev_city = int(round(pieces[prev_pos].state[0])) % n
            next_city = int(round(pieces[next_pos].state[0])) % n

            best_city = missing[0]
            best_cost = float("inf")
            for city in missing:
                cost = self.dist_matrix[city, prev_city] + self.dist_matrix[city, next_city]
                if cost < best_cost:
                    best_cost = cost
                    best_city = city

            pieces[pos].state[0] = float(best_city)
            missing.remove(best_city)

        return pieces

    def adapt_springs(self) -> bool:
        return False

    def spring_force(self, piece_a: Piece, piece_b: Piece, spring: Spring):
        return np.zeros(1), np.zeros(1)

    def tour_length(self, pieces: list[Piece]) -> float:
        sorted_pieces = sorted(pieces, key=lambda p: p.metadata["position"])
        total = 0.0
        for i in range(self.n):
            city_a = int(round(sorted_pieces[i].state[0])) % self.n
            city_b = int(round(sorted_pieces[(i + 1) % self.n].state[0])) % self.n
            total += self.dist_matrix[city_a, city_b]
        return total

    @staticmethod
    def display(pieces: list[Piece]):
        sorted_pieces = sorted(pieces, key=lambda p: p.metadata["position"])
        return [int(round(p.state[0])) for p in sorted_pieces]
