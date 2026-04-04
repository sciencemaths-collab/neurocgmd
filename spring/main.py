#!/usr/bin/env python3
"""
SPRING: Structured Piece Relaxation via Iterative eNergy-Guided descent
=======================================================================

One engine. Four totally different problems. Same algorithm.

Demo 1: Tile Puzzle     (discrete, combinatorial)
Demo 2: Chain Folding   (continuous, physics-based)
Demo 3: Graph Coloring  (constraint satisfaction)
Demo 4: TSP             (NP-hard optimization)

This proves the architecture is general.
"""

import numpy as np
from spring.engine import SpringEngine
from spring.demos.tile_puzzle import TilePuzzle
from spring.demos.chain_fold import ChainFold
from spring.demos.graph_color import GraphColoring
from spring.demos.tsp import TSP
from spring import visualize as viz


def demo_tile_puzzle():
    print("=" * 60)
    print("DEMO 1: TILE PUZZLE (Discrete Combinatorial)")
    print("=" * 60)

    tiles = [4, 2, 0, 3, 1]
    print(f"\n  Input (shuffled):  {tiles}")
    print(f"  Target (sorted):   {list(range(len(tiles)))}")

    problem = TilePuzzle()
    engine = SpringEngine(
        max_iterations=200,
        convergence_threshold=1e-8,
        temperature=2.0,
        cooling_rate=0.98,
        momentum=0.0,
        verbose=True,
    )

    print("\n  Solving...\n")
    np.random.seed(42)
    best, history = engine.solve(problem, tiles)

    result = TilePuzzle.display(best.pieces)
    print(f"\n  Result: {result}")
    print(f"  Energy: {best.energy_total:.4f}")
    print(f"  Solved: {'YES' if result == list(range(len(tiles))) else 'NO (local min)'}")

    viz.plot_energy_trajectory(history, "Tile Puzzle — Energy", "tile_energy.png")
    viz.plot_action_trajectory(history, "Tile Puzzle — Action", "tile_action.png")
    return history


def demo_chain_fold():
    print("\n" + "=" * 60)
    print("DEMO 2: CHAIN FOLDING (Continuous Physics-Based)")
    print("=" * 60)

    sequence = "HPPHPPHH"
    print(f"\n  Sequence: {sequence}")
    print(f"  H = hydrophobic (attract each other)")
    print(f"  P = polar (neutral)")
    print(f"  Goal: fold so H residues cluster together\n")

    problem = ChainFold(
        bond_length=1.0,
        bond_stiffness=10.0,
        attract_stiffness=3.0,
    )
    engine = SpringEngine(
        max_iterations=500,
        convergence_threshold=1e-6,
        temperature=1.5,
        cooling_rate=0.995,
        momentum=0.3,
        damping=0.02,
        reheat_patience=80,
        verbose=True,
    )

    print("  Solving...\n")
    np.random.seed(42)
    best, history = engine.solve(problem, sequence)

    compactness = ChainFold.compactness(best.pieces)
    print(f"\n  Energy: {best.energy_total:.4f}")
    print(f"  Compactness: {compactness:.3f}")

    h_pieces = [p for p in best.pieces if p.metadata["type"] == "H"]
    print(f"  H-H distances:")
    for i, a in enumerate(h_pieces):
        for b in h_pieces[i + 1:]:
            d = np.linalg.norm(a.state - b.state)
            print(f"    H{a.piece_id} — H{b.piece_id}: {d:.2f}")

    reduction = (1 - best.energy_total / max(history[0].energy_total, 1e-10)) * 100
    print(f"  Energy reduction: {reduction:.1f}%")

    viz.plot_energy_trajectory(history, "Chain Fold — Energy", "fold_energy.png")
    viz.plot_action_trajectory(history, "Chain Fold — Action", "fold_action.png")
    viz.plot_chain_fold(best.pieces, best.springs, sequence, "fold_structure.png")
    if best.stiffness_matrix is not None:
        viz.plot_stiffness_matrix(best.stiffness_matrix, "Chain Fold — Stiffness K", "fold_stiffness.png")
    return history


def demo_graph_coloring():
    print("\n" + "=" * 60)
    print("DEMO 3: GRAPH COLORING (Constraint Satisfaction)")
    print("=" * 60)

    # Petersen-like graph — 10 nodes, heavily connected
    graph = {
        "nodes": 10,
        "edges": [
            (0, 1), (0, 4), (0, 5),
            (1, 2), (1, 6),
            (2, 3), (2, 7),
            (3, 4), (3, 8),
            (4, 9),
            (5, 7), (5, 8),
            (6, 8), (6, 9),
            (7, 9),
        ]
    }
    print(f"\n  Nodes: {graph['nodes']}")
    print(f"  Edges: {len(graph['edges'])}")
    print(f"  Colors: 3 (R, G, B)")

    problem = GraphColoring(num_colors=3)
    engine = SpringEngine(
        max_iterations=300,
        convergence_threshold=1e-8,
        temperature=2.0,
        cooling_rate=0.98,
        momentum=0.0,
        reheat_patience=40,
        verbose=True,
    )

    print("\n  Solving...\n")
    np.random.seed(42)
    best, history = engine.solve(problem, graph)

    coloring = GraphColoring.display(best.pieces)
    conflicts = GraphColoring.count_conflicts(best.pieces, best.springs)
    print(f"\n  Coloring: {coloring}")
    print(f"  Conflicts: {conflicts}")
    print(f"  Solved: {'YES' if conflicts == 0 else 'NO'}")

    viz.plot_energy_trajectory(history, "Graph Coloring — Energy", "graph_energy.png")
    viz.plot_graph_coloring(best.pieces, best.springs, "graph_coloring.png")
    return history


def demo_tsp():
    print("\n" + "=" * 60)
    print("DEMO 4: TRAVELING SALESMAN (NP-Hard Optimization)")
    print("=" * 60)

    # 15 random cities
    np.random.seed(123)
    cities = np.random.rand(15, 2) * 100
    print(f"\n  Cities: {len(cities)}")

    problem = TSP()
    engine = SpringEngine(
        max_iterations=500,
        convergence_threshold=1e-8,
        temperature=3.0,
        cooling_rate=0.995,
        momentum=0.0,
        reheat_patience=60,
        verbose=True,
    )

    print("\n  Solving...\n")
    np.random.seed(42)
    best, history = engine.solve(problem, cities)

    tour = TSP.display(best.pieces)
    length = problem.tour_length(best.pieces)
    print(f"\n  Tour: {tour}")
    print(f"  Tour length: {length:.2f}")

    start_length = problem.tour_length(history[0].pieces)
    reduction = (1 - length / max(start_length, 1e-10)) * 100
    print(f"  Starting length: {start_length:.2f}")
    print(f"  Reduction: {reduction:.1f}%")

    viz.plot_energy_trajectory(history, "TSP — Energy", "tsp_energy.png")
    viz.plot_tsp_tour(best.pieces, cities, "tsp_tour.png")
    return history


def main():
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║  SPRING — Universal Optimization Architecture   ║")
    print("  ║  One engine. Any problem.                       ║")
    print("  ╚══════════════════════════════════════════════════╝")

    results = {}
    results["Tile Puzzle"] = demo_tile_puzzle()
    results["Chain Fold"] = demo_chain_fold()
    results["Graph Coloring"] = demo_graph_coloring()
    results["TSP"] = demo_tsp()

    # Comparison plot
    viz.plot_all_energies(results, "comparison.png")

    print("\n" + "=" * 60)
    print("SAME ENGINE solved all 4 problems:")
    print("  1. Tile Puzzle      — discrete combinatorial")
    print("  2. Chain Folding    — continuous physics")
    print("  3. Graph Coloring   — constraint satisfaction")
    print("  4. TSP              — NP-hard optimization")
    print()
    print("The only difference: 5 methods per problem.")
    print(f"Visualizations saved to: {viz.OUTPUT_DIR}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
