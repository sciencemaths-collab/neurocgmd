"""
SPRING Visualizer
=================
Generates plots showing:
  1. Energy trajectory over iterations (all demos)
  2. Chain fold spatial layout (2D positions)
  3. TSP tour on a map
  4. Stiffness matrix heatmap
  5. Action trajectory (least-action path cost)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "spring_output"


def ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


def plot_energy_trajectory(history, title="Energy Trajectory", filename="energy.png"):
    """Plot E_local, E_coupling, E_total over iterations."""
    out = ensure_output_dir()
    iters = [s.iteration for s in history]
    e_local = [s.energy_local for s in history]
    e_coupling = [s.energy_coupling for s in history]
    e_total = [s.energy_total for s in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, e_total, "k-", linewidth=2, label="E_total")
    ax.plot(iters, e_local, "b--", linewidth=1, alpha=0.7, label="E_local")
    ax.plot(iters, e_coupling, "r--", linewidth=1, alpha=0.7, label="E_spring")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("symlog", linthresh=1.0)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out / filename}")


def plot_action_trajectory(history, title="Action Trajectory", filename="action.png"):
    """Plot accumulated action (least-action path cost) over iterations."""
    out = ensure_output_dir()
    iters = [s.iteration for s in history]
    actions = [s.action for s in history]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(iters, actions, "purple", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accumulated Action")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out / filename}")


def plot_chain_fold(pieces, springs, sequence, filename="fold.png"):
    """Plot 2D chain fold with residue types colored."""
    out = ensure_output_dir()

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw springs
    for s in springs:
        pa = pieces[s.a].state
        pb = pieces[s.b].state
        style = "-" if s.metadata.get("type") == "backbone" else ":"
        color = "#333333" if s.metadata.get("type") == "backbone" else "#cc4444"
        alpha = 0.8 if s.metadata.get("type") == "backbone" else 0.3
        lw = 2 if s.metadata.get("type") == "backbone" else 1
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], style, color=color, alpha=alpha, linewidth=lw)

    # Draw residues
    for p in pieces:
        color = "#e74c3c" if p.metadata["type"] == "H" else "#3498db"
        ax.scatter(p.state[0], p.state[1], s=200, c=color, zorder=5,
                   edgecolors="black", linewidth=1.5)
        ax.annotate(f"{p.metadata['type']}{p.piece_id}", (p.state[0], p.state[1]),
                    ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    ax.set_title(f"Chain Fold: {sequence}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    # Legend
    ax.scatter([], [], c="#e74c3c", s=100, label="H (hydrophobic)")
    ax.scatter([], [], c="#3498db", s=100, label="P (polar)")
    ax.plot([], [], "-", color="#333", label="backbone")
    ax.plot([], [], ":", color="#cc4444", label="H-H attraction")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out / filename}")


def plot_tsp_tour(pieces, cities, filename="tsp.png"):
    """Plot TSP tour on 2D map."""
    out = ensure_output_dir()

    sorted_pieces = sorted(pieces, key=lambda p: p.metadata["position"])
    tour = [int(round(p.state[0])) for p in sorted_pieces]
    n = len(cities)
    tour = [np.clip(c, 0, n - 1) for c in tour]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw tour edges
    for i in range(len(tour)):
        c1 = cities[tour[i]]
        c2 = cities[tour[(i + 1) % len(tour)]]
        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], "b-", linewidth=1.5, alpha=0.6)

    # Draw cities
    ax.scatter(cities[:, 0], cities[:, 1], s=120, c="red", zorder=5,
               edgecolors="black", linewidth=1.5)
    for i, (x, y) in enumerate(cities):
        ax.annotate(str(i), (x, y), ha="center", va="center", fontsize=7,
                    fontweight="bold", color="white")

    # Mark start
    start = cities[tour[0]]
    ax.scatter([start[0]], [start[1]], s=250, c="green", zorder=6,
               edgecolors="black", linewidth=2, marker="*")

    ax.set_title(f"TSP Tour ({len(cities)} cities)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out / filename}")


def plot_graph_coloring(pieces, springs, filename="graph_color.png"):
    """Plot colored graph."""
    out = ensure_output_dir()
    color_map = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]

    n = len(pieces)
    # Layout: circular
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw edges
    for s in springs:
        pa = positions[s.a]
        pb = positions[s.b]
        same_color = np.argmax(pieces[s.a].state) == np.argmax(pieces[s.b].state)
        style = "-" if not same_color else "-"
        color = "#cccccc" if not same_color else "#ff0000"
        lw = 1 if not same_color else 3
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], style, color=color, linewidth=lw, alpha=0.6)

    # Draw nodes
    for i, p in enumerate(pieces):
        c_idx = int(np.argmax(p.state))
        c = color_map[c_idx % len(color_map)]
        ax.scatter(positions[i, 0], positions[i, 1], s=400, c=c, zorder=5,
                   edgecolors="black", linewidth=2)
        ax.annotate(str(i), positions[i], ha="center", va="center",
                    fontsize=12, fontweight="bold", color="white")

    ax.set_title(f"Graph Coloring ({n} nodes)")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out / filename}")


def plot_stiffness_matrix(K, title="Stiffness Matrix K", filename="stiffness.png"):
    """Heatmap of the stiffness matrix."""
    out = ensure_output_dir()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(K, cmap="viridis", aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("Piece j")
    ax.set_ylabel("Piece i")
    fig.colorbar(im, ax=ax, label="Stiffness")
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out / filename}")


def plot_all_energies(results: dict, filename="comparison.png"):
    """Compare energy trajectories across all demos."""
    out = ensure_output_dir()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (name, history) in enumerate(results.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        iters = [s.iteration for s in history]
        e_total = [s.energy_total for s in history]
        ax.plot(iters, e_total, "k-", linewidth=2)
        ax.set_title(name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy")
        ax.grid(True, alpha=0.3)
        if max(e_total) > 10 * min(max(e_total[-1:], default=1), 1):
            ax.set_yscale("symlog", linthresh=1.0)

    fig.suptitle("SPRING — Same Engine, Different Problems", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out / filename}")
