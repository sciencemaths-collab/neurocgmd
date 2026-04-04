"""Generate PNG plots from partial or complete simulation output."""

from __future__ import annotations

import csv
import json
import sys
from math import sqrt, log
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from core.state import SimulationState
from validation.adaptive_analysis import RDFCalculator, RMSDTracker
from validation.molecular_observables import (
    SASACalculator, RadiusOfGyration, HydrogenBondAnalyzer, ResidueEnergyDecomposition,
    ContactMapCalculator,
)
from scripts.plot_style import setup_style, styled_figure, add_watermark, smooth, CHAIN_COLORS, ENERGY_COLORS
from scripts.plot_binding import run_binding_analysis


def load_energies(path: Path):
    steps, pe, ke, total, times = [], [], [], [], []
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) >= 8 and row[0] != "stage":
                try:
                    s, t, p, k, tot = int(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])
                except (ValueError, IndexError):
                    continue
                steps.append(s)
                times.append(t)
                pe.append(p)
                ke.append(k)
                total.append(tot)
    # Ensure all lists are same length (partial rows from live file)
    n = min(len(steps), len(times), len(pe), len(ke), len(total))
    return steps[:n], times[:n], pe[:n], ke[:n], total[:n]


def load_trajectory(path: Path) -> list[SimulationState]:
    states = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    frame = json.loads(line)
                    states.append(SimulationState.from_dict(frame["state"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return states


def write_cg_trajectory_pdb(traj_states, bundle_path: Path, out_path: Path):
    """Write CG trajectory as multi-model PDB for VMD/PyMOL."""
    # Get bead metadata from bundle
    bd = json.loads(bundle_path.read_text()) if bundle_path.exists() else {}
    entities = bd.get("import_summary", {}).get("entities", [])

    # Build chain assignment: first entity = chain A, second = chain B, etc.
    chain_map: dict[int, str] = {}
    offset = 0
    chain_labels = "ABCDEFGHIJ"
    for idx, ent in enumerate(entities):
        bc = ent.get("bead_count", 0)
        ch = chain_labels[idx] if idx < len(chain_labels) else "X"
        for i in range(offset, offset + bc):
            chain_map[i] = ch
        offset += bc

    lines: list[str] = []
    for model_idx, state in enumerate(traj_states):
        lines.append(f"MODEL     {model_idx + 1:>4d}")
        lines.append(f"REMARK   SIM_TIME  {state.time:.3f} ps   STEP {state.step}")
        for pi in range(state.particle_count):
            x, y, z = state.particles.positions[pi]
            # PDB coordinates are in Angstroms, our positions are in nm
            x_a, y_a, z_a = x * 10.0, y * 10.0, z * 10.0
            chain = chain_map.get(pi, "A")
            label = state.particles.labels[pi] if pi < len(state.particles.labels) else "CG"
            atom_name = label[:4].ljust(4)
            resname = "CG" + chain
            resseq = pi + 1
            serial = pi + 1
            line = (
                f"ATOM  {serial:>5d} {atom_name:4s} {resname:>3s} {chain:1s}{resseq:>4d}    "
                f"{x_a:>8.3f}{y_a:>8.3f}{z_a:>8.3f}{1.0:>6.2f}{0.0:>6.2f}          {'C':>2s}"
            )
            lines.append(line)
        lines.append("ENDMDL")
    lines.append("END")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_cg_snapshot_pdb(state, bundle_path: Path, out_path: Path, title: str = ""):
    """Write a single CG frame as PDB."""
    bd = json.loads(bundle_path.read_text()) if bundle_path.exists() else {}
    entities = bd.get("import_summary", {}).get("entities", [])
    chain_map: dict[int, str] = {}
    offset = 0
    chain_labels = "ABCDEFGHIJ"
    for idx, ent in enumerate(entities):
        bc = ent.get("bead_count", 0)
        ch = chain_labels[idx] if idx < len(chain_labels) else "X"
        for i in range(offset, offset + bc):
            chain_map[i] = ch
        offset += bc

    lines: list[str] = []
    if title:
        lines.append(f"REMARK   {title}")
    lines.append(f"REMARK   SIM_TIME  {state.time:.3f} ps   STEP {state.step}")
    for pi in range(state.particle_count):
        x, y, z = state.particles.positions[pi]
        x_a, y_a, z_a = x * 10.0, y * 10.0, z * 10.0
        chain = chain_map.get(pi, "A")
        label = state.particles.labels[pi] if pi < len(state.particles.labels) else "CG"
        atom_name = label[:4].ljust(4)
        resname = "CG" + chain
        resseq = pi + 1
        serial = pi + 1
        line = (
            f"ATOM  {serial:>5d} {atom_name:4s} {resname:>3s} {chain:1s}{resseq:>4d}    "
            f"{x_a:>8.3f}{y_a:>8.3f}{z_a:>8.3f}{1.0:>6.2f}{0.0:>6.2f}          {'C':>2s}"
        )
        lines.append(line)

    # Add CONECT records for sequential beads within same chain
    prev_chain = None
    for pi in range(state.particle_count):
        ch = chain_map.get(pi, "A")
        if ch == prev_chain and pi > 0:
            lines.append(f"CONECT{pi:>5d}{pi + 1:>5d}")
        prev_chain = ch

    lines.append("END")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_aa_pdb(pdb_path: Path):
    """Parse AA PDB into list of dicts with full residue and atom metadata.

    Fields per atom:
      line     – raw PDB line (for back-mapped PDB export)
      chain    – chain ID (single char)
      resseq   – residue sequence number
      resname  – 3-letter amino acid code (GLY, ARG, ASP, …)
      name     – atom name (CA, N, O, CB, …)
      element  – element symbol (C, N, O, S, …)
      x, y, z  – coordinates in Angstroms
    """
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atoms.append({
                    "line": line.rstrip(),
                    "chain": line[21],
                    "resseq": int(line[22:26].strip()),
                    "resname": line[17:20].strip(),
                    "name": line[12:16].strip(),
                    "element": line[76:78].strip() if len(line) > 77 else line[12:14].strip()[0],
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                })
    return atoms


def _build_bead_to_atom_mapping(aa_atoms, entities, bead_labels, cg_initial_positions):
    """Map each CG bead to its AA atoms using bead labels and initial CG positions.

    Bead labels like 'chain_a_support_3_10' encode the chain and residue range.
    Uses the actual initial CG positions (in nm) as reference centroids.

    Returns:
        bead_atom_offsets: list of lists of (atom_idx, dx, dy, dz) —
            per-atom offset from the bead center in Angstroms
    """
    import re

    # Index AA atoms by (chain, resseq)
    chain_res_atoms: dict[tuple[str, int], list[int]] = {}
    for ai, atom in enumerate(aa_atoms):
        key = (atom["chain"], atom["resseq"])
        if key not in chain_res_atoms:
            chain_res_atoms[key] = []
        chain_res_atoms[key].append(ai)

    # Map entity_id -> chain_ids
    entity_chains: dict[str, list[str]] = {}
    for ent in entities:
        eid = ent.get("entity_id", "")
        entity_chains[eid] = ent.get("chain_ids", [])

    bead_atom_offsets: list[list[tuple[int, float, float, float]]] = []

    for bi, label in enumerate(bead_labels):
        # Parse label: "chain_X_support_START_END" or similar
        # Extract chain entity and residue range
        match = re.match(r"(chain_\w+)_\w+_(\d+)_(\d+)", label)
        atom_indices = []

        if match:
            entity_id = match.group(1)
            res_start = int(match.group(2))
            res_end = int(match.group(3))
            chains = entity_chains.get(entity_id, [])

            for chain in chains:
                for resseq in range(res_start, res_end + 1):
                    key = (chain, resseq)
                    if key in chain_res_atoms:
                        atom_indices.extend(chain_res_atoms[key])

        # CG bead center in Angstroms
        cg_x = cg_initial_positions[bi][0] * 10.0
        cg_y = cg_initial_positions[bi][1] * 10.0
        cg_z = cg_initial_positions[bi][2] * 10.0

        # Compute per-atom offsets from bead center
        offsets = []
        for ai in atom_indices:
            dx = aa_atoms[ai]["x"] - cg_x
            dy = aa_atoms[ai]["y"] - cg_y
            dz = aa_atoms[ai]["z"] - cg_z
            offsets.append((ai, dx, dy, dz))

        bead_atom_offsets.append(offsets)

    return bead_atom_offsets


def _precompute_atom_bead_weights(aa_atoms, cg_initial_positions, bead_atom_offsets):
    """Precompute interpolation weights: each atom gets distance-weighted
    influence from the nearest CG beads.

    Returns per-atom list of [(bead_idx, weight), ...] and the original AA coords.
    """
    n_beads = len(cg_initial_positions)
    # CG initial positions in Angstroms
    cg_ref = [(p[0] * 10.0, p[1] * 10.0, p[2] * 10.0) for p in cg_initial_positions]

    # Build set of all mapped atom indices
    mapped_atoms: set[int] = set()
    for offsets in bead_atom_offsets:
        for ai, _, _, _ in offsets:
            mapped_atoms.add(ai)

    atom_weights: list[list[tuple[int, float]]] = [[] for _ in aa_atoms]

    # Build bead-to-chain mapping from bead_atom_offsets
    import re
    bead_chain: dict[int, str] = {}
    for bi, offsets in enumerate(bead_atom_offsets):
        for ai_off, _, _, _ in offsets:
            bead_chain[bi] = aa_atoms[ai_off]["chain"]
            break

    for ai in mapped_atoms:
        ax, ay, az = aa_atoms[ai]["x"], aa_atoms[ai]["y"], aa_atoms[ai]["z"]
        atom_chain = aa_atoms[ai]["chain"]

        # Only consider beads from the same chain
        dists = []
        for bi in range(n_beads):
            if bead_chain.get(bi) != atom_chain:
                continue
            dx = ax - cg_ref[bi][0]
            dy = ay - cg_ref[bi][1]
            dz = az - cg_ref[bi][2]
            d = sqrt(dx * dx + dy * dy + dz * dz)
            dists.append((d, bi))

        if not dists:
            # Fallback: use all beads
            for bi in range(n_beads):
                dx = ax - cg_ref[bi][0]
                dy = ay - cg_ref[bi][1]
                dz = az - cg_ref[bi][2]
                d = sqrt(dx * dx + dy * dy + dz * dz)
                dists.append((d, bi))

        dists.sort()
        # Use nearest 2 same-chain beads
        top_k = dists[:2]
        if top_k[0][0] < 0.01:
            atom_weights[ai] = [(top_k[0][1], 1.0)]
        else:
            inv_dists = [(1.0 / d, bi) for d, bi in top_k]
            total_inv = sum(w for w, _ in inv_dists)
            atom_weights[ai] = [(bi, w / total_inv) for w, bi in inv_dists]

    return atom_weights, mapped_atoms


def backmap_cg_to_aa(state, aa_atoms, cg_initial_positions, atom_weights, mapped_atoms):
    """Interpolation-based back-mapping.

    For each AA atom:
      1. Compute displacement of each nearby CG bead: delta_i = cg_pos_now_i - cg_pos_initial_i
      2. Weighted average: delta = sum(w_i * delta_i)
      3. New atom pos = original_aa_pos + delta

    This ensures:
      - t=0 back-map is identical to the original AA (delta=0)
      - Atoms near bead boundaries interpolate smoothly between neighbors
      - No gaps or breaks

    Returns list of (x, y, z) in Angstroms for each AA atom.
    """
    new_coords = [(a["x"], a["y"], a["z"]) for a in aa_atoms]

    # CG reference positions in Angstroms
    cg_ref = [(p[0] * 10.0, p[1] * 10.0, p[2] * 10.0) for p in cg_initial_positions]

    # Current CG positions in Angstroms
    n_beads = state.particle_count
    cg_now = [(state.particles.positions[bi][0] * 10.0,
               state.particles.positions[bi][1] * 10.0,
               state.particles.positions[bi][2] * 10.0) for bi in range(n_beads)]

    # Precompute per-bead displacements
    bead_delta = [(cg_now[bi][0] - cg_ref[bi][0],
                   cg_now[bi][1] - cg_ref[bi][1],
                   cg_now[bi][2] - cg_ref[bi][2]) for bi in range(n_beads)]

    for ai in mapped_atoms:
        weights = atom_weights[ai]
        if not weights:
            continue
        # Weighted average displacement
        dx = sum(w * bead_delta[bi][0] for bi, w in weights)
        dy = sum(w * bead_delta[bi][1] for bi, w in weights)
        dz = sum(w * bead_delta[bi][2] for bi, w in weights)

        ox, oy, oz = aa_atoms[ai]["x"], aa_atoms[ai]["y"], aa_atoms[ai]["z"]
        new_coords[ai] = (ox + dx, oy + dy, oz + dz)

    # Bond relaxation: fix stretched peptide bonds at bead boundaries.
    # Works on actual C(i)-N(i+1) peptide bonds and CA-CA distances.
    _RELAX_ITERATIONS = 8

    # Index atoms by (chain, resseq, name) for fast lookup
    atom_lookup: dict[tuple[str, int, str], int] = {}
    by_chain_res: dict[tuple[str, int], list[int]] = {}
    for ai in mapped_atoms:
        atom = aa_atoms[ai]
        name = atom["name"]
        chain = atom["chain"]
        resseq = atom["resseq"]
        atom_lookup[(chain, resseq, name)] = ai
        key = (chain, resseq)
        if key not in by_chain_res:
            by_chain_res[key] = []
        by_chain_res[key].append(ai)

    # Collect all peptide bond pairs: C(i) - N(i+1) in same chain
    peptide_bonds: list[tuple[int, int, float]] = []  # (C_idx, N_idx, target_dist)
    ca_pairs: list[tuple[int, int, float]] = []

    chains_residues: dict[str, list[int]] = {}
    for chain, resseq in by_chain_res:
        if chain not in chains_residues:
            chains_residues[chain] = []
        chains_residues[chain].append(resseq)
    for chain in chains_residues:
        chains_residues[chain].sort()

    for chain, residues in chains_residues.items():
        for k in range(len(residues) - 1):
            r1, r2 = residues[k], residues[k + 1]
            if r2 - r1 > 2:
                continue  # skip non-sequential
            c_key = (chain, r1, "C")
            n_key = (chain, r2, "N")
            ca1_key = (chain, r1, "CA")
            ca2_key = (chain, r2, "CA")
            if c_key in atom_lookup and n_key in atom_lookup:
                peptide_bonds.append((atom_lookup[c_key], atom_lookup[n_key], 1.33))
            if ca1_key in atom_lookup and ca2_key in atom_lookup:
                ca_pairs.append((atom_lookup[ca1_key], atom_lookup[ca2_key], 3.80))

    for _iteration in range(_RELAX_ITERATIONS):
        # Fix peptide C-N bonds (stretch and compress)
        for i, j, target in peptide_bonds:
            xi, yi, zi = new_coords[i]
            xj, yj, zj = new_coords[j]
            dx, dy, dz = xj - xi, yj - yi, zj - zi
            d = sqrt(dx * dx + dy * dy + dz * dz)
            if d < target * 0.6 and d > 0.01:
                # Too short — push apart
                deficit = target - d
                frac = 0.45 * deficit / d
                cx, cy, cz = dx * frac, dy * frac, dz * frac
                new_coords[i] = (xi - cx, yi - cy, zi - cz)
                new_coords[j] = (xj + cx, yj + cy, zj + cz)
                res_i = (aa_atoms[i]["chain"], aa_atoms[i]["resseq"])
                res_j = (aa_atoms[j]["chain"], aa_atoms[j]["resseq"])
                for ai2 in by_chain_res.get(res_i, []):
                    if ai2 != i:
                        ox, oy, oz = new_coords[ai2]
                        new_coords[ai2] = (ox - cx * 0.5, oy - cy * 0.5, oz - cz * 0.5)
                for ai2 in by_chain_res.get(res_j, []):
                    if ai2 != j:
                        ox, oy, oz = new_coords[ai2]
                        new_coords[ai2] = (ox + cx * 0.5, oy + cy * 0.5, oz + cz * 0.5)
            elif d > target * 1.5 and d > 0.01:
                excess = d - target
                frac = 0.45 * excess / d
                cx, cy, cz = dx * frac, dy * frac, dz * frac
                new_coords[i] = (xi + cx, yi + cy, zi + cz)
                new_coords[j] = (xj - cx, yj - cy, zj - cz)
                # Drag entire residue along
                res_i = (aa_atoms[i]["chain"], aa_atoms[i]["resseq"])
                res_j = (aa_atoms[j]["chain"], aa_atoms[j]["resseq"])
                for ai2 in by_chain_res.get(res_i, []):
                    if ai2 != i:
                        ox, oy, oz = new_coords[ai2]
                        new_coords[ai2] = (ox + cx * 0.5, oy + cy * 0.5, oz + cz * 0.5)
                for ai2 in by_chain_res.get(res_j, []):
                    if ai2 != j:
                        ox, oy, oz = new_coords[ai2]
                        new_coords[ai2] = (ox - cx * 0.5, oy - cy * 0.5, oz - cz * 0.5)

        # Fix CA-CA distances
        for i, j, target in ca_pairs:
            xi, yi, zi = new_coords[i]
            xj, yj, zj = new_coords[j]
            dx, dy, dz = xj - xi, yj - yi, zj - zi
            d = sqrt(dx * dx + dy * dy + dz * dz)
            if d > target * 1.15 and d > 0.01:
                excess = d - target
                frac = 0.3 * excess / d
                cx, cy, cz = dx * frac, dy * frac, dz * frac
                new_coords[i] = (xi + cx, yi + cy, zi + cz)
                new_coords[j] = (xj - cx, yj - cy, zj - cz)

    return new_coords


def write_aa_backmapped_pdb(aa_atoms, new_coords, out_path: Path, title: str = ""):
    """Write back-mapped AA structure as PDB."""
    lines = []
    if title:
        lines.append(f"REMARK   {title}")
    for ai, atom in enumerate(aa_atoms):
        x, y, z = new_coords[ai]
        old_line = atom["line"]
        # Replace coordinates in the PDB line (columns 31-54)
        new_line = f"{old_line[:30]}{x:>8.3f}{y:>8.3f}{z:>8.3f}{old_line[54:]}"
        lines.append(new_line)
    lines.append("END")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_aa_backmapped_trajectory(traj_states, aa_atoms, cg_initial_positions,
                                    atom_weights, mapped_atoms, out_path: Path):
    """Write multi-model back-mapped AA trajectory PDB."""
    lines = []
    for model_idx, state in enumerate(traj_states):
        lines.append(f"MODEL     {model_idx + 1:>4d}")
        lines.append(f"REMARK   SIM_TIME  {state.time:.3f} ps   STEP {state.step}")
        new_coords = backmap_cg_to_aa(state, aa_atoms, cg_initial_positions,
                                       atom_weights, mapped_atoms)
        for ai, atom in enumerate(aa_atoms):
            x, y, z = new_coords[ai]
            old_line = atom["line"]
            new_line = f"{old_line[:30]}{x:>8.3f}{y:>8.3f}{z:>8.3f}{old_line[54:]}"
            lines.append(new_line)
        lines.append("ENDMDL")
    lines.append("END")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_structure_snapshots(traj_states, bundle_path: Path, plots_dir: Path):
    """Plot 2D projections of CG structure at key timepoints."""
    bd = json.loads(bundle_path.read_text()) if bundle_path.exists() else {}
    entities = bd.get("import_summary", {}).get("entities", [])
    chain_map: dict[int, str] = {}
    offset = 0
    for idx, ent in enumerate(entities):
        bc = ent.get("bead_count", 0)
        for i in range(offset, offset + bc):
            chain_map[i] = idx
        offset += bc

    n_snapshots = min(6, len(traj_states))
    indices = [int(i * (len(traj_states) - 1) / max(1, n_snapshots - 1)) for i in range(n_snapshots)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=150, facecolor="white")
    axes = axes.flatten()

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    for ax_idx, frame_idx in enumerate(indices):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        state = traj_states[frame_idx]
        pos = state.particles.positions
        n = state.particle_count

        for pi in range(n):
            entity_idx = chain_map.get(pi, 0)
            color = colors[entity_idx % len(colors)]
            ax.scatter(pos[pi][0], pos[pi][1], c=color, s=80, edgecolors="black",
                       linewidths=0.3, zorder=3, alpha=0.85)

        # Draw bonds between sequential beads in same chain
        prev_entity = None
        for pi in range(n):
            eidx = chain_map.get(pi, 0)
            if eidx == prev_entity and pi > 0:
                ax.plot([pos[pi-1][0], pos[pi][0]], [pos[pi-1][1], pos[pi][1]],
                        color="gray", linewidth=0.8, zorder=2)
            prev_entity = eidx

        ax.set_title(f"t = {state.time:.1f} ps (step {state.step})", fontsize=9)
        ax.set_xlabel("x (nm)", fontsize=8)
        ax.set_ylabel("y (nm)", fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    fig.suptitle("CG Structure Snapshots (XY projection)\nRed = Barnase, Blue = Barstar", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(plots_dir / "structure_snapshots.png", bbox_inches="tight")
    plt.close(fig)


def main(output_dir: str):
    out = Path(output_dir)
    plots_dir = out / "plots"
    plots_dir.mkdir(exist_ok=True)

    energy_path = out / "energies.csv"
    traj_path = out / "traj.jsonl"

    if not energy_path.exists():
        print("No energies.csv found.")
        return

    # --- Load data ---
    print("Loading energies...")
    steps, times, pe, ke, total = load_energies(energy_path)
    print(f"  {len(steps)} energy samples.")

    print("Loading trajectory...")
    traj_states = load_trajectory(traj_path) if traj_path.exists() else []
    print(f"  {len(traj_states)} frames loaded.")

    setup_style()
    fig_kw = dict(dpi=180, facecolor="#FAFAFA")

    # ======== 1. Energy time series — individual plots ========
    print("Plotting energy...")

    fig, ax = plt.subplots(figsize=(10, 4), **fig_kw)
    ax.plot(times, pe, color="#2166ac", linewidth=0.8)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Potential Energy (kJ/mol)")
    ax.set_title("Potential Energy vs Time")
    add_watermark(fig)
    fig.tight_layout()
    fig.savefig(plots_dir / "potential_energy.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), **fig_kw)
    ax.plot(times, ke, color="#4daf4a", linewidth=0.8)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Kinetic Energy (kJ/mol)")
    ax.set_title("Kinetic Energy vs Time")
    add_watermark(fig)
    fig.tight_layout()
    fig.savefig(plots_dir / "kinetic_energy.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), **fig_kw)
    ax.plot(times, total, color="#b2182b", linewidth=0.8)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Total Energy (kJ/mol)")
    ax.set_title("Total Energy vs Time")
    add_watermark(fig)
    fig.tight_layout()
    fig.savefig(plots_dir / "total_energy.png")
    plt.close(fig)

    if not traj_states:
        print("No trajectory frames — skipping remaining plots.")
        return

    # ======== 2. RMSD ========
    print("Computing RMSD...")
    rmsd_tracker = RMSDTracker(reference_positions=traj_states[0].particles.positions)
    rmsd_times, rmsd_vals = [], []
    for state in traj_states:
        rmsd_tracker.record(state.step, state)
        rmsd_times.append(state.time)
        rmsd_vals.append(rmsd_tracker._history[-1][1])

    fig, ax = plt.subplots(figsize=(10, 4), **fig_kw)
    ax.plot(rmsd_times, rmsd_vals, color="#d6604d", linewidth=0.8)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("RMSD (nm)")
    ax.set_title("Root Mean Square Deviation vs Time")
    add_watermark(fig)
    fig.tight_layout()
    fig.savefig(plots_dir / "rmsd.png")
    plt.close(fig)

    # ======== 3. RMSF per particle ========
    print("Computing RMSF...")
    rmsf = rmsd_tracker.rmsf_per_particle()
    if rmsf:
        fig, ax = plt.subplots(figsize=(10, 4), **fig_kw)
        ax.bar(range(len(rmsf)), rmsf, color="#4393c3", edgecolor="none")
        ax.set_xlabel("Bead Index")
        ax.set_ylabel("RMSF (nm)")
        ax.set_title("Root Mean Square Fluctuation per Bead")
        add_watermark(fig)
        fig.tight_layout()
        fig.savefig(plots_dir / "rmsf.png")
        plt.close(fig)

    # ======== 4. RDF ========
    print("Computing RDF...")
    rdf_calc = RDFCalculator(cutoff=2.0, n_bins=100)
    stride = max(1, len(traj_states) // 200)
    for i in range(0, len(traj_states), stride):
        rdf_calc.accumulate(traj_states[i])
    r_vals, g_vals = rdf_calc.compute_rdf()

    fig, ax = plt.subplots(figsize=(10, 4), **fig_kw)
    ax.plot(r_vals, g_vals, color="#762a83", linewidth=1.2)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Distance r (nm)")
    ax.set_ylabel("g(r)")
    ax.set_title("Radial Distribution Function")
    add_watermark(fig)
    fig.tight_layout()
    fig.savefig(plots_dir / "rdf.png")
    plt.close(fig)

    # ======== 5. SASA and Rg — individual plots ========
    print("Computing SASA and Rg...")
    sasa_calc = SASACalculator()
    rg_calc = RadiusOfGyration()
    sasa_times, sasa_vals = [], []
    rg_times, rg_vals = [], []
    obs_stride = max(1, len(traj_states) // 100)
    for i in range(0, len(traj_states), obs_stride):
        s = traj_states[i]
        positions = s.particles.positions
        masses = s.particles.masses
        n = len(positions)
        radii = tuple(0.24 for _ in range(n))
        sasa_result = sasa_calc.compute(positions, radii)
        rg_result = rg_calc.compute(positions, masses)
        sasa_times.append(s.time)
        sasa_vals.append(sasa_result.total_sasa)
        rg_times.append(s.time)
        rg_vals.append(rg_result.radius_of_gyration)

    fig, ax = plt.subplots(figsize=(10, 4), **fig_kw)
    ax.plot(sasa_times, sasa_vals, color="#e08214", linewidth=0.8)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("SASA (nm²)")
    ax.set_title("Solvent Accessible Surface Area vs Time")
    add_watermark(fig)
    fig.tight_layout()
    fig.savefig(plots_dir / "sasa.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), **fig_kw)
    ax.plot(rg_times, rg_vals, color="#1b7837", linewidth=0.8)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Radius of Gyration (nm)")
    ax.set_title("Radius of Gyration vs Time")
    add_watermark(fig)
    fig.tight_layout()
    fig.savefig(plots_dir / "rg.png")
    plt.close(fig)

    # ======== 5b. AA-level hydrogen bonds (on back-mapped structures) ========
    print("Computing AA-level hydrogen bonds from back-mapped structures...")
    try:
        bundle_path_hb = out / "prepared_bundle.json"
        aa_source_hb = REPO_ROOT / "benchmarks" / "reference_cases" / "data" / "1BRS.pdb"

        if bundle_path_hb.exists() and aa_source_hb.exists():
            from sampling.scenarios import ImportedProteinComplexScenario
            from scripts.live_dashboard import build_dashboard_context_for_scenario
            # Set up back-mapping for H-bond analysis
            aa_atoms_hb = _parse_aa_pdb(aa_source_hb)
            bd_hb = json.loads(bundle_path_hb.read_text())
            entities_hb = bd_hb.get("import_summary", {}).get("entities", [])
            bead_labels_hb = traj_states[0].particles.labels
            cg_init_hb = traj_states[0].particles.positions
            offsets_hb = _build_bead_to_atom_mapping(aa_atoms_hb, entities_hb, bead_labels_hb, cg_init_hb)
            weights_hb, mapped_hb = _precompute_atom_bead_weights(aa_atoms_hb, cg_init_hb, offsets_hb)

            # AA-level H-bond detection: backbone + side-chain
            # Donor-acceptor distance < 3.5 A AND angle > 120 degrees
            # Donors: backbone N (not PRO), Lys NZ, Arg NH1/NH2/NE,
            #         Asn ND2, Gln NE2, His ND1/NE2, Trp NE1,
            #         Ser OG, Thr OG1, Tyr OH
            # Acceptors: backbone O, Asp OD1/OD2, Glu OE1/OE2,
            #            Asn OD1, Gln OE1, His ND1/NE2, Ser OG,
            #            Thr OG1, Tyr OH
            import math as _math
            _HBOND_CUTOFF = 3.5  # Angstroms
            _HBOND_CUTOFF_SQ = _HBOND_CUTOFF ** 2
            _HBOND_ANGLE_MIN = 120.0  # degrees
            _COS_ANGLE_MIN = _math.cos(_math.radians(_HBOND_ANGLE_MIN))

            _SC_DONORS = {
                "ARG": {"NE", "NH1", "NH2"}, "ASN": {"ND2"}, "GLN": {"NE2"},
                "HIS": {"ND1", "NE2"}, "LYS": {"NZ"}, "SER": {"OG"},
                "THR": {"OG1"}, "TRP": {"NE1"}, "TYR": {"OH"},
            }
            _SC_ACCEPTORS = {
                "ASN": {"OD1"}, "ASP": {"OD1", "OD2"}, "GLN": {"OE1"},
                "GLU": {"OE1", "OE2"}, "HIS": {"ND1", "NE2"}, "SER": {"OG"},
                "THR": {"OG1"}, "TYR": {"OH"},
            }

            # Build per-residue donor/acceptor/CA indices
            _res_hb_info: dict[tuple[str, int], dict] = {}
            for ai, atom in enumerate(aa_atoms_hb):
                key = (atom["chain"], atom["resseq"])
                name = atom["name"]
                resname = atom["resname"]
                if key not in _res_hb_info:
                    _res_hb_info[key] = {
                        "resname": resname, "donors": [], "acceptors": [],
                        "ca": None,
                    }
                info = _res_hb_info[key]
                if name == "CA":
                    info["ca"] = ai
                if name == "N" and resname != "PRO":
                    info["donors"].append(ai)
                if name == "O":
                    info["acceptors"].append(ai)
                if resname in _SC_DONORS and name in _SC_DONORS[resname]:
                    info["donors"].append(ai)
                if resname in _SC_ACCEPTORS and name in _SC_ACCEPTORS[resname]:
                    info["acceptors"].append(ai)

            def _hbond_valid(coords, d_idx, a_idx, ca_idx):
                dx = coords[d_idx][0] - coords[a_idx][0]
                dy = coords[d_idx][1] - coords[a_idx][1]
                dz = coords[d_idx][2] - coords[a_idx][2]
                dsq = dx*dx + dy*dy + dz*dz
                if dsq > _HBOND_CUTOFF_SQ or dsq < 0.01:
                    return False
                if ca_idx is None:
                    return True
                # Angle: CA→D should be anti-parallel to D→A for good H-bond
                cd_x = coords[d_idx][0] - coords[ca_idx][0]
                cd_y = coords[d_idx][1] - coords[ca_idx][1]
                cd_z = coords[d_idx][2] - coords[ca_idx][2]
                da_x = coords[a_idx][0] - coords[d_idx][0]
                da_y = coords[a_idx][1] - coords[d_idx][1]
                da_z = coords[a_idx][2] - coords[d_idx][2]
                dot = cd_x*da_x + cd_y*da_y + cd_z*da_z
                m1sq = cd_x**2 + cd_y**2 + cd_z**2
                m2sq = da_x**2 + da_y**2 + da_z**2
                if m1sq < 0.01 or m2sq < 0.01:
                    return True
                cos_a = dot / _math.sqrt(m1sq * m2sq)
                return cos_a < _COS_ANGLE_MIN  # angle > 120°

            all_res_keys = sorted(_res_hb_info.keys())
            hbond_times, hbond_counts = [], []
            hbond_pair_presence: dict[tuple[str, int, str, int], list[int]] = {}
            hb_stride = max(1, len(traj_states) // 100)

            for fi in range(0, len(traj_states), hb_stride):
                coords = backmap_cg_to_aa(traj_states[fi], aa_atoms_hb, cg_init_hb,
                                          weights_hb, mapped_hb)
                count = 0
                observed: set[tuple[str, int, str, int]] = set()
                for rk_d in all_res_keys:
                    info_d = _res_hb_info[rk_d]
                    if not info_d["donors"]:
                        continue
                    for rk_a in all_res_keys:
                        if rk_d[0] == rk_a[0] and abs(rk_d[1] - rk_a[1]) <= 1:
                            continue  # skip same and adjacent residues in same chain
                        info_a = _res_hb_info[rk_a]
                        if not info_a["acceptors"]:
                            continue
                        found = False
                        for d_idx in info_d["donors"]:
                            for a_idx in info_a["acceptors"]:
                                if _hbond_valid(coords, d_idx, a_idx, info_d["ca"]):
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            count += 1
                            pair_key = (rk_d[0], rk_d[1], rk_a[0], rk_a[1])
                            observed.add(pair_key)

                hbond_times.append(traj_states[fi].time)
                hbond_counts.append(count)

                # Track occupancy
                all_pairs = set(hbond_pair_presence.keys()) | observed
                for pk in all_pairs:
                    if pk not in hbond_pair_presence:
                        hbond_pair_presence[pk] = [0] * (len(hbond_times) - 1)
                    hbond_pair_presence[pk].append(1 if pk in observed else 0)

            # H-bond count time series
            fig, ax = plt.subplots(figsize=(10, 4), dpi=150, facecolor="white")
            ax.plot(hbond_times, hbond_counts, color="#e41a1c", linewidth=0.8)
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("Number of H-bonds")
            ax.set_title("Hydrogen Bonds (AA back-mapped, D-A < 3.5 \u00c5, angle > 120\u00b0)")
            add_watermark(fig)
            fig.tight_layout()
            fig.savefig(plots_dir / "hbonds_timeseries.png")
            plt.close(fig)

            # H-bond occupancy — top pairs (with actual residue names)
            occupancy_aa: dict[str, float] = {}
            for pk, presence in hbond_pair_presence.items():
                n_frames = len(presence)
                if n_frames > 0:
                    occ = sum(presence) / n_frames
                    if occ > 0.1:  # only show pairs present >10% of frames
                        rn_d = _res_hb_info.get((pk[0], pk[1]), {}).get("resname", "?")
                        rn_a = _res_hb_info.get((pk[2], pk[3]), {}).get("resname", "?")
                        label = f"{pk[0]}:{rn_d}{pk[1]} \u2192 {pk[2]}:{rn_a}{pk[3]}"
                        occupancy_aa[label] = occ

            if occupancy_aa:
                sorted_occ = sorted(occupancy_aa.items(), key=lambda x: -x[1])[:25]
                pair_labels = [p for p, _ in sorted_occ]
                occ_vals = [v for _, v in sorted_occ]
                fig, ax = plt.subplots(figsize=(12, max(4, len(pair_labels) * 0.25)),
                                       dpi=150, facecolor="white")
                ax.barh(range(len(pair_labels)), occ_vals, color="#377eb8")
                ax.set_yticks(range(len(pair_labels)))
                ax.set_yticklabels(pair_labels, fontsize=7)
                ax.set_xlabel("Occupancy (fraction of frames)")
                ax.set_title("Top H-bond Pair Occupancy (AA level, backbone + side-chain)")
                ax.invert_yaxis()
                fig.tight_layout()
                fig.savefig(plots_dir / "hbond_occupancy.png")
                plt.close(fig)

            # ======== 5c. Per-residue energy decomposition (CG level) ========
            print("Computing residue energy decomposition...")
            from prepare import PreparationPipeline
            bundle_obj = PreparationPipeline().load_bundle(bundle_path_hb)
            scenario_obj = ImportedProteinComplexScenario(spec=bundle_obj.scenario_spec)
            context_obj = build_dashboard_context_for_scenario(
                scenario_obj, initial_state_override=traj_states[0])
            topology = context_obj.engine.setup.topology

            energy_decomp = ResidueEnergyDecomposition()
            last_state = traj_states[-1]
            n_part = last_state.particle_count
            dummy_forces = tuple((0.0, 0.0, 0.0) for _ in range(n_part))
            last_pe = last_state.potential_energy if last_state.potential_energy is not None else 0.0
            decomp = energy_decomp.decompose(last_state, topology, dummy_forces, last_pe)

            bead_indices = list(range(n_part))

            fig, ax = plt.subplots(figsize=(12, 4), dpi=150, facecolor="white")
            ax.bar(bead_indices, decomp.per_particle_bonded, color="#66c2a5", edgecolor="none")
            ax.set_xlabel("Bead Index")
            ax.set_ylabel("Bonded Energy (kJ/mol)")
            ax.set_title("Per-Bead Bonded Energy Decomposition")
            add_watermark(fig)
            fig.tight_layout()
            fig.savefig(plots_dir / "energy_decomposition_bonded.png")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(12, 4), dpi=150, facecolor="white")
            ax.bar(bead_indices, decomp.per_particle_nonbonded, color="#fc8d62", edgecolor="none")
            ax.set_xlabel("Bead Index")
            ax.set_ylabel("Nonbonded Energy (kJ/mol)")
            ax.set_title("Per-Bead Nonbonded Energy Decomposition")
            add_watermark(fig)
            fig.tight_layout()
            fig.savefig(plots_dir / "energy_decomposition_nonbonded.png")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(12, 4), dpi=150, facecolor="white")
            ax.bar(bead_indices, decomp.per_particle_total, color="#8da0cb", edgecolor="none")
            ax.set_xlabel("Bead Index")
            ax.set_ylabel("Total Energy (kJ/mol)")
            ax.set_title("Per-Bead Total Energy Decomposition")
            add_watermark(fig)
            fig.tight_layout()
            fig.savefig(plots_dir / "energy_decomposition_total.png")
            plt.close(fig)

            # ======== 5d. Inter-entity interaction energy ========
            print("Computing inter-entity interaction energies...")
            entities_info_ie = bd_hb.get("import_summary", {}).get("entities", [])
            if len(entities_info_ie) >= 2:
                entity_groups_idx = []
                offset_ie = 0
                for ent in entities_info_ie:
                    bc = ent.get("bead_count", 0)
                    entity_groups_idx.append(tuple(range(offset_ie, offset_ie + bc)))
                    offset_ie += bc

                ie_times, ie_vals = [], []
                ie_stride = max(1, len(traj_states) // 50)
                for i in range(0, len(traj_states), ie_stride):
                    s = traj_states[i]
                    total_ie = 0.0
                    for gi in range(len(entity_groups_idx)):
                        for gj in range(gi + 1, len(entity_groups_idx)):
                            total_ie += energy_decomp.interaction_energy_between_groups(
                                s, topology, entity_groups_idx[gi], entity_groups_idx[gj])
                    ie_times.append(s.time)
                    ie_vals.append(total_ie)

                fig, ax = plt.subplots(figsize=(10, 4), dpi=150, facecolor="white")
                ax.plot(ie_times, ie_vals, color="#984ea3", linewidth=1.0)
                ax.set_xlabel("Time (ps)")
                ax.set_ylabel("Interaction Energy (kJ/mol)")
                ax.set_title("Inter-Entity Interaction Energy vs Time")
                add_watermark(fig)
                fig.tight_layout()
                fig.savefig(plots_dir / "interaction_energy.png")
                plt.close(fig)

            # ======== 5e. QCloud event analysis (from run_summary) ========
            print("Loading QCloud event analysis...")
            run_summary_path = out / "run_summary.json"
            if run_summary_path.exists():
                rs_data = json.loads(run_summary_path.read_text())
                rs_meta = rs_data.get("metadata", {})
                qcloud_summary = rs_meta.get("qcloud_event_analysis", {})
                qcloud_events = rs_meta.get("qcloud_detected_events", [])

                if qcloud_summary.get("total_correction_cycles", 0) > 0 or qcloud_events:
                    # QCloud correction cycle count bar chart
                    summary_labels = [
                        "Bond Forming", "Bond Breaking",
                        "Conformational Shift", "Interface Rearrangement",
                    ]
                    summary_vals = [
                        qcloud_summary.get("bond_forming_events", 0),
                        qcloud_summary.get("bond_breaking_events", 0),
                        qcloud_summary.get("conformational_shifts", 0),
                        qcloud_summary.get("interface_rearrangements", 0),
                    ]
                    fig, ax = plt.subplots(figsize=(8, 4), dpi=150, facecolor="white")
                    colors_ev = ["#66BB6A", "#EF5350", "#42A5F5", "#FFA726"]
                    ax.bar(summary_labels, summary_vals, color=colors_ev)
                    ax.set_ylabel("Event Count")
                    ax.set_title(f"QCloud Detected Structural Events "
                                 f"({qcloud_summary.get('total_events_detected', 0)} total, "
                                 f"{qcloud_summary.get('total_correction_cycles', 0)} cycles)")
                    fig.tight_layout()
                    fig.savefig(plots_dir / "qcloud_events.png")
                    plt.close(fig)

                    # Event timeline
                    if qcloud_events:
                        ev_times = [e["time"] for e in qcloud_events]
                        ev_mags = [e["correction_magnitude"] for e in qcloud_events]
                        ev_kinds = [e["kind"] for e in qcloud_events]
                        kind_colors = {
                            "bond_forming": "#66BB6A",
                            "bond_breaking": "#EF5350",
                            "conformational_shift": "#42A5F5",
                            "interface_rearrangement": "#FFA726",
                        }
                        fig, ax = plt.subplots(figsize=(10, 4), dpi=150, facecolor="white")
                        for i, (t, m, k) in enumerate(zip(ev_times, ev_mags, ev_kinds)):
                            ax.scatter(t, m, c=kind_colors.get(k, "gray"), s=60,
                                       edgecolors="black", linewidths=0.5, zorder=3)
                        ax.set_xlabel("Time (ps)")
                        ax.set_ylabel("Correction Magnitude")
                        ax.set_title("QCloud Structural Events Timeline")
                        # Legend
                        from matplotlib.lines import Line2D
                        legend_elements = [
                            Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=c, markersize=8, label=k.replace("_", " ").title())
                            for k, c in kind_colors.items()
                        ]
                        ax.legend(handles=legend_elements, fontsize=8)
                        fig.tight_layout()
                        fig.savefig(plots_dir / "qcloud_event_timeline.png")
                        plt.close(fig)

                    print(f"  QCloud events: {qcloud_summary.get('total_events_detected', 0)} detected")
                else:
                    print("  No QCloud correction cycles recorded in run summary.")
            else:
                print("  No run_summary.json found (run analyze after run completes).")

            print("  H-bonds, energy decomposition, interaction energy, and QCloud events done.")
        else:
            print("  Skipping (no prepared bundle or AA reference).")
            topology = None
    except Exception as exc:
        import traceback
        print(f"  Warning: H-bond/energy analysis failed: {exc}")
        traceback.print_exc()
        topology = None

    # ======== 6. Reaction coordinate + PMF ========
    print("Computing reaction coordinate and PMF...")
    bundle_path = out / "prepared_bundle.json"
    group_a, group_b = [], []
    if bundle_path.exists():
        bd = json.loads(bundle_path.read_text())
        entities = bd.get("import_summary", {}).get("entities", [])
        if len(entities) >= 2:
            offset = 0
            for idx, ent in enumerate(entities):
                bc = ent.get("bead_count", 0)
                if idx == 0:
                    group_a = list(range(offset, offset + bc))
                elif idx == 1:
                    group_b = list(range(offset, offset + bc))
                offset += bc
    if not group_a and traj_states:
        n = traj_states[0].particle_count
        group_a = list(range(n // 2))
        group_b = list(range(n // 2, n))

    cv_history, cv_times = [], []
    for state in traj_states:
        pos = state.particles.positions
        masses = state.particles.masses
        def _com(indices):
            tm = sum(masses[i] for i in indices)
            if tm == 0: tm = 1.0
            return (
                sum(masses[i] * pos[i][0] for i in indices) / tm,
                sum(masses[i] * pos[i][1] for i in indices) / tm,
                sum(masses[i] * pos[i][2] for i in indices) / tm,
            )
        ca, cb = _com(group_a), _com(group_b)
        d = sqrt(sum((ca[i] - cb[i]) ** 2 for i in range(3)))
        cv_history.append(d)
        cv_times.append(state.time)

    fig, ax = plt.subplots(figsize=(10, 4), **fig_kw)
    ax.plot(cv_times, cv_history, color="#e7298a", linewidth=0.8)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Center-of-Mass Distance (nm)")
    ax.set_title("Inter-Entity COM Distance vs Time")
    add_watermark(fig)
    fig.tight_layout()
    fig.savefig(plots_dir / "reaction_coordinate.png")
    plt.close(fig)

    pmf_r, pmf_v = [], []
    if cv_history:
        n_bins = 50
        cv_min, cv_max = min(cv_history), max(cv_history)
        if cv_max > cv_min:
            bw = (cv_max - cv_min) / n_bins
            hist = [0] * n_bins
            for cv in cv_history:
                b = min(int((cv - cv_min) / bw), n_bins - 1)
                hist[b] += 1
            max_c = max(hist)
            for i in range(n_bins):
                pmf_r.append(cv_min + (i + 0.5) * bw)
                if hist[i] > 0:
                    pmf_v.append(-log(hist[i] / max_c))
                else:
                    pmf_v.append(float("nan"))
            fig, ax = plt.subplots(figsize=(10, 4), **fig_kw)
            ax.plot(pmf_r, pmf_v, color="#7570b3", linewidth=1.5, marker="o", markersize=3)
            ax.set_xlabel("COM Distance (nm)")
            ax.set_ylabel("Free Energy (kT)")
            ax.set_title("Potential of Mean Force (Boltzmann Inversion)")
            add_watermark(fig)
            fig.tight_layout()
            fig.savefig(plots_dir / "pmf.png")
            plt.close(fig)

    # ======== 7. Structure snapshots ========
    print("Plotting structure snapshots...")
    bundle_path = out / "prepared_bundle.json"
    plot_structure_snapshots(traj_states, bundle_path, plots_dir)

    # ======== 7b. Automatic binding interaction analysis ========
    # (Moved after back-mapping — see section 8c below)

    # ======== 8. Export CG trajectory PDB ========
    print("Exporting CG trajectory PDB...")
    # Full trajectory (sampled to keep file size reasonable)
    traj_stride = max(1, len(traj_states) // 200)
    sampled_states = [traj_states[i] for i in range(0, len(traj_states), traj_stride)]
    write_cg_trajectory_pdb(sampled_states, bundle_path, out / "cg_trajectory.pdb")
    print(f"  Wrote {len(sampled_states)} frames to cg_trajectory.pdb")

    # First and last snapshots
    write_cg_snapshot_pdb(traj_states[0], bundle_path, out / "cg_initial.pdb", title="Initial CG structure")
    write_cg_snapshot_pdb(traj_states[-1], bundle_path, out / "cg_final.pdb", title="Final CG structure")
    print("  Wrote cg_initial.pdb and cg_final.pdb")

    # Copy reference AA PDB if available
    aa_source = REPO_ROOT / "benchmarks" / "reference_cases" / "data" / "1BRS.pdb"
    aa_dest = out / "aa_reference.pdb"
    if aa_source.exists() and not aa_dest.exists():
        import shutil
        shutil.copy2(aa_source, aa_dest)
        print(f"  Copied AA reference structure to aa_reference.pdb")

    # ======== 8b. Back-map CG → AA trajectory ========
    if aa_source.exists():
        print("Back-mapping CG → AA trajectory...")
        aa_atoms = _parse_aa_pdb(aa_source)
        bd = json.loads(bundle_path.read_text()) if bundle_path.exists() else {}
        entities = bd.get("import_summary", {}).get("entities", [])
        bead_labels = traj_states[0].particles.labels
        cg_initial_positions = traj_states[0].particles.positions
        bead_atom_offsets = _build_bead_to_atom_mapping(aa_atoms, entities, bead_labels, cg_initial_positions)

        # Precompute interpolation weights (distance-weighted from nearest beads)
        print("  Precomputing interpolation weights...")
        atom_weights, mapped_atoms = _precompute_atom_bead_weights(
            aa_atoms, cg_initial_positions, bead_atom_offsets)

        def _backmap(state):
            return backmap_cg_to_aa(state, aa_atoms, cg_initial_positions,
                                    atom_weights, mapped_atoms)

        # Write initial + final back-mapped AA snapshots
        write_aa_backmapped_pdb(
            aa_atoms, _backmap(traj_states[0]),
            out / "aa_backmapped_initial.pdb",
            title="Back-mapped AA structure (initial)",
        )
        write_aa_backmapped_pdb(
            aa_atoms, _backmap(traj_states[-1]),
            out / "aa_backmapped_final.pdb",
            title="Back-mapped AA structure (final)",
        )
        print("  Wrote aa_backmapped_initial.pdb and aa_backmapped_final.pdb")

        # Combined: initial + final in one file
        combined_lines = ["REMARK   Combined initial and final back-mapped AA structures"]
        combined_lines.append("REMARK   MODEL 1 = initial (t=0), MODEL 2 = final")
        for model_idx, frame_state in enumerate([traj_states[0], traj_states[-1]]):
            combined_lines.append(f"MODEL     {model_idx + 1:>4d}")
            coords = _backmap(frame_state)
            for ai, atom in enumerate(aa_atoms):
                x, y, z = coords[ai]
                old_line = atom["line"]
                combined_lines.append(f"{old_line[:30]}{x:>8.3f}{y:>8.3f}{z:>8.3f}{old_line[54:]}")
            combined_lines.append("ENDMDL")
        combined_lines.append("END")
        (out / "aa_backmapped_combined.pdb").write_text("\n".join(combined_lines), encoding="utf-8")
        print("  Wrote aa_backmapped_combined.pdb (initial + final in one file)")

        # Full back-mapped AA trajectory (sampled)
        traj_sample_stride = max(1, len(traj_states) // 100)
        sampled_for_aa = [traj_states[i] for i in range(0, len(traj_states), traj_sample_stride)]
        write_aa_backmapped_trajectory(sampled_for_aa, aa_atoms, cg_initial_positions,
                                       atom_weights, mapped_atoms,
                                       out / "aa_backmapped_trajectory.pdb")
        print(f"  Wrote aa_backmapped_trajectory.pdb ({len(sampled_for_aa)} frames)")

    # ======== 8c. Binding interaction analysis (with AA back-mapping) ========
    bundle_path_bind = out / "prepared_bundle.json"
    if bundle_path_bind.exists():
        print("Running binding interaction analysis (CG + AA cooperative)...")
        try:
            # Pass AA back-mapping data if available so binding analysis
            # runs at the AA residue level (carries QCloud + CG info)
            if aa_source.exists():
                aa_atoms_bind = _parse_aa_pdb(aa_source)
                bd_bind = json.loads(bundle_path_bind.read_text())
                entities_bind = bd_bind.get("import_summary", {}).get("entities", [])
                bead_labels_bind = traj_states[0].particles.labels
                cg_init_bind = traj_states[0].particles.positions
                offsets_bind = _build_bead_to_atom_mapping(
                    aa_atoms_bind, entities_bind, bead_labels_bind, cg_init_bind)
                weights_bind, mapped_bind = _precompute_atom_bead_weights(
                    aa_atoms_bind, cg_init_bind, offsets_bind)

                def _backmap_bind(state):
                    return backmap_cg_to_aa(state, aa_atoms_bind, cg_init_bind,
                                           weights_bind, mapped_bind)

                run_binding_analysis(
                    traj_states, bundle_path_bind, plots_dir,
                    aa_atoms=aa_atoms_bind,
                    cg_initial_positions=cg_init_bind,
                    atom_weights=weights_bind,
                    mapped_atoms=mapped_bind,
                    backmap_fn=_backmap_bind,
                )
            else:
                run_binding_analysis(traj_states, bundle_path_bind, plots_dir)
        except Exception as exc:
            import traceback
            print(f"  Warning: Binding analysis failed: {exc}")
            traceback.print_exc()

    # ======== Summary ========
    print("\n" + "=" * 70)
    print("  NeuroCGMD Analysis Complete — Cooperative Architecture Output")
    print("=" * 70)

    print("\n  CG-LEVEL (collective dynamics):")
    print("  ├── potential_energy.png              Potential energy vs time")
    print("  ├── kinetic_energy.png                Kinetic energy vs time")
    print("  ├── total_energy.png                  Total energy vs time")
    print("  ├── rmsd.png                          RMSD vs time")
    print("  ├── rmsf.png                          RMSF per bead")
    print("  ├── rdf.png                           Radial distribution function")
    print("  ├── sasa.png                          Solvent-accessible surface area")
    print("  ├── rg.png                            Radius of gyration")
    print("  ├── reaction_coordinate.png           COM distance vs time")
    print("  ├── pmf.png                           Potential of mean force")
    print("  ├── energy_decomposition_bonded.png   Per-bead bonded energy")
    print("  ├── energy_decomposition_nonbonded.png Per-bead nonbonded energy")
    print("  ├── energy_decomposition_total.png    Per-bead total energy")
    print("  ├── interaction_energy.png            Inter-entity interaction energy")
    print("  ├── free_energy_landscape_2d.png      COM distance vs Rg landscape")
    print("  ├── contact_map.png                   CG bead-bead contact frequency")
    print("  ├── binding_energy_pairs.png          Binding energy per chain pair")
    print("  ├── binding_energy_summary.png        Mean binding energies bar chart")
    print("  ├── binding_dashboard.png             4-panel binding overview")
    print("  ├── interface_residues.png            Interface bead identification")
    print("  └── structure_snapshots.png           CG structure evolution")

    print("\n  AA-LEVEL (back-mapped from CG+QCloud):")
    print("  ├── hbonds_timeseries.png             H-bond count over time")
    print("  ├── hbond_occupancy.png               Top H-bond pair occupancy")
    print("  ├── aa_contact_map_*.png              Residue-residue contact map")
    print("  ├── aa_top_residue_pairs_*.png        Top interface pairs")
    print("  ├── aa_residue_binding_*.png          Per-residue binding contribution")
    print("  └── aa_interface_hbonds_*.png         Inter-chain H-bond network")

    print("\n  QCLOUD (quantum correction feedback):")
    print("  ├── qcloud_events.png                 Detected structural events")
    print("  └── qcloud_event_timeline.png         Event timeline")

    print("\n  TRAJECTORIES:")
    print("  ├── cg_trajectory.pdb            CG multi-model trajectory")
    print("  ├── cg_initial.pdb / cg_final.pdb")
    print("  ├── aa_backmapped_trajectory.pdb Full AA trajectory (CG+QCloud→AA)")
    print("  ├── aa_backmapped_initial/final.pdb")
    print("  ├── aa_backmapped_combined.pdb   Initial+final in one file")
    print("  └── aa_reference.pdb             Original crystal structure")

    print("\n  Architecture: CG dynamics → QCloud corrections → ML residual")
    print("  → back-map to AA. One coherent information stream.")
    print("=" * 70)


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/barnase_barstar_20ns"
    main(output_dir)
