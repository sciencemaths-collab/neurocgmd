"""Automatic binding interaction analysis and visualization.

Detects system type (multi-chain, protein-ligand, complex assembly) and
generates appropriate binding analysis plots:
- Inter-chain contact maps
- Pairwise binding energy decomposition
- Interface residue identification
- H-bond network at the binding interface
- Binding free energy timeline
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import sqrt, log
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from scripts.plot_style import (
    CHAIN_COLORS, ENERGY_COLORS, CONTACT_CMAP, ENERGY_LANDSCAPE_CMAP,
    setup_style, styled_figure, add_watermark, smooth,
)


@dataclass
class EntityInfo:
    entity_id: str
    chain_ids: list[str]
    bead_count: int
    bead_indices: list[int]


def _load_entities(bundle_path: Path) -> list[EntityInfo]:
    """Parse entity info from prepared bundle."""
    bd = json.loads(bundle_path.read_text())
    entities = bd.get("import_summary", {}).get("entities", [])
    result = []
    offset = 0
    for ent in entities:
        bc = ent.get("bead_count", 0)
        result.append(EntityInfo(
            entity_id=ent.get("entity_id", f"entity_{len(result)}"),
            chain_ids=ent.get("chain_ids", []),
            bead_count=bc,
            bead_indices=list(range(offset, offset + bc)),
        ))
        offset += bc
    return result


def _distance(a, b) -> float:
    return sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def plot_inter_chain_contact_map(traj_states, entities: list[EntityInfo],
                                  plots_dir: Path, contact_cutoff: float = 1.2):
    """Plot contact frequency map between all entity pairs."""
    n_total = sum(e.bead_count for e in entities)
    contact_freq = [[0.0] * n_total for _ in range(n_total)]
    n_frames = len(traj_states)
    stride = max(1, n_frames // 100)

    for fi in range(0, n_frames, stride):
        pos = traj_states[fi].particles.positions
        for i in range(n_total):
            for j in range(i + 1, n_total):
                d = _distance(pos[i], pos[j])
                if d < contact_cutoff:
                    contact_freq[i][j] += 1
                    contact_freq[j][i] += 1

    n_sampled = len(range(0, n_frames, stride))
    for i in range(n_total):
        for j in range(n_total):
            contact_freq[i][j] /= max(n_sampled, 1)

    arr = np.array(contact_freq)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(arr, cmap=CONTACT_CMAP, aspect="equal", origin="lower",
                   vmin=0, vmax=1, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Contact Frequency", fontsize=10)

    # Draw entity boundaries
    offset = 0
    for i, ent in enumerate(entities):
        ax.axhline(y=offset - 0.5, color="#333", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axvline(x=offset - 0.5, color="#333", linewidth=0.8, linestyle="--", alpha=0.5)
        mid = offset + ent.bead_count / 2
        label = "/".join(ent.chain_ids)
        ax.text(mid, -2.5, label, ha="center", fontsize=8, fontweight="bold",
                color=CHAIN_COLORS[i % len(CHAIN_COLORS)])
        ax.text(-2.5, mid, label, ha="center", va="center", fontsize=8, fontweight="bold",
                color=CHAIN_COLORS[i % len(CHAIN_COLORS)], rotation=90)
        offset += ent.bead_count

    ax.set_title("Inter-Chain Contact Map", fontsize=13, fontweight="bold")
    ax.set_xlabel("Bead Index")
    ax.set_ylabel("Bead Index")
    add_watermark(fig)
    fig.savefig(plots_dir / "contact_map.png")
    plt.close(fig)


def plot_pairwise_binding_energy(traj_states, entities: list[EntityInfo],
                                  plots_dir: Path):
    """Plot binding energy between biological binding pairs over time."""
    if len(entities) < 2:
        return

    epsilon = 2.0
    sigma = 0.47

    bio_pairs = _identify_biological_pairs(entities)

    pair_labels = []
    pair_energies: dict[str, list[float]] = {}
    pair_times: list[float] = []

    for ei, ej in bio_pairs:
        label = f"{'/'.join(entities[ei].chain_ids)}-{'/'.join(entities[ej].chain_ids)}"
        pair_labels.append(label)
        pair_energies[label] = []

    stride = max(1, len(traj_states) // 80)
    for fi in range(0, len(traj_states), stride):
        pos = traj_states[fi].particles.positions
        pair_times.append(traj_states[fi].time)
        for ei, ej in bio_pairs:
            label = f"{'/'.join(entities[ei].chain_ids)}-{'/'.join(entities[ej].chain_ids)}"
            ie = 0.0
            for i in entities[ei].bead_indices:
                for j in entities[ej].bead_indices:
                    r = _distance(pos[i], pos[j])
                    if r < 1e-9:
                        continue
                    sr6 = (sigma / r) ** 6
                    ie += 4.0 * epsilon * (sr6 * sr6 - sr6)
            pair_energies[label].append(ie)

    # Plot each pair
    n_pairs = len(pair_labels)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(10, 3 * n_pairs), sharex=True)
    if n_pairs == 1:
        axes = [axes]

    for idx, label in enumerate(pair_labels):
        color = CHAIN_COLORS[idx % len(CHAIN_COLORS)]
        vals = pair_energies[label]
        axes[idx].plot(pair_times, vals, color=color, linewidth=1.0, alpha=0.4)
        axes[idx].plot(pair_times, smooth(vals, 5), color=color, linewidth=1.8)
        axes[idx].set_ylabel("E (kJ/mol)", fontsize=9)
        axes[idx].set_title(f"Binding: {label}", fontsize=11, fontweight="bold", color=color)
        axes[idx].axhline(y=0, color="#999", linewidth=0.5, linestyle="--")

    axes[-1].set_xlabel("Time (ps)")
    fig.suptitle("Pairwise Binding Energy Decomposition", fontsize=14, fontweight="bold", y=1.01)
    add_watermark(fig)
    fig.tight_layout()
    fig.savefig(plots_dir / "binding_energy_pairs.png")
    plt.close(fig)

    # Summary bar chart of mean binding energies
    fig, ax = plt.subplots(figsize=(max(6, n_pairs * 1.5), 5))
    means = [sum(pair_energies[l]) / len(pair_energies[l]) if pair_energies[l] else 0 for l in pair_labels]
    colors = [CHAIN_COLORS[i % len(CHAIN_COLORS)] for i in range(n_pairs)]
    bars = ax.bar(pair_labels, means, color=colors, edgecolor="#333", linewidth=0.5)
    ax.axhline(y=0, color="#999", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Mean Binding Energy (kJ/mol)")
    ax.set_title("Mean Pairwise Binding Energies", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=9, fontweight="bold")
    add_watermark(fig)
    fig.tight_layout()
    fig.savefig(plots_dir / "binding_energy_summary.png")
    plt.close(fig)


def plot_interface_residues(traj_states, entities: list[EntityInfo],
                            plots_dir: Path, contact_cutoff: float = 1.0):
    """Identify and plot interface residues — beads that are in contact
    with another entity at least 50% of the time."""
    if len(entities) < 2:
        return

    n_total = sum(e.bead_count for e in entities)
    # Track per-bead inter-entity contact frequency
    inter_contact_count = [0] * n_total
    n_frames = len(traj_states)
    stride = max(1, n_frames // 100)
    n_sampled = 0

    entity_of_bead = {}
    for ei, ent in enumerate(entities):
        for bi in ent.bead_indices:
            entity_of_bead[bi] = ei

    for fi in range(0, n_frames, stride):
        pos = traj_states[fi].particles.positions
        n_sampled += 1
        for i in range(n_total):
            for j in range(i + 1, n_total):
                if entity_of_bead[i] == entity_of_bead[j]:
                    continue
                d = _distance(pos[i], pos[j])
                if d < contact_cutoff:
                    inter_contact_count[i] += 1
                    inter_contact_count[j] += 1

    freq = [c / max(n_sampled, 1) for c in inter_contact_count]

    fig, ax = plt.subplots(figsize=(12, 4))
    offset = 0
    for ei, ent in enumerate(entities):
        color = CHAIN_COLORS[ei % len(CHAIN_COLORS)]
        indices = list(range(offset, offset + ent.bead_count))
        vals = [freq[i] for i in ent.bead_indices]
        ax.bar(indices, vals, color=color, edgecolor="none", alpha=0.85,
               label="/".join(ent.chain_ids))
        offset += ent.bead_count

    ax.axhline(y=0.5, color="#E63946", linewidth=1.0, linestyle="--",
               alpha=0.7, label="Interface threshold (50%)")
    ax.set_xlabel("Bead Index")
    ax.set_ylabel("Inter-Entity Contact Frequency")
    ax.set_title("Interface Residues (Beads in Cross-Entity Contact)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    add_watermark(fig)
    fig.tight_layout()
    fig.savefig(plots_dir / "interface_residues.png")
    plt.close(fig)


def plot_2d_free_energy_landscape(traj_states, entities: list[EntityInfo],
                                   plots_dir: Path):
    """Plot 2D free energy landscape: COM distance vs Rg."""
    if len(entities) < 2:
        return

    com_dists = []
    rg_vals = []

    for state in traj_states:
        pos = state.particles.positions
        masses = state.particles.masses

        def _com(indices):
            tm = sum(masses[i] for i in indices)
            if tm == 0:
                tm = 1.0
            return tuple(sum(masses[i] * pos[i][d] for i in indices) / tm for d in range(3))

        com_a = _com(entities[0].bead_indices)
        com_b = _com(entities[1].bead_indices)
        com_dists.append(_distance(com_a, com_b))

        # Overall Rg
        all_indices = [i for e in entities for i in e.bead_indices]
        com_all = _com(all_indices)
        rg_sq = sum(masses[i] * sum((pos[i][d] - com_all[d]) ** 2 for d in range(3))
                     for i in all_indices) / sum(masses[i] for i in all_indices)
        rg_vals.append(sqrt(max(0, rg_sq)))

    # 2D histogram → free energy
    n_bins = 40
    x = np.array(com_dists)
    y = np.array(rg_vals)

    hist, xedges, yedges = np.histogram2d(x, y, bins=n_bins)
    hist = hist.T
    # Convert to free energy: F = -ln(P)
    hist_max = hist.max()
    with np.errstate(divide="ignore"):
        fe = -np.log(hist / hist_max)
    fe[np.isinf(fe)] = np.nan

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(fe, origin="lower", aspect="auto",
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap=ENERGY_LANDSCAPE_CMAP, interpolation="bicubic")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Free Energy (kT)", fontsize=10)

    # Overlay trajectory path
    ax.plot(com_dists, rg_vals, color="white", linewidth=0.3, alpha=0.3)
    ax.scatter(com_dists[0], rg_vals[0], c="#2A9D8F", s=80, edgecolors="white",
               linewidths=1.5, zorder=5, label="Start")
    ax.scatter(com_dists[-1], rg_vals[-1], c="#E63946", s=80, edgecolors="white",
               linewidths=1.5, zorder=5, label="End")

    ax.set_xlabel(f"COM Distance ({'/'.join(entities[0].chain_ids)}"
                  f"-{'/'.join(entities[1].chain_ids)}) (nm)")
    ax.set_ylabel("Radius of Gyration (nm)")
    ax.set_title("2D Free Energy Landscape", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    add_watermark(fig)
    fig.tight_layout()
    fig.savefig(plots_dir / "free_energy_landscape_2d.png")
    plt.close(fig)


def plot_binding_interaction_dashboard(traj_states, entities: list[EntityInfo],
                                       plots_dir: Path):
    """All-in-one binding interaction dashboard."""
    if len(entities) < 2:
        return

    epsilon, sigma = 2.0, 0.47
    contact_cutoff = 1.0

    # Compute time series
    times, com_dists, n_contacts_list, ie_list = [], [], [], []
    entity_of_bead = {}
    for ei, ent in enumerate(entities):
        for bi in ent.bead_indices:
            entity_of_bead[bi] = ei

    n_total = sum(e.bead_count for e in entities)
    stride = max(1, len(traj_states) // 100)

    for fi in range(0, len(traj_states), stride):
        state = traj_states[fi]
        pos = state.particles.positions
        masses = state.particles.masses
        times.append(state.time)

        # COM distance (first two entities)
        def _com(indices):
            tm = sum(masses[i] for i in indices)
            if tm == 0: tm = 1.0
            return tuple(sum(masses[i] * pos[i][d] for i in indices) / tm for d in range(3))

        com_a = _com(entities[0].bead_indices)
        com_b = _com(entities[1].bead_indices)
        com_dists.append(_distance(com_a, com_b))

        # Inter-entity contacts and interaction energy
        n_contacts = 0
        total_ie = 0.0
        for i in range(n_total):
            for j in range(i + 1, n_total):
                if entity_of_bead.get(i) == entity_of_bead.get(j):
                    continue
                r = _distance(pos[i], pos[j])
                if r < contact_cutoff:
                    n_contacts += 1
                if r > 1e-9:
                    sr6 = (sigma / r) ** 6
                    total_ie += 4.0 * epsilon * (sr6 * sr6 - sr6)
        n_contacts_list.append(n_contacts)
        ie_list.append(total_ie)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)

    # Panel 1: COM distance
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times, com_dists, color="#457B9D", linewidth=0.6, alpha=0.4)
    ax1.plot(times, smooth(com_dists, 5), color="#1D3557", linewidth=1.8)
    ax1.set_ylabel("COM Distance (nm)")
    ax1.set_title("Binding Progress", fontweight="bold")
    ax1.set_xlabel("Time (ps)")

    # Panel 2: Interface contacts
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(times, n_contacts_list, color="#2A9D8F", alpha=0.3)
    ax2.plot(times, smooth(n_contacts_list, 5), color="#264653", linewidth=1.8)
    ax2.set_ylabel("Inter-Entity Contacts")
    ax2.set_title("Interface Formation", fontweight="bold")
    ax2.set_xlabel("Time (ps)")

    # Panel 3: Interaction energy
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(times, ie_list, color="#E9C46A", linewidth=0.6, alpha=0.4)
    ax3.plot(times, smooth(ie_list, 5), color="#E76F51", linewidth=1.8)
    ax3.axhline(y=0, color="#999", linewidth=0.5, linestyle="--")
    ax3.set_ylabel("Interaction Energy (kJ/mol)")
    ax3.set_title("Binding Energetics", fontweight="bold")
    ax3.set_xlabel("Time (ps)")

    # Panel 4: COM distance vs contacts scatter
    ax4 = fig.add_subplot(gs[1, 1])
    colors_sc = np.linspace(0, 1, len(times))
    sc = ax4.scatter(com_dists, n_contacts_list, c=colors_sc, cmap="coolwarm",
                     s=20, edgecolors="none", alpha=0.7)
    ax4.set_xlabel("COM Distance (nm)")
    ax4.set_ylabel("Interface Contacts")
    ax4.set_title("Binding Correlation", fontweight="bold")
    cbar = fig.colorbar(sc, ax=ax4, shrink=0.8)
    cbar.set_label("Time progression", fontsize=8)

    fig.suptitle("Binding Interaction Analysis", fontsize=15, fontweight="bold", y=1.01)
    add_watermark(fig)
    fig.tight_layout()
    fig.savefig(plots_dir / "binding_dashboard.png")
    plt.close(fig)


def _identify_biological_pairs(entities: list[EntityInfo]) -> list[tuple[int, int]]:
    """For complex assemblies (e.g. 1BRS with 3 barnase + 3 barstar copies),
    identify the biologically relevant binding pairs instead of all N*(N-1)/2.

    Heuristic: group entities by approximate bead count (within 20% = same type),
    then pair different-type entities 1:1 by index order.
    Falls back to all pairs if grouping fails."""
    if len(entities) <= 2:
        return [(0, 1)] if len(entities) == 2 else []

    # Group by bead count: exact match first, then merge groups within 15%
    exact_groups: dict[int, list[int]] = {}
    for i, ent in enumerate(entities):
        bc = ent.bead_count
        if bc not in exact_groups:
            exact_groups[bc] = []
        exact_groups[bc].append(i)

    # Merge groups whose bead counts are within 15% of each other
    group_keys = sorted(exact_groups.keys())
    groups: list[list[int]] = []
    merged: set[int] = set()
    for k in group_keys:
        if k in merged:
            continue
        current = list(exact_groups[k])
        for k2 in group_keys:
            if k2 != k and k2 not in merged and abs(k2 - k) / max(k, 1) < 0.15:
                current.extend(exact_groups[k2])
                merged.add(k2)
        merged.add(k)
        groups.append(current)

    if len(groups) == 2:
        # Two protein types — pair them 1:1 by index order (A-D, B-E, C-F)
        g1, g2 = groups[0], groups[1]
        pairs = list(zip(g1, g2))
        return pairs
    else:
        # Can't cleanly group — return all cross-entity pairs
        return [(i, j) for i in range(len(entities)) for j in range(i + 1, len(entities))]


def plot_aa_residue_interaction_analysis(traj_states, entities: list[EntityInfo],
                                          plots_dir: Path,
                                          aa_atoms: list[dict],
                                          cg_initial_positions,
                                          atom_weights: list,
                                          mapped_atoms: set[int],
                                          backmap_fn,
                                          pairs: list[tuple[int, int]] | None = None):
    """AA-level residue-residue interaction analysis from back-mapped structures.

    The back-mapped AA coordinates carry the quantum-classical-CG cooperative
    information: CG dynamics guide overall motion, QCloud corrections refine
    the forces, and back-mapping reconstructs the AA detail. This function
    analyzes interactions at the true residue level (actual amino acids),
    not CG beads.

    Produces per binding pair:
    - Residue-residue contact frequency heatmap (AA level)
    - Residue-residue closest-approach distance heatmap
    - Top interacting residue pairs (by contact frequency + H-bond presence)
    - Per-residue binding contribution profile
    - Inter-chain H-bond network at the interface
    """
    if len(entities) < 2:
        return

    if pairs is None:
        pairs = _identify_biological_pairs(entities)

    # Build residue index from AA atoms: (chain, resseq) → full metadata
    from collections import OrderedDict

    # H-bond donor/acceptor chemistry:
    # Donors: backbone N (except PRO), side-chain NH/OH groups
    # Acceptors: backbone O, side-chain C=O and O/N groups
    _SIDECHAIN_DONORS = {
        # resname → list of atom names that can donate H-bonds
        "ARG": ["NE", "NH1", "NH2"],
        "ASN": ["ND2"],
        "GLN": ["NE2"],
        "HIS": ["ND1", "NE2"],
        "LYS": ["NZ"],
        "SER": ["OG"],
        "THR": ["OG1"],
        "TRP": ["NE1"],
        "TYR": ["OH"],
        "CYS": ["SG"],  # weak donor
    }
    _SIDECHAIN_ACCEPTORS = {
        # resname → list of atom names that can accept H-bonds
        "ASN": ["OD1"],
        "ASP": ["OD1", "OD2"],
        "GLN": ["OE1"],
        "GLU": ["OE1", "OE2"],
        "HIS": ["ND1", "NE2"],
        "MET": ["SD"],
        "SER": ["OG"],
        "THR": ["OG1"],
        "TYR": ["OH"],
    }

    residue_info: dict[tuple[str, int], dict] = OrderedDict()
    for ai, atom in enumerate(aa_atoms):
        key = (atom["chain"], atom["resseq"])
        name = atom.get("name") or atom["line"][12:16].strip()
        resname = atom.get("resname") or atom["line"][17:20].strip()
        if key not in residue_info:
            residue_info[key] = {
                "resname": resname,
                "chain": atom["chain"],
                "resseq": atom["resseq"],
                "atom_indices": [],
                "heavy_atom_indices": [],  # non-H atoms for contact calculation
                "ca_index": None,          # CA atom for fast pre-filtering
                "backbone_N": None,
                "backbone_O": None,
                "backbone_C": None,        # for angle geometry
                "backbone_CA": None,
                "donor_atoms": [],         # (atom_idx, name) — backbone + sidechain
                "acceptor_atoms": [],      # (atom_idx, name) — backbone + sidechain
            }
        info = residue_info[key]
        info["atom_indices"].append(ai)

        # Track heavy atoms (exclude H)
        element = atom.get("element", name[0])
        if element != "H":
            info["heavy_atom_indices"].append(ai)

        # Backbone atoms
        if name == "CA":
            info["ca_index"] = ai
            info["backbone_CA"] = ai
        elif name == "N":
            info["backbone_N"] = ai
            # Backbone N is a donor (except PRO)
            if resname != "PRO":
                info["donor_atoms"].append((ai, "N"))
        elif name == "O":
            info["backbone_O"] = ai
            info["acceptor_atoms"].append((ai, "O"))
        elif name == "C":
            info["backbone_C"] = ai

        # Side-chain donors
        if resname in _SIDECHAIN_DONORS and name in _SIDECHAIN_DONORS[resname]:
            info["donor_atoms"].append((ai, name))
        # Side-chain acceptors
        if resname in _SIDECHAIN_ACCEPTORS and name in _SIDECHAIN_ACCEPTORS[resname]:
            info["acceptor_atoms"].append((ai, name))

    # Group residues by chain
    chain_residues: dict[str, list[tuple[str, int]]] = {}
    for key in residue_info:
        ch = key[0]
        if ch not in chain_residues:
            chain_residues[ch] = []
        chain_residues[ch].append(key)

    # Map entity chain_ids to their residue keys
    entity_residue_keys: dict[int, list[tuple[str, int]]] = {}
    for ei, ent in enumerate(entities):
        keys = []
        for ch in ent.chain_ids:
            keys.extend(chain_residues.get(ch, []))
        entity_residue_keys[ei] = keys

    # Cutoffs (Angstroms)
    CONTACT_CUTOFF = 4.5      # heavy-atom closest approach for "contact"
    CONTACT_CUTOFF_SQ = CONTACT_CUTOFF ** 2
    CA_PREFILTER = 12.0       # CA-CA prefilter (skip if CAs too far apart)
    CA_PREFILTER_SQ = CA_PREFILTER ** 2
    HBOND_DIST_CUTOFF = 3.5   # donor-acceptor distance cutoff
    HBOND_DIST_CUTOFF_SQ = HBOND_DIST_CUTOFF ** 2
    HBOND_ANGLE_MIN = 120.0   # minimum D-H…A angle (degrees) — 120° is permissive

    import math
    cos_angle_min = math.cos(math.radians(HBOND_ANGLE_MIN))

    def _check_hbond_geometry(coords, donor_idx, acceptor_idx, ca_donor_idx):
        """Check H-bond geometry: distance < 3.5A and angle > 120 degrees.

        Since we don't have explicit H positions in the back-mapped structure,
        we approximate the D-H direction using the CA→donor vector (the H atom
        is roughly along the donor's bonding direction away from the backbone).

        Returns True if the donor-acceptor pair forms a valid H-bond.
        """
        dx = coords[donor_idx][0] - coords[acceptor_idx][0]
        dy = coords[donor_idx][1] - coords[acceptor_idx][1]
        dz = coords[donor_idx][2] - coords[acceptor_idx][2]
        dist_sq = dx * dx + dy * dy + dz * dz
        if dist_sq > HBOND_DIST_CUTOFF_SQ or dist_sq < 0.01:
            return False

        # Angle check: approximate H position direction from CA→Donor vector
        if ca_donor_idx is None:
            # No CA — accept on distance alone (rare edge case)
            return True

        # Vector CA → Donor (approximates direction of the N-H bond)
        ca_d_x = coords[donor_idx][0] - coords[ca_donor_idx][0]
        ca_d_y = coords[donor_idx][1] - coords[ca_donor_idx][1]
        ca_d_z = coords[donor_idx][2] - coords[ca_donor_idx][2]

        # Vector Donor → Acceptor
        d_a_x = coords[acceptor_idx][0] - coords[donor_idx][0]
        d_a_y = coords[acceptor_idx][1] - coords[donor_idx][1]
        d_a_z = coords[acceptor_idx][2] - coords[donor_idx][2]

        # cos(angle) = dot(CA→D, D→A) / (|CA→D| * |D→A|)
        dot = ca_d_x * d_a_x + ca_d_y * d_a_y + ca_d_z * d_a_z
        mag1_sq = ca_d_x**2 + ca_d_y**2 + ca_d_z**2
        mag2_sq = d_a_x**2 + d_a_y**2 + d_a_z**2
        if mag1_sq < 0.01 or mag2_sq < 0.01:
            return dist_sq < HBOND_DIST_CUTOFF_SQ  # fallback

        cos_angle = dot / math.sqrt(mag1_sq * mag2_sq)
        # We want the angle to be roughly linear (>120°)
        # cos(120°) = -0.5, so cos_angle > -0.5 means angle < 120° (bad)
        # We want angle > 120° → cos_angle < cos(120°) = -0.5
        # Actually: angle = acos(cos_angle). For >120°, cos < -0.5.
        # But the CA→D direction is opposite to the H position...
        # The H is on the OPPOSITE side of the donor from CA.
        # So the H-bond direction D→A should be roughly OPPOSITE to CA→D.
        # That means D→A and CA→D should be anti-parallel → cos ≈ -1 → angle ≈ 180°
        # Accept if cos < cos_angle_min (i.e., angle > HBOND_ANGLE_MIN)
        return cos_angle < cos_angle_min

    stride = max(1, len(traj_states) // 50)
    n_sampled = len(range(0, len(traj_states), stride))

    for pair_idx, (ei, ej) in enumerate(pairs):
        ent_a, ent_b = entities[ei], entities[ej]
        label_a = "/".join(ent_a.chain_ids)
        label_b = "/".join(ent_b.chain_ids)
        res_keys_a = entity_residue_keys[ei]
        res_keys_b = entity_residue_keys[ej]
        na, nb = len(res_keys_a), len(res_keys_b)

        if na == 0 or nb == 0:
            continue

        # Accumulate: contact frequency, H-bond count (backbone + sidechain)
        contact_freq = [[0] * nb for _ in range(na)]
        hbond_freq = [[0] * nb for _ in range(na)]

        for fi in range(0, len(traj_states), stride):
            coords = backmap_fn(traj_states[fi])  # list of (x,y,z) in Angstroms

            for ri, rkey_a in enumerate(res_keys_a):
                info_a = residue_info[rkey_a]
                ca_a = info_a["ca_index"]

                for rj, rkey_b in enumerate(res_keys_b):
                    info_b = residue_info[rkey_b]
                    ca_b = info_b["ca_index"]

                    # Fast CA-CA prefilter: skip if CAs > 12A apart
                    if ca_a is not None and ca_b is not None:
                        dx = coords[ca_a][0] - coords[ca_b][0]
                        dy = coords[ca_a][1] - coords[ca_b][1]
                        dz = coords[ca_a][2] - coords[ca_b][2]
                        if dx*dx + dy*dy + dz*dz > CA_PREFILTER_SQ:
                            continue

                    # Heavy-atom closest approach for contact detection
                    min_d_sq = 1e12
                    for ai_a in info_a["heavy_atom_indices"]:
                        xa, ya, za = coords[ai_a]
                        for ai_b in info_b["heavy_atom_indices"]:
                            xb, yb, zb = coords[ai_b]
                            dsq = (xa - xb)**2 + (ya - yb)**2 + (za - zb)**2
                            if dsq < min_d_sq:
                                min_d_sq = dsq

                    if min_d_sq < CONTACT_CUTOFF_SQ:
                        contact_freq[ri][rj] += 1

                    # H-bond detection: all donor-acceptor pairs with angle check
                    # Forward: donors in A → acceptors in B
                    has_hbond = False
                    for d_idx, d_name in info_a["donor_atoms"]:
                        for a_idx, a_name in info_b["acceptor_atoms"]:
                            if _check_hbond_geometry(coords, d_idx, a_idx,
                                                     info_a["backbone_CA"]):
                                has_hbond = True
                                break
                        if has_hbond:
                            break
                    # Reverse: donors in B → acceptors in A
                    if not has_hbond:
                        for d_idx, d_name in info_b["donor_atoms"]:
                            for a_idx, a_name in info_a["acceptor_atoms"]:
                                if _check_hbond_geometry(coords, d_idx, a_idx,
                                                         info_b["backbone_CA"]):
                                    has_hbond = True
                                    break
                            if has_hbond:
                                break
                    if has_hbond:
                        hbond_freq[ri][rj] += 1

        # Normalize
        for i in range(na):
            for j in range(nb):
                contact_freq[i][j] /= max(n_sampled, 1)
                hbond_freq[i][j] /= max(n_sampled, 1)

        # Build residue name labels
        res_labels_a = [f"{residue_info[k]['resname']}{k[1]}" for k in res_keys_a]
        res_labels_b = [f"{residue_info[k]['resname']}{k[1]}" for k in res_keys_b]

        suffix = f"_{label_a}_{label_b}".replace("/", "")

        # --- 1. Contact frequency heatmap ---
        arr_cf = np.array(contact_freq)
        fig, ax = plt.subplots(figsize=(max(8, nb * 0.12 + 3), max(6, na * 0.12 + 3)))
        im = ax.imshow(arr_cf, aspect="auto", origin="lower",
                       cmap=CONTACT_CMAP, vmin=0, vmax=1, interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Contact Frequency", fontsize=9)
        ax.set_xlabel(f"Residue in Chain {label_b}", fontsize=10)
        ax.set_ylabel(f"Residue in Chain {label_a}", fontsize=10)
        ax.set_title(f"AA Residue-Residue Contacts: {label_a} — {label_b}\n"
                     f"(Back-mapped from CG+QCloud, cutoff {CONTACT_CUTOFF} \u00c5)",
                     fontsize=11, fontweight="bold")
        # Tick labels at intervals
        xt_stride = max(1, nb // 20)
        yt_stride = max(1, na // 20)
        ax.set_xticks(range(0, nb, xt_stride))
        ax.set_xticklabels([res_labels_b[i] for i in range(0, nb, xt_stride)],
                           fontsize=5, rotation=90)
        ax.set_yticks(range(0, na, yt_stride))
        ax.set_yticklabels([res_labels_a[i] for i in range(0, na, yt_stride)], fontsize=5)
        add_watermark(fig)
        fig.tight_layout()
        fig.savefig(plots_dir / f"aa_contact_map{suffix}.png")
        plt.close(fig)

        # --- 2. Top interacting residue pairs ---
        scored_pairs = []
        for i in range(na):
            for j in range(nb):
                score = contact_freq[i][j]
                if score > 0.05:  # at least 5% contact occupancy
                    scored_pairs.append((i, j, score, hbond_freq[i][j]))

        scored_pairs.sort(key=lambda x: -(x[2] + x[3]))  # rank by contact + H-bond
        top_n = min(30, len(scored_pairs))
        top_sp = scored_pairs[:top_n]

        if top_sp:
            pair_names = []
            contact_vals = []
            hbond_vals = []
            for i, j, cf, hf in top_sp:
                pair_names.append(f"{label_a}:{res_labels_a[i]} — {label_b}:{res_labels_b[j]}")
                contact_vals.append(cf)
                hbond_vals.append(hf)

            fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
            y_pos = np.arange(len(pair_names))
            # Stacked: contact in blue, H-bond overlay in red
            ax.barh(y_pos, contact_vals, color="#457B9D", edgecolor="#333",
                    linewidth=0.3, label="Contact frequency", alpha=0.85)
            ax.barh(y_pos, hbond_vals, color="#E63946", edgecolor="#333",
                    linewidth=0.3, label="H-bond frequency", alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(pair_names, fontsize=7)
            ax.set_xlabel("Frequency (fraction of frames)")
            ax.set_title(f"Top AA Interface Pairs: {label_a} — {label_b}\n"
                         f"(Quantum-Classical-CG Cooperative Back-mapping)",
                         fontsize=11, fontweight="bold")
            ax.legend(loc="lower right", fontsize=8)
            ax.invert_yaxis()
            add_watermark(fig)
            fig.tight_layout()
            fig.savefig(plots_dir / f"aa_top_residue_pairs{suffix}.png")
            plt.close(fig)

        # --- 3. Per-residue binding contribution (contact frequency sum) ---
        contrib_a = [sum(contact_freq[i][j] for j in range(nb)) for i in range(na)]
        contrib_b = [sum(contact_freq[i][j] for i in range(na)) for j in range(nb)]
        # H-bond contribution
        hb_contrib_a = [sum(hbond_freq[i][j] for j in range(nb)) for i in range(na)]
        hb_contrib_b = [sum(hbond_freq[i][j] for i in range(na)) for j in range(nb)]

        fig, axes = plt.subplots(2, 1, figsize=(max(12, na * 0.1), 8), sharex=False)

        x_a = np.arange(na)
        axes[0].bar(x_a, contrib_a, color="#457B9D", edgecolor="none", alpha=0.7,
                    label="Contact score")
        axes[0].bar(x_a, hb_contrib_a, color="#E63946", edgecolor="none", alpha=0.85,
                    label="H-bond score")
        axes[0].set_ylabel("Interface Score")
        axes[0].set_title(f"Chain {label_a} — Per-Residue Interface Contribution (AA)",
                          fontweight="bold")
        axes[0].legend(fontsize=8, loc="upper right")
        # Mark hotspot residues
        hotspot_thresh_a = sorted(contrib_a, reverse=True)[min(9, na - 1)] if na > 0 else 0
        for i in range(na):
            if contrib_a[i] >= hotspot_thresh_a and contrib_a[i] > 0:
                axes[0].text(i, contrib_a[i], res_labels_a[i], fontsize=5,
                             ha="center", va="bottom", rotation=90, color="#1D3557")
        xt_a_stride = max(1, na // 30)
        axes[0].set_xticks(range(0, na, xt_a_stride))
        axes[0].set_xticklabels([res_labels_a[i] for i in range(0, na, xt_a_stride)],
                                fontsize=5, rotation=90)

        x_b = np.arange(nb)
        axes[1].bar(x_b, contrib_b, color="#2A9D8F", edgecolor="none", alpha=0.7,
                    label="Contact score")
        axes[1].bar(x_b, hb_contrib_b, color="#E63946", edgecolor="none", alpha=0.85,
                    label="H-bond score")
        axes[1].set_ylabel("Interface Score")
        axes[1].set_title(f"Chain {label_b} — Per-Residue Interface Contribution (AA)",
                          fontweight="bold")
        axes[1].legend(fontsize=8, loc="upper right")
        hotspot_thresh_b = sorted(contrib_b, reverse=True)[min(9, nb - 1)] if nb > 0 else 0
        for j in range(nb):
            if contrib_b[j] >= hotspot_thresh_b and contrib_b[j] > 0:
                axes[1].text(j, contrib_b[j], res_labels_b[j], fontsize=5,
                             ha="center", va="bottom", rotation=90, color="#1D3557")
        xt_b_stride = max(1, nb // 30)
        axes[1].set_xticks(range(0, nb, xt_b_stride))
        axes[1].set_xticklabels([res_labels_b[j] for j in range(0, nb, xt_b_stride)],
                                fontsize=5, rotation=90)

        fig.suptitle("Residue-Level Binding Contribution (AA from CG+QCloud Back-mapping)",
                     fontsize=12, fontweight="bold", y=1.02)
        add_watermark(fig)
        fig.tight_layout()
        fig.savefig(plots_dir / f"aa_residue_binding_contribution{suffix}.png")
        plt.close(fig)

        # --- 4. Interface H-bond network ---
        hb_pairs_significant = [(i, j, hbond_freq[i][j])
                                for i in range(na) for j in range(nb)
                                if hbond_freq[i][j] > 0.1]  # >10% occupancy
        if hb_pairs_significant:
            hb_pairs_significant.sort(key=lambda x: -x[2])
            top_hb = hb_pairs_significant[:20]

            fig, ax = plt.subplots(figsize=(10, max(4, len(top_hb) * 0.35)))
            hb_names = [f"{label_a}:{res_labels_a[i]}(N) — {label_b}:{res_labels_b[j]}(O)"
                        for i, j, _ in top_hb]
            hb_occ = [occ for _, _, occ in top_hb]
            colors_hb = plt.cm.Reds(np.linspace(0.4, 0.9, len(hb_occ)))
            ax.barh(range(len(hb_names)), hb_occ, color=colors_hb,
                    edgecolor="#333", linewidth=0.3)
            ax.set_yticks(range(len(hb_names)))
            ax.set_yticklabels(hb_names, fontsize=7)
            ax.set_xlabel("H-bond Occupancy (fraction of frames)")
            ax.set_title(f"Inter-Chain Backbone H-bonds: {label_a} — {label_b}\n"
                         f"(AA-level from CG+QCloud back-mapping)",
                         fontsize=11, fontweight="bold")
            ax.invert_yaxis()
            add_watermark(fig)
            fig.tight_layout()
            fig.savefig(plots_dir / f"aa_interface_hbonds{suffix}.png")
            plt.close(fig)

        n_contacts_total = sum(1 for i in range(na) for j in range(nb) if contact_freq[i][j] > 0.5)
        n_hbonds_total = len(hb_pairs_significant)
        print(f"    {label_a}-{label_b}: {na}x{nb} residues, "
              f"{n_contacts_total} persistent contacts (>50%), "
              f"{n_hbonds_total} inter-chain H-bonds (>10%)")


def run_binding_analysis(traj_states, bundle_path: Path, plots_dir: Path,
                         aa_atoms=None, cg_initial_positions=None,
                         atom_weights=None, mapped_atoms=None, backmap_fn=None):
    """Auto-detect system type and run appropriate binding analysis.

    When aa_atoms and backmap_fn are provided, performs AA-level residue-residue
    interaction analysis using back-mapped coordinates that carry the full
    quantum-classical-CG cooperative information.
    """
    entities = _load_entities(bundle_path)

    if len(entities) < 2:
        print("  Single entity — skipping binding analysis.")
        return

    n_entities = len(entities)
    chain_desc = ", ".join(f"{'/'.join(e.chain_ids)}({e.bead_count})" for e in entities)
    print(f"  Detected {n_entities} entities: {chain_desc}")

    if n_entities == 2:
        system_type = "protein-protein" if all(e.bead_count > 5 for e in entities) else "protein-ligand"
    else:
        system_type = "complex assembly"
    print(f"  System type: {system_type}")

    # Identify biologically relevant pairs for complex assemblies
    bio_pairs = _identify_biological_pairs(entities)
    pair_desc = ", ".join(
        f"{'/'.join(entities[i].chain_ids)}-{'/'.join(entities[j].chain_ids)}"
        for i, j in bio_pairs
    )
    print(f"  Biological binding pairs: {pair_desc}")

    print("  Plotting CG contact map...")
    plot_inter_chain_contact_map(traj_states, entities, plots_dir)

    print("  Plotting pairwise binding energies (biological pairs only)...")
    plot_pairwise_binding_energy(traj_states, entities, plots_dir)

    print("  Plotting CG interface residues...")
    plot_interface_residues(traj_states, entities, plots_dir)

    # AA-level residue-residue analysis (the core cooperative analysis)
    if aa_atoms is not None and backmap_fn is not None:
        print("  Running AA-level residue-residue interaction analysis "
              "(CG+QCloud back-mapped)...")
        plot_aa_residue_interaction_analysis(
            traj_states, entities, plots_dir,
            aa_atoms=aa_atoms,
            cg_initial_positions=cg_initial_positions,
            atom_weights=atom_weights,
            mapped_atoms=mapped_atoms,
            backmap_fn=backmap_fn,
            pairs=bio_pairs,
        )
    else:
        print("  (No AA reference — skipping AA-level residue analysis)")

    print("  Plotting 2D free energy landscape...")
    plot_2d_free_energy_landscape(traj_states, entities, plots_dir)

    print("  Plotting binding dashboard...")
    plot_binding_interaction_dashboard(traj_states, entities, plots_dir)

    print(f"  Binding analysis complete ({system_type}).")
