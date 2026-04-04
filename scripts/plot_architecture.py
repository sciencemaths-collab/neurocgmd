"""Generate the NeuroCGMD architecture diagram.

Publication-quality diagram of the full cooperative pipeline:
  Input → CG Dynamics + QCloud + ML → Back-mapping → AA Analysis
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Palette ──────────────────────────────────────────────────────────────────
BG        = "#0D1117"
CARD      = "#161B22"
BORDER    = "#30363D"
TEXT      = "#E6EDF3"
MUTED     = "#8B949E"
BLUE      = "#58A6FF"
TEAL      = "#3FB9A2"
PURPLE    = "#BC8CFF"
GOLD      = "#F0C75E"
GREEN     = "#56D364"
PINK      = "#F778BA"
ORANGE    = "#F0883E"
RED       = "#F85149"


# ── Drawing primitives ───────────────────────────────────────────────────────

def box(ax, x, y, w, h, color, label, sub=None, fs=11, sfs=8, radius=0.012):
    """Draw a rounded card with label and optional subtitle."""
    r = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius}",
                       facecolor=color + "18", edgecolor=color + "90",
                       linewidth=1.2, transform=ax.transAxes)
    ax.add_patch(r)
    ty = y + h * 0.62 if sub else y + h * 0.5
    ax.text(x + w/2, ty, label, fontsize=fs, color=color, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)
    if sub:
        ax.text(x + w/2, y + h * 0.28, sub, fontsize=sfs, color=MUTED,
                ha="center", va="center", transform=ax.transAxes, style="italic")


def arrow(ax, x1, y1, x2, y2, color, lw=1.6, style="-|>", ms=14):
    """Draw a straight arrow."""
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                        linewidth=lw, mutation_scale=ms, alpha=0.75,
                        transform=ax.transAxes)
    ax.add_patch(a)


def curved_arrow(ax, x1, y1, x2, y2, color, rad=0.2, lw=1.4):
    """Draw a curved arrow."""
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        connectionstyle=f"arc3,rad={rad}",
                        arrowstyle="-|>", color=color,
                        linewidth=lw, mutation_scale=13, alpha=0.65,
                        transform=ax.transAxes)
    ax.add_patch(a)


def glow(ax, x, y, text, fs=12, color=TEXT, bold=True, **kw):
    """Draw text with subtle glow."""
    w = "bold" if bold else "normal"
    defaults = dict(ha="center", va="center")
    defaults.update(kw)
    t = ax.text(x, y, text, fontsize=fs, color=color, fontweight=w,
                transform=ax.transAxes, **defaults)
    t.set_path_effects([pe.withStroke(linewidth=3, foreground=color + "25"), pe.Normal()])
    return t


def label(ax, x, y, text, fs=8, color=MUTED, **kw):
    """Small annotation label with background."""
    ax.text(x, y, text, fontsize=fs, color=color, ha="center", va="center",
            transform=ax.transAxes, style="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=BG + "DD",
                      edgecolor=BORDER, linewidth=0.5), **kw)


def section_box(ax, x, y, w, h, color, title=None, title_fs=15):
    """Draw a large section outline with optional title."""
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                       facecolor=CARD, edgecolor=color + "50",
                       linewidth=1.8, transform=ax.transAxes)
    ax.add_patch(r)
    if title:
        glow(ax, x + w/2, y + h - 0.018, title, fs=title_fs, color=color)


def hline(ax, x1, x2, y, color=BORDER, lw=0.8, alpha=0.4):
    ax.plot([x1, x2], [y, y], color=color, linewidth=lw, alpha=alpha,
            transform=ax.transAxes)


# ── Main diagram ─────────────────────────────────────────────────────────────

def main(output_dir: str | None = None):
    fig = plt.figure(figsize=(22, 30), facecolor=BG)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor(BG); ax.axis("off")

    # ── TITLE ────────────────────────────────────────────────────────────
    glow(ax, 0.5, 0.976, "NeuroCGMD", fs=34, color=BLUE)
    glow(ax, 0.5, 0.960, "Quantum-Classical-CG-ML Cooperative Molecular Dynamics Engine",
         fs=13, color=MUTED, bold=False)
    hline(ax, 0.1, 0.9, 0.950)
    ax.text(0.5, 0.943, "One simulation  ·  One information stream  ·  All-atom accuracy at CG speed",
            fontsize=11, color=TEAL, ha="center", va="center",
            transform=ax.transAxes, style="italic")

    # ── LAYER 1: INPUT (y = 0.905) ───────────────────────────────────────
    iy = 0.905
    glow(ax, 0.065, iy + 0.005, "INPUT", fs=10, color=MUTED, ha="left")

    bw, bh = 0.18, 0.035
    box(ax, 0.13, iy - bh/2, bw, bh, MUTED, "PDB Structure", fs=10)
    box(ax, 0.34, iy - bh/2, bw, bh, MUTED, "TOML Config", fs=10)
    box(ax, 0.55, iy - bh/2, 0.20, bh, ORANGE, "Preparation", "CG mapping + topology", fs=10, sfs=7)
    box(ax, 0.78, iy - bh/2, bw, bh, BLUE, "CG System", "Beads, bonds, angles", fs=10, sfs=7)

    arrow(ax, 0.31, iy, 0.34, iy, MUTED)
    arrow(ax, 0.52, iy, 0.55, iy, ORANGE)
    arrow(ax, 0.75, iy, 0.78, iy, BLUE)

    # ── LAYER 2: SIMULATION ENGINE (y = 0.56 to 0.86) ───────────────────
    ey_top, ey_bot = 0.865, 0.545
    section_box(ax, 0.04, ey_bot, 0.92, ey_top - ey_bot, ORANGE, "SIMULATION ENGINE", 16)

    ax.text(0.5, ey_top - 0.040,
            "Hybrid production loop  ·  eval_stride controls full vs lightweight evaluation",
            fontsize=9, color=MUTED, ha="center", va="center", transform=ax.transAxes)

    # Column layout: 3 equal columns with gaps
    col_w = 0.26
    col_gap = 0.03
    col1_x = 0.07   # CG
    col2_x = col1_x + col_w + col_gap   # QCloud
    col3_x = col2_x + col_w + col_gap   # ML

    row_top = ey_top - 0.065
    bx_h = 0.042
    bx_gap = 0.008

    def col_box(col_x, row, color, title, sub=None):
        y = row_top - row * (bx_h + bx_gap)
        box(ax, col_x, y, col_w, bx_h, color, title, sub, fs=10, sfs=7)
        return y

    # Column headers
    glow(ax, col1_x + col_w/2, row_top + 0.020, "CG DYNAMICS", fs=12, color=BLUE)
    glow(ax, col2_x + col_w/2, row_top + 0.020, "QCLOUD QUANTUM", fs=12, color=PURPLE)
    glow(ax, col3_x + col_w/2, row_top + 0.020, "ML RESIDUAL", fs=12, color=GOLD)

    # CG column
    cy0 = col_box(col1_x, 0, BLUE, "Langevin Integrator", "Velocity-Verlet + BAOAB thermostat")
    cy1 = col_box(col1_x, 1, BLUE, "Classical Force Field", "Harmonic bonds + LJ (shifted at cutoff)")
    cy2 = col_box(col1_x, 2, BLUE, "Neighbor List", "O(N) cell-list with cutoff")
    cy3 = col_box(col1_x, 3, GREEN, "Lightweight Path", "Classical-only between eval steps")

    for r in range(3):
        y1 = row_top - r * (bx_h + bx_gap)
        y2 = row_top - (r+1) * (bx_h + bx_gap) + bx_h
        arrow(ax, col1_x + col_w/2, y1, col1_x + col_w/2, y2, BLUE, lw=1.2)

    # QCloud column
    qy0 = col_box(col2_x, 0, PURPLE, "Region Selector", "Priority-based subgraph selection")
    qy1 = col_box(col2_x, 1, PURPLE, "Quantum Correction", "AA-level force deltas on regions")
    qy2 = col_box(col2_x, 2, PURPLE, "Event Analyzer", "Bond forming/breaking detection")
    qy3 = col_box(col2_x, 3, PURPLE, "Feedback Loop", "Correction magnitudes → priority scores")

    for r in range(3):
        y1 = row_top - r * (bx_h + bx_gap)
        y2 = row_top - (r+1) * (bx_h + bx_gap) + bx_h
        arrow(ax, col2_x + col_w/2, y1, col2_x + col_w/2, y2, PURPLE, lw=1.2)

    # Feedback curved arrow (bottom → top right)
    fb_bot = row_top - 3*(bx_h + bx_gap) + bx_h/2
    fb_top = row_top + bx_h/2
    curved_arrow(ax, col2_x + col_w + 0.005, fb_bot,
                 col2_x + col_w + 0.005, fb_top, PURPLE, rad=0.25, lw=1.3)
    label(ax, col2_x + col_w + 0.040, (fb_bot + fb_top)/2, "adaptive\nfocus", fs=7, color=PURPLE)

    # ML column
    my0 = col_box(col3_x, 0, GOLD, "Residual Predictor", "Learns CG → QCloud correction patterns")
    my1 = col_box(col3_x, 1, GOLD, "On-the-Fly Training", "Updates every eval step from QCloud")
    my2 = col_box(col3_x, 2, GOLD, "Force Composition", "F = F_CG + F_QCloud + F_ML")
    my3 = col_box(col3_x, 3, GOLD, "Drift Control", "Energy conservation monitoring")

    for r in range(3):
        y1 = row_top - r * (bx_h + bx_gap)
        y2 = row_top - (r+1) * (bx_h + bx_gap) + bx_h
        arrow(ax, col3_x + col_w/2, y1, col3_x + col_w/2, y2, GOLD, lw=1.2)

    # ── Cross-layer arrows ──
    mid_y1 = row_top - 0.5*(bx_h + bx_gap) + bx_h/2   # between row 0 and 1
    mid_y2 = row_top - 1.5*(bx_h + bx_gap) + bx_h/2   # between row 1 and 2

    # CG → QCloud
    arrow(ax, col1_x + col_w, mid_y1, col2_x, mid_y1, MUTED, lw=2.0)
    label(ax, (col1_x + col_w + col2_x)/2, mid_y1 + 0.016, "CG forces +\npositions", fs=7)

    # QCloud → ML
    arrow(ax, col2_x + col_w, mid_y1, col3_x, mid_y1, MUTED, lw=2.0)
    label(ax, (col2_x + col_w + col3_x)/2, mid_y1 + 0.016, "corrections as\ntraining data", fs=7)

    # ── Force equation bar ──
    eq_y = ey_bot + 0.030
    eq_rect = FancyBboxPatch((0.10, eq_y - 0.013), 0.80, 0.026,
                             boxstyle="round,pad=0.005",
                             facecolor=GREEN + "10", edgecolor=GREEN + "40",
                             linewidth=1.0, transform=ax.transAxes)
    ax.add_patch(eq_rect)

    # Mathematical formula with proper notation
    ax.text(0.50, eq_y, r"$\mathbf{F}_{\mathrm{total}} = \mathbf{F}_{\mathrm{CG}} + \Delta\mathbf{F}_{\mathrm{QCloud}} + \alpha\,\Delta\mathbf{F}_{\mathrm{ML}}$          $\alpha = 0.35$ when QCloud active",
            fontsize=12, color=GREEN, ha="center", va="center",
            transform=ax.transAxes, fontfamily="serif")

    # Arrows down from each column to the equation bar
    for cx in [col1_x, col2_x, col3_x]:
        bot_of_col = row_top - 3*(bx_h + bx_gap)
        arrow(ax, cx + col_w/2, bot_of_col, cx + col_w/2, eq_y + 0.015, GREEN, lw=1.3)

    # ── eval_stride note ──
    ax.text(0.50, ey_bot + 0.008,
            "eval_stride: full QCloud+ML pipeline every N steps  |  lightweight classical between  |  ~1000 steps/min",
            fontsize=8, color=GREEN, ha="center", va="center",
            transform=ax.transAxes, style="italic")

    # ── LAYER 3: INTEGRATOR (y = 0.49 to 0.53) ──────────────────────────
    int_y = 0.505
    int_h = 0.032
    section_box(ax, 0.04, int_y - int_h/2, 0.92, int_h, BLUE)
    ax.text(0.12, int_y, "INTEGRATOR", fontsize=11, color=BLUE, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)
    ax.text(0.55, int_y,
            r"$\mathbf{v}_{n+\frac{1}{2}} = \mathbf{v}_n + \frac{\Delta t}{2m}\,\mathbf{F}_{\mathrm{total}}$"
            r"          "
            r"$\mathbf{x}_{n+1} = \mathbf{x}_n + \Delta t\,\mathbf{v}_{n+\frac{1}{2}}$"
            r"          "
            r"$\mathbf{v}_{n+1} = \mathbf{v}_{n+\frac{1}{2}} + \frac{\Delta t}{2m}\,\mathbf{F}_{n+1}$",
            fontsize=10, color=TEXT, ha="center", va="center",
            transform=ax.transAxes, fontfamily="serif")

    arrow(ax, 0.50, ey_bot, 0.50, int_y + int_h/2, BLUE, lw=2.5)

    # ── LAYER 4: BACK-MAPPING (y = 0.43 to 0.47) ────────────────────────
    bm_y = 0.445
    bm_h = 0.040
    section_box(ax, 0.04, bm_y - bm_h/2, 0.92, bm_h, TEAL)
    glow(ax, 0.12, bm_y + 0.004, "BACK-MAPPING", fs=12, color=TEAL, ha="center")
    ax.text(0.55, bm_y + 0.005,
            r"$\mathbf{r}_{\mathrm{AA}}^{(a)}(t) = \mathbf{r}_{\mathrm{AA}}^{(a)}(0) + \sum_i w_i \left[\mathbf{r}_{\mathrm{CG}}^{(i)}(t) - \mathbf{r}_{\mathrm{CG}}^{(i)}(0)\right]$",
            fontsize=11, color=TEAL, ha="center", va="center",
            transform=ax.transAxes, fontfamily="serif")
    ax.text(0.50, bm_y - 0.013,
            "AA coordinates carry CG dynamics + QCloud corrections + ML residual — one coherent stream",
            fontsize=8, color=MUTED, ha="center", va="center",
            transform=ax.transAxes, style="italic")

    arrow(ax, 0.50, int_y - int_h/2, 0.50, bm_y + bm_h/2, TEAL, lw=2.5)

    # ── LAYER 5: ANALYSIS ROUTING (y = 0.10 to 0.39) ────────────────────
    an_top = 0.400
    an_bot = 0.095
    section_box(ax, 0.04, an_bot, 0.92, an_top - an_bot, PINK,
                "INTELLIGENT ANALYSIS ROUTING", 14)

    ax.text(0.50, an_top - 0.035,
            "Each analysis routes to the level where it is most meaningful",
            fontsize=9, color=MUTED, ha="center", va="center",
            transform=ax.transAxes, style="italic")

    arrow(ax, 0.50, bm_y - bm_h/2, 0.50, an_top, PINK, lw=2.5)

    # Three analysis columns
    acol_w = 0.27
    acol_gap = 0.02
    a1_x = 0.065
    a2_x = a1_x + acol_w + acol_gap
    a3_x = a2_x + acol_w + acol_gap
    ahead_y = an_top - 0.055

    box(ax, a1_x, ahead_y, acol_w, 0.030, BLUE, "CG-Level Analysis",
        "Collective dynamics", fs=11, sfs=8)
    box(ax, a2_x, ahead_y, acol_w, 0.030, TEAL, "AA-Level Analysis",
        "Back-mapped atomic detail", fs=11, sfs=8)
    box(ax, a3_x, ahead_y, acol_w, 0.030, PURPLE, "QCloud Analysis",
        "Quantum correction feedback", fs=11, sfs=8)

    # Item lists
    def items(ax, x, start_y, color, entries):
        for i, (name, desc) in enumerate(entries):
            iy = start_y - i * 0.020
            ax.text(x + 0.008, iy, "›", fontsize=10, color=color,
                    transform=ax.transAxes, va="center")
            ax.text(x + 0.020, iy, name, fontsize=9, color=TEXT,
                    transform=ax.transAxes, va="center", fontweight="bold")
            ax.text(x + acol_w - 0.008, iy, desc, fontsize=7.5, color=MUTED,
                    transform=ax.transAxes, va="center", ha="right", style="italic")

    list_y = ahead_y - 0.015

    items(ax, a1_x, list_y, BLUE, [
        ("Energy", "PE / KE / Total"),
        ("RMSD / RMSF", "Drift & fluctuation"),
        ("RDF g(r)", "Pair correlation"),
        ("SASA + Rg", "Surface & compactness"),
        ("PMF", "Boltzmann inversion"),
        ("Reaction Coord.", "COM distance"),
        ("Free Energy", "2D landscape"),
        ("Contact Map", "Bead-bead frequency"),
        ("Binding Dashboard", "4-panel overview"),
    ])

    items(ax, a2_x, list_y, TEAL, [
        ("Residue Contacts", "Per-amino-acid map"),
        ("Top Interface Pairs", "Contact + H-bond ranked"),
        ("Per-Residue Binding", "Hotspot profiles"),
        ("Inter-Chain H-bonds", "N-O network (angle)"),
        ("H-bond Time Series", "Count evolution"),
        ("H-bond Occupancy", "Pair persistence"),
        ("AA Trajectory", "PDB (CG→AA)"),
    ])

    items(ax, a3_x, list_y, PURPLE, [
        ("Structural Events", "Bond form/break"),
        ("Event Timeline", "Magnitudes over time"),
        ("Adaptive Focus", "Where QCloud spent"),
        ("Energy Decomp.", "Per-bead breakdown"),
        ("Interaction Energy", "Inter-entity LJ"),
    ])

    # ── LAYER 6: KEY PRINCIPLES (bottom) ─────────────────────────────────
    py = 0.045
    pw = 0.155
    pstart = 0.065
    principles = [
        ("Speed", "CG dynamics\n~1000 steps/min", BLUE),
        ("Accuracy", "QCloud quantum\ncorrections", PURPLE),
        ("Learning", "ML residual\non-the-fly", GOLD),
        ("Detail", "Full AA from\nCG+QCloud", TEAL),
        ("Smart", "Analysis routed\nto right level", PINK),
    ]
    for i, (title, desc, color) in enumerate(principles):
        px = pstart + i * (pw + 0.012)
        box(ax, px, py - 0.018, pw, 0.045, color, title, desc, fs=10, sfs=7.5)

    # ── Version ──────────────────────────────────────────────────────────
    ax.text(0.50, 0.010,
            "NeuroCGMD v1.0.0  ·  Quantum-Classical-CG-ML Cooperative MD Engine",
            fontsize=8, color=MUTED + "60", ha="center", va="center",
            transform=ax.transAxes)

    # ── Save ─────────────────────────────────────────────────────────────
    if output_dir:
        out_path = Path(output_dir) / "plots" / "architecture.png"
    else:
        out_path = Path("outputs/barnase_barstar_20ns/plots/architecture.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Architecture diagram saved to {out_path}")

    # Also copy to website assets
    web_path = Path(__file__).resolve().parents[1] / "docs" / "website" / "assets" / "plots" / "architecture.png"
    if web_path.parent.exists():
        import shutil
        shutil.copy2(out_path, web_path)
        print(f"  Copied to {web_path}")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else None
    main(output_dir)
