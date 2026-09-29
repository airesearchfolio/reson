#!/usr/bin/env python3
"""
Publication-ready alpha/beta sweep plot.
All values are embedded: no external CSV required.

Outputs:
  fig_alpha_beta_tradeoff.pdf
  fig_alpha_beta_tradeoff.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------
# DATA -- preserved from the existing sweep figure
# ---------------------------------------------------------------------

# (a) alpha sweep at fixed beta=0.34
alpha = np.array([0.135, 0.145, 0.155])
alpha_tpr = np.array([85.0, 89.5, 94.0])
alpha_fidelity_loss = np.array([4.8, 5.0, 5.9])

# (b) beta sweep at fixed alpha=0.155
beta = np.array([0.30, 0.32, 0.34, 0.36, 0.38, 0.40])
beta_tpr = np.array([83.5, 89.5, 94.0, 96.5, 97.5, 100.0])
beta_fidelity_loss = np.array([4.0, 4.8, 5.9, 7.3, 9.0, 11.5])

SELECTED_ALPHA = 0.155
SELECTED_BETA = 0.40

# ---------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------

BLUE = "#1f77b4"
ORANGE = "#d95f02"
GRAY = "#777777"

# Curve colors — KEEP UNCHANGED
BLUE = "#1f77b4"
ORANGE = "#d95f02"

# Axis typography
BLUE_AXIS = "#155A8A"
ORANGE_AXIS = "#8B2F00"   # much darker burnt orange

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "axes.linewidth": 1.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1))
# Deliberately reserve a clean band at the top for one shared legend.
fig.subplots_adjust(
    left=0.085,
    right=0.925,
    bottom=0.14,
    top=0.82,       # plots move upward toward legend
    wspace=0.43
)

def make_panel(ax, x, tpr, cost, xlabel, title, selected_x,
               xlim, xticks):
    ax2 = ax.twinx()

    # Detection
    l1, = ax.plot(
        x, tpr,
        color=BLUE, marker="o", markersize=7,
        linewidth=2.0, zorder=4,
        label="Detection (TPR@1% FPR)"
    )

    # Fidelity loss
    l2, = ax2.plot(
        x, cost,
        color=ORANGE, marker="s", markersize=6.5,
        linewidth=2.0, linestyle="--", zorder=3,
        label=r"Fidelity loss ($-\Delta$CLIP)"
    )

    # Selected parameter: neutral guide rather than a third data color
    ax.axvline(
        selected_x, color=GRAY, linestyle=":", linewidth=1.8,
        alpha=0.85, zorder=1
    )

    # Open black rings emphasize the selected values on BOTH objectives.
    idx = int(np.argmin(np.abs(x - selected_x)))
    ax.scatter(
        [x[idx]], [tpr[idx]], s=210,
        facecolors="none", edgecolors="black",
        linewidths=1.5, zorder=6
    )
    ax2.scatter(
        [x[idx]], [cost[idx]], s=180,
        facecolors="none", edgecolors="black",
        linewidths=1.4, zorder=6
    )

    ax.set_title(title, pad=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Detection TPR (%)", color=BLUE, labelpad=7)
    ax.tick_params(axis="y", colors=BLUE)

    ax2.set_ylabel(
    r"Fidelity loss ($-\Delta$CLIP, %)",
    color=ORANGE_AXIS,
    labelpad=10,
    fontweight="medium"
    )

    ax2.tick_params(
        axis="y",
        colors=ORANGE_AXIS,
        width=1.2,
        length=4
    )

    ax2.spines["right"].set_color(ORANGE_AXIS)
    
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_ylim(78, 102)
    ax.set_yticks([80, 85, 90, 95, 100])

    ax2.set_ylim(0, 12.5)
    ax2.set_yticks([0, 2.5, 5, 7.5, 10, 12.5])

    # Light horizontal grid only: avoids visual clutter from two axes.
    ax.grid(axis="y", linestyle=":", linewidth=0.9,
            alpha=0.35, color=GRAY)
    ax.grid(axis="x", visible=False)

    return l1, l2

l1, l2 = make_panel(
    axes[0],
    alpha, alpha_tpr, alpha_fidelity_loss,
    r"Injection strength $\alpha$",
    r"(a) Injection-strength trade-off",
    SELECTED_ALPHA,
    (0.134, 0.156),
    [0.135, 0.140, 0.145, 0.150, 0.155]
)

make_panel(
    axes[1],
    beta, beta_tpr, beta_fidelity_loss,
    r"Carrier mixing $\beta$",
    r"(b) Carrier-mixing trade-off",
    SELECTED_BETA,
    (0.295, 0.405),
    [0.30, 0.32, 0.34, 0.36, 0.38, 0.40]
)

# Shared legend: semantic colors correspond to objectives.
selected_handle = Line2D(
    [0], [0], marker="o", linestyle="None",
    markerfacecolor="none", markeredgecolor="black",
    markeredgewidth=1.4, markersize=9,
    label="Selected operating point"
)

fig.legend(
    handles=[l1, l2, selected_handle],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.985),
    ncol=3,
    frameon=False,
    columnspacing=2.0,
    handlelength=2.5
)

# # Sequential-selection note. Kept outside plotting area.
# fig.text(
#     0.5, 0.055,
#     r"Sequential selection: $\alpha=0.155$ at fixed $\beta=0.34$; "
#     r"then $\beta=0.40$ at fixed $\alpha=0.155$.",
#     ha="center", va="center", fontsize=11
# )

fig.savefig("fig_alpha_beta_tradeoff.pdf", bbox_inches="tight")
fig.savefig("fig_alpha_beta_tradeoff.png", dpi=400, bbox_inches="tight")
print("Saved fig_alpha_beta_tradeoff.pdf")
print("Saved fig_alpha_beta_tradeoff.png")

plt.close(fig)
