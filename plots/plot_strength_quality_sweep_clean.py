#!/usr/bin/env python3
"""
Publication-ready alpha/beta detection--fidelity trade-off figure.

All values are embedded; no external CSV required.

Outputs:
    fig_alpha_beta_tradeoff.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# DATA
# ============================================================

# Alpha sweep:
# beta fixed at 0.34
# N = 200 matched pairs per configuration
alpha = np.array([0.135, 0.145, 0.155])

alpha_tpr = np.array([
    85.0,
    89.5,
    94.0
])

# Relative CLIP change (%)
alpha_dclip = np.array([
    -4.75,
    -5.01,
    -5.83
])


# Beta sweep:
# alpha fixed at 0.155
# N = 200 matched pairs per configuration
beta = np.array([
    0.30,
    0.32,
    0.34,
    0.36,
    0.38,
    0.40
])

beta_tpr = np.array([
    83.5,
    89.5,
    94.0,
    96.5,
    97.5,
    100.0
])

# Relative CLIP change (%)
beta_dclip = np.array([
    -3.90,
    -4.67,
    -5.83,
    -7.31,
    -9.18,
    -11.43
])


SELECTED_ALPHA = 0.155
SELECTED_BETA = 0.40


# ============================================================
# COLORS
# ============================================================

# Curves
BLUE = "#1f77b4"
ORANGE = "#d95f02"

# Darker versions for axis text/ticks
BLUE_AXIS = "#155A8A"
ORANGE_AXIS = "#8F3600"

# Neutral selection marker / guide
GRAY = "#777777"


# ============================================================
# MATPLOTLIB STYLE
# ============================================================

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


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.4, 5.0)
)

# Extra space:
# top -> shared legend
# middle -> dual y-axis labels do not collide
fig.subplots_adjust(
    left=0.085,
    right=0.925,
    bottom=0.16,
    top=0.76,
    wspace=0.43
)


# ============================================================
# PANEL FUNCTION
# ============================================================

def make_panel(
    ax,
    x,
    tpr,
    dclip,
    xlabel,
    title,
    selected_x,
    xlim,
    xticks
):
    """
    Plot detection TPR on left y-axis and fidelity loss (-Delta CLIP)
    on right y-axis.
    """

    ax2 = ax.twinx()

    # --------------------------------------------------------
    # Detection curve
    # --------------------------------------------------------

    detection_line, = ax.plot(
        x,
        tpr,
        color=BLUE,
        marker="o",
        markersize=7,
        markerfacecolor=BLUE,
        markeredgecolor=BLUE,
        linewidth=2.8,
        zorder=4,
        label=r"Detection (TPR@1\% FPR)"
    )

    # --------------------------------------------------------
    # Fidelity-loss curve
    #
    # dCLIP is negative in the experiment.
    # Plot magnitude so larger value = greater fidelity loss.
    # --------------------------------------------------------

    fidelity_loss = -dclip

    fidelity_line, = ax2.plot(
        x,
        fidelity_loss,
        color=ORANGE,
        marker="s",
        markersize=6.5,
        markerfacecolor=ORANGE,
        markeredgecolor=ORANGE,
        linewidth=2.5,
        linestyle="--",
        zorder=3,
        label=r"Fidelity loss ($-\Delta$CLIP)"
    )

    # --------------------------------------------------------
    # Selected parameter guide
    # --------------------------------------------------------

    ax.axvline(
        selected_x,
        color=GRAY,
        linestyle=":",
        linewidth=1.7,
        alpha=0.75,
        zorder=1
    )

    idx = int(np.argmin(np.abs(x - selected_x)))

    # Black open ring around selected TPR
    ax.scatter(
        [x[idx]],
        [tpr[idx]],
        s=210,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        zorder=7
    )

    # Black open ring around selected fidelity point
    ax2.scatter(
        [x[idx]],
        [fidelity_loss[idx]],
        s=180,
        facecolors="none",
        edgecolors="black",
        linewidths=1.4,
        zorder=7
    )

    # --------------------------------------------------------
    # Titles / x-axis
    # --------------------------------------------------------

    ax.set_title(
        title,
        pad=14,
        fontweight="normal"
    )

    ax.set_xlabel(
        xlabel,
        labelpad=7
    )

    # --------------------------------------------------------
    # LEFT Y AXIS
    # --------------------------------------------------------

    ax.set_ylabel(
        "Detection TPR (%)",
        color=BLUE_AXIS,
        labelpad=8,
        fontweight="medium"
    )

    ax.tick_params(
        axis="y",
        colors=BLUE_AXIS,
        width=1.2,
        length=4
    )

    ax.spines["left"].set_color(BLUE_AXIS)
    ax.spines["left"].set_linewidth(1.2)

    # --------------------------------------------------------
    # RIGHT Y AXIS
    # --------------------------------------------------------

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

    # Dark right spine makes association obvious
    ax2.spines["right"].set_color(ORANGE_AXIS)
    ax2.spines["right"].set_linewidth(1.2)

    # --------------------------------------------------------
    # AXIS RANGES
    # --------------------------------------------------------

    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)

    ax.set_ylim(78, 102)
    ax.set_yticks([
        80,
        85,
        90,
        95,
        100
    ])

    ax2.set_ylim(0, 12.5)
    ax2.set_yticks([
        0,
        2.5,
        5.0,
        7.5,
        10.0,
        12.5
    ])

    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------

    # Horizontal grid only.
    # This keeps a dual-axis plot substantially cleaner.
    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.9,
        alpha=0.30,
        color=GRAY
    )

    ax.grid(
        axis="x",
        visible=False
    )

    # Keep top/bottom spines neutral
    ax.spines["top"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")

    # Hide duplicate top spine from twin axis
    ax2.spines["top"].set_visible(False)

    return detection_line, fidelity_line


# ============================================================
# PANEL A -- ALPHA
# ============================================================

detection_line, fidelity_line = make_panel(
    axes[0],
    alpha,
    alpha_tpr,
    alpha_dclip,
    xlabel=r"Injection strength $\alpha$",
    title=r"(a) Injection-strength trade-off",
    selected_x=SELECTED_ALPHA,
    xlim=(0.134, 0.156),
    xticks=[
        0.135,
        0.140,
        0.145,
        0.150,
        0.155
    ]
)


# ============================================================
# PANEL B -- BETA
# ============================================================

make_panel(
    axes[1],
    beta,
    beta_tpr,
    beta_dclip,
    xlabel=r"Carrier mixing $\beta$",
    title=r"(b) Carrier-mixing trade-off",
    selected_x=SELECTED_BETA,
    xlim=(0.295, 0.405),
    xticks=[
        0.30,
        0.32,
        0.34,
        0.36,
        0.38,
        0.40
    ]
)


# ============================================================
# SHARED LEGEND
# ============================================================

# Only the two quantities in the legend.
# The black circles are explained in the LaTeX caption instead.
fig.legend(
    handles=[
        detection_line,
        fidelity_line
    ],
    labels=[
        r"Detection (TPR@1\% FPR)",
        r"Fidelity loss ($-\Delta$CLIP)"
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.975),
    ncol=2,
    frameon=False,
    columnspacing=3.0,
    handlelength=2.8,
    handletextpad=0.8
)


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    "fig_alpha_beta_tradeoff.png",
    dpi=400,
    bbox_inches="tight",
    pad_inches=0.06
)

plt.close(fig)

print("Saved: fig_alpha_beta_tradeoff.png")