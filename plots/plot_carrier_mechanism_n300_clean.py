#!/usr/bin/env python3
"""
Publication-ready controlled carrier-mechanism ablation.

Original UNSWAPPED mapping:
    beta = 0.0  -> white carrier
    beta = 0.4  -> blended carrier
    beta = 1.0  -> structured carrier

N = 300 matched prompts per condition.

All values are embedded.
No external CSV is required.

Output:
    fig_carrier_mechanism_n300.png
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# DATA
# ============================================================

x = np.arange(5)
depths = ["G0", "G1", "G2", "G3", "G4"]

series = [
    {
        "name": r"$\beta=0$ (white)",
        "marker": "o",

        # Frozen-threshold TPR (%)
        "tpr": np.array([
            19.0,
            3.3333333333,
            3.3333333333,
            1.3333333333,
            1.6666666667,
        ]),

        # 95% confidence intervals (%)
        "ci": np.array([
            [14.9634850334, 23.8203791825],
            [1.8204947339, 6.0261535804],
            [1.8204947339, 6.0261535804],
            [0.5196961739, 3.3775530253],
            [0.7139477351, 3.8415394833],
        ]),
    },

    {
        "name": r"$\beta=0.4$ (blend)",
        "marker": "s",

        "tpr": np.array([
            93.3333333333,
            87.6666666667,
            74.3333333333,
            61.6666666667,
            64.3333333333,
        ]),

        "ci": np.array([
            [89.9276940484, 95.6432484455],
            [83.4626275971, 90.9182685706],
            [69.1047091230, 78.9466662774],
            [56.0503139167, 66.9880167547],
            [58.7629765652, 69.5412582597],
        ]),
    },

    {
        "name": r"$\beta=1$ (structured)",
        "marker": "^",

        "tpr": np.array([
            98.3333333333,
            95.6666666667,
            88.3333333333,
            79.6666666667,
            81.6666666667,
        ]),

        "ci": np.array([
            [96.1584605167, 99.2860522649],
            [92.7282534366, 97.4503551916],
            [84.2066856676, 91.4906865385],
            [74.7505688715, 83.8326148358],
            [76.8971382323, 85.6354735898],
        ]),
    },
]


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "legend.fontsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.9,
})


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.15, 2.55)
)

# Compact vertical layout.
#
# top=0.80 moves the panels closer to the legend.
# bottom=0.18 leaves enough room for x-axis labels.
fig.subplots_adjust(
    left=0.085,
    right=0.985,
    bottom=0.18,
    top=0.80,
    wspace=0.31
)


# ============================================================
# PLOT SERIES
# ============================================================

legend_handles = []

for item in series:

    vals = item["tpr"]
    ci = item["ci"]

    # --------------------------------------------------------
    # 95% CI error bars
    # --------------------------------------------------------

    lower_error = vals - ci[:, 0]
    upper_error = ci[:, 1] - vals

    yerr = np.vstack([
        lower_error,
        upper_error
    ])

    # --------------------------------------------------------
    # PANEL A:
    # Absolute frozen-threshold TPR
    # --------------------------------------------------------

    h = axes[0].errorbar(
        x,
        vals,
        yerr=yerr,
        marker=item["marker"],
        markersize=5.2,
        linewidth=2.0,
        capsize=2.5,
        capthick=1.0,
        elinewidth=1.0,
        label=item["name"],
    )

    legend_handles.append(h)

    # --------------------------------------------------------
    # PANEL B:
    # TPR retention relative to each condition's G0
    # --------------------------------------------------------

    retention = vals / vals[0] * 100.0

    axes[1].plot(
        x,
        retention,
        marker=item["marker"],
        markersize=5.2,
        linewidth=2.0,
        label=item["name"],
    )


# ============================================================
# COMMON AXIS FORMATTING
# ============================================================

for ax in axes:

    ax.set_xticks(
        x,
        depths
    )

    ax.set_xlabel(
        "Lineage depth",
        labelpad=4
    )

    ax.set_ylim(
        0,
        105
    )

    ax.set_yticks([
        0,
        20,
        40,
        60,
        80,
        100
    ])

    # Light horizontal grid only
    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.7,
        alpha=0.45
    )

    ax.grid(
        axis="x",
        visible=False
    )


# ============================================================
# PANEL A
# ============================================================

axes[0].set_ylabel(
    "Frozen-threshold TPR (%)",
    labelpad=5
)

axes[0].set_title(
    "(a) Absolute persistence",
    pad=5
)


# ============================================================
# PANEL B
# ============================================================

axes[1].set_ylabel(
    "TPR retention vs. G0 (%)",
    labelpad=5
)

axes[1].set_title(
    "(b) G0-normalized persistence",
    pad=5
)


# ============================================================
# SHARED LEGEND
# ============================================================

# IMPORTANT:
# This uses legend_handles from the carrier series.
# There is NO detection_line / fidelity_line in this script.

fig.legend(
    legend_handles,
    [s["name"] for s in series],

    loc="upper center",

    # Close to panel titles without overlapping them
    bbox_to_anchor=(0.5, 0.965),

    ncol=3,
    frameon=False,

    # Compact legend spacing
    columnspacing=1.8,
    handlelength=2.2,
    handletextpad=0.5,

    borderaxespad=0.0
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    "fig_carrier_mechanism_n300.png",
    dpi=400,
    bbox_inches="tight",
    pad_inches=0.03
)

plt.close(fig)

print("Saved: fig_carrier_mechanism_n300.png")