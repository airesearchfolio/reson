#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path


# ============================================================
# OUTPUT
# ============================================================

OUT_DIR = Path(
    "workspace/8/"
    "paper_experiments/plot_scripts"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "payload_capacity_ablation_combined.png"
OUT_PDF = OUT_DIR / "payload_capacity_ablation_combined.pdf"


# ============================================================
# DATA
# ============================================================

depths = ["G0", "G1", "G2", "G3", "G4"]
x_depth = np.arange(len(depths))

# ------------------------------------------------------------
# 1-bit
# ------------------------------------------------------------
# From final lineage evaluation.
payload_1bit = np.array([
    1.000,
    0.999,
    1.000,
    0.997,
    0.999,
])


# ------------------------------------------------------------
# 2-bit
# ------------------------------------------------------------
# bit0_accuracy = 1.0
# bit1_accuracy = 1.0
# at every lineage depth.
payload_2bit = np.array([
    1.000,
    1.000,
    1.000,
    1.000,
    1.000,
])


# ------------------------------------------------------------
# 4-bit
# IMPORTANT:
# Replace with exact mean_bit_accuracy from:
#
# final_lineage_4bit_evaluation/
# lineage_4bit_metrics_summary.csv
#
# These are currently the rounded values used in the
# previous visualization.
# ------------------------------------------------------------

payload_4bit = np.array([
    0.990,
    0.980,
    0.960,
    0.950,
    0.940,
])


# ------------------------------------------------------------
# 8-bit
# Exact values supplied from final G0-G4 evaluation.
# ------------------------------------------------------------

payload_8bit = np.array([
    0.951875,
    0.917375,
    0.889125,
    0.871000,
    0.867500,
])


# ============================================================
# PAPER STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,

    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,

    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,

    "legend.fontsize": 9,

    "axes.linewidth": 0.85,

    "figure.dpi": 150,
    "savefig.dpi": 600,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# COLORS
# ============================================================
# Panel (a): same payload colors as your current plot.
# 1-bit = blue
# 2-bit = green
# 4-bit = orange
# 8-bit = yellow/gold
# ============================================================

payload_colors = {
    "1-bit": "#4285F4",
    "2-bit": "#5DBB7A",
    "4-bit": "#F47C48",
    "8-bit": "#F6C344",
}

payload_markers = {
    "1-bit": "o",
    "2-bit": "o",
    "4-bit": "o",
    "8-bit": "o",
}


# ============================================================
# Panel (b): lineage-depth colors
# ============================================================

depth_colors = {
    "G0": "#1f77b4",
    "G1": "#ff7f0e",
    "G2": "#2ca02c",
    "G3": "#d62728",
    "G4": "#9467bd",
}

depth_markers = {
    "G0": "o",
    "G1": "s",
    "G2": "^",
    "G3": "D",
    "G4": "P",
}


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.2, 3.65),
)

ax1, ax2 = axes


# ============================================================
# PANEL (a)
# LINEAGE-DEPTH VIEW
# ============================================================

series_a = [
    ("1-bit", payload_1bit),
    ("2-bit", payload_2bit),
    ("4-bit", payload_4bit),
    ("8-bit", payload_8bit),
]

for label, values in series_a:

    ax1.plot(
        x_depth,
        values,
        color=payload_colors[label],
        marker=payload_markers[label],
        markersize=5.7,
        linewidth=2.0,
        label=label,
        markeredgecolor="white",
        markeredgewidth=0.55,
        zorder=3,
    )


# Chance level
ax1.axhline(
    0.50,
    color="0.50",
    linestyle=":",
    linewidth=1.0,
    zorder=1,
)


ax1.set_xlim(-0.18, 4.18)
ax1.set_ylim(0.48, 1.015)

ax1.set_xticks(x_depth)
ax1.set_xticklabels(depths)

ax1.set_yticks([
    0.50,
    0.625,
    0.75,
    0.875,
    1.00,
])

ax1.yaxis.set_major_formatter(
    PercentFormatter(
        xmax=1.0,
        decimals=0,
    )
)

ax1.set_xlabel("Lineage depth")
ax1.set_ylabel("Per-bit recovery accuracy")

ax1.set_title(
    "(a) Lineage-depth view",
    pad=8,
    fontweight="bold",
)


# Grid
ax1.grid(
    axis="y",
    linestyle=":",
    linewidth=0.7,
    alpha=0.38,
)

ax1.grid(
    axis="x",
    linestyle=":",
    linewidth=0.45,
    alpha=0.20,
)

ax1.set_axisbelow(True)


# Legend
ax1.legend(
    loc="lower left",
    ncol=2,
    frameon=False,
    handlelength=2.0,
    columnspacing=1.0,
    handletextpad=0.45,
    borderaxespad=0.55,
    labelspacing=0.35,
)


# ============================================================
# PANEL (b)
# PAYLOAD-CAPACITY VIEW
# ============================================================

payload_bits = np.array([1, 2, 4, 8])

# Build each lineage-depth trajectory across payload sizes.
capacity_by_depth = {
    "G0": [
        payload_1bit[0],
        payload_2bit[0],
        payload_4bit[0],
        payload_8bit[0],
    ],

    "G1": [
        payload_1bit[1],
        payload_2bit[1],
        payload_4bit[1],
        payload_8bit[1],
    ],

    "G2": [
        payload_1bit[2],
        payload_2bit[2],
        payload_4bit[2],
        payload_8bit[2],
    ],

    "G3": [
        payload_1bit[3],
        payload_2bit[3],
        payload_4bit[3],
        payload_8bit[3],
    ],

    "G4": [
        payload_1bit[4],
        payload_2bit[4],
        payload_4bit[4],
        payload_8bit[4],
    ],
}


for depth in depths:

    values = np.array(capacity_by_depth[depth])

    # Connecting line
    ax2.plot(
        payload_bits,
        values,
        color=depth_colors[depth],
        linewidth=1.7,
        alpha=0.92,
        zorder=2,
    )

    # Scatter points
    ax2.scatter(
        payload_bits,
        values,
        color=depth_colors[depth],
        marker=depth_markers[depth],
        s=45,
        edgecolors="white",
        linewidths=0.55,
        label=depth,
        zorder=3,
    )


ax2.set_xlim(0.65, 8.35)
ax2.set_ylim(0.84, 1.015)

ax2.set_xticks(payload_bits)
ax2.set_xticklabels(["1", "2", "4", "8"])

ax2.set_yticks([
    0.85,
    0.90,
    0.95,
    1.00,
])

ax2.yaxis.set_major_formatter(
    PercentFormatter(
        xmax=1.0,
        decimals=0,
    )
)

ax2.set_xlabel("Payload size (bits)")
ax2.set_ylabel("Per-bit recovery accuracy")

ax2.set_title(
    "(b) Payload-capacity view",
    pad=8,
    fontweight="bold",
)


# Grid
ax2.grid(
    axis="y",
    linestyle=":",
    linewidth=0.7,
    alpha=0.38,
)

ax2.grid(
    axis="x",
    linestyle=":",
    linewidth=0.45,
    alpha=0.20,
)

ax2.set_axisbelow(True)


# Legend
ax2.legend(
    loc="lower left",
    ncol=3,
    frameon=False,
    handlelength=1.5,
    columnspacing=0.9,
    handletextpad=0.35,
    borderaxespad=0.55,
    labelspacing=0.35,
)


# ============================================================
# SPINES
# ============================================================

for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)


# ============================================================
# LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.075,
    right=0.995,
    bottom=0.17,
    top=0.91,
    wspace=0.25,
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    OUT_PNG,
    bbox_inches="tight",
    pad_inches=0.025,
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
    pad_inches=0.025,
)

plt.close(fig)

print(f"Saved PNG: {OUT_PNG}")
print(f"Saved PDF: {OUT_PDF}")