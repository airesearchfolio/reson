#!/usr/bin/env python3
"""
Plot RESON perturbation robustness.

Produces:
  perturbation_robustness.png
  perturbation_robustness.pdf

Plot:
  x-axis  : perturbation severity
  y-axis  : accuracy / rate
  lines   : frozen-threshold detection TPR + payload accuracy

The script can combine:
  1. original perturbation_metrics_summary.csv
  2. dense perturbation_metrics_summary.csv

Duplicate conditions are automatically removed.
"""

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--original",
        type=str,
        default=None,
        help="Original perturbation_metrics_summary.csv",
    )

    p.add_argument(
        "--dense",
        type=str,
        required=True,
        help="Dense perturbation_metrics_summary.csv",
    )

    p.add_argument(
        "--out-dir",
        type=str,
        default=".",
    )

    return p.parse_args()


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def normalize_columns(df):

    # support slightly different evaluator naming
    aliases = {
        "tpr_frozen": [
            "tpr_frozen",
            "watermark_detection_rate",
            "frozen_tpr",
        ],
        "bit_accuracy": [
            "bit_accuracy",
            "payload_accuracy",
            "mean_bit_accuracy",
        ],
    }

    for canonical, possibilities in aliases.items():
        if canonical not in df.columns:
            for x in possibilities:
                if x in df.columns:
                    df[canonical] = df[x]
                    break

    required = [
        "condition",
        "attack",
        "strength",
        "tpr_frozen",
        "bit_accuracy",
    ]

    missing = [x for x in required if x not in df.columns]

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return df


def severity_value(row):

    s = str(row["strength"]).lower()

    nums = re.findall(r"[-+]?\d*\.?\d+", s)

    if not nums:
        return 0.0

    v = float(nums[0])

    attack = str(row["attack"]).lower()

    # For JPEG, lower quality = stronger attack.
    # Convert so severity increases left -> right.
    if attack == "jpeg":
        return 100.0 - v

    # Crop keep fraction:
    # keep 0.9 -> severity .1
    # keep 0.5 -> severity .5
    if attack == "center_crop":
        return 1.0 - v

    # Resize similarly:
    # x0.75 is mild; x0.10 severe
    if attack == "resize":
        return 1.0 - v

    # brightness:
    # distance from identity 1.0
    if attack == "brightness":
        return abs(v - 1.0)

    # contrast:
    if attack == "contrast":
        return abs(v - 1.0)

    # blur/noise/rotation: larger = stronger
    return v


def pretty_strength(row):

    attack = str(row["attack"])
    s = str(row["strength"])

    if attack == "jpeg":
        return s.upper()

    if attack == "gaussian_blur":
        return s.replace("r", "r=")

    if attack == "gaussian_noise":
        return s.replace("std", "σ=")

    if attack == "resize":
        return s.replace("x", "×")

    if attack == "center_crop":
        x = float(re.findall(r"\d*\.?\d+", s)[0])
        return f"{int(round(x*100))}%"

    if attack in ["brightness", "contrast"]:
        return s.replace("x", "×")

    if attack == "rotation":
        x = re.findall(r"\d*\.?\d+", s)[0]
        return f"{x}°"

    return s


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    args = parse_args()

    dfs = []

    if args.original:
        d = pd.read_csv(args.original)
        d["source"] = "original"
        dfs.append(d)

    dense = pd.read_csv(args.dense)
    dense["source"] = "dense"
    dfs.append(dense)

    df = pd.concat(dfs, ignore_index=True)

    df = normalize_columns(df)

    # Dense result wins if a condition exists in both files
    df["_priority"] = (df["source"] == "dense").astype(int)

    df = (
        df.sort_values("_priority")
          .drop_duplicates("condition", keep="last")
          .copy()
    )

    # Do not plot unperturbed identity condition.
    df = df[
        (df["attack"].astype(str).str.lower() != "none")
        & (df["condition"].astype(str).str.lower() != "none")
    ].copy()

    df["severity"] = df.apply(severity_value, axis=1)
    df["display_strength"] = df.apply(pretty_strength, axis=1)

    families = [
        "jpeg",
        "gaussian_blur",
        "gaussian_noise",
        "resize",
        "center_crop",
        "brightness",
        "contrast",
        "rotation",
    ]

    families = [
        f for f in families
        if f in set(df["attack"])
    ]

    # -----------------------------------------------------
    # Figure
    # -----------------------------------------------------

    ncols = 4
    nrows = int(np.ceil(len(families) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14.5, 3.7 * nrows),
        sharey=True,
    )

    axes = np.asarray(axes).reshape(-1)

    title_map = {
        "jpeg": "JPEG compression",
        "gaussian_blur": "Gaussian blur",
        "gaussian_noise": "Gaussian noise",
        "resize": "Resize",
        "center_crop": "Center crop",
        "brightness": "Brightness",
        "contrast": "Contrast",
        "rotation": "Rotation",
    }

    # Same colors consistently across all panels
    DET_COLOR = "#3B6FB6"
    BIT_COLOR = "#D97932"

    for ax, family in zip(axes, families):

        g = df[df.attack == family].copy()

        #===== START : Correct perturbation x-axis ordering =====#
        if family in ["brightness", "contrast"]:
            # For photometric factors, show the actual factor from low -> high.
            # This avoids connecting values in distance-from-identity order.
            g["_plot_order"] = g["strength"].astype(str).str.extract(
                r"([-+]?\d*\.?\d+)", expand=False
            ).astype(float)
            g = g.sort_values("_plot_order")
        else:
            # Other perturbations progress from mild -> severe.
            g = g.sort_values("severity")
        #===== END : Correct perturbation x-axis ordering =====#

        x = np.arange(len(g))

        # Detection
        ax.plot(
            x,
            g["tpr_frozen"],
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=DET_COLOR,
            label="TPR (frozen threshold)"
        )

        # Payload
        ax.plot(
            x,
            g["bit_accuracy"],
            marker="s",
            linewidth=2.2,
            markersize=5.5,
            color=BIT_COLOR,
            label="Payload bit accuracy",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            g["display_strength"],
            fontsize=8.5,
        )

        ax.set_title(
            title_map.get(family, family),
            fontsize=11,
            fontweight="bold",
        )

        ax.set_ylim(0, 1.04)

        ax.set_yticks(
            np.arange(0, 1.01, 0.2)
        )

        ax.grid(
            axis="y",
            alpha=0.22,
            linewidth=0.7,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # # emphasize severe crop failure without annotation clutter
        # if family == "center_crop":
        #     ax.axhline(
        #         0.5,
        #         linestyle=":",
        #         linewidth=0.8,
        #         color="gray",
        #         alpha=0.5,
        #     )

    # remove unused axes
    for ax in axes[len(families):]:
        ax.axis("off")

    # common labels
    fig.supylabel(
        "Rate",
        fontsize=12,
    )

    # global legend
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=10.5,
    )

    fig.tight_layout(
        rect=[0.025, 0.02, 1, 0.96]
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    png = out / "perturbation_robustness.png"
    pdf = out / "perturbation_robustness.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nSaved:")
    print(png)
    print(pdf)

    print("\nConditions plotted:")
    print("\nConditions plotted:")
    print(
        df.sort_values(["attack", "severity"])[
            [
                "attack",
                "condition",
                "tpr_frozen",
                "bit_accuracy",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()