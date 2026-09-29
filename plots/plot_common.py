#!/usr/bin/env python3
# Publication plotting utilities for RESON / ICLR 2027.
# Uses matplotlib only. No seaborn. Saves PDF + PNG (300 dpi).
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DEPTHS = ["G0","G1","G2","G3","G4"]
TAU = 0.500754654

def save(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print("Saved:", stem.with_suffix(".pdf"))
    print("Saved:", stem.with_suffix(".png"))
