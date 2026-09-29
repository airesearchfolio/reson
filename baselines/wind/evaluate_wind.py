#!/usr/bin/env python3
"""
Launch the existing WIND native lineage evaluator on the fair N=300 lineage.

IMPORTANT: this is WIND-specific. Tree-Ring, Gaussian Shading, and SERUM must
be evaluated with their own native detectors.

Expected fair lineage:
  workspace/fair_sd21_main_table/wind_lineage_n300/
    manifests/generation_0.csv
    manifests/generation_1_flux.csv
    manifests/generation_2_realvis.csv
    manifests/generation_3_dreamshaper.csv
    manifests/generation_4_flux.csv

Place this launcher in fair_sd21_main_table/. It searches for the existing
wind_lineage_auc_corrected.py / wind_lineage_auc.py in known experiment roots.
"""
from pathlib import Path
import subprocess, sys

EXP2=Path("workspace/2")
ROOT=EXP2/"fair_sd21_main_table"/"wind_lineage_n300"
MAN=ROOT/"manifests"

needed=[
 MAN/"generation_0.csv", MAN/"generation_1_flux.csv",
 MAN/"generation_2_realvis.csv", MAN/"generation_3_dreamshaper.csv",
 MAN/"generation_4_flux.csv"
]
missing=[str(p) for p in needed if not p.exists()]
if missing:
    raise FileNotFoundError("Missing lineage manifests:\n"+"\n".join(missing))

candidates=[
 Path(__file__).resolve().with_name("wind_lineage_auc_fair300.py"),
 EXP2/"wind_lineage_auc_corrected.py",
 EXP2/"wind_lineage_auc.py",
 EXP2/"baselines"/"Hidden-in-the-Noise"/"wind_lineage_auc_corrected.py",
 EXP2/"baselines"/"Hidden-in-the-Noise"/"wind_lineage_auc.py",
]
ev=next((p for p in candidates if p.exists()),None)
if ev is None:
    raise FileNotFoundError(
        "Could not find existing WIND evaluator. Looked in:\n"+
        "\n".join(str(p) for p in candidates)
    )

print("Evaluator:",ev)
print("Lineage root:",ROOT)
print("Manifest dir:",MAN)
print("N/depth: 300")
cmd=[
 sys.executable,str(ev),
 "--wind-root",str(ROOT),
 "--manifest-dir",str(MAN),
 "--n","300",
 "--target_fpr","0.01",
 "--gpu_id","0",
 "--resume",
]
print("Running:"," ".join(cmd),flush=True)
subprocess.run(cmd,check=True)
