#!/usr/bin/env python3
"""Merge final RESON SD2.1 generation shard manifests and run integrity checks."""
import argparse
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--out-dir", default="workspace/reson_sd21_canonical_10k")
p.add_argument("--expected", type=int, default=10000)
args = p.parse_args()

root = Path(args.out_dir)
files = sorted((root / "shards").glob("manifest_shard_*_of_*.csv"))
if not files:
    raise SystemExit("No shard manifests found.")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.sort_values("sample_id").reset_index(drop=True)

if len(df) != args.expected:
    raise RuntimeError(f"Expected {args.expected} rows, got {len(df)}")
if df["sample_id"].duplicated().any():
    raise RuntimeError("Duplicate sample_id found.")
if df["generation_seed"].isna().any() or df["prompt"].isna().any():
    raise RuntimeError("Missing prompt or generation_seed.")
if not set(df["bit"].astype(int).unique()).issubset({0, 1}):
    raise RuntimeError("Invalid payload bit.")

missing = []
for c in ["clean_path", "wm_path"]:
    for x in df[c]:
        if not Path(x).exists():
            missing.append(x)
if missing:
    raise RuntimeError(f"{len(missing)} images missing; first: {missing[:5]}")

manifest = root / "manifest.csv"
df.to_csv(manifest, index=False)

print("FINAL INTEGRITY CHECK")
print("---------------------")
print(f"Pairs       : {len(df)}")
print(f"Images      : {2*len(df)}")
print(f"Unique IDs  : {df['sample_id'].nunique()}")
print(f"Unique seeds: {df['generation_seed'].nunique()}")
print(f"Bits        : {df['bit'].value_counts().sort_index().to_dict()}")
if "split" in df:
    print(f"Splits      : {df['split'].value_counts().to_dict()}")
print(f"Manifest    : {manifest}")
