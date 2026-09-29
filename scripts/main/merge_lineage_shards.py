#!/usr/bin/env python3
"""Merge two or more disjoint lineage-generation shard manifests."""
import argparse
from pathlib import Path
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument("--order-dir", required=True)
ap.add_argument("--num-shards", type=int, default=2)
ap.add_argument("--expected-n", type=int, default=200)
a=ap.parse_args()
od=Path(a.order_dir)

for depth in ("g1","g2","g3","g4"):
    fs=[od/f"manifest_{depth}_shard{i}.csv" for i in range(a.num_shards)]
    missing=[str(f) for f in fs if not f.exists()]
    if missing:
        raise FileNotFoundError(f"{depth}: missing shard manifests: {missing}")
    df=pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    if df.sample_id.duplicated().any():
        dup=df.loc[df.sample_id.duplicated(),"sample_id"].tolist()[:10]
        raise RuntimeError(f"{depth}: duplicate sample IDs across shards: {dup}")
    df=df.sort_values("sample_id").reset_index(drop=True)
    if len(df)!=a.expected_n:
        raise RuntimeError(f"{depth}: expected {a.expected_n} rows, got {len(df)}")
    out=od/f"manifest_{depth}.csv"
    df.to_csv(out,index=False)
    print(f"{depth}: merged {len(df)} rows -> {out}")
print("All lineage manifests merged successfully.")
