#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

root = Path("workspace/reson_sd21_canonical_10k/alternative_lineages/order_a")
files = [root/"manifest_g4_shard0.csv", root/"manifest_g4_shard1.csv"]
for f in files:
    if not f.exists():
        raise FileNotFoundError(f)
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
if df.sample_id.duplicated().any():
    raise RuntimeError("Duplicate sample IDs across G4 shards")
df = df.sort_values("sample_id").reset_index(drop=True)
if len(df) != 1000:
    raise RuntimeError(f"Expected 1000 G4 rows, got {len(df)}")
out = root/"manifest_g4.csv"
df.to_csv(out, index=False)
print(f"Merged {len(df)} rows -> {out}")
