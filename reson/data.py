#!/usr/bin/env python3
"""
stage6/data.py -- builds train/val/test prompt splits from a large
prompt pool.

Reuses prompts + seeds from an existing manifest (any CSV with a
'prompt' column, optionally an 'attack_seed' column) -- we only take the
prompt TEXT and SEED, never any pre-existing watermarked image path,
since those images (if present in the source CSV) were made with
whatever OTHER method built them, not this one.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from config import Config


def build_split(
    config: Config,
    source_csv: Path,
    n_total: int = 500,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> pd.DataFrame:
    """train_frac + val_frac + test_frac must sum to 1.0 (test = remainder)."""
    if config.split_csv.exists():
        df = pd.read_csv(config.split_csv)
        print(f"Using existing split: {config.split_csv}  (n={len(df)})")
        return df

    raw = pd.read_csv(source_csv)
    if "prompt" not in raw.columns:
        raise KeyError(f"{source_csv} must contain a 'prompt' column.")

    # Dedupe by prompt text: a prompt can appear multiple times in a
    # multi-depth/multi-generator source manifest, we only want one row.
    raw = raw.drop_duplicates(subset=["prompt"]).reset_index(drop=True)

    if len(raw) < n_total:
        raise ValueError(
            f"Requested {n_total} prompts, only {len(raw)} unique prompts available."
        )

    picked = raw.sample(n=n_total, random_state=config.seed).reset_index(drop=True)

    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    n_test = n_total - n_train - n_val

    picked["split"] = (["train"] * n_train) + (["val"] * n_val) + (["test"] * n_test)

    # Balanced 0/1 bit assignment within each split.
    bits = np.zeros(len(picked), dtype=np.int64)
    for split_name in ["train", "val", "test"]:
        idx = np.where(picked["split"].values == split_name)[0]
        local_bits = np.array([i % 2 for i in range(len(idx))], dtype=np.int64)
        rng = np.random.default_rng(config.seed + (abs(hash(split_name)) % 10000))
        rng.shuffle(local_bits)
        bits[idx] = local_bits
    picked["bit"] = bits

    if "attack_seed" in picked.columns:
        seeds = picked["attack_seed"].astype(int)
    elif "generation_seed" in picked.columns:
        seeds = picked["generation_seed"].astype(int)
    else:
        seeds = config.seed + 100000 + picked.index

    out = pd.DataFrame({
        "prompt_id": picked.index,
        "prompt": picked["prompt"],
        "split": picked["split"],
        "bit": picked["bit"],
        "generation_seed": seeds,
    })

    config.split_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.split_csv, index=False)

    print("\nSplit sizes:")
    print(
        out.groupby(["split", "bit"]).size()
        .rename("n").reset_index().to_string(index=False)
    )
    print("Saved:", config.split_csv)
    return out


def load_split(config: Config) -> pd.DataFrame:
    if not config.split_csv.exists():
        raise FileNotFoundError(
            f"{config.split_csv} missing. Run: "
            "python prepare_data.py --stage split --source-csv <path> --n-total 500"
        )
    return pd.read_csv(config.split_csv)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--source-csv", type=Path, required=True)
    ap.add_argument("--n-total", type=int, default=500)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    cfg = Config()
    cfg.make_dirs()
    build_split(cfg, args.source_csv, args.n_total, args.train_frac, args.val_frac)
