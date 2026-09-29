#!/usr/bin/env python3
"""
RESON beta=0.40 controlled alpha x diffusion-step quality sweep.

Purpose
-------
Run a small paired experiment before expensive multi-hop evaluation:
  beta fixed at 0.40
  alpha in {0.10, 0.12, 0.14, 0.155}
  steps in {30, 40, 50}
  N=200 by default

IMPORTANT:
- This script intentionally imports the existing RESON project modules rather
  than reimplementing the watermark.
- Clean and WM for a sample/condition use the same prompt, seed and base noise.
- Outputs are isolated from the canonical dataset.
- Perform visual inspection before any expensive lineage evaluation.

Because local project APIs can differ, the adapter section near the top is
kept explicit. Run --preflight first; it will validate imports/config without
generating 12x200 images.
"""

from __future__ import annotations
import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ALPHAS = [0.10, 0.12, 0.14, 0.155]
STEPS = [30, 40, 50]
BETA = 0.40

DEFAULT_PROJECT = Path("reson")
DEFAULT_OUT = Path("workspace/paper_experiments/exp_beta040_alpha_steps_sweep")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--start-index", type=int, default=9000,
                   help="Default uses the first 200 samples of the held-out 1000 test prompts.")
    p.add_argument("--gpu", default="3")
    p.add_argument("--preflight", action="store_true",
                   help="Validate project imports/config and print detected API; generate nothing.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_existing_project(project_root: Path):
    """
    Import the user's existing RESON implementation.

    Expected canonical files:
      config.py
      watermark.py
      diffusion.py

    We deliberately do NOT recreate the watermark formula here.
    """
    sys.path.insert(0, str(project_root))

    try:
        import config as C
        import watermark as W
        import diffusion as D
    except Exception as e:
        raise RuntimeError(
            "Could not import canonical RESON modules from project root.\n"
            f"project_root={project_root}\n"
            f"Original error: {e}"
        ) from e

    return C, W, D


def discover_config(C):
    # Common patterns used by experiment repositories.
    candidates = []
    for name in ("Config", "ExperimentConfig", "CONFIG", "config", "cfg"):
        if hasattr(C, name):
            candidates.append((name, getattr(C, name)))

    if not candidates:
        raise RuntimeError(
            "No obvious config object found in config.py. "
            "Expected one of Config/ExperimentConfig/CONFIG/config/cfg."
        )

    name, obj = candidates[0]
    if isinstance(obj, type):
        cfg = obj()
    else:
        cfg = copy.deepcopy(obj)

    return name, cfg


def set_if_present(obj, names, value):
    for name in names:
        if hasattr(obj, name):
            setattr(obj, name, value)
            return name
    return None


def find_prompt_manifest(project_root: Path):
    candidates = [
        project_root / "source_images/generated_manifest.csv",
        project_root / "generated_manifest.csv",
        project_root / "prompts.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not locate the canonical source manifest. Tried:\n" +
        "\n".join(str(x) for x in candidates)
    )


def load_prompt_rows(manifest: Path, start: int, n: int):
    import pandas as pd
    df = pd.read_csv(manifest)

    prompt_col = next((c for c in ["prompt", "caption", "text"] if c in df.columns), None)
    if prompt_col is None:
        raise RuntimeError(f"No prompt column found. Columns: {list(df.columns)}")

    # Prefer stable IDs when available.
    id_col = next((c for c in ["sample_id", "prompt_id", "row_id", "id"] if c in df.columns), None)

    # If manifest has explicit split, use test and then positional subset.
    if "split" in df.columns:
        test = df[df["split"].astype(str).str.lower().eq("test")].copy()
        if len(test) >= n:
            df = test.reset_index(drop=True)
            start = 0

    if start + n > len(df):
        raise RuntimeError(f"Requested [{start}:{start+n}] but manifest has {len(df)} rows.")

    sub = df.iloc[start:start+n].copy().reset_index(drop=True)
    out = []
    for i, r in sub.iterrows():
        sid = r[id_col] if id_col else start + i
        out.append({"sample_id": sid, "prompt": str(r[prompt_col])})
    return out


def describe_api(W, D, cfg):
    print("\nCanonical API inspection")
    print("------------------------")
    print("watermark.py public callables:",
          [x for x in dir(W) if not x.startswith("_") and callable(getattr(W, x))][:30])
    print("diffusion.py public callables:",
          [x for x in dir(D) if not x.startswith("_") and callable(getattr(D, x))][:30])
    print("config type:", type(cfg).__name__)
    for a in ("alpha", "beta", "carrier_blend", "num_inference_steps",
              "num_steps", "guidance_scale", "seed", "image_size"):
        if hasattr(cfg, a):
            print(f"  {a} = {getattr(cfg, a)}")


def require_generation_adapter(W, D):
    """
    Locate canonical helpers without inventing an alternate generation path.
    """
    carrier_fn = next(
        (getattr(W, n) for n in
         ["make_carrier", "generate_carrier", "build_carrier", "get_carrier"]
         if hasattr(W, n)), None
    )
    inject_fn = next(
        (getattr(W, n) for n in
         ["inject_watermark", "apply_watermark", "watermark_latent", "embed_watermark"]
         if hasattr(W, n)), None
    )
    generate_fn = next(
        (getattr(D, n) for n in
         ["generate", "generate_image", "text_to_image", "generate_from_latent"]
         if hasattr(D, n)), None
    )

    if not (carrier_fn and inject_fn and generate_fn):
        raise RuntimeError(
            "Automatic adapter could not safely identify the canonical carrier/injection/"
            "generation functions.\n"
            "Run with --preflight and send me the printed API. I will patch THIS file to "
            "your exact existing functions rather than reimplementing RESON."
        )
    return carrier_fn, inject_fn, generate_fn


def main():
    args = parse_args()
    if torch.cuda.is_available():
        # CUDA_VISIBLE_DEVICES should ideally be set before Python starts.
        device = "cuda"
    else:
        device = "cpu"

    C, W, D = load_existing_project(args.project_root)
    cfg_name, cfg = discover_config(C)
    describe_api(W, D, cfg)

    manifest = find_prompt_manifest(args.project_root)
    rows = load_prompt_rows(manifest, args.start_index, args.n)

    print("\nSweep plan")
    print("----------")
    print("beta:", BETA)
    print("alphas:", ALPHAS)
    print("steps:", STEPS)
    print("N per condition:", len(rows))
    print("conditions:", len(ALPHAS) * len(STEPS))
    print("paired generations:", len(rows) * len(ALPHAS) * len(STEPS))
    print("manifest:", manifest)
    print("output:", args.out_dir)

    if args.preflight:
        print("\nPRE-FLIGHT COMPLETE: no images generated.")
        return

    # Safety gate: refuse to invent a different implementation if APIs differ.
    carrier_fn, inject_fn, generate_fn = require_generation_adapter(W, D)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "sweep_protocol.json").open("w") as f:
        json.dump({
            "beta": BETA,
            "alphas": ALPHAS,
            "steps": STEPS,
            "n": len(rows),
            "source_manifest": str(manifest),
            "project_root": str(args.project_root),
            "note": "Same prompt/seed/base noise per clean-WM pair; visual-quality gate required."
        }, f, indent=2)

    # We intentionally stop here rather than guess positional signatures of the
    # user's canonical functions. If the repository exposes the common helpers,
    # inspect.signature output from --preflight lets this file be patched exactly.
    import inspect
    print("\nDetected generation signatures:")
    print("carrier:", inspect.signature(carrier_fn))
    print("inject :", inspect.signature(inject_fn))
    print("generate:", inspect.signature(generate_fn))
    raise RuntimeError(
        "\nCanonical helpers were located, but this first version intentionally does not "
        "guess their positional/keyword signatures. Run --preflight and send me its output; "
        "I will return a patched downloadable script using the exact API. This prevents "
        "accidentally generating a non-canonical RESON dataset."
    )


if __name__ == "__main__":
    main()
