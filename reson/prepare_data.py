#!/usr/bin/env python3
"""
stage6/prepare_data.py -- builds everything the detector needs to
train, in three stages:

  --stage split        build train/val/test prompt split (from any CSV
                        with a 'prompt' column, e.g. SERUM's own large
                        manifest -- prompts/seeds only, never their
                        watermarked image paths)
  --stage source        generate clean + watermarked SOURCE images for
                        every split (SDXL, alpha=0 vs alpha=RESON carrier)
  --stage regen-train    regenerate train/val images through the SEEN
                        generators (RealVis, DreamShaper) at seen
                        strengths, for both clean and watermarked --
                        this is what lets the detector practice on
                        attacked images, not just clean ones

Usage
-----
python prepare_data.py --stage split --source-csv /path/to/manifest.csv --n-total 500
python prepare_data.py --stage source
python prepare_data.py --stage regen-train
# or all at once (source-csv required for 'split'):
python prepare_data.py --stage all --source-csv /path/to/manifest.csv --n-total 500
"""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from config import Config
from watermark import Watermark
from data import build_split, load_split
import diffusion as D


# ---------------------------------------------------------------------------
# Stage: source images (clean + watermarked, SDXL, generation-time)
# ---------------------------------------------------------------------------

def generate_source(config: Config, wm: Watermark) -> pd.DataFrame:
    split_df = load_split(config)
    config.source_dir.mkdir(parents=True, exist_ok=True)

    pipe = D.load_sdxl_txt2img(config)
    device = config.get_device()
    rows = []

    try:
        for _, r in tqdm(split_df.iterrows(), total=len(split_df), desc="source generation"):
            split_name = str(r["split"])
            bit = int(r["bit"])
            folder = config.source_dir / split_name
            folder.mkdir(parents=True, exist_ok=True)

            clean_path = folder / f"row{int(r.prompt_id):04d}__clean.png"
            wm_path = folder / f"row{int(r.prompt_id):04d}__bit{bit}.png"

            if not clean_path.exists():
                null_latent = wm.null_latent(int(r.generation_seed), device)
                D.generate_from_latent(pipe, null_latent, r.prompt, config).save(clean_path)

            if not wm_path.exists():
                wm_latent = wm.inject(bit, int(r.generation_seed), device)
                D.generate_from_latent(pipe, wm_latent, r.prompt, config).save(wm_path)

            rows.append({
                "prompt_id": int(r.prompt_id), "prompt": r.prompt, "split": split_name,
                "bit": bit, "generation_seed": int(r.generation_seed),
                "alpha": config.watermark.noise_mix_alpha,
                "carrier_blend": config.watermark.carrier_blend,
                "clean_image_path": str(clean_path.resolve()),
                "wm_image_path": str(wm_path.resolve()),
            })
    finally:
        D.cleanup_pipe(pipe)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(config.source_manifest, index=False)
    print("Saved:", config.source_manifest)
    return manifest


# ---------------------------------------------------------------------------
# Stage: regeneration-aware TRAIN/VAL data (seen generators only)
# ---------------------------------------------------------------------------

def generate_regen_train_data(config: Config, only_generator: str | None = None) -> None:
    if not config.source_manifest.exists():
        raise FileNotFoundError(f"{config.source_manifest} missing. Run --stage source first.")
    source = pd.read_csv(config.source_manifest)

    generator_ids = {
        "realvis": config.realvis_model_id,
        "dreamshaper": config.dreamshaper_model_id,
    }
    if only_generator is not None:
        if only_generator not in generator_ids:
            raise ValueError(f"Unknown generator '{only_generator}', choose from {list(generator_ids)}")
        generator_ids = {only_generator: generator_ids[only_generator]}

    for gen_name, model_id in generator_ids.items():
        pipe = D.load_attack_pipe("sdxl", model_id, config)

        for split in ["train", "val"]:
            rows_out = []
            split_df = source[source["split"] == split].reset_index(drop=True)

            for strength in config.training.train_strengths:
                img_dir = config.regen_train_dir / "images" / gen_name / split / f"strength_{strength:.2f}"
                (img_dir / "clean").mkdir(parents=True, exist_ok=True)
                (img_dir / "wm").mkdir(parents=True, exist_ok=True)

                for _, r in tqdm(
                    split_df.iterrows(), total=len(split_df),
                    desc=f"{gen_name} {split} strength={strength:.2f}",
                ):
                    pid = int(r.prompt_id)
                    seed = D.stable_seed(config.seed, "trainregen", pid, gen_name, f"{strength:.2f}")

                    clean_out = img_dir / "clean" / f"{pid:06d}.png"
                    wm_out = img_dir / "wm" / f"{pid:06d}.png"

                    if not clean_out.exists():
                        with Image.open(r.clean_image_path) as im:
                            init = im.convert("RGB").resize((config.image_size, config.image_size), Image.Resampling.LANCZOS)
                        D.run_attack(pipe, "sdxl", init, r.prompt, strength, seed, config).save(clean_out)

                    if not wm_out.exists():
                        with Image.open(r.wm_image_path) as im:
                            init = im.convert("RGB").resize((config.image_size, config.image_size), Image.Resampling.LANCZOS)
                        D.run_attack(pipe, "sdxl", init, r.prompt, strength, seed, config).save(wm_out)

                    rows_out.append({
                        "sample_id": pid, "split": split, "bit": int(r.bit),
                        "generator": gen_name, "strength": float(strength),
                        "clean_image_path": str(clean_out.resolve()),
                        "wm_image_path": str(wm_out.resolve()),
                    })

            mp = config.regen_train_manifest_dir / f"{gen_name}_{split}.csv"
            pd.DataFrame(rows_out).to_csv(mp, index=False)
            print("Saved:", mp)

        D.cleanup_pipe(pipe)

    print("\nRegeneration-aware training data complete.")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["split", "source", "regen-train", "all"], required=True)
    ap.add_argument("--source-csv", type=Path, default=None, help="Required for --stage split/all.")
    ap.add_argument("--n-total", type=int, default=500)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--generator", choices=["realvis", "dreamshaper"], default=None, help="Run only this generator (for splitting regen-train across multiple GPUs).")

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    config = Config()
    config.make_dirs()

    if args.stage in ("split", "all"):
        if args.source_csv is None:
            raise ValueError("--source-csv is required for --stage split/all")
        build_split(config, args.source_csv, args.n_total, args.train_frac, args.val_frac)

    if args.stage in ("source", "all"):
        wm = Watermark(config).build()
        generate_source(config, wm)

    if args.stage in ("regen-train", "all"):
        generate_regen_train_data(config, only_generator=args.generator)


if __name__ == "__main__":
    main()
