#!/usr/bin/env python3
"""
Multi-GPU RESON + SD2.1 generation with deterministic per-sample latents.

Designed for controlled N=1000 experiments:
  - SD2.1: sd2-community/stable-diffusion-2-1-base
  - 512x512, 50 steps, CFG 7.5
  - RESON beta=.40, alpha=.155 by default
  - clean/WM use EXACTLY the same base Gaussian latent
  - each sample latent is determined by GLOBAL sample ID, independent of:
      * GPU assignment
      * number of GPUs
      * batch size
      * shard ordering
  - each worker writes its own shard manifest
  - parent process merges manifests after all workers finish

Example:
python run_reson_sd21_multigpu.py \
  --prompts reson/reson_final_diverse_1000_prompts.txt \
  --out-root workspace/paper_experiments/reson_sd21_alpha0155_1k \
  --limit 1000 \
  --gpus 2,3,4,5 \
  --alpha 0.155 \
  --beta 0.40 \
  --steps 50 \
  --guidance 7.5 \
  --batch-size 8 \
  --seed 42
"""

import argparse
import csv
import math
import multiprocessing as mp
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--gpus", default="2,3,4,5",
                   help="Physical GPU IDs, e.g. 2,3,4,5")
    p.add_argument("--model-id",
                   default="sd2-community/stable-diffusion-2-1-base")
    p.add_argument("--alpha", type=float, default=0.155)
    p.add_argument("--beta", type=float, default=0.40)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed; sample i uses deterministic seed derived from this + global ID.")
    p.add_argument("--carrier-seed", type=int, default=20260903 + 55555)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_prompts(path, limit):
    xs = [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines()
          if x.strip()]
    if limit > 0:
        xs = xs[:limit]
    if not xs:
        raise RuntimeError("No prompts found.")
    return xs


def stable_sample_seed(base_seed, sample_id):
    # Simple explicit mapping recorded in manifest.
    # Keeps latent invariant to sharding and batch size.
    return int(base_seed + sample_id)


def make_carrier(beta, seed, device, dtype):
    # Carrier RNG is completely separate from per-image RNG.
    g = torch.Generator(device="cpu").manual_seed(seed)
    white = torch.randn((1, 4, 64, 64), generator=g, dtype=torch.float32)

    smooth = white.clone()
    for _ in range(8):
        smooth = F.avg_pool2d(smooth, kernel_size=3, stride=1, padding=1)

    def norm(x):
        return (x - x.mean()) / x.std().clamp_min(1e-8)

    white = norm(white)
    smooth = norm(smooth)
    carrier = beta * smooth + (1.0 - beta) * white
    carrier = norm(carrier)
    return carrier.to(device=device, dtype=dtype)


def latent_for_sample(base_seed, sample_id):
    seed = stable_sample_seed(base_seed, sample_id)
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn((4, 64, 64), generator=g, dtype=torch.float32)
    return z, seed


def worker(rank, physical_gpu, sample_ids, prompts, cfg):
    # Restrict this subprocess to one physical GPU BEFORE importing diffusers.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)

    from diffusers import AutoPipelineForText2Image

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dtype = torch.float16

    # Keep setup deterministic where practical.
    random.seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)
    torch.manual_seed(cfg["seed"] + rank)
    torch.cuda.manual_seed_all(cfg["seed"] + rank)

    out_root = Path(cfg["out_root"])
    image_dir = out_root / "images"
    shard_dir = out_root / "shards"
    image_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    print(f"[worker {rank}] physical GPU {physical_gpu}: "
          f"{len(sample_ids)} samples", flush=True)

    pipe = AutoPipelineForText2Image.from_pretrained(
        cfg["model_id"],
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    carrier = make_carrier(
        cfg["beta"], cfg["carrier_seed"], device, dtype
    )

    rows = []

    for start in range(0, len(sample_ids), cfg["batch_size"]):
        ids = sample_ids[start:start + cfg["batch_size"]]

        batch_prompts = [prompts[i] for i in ids]

        zs = []
        latent_seeds = []
        for i in ids:
            z, zseed = latent_for_sample(cfg["seed"], i)
            zs.append(z)
            latent_seeds.append(zseed)

        z = torch.stack(zs).to(device=device, dtype=dtype)

        bits = torch.tensor(
            [i % 2 for i in ids],
            device=device,
            dtype=dtype,
        )
        signs = (2.0 * bits - 1.0).view(-1, 1, 1, 1)

        z_wm = (
            z + cfg["alpha"] * signs * carrier
        ) / math.sqrt(1.0 + cfg["alpha"] ** 2)

        common = dict(
            prompt=batch_prompts,
            negative_prompt=[""] * len(ids),
            num_inference_steps=cfg["steps"],
            guidance_scale=cfg["guidance"],
        )

        with torch.inference_mode():
            clean_images = pipe(latents=z.clone(), **common).images
            wm_images = pipe(latents=z_wm.clone(), **common).images

        for j, i in enumerate(ids):
            clean_path = image_dir / f"{i:05d}.png"
            wm_path = image_dir / f"{i:05d}_wm.png"

            if cfg["overwrite"] or not clean_path.exists():
                clean_images[j].save(clean_path)
            if cfg["overwrite"] or not wm_path.exists():
                wm_images[j].save(wm_path)

            rows.append({
                "sample_id": i,
                "prompt": prompts[i],
                "bit": int(i % 2),
                "latent_seed": latent_seeds[j],
                "base_seed": cfg["seed"],
                "worker_rank": rank,
                "physical_gpu": physical_gpu,
                "model_id": cfg["model_id"],
                "scheduler": pipe.scheduler.__class__.__name__,
                "steps": cfg["steps"],
                "guidance": cfg["guidance"],
                "alpha": cfg["alpha"],
                "beta": cfg["beta"],
                "carrier_seed": cfg["carrier_seed"],
                "clean_image_path": str(clean_path.resolve()),
                "wm_image_path": str(wm_path.resolve()),
            })

        print(
            f"[worker {rank} | GPU {physical_gpu}] "
            f"{min(start + len(ids), len(sample_ids))}/{len(sample_ids)}",
            flush=True,
        )

    shard_manifest = shard_dir / f"manifest_shard_{rank:02d}.csv"
    with shard_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[worker {rank}] DONE -> {shard_manifest}", flush=True)


def merge_manifests(out_root, num_workers, expected_n):
    import pandas as pd

    out_root = Path(out_root)
    dfs = []
    for rank in range(num_workers):
        p = out_root / "shards" / f"manifest_shard_{rank:02d}.csv"
        if not p.exists():
            raise RuntimeError(f"Missing shard manifest: {p}")
        dfs.append(pd.read_csv(p))

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("sample_id").reset_index(drop=True)

    if len(df) != expected_n:
        raise RuntimeError(
            f"Merged manifest has {len(df)} rows; expected {expected_n}"
        )
    if df["sample_id"].nunique() != expected_n:
        raise RuntimeError("Duplicate/missing sample IDs in merged manifest.")

    expected_ids = list(range(expected_n))
    if df["sample_id"].tolist() != expected_ids:
        raise RuntimeError("Merged sample IDs are not exactly 0..N-1.")

    # Verify all generated files exist.
    missing = []
    for col in ["clean_image_path", "wm_image_path"]:
        for p in df[col]:
            if not Path(p).exists():
                missing.append(p)
    if missing:
        raise RuntimeError(
            f"{len(missing)} generated files missing; first: {missing[0]}"
        )

    merged = out_root / "manifest.csv"
    df.to_csv(merged, index=False)

    print("\n==============================================")
    print("ALL WORKERS COMPLETE")
    print("Merged manifest:", merged)
    print("Pairs:", len(df))
    print("Images:", len(df) * 2)
    print("==============================================")
    return merged


def save_carrier_copy(out_root, beta, seed):
    # Save exact float32 carrier once for experiment provenance.
    carrier = make_carrier(
        beta=beta,
        seed=seed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    path = Path(out_root) / f"carrier_beta{beta:.3f}.npy"
    np.save(path, carrier.numpy())
    return path


def main():
    args = parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("No GPUs supplied.")

    prompts = load_prompts(args.prompts, args.limit)
    n = len(prompts)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Contiguous balanced shards.
    all_ids = np.arange(n)
    shards = [x.tolist() for x in np.array_split(all_ids, len(gpu_ids))]

    cfg = {
        "out_root": str(out_root),
        "model_id": args.model_id,
        "alpha": args.alpha,
        "beta": args.beta,
        "steps": args.steps,
        "guidance": args.guidance,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "carrier_seed": args.carrier_seed,
        "overwrite": args.overwrite,
    }

    carrier_path = save_carrier_copy(
        out_root, args.beta, args.carrier_seed
    )

    # Save experiment configuration.
    import json
    with (out_root / "generation_config.json").open("w") as f:
        json.dump({
            **cfg,
            "prompts_file": str(Path(args.prompts).resolve()),
            "n": n,
            "physical_gpus": gpu_ids,
            "latent_seed_rule": "latent_seed = base_seed + global_sample_id",
            "carrier_path": str(carrier_path.resolve()),
        }, f, indent=2)

    print("N:", n)
    print("GPUs:", gpu_ids)
    print("Shard sizes:", [len(x) for x in shards])
    print("alpha/beta:", args.alpha, args.beta)
    print("steps/CFG:", args.steps, args.guidance)
    print("base seed:", args.seed)
    print("latent rule: base_seed + global sample_id")
    print("carrier:", carrier_path)

    ctx = mp.get_context("spawn")
    processes = []

    for rank, (gpu, ids) in enumerate(zip(gpu_ids, shards)):
        p = ctx.Process(
            target=worker,
            args=(rank, gpu, ids, prompts, cfg),
        )
        p.start()
        processes.append(p)

    failed = []
    for rank, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            failed.append((rank, p.exitcode))

    if failed:
        raise RuntimeError(f"Worker failures: {failed}")

    merge_manifests(out_root, len(gpu_ids), n)


if __name__ == "__main__":
    mp.freeze_support()
    main()
