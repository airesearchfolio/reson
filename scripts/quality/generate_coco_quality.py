#!/usr/bin/env python3
"""
Generate final RESON SD2.1 COCO quality samples.

Purpose:
  Separate quality experiment only. Does NOT use detector train/val/test data.
  Generates paired Clean / RESON images from COCO captions with the canonical
  SD2.1 RESON configuration.

Canonical settings:
  SD2.1 base, 512x512, 50 steps, CFG 7.5, alpha=0.155, beta=0.40.

Designed to be:
  - resumable
  - deterministic
  - shardable across GPUs
  - compatible with the project's canonical Watermark implementation

IMPORTANT:
  This script creates generated images and a manifest. It does not compute FID.
"""
import argparse, csv, hashlib, json, os, sys
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline


def stable_int(*parts, mod=2**31-1):
    h = hashlib.sha256("||".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") % mod


def load_coco_captions(path):
    with open(path, "r") as f:
        data = json.load(f)
    images = {int(x["id"]): x for x in data["images"]}
    anns = sorted(data["annotations"], key=lambda x: int(x["id"]))
    rows = []
    for a in anns:
        iid = int(a["image_id"])
        if iid not in images:
            continue
        cap = str(a["caption"]).strip()
        if not cap:
            continue
        rows.append({
            "annotation_id": int(a["id"]),
            "image_id": iid,
            "file_name": images[iid]["file_name"],
            "prompt": cap,
        })
    return rows


def select_rows(rows, n, selection_seed):
    # Deterministic annotation-level selection. Multiple captions from one COCO
    # image are permitted, but generated samples are unique caption/seed pairs.
    keyed = []
    for r in rows:
        k = stable_int("reson-coco-final", selection_seed, r["annotation_id"], mod=2**63-1)
        keyed.append((k, r["annotation_id"], r))
    keyed.sort(key=lambda x: (x[0], x[1]))
    if len(keyed) < n:
        raise RuntimeError(f"Only {len(keyed)} captions available; requested {n}")
    return [x[2] for x in keyed[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="reson")
    ap.add_argument("--captions", default="workspace/paper_experiments/exp_16_coco_fid/annotations/captions_val2014.json")
    ap.add_argument("--coco-images", default="datasets/coco2014/val2014")
    ap.add_argument("--out-root", default="workspace/paper_experiments/exp_16_coco_fid/sd21_canonical_final_n10000")
    ap.add_argument("--model-id", default="sd2-community/stable-diffusion-2-1-base")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--selection-seed", type=int, default=20260920)
    ap.add_argument("--generation-seed-base", type=int, default=20260903)
    ap.add_argument("--alpha", type=float, default=0.155)
    ap.add_argument("--beta", type=float, default=0.40)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.project_root)
    sys.path.insert(0, str(root))
    from config import Config
    from watermark import Watermark

    out = Path(args.out_root)
    clean_dir = out / "clean"
    wm_dir = out / "reson"
    ref_dir = out / "real_reference_unique"
    shard_dir = out / "shards"
    for d in (clean_dir, wm_dir, ref_dir, shard_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows = select_rows(load_coco_captions(args.captions), args.n, args.selection_seed)
    selected = [(i, r) for i, r in enumerate(rows) if i % args.num_shards == args.shard_id]
    if args.limit is not None:
        selected = selected[:args.limit]

    # Copy/symlink unique real reference images selected by the 10K captions.
    # FID evaluation script can instead use the full COCO val2014 directory;
    # this folder is retained for transparent audit of selected references.
    for _, r in selected:
        src = Path(args.coco_images) / r["file_name"]
        dst = ref_dir / r["file_name"]
        if src.exists() and not dst.exists():
            try:
                dst.symlink_to(src)
            except Exception:
                pass

    cfg = Config()
    # Set canonical watermark fields explicitly.
    cfg.watermark.noise_mix_alpha = args.alpha
    cfg.watermark.carrier_blend = args.beta
    wm = Watermark(cfg).build()

    # Use FP32 for SD2.1 here. In this environment, passing externally-created
    # FP32 latents into an FP16 UNet can produce a timestep embedding
    # Float-vs-Half dtype mismatch. FP32 matches the canonical latent dtype and
    # avoids that failure.
    dtype = torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.set_progress_bar_config(disable=True)
    device = pipe._execution_device

    manifest_path = shard_dir / f"manifest_shard_{args.shard_id:02d}.csv"
    fields = ["sample_id","annotation_id","image_id","file_name","prompt",
              "generation_seed","bit","clean_path","wm_path","shard_id"]
    existing = set()
    if manifest_path.exists() and not args.overwrite:
        with open(manifest_path, newline="") as f:
            for r in csv.DictReader(f):
                existing.add(int(r["sample_id"]))

    mode = "a" if manifest_path.exists() and not args.overwrite else "w"
    with open(manifest_path, mode, newline="") as mf:
        w = csv.DictWriter(mf, fieldnames=fields)
        if mode == "w":
            w.writeheader()

        for j, (sid, r) in enumerate(selected, 1):
            cp = clean_dir / f"{sid:05d}.png"
            wp = wm_dir / f"{sid:05d}_wm.png"
            if sid in existing and cp.exists() and wp.exists() and not args.overwrite:
                continue

            seed = args.generation_seed_base + sid
            bit = sid % 2

            z0 = wm.null_latent(seed=seed, device=device)
            zw = wm.inject(bit=bit, seed=seed, device=device, alpha=args.alpha)
            # Explicit dtype/device alignment with the diffusion UNet.
            unet_dtype = next(pipe.unet.parameters()).dtype
            z0 = z0.to(device=device, dtype=unet_dtype)
            zw = zw.to(device=device, dtype=unet_dtype)

            common = dict(
                prompt=r["prompt"],
                negative_prompt="",
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                height=args.height,
                width=args.width,
            )
            with torch.inference_mode():
                if args.overwrite or not cp.exists():
                    im = pipe(latents=z0, **common).images[0]
                    im.save(cp)
                if args.overwrite or not wp.exists():
                    imw = pipe(latents=zw, **common).images[0]
                    imw.save(wp)

            w.writerow({
                "sample_id": sid,
                "annotation_id": r["annotation_id"],
                "image_id": r["image_id"],
                "file_name": r["file_name"],
                "prompt": r["prompt"],
                "generation_seed": seed,
                "bit": bit,
                "clean_path": str(cp),
                "wm_path": str(wp),
                "shard_id": args.shard_id,
            })
            mf.flush()
            if j % 25 == 0 or j == len(selected):
                print(f"[shard {args.shard_id}] {j}/{len(selected)}", flush=True)

    print("DONE:", manifest_path)


if __name__ == "__main__":
    main()
