#!/usr/bin/env python3
"""
Generate ONLY G4 for RESON alternative lineage Order A, using DreamShaperXL.

Input:
  alternative_lineages/order_a/manifest_g3.csv

Output:
  alternative_lineages/order_a/g4/
      00000.png
      00000_wm.png
      ...
  manifest_g4_shard0.csv / manifest_g4_shard1.csv when sharded

The clean and watermarked branches use the SAME prompt, transition seed,
strength, and img2img settings. The watermark is NOT reinserted.

Recommended:
  GPU 2: --num-shards 2 --shard-id 0
  GPU 3: --num-shards 2 --shard-id 1
"""
import argparse
import gc
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image

ROOT = Path("workspace/reson_sd21_canonical_10k")
ORDER = ROOT / "alternative_lineages" / "order_a"
INPUT_MANIFEST = ORDER / "manifest_g3.csv"
OUTDIR = ORDER / "g4"
MODEL_ID = "Lykon/dreamshaper-xl-1-0"

def load_pipe():
    print(f"Loading {MODEL_ID}", flush=True)
    try:
        pipe = AutoPipelineForImage2Image.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, variant="fp16"
        )
    except Exception as e:
        print(f"fp16 variant unavailable ({e}); retrying without variant.", flush=True)
        pipe = AutoPipelineForImage2Image.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16
        )
    pipe = pipe.to("cuda")
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return pipe

def generate(pipe, image_path, prompt, seed, strength):
    image = Image.open(image_path).convert("RGB")
    # Preserve the previous lineage resolution; make dimensions divisible by 8.
    w, h = image.size
    w = max(64, (w // 8) * 8)
    h = max(64, (h // 8) * 8)
    if image.size != (w, h):
        image = image.resize((w, h), Image.Resampling.LANCZOS)

    gen = torch.Generator(device="cuda").manual_seed(int(seed))
    out = pipe(
        prompt=str(prompt),
        image=image,
        strength=float(strength),
        guidance_scale=7.0,
        num_inference_steps=30,
        generator=gen,
    ).images[0]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--base-seed", type=int, default=20260903)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--smoke-n", type=int, default=0,
                    help="If >0, generate only this many new pairs for this shard.")
    args = ap.parse_args()

    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(f"Missing G3 manifest: {INPUT_MANIFEST}")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid shard configuration")

    df = pd.read_csv(INPUT_MANIFEST).head(args.n).copy()
    required = {"sample_id", "prompt", "clean_path", "wm_path"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"G3 manifest missing columns: {missing}")

    # Split the first N samples deterministically across GPUs.
    df = df.iloc[args.shard_id::args.num_shards].copy()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    manifest = (ORDER / "manifest_g4.csv" if args.num_shards == 1
                else ORDER / f"manifest_g4_shard{args.shard_id}.csv")

    if manifest.exists():
        old = pd.read_csv(manifest)
        rows = old.to_dict("records")
        done = set(old["sample_id"].astype(int))
    else:
        rows, done = [], set()

    todo = df[~df["sample_id"].astype(int).isin(done)].copy()
    print(f"Order A G4 shard {args.shard_id}/{args.num_shards}: "
          f"{len(done)} existing, {len(todo)} remaining", flush=True)

    pipe = load_pipe()
    made = 0

    for _, r in todo.iterrows():
        sid = int(r["sample_id"])
        # Dedicated G4 transition seed; identical for clean/wm pair.
        seed = int(args.base_seed + sid + 4_000_000)

        clean_out = OUTDIR / f"{sid:05d}.png"
        wm_out = OUTDIR / f"{sid:05d}_wm.png"

        clean_img = generate(pipe, r["clean_path"], r["prompt"], seed, args.strength)
        wm_img = generate(pipe, r["wm_path"], r["prompt"], seed, args.strength)
        clean_img.save(clean_out)
        wm_img.save(wm_out)

        rows.append({
            "sample_id": sid,
            "split": r.get("split", "test"),
            "chain": "order_a",
            "depth": "g4",
            "family": "sdxl",
            "model_id": MODEL_ID,
            "prompt": r["prompt"],
            "transition_seed": seed,
            "strength": args.strength,
            "clean_path": str(clean_out),
            "wm_path": str(wm_out),
        })
        outdf = pd.DataFrame(rows).drop_duplicates("sample_id", keep="last")
        outdf = outdf.sort_values("sample_id")
        outdf.to_csv(manifest, index=False)

        made += 1
        if made % 10 == 0:
            print(f"generated {made}/{len(todo)} new pairs", flush=True)
        if args.smoke_n > 0 and made >= args.smoke_n:
            print(f"Smoke test complete. Inspect {OUTDIR}", flush=True)
            break

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Saved shard manifest: {manifest}", flush=True)

if __name__ == "__main__":
    main()
