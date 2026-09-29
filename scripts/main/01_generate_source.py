#!/usr/bin/env python3
"""
Final paper-scale RESON SD2.1 G0 generation.

IMPORTANT:
- Uses the existing prompt split CSV exactly as supplied.
- Clean/WM for each sample use the same prompt, generation_seed, and base latent.
- Uses the project's canonical WatermarkEncoder from experiments/6/src_large/watermark.py.
- Does NOT reimplement the carrier.
- Defaults to physical GPU 7. For multi-GPU, launch one process per GPU/shard.
- Saves clean and WM images together: XXXXX.png and XXXXX_wm.png.
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


DEFAULT_PROJECT = "reson"
DEFAULT_PROMPTS = f"{DEFAULT_PROJECT}/prompt_split.csv"
DEFAULT_OUT = "workspace/reson_sd21_canonical_10k"
MODEL_ID = "sd2-community/stable-diffusion-2-1-base"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=DEFAULT_PROJECT)
    p.add_argument("--prompt-csv", default=DEFAULT_PROMPTS)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--gpu", type=int, default=7, help="Physical GPU id; default 7.")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.155)
    p.add_argument("--beta", type=float, default=0.40)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def first_existing(row, names):
    for n in names:
        if n in row and pd.notna(row[n]):
            return row[n]
    return None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def append_csv(path, fieldnames, row):
    path = Path(path)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def write_json_atomic(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def main():
    args = parse_args()
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("Require 0 <= shard-id < num-shards")

    # Set physical GPU before importing diffusers/project modules.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    project_root = Path(args.project_root).resolve()
    prompt_csv = Path(args.prompt_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    image_dir = out_dir / "images"
    shard_dir = out_dir / "shards"
    log_dir = out_dir / "logs"
    image_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    progress_log = log_dir / f"generation_shard_{args.shard_id:02d}_of_{args.num_shards:02d}.csv"
    failure_log = log_dir / f"failures_shard_{args.shard_id:02d}_of_{args.num_shards:02d}.log"
    summary_path = log_dir / f"summary_shard_{args.shard_id:02d}_of_{args.num_shards:02d}.json"
    run_start_wall = time.time()
    run_start_utc = utc_now()

    if not prompt_csv.exists():
        raise FileNotFoundError(prompt_csv)
    if not (project_root / "watermark.py").exists():
        raise FileNotFoundError(
            f"Canonical watermark.py not found at {project_root / 'watermark.py'}"
        )

    sys.path.insert(0, str(project_root))

    # Canonical project implementation: do not duplicate carrier math here.
    from config import Config
    from watermark import Watermark
    from diffusers import StableDiffusionPipeline

    cfg = Config()
    # Freeze final operating point in the canonical config object.
    if hasattr(cfg, "watermark"):
        # Names used by the canonical watermark.py implementation.
        cfg.watermark.noise_mix_alpha = args.alpha
        cfg.watermark.carrier_blend = args.beta
    else:
        raise RuntimeError("Config() has no .watermark field; inspect canonical config.py.")

    watermark = Watermark(cfg).build()

    df = pd.read_csv(prompt_csv)
    if len(df) != 10000:
        raise RuntimeError(f"Expected 10,000 rows, found {len(df)}")

    # Preserve original row order and split. We never reshuffle.
    rows = []
    for idx, row in df.iterrows():
        d = row.to_dict()
        sample_id = first_existing(d, ["sample_id", "id", "index"])
        if sample_id is None:
            sample_id = idx
        prompt = first_existing(d, ["prompt", "text", "caption"])
        bit = first_existing(d, ["bit", "payload_bit", "watermark_bit"])
        seed = first_existing(d, ["generation_seed", "seed", "gen_seed"])
        split = first_existing(d, ["split", "partition", "subset"])

        if prompt is None or seed is None or bit is None:
            raise RuntimeError(
                "CSV must contain prompt, generation_seed/seed, and bit columns. "
                f"Problem at row {idx}."
            )
        rows.append({
            **d,
            "_row_index": idx,
            "_sample_id": int(sample_id),
            "_prompt": str(prompt),
            "_bit": int(bit),
            "_seed": int(seed),
            "_split": "" if split is None else str(split),
        })

    rows = [r for r in rows if r["_row_index"] % args.num_shards == args.shard_id]
    print(f"GPU physical {args.gpu} | shard {args.shard_id}/{args.num_shards} | pairs {len(rows)}")
    print(f"Prompt CSV: {prompt_csv}")
    print(f"Output: {out_dir}")
    print(f"alpha={args.alpha} beta={args.beta} steps={args.steps} CFG={args.guidance}")
    print("Carrier: canonical project WatermarkEncoder")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # Keep scheduler from the SD2.1 pipeline unchanged (PNDM in the pilot).
    vae_scale = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    latent_h = args.height // vae_scale
    latent_w = args.width // vae_scale

    try:
        import diffusers
        diffusers_version = diffusers.__version__
    except Exception:
        diffusers_version = "unknown"

    provenance = {
        "run_start_utc": run_start_utc,
        "model_id": MODEL_ID,
        "prompt_csv": str(prompt_csv),
        "project_root": str(project_root),
        "canonical_watermark_py": str(project_root / "watermark.py"),
        "alpha": args.alpha,
        "beta": args.beta,
        "steps": args.steps,
        "guidance": args.guidance,
        "height": args.height,
        "width": args.width,
        "physical_gpu": args.gpu,
        "visible_device": device,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "scheduler": pipe.scheduler.__class__.__name__,
        "torch_version": torch.__version__,
        "diffusers_version": diffusers_version,
        "python_version": platform.python_version(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "carrier_implementation": "canonical project Watermark.build/inject/null_latent",
    }
    write_json_atomic(
        log_dir / f"provenance_shard_{args.shard_id:02d}_of_{args.num_shards:02d}.json",
        provenance,
    )

    log_fields = [
        "timestamp_utc", "sample_id", "split", "bit", "generation_seed",
        "status", "elapsed_sec", "clean_path", "wm_path", "error"
    ]
    completed = skipped = failed = 0
    manifest_rows = []
    for r in tqdm(rows, desc=f"shard{args.shard_id}-gpu{args.gpu}"):
        sid = r["_sample_id"]
        prompt = r["_prompt"]
        bit = r["_bit"]
        seed = r["_seed"]

        clean_path = image_dir / f"{sid:05d}.png"
        wm_path = image_dir / f"{sid:05d}_wm.png"
        sample_t0 = time.time()
        status = "generated"
        err = ""

        try:
            pair_complete = clean_path.exists() and wm_path.exists()
            if pair_complete and not args.overwrite:
                status = "skipped_existing"
                skipped += 1
            else:
                # If only one side exists, regenerate BOTH to preserve an exact pair.
                # Exact canonical project API: both branches derive from the
                # same generation seed; Watermark.inject applies the frozen carrier.
                z = watermark.null_latent(seed=seed, device=device).to(dtype=torch.float16)
                z_wm = watermark.inject(
                    bit=bit, seed=seed, device=device, alpha=args.alpha
                ).to(dtype=torch.float16)

                with torch.inference_mode():
                    clean = pipe(
                        prompt=prompt,
                        latents=z.clone(),
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        height=args.height,
                        width=args.width,
                    ).images[0]

                    wm = pipe(
                        prompt=prompt,
                        latents=z_wm,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        height=args.height,
                        width=args.width,
                    ).images[0]

                # Save through temporary files so a killed process is less likely
                # to leave a file that looks complete.
                clean_tmp = clean_path.with_name(clean_path.stem + ".tmp.png")
                wm_tmp = wm_path.with_name(wm_path.stem + ".tmp.png")
                clean.save(clean_tmp)
                wm.save(wm_tmp)
                os.replace(clean_tmp, clean_path)
                os.replace(wm_tmp, wm_path)
                completed += 1

            out = {k: v for k, v in r.items() if not k.startswith("_")}
            out.update({
                "sample_id": sid,
                "prompt": prompt,
                "bit": bit,
                "generation_seed": seed,
                "split": r["_split"],
                "clean_path": str(clean_path),
                "wm_path": str(wm_path),
                "source_model": MODEL_ID,
                "alpha": args.alpha,
                "beta": args.beta,
                "steps": args.steps,
                "guidance": args.guidance,
                "scheduler": pipe.scheduler.__class__.__name__,
                "carrier_impl": str(project_root / "watermark.py"),
                "shard_id": args.shard_id,
            })
            manifest_rows.append(out)

        except Exception as e:
            failed += 1
            status = "failed"
            err = f"{type(e).__name__}: {e}"
            with failure_log.open("a", encoding="utf-8") as f:
                f.write(f"\n[{utc_now()}] sample_id={sid} seed={seed} bit={bit}\n")
                f.write(traceback.format_exc())
                f.flush()
            # Remove incomplete temp files, if any.
            for tmp in [
                clean_path.with_name(clean_path.stem + ".tmp.png"),
                wm_path.with_name(wm_path.stem + ".tmp.png"),
            ]:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass

        append_csv(progress_log, log_fields, {
            "timestamp_utc": utc_now(),
            "sample_id": sid,
            "split": r["_split"],
            "bit": bit,
            "generation_seed": seed,
            "status": status,
            "elapsed_sec": round(time.time() - sample_t0, 4),
            "clean_path": str(clean_path),
            "wm_path": str(wm_path),
            "error": err,
        })

        # Continuously refresh a crash-tolerant run summary.
        write_json_atomic(summary_path, {
            **provenance,
            "last_update_utc": utc_now(),
            "assigned_pairs": len(rows),
            "generated_this_run": completed,
            "skipped_existing_this_run": skipped,
            "failed_this_run": failed,
            "processed_this_run": completed + skipped + failed,
            "elapsed_sec": round(time.time() - run_start_wall, 2),
            "progress_log": str(progress_log),
            "failure_log": str(failure_log),
        })

    shard_manifest = shard_dir / f"manifest_shard_{args.shard_id:02d}_of_{args.num_shards:02d}.csv"
    pd.DataFrame(manifest_rows).to_csv(shard_manifest, index=False)

    config_out = shard_dir / f"config_shard_{args.shard_id:02d}_of_{args.num_shards:02d}.json"
    with open(config_out, "w") as f:
        json.dump({
            "model_id": MODEL_ID,
            "prompt_csv": str(prompt_csv),
            "project_root": str(project_root),
            "canonical_watermark_py": str(project_root / "watermark.py"),
            "alpha": args.alpha,
            "beta": args.beta,
            "steps": args.steps,
            "guidance": args.guidance,
            "height": args.height,
            "width": args.width,
            "physical_gpu": args.gpu,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "scheduler": pipe.scheduler.__class__.__name__,
        }, f, indent=2)

    final_summary = {
        **provenance,
        "run_end_utc": utc_now(),
        "assigned_pairs": len(rows),
        "manifest_rows": len(manifest_rows),
        "generated_this_run": completed,
        "skipped_existing_this_run": skipped,
        "failed_this_run": failed,
        "elapsed_sec": round(time.time() - run_start_wall, 2),
        "shard_manifest": str(shard_manifest),
        "progress_log": str(progress_log),
        "failure_log": str(failure_log),
    }
    write_json_atomic(summary_path, final_summary)

    print("\nFINAL SHARD SUMMARY")
    print("-------------------")
    print(f"Assigned pairs : {len(rows)}")
    print(f"Generated      : {completed}")
    print(f"Skipped/resume : {skipped}")
    print(f"Failed         : {failed}")
    print(f"Elapsed sec    : {final_summary['elapsed_sec']}")
    print(f"Shard manifest : {shard_manifest}")
    print(f"Progress log   : {progress_log}")
    print(f"Failure log    : {failure_log}")
    print(f"Summary JSON   : {summary_path}")
    print(f"Images         : {image_dir}")


if __name__ == "__main__":
    main()
