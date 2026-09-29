#!/usr/bin/env python3
"""
RESON lineage-order robustness generator.

Runs THREE alternative permutations of the SAME downstream model multiset used
in the canonical RESON lineage:

Canonical downstream multiset:
    FLUX, RealVisXL, DreamShaperXL, FLUX

This script launches one complete order per GPU, simultaneously.

Default alternative orders:
    order_p1: RealVisXL -> FLUX -> DreamShaperXL -> FLUX
    order_p2: DreamShaperXL -> FLUX -> RealVisXL -> FLUX
    order_p3: FLUX -> DreamShaperXL -> FLUX -> RealVisXL

G0 is NOT regenerated. Each order starts from the same N held-out canonical
test pairs and propagates clean/WM images through G1..G4 with no watermark
reinsertion.

Example:
    python generate_order_permutations.py --gpus 2,3,4 --n 300

Resume-safe:
    Existing per-stage manifest rows are reused. Re-running resumes missing IDs.

Visual-quality gate:
    After G1, the script prints a reminder to inspect generated clean/WM pairs.
    Use --stop-after-g1 to stop all orders after G1 for a manual quality check.
"""

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT_DEFAULT = Path(
    "workspace/reson_sd21_canonical_10k"
)

# Exact model set from the canonical G1-G4 lineage.
FLUX = ("flux", "black-forest-labs/FLUX.1-dev", "flux")
REALVIS = ("sdxl", "SG161222/RealVisXL_V5.0", "realvis_xl")
DREAM = ("sdxl", "Lykon/dreamshaper-xl-1-0", "dreamshaper_xl")

ORDERS = {
    # Same multiset as canonical [FLUX, RealVis, DreamShaper, FLUX],
    # but three different permutations.
    "order_p1": [REALVIS, FLUX, DREAM, FLUX],
    "order_p2": [DREAM, FLUX, REALVIS, FLUX],
    "order_p3": [FLUX, DREAM, FLUX, REALVIS],
}


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    p.add_argument("--gpus", default="2,3,4",
                   help="Three physical GPU IDs, e.g. 2,3,4")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--strength", type=float, default=0.5)
    p.add_argument("--base-seed", type=int, default=20260903)
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--flux-guidance-scale", type=float, default=3.5)
    p.add_argument("--stop-after-g1", action="store_true")
    p.add_argument("--worker-order", choices=list(ORDERS), default=None,
                   help=argparse.SUPPRESS)
    return p.parse_args()


def canonical_test_subset(root: Path, n: int):
    mf = root / "manifest.csv"
    if not mf.exists():
        raise FileNotFoundError(mf)
    d = pd.read_csv(mf)
    if "split" not in d.columns:
        raise RuntimeError("Canonical manifest has no split column.")
    d = d[d["split"].astype(str).str.lower().str.strip() == "test"].copy()
    d = d.sort_values("sample_id").head(n).copy()
    if len(d) != n:
        raise RuntimeError(f"Expected {n} held-out test rows, found {len(d)}.")

    # Recover paths when the canonical manifest does not explicitly store them.
    if "clean_path" not in d.columns or "wm_path" not in d.columns:
        def pick(sid, wm):
            names = [
                root / "images" / (f"{int(sid):05d}_wm.png" if wm else f"{int(sid):05d}.png"),
                root / (f"{int(sid):05d}_wm.png" if wm else f"{int(sid):05d}.png"),
            ]
            for x in names:
                if x.exists():
                    return str(x)
            return str(names[0])
        d["clean_path"] = [pick(x, False) for x in d.sample_id]
        d["wm_path"] = [pick(x, True) for x in d.sample_id]

    pc = "prompt" if "prompt" in d.columns else "caption"
    if pc != "prompt":
        d["prompt"] = d[pc].astype(str)

    for c in ["clean_path", "wm_path"]:
        missing = [x for x in d[c] if not Path(x).exists()]
        if missing:
            raise RuntimeError(f"{len(missing)} missing canonical {c} files; first: {missing[:3]}")

    return d[["sample_id", "prompt", "clean_path", "wm_path"]].copy()


def load_pipe(family, model_id):
    import torch
    from diffusers import AutoPipelineForImage2Image

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    pipe = pipe.to("cuda")
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return pipe


def free_pipe(pipe):
    del pipe
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass


def generate_one(pipe, family, prompt, src, seed, strength, steps, guidance, flux_guidance):
    import torch
    from PIL import Image

    image = Image.open(src).convert("RGB")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    kw = dict(
        prompt=str(prompt),
        image=image,
        strength=float(strength),
        num_inference_steps=int(steps),
        generator=gen,
    )
    if family == "flux":
        kw["guidance_scale"] = float(flux_guidance)
    else:
        kw["guidance_scale"] = float(guidance)

    with torch.inference_mode():
        return pipe(**kw).images[0]


def run_order(a, order_name):
    chain = ORDERS[order_name]
    out_root = a.root / "alternative_lineages" / order_name
    out_root.mkdir(parents=True, exist_ok=True)

    base = canonical_test_subset(a.root, a.n)
    prev = base.copy()

    config = {
        "order": order_name,
        "canonical_g0": "Stable Diffusion 2.1",
        "n": a.n,
        "split": "test",
        "strength": a.strength,
        "base_seed": a.base_seed,
        "watermark_reinserted_after_g0": False,
        "chain": [
            [f"g{i}", fam, mid, short]
            for i, (fam, mid, short) in enumerate(chain, start=1)
        ],
    }
    (out_root / "run_config.json").write_text(json.dumps(config, indent=2))

    print(f"\n[{order_name}] Starting on visible CUDA device 0", flush=True)
    print(f"[{order_name}] N={a.n}; held-out test IDs "
          f"{int(base.sample_id.min())}..{int(base.sample_id.max())}", flush=True)

    for gi, (family, model_id, short_name) in enumerate(chain, start=1):
        depth = f"g{gi}"
        stage_dir = out_root / depth
        stage_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_root / f"manifest_{depth}.csv"

        if manifest_path.exists():
            existing = pd.read_csv(manifest_path)
            done = set(existing.sample_id.astype(int))
        else:
            existing = pd.DataFrame()
            done = set()

        todo = prev[~prev.sample_id.astype(int).isin(done)].copy()
        print(f"[{order_name}] {depth}: {short_name}; "
              f"done={len(done)}, todo={len(todo)}", flush=True)

        if len(todo):
            pipe = load_pipe(family, model_id)
            rows = [] if existing.empty else existing.to_dict("records")

            for k, r in enumerate(todo.itertuples(index=False), start=1):
                sid = int(r.sample_id)
                # Same transition seed for the matched clean/WM pair.
                seed = int(a.base_seed + gi * 1_000_000 + sid)

                cp = stage_dir / f"{sid:05d}.png"
                wp = stage_dir / f"{sid:05d}_wm.png"

                clean = generate_one(
                    pipe, family, r.prompt, r.clean_path, seed,
                    a.strength, a.num_inference_steps,
                    a.guidance_scale, a.flux_guidance_scale
                )
                wm = generate_one(
                    pipe, family, r.prompt, r.wm_path, seed,
                    a.strength, a.num_inference_steps,
                    a.guidance_scale, a.flux_guidance_scale
                )
                clean.save(cp)
                wm.save(wp)

                rows.append({
                    "sample_id": sid,
                    "split": "test",
                    "chain": order_name,
                    "depth": depth,
                    "family": family,
                    "model_id": model_id,
                    "prompt": r.prompt,
                    "transition_seed": seed,
                    "strength": a.strength,
                    "clean_path": str(cp),
                    "wm_path": str(wp),
                })

                # Save frequently so interrupted jobs resume safely.
                if k % 5 == 0 or k == len(todo):
                    cur = (pd.DataFrame(rows)
                           .drop_duplicates("sample_id", keep="last")
                           .sort_values("sample_id"))
                    cur.to_csv(manifest_path, index=False)
                    print(f"[{order_name}] {depth}: {len(cur)}/{a.n}", flush=True)

            free_pipe(pipe)

        cur = pd.read_csv(manifest_path).drop_duplicates("sample_id").sort_values("sample_id")
        if len(cur) != a.n:
            raise RuntimeError(f"[{order_name}] {depth} incomplete: {len(cur)}/{a.n}")

        prev = cur[["sample_id", "prompt", "clean_path", "wm_path"]].copy()

        if gi == 1:
            print(
                f"\n[{order_name}] VISUAL QUALITY GATE:\n"
                f"  Inspect paired images in: {stage_dir}\n"
                f"  e.g. <id>.png versus <id>_wm.png\n",
                flush=True,
            )
            if a.stop_after_g1:
                print(f"[{order_name}] Stopping after G1 as requested.", flush=True)
                return

    print(f"[{order_name}] COMPLETE through G4.", flush=True)


def launch(a):
    gpu_ids = [x.strip() for x in a.gpus.split(",") if x.strip()]
    if len(gpu_ids) != 3:
        raise RuntimeError("--gpus must contain exactly three GPU IDs, e.g. --gpus 2,3,4")

    script = Path(__file__).resolve()
    orders = list(ORDERS)
    procs = []

    log_dir = a.root / "alternative_lineages" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for gpu, order in zip(gpu_ids, orders):
        log = log_dir / f"{order}_gpu{gpu}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu

        cmd = [
            sys.executable, str(script),
            "--root", str(a.root),
            "--n", str(a.n),
            "--strength", str(a.strength),
            "--base-seed", str(a.base_seed),
            "--num-inference-steps", str(a.num_inference_steps),
            "--guidance-scale", str(a.guidance_scale),
            "--flux-guidance-scale", str(a.flux_guidance_scale),
            "--worker-order", order,
        ]
        if a.stop_after_g1:
            cmd.append("--stop-after-g1")

        fh = open(log, "a")
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        procs.append((order, gpu, p, fh, log))
        print(f"Launched {order} on GPU {gpu}: PID={p.pid}")
        print(f"  log: {log}")

    failed = False
    for order, gpu, p, fh, log in procs:
        rc = p.wait()
        fh.close()
        print(f"{order} on GPU {gpu} exited with code {rc}; log={log}")
        if rc != 0:
            failed = True

    if failed:
        raise SystemExit(1)

    print("\nAll three lineage-order jobs completed.")


def main():
    a = cli()
    if a.worker_order:
        run_order(a, a.worker_order)
    else:
        launch(a)


if __name__ == "__main__":
    main()
