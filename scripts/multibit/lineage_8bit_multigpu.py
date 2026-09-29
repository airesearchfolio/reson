#!/usr/bin/env python3
"""
RESON 8-bit canonical G0->G4 lineage generator, N=1000 held-out test pairs.

Protocol:
  G0: existing SD2.1 8-bit test set (sample_id 9000..9999)
  G1: FLUX.1-dev
  G2: RealVisXL V5.0
  G3: DreamShaper XL 1.0
  G4: FLUX.1-dev

- Uses existing G0 clean + 8-bit watermarked images; DOES NOT regenerate G0.
- No watermark reinsertion after G0.
- Four-GPU sharding supported (default physical GPUs 4,5,6,7).
- Resume-safe: existing output images are reused.
- Same regeneration strength=.50 and deterministic transition seed convention
  used by the RESON lineage scripts.
- Writes manifest_g1.csv ... manifest_g4.csv plus manifest_g1_g4.csv.
"""

import argparse, gc, hashlib, json, os, subprocess, sys
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

ROOT_DEFAULT = Path(
    "workspace/paper_experiments/"
    "exp_16_coco_fid/sd21_8bit_alpha0155_beta040_n10000"
)
PROJECT_DEFAULT = Path("reson")

CHAIN = {
    1: ("flux", "black-forest-labs/FLUX.1-dev", "flux"),
    2: ("sdxl", "SG161222/RealVisXL_V5.0", "realvis_xl"),
    3: ("sdxl", "Lykon/dreamshaper-xl-1-0", "dreamshaper_xl"),
    4: ("flux", "black-forest-labs/FLUX.1-dev", "flux"),
}

def stable_seed(base, *parts):
    h = hashlib.sha256("|".join(map(str, (base,) + parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31 - 1)

def atomic_save(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.png")
    img.save(tmp)
    os.replace(tmp, path)

def load_test(root, expected_n):
    mf = root / "manifest_final_10000.csv"
    if not mf.exists():
        raise FileNotFoundError(mf)
    df = pd.read_csv(mf)
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df = df[df["split"] == "test"].copy().sort_values("sample_id").reset_index(drop=True)
    if len(df) != expected_n:
        raise RuntimeError(f"Expected {expected_n} test rows, got {len(df)}")
    if df["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate test sample IDs")
    if "prompt" not in df:
        raise RuntimeError("manifest_final_10000.csv has no prompt column")
    if "wm8_path" not in df:
        raise RuntimeError("manifest_final_10000.csv has no wm8_path column")
    for c in ["clean_path", "wm8_path"]:
        missing = [p for p in df[c].astype(str) if not Path(p).exists()]
        if missing:
            raise RuntimeError(f"{c}: {len(missing)} missing files; first={missing[0]}")
    return df

def stage_manifest(out, depth):
    return out / f"manifest_g{depth}.csv"

def load_previous_paths(root, out, df, depth):
    if depth == 1:
        return (
            {int(r.sample_id): Path(str(r.clean_path)) for _, r in df.iterrows()},
            {int(r.sample_id): Path(str(r.wm8_path)) for _, r in df.iterrows()},
        )
    pm = stage_manifest(out, depth - 1)
    if not pm.exists():
        raise FileNotFoundError(f"Previous stage manifest missing: {pm}")
    p = pd.read_csv(pm)
    if len(p) != len(df) or p.sample_id.nunique() != len(df):
        raise RuntimeError(f"Incomplete previous manifest {pm}: {len(p)}/{len(df)}")
    return (
        {int(r.sample_id): Path(str(r.clean_path)) for _, r in p.iterrows()},
        {int(r.sample_id): Path(str(r.wm_path)) for _, r in p.iterrows()},
    )

def worker(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    root, out = Path(args.data_root), Path(args.out_root)
    df = load_test(root, args.expected_n)
    df = df.iloc[args.shard_id::args.num_shards].copy()

    sys.path.insert(0, str(args.project_root))
    import diffusion as D
    from config import Config
    cfg = Config()

    # Previous-stage maps must be built from the FULL test set.
    full = load_test(root, args.expected_n)
    prevc, prevw = load_previous_paths(root, out, full, args.depth)

    family, model_id, prefix = CHAIN[args.depth]
    od = out / f"g{args.depth}"
    shard_dir = out / "shards" / f"g{args.depth}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    print(f"[G{args.depth} shard {args.shard_id}] GPU={args.physical_gpu} "
          f"N={len(df)} model={model_id}", flush=True)

    pipe = D.load_attack_pipe(family, model_id, cfg)
    rows = []
    for k, (_, r) in enumerate(tqdm(df.iterrows(), total=len(df),
                                    desc=f"G{args.depth} shard {args.shard_id}"), 1):
        sid = int(r.sample_id)
        prompt = str(r.prompt)
        cp = od / f"{prefix}_regen_{sid:05d}.png"
        wp = od / f"{prefix}_regen_{sid:05d}_wm.png"
        sd = stable_seed(args.base_seed, "evallineage", sid,
                         f"g{args.depth}", args.strength)

        if not cp.exists():
            with Image.open(prevc[sid]) as im:
                x = im.convert("RGB")
            atomic_save(D.run_attack(pipe, family, x, prompt,
                                     args.strength, sd, cfg), cp)

        if not wp.exists():
            with Image.open(prevw[sid]) as im:
                x = im.convert("RGB")
            atomic_save(D.run_attack(pipe, family, x, prompt,
                                     args.strength, sd, cfg), wp)

        row = {
            "sample_id": sid, "split": "test",
            "message_int": int(r.message_int),
            **{f"bit{i}": int(r[f"bit{i}"]) for i in range(8)},
            "prompt": prompt, "depth": f"g{args.depth}",
            "generator": prefix, "family": family, "model_id": model_id,
            "transition_seed": sd, "strength": args.strength,
            "clean_path": str(cp.resolve()), "wm_path": str(wp.resolve()),
            "payload_bits": 8, "watermark_reinserted": False,
            "shard_id": args.shard_id,
        }
        rows.append(row)
        if k % 25 == 0:
            print(f"[G{args.depth} shard {args.shard_id}] {k}/{len(df)}", flush=True)

    sm = shard_dir / f"shard_{args.shard_id}.csv"
    pd.DataFrame(rows).to_csv(sm, index=False)
    print(f"DONE shard -> {sm}", flush=True)

def merge_stage(out, depth, expected_n, num_shards):
    fs = [out / "shards" / f"g{depth}" / f"shard_{i}.csv"
          for i in range(num_shards)]
    for f in fs:
        if not f.exists():
            raise FileNotFoundError(f)
    m = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    m = m.sort_values("sample_id").drop_duplicates("sample_id", keep="last")
    if len(m) != expected_n:
        raise RuntimeError(f"G{depth} merge has {len(m)}/{expected_n} rows")
    for c in ["clean_path", "wm_path"]:
        bad = [p for p in m[c].astype(str) if not Path(p).exists()]
        if bad:
            raise RuntimeError(f"G{depth}: missing {c}; first={bad[0]}")
    dest = stage_manifest(out, depth)
    m.to_csv(dest, index=False)
    print(f"[MERGE] G{depth}: {len(m)} -> {dest}", flush=True)

def launch(args):
    out = Path(args.out_root)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if len(gpus) != 4:
        raise RuntimeError("--gpus must contain exactly four GPU IDs")

    config = {
        "experiment": "RESON 8-bit canonical lineage",
        "data_root": str(Path(args.data_root)),
        "checkpoint": str(Path(args.data_root) / "final_detector_8bit" / "detector_best.pt"),
        "expected_n": args.expected_n,
        "lineage": ["FLUX", "RealVisXL", "DreamShaperXL", "FLUX"],
        "strength": args.strength, "base_seed": args.base_seed,
        "gpus": gpus, "watermark_reinserted_after_g0": False,
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2))

    # Validate G0 before spending GPU time.
    load_test(Path(args.data_root), args.expected_n)
    print(f"G0 validation passed: {args.expected_n} existing 8-bit test pairs.", flush=True)

    for depth in range(args.start_depth, args.stop_depth + 1):
        print(f"\n========== START G{depth} ==========", flush=True)
        procs = []
        for shard_id, gpu in enumerate(gpus):
            log = out / "logs" / f"g{depth}_gpu{gpu}_shard{shard_id}.log"
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--worker", "--depth", str(depth),
                "--shard-id", str(shard_id), "--num-shards", "4",
                "--physical-gpu", str(gpu),
                "--data-root", str(args.data_root),
                "--out-root", str(args.out_root),
                "--project-root", str(args.project_root),
                "--expected-n", str(args.expected_n),
                "--base-seed", str(args.base_seed),
                "--strength", str(args.strength),
            ]
            fh = open(log, "w")
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            procs.append((p, fh, log))
            print(f"[LAUNCH] G{depth} shard {shard_id} -> GPU {gpu}; log={log}", flush=True)

        failed = []
        for p, fh, log in procs:
            rc = p.wait()
            fh.close()
            if rc != 0:
                failed.append((rc, log))
        if failed:
            for rc, log in failed:
                print(f"FAILED rc={rc}: {log}", file=sys.stderr)
            raise RuntimeError(f"G{depth} failed on {len(failed)} shard(s)")

        merge_stage(out, depth, args.expected_n, 4)
        print(f"========== G{depth} COMPLETE ==========", flush=True)

    frames = []
    for d in range(1, 5):
        f = stage_manifest(out, d)
        if f.exists():
            frames.append(pd.read_csv(f))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out / "manifest_g1_g4.csv", index=False)

    print("\nDONE. Status:", flush=True)
    for d in range(1, 5):
        f = stage_manifest(out, d)
        if f.exists():
            x = pd.read_csv(f)
            print(f"G{d}: {len(x)} rows -> {f}", flush=True)
        else:
            print(f"G{d}: pending", flush=True)

def status(args):
    root, out = Path(args.data_root), Path(args.out_root)
    df = load_test(root, args.expected_n)
    print(f"G0: {len(df)} existing test pairs")
    for d in range(1, 5):
        f = stage_manifest(out, d)
        if not f.exists():
            print(f"G{d}: pending")
        else:
            x = pd.read_csv(f)
            valid = 0
            if "clean_path" in x and "wm_path" in x:
                valid = sum(Path(c).exists() and Path(w).exists()
                            for c, w in zip(x.clean_path.astype(str), x.wm_path.astype(str)))
            print(f"G{d}: rows={len(x)} valid_pairs={valid} {f}")

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=ROOT_DEFAULT)
    p.add_argument("--out-root", type=Path,
                   default=ROOT_DEFAULT / "lineage_8bit_test1000")
    p.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    p.add_argument("--gpus", default="4,5,6,7")
    p.add_argument("--expected-n", type=int, default=1000)
    p.add_argument("--strength", type=float, default=0.50)
    p.add_argument("--base-seed", type=int, default=20260903)
    p.add_argument("--start-depth", type=int, default=1)
    p.add_argument("--stop-depth", type=int, default=4)
    p.add_argument("--stage", choices=["lineage", "status"], default="lineage")

    # internal worker args
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--depth", type=int, choices=[1,2,3,4], help=argparse.SUPPRESS)
    p.add_argument("--shard-id", type=int, help=argparse.SUPPRESS)
    p.add_argument("--num-shards", type=int, default=4, help=argparse.SUPPRESS)
    p.add_argument("--physical-gpu", type=int, help=argparse.SUPPRESS)
    return p.parse_args()

if __name__ == "__main__":
    a = cli()
    if a.worker:
        worker(a)
    elif a.stage == "status":
        status(a)
    else:
        launch(a)
