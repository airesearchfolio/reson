#!/usr/bin/env python3
"""Launch final N=10K COCO RESON generation across selected GPUs."""
import argparse, os, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="5,7")
    ap.add_argument("--worker", default=str(Path(__file__).with_name("generate_coco_quality.py")))
    ap.add_argument("--out-root", default="workspace/paper_experiments/exp_16_coco_fid/sd21_canonical_final_n10000")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--extra", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    out = Path(args.out_root)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    procs = []
    for sid, gpu in enumerate(gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log = open(logs / f"gpu{gpu}_shard{sid}.log", "a")
        cmd = [
            sys.executable, args.worker,
            "--out-root", args.out_root,
            "--n", str(args.n),
            "--shard-id", str(sid),
            "--num-shards", str(len(gpus)),
        ] + (args.extra or [])
        print("START:", " ".join(cmd), "GPU", gpu)
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((p, log, gpu))

    rc = 0
    for p, log, gpu in procs:
        r = p.wait()
        log.close()
        print(f"GPU {gpu}: rc={r}")
        rc = max(rc, r)
    raise SystemExit(rc)

if __name__ == "__main__":
    main()
