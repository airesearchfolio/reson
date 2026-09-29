#!/usr/bin/env python3
"""
RESON SD2.1 8-bit COCO experiment using EXACTLY the same 10K COCO captions,
sample_ids, generation seeds, and clean images as the current canonical COCO run.

IMPORTANT:
- Reuses existing canonical CLEAN images; does not regenerate them.
- Generates only 8-bit RESON images.
- Same COCO caption selection:
    captions_val2014.json
    selection_seed=20260920
    generation_seed = 20260903 + sample_id
- Same SD2.1 / 512 / 50 steps / CFG 7.5 / alpha=.155 / beta=.40.
- 8-bit messages use sample_id % 256; across 10K samples all 256 messages are represented nearly uniformly.
- Aggregate carrier is normalized so 8 bits do not multiply watermark energy.
- Resume-safe and shardable.
- Start with --limit 100 per worker (200 total) for visual gate.
"""
import argparse, csv, hashlib, json, os, sys
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from tqdm.auto import tqdm


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
        if cap:
            rows.append({
                "annotation_id": int(a["id"]),
                "image_id": iid,
                "file_name": images[iid]["file_name"],
                "prompt": cap,
            })
    return rows


def select_rows(rows, n, selection_seed):
    keyed = []
    for r in rows:
        k = stable_int("reson-coco-final", selection_seed, r["annotation_id"], mod=2**63-1)
        keyed.append((k, r["annotation_id"], r))
    keyed.sort(key=lambda x: (x[0], x[1]))
    if len(keyed) < n:
        raise RuntimeError(f"Only {len(keyed)} captions; requested {n}")
    return [x[2] for x in keyed[:n]]


def bits8(sid):
    m = int(sid) % 256
    return [(m >> j) & 1 for j in range(8)]


def extract_canonical_carrier(wm, seed, device, alpha):
    z = wm.null_latent(seed=seed, device=device).float()
    zp = wm.inject(bit=1, seed=seed, device=device, alpha=alpha).float()
    c = (np.sqrt(1.0 + alpha * alpha) * zp - z) / alpha
    return c


def make_eight_carriers(c):
    # Deterministic transforms of the SAME canonical structured carrier.
    # Gram-Schmidt makes eight orthogonal payload directions.
    # Each carrier is rescaled to the canonical carrier's L2 norm.
    specs = [
        ("id",       0,  0),
        ("roll",    11, 17),
        ("flipx",   23,  7),
        ("flipy",    5, 29),
        ("roll",    31, 13),
        ("flipx",   19, 37),
        ("flipy",   41,  3),
        ("flipxy",  17, 43),
    ]
    cand=[]
    for kind,dy,dx in specs:
        x=torch.roll(c, shifts=(dy,dx), dims=(-2,-1))
        if kind=="flipx":
            x=torch.flip(x,dims=(-1,))
        elif kind=="flipy":
            x=torch.flip(x,dims=(-2,))
        elif kind=="flipxy":
            x=torch.flip(x,dims=(-2,-1))
        cand.append(x)

    target=torch.linalg.vector_norm(c.reshape(-1))
    qs=[]
    for x in cand:
        v=x.reshape(-1).clone()
        for q in qs:
            v=v-torch.dot(v,q)*q
        nv=torch.linalg.vector_norm(v)
        if nv < 1e-8:
            raise RuntimeError("Degenerate 8-carrier basis")
        qs.append(v/nv)
    return [q.reshape_as(c)*target for q in qs]


def inject8(z, carriers, bits, alpha):
    # 1/sqrt(8) superposition, then exact L2 renormalization.
    # Thus total watermark energy matches the canonical 1-bit/8-bit construction.
    signs=torch.tensor([2*b-1 for b in bits],device=z.device,dtype=z.dtype)
    u=sum(signs[j]*carriers[j].to(z.device,z.dtype) for j in range(8))/np.sqrt(8.0)
    target=torch.linalg.vector_norm(carriers[0].to(z.device,z.dtype).reshape(-1))
    u=u*(target/torch.linalg.vector_norm(u.reshape(-1)))
    return (z+alpha*u)/np.sqrt(1.0+alpha*alpha)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="reson")
    ap.add_argument("--captions", default="workspace/paper_experiments/exp_16_coco_fid/annotations/captions_val2014.json")
    ap.add_argument("--canonical-coco-root", default="workspace/paper_experiments/exp_16_coco_fid/sd21_canonical_final_n10000")
    ap.add_argument("--out-root", default="workspace/paper_experiments/exp_16_coco_fid/sd21_8bit_alpha0155_beta040_n10000")
    ap.add_argument("--model-id", default="sd2-community/stable-diffusion-2-1-base")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--selection-seed", type=int, default=20260920)
    ap.add_argument("--generation-seed-base", type=int, default=20260903)
    ap.add_argument("--alpha", type=float, default=.155)
    ap.add_argument("--beta", type=float, default=.40)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--worker-id", type=int, choices=[0,1], required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit assigned samples. Use 100 per worker for 200-pair visual gate.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.project_root)
    from config import Config
    from watermark import Watermark

    canonical = Path(args.canonical_coco_root)
    clean_dir = canonical / "clean"
    out = Path(args.out_root)
    wm8_dir = out / "reson_8bit"
    shard_dir = out / "shards"
    wm8_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    rows = select_rows(load_coco_captions(args.captions), args.n, args.selection_seed)
    selected = [(sid, r) for sid, r in enumerate(rows) if sid % 2 == args.worker_id]
    if args.limit is not None:
        selected = selected[:args.limit]

    cfg = Config()
    cfg.watermark.noise_mix_alpha = args.alpha
    cfg.watermark.carrier_blend = args.beta
    wm = Watermark(cfg).build()

    # Match current canonical COCO generation: FP32 SD2.1.
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    device = pipe._execution_device

    c = extract_canonical_carrier(wm, 20260920, device, args.alpha)
    carriers = make_eight_carriers(c)
    gram = np.array([
        [float(torch.dot(x.reshape(-1), y.reshape(-1)) /
               (torch.linalg.vector_norm(x.reshape(-1))*torch.linalg.vector_norm(y.reshape(-1))))
         for y in carriers] for x in carriers
    ])
    print("8-carrier cosine Gram matrix:")
    print(np.round(gram, 6), flush=True)

    mfpath = shard_dir / f"manifest_worker_{args.worker_id:02d}.csv"
    fields = [
        "sample_id","annotation_id","image_id","file_name","prompt","generation_seed",
        "split","message_int","bit0","bit1","bit2","bit3","bit4","bit5","bit6","bit7",
        "clean_path","wm8_path","alpha","beta","payload_bits","worker_id"
    ]
    existing = set()
    if mfpath.exists() and not args.overwrite:
        with open(mfpath, newline="") as f:
            existing = {int(r["sample_id"]) for r in csv.DictReader(f)}

    pending = []
    missing_clean = []
    for sid, r in selected:
        cp = clean_dir / f"{sid:05d}.png"
        wp = wm8_dir / f"{sid:05d}_wm.png"
        if not cp.exists():
            missing_clean.append(str(cp))
        if args.overwrite or not wp.exists():
            pending.append((sid, r))
    if missing_clean:
        raise RuntimeError(
            f"{len(missing_clean)} canonical clean images are not ready yet. "
            f"First missing: {missing_clean[0]}\n"
            "Wait for the current canonical COCO generation to create those clean images, "
            "or choose sample IDs already complete."
        )

    mode = "a" if mfpath.exists() and not args.overwrite else "w"
    with open(mfpath, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if mode == "w":
            w.writeheader()

        print(f"[worker {args.worker_id}] assigned={len(selected)} "
              f"already_complete={len(selected)-len(pending)} pending={len(pending)}", flush=True)

        for sid, r in tqdm(pending, desc=f"8bit worker {args.worker_id}", unit="wm", dynamic_ncols=True):
            seed = args.generation_seed_base + sid
            b = bits8(sid)
            z = wm.null_latent(seed=seed, device=device).float()
            zw = inject8(z, carriers, b, args.alpha)
            zw = zw.to(device=device, dtype=next(pipe.unet.parameters()).dtype)

            common = dict(
                prompt=r["prompt"], negative_prompt="",
                num_inference_steps=args.steps, guidance_scale=args.guidance,
                height=512, width=512
            )
            wp = wm8_dir / f"{sid:05d}_wm.png"
            with torch.inference_mode():
                im = pipe(latents=zw, **common).images[0]
            tmp = wp.with_name(wp.stem + ".tmp.png")
            im.save(tmp)
            os.replace(tmp, wp)

            split = "train" if sid < 8000 else ("val" if sid < 9000 else "test")
            w.writerow({
                "sample_id":sid,"annotation_id":r["annotation_id"],"image_id":r["image_id"],
                "file_name":r["file_name"],"prompt":r["prompt"],"generation_seed":seed,
                "split":split,"message_int":sid%256,
                "bit0":b[0],"bit1":b[1],"bit2":b[2],"bit3":b[3],
                "bit4":b[4],"bit5":b[5],"bit6":b[6],"bit7":b[7],
                "clean_path":str(clean_dir/f"{sid:05d}.png"),"wm8_path":str(wp),
                "alpha":args.alpha,"beta":args.beta,"payload_bits":8,"worker_id":args.worker_id
            })
            f.flush()

    meta = {
        "same_caption_selection_as_canonical_coco": True,
        "selection_seed": args.selection_seed,
        "generation_seed_rule": f"{args.generation_seed_base} + sample_id",
        "canonical_clean_root": str(clean_dir),
        "model": args.model_id, "alpha":args.alpha, "beta":args.beta,
        "steps":args.steps, "guidance":args.guidance, "payload_bits":8,
        "carrier_cosine_gram":gram.tolist(),
        "split_rule":"sample_id 0:7999 train, 8000:8999 val, 9000:9999 test",
        "message_rule":"sample_id mod 256"
    }
    (out/"experiment_config.json").write_text(json.dumps(meta, indent=2))
    print("DONE:", mfpath, flush=True)


if __name__ == "__main__":
    main()
