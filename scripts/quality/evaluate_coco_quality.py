#!/usr/bin/env python3
"""
RESON COCO quality evaluator V3.

Fixes:
- uses open_clip instead of the incompatible `clip` package
- caches Inception pool3 statistics to .npz immediately
- prints/saves FID before CLIP evaluation
- reuses COCO-real and clean statistics on future runs
- native-size COCO real images use batch_size=1

Expected root:
  .../sd21_canonical_final_n10000/
    clean/
    reson/
    manifest_final_10000.csv
"""

import argparse, csv, json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from pytorch_fid.inception import InceptionV3
import open_clip


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class ImagePathDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.tf = transforms.ToTensor()
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def inception_stats(paths, device, batch_size, cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        x = np.load(cache_path)
        print(f"Using cached stats: {cache_path}", flush=True)
        return x["mu"], x["sigma"], int(x["n"])

    block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block]).to(device).eval()
    ds = ImagePathDataset(paths)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=0, pin_memory=True)

    n = 0
    sum_x = np.zeros(2048, dtype=np.float64)
    sum_xx = np.zeros((2048, 2048), dtype=np.float64)

    for batch in tqdm(dl):
        batch = batch.to(device, non_blocking=True)
        pred = model(batch)[0]
        if pred.shape[2:] != (1, 1):
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, (1, 1))
        a = pred.squeeze(3).squeeze(2).cpu().numpy().astype(np.float64)
        n += a.shape[0]
        sum_x += a.sum(axis=0)
        sum_xx += a.T @ a

    mu = sum_x / n
    sigma = (sum_xx - n * np.outer(mu, mu)) / (n - 1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, mu=mu, sigma=sigma, n=np.int64(n))
    print(f"Saved stats immediately: {cache_path}", flush=True)
    return mu, sigma, n


def fid(mu1, s1, mu2, s2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2*np.trace(covmean))


def read_manifest(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [int(float(r["sample_id"])) for r in rows]
    assert len(rows) == 10000 and len(set(ids)) == 10000
    return rows


def resolve_prompt(r):
    for k in ("prompt", "caption", "text"):
        if k in r and r[k]:
            return r[k]
    raise KeyError("No prompt/caption/text column found in manifest")


def image_for_id(folder, sid, wm=False):
    # Prefer canonical names but tolerate zero-padding differences.
    candidates = [
        folder / (f"{sid:05d}_wm.png" if wm else f"{sid:05d}.png"),
        folder / (f"{sid}_wm.png" if wm else f"{sid}.png"),
    ]
    for p in candidates:
        if p.exists(): return p
    matches = list(folder.glob(f"*{sid:05d}*")) + list(folder.glob(f"*{sid}*"))
    matches = [p for p in matches if p.suffix.lower() in IMG_EXTS]
    if len(matches) == 1: return matches[0]
    raise FileNotFoundError(f"Cannot uniquely resolve sample {sid} in {folder}")


@torch.no_grad()
def clip_eval(rows, clean_dir, wm_dir, device, batch_size=128):
    # OpenCLIP equivalent architecture to OpenAI CLIP ViT-B/32.
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    clean_scores, wm_scores = [], []
    for start in tqdm(range(0, len(rows), batch_size), desc="CLIP batches"):
        rr = rows[start:start+batch_size]
        prompts = [resolve_prompt(r) for r in rr]
        ids = [int(float(r["sample_id"])) for r in rr]
        ci = torch.stack([preprocess(Image.open(image_for_id(clean_dir,s)).convert("RGB")) for s in ids]).to(device)
        wi = torch.stack([preprocess(Image.open(image_for_id(wm_dir,s,True)).convert("RGB")) for s in ids]).to(device)
        tx = tokenizer(prompts).to(device)

        tf = model.encode_text(tx); tf = tf / tf.norm(dim=-1, keepdim=True)
        cf = model.encode_image(ci); cf = cf / cf.norm(dim=-1, keepdim=True)
        wf = model.encode_image(wi); wf = wf / wf.norm(dim=-1, keepdim=True)
        clean_scores.extend((cf*tf).sum(-1).cpu().numpy().tolist())
        wm_scores.extend((wf*tf).sum(-1).cpu().numpy().tolist())

    c=np.asarray(clean_scores); w=np.asarray(wm_scores)
    return {
        "model": "OpenCLIP ViT-B-32 pretrained=openai",
        "clean_mean": float(c.mean()),
        "reson_mean": float(w.mean()),
        "delta_reson_minus_clean": float((w-c).mean()),
        "reson_higher_fraction": float((w>c).mean()),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--real-dir", default="datasets/coco2014/val2014")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gen-batch-size", type=int, default=64)
    ap.add_argument("--clip-batch-size", type=int, default=128)
    ap.add_argument("--skip-clip", action="store_true")
    args=ap.parse_args()

    root=Path(args.root)
    manifest=root/"manifest_final_10000.csv"
    rows=read_manifest(manifest)
    clean=root/"clean"; wm=root/"reson"
    real=Path(args.real_dir)

    real_paths=sorted(p for p in real.iterdir() if p.suffix.lower() in IMG_EXTS)
    clean_paths=sorted(p for p in clean.iterdir() if p.suffix.lower() in IMG_EXTS)
    wm_paths=sorted(p for p in wm.iterdir() if p.suffix.lower() in IMG_EXTS)
    assert len(clean_paths)==10000 and len(wm_paths)==10000
    print(f"Validated generated files: clean={len(clean_paths)}, RESON={len(wm_paths)}")
    print(f"Real COCO reference images: {len(real_paths)}")

    cache=root/"quality_cache"
    print("\n=== FID statistics: COCO real reference ===")
    rmu,rsig,rn=inception_stats(real_paths,args.device,1,cache/"coco_val2014_real_pool3.npz")
    print("\n=== FID statistics: Clean SD2.1 ===")
    cmu,csig,cn=inception_stats(clean_paths,args.device,args.gen_batch_size,cache/"clean_pool3.npz")
    print("\n=== FID statistics: RESON 1-bit ===")
    wmu,wsig,wn=inception_stats(wm_paths,args.device,args.gen_batch_size,cache/"reson_1bit_pool3.npz")

    fclean=fid(rmu,rsig,cmu,csig)
    fwm=fid(rmu,rsig,wmu,wsig)
    out={
        "reference_images":rn,"clean_images":cn,"reson_images":wn,
        "fid_coco_real_vs_clean":fclean,
        "fid_coco_real_vs_reson_1bit":fwm,
        "delta_fid_reson_minus_clean":fwm-fclean,
        "fid_protocol":"pytorch-fid InceptionV3 pool3=2048; native-size COCO real; real batch=1"
    }
    # Save FID BEFORE CLIP.
    result_path=root/"final_quality_results_v3.json"
    result_path.write_text(json.dumps(out,indent=2))
    print("\n=== FID RESULTS (already saved) ===")
    print(json.dumps(out,indent=2),flush=True)
    print("Saved:",result_path,flush=True)

    if not args.skip_clip:
        print("\n=== CLIP ===",flush=True)
        out["clip"]=clip_eval(rows,clean,wm,args.device,args.clip_batch_size)
        result_path.write_text(json.dumps(out,indent=2))
        print(json.dumps(out["clip"],indent=2),flush=True)
        print("Updated:",result_path,flush=True)


if __name__=="__main__":
    main()
