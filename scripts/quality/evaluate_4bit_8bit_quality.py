#!/usr/bin/env python3
"""
RESON 4-bit / 8-bit G0 quality evaluator.

Matches the final 1-bit V3 quality protocol:
- COCO 2014 val real reference: datasets/coco2014/val2014
- FID features: pytorch-fid InceptionV3 pool3=2048
- native-size COCO reference evaluated with batch_size=1
- OpenCLIP ViT-B-32 pretrained=openai
- evaluates all 10,000 generated images
- no regeneration

Clean FID/CLIP are reused from the final 1-bit JSON because 4-bit and 8-bit
manifests point to the same canonical clean images.
"""

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import linalg
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from pytorch_fid.inception import InceptionV3
import open_clip

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXP_ROOT = Path("workspace/paper_experiments/exp_16_coco_fid")
ONEBIT_JSON = EXP_ROOT / "sd21_canonical_final_n10000" / "final_quality_results_v3.json"

CONFIGS = {
    4: {
        "root": EXP_ROOT / "sd21_4bit_alpha0155_beta040_n10000",
        "wm_col": "wm4_path",
    },
    8: {
        "root": EXP_ROOT / "sd21_8bit_alpha0155_beta040_n10000",
        "wm_col": "wm8_path",
    },
}


class ImagePathDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.tf = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


def collate_images(batch):
    # Generated images are expected to have the same dimensions.
    return torch.stack(batch, 0)


def inception_stats(paths, device, batch_size, cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        z = np.load(cache_path)
        print(f"Using cached stats: {cache_path}", flush=True)
        return z["mu"], z["sigma"], int(z["n"])

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device).eval()

    ds = ImagePathDataset(paths)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=True, collate_fn=collate_images
    )

    feats = []
    with torch.inference_mode():
        for x in tqdm(loader, desc=f"Inception n={len(ds)}"):
            x = x.to(device, non_blocking=True)
            pred = model(x)[0]
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = torch.nn.functional.adaptive_avg_pool2d(pred, (1, 1))
            feats.append(pred.squeeze(3).squeeze(2).cpu().numpy())

    act = np.concatenate(feats, axis=0)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, mu=mu, sigma=sigma, n=len(paths))
    return mu, sigma, len(paths)


def fid(mu1, s1, mu2, s2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        offset = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + offset).dot(s2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def validate_manifest(bits, root, wm_col, expected_n):
    mf = root / "manifest_final_10000.csv"
    if not mf.exists():
        raise FileNotFoundError(mf)
    df = pd.read_csv(mf).sort_values("sample_id").reset_index(drop=True)

    required = {"sample_id", "prompt", "clean_path", wm_col}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{bits}-bit missing manifest columns: {sorted(missing)}")
    if len(df) != expected_n or df.sample_id.nunique() != expected_n:
        raise RuntimeError(
            f"{bits}-bit expected {expected_n} unique rows; "
            f"got rows={len(df)}, unique={df.sample_id.nunique()}"
        )

    for col in ["clean_path", wm_col]:
        bad = [p for p in df[col].astype(str) if not Path(p).exists()]
        if bad:
            raise RuntimeError(
                f"{bits}-bit {col}: {len(bad)} missing files; first={bad[0]}"
            )
    return df


def clip_eval(df, image_col, device, batch_size):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    all_scores = []
    ids = []

    with torch.inference_mode():
        for start in tqdm(range(0, len(df), batch_size), desc=f"CLIP {image_col}"):
            b = df.iloc[start:start+batch_size]
            ims = []
            for p in b[image_col].astype(str):
                with Image.open(p) as im:
                    ims.append(preprocess(im.convert("RGB")))
            ims = torch.stack(ims).to(device)
            txt = tokenizer(b["prompt"].astype(str).tolist()).to(device)

            fi = model.encode_image(ims)
            ft = model.encode_text(txt)
            fi = fi / fi.norm(dim=-1, keepdim=True)
            ft = ft / ft.norm(dim=-1, keepdim=True)
            score = (fi * ft).sum(dim=-1)

            all_scores.extend(score.cpu().numpy().astype(float).tolist())
            ids.extend(b.sample_id.astype(int).tolist())

    return pd.DataFrame({"sample_id": ids, "clip_score": all_scores})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir",
                    default="datasets/coco2014/val2014")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fid-batch-size", type=int, default=64)
    ap.add_argument("--clip-batch-size", type=int, default=128)
    ap.add_argument("--expected-n", type=int, default=10000)
    ap.add_argument("--expected-real-n", type=int, default=40504)
    ap.add_argument("--skip-fid", action="store_true")
    ap.add_argument("--skip-clip", action="store_true")
    ap.add_argument("--out-dir", type=Path,
                    default=EXP_ROOT / "payload_quality_4bit_8bit")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    real_dir = Path(a.real_dir)
    if not real_dir.exists():
        raise FileNotFoundError(f"COCO real directory not found: {real_dir}")

    real_paths = sorted(p for p in real_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in IMG_EXTS)
    print(f"Real COCO reference images: {len(real_paths)}", flush=True)
    if len(real_paths) != a.expected_real_n:
        raise RuntimeError(
            f"Expected {a.expected_real_n} real COCO images, got {len(real_paths)}"
        )

    if not ONEBIT_JSON.exists():
        raise FileNotFoundError(f"Missing final 1-bit quality JSON: {ONEBIT_JSON}")
    one = json.loads(ONEBIT_JSON.read_text())

    dfs = {}
    for bits, cfg in CONFIGS.items():
        dfs[bits] = validate_manifest(
            bits, cfg["root"], cfg["wm_col"], a.expected_n
        )
        print(f"{bits}-bit manifest OK: {len(dfs[bits])}", flush=True)

    # The payload comparison is only apples-to-apples if the same source set is used.
    cols = ["sample_id", "prompt", "clean_path"]
    if not dfs[4][cols].equals(dfs[8][cols]):
        raise RuntimeError("4-bit and 8-bit manifests do not use identical source examples.")
    print("4-bit and 8-bit source sets match exactly.", flush=True)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    cache = a.out_dir / "quality_cache"

    out = {
        "reference_images": len(real_paths),
        "generated_images_per_payload": a.expected_n,
        "fid_protocol": one.get(
            "fid_protocol",
            "pytorch-fid InceptionV3 pool3=2048; native-size COCO real; real batch=1"
        ),
        "clip_model": "OpenCLIP ViT-B-32 pretrained=openai",
        "clean_reused_from_final_1bit": {
            "fid": one["fid_coco_real_vs_clean"],
            "clip_mean": one["clip"]["clean_mean"],
        },
        "one_bit_reused": {
            "fid": one["fid_coco_real_vs_reson_1bit"],
            "clip_mean": one["clip"]["reson_mean"],
        },
        "payloads": {}
    }

    rmu = rsig = None
    if not a.skip_fid:
        print("\n=== FID statistics: COCO real reference ===", flush=True)
        # Match V3 exactly: native-size real images use batch_size=1.
        rmu, rsig, rn = inception_stats(
            real_paths, a.device, 1, cache / "coco_val2014_real_pool3.npz"
        )
        if rn != a.expected_real_n:
            raise RuntimeError(f"Real stats n={rn}, expected {a.expected_real_n}")

    for bits in (4, 8):
        cfg = CONFIGS[bits]
        df = dfs[bits]
        entry = {"n": len(df)}

        print("\n" + "="*72, flush=True)
        print(f"{bits}-BIT QUALITY", flush=True)
        print("="*72, flush=True)

        if not a.skip_fid:
            wm_paths = [Path(p) for p in df[cfg["wm_col"]].astype(str)]
            print(f"\n=== FID statistics: RESON {bits}-bit ===", flush=True)
            wmu, wsig, wn = inception_stats(
                wm_paths, a.device, a.fid_batch_size,
                cache / f"reson_{bits}bit_pool3.npz"
            )
            fwm = fid(rmu, rsig, wmu, wsig)
            entry["fid_coco_real_vs_wm"] = fwm
            entry["delta_fid_vs_clean"] = (
                fwm - float(one["fid_coco_real_vs_clean"])
            )
            print(f"FID COCO real vs {bits}-bit = {fwm:.8f}", flush=True)

            # Save immediately so a later CLIP failure cannot lose FID.
            out["payloads"][str(bits)] = entry
            (a.out_dir / "payload_quality_4bit_8bit.json").write_text(
                json.dumps(out, indent=2)
            )

        if not a.skip_clip:
            print(f"\n=== CLIP: RESON {bits}-bit ===", flush=True)
            per = clip_eval(df, cfg["wm_col"], a.device, a.clip_batch_size)
            per.to_csv(a.out_dir / f"{bits}bit_clip_per_image.csv", index=False)
            entry["clip_mean"] = float(per.clip_score.mean())
            entry["clip_std"] = float(per.clip_score.std(ddof=1))
            entry["delta_clip_vs_clean"] = (
                entry["clip_mean"] - float(one["clip"]["clean_mean"])
            )
            print(f"CLIP {bits}-bit = {entry['clip_mean']:.8f}", flush=True)

        out["payloads"][str(bits)] = entry
        (a.out_dir / "payload_quality_4bit_8bit.json").write_text(
            json.dumps(out, indent=2)
        )

    rows = [{
        "bits": 1,
        "n": int(one["reson_images"]),
        "fid": float(one["fid_coco_real_vs_reson_1bit"]),
        "delta_fid_vs_clean": float(one["delta_fid_reson_minus_clean"]),
        "clip": float(one["clip"]["reson_mean"]),
        "delta_clip_vs_clean": float(one["clip"]["delta_reson_minus_clean"]),
    }]
    for bits in (4, 8):
        e = out["payloads"][str(bits)]
        rows.append({
            "bits": bits,
            "n": e["n"],
            "fid": e.get("fid_coco_real_vs_wm", np.nan),
            "delta_fid_vs_clean": e.get("delta_fid_vs_clean", np.nan),
            "clip": e.get("clip_mean", np.nan),
            "delta_clip_vs_clean": e.get("delta_clip_vs_clean", np.nan),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(a.out_dir / "payload_quality_summary.csv", index=False)

    print("\n=== FINAL SUMMARY ===", flush=True)
    print(f"Clean FID  : {one['fid_coco_real_vs_clean']:.8f}", flush=True)
    print(f"Clean CLIP : {one['clip']['clean_mean']:.8f}", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nSaved to: {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
