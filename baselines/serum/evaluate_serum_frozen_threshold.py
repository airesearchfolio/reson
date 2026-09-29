#!/usr/bin/env python3
"""
SERUM canonical N=300 lineage evaluation with ONE frozen threshold (G0-G4).

Why this exists
---------------
SERUM's native evaluator (src.evaluation.eval) picks a new threshold for every
call ("... TPR @ 1% FPR ... Thr: x") using the clean images it is given. Called
once per depth, that is a per-depth oracle threshold chosen on the test images.
RESON uses a single threshold frozen on validation data. This script scores
images directly and applies one threshold, calibrated once on held-out clean
images, to every depth.

Outputs (in --out-dir)
  serum_scores_g0_g4.csv          per-image scores (depth, sample_id, label, score)
  serum_frozen_summary.csv        TPR / realized FPR (95% Wilson CI), AUC, oracle diagnostic
  calibration.json                threshold, calibration set, n

Correctness check (strongly recommended)
  --native-log-dir points to the native evaluator logs (G0.log ...). The script
  compares its AUC per depth with the native AUC. If they differ by more than
  ~0.005, the preprocessing (VAE latent mode / scaling) does not match SERUM's
  and the scores must not be used. Adjust --latent-mode / --scale and rerun.
"""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ROOT = Path("workspace/fair_sd21_main_table/serum_canonical_n300")
SERUM_ROOT = Path("third_party/SERUM")
CKPT = SERUM_ROOT / "results/SERUM_stage15_full/checkpoints/checkpoint_epoch_50.pt"
G0_CLEAN_POOL = SERUM_ROOT / "results/serum_sd_120/gen_images/clean"
VAE_ID = "sd2-community/stable-diffusion-2-1-base"
DEPTH_NAMES = {1: "flux", 2: "realvis", 3: "dreamshaper", 4: "flux"}
TARGET_FPR = 0.01


# --------------------------------------------------------------------------- #
# SERUM detector (architecture from the SERUM paper, Fig. 5 / App. J)
# --------------------------------------------------------------------------- #
class SerumDetector(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [4, 32, 64, 128, 256]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3, 1, 1), nn.BatchNorm2d(ch[i + 1]),
                       nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(4096, 512), nn.ReLU(),
                                  nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def extract_state_dict(obj):
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    for k in ["score_model_state_dict", "model_state_dict", "state_dict", "model", "detector", "net"]:
        if isinstance(obj, dict) and k in obj:
            return extract_state_dict(obj[k])
    if isinstance(obj, dict) and all(torch.is_tensor(v) for v in obj.values()):
        return obj
    raise RuntimeError(f"Unrecognised checkpoint format; top-level keys: {list(obj)[:20]}")


def load_detector(ckpt_path, device):
    sd = extract_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model = SerumDetector()
    try:
        model.load_state_dict(sd, strict=True)
        print("Checkpoint loaded by name (strict).")
    except RuntimeError:
        # Parameter names differ from ours: map tensors in order, requiring exact shape matches.
        target = model.state_dict()
        src = [(k, v) for k, v in sd.items() if not k.endswith("num_batches_tracked")]
        dst = [k for k in target if not k.endswith("num_batches_tracked")]
        if len(src) != len(dst):
            raise RuntimeError(f"Tensor count mismatch: checkpoint {len(src)} vs model {len(dst)}. "
                               "Use SERUM's own model class instead (see --help).")
        new = {}
        for (sk, sv), dk in zip(src, dst):
            if tuple(sv.shape) != tuple(target[dk].shape):
                raise RuntimeError(f"Shape mismatch {sk}{tuple(sv.shape)} -> {dk}{tuple(target[dk].shape)}")
            new[dk] = sv
        model.load_state_dict(new, strict=False)
        print(f"Checkpoint loaded by order ({len(new)} tensors, all shapes matched).")
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def score_paths(paths, vae, det, device, bs, latent_mode, scale):
    out = []
    for i in tqdm(range(0, len(paths), bs), leave=False):
        ims = []
        for p in paths[i:i + bs]:
            with Image.open(p) as im:
                im = im.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
            ims.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 127.5 - 1.0)
        x = torch.stack(ims).to(device, dtype=vae.dtype)
        dist = vae.encode(x).latent_dist
        z = (dist.mean if latent_mode == "mean" else dist.sample()) * scale
        out.append(det(z.float()).float().cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def path_cols(df):
    wc = next((c for c in ["wm_image_path", "watermarked_image_path", "wm_path"] if c in df), None)
    cc = next((c for c in ["alpha0_image_path", "unwatermarked_image_path", "clean_image_path", "clean_path"] if c in df), None)
    if not wc or not cc:
        raise RuntimeError(f"Cannot identify path columns: {list(df.columns)}")
    return cc, wc


def depth_manifest(root, d):
    cands = [root / "generation_0.csv", root / "manifests" / "generation_0.csv"] if d == 0 else \
            [root / f"generation_{d}.csv", root / "manifests" / f"generation_{d}_{DEPTH_NAMES[d]}.csv"]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"No manifest for G{d} under {root}")


def load_depth(root, d, n):
    df = pd.read_csv(depth_manifest(root, d)).sort_values("sample_id").head(n).reset_index(drop=True)
    if len(df) != n:
        raise RuntimeError(f"G{d}: {len(df)} rows, need {n}")
    cc, wc = path_cols(df)
    for c in (cc, wc):
        missing = [p for p in df[c] if not Path(str(p)).exists()]
        if missing:
            raise FileNotFoundError(f"G{d}: {len(missing)} missing files, e.g. {missing[0]}")
    return df, cc, wc


def calibration_paths(args, test_g0_clean):
    """Clean images disjoint from the 300 test images, used only to set the threshold."""
    if args.calib_dir:
        pool = sorted(p for p in Path(args.calib_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    else:
        pool = sorted(p for p in G0_CLEAN_POOL.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    test = {Path(str(p)).resolve() for p in test_g0_clean}
    pool = [p for p in pool if p.resolve() not in test]
    if len(pool) < args.min_calib:
        raise RuntimeError(f"Only {len(pool)} calibration images disjoint from the test set; need >= {args.min_calib}. "
                           "Generate clean SERUM-pipeline G0 images from held-out (validation) prompts and pass --calib-dir.")
    return pool[: args.max_calib]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (100 * (c - h), 100 * (c + h))


def oracle_tpr(clean, wm, target=TARGET_FPR):
    """Diagnostic only: best test-set threshold with FPR <= target."""
    best = (0.0, float("nan"), 0.0)
    for t in np.unique(np.concatenate([clean, wm])):
        fpr = float((clean > t).mean())
        if fpr <= target:
            tpr = float((wm > t).mean())
            if tpr > best[0]:
                best = (tpr, t, fpr)
    return best


def native_auc(log_dir, d):
    p = Path(log_dir) / f"G{d}.log"
    if not p.exists():
        return None
    m = re.search(r"ROC AUC: ([0-9.]+)", p.read_text())
    return float(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT, help="folder with generation_*.csv manifests")
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--vae", default=VAE_ID)
    ap.add_argument("--latent-mode", choices=["mean", "sample"], default="mean")
    ap.add_argument("--scale", type=float, default=0.18215, help="VAE latent scaling factor used by SERUM")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--calib-dir", type=Path, default=None,
                    help="clean SERUM-pipeline images NOT in the test set (default: unused files in the G0 clean pool)")
    ap.add_argument("--min-calib", type=int, default=300)
    ap.add_argument("--max-calib", type=int, default=1000)
    ap.add_argument("--native-log-dir", type=Path, default=ROOT / "native_evaluation")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "frozen_threshold_evaluation")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.manual_seed(0)

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae", torch_dtype=torch.float16).to(device).eval()
    det = load_detector(args.checkpoint, device)

    # ---- load all depths first (fail early on missing files) ----
    data = {d: load_depth(args.root, d, args.n) for d in range(5)}
    ids0 = data[0][0].sample_id.tolist()
    for d in range(1, 5):
        if data[d][0].sample_id.tolist() != ids0:
            raise RuntimeError(f"G{d} sample_ids differ from G0; lineage rows are not aligned")

    # ---- calibration: one threshold, frozen for all depths ----
    g0_df, g0_cc, _ = data[0]
    calib = calibration_paths(args, g0_df[g0_cc])
    cal_scores = score_paths([str(p) for p in calib], vae, det, device, args.batch_size, args.latent_mode, args.scale)
    thr = float(np.quantile(cal_scores, 1 - TARGET_FPR, method="higher"))  # detect if score > thr
    cal_fpr = float((cal_scores > thr).mean())
    json.dump(dict(threshold=thr, rule="watermarked if score > threshold", target_fpr=TARGET_FPR,
                   calibration_fpr=cal_fpr, n_calibration=len(calib),
                   calibration_source=str(args.calib_dir or G0_CLEAN_POOL),
                   latent_mode=args.latent_mode, scale=args.scale, checkpoint=str(args.checkpoint)),
              open(args.out_dir / "calibration.json", "w"), indent=2)
    print(f"Frozen threshold {thr:.6f} from {len(calib)} held-out clean images (calibration FPR {100*cal_fpr:.2f}%)")

    # ---- score every depth with the same threshold ----
    rows, per_img = [], []
    for d in range(5):
        df, cc, wc = data[d]
        cs = score_paths(df[cc].astype(str).tolist(), vae, det, device, args.batch_size, args.latent_mode, args.scale)
        ws = score_paths(df[wc].astype(str).tolist(), vae, det, device, args.batch_size, args.latent_mode, args.scale)
        for sid, c, w in zip(df.sample_id, cs, ws):
            per_img += [dict(depth=f"G{d}", sample_id=int(sid), label=0, score=float(c)),
                        dict(depth=f"G{d}", sample_id=int(sid), label=1, score=float(w))]
        n = len(df)
        k_tp, k_fp = int((ws > thr).sum()), int((cs > thr).sum())
        auc = roc_auc_score(np.r_[np.zeros(n), np.ones(n)], np.r_[cs, ws])
        o_tpr, o_thr, o_fpr = oracle_tpr(cs, ws)
        nat = native_auc(args.native_log_dir, d)
        rows.append(dict(
            depth=f"G{d}", n_clean=n, n_wm=n, threshold=thr,
            tp=k_tp, tpr_frozen=100 * k_tp / n, tpr_ci95=wilson(k_tp, n),
            fp=k_fp, realized_fpr_frozen=100 * k_fp / n, fpr_ci95=wilson(k_fp, n),
            auc=auc, native_auc=nat, auc_abs_diff=(abs(auc - nat) if nat is not None else None),
            oracle_tpr_le1pct=100 * o_tpr, oracle_threshold=o_thr, oracle_fpr=100 * o_fpr))
        print(f"G{d}: TPR {100*k_tp/n:6.2f}%  FPR {100*k_fp/n:5.2f}%  AUC {auc:.4f}"
              + (f"  (native AUC {nat:.4f})" if nat is not None else ""))

    pd.DataFrame(per_img).to_csv(args.out_dir / "serum_scores_g0_g4.csv", index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(args.out_dir / "serum_frozen_summary.csv", index=False)
    print("\n" + summ[["depth", "tp", "tpr_frozen", "realized_fpr_frozen", "auc", "native_auc", "oracle_tpr_le1pct"]]
          .to_string(index=False))

    bad = summ.dropna(subset=["auc_abs_diff"])
    if len(bad) and (bad.auc_abs_diff > 0.005).any():
        print("\nWARNING: AUC differs from SERUM's native evaluator by > 0.005 at some depth. "
              "Preprocessing does not match SERUM's; do NOT report these scores. "
              "Try --latent-mode sample and/or --scale 1.0, or load SERUM's own model/preprocessing.")
    print("\nSaved to", args.out_dir)


if __name__ == "__main__":
    main()