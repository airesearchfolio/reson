#!/usr/bin/env python3
"""
Evaluate the FINAL RESON 8-bit detector on the untouched G0 held-out test set.

Matches train_8bit_detector.py exactly:
  - SD2.1 VAE
  - posterior sample() * scaling_factor
  - same Joint8BitDecoder architecture
  - 1,000 held-out test pairs from manifest_final_10000.csv
  - PRIMARY operating point = validation-frozen threshold stored/recovered
    from final_detector_8bit/results.json
  - SECONDARY diagnostic = test-set TPR at <=1% FPR

No retraining. No test-set threshold tuning for the primary result.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from tqdm import tqdm


class Joint8BitDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.presence = nn.Sequential(
            nn.Flatten(), nn.Linear(4096, 512), nn.ReLU(), nn.Linear(512, 1)
        )
        self.payload = nn.Sequential(
            nn.Flatten(), nn.Linear(4096, 512), nn.ReLU(), nn.Linear(512, 8)
        )

    def forward(self, z):
        h = self.features(z)
        return self.presence(h).squeeze(1), self.payload(h)


def load_rgb(path):
    return Image.open(path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)


def tensorize(img):
    a = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1)


@torch.no_grad()
def encode(vae, x):
    # EXACT match to the final 8-bit trainer.
    return (vae.encode(x).latent_dist.sample() * vae.config.scaling_factor).float()


def op_at_fpr(y, s, target=0.01):
    fpr, tpr, thr = roc_curve(y, s)
    ok = np.where(fpr <= target)[0]
    j = ok[np.argmax(tpr[ok])]
    return float(tpr[j]), float(fpr[j]), float(thr[j])


@torch.no_grad()
def infer_pairs(df, model, vae, device, batch_size):
    items = []
    for _, r in df.iterrows():
        bits = [int(float(r[f"bit{i}"])) for i in range(8)]
        sid = int(float(r["sample_id"]))
        clean = r["clean_path"]
        wm = r["wm8_path"]
        # Clean examples are used only for presence detection. Payload metrics
        # are computed only on watermarked examples.
        items.append((clean, 0, bits, sid, "clean"))
        items.append((wm, 1, bits, sid, "wm"))

    records = []
    for st in tqdm(range(0, len(items), batch_size), desc="G0 detector", dynamic_ncols=True):
        chunk = items[st:st + batch_size]
        x = torch.stack([tensorize(load_rgb(q[0])) for q in chunk]).to(device)
        z = encode(vae, x)
        presence_logits, bit_logits = model(z)

        presence_prob = torch.sigmoid(presence_logits).cpu().numpy()
        bit_prob = torch.sigmoid(bit_logits).cpu().numpy()

        for q, pp, bp in zip(chunk, presence_prob, bit_prob):
            path, label, bits, sid, kind = q
            rec = {
                "sample_id": sid,
                "kind": kind,
                "path": str(path),
                "label": label,
                "presence_score": float(pp),
            }
            for i in range(8):
                rec[f"true_bit{i}"] = bits[i]
                rec[f"bit_prob{i}"] = float(bp[i])
            records.append(rec)

    return pd.DataFrame(records)


def compute_metrics(pred, frozen_threshold):
    y = pred["label"].to_numpy().astype(int)
    s = pred["presence_score"].to_numpy()

    auc = float(roc_auc_score(y, s))

    detected = s >= frozen_threshold
    tpr_frozen = float(detected[y == 1].mean())
    fpr_frozen = float(detected[y == 0].mean())

    wm = pred[pred["kind"] == "wm"].copy()
    per_bit = {}
    correct_matrix = []

    for i in range(8):
        bhat = (wm[f"bit_prob{i}"].to_numpy() >= 0.5).astype(int)
        btrue = wm[f"true_bit{i}"].to_numpy().astype(int)
        correct = bhat == btrue
        correct_matrix.append(correct)
        per_bit[f"bit{i}"] = float(correct.mean())

    correct_matrix = np.stack(correct_matrix, axis=1)
    mean_bit_accuracy = float(correct_matrix.mean())
    exact_8bit_accuracy = float(np.all(correct_matrix, axis=1).mean())

    diag_tpr, diag_fpr, diag_thr = op_at_fpr(y, s, 0.01)

    return {
        "n_pairs": int(len(wm)),
        "auc": auc,
        "frozen_threshold": float(frozen_threshold),
        "tpr_frozen": tpr_frozen,
        "fpr_frozen": fpr_frozen,
        "per_bit_accuracy": per_bit,
        "mean_bit_accuracy": mean_bit_accuracy,
        "exact_8bit_message_accuracy": exact_8bit_accuracy,
        "diagnostic_tpr_at_le_1pct_fpr": diag_tpr,
        "diagnostic_fpr": diag_fpr,
        "diagnostic_threshold": diag_thr,
        "depth": "g0",
    }


def recover_frozen_threshold(results_path):
    if not results_path.exists():
        raise FileNotFoundError(
            f"Missing {results_path}. The frozen threshold must come from the "
            "8-bit validation set, not from the G0 test set."
        )
    result = json.loads(results_path.read_text())
    key = "validation_threshold_at_le_1pct_fpr"
    if key not in result:
        raise KeyError(f"{results_path} does not contain '{key}'")
    return float(result[key])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        default="workspace/paper_experiments/"
                "exp_16_coco_fid/sd21_8bit_alpha0155_beta040_n10000",
    )
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Optional override. Normally leave unset so the validation-frozen threshold is loaded from results.json.")
    ap.add_argument("--vae-id", default="sd2-community/stable-diffusion-2-1-base")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    root = Path(args.data_root)
    detector_dir = root / "final_detector_8bit"
    checkpoint = Path(args.checkpoint) if args.checkpoint else detector_dir / "detector_best.pt"
    training_results = detector_dir / "results.json"
    out = root / "final_g0_8bit_evaluation"
    out.mkdir(parents=True, exist_ok=True)

    manifest = root / "manifest_final_10000.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    frozen_threshold = (
        float(args.threshold)
        if args.threshold is not None
        else recover_frozen_threshold(training_results)
    )

    print("Data root:", root)
    print("Manifest:", manifest)
    print("Checkpoint:", checkpoint)
    print("Frozen validation threshold:", frozen_threshold)
    print("Output:", out)

    df = pd.read_csv(manifest)
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    test = df[df["split"] == "test"].copy().sort_values("sample_id")

    if len(test) != 1000:
        raise RuntimeError(f"Expected exactly 1000 held-out test rows; found {len(test)}")

    required = ["sample_id", "clean_path", "wm8_path"] + [f"bit{i}" for i in range(8)]
    missing = [c for c in required if c not in test.columns]
    if missing:
        raise KeyError(f"Manifest missing columns: {missing}")

    print(f"Held-out G0 test pairs: {len(test)}")

    device = torch.device(args.device)

    model = Joint8BitDecoder().to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict):
        sd = state.get("model_state_dict", state.get("model", state.get("state_dict", state)))
    else:
        sd = state
    model.load_state_dict(sd, strict=True)
    model.eval()

    vae = AutoencoderKL.from_pretrained(
        args.vae_id, subfolder="vae", torch_dtype=torch.float32
    ).to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    print("\n" + "=" * 72)
    print("EVALUATING 8-BIT G0")
    print("=" * 72)

    pred = infer_pairs(test, model, vae, device, args.batch_size)
    pred.to_csv(out / "predictions_g0.csv", index=False)

    metrics = compute_metrics(pred, frozen_threshold)

    (out / "results_g0.json").write_text(json.dumps(metrics, indent=2))

    summary = {
        "depth": "g0",
        "n_pairs": metrics["n_pairs"],
        "auc": metrics["auc"],
        "frozen_threshold": metrics["frozen_threshold"],
        "frozen_tpr": metrics["tpr_frozen"],
        "frozen_fpr": metrics["fpr_frozen"],
        **metrics["per_bit_accuracy"],
        "mean_bit_accuracy": metrics["mean_bit_accuracy"],
        "exact_8bit_message_accuracy": metrics["exact_8bit_message_accuracy"],
        "diagnostic_tpr_at_le_1pct_fpr": metrics["diagnostic_tpr_at_le_1pct_fpr"],
        "diagnostic_fpr": metrics["diagnostic_fpr"],
        "diagnostic_threshold": metrics["diagnostic_threshold"],
    }
    pd.DataFrame([summary]).to_csv(out / "g0_8bit_metrics_summary.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print("\nDONE:", out)


if __name__ == "__main__":
    main()
