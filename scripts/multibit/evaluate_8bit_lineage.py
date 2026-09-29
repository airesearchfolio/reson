#!/usr/bin/env python3
"""
Evaluate FINAL RESON 8-bit detector on G0-G4 canonical lineage.

Primary:
    Validation-frozen presence threshold from final_detector_8bit/results.json.

Payload:
    - mean per-bit accuracy across 8 bits
    - individual bit accuracies
    - exact 8-bit message recovery

Secondary diagnostic:
    Per-depth ROC TPR at <=1% FPR.

IMPORTANT:
    - No retraining.
    - No threshold recalibration for the primary result.
    - No watermark reinsertion after G0.
    - G0-G4 each contain exactly 1,000 held-out pairs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from diffusers import AutoencoderKL


# ============================================================
# MODEL
# ============================================================

class Joint8BitDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.presence = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.payload = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 8),
        )

    def forward(self, z):
        h = self.features(z)
        return self.presence(h).squeeze(1), self.payload(h)


# ============================================================
# IMAGE / LATENT
# ============================================================

def load_rgb(path):
    return (
        Image.open(path)
        .convert("RGB")
        .resize((512, 512), Image.Resampling.LANCZOS)
    )


def tensorize(img):
    a = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1)


@torch.no_grad()
def encode(vae, x):
    # EXACT match to final 8-bit trainer.
    return (
        vae.encode(x).latent_dist.sample()
        * vae.config.scaling_factor
    ).float()


# ============================================================
# OPERATING POINT
# ============================================================

def op_at_fpr(y, s, target=0.01):

    fpr, tpr, thr = roc_curve(y, s)

    ok = np.where(fpr <= target)[0]

    if len(ok) == 0:
        raise RuntimeError(
            f"No ROC operating point with FPR <= {target}"
        )

    j = ok[np.argmax(tpr[ok])]

    return (
        float(tpr[j]),
        float(fpr[j]),
        float(thr[j]),
    )


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def infer_pairs(
    df,
    model,
    vae,
    device,
    batch_size,
    depth,
):

    items = []

    for _, r in df.iterrows():

        bits = [
            int(float(r[f"bit{i}"]))
            for i in range(8)
        ]

        sid = int(float(r["sample_id"]))

        clean = r["clean_path"]
        wm = r["wm_path"]

        items.append(
            (clean, 0, bits, sid, "clean")
        )

        items.append(
            (wm, 1, bits, sid, "wm")
        )

    records = []

    for st in tqdm(
        range(0, len(items), batch_size),
        desc=f"{depth.upper()} detector",
        dynamic_ncols=True,
    ):

        chunk = items[
            st:st + batch_size
        ]

        x = torch.stack([
            tensorize(load_rgb(q[0]))
            for q in chunk
        ]).to(device)

        z = encode(vae, x)

        presence_logits, bit_logits = model(z)

        presence_prob = (
            torch.sigmoid(presence_logits)
            .cpu()
            .numpy()
        )

        bit_prob = (
            torch.sigmoid(bit_logits)
            .cpu()
            .numpy()
        )

        for q, pp, bp in zip(
            chunk,
            presence_prob,
            bit_prob,
        ):

            path, label, bits, sid, kind = q

            rec = {
                "sample_id": sid,
                "depth": depth,
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


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    pred,
    frozen_threshold,
    depth,
):

    y = (
        pred["label"]
        .to_numpy()
        .astype(int)
    )

    s = (
        pred["presence_score"]
        .to_numpy()
    )

    auc = float(
        roc_auc_score(y, s)
    )

    detected = (
        s >= frozen_threshold
    )

    tpr_frozen = float(
        detected[y == 1].mean()
    )

    fpr_frozen = float(
        detected[y == 0].mean()
    )

    # Payload evaluated ONLY on WM examples.
    wm = (
        pred[
            pred["kind"] == "wm"
        ]
        .copy()
    )

    per_bit = {}

    correct_matrix = []

    for i in range(8):

        bhat = (
            wm[f"bit_prob{i}"]
            .to_numpy()
            >= 0.5
        ).astype(int)

        btrue = (
            wm[f"true_bit{i}"]
            .to_numpy()
            .astype(int)
        )

        correct = (
            bhat == btrue
        )

        correct_matrix.append(
            correct
        )

        per_bit[f"bit{i}"] = float(
            correct.mean()
        )

    correct_matrix = np.stack(
        correct_matrix,
        axis=1,
    )

    # Accuracy over all N*8 individual bits.
    mean_bit_accuracy = float(
        correct_matrix.mean()
    )

    # Fraction of images for which ALL 8 bits are correct.
    exact_8bit_accuracy = float(
        np.all(
            correct_matrix,
            axis=1,
        ).mean()
    )

    diag_tpr, diag_fpr, diag_thr = (
        op_at_fpr(
            y,
            s,
            0.01,
        )
    )

    return {

        "depth": depth,

        "n_pairs": int(
            len(wm)
        ),

        "auc": auc,

        "frozen_threshold":
            float(frozen_threshold),

        "tpr_frozen":
            tpr_frozen,

        "fpr_frozen":
            fpr_frozen,

        "per_bit_accuracy":
            per_bit,

        "mean_bit_accuracy":
            mean_bit_accuracy,

        "exact_8bit_message_accuracy":
            exact_8bit_accuracy,

        "diagnostic_tpr_at_le_1pct_fpr":
            diag_tpr,

        "diagnostic_fpr":
            diag_fpr,

        "diagnostic_threshold":
            diag_thr,
    }


# ============================================================
# FROZEN THRESHOLD
# ============================================================

def recover_frozen_threshold(
    results_path,
):

    if not results_path.exists():

        raise FileNotFoundError(
            f"Missing {results_path}. "
            "The frozen threshold must come "
            "from the 8-bit validation set."
        )

    result = json.loads(
        results_path.read_text()
    )

    key = (
        "validation_threshold_at_le_1pct_fpr"
    )

    if key not in result:

        raise KeyError(
            f"{results_path} does not "
            f"contain '{key}'"
        )

    return float(
        result[key]
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_depth(
    df,
    depth,
    expected_n=1000,
):

    required = (
        ["sample_id", "clean_path", "wm_path"]
        + [
            f"bit{i}"
            for i in range(8)
        ]
    )

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise KeyError(
            f"{depth}: missing columns: "
            f"{missing}"
        )

    if len(df) != expected_n:

        raise RuntimeError(
            f"{depth}: expected "
            f"{expected_n} rows; "
            f"found {len(df)}"
        )

    if df["sample_id"].duplicated().any():

        raise RuntimeError(
            f"{depth}: duplicate sample IDs"
        )

    # Validate actual files before GPU inference.
    for col in [
        "clean_path",
        "wm_path",
    ]:

        missing_files = [
            p
            for p in df[col].astype(str)
            if not Path(p).exists()
        ]

        if missing_files:

            raise FileNotFoundError(
                f"{depth}: missing {col}; "
                f"first missing file = "
                f"{missing_files[0]}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data-root",
        default=(
            "./watermark/"
            "experiments/8/paper_experiments/"
            "exp_16_coco_fid/"
            "sd21_8bit_alpha0155_beta040_n10000"
        ),
    )

    ap.add_argument(
        "--lineage-root",
        default=None,
    )

    ap.add_argument(
        "--checkpoint",
        default=None,
    )

    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional override. Normally leave unset "
            "so validation-frozen threshold is loaded "
            "from final_detector_8bit/results.json."
        ),
    )

    ap.add_argument(
        "--vae-id",
        default=(
            "sd2-community/"
            "stable-diffusion-2-1-base"
        ),
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--device",
        default="cuda",
    )

    args = ap.parse_args()


    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    root = Path(
        args.data_root
    )

    lineage = (
        Path(args.lineage_root)
        if args.lineage_root
        else root / "lineage_8bit_test1000"
    )

    detector_dir = (
        root / "final_detector_8bit"
    )

    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else detector_dir / "detector_best.pt"
    )

    training_results = (
        detector_dir / "results.json"
    )

    out = (
        root
        / "final_lineage_8bit_evaluation"
    )

    out.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Frozen threshold
    # --------------------------------------------------------

    frozen_threshold = (

        float(args.threshold)

        if args.threshold is not None

        else recover_frozen_threshold(
            training_results
        )
    )


    print(
        "Data root:",
        root,
    )

    print(
        "Lineage root:",
        lineage,
    )

    print(
        "Checkpoint:",
        checkpoint,
    )

    print(
        "Frozen validation threshold:",
        frozen_threshold,
    )

    print(
        "Output:",
        out,
    )


    # --------------------------------------------------------
    # G0
    # --------------------------------------------------------

    base_manifest = (
        root / "manifest_final_10000.csv"
    )

    base = pd.read_csv(
        base_manifest
    )

    base["split"] = (
        base["split"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    base = (
        base[
            base["split"] == "test"
        ]
        .copy()
        .sort_values("sample_id")
    )


    # Normalize G0 column name:
    # wm8_path -> wm_path
    g0_columns = (
        [
            "sample_id",
            "clean_path",
            "wm8_path",
        ]
        + [
            f"bit{i}"
            for i in range(8)
        ]
    )

    missing = [
        c
        for c in g0_columns
        if c not in base.columns
    ]

    if missing:

        raise KeyError(
            "G0 manifest missing columns: "
            f"{missing}"
        )


    g0 = (
        base[g0_columns]
        .copy()
        .rename(
            columns={
                "wm8_path": "wm_path"
            }
        )
    )


    # --------------------------------------------------------
    # G1-G4
    # --------------------------------------------------------

    depths = {
        "g0": g0
    }

    for depth in [
        "g1",
        "g2",
        "g3",
        "g4",
    ]:

        path = (
            lineage
            / f"manifest_{depth}.csv"
        )

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        depths[depth] = (
            pd.read_csv(path)
            .sort_values("sample_id")
            .reset_index(drop=True)
        )


    # --------------------------------------------------------
    # Validate ALL depths BEFORE loading GPU model
    # --------------------------------------------------------

    print(
        "\nValidating G0-G4..."
    )

    for depth, df in depths.items():

        validate_depth(
            df,
            depth,
            expected_n=1000,
        )

        print(
            f"{depth.upper()}: "
            f"{len(df)} valid pairs"
        )


    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    device = torch.device(
        args.device
    )


    model = (
        Joint8BitDecoder()
        .to(device)
    )


    state = torch.load(
        checkpoint,
        map_location=device,
    )


    if isinstance(state, dict):

        sd = state.get(
            "model_state_dict",
            state.get(
                "model",
                state.get(
                    "state_dict",
                    state,
                ),
            ),
        )

    else:

        sd = state


    model.load_state_dict(
        sd,
        strict=True,
    )

    model.eval()


    vae = (
        AutoencoderKL
        .from_pretrained(
            args.vae_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        .to(device)
        .eval()
    )


    for p in vae.parameters():
        p.requires_grad = False


    # --------------------------------------------------------
    # Evaluate G0-G4
    # --------------------------------------------------------

    results = {}


    for depth, df in depths.items():

        print(
            "\n"
            + "=" * 72
        )

        print(
            f"EVALUATING 8-BIT "
            f"{depth.upper()}"
        )

        print(
            "=" * 72
        )


        pred = infer_pairs(
            df,
            model,
            vae,
            device,
            args.batch_size,
            depth,
        )


        pred.to_csv(
            out
            / f"predictions_{depth}.csv",
            index=False,
        )


        m = compute_metrics(
            pred,
            frozen_threshold,
            depth,
        )


        results[depth] = m


        print(
            json.dumps(
                m,
                indent=2,
            )
        )


    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------

    rows = []


    for depth in [
        "g0",
        "g1",
        "g2",
        "g3",
        "g4",
    ]:

        m = results[depth]


        row = {

            "depth":
                depth,

            "n_pairs":
                m["n_pairs"],

            "auc":
                m["auc"],

            "frozen_threshold":
                m["frozen_threshold"],

            "frozen_tpr":
                m["tpr_frozen"],

            "frozen_fpr":
                m["fpr_frozen"],

            "mean_bit_accuracy":
                m["mean_bit_accuracy"],

            "exact_8bit_message_accuracy":
                m[
                    "exact_8bit_message_accuracy"
                ],

            "diagnostic_tpr_at_le_1pct_fpr":
                m[
                    "diagnostic_tpr_at_le_1pct_fpr"
                ],

            "diagnostic_fpr":
                m[
                    "diagnostic_fpr"
                ],

            "diagnostic_threshold":
                m[
                    "diagnostic_threshold"
                ],
        }


        for i in range(8):

            row[f"bit{i}"] = (
                m[
                    "per_bit_accuracy"
                ][f"bit{i}"]
            )


        rows.append(row)


    summary = pd.DataFrame(
        rows
    )


    summary.to_csv(
        out
        / "lineage_8bit_metrics_summary.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Full JSON
    # --------------------------------------------------------

    (
        out
        / "results_g0_g4.json"
    ).write_text(
        json.dumps(
            results,
            indent=2,
        )
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "8-BIT G0-G4 EVALUATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        summary[
            [
                "depth",
                "n_pairs",
                "frozen_tpr",
                "frozen_fpr",
                "mean_bit_accuracy",
                "exact_8bit_message_accuracy",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nSummary:",
        out
        / "lineage_8bit_metrics_summary.csv",
    )


if __name__ == "__main__":
    main()