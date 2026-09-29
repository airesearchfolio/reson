#!/usr/bin/env python3
"""
RESON SD2.1 — Dense Perturbation Sweep
Generate ONLY the missing perturbation severity levels.

Canonical source:
    reson_sd21_canonical_10k

Properties
----------
- Uses canonical held-out TEST split (N=1000)
- Generates clean + watermarked perturbations
- NO GPU required
- Four independent CPU workers
- Existing files are skipped
- Does NOT modify canonical perturbation outputs
- Uses exactly the same PIL/NumPy transformations as the canonical generator
- Deterministic noise using condition-specific seeds

Example:
    python generate_perturbations.py \
        --worker-id 0 --num-workers 4

Run worker IDs 0,1,2,3 simultaneously.
"""

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm


# ============================================================
# CONDITION
# ============================================================

@dataclass
class Condition:
    name: str
    attack: str
    strength: str
    fn: Callable[[Image.Image, np.random.Generator], Image.Image]


# ============================================================
# EXACT TRANSFORM IMPLEMENTATIONS FROM CANONICAL GENERATOR
# ============================================================

def jpeg_fn(quality: int):
    def _f(img, rng):
        buf = io.BytesIO()
        img.save(
            buf,
            format="JPEG",
            quality=int(quality),
            subsampling=0
        )
        buf.seek(0)

        out = Image.open(buf).convert("RGB")
        return out.copy()

    return _f


def blur_fn(radius: float):
    def _f(img, rng):
        return img.filter(
            ImageFilter.GaussianBlur(
                radius=float(radius)
            )
        )

    return _f


def noise_fn(std: float):
    def _f(img, rng):

        arr = (
            np.asarray(img)
            .astype(np.float32)
            / 255.0
        )

        noise = rng.normal(
            0.0,
            float(std),
            size=arr.shape
        ).astype(np.float32)

        arr = np.clip(
            arr + noise,
            0.0,
            1.0
        )

        return Image.fromarray(
            np.round(arr * 255.0)
            .astype(np.uint8),
            mode="RGB"
        )

    return _f


def resize_fn(scale: float):
    def _f(img, rng):

        w, h = img.size

        nw = max(
            1,
            int(round(w * float(scale)))
        )

        nh = max(
            1,
            int(round(h * float(scale)))
        )

        small = img.resize(
            (nw, nh),
            Image.Resampling.LANCZOS
        )

        return small.resize(
            (w, h),
            Image.Resampling.LANCZOS
        )

    return _f


def crop_fn(keep_fraction: float):
    def _f(img, rng):

        w, h = img.size

        cw = max(
            1,
            int(round(w * float(keep_fraction)))
        )

        ch = max(
            1,
            int(round(h * float(keep_fraction)))
        )

        left = (w - cw) // 2
        top = (h - ch) // 2

        cropped = img.crop(
            (
                left,
                top,
                left + cw,
                top + ch
            )
        )

        return cropped.resize(
            (w, h),
            Image.Resampling.LANCZOS
        )

    return _f


def brightness_fn(factor: float):
    def _f(img, rng):

        return ImageEnhance.Brightness(
            img
        ).enhance(
            float(factor)
        )

    return _f


def contrast_fn(factor: float):
    def _f(img, rng):

        return ImageEnhance.Contrast(
            img
        ).enhance(
            float(factor)
        )

    return _f


def rotate_fn(degrees: float):
    def _f(img, rng):

        return img.rotate(
            float(degrees),
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=(0, 0, 0),
        )

    return _f


# ============================================================
# ONLY NEW / MISSING CONDITIONS
# ============================================================

def new_conditions():

    return [

        # ----------------------------------------------------
        # JPEG
        # existing = 90, 70, 50
        # ----------------------------------------------------

        Condition(
            "jpeg_q30",
            "jpeg",
            "q30",
            jpeg_fn(30)
        ),

        Condition(
            "jpeg_q10",
            "jpeg",
            "q10",
            jpeg_fn(10)
        ),


        # ----------------------------------------------------
        # Gaussian Blur
        # existing = 0.5, 1.0, 2.0
        # ----------------------------------------------------

        Condition(
            "gaussian_blur_r3.0",
            "gaussian_blur",
            "r3.0",
            blur_fn(3.0)
        ),

        Condition(
            "gaussian_blur_r4.0",
            "gaussian_blur",
            "r4.0",
            blur_fn(4.0)
        ),


        # ----------------------------------------------------
        # Gaussian Noise
        # existing = .01, .03, .05
        # ----------------------------------------------------

        Condition(
            "gaussian_noise_std0.07",
            "gaussian_noise",
            "std0.07",
            noise_fn(0.07)
        ),

        Condition(
            "gaussian_noise_std0.10",
            "gaussian_noise",
            "std0.10",
            noise_fn(0.10)
        ),


        # ----------------------------------------------------
        # Resize
        # existing = .75, .50, .25
        # ----------------------------------------------------

        Condition(
            "resize_x0.15",
            "resize",
            "x0.15",
            resize_fn(0.15)
        ),

        Condition(
            "resize_x0.10",
            "resize",
            "x0.10",
            resize_fn(0.10)
        ),


        # ----------------------------------------------------
        # Center Crop
        # existing = .90, .80, .70
        # ----------------------------------------------------

        Condition(
            "center_crop_keep0.60",
            "center_crop",
            "keep0.60",
            crop_fn(0.60)
        ),

        Condition(
            "center_crop_keep0.50",
            "center_crop",
            "keep0.50",
            crop_fn(0.50)
        ),


        # ----------------------------------------------------
        # Brightness
        # existing = .80, 1.20
        # ----------------------------------------------------

        Condition(
            "brightness_x0.40",
            "brightness",
            "x0.40",
            brightness_fn(0.40)
        ),

        Condition(
            "brightness_x0.60",
            "brightness",
            "x0.60",
            brightness_fn(0.60)
        ),

        Condition(
            "brightness_x1.40",
            "brightness",
            "x1.40",
            brightness_fn(1.40)
        ),

        Condition(
            "brightness_x1.60",
            "brightness",
            "x1.60",
            brightness_fn(1.60)
        ),


        # ----------------------------------------------------
        # Contrast
        # existing = .80, 1.20
        # ----------------------------------------------------

        Condition(
            "contrast_x0.40",
            "contrast",
            "x0.40",
            contrast_fn(0.40)
        ),

        Condition(
            "contrast_x0.60",
            "contrast",
            "x0.60",
            contrast_fn(0.60)
        ),

        Condition(
            "contrast_x1.40",
            "contrast",
            "x1.40",
            contrast_fn(1.40)
        ),

        Condition(
            "contrast_x1.60",
            "contrast",
            "x1.60",
            contrast_fn(1.60)
        ),


        # ----------------------------------------------------
        # Rotation
        # existing = 2, 5
        # ----------------------------------------------------

        Condition(
            "rotation_deg10",
            "rotation",
            "deg10",
            rotate_fn(10)
        ),

        Condition(
            "rotation_deg15",
            "rotation",
            "deg15",
            rotate_fn(15)
        ),

        Condition(
            "rotation_deg20",
            "rotation",
            "deg20",
            rotate_fn(20)
        ),
    ]


# ============================================================
# DETERMINISTIC CONDITION SEED
# EXACT SAME FORMULA AS CANONICAL GENERATOR
# ============================================================

def condition_seed(
    base_seed: int,
    sample_id: int,
    condition_name: str
) -> int:

    return (
        int(base_seed)
        + 1000003 * int(sample_id)
        + sum(
            (i + 1) * ord(c)
            for i, c in enumerate(condition_name)
        )
    ) % (2**32)


# ============================================================
# RESOLVE CANONICAL G0 PAIR
# ============================================================

def resolve_pair(root: Path, row):

    sid = int(row["sample_id"])

    clean_cols = [
        "clean_path",
        "clean_image_path",
        "source_clean_path",
    ]

    wm_cols = [
        "wm_path",
        "wm_image_path",
        "watermarked_path",
        "source_wm_path",
    ]

    clean = None
    wm = None

    for c in clean_cols:

        if (
            c in row.index
            and pd.notna(row[c])
            and str(row[c]).strip()
        ):

            q = Path(str(row[c]))

            if q.exists():
                clean = q
                break


    for c in wm_cols:

        if (
            c in row.index
            and pd.notna(row[c])
            and str(row[c]).strip()
        ):

            q = Path(str(row[c]))

            if q.exists():
                wm = q
                break


    if clean is None:

        candidates = [

            root
            / "images"
            / f"{sid:05d}.png",

            root
            / f"{sid:05d}.png",
        ]

        clean = next(
            (
                q
                for q in candidates
                if q.exists()
            ),
            None
        )


    if wm is None:

        candidates = [

            root
            / "images"
            / f"{sid:05d}_wm.png",

            root
            / f"{sid:05d}_wm.png",
        ]

        wm = next(
            (
                q
                for q in candidates
                if q.exists()
            ),
            None
        )


    if clean is None or wm is None:

        raise FileNotFoundError(
            f"Could not resolve canonical G0 pair "
            f"for sample_id={sid}\n"
            f"clean={clean}\n"
            f"wm={wm}"
        )

    return clean, wm


# ============================================================
# ATOMIC PNG SAVE
# ============================================================

def save_png(img, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    tmp = path.with_suffix(
        ".tmp.png"
    )

    img.save(
        tmp,
        format="PNG",
        compress_level=3
    )

    tmp.replace(path)


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        default=(
            "./watermark/"
            "experiments/8/paper_experiments/"
            "reson_sd21_canonical_10k"
        )
    )

    ap.add_argument(
        "--output-root",
        default=None
    )

    ap.add_argument(
        "--worker-id",
        type=int,
        required=True
    )

    ap.add_argument(
        "--num-workers",
        type=int,
        default=4
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=2026
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None
    )

    ap.add_argument(
        "--overwrite",
        action="store_true"
    )

    args = ap.parse_args()


    # --------------------------------------------------------
    # Validate worker
    # --------------------------------------------------------

    if not (
        0 <= args.worker_id < args.num_workers
    ):

        raise ValueError(
            "worker-id must satisfy "
            "0 <= worker-id < num-workers"
        )


    root = Path(args.root)

    manifest_path = (
        root / "manifest.csv"
    )


    output_root = (

        Path(args.output_root)

        if args.output_root

        else root / "perturbations_dense_new"
    )


    output_root.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Load canonical test split
    # --------------------------------------------------------

    df = pd.read_csv(
        manifest_path
    )


    required = {
        "sample_id",
        "split"
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Manifest missing columns: "
            f"{sorted(missing)}"
        )


    test = df[
        df["split"]
        .astype(str)
        .str.lower()
        == "test"
    ].copy()


    test = (
        test
        .sort_values("sample_id")
        .reset_index(drop=True)
    )


    if (
        args.limit is None
        and len(test) != 1000
    ):

        raise RuntimeError(
            "Expected canonical "
            f"1000-row test split; "
            f"found {len(test)}"
        )


    if args.limit is not None:

        test = (
            test
            .head(args.limit)
            .copy()
        )


    if test["sample_id"].duplicated().any():

        raise RuntimeError(
            "Duplicate sample_id values."
        )


    # --------------------------------------------------------
    # Validate source pairs
    # --------------------------------------------------------

    pairs = {}

    for _, row in test.iterrows():

        sid = int(
            row["sample_id"]
        )

        pairs[sid] = resolve_pair(
            root,
            row
        )


    # --------------------------------------------------------
    # Split CONDITIONS across workers
    #
    # Worker 0 gets conditions 0,4,8,...
    # Worker 1 gets conditions 1,5,9,...
    # etc.
    # --------------------------------------------------------

    all_conditions = new_conditions()

    worker_conditions = (

        all_conditions[
            args.worker_id
            :: args.num_workers
        ]
    )


    print(
        f"\nWorker {args.worker_id}/"
        f"{args.num_workers}"
    )

    print(
        f"Canonical test N = {len(test)}"
    )

    print(
        f"Total new conditions = "
        f"{len(all_conditions)}"
    )

    print(
        f"This worker conditions = "
        f"{len(worker_conditions)}"
    )


    for c in worker_conditions:

        print(
            f"  - {c.name}"
        )


    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    rows = []


    for cond in worker_conditions:

        print(
            f"\n========== "
            f"{cond.name} "
            f"=========="
        )


        cond_dir = (
            output_root
            / cond.name
        )


        cond_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        for _, row in tqdm(
            test.iterrows(),
            total=len(test),
            desc=cond.name
        ):

            sid = int(
                row["sample_id"]
            )


            source_clean, source_wm = (
                pairs[sid]
            )


            clean_out = (
                cond_dir
                / f"{sid:05d}.png"
            )


            wm_out = (
                cond_dir
                / f"{sid:05d}_wm.png"
            )


            seed = condition_seed(
                args.seed,
                sid,
                cond.name
            )


            # ----------------------------------------------
            # Clean
            # ----------------------------------------------

            if (
                args.overwrite
                or not clean_out.exists()
            ):

                with Image.open(
                    source_clean
                ) as im:

                    img = im.convert("RGB")

                    rng = (
                        np.random.default_rng(
                            seed
                        )
                    )

                    attacked = cond.fn(
                        img,
                        rng
                    )

                    save_png(
                        attacked,
                        clean_out
                    )


            # ----------------------------------------------
            # Watermarked
            #
            # IMPORTANT:
            # Same seed so stochastic noise realization
            # is matched between clean/watermarked pair.
            # ----------------------------------------------

            if (
                args.overwrite
                or not wm_out.exists()
            ):

                with Image.open(
                    source_wm
                ) as im:

                    img = im.convert("RGB")

                    rng = (
                        np.random.default_rng(
                            seed
                        )
                    )

                    attacked = cond.fn(
                        img,
                        rng
                    )

                    save_png(
                        attacked,
                        wm_out
                    )


            rows.append({
                "sample_id": sid,
                "split": "test",
                "condition": cond.name,
                "attack": cond.attack,
                "strength": cond.strength,
                "clean_path": str(
                    clean_out.resolve()
                ),
                "wm_path": str(
                    wm_out.resolve()
                ),
                "condition_seed": seed,
            })


    # --------------------------------------------------------
    # Worker manifest
    # --------------------------------------------------------

    manifest = pd.DataFrame(
        rows
    )


    manifest_out = (

        output_root
        / (
            f"manifest_worker"
            f"{args.worker_id}.csv"
        )
    )


    manifest.to_csv(
        manifest_out,
        index=False
    )


    print(
        "\n================================"
    )

    print(
        f"WORKER {args.worker_id} COMPLETE"
    )

    print(
        f"Rows: {len(manifest)}"
    )

    print(
        f"Manifest: {manifest_out}"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()