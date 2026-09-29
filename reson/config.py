#!/usr/bin/env python3
"""
stage6/config.py -- one place for all settings, mirroring SERUM's
Config-object pattern so alpha, blend, paths, etc. aren't scattered
across CLI flags in every script.
"""

from pathlib import Path
from dataclasses import dataclass, field
import torch


@dataclass
class WatermarkConfig:
    noise_mix_alpha: float = 0.155        # injection strength alpha
    carrier_blend: float = 0.40          # carrier blend beta (structured : white)
    latent_channels: int = 4
    latent_h: int = 64
    latent_w: int = 64


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 5e-5
    presence_weight: float = 1.0
    bit_weight: float = 0.5
    margin: float = 1.5
    use_margin_loss: bool = True
    patience: int = 8
    sanity_every: int = 5
    train_generators: tuple = ("realvis", "dreamshaper")
    train_strengths: tuple = (0.30, 0.50)


@dataclass
class EvalConfig:
    test_limit: int = 120
    bootstrap: int = 2000
    eval_strength: float = 0.50
    eval_steps: int = 30
    sdxl_guidance: float = 7.5
    flux_guidance: float = 3.5
    flux_max_sequence_length: int = 512
    # canonical multi-hop chain: depth -> (short_name, family, model_id_attr)
    # family is one of {"flux", "sdxl"}; model_id_attr looks up the actual
    # model id from this Config object at runtime.
    chain: tuple = (
        (1, "flux", "flux", "flux_model_id"),
        (2, "realvis", "sdxl", "realvis_model_id"),
        (3, "dreamshaper", "sdxl", "dreamshaper_model_id"),
        (4, "flux", "flux", "flux_model_id"),
    )


@dataclass
class Config:
    root: Path = Path("reson")
    seed: int = 20260903

    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    image_size: int = 512
    decoder_image_size: int = 256   # unused now (latent-space detector), kept for reference
    generation_steps: int = 30
    guidance: float = 7.0

    sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    flux_model_id: str = "black-forest-labs/FLUX.1-dev"
    realvis_model_id: str = "SG161222/RealVisXL_V5.0"
    dreamshaper_model_id: str = "Lykon/dreamshaper-xl-1-0"
    vae_fp16fix_id: str = "madebyollin/sdxl-vae-fp16-fix"

    # ---- derived paths -----------------------------------------------

    @property
    def split_csv(self) -> Path:
        return self.root / "prompt_split.csv"

    @property
    def carrier_path(self) -> Path:
        b = self.watermark.carrier_blend
        return self.root / f"carrier_blend{b:.2f}.npy"

    @property
    def source_dir(self) -> Path:
        return self.root / "source_images"

    @property
    def source_manifest(self) -> Path:
        return self.source_dir / "generated_manifest.csv"

    @property
    def regen_train_dir(self) -> Path:
        return self.root / "regen_train_data"

    @property
    def regen_train_manifest_dir(self) -> Path:
        return self.regen_train_dir / "manifests"

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def detector_ckpt(self) -> Path:
        return self.checkpoint_dir / "detector_best.pt"

    @property
    def history_csv(self) -> Path:
        return self.checkpoint_dir / "detector_history.csv"

    @property
    def lineage_dir(self) -> Path:
        return self.root / "lineage"

    @property
    def lineage_manifest_dir(self) -> Path:
        return self.lineage_dir / "manifests"

    @property
    def lineage_image_dir(self) -> Path:
        return self.lineage_dir / "images"

    @property
    def eval_dir(self) -> Path:
        return self.root / "evaluation"

    def get_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def make_dirs(self) -> None:
        for p in [
            self.root, self.source_dir, self.regen_train_dir,
            self.regen_train_manifest_dir, self.checkpoint_dir,
            self.lineage_dir, self.lineage_manifest_dir,
            self.lineage_image_dir, self.eval_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)
