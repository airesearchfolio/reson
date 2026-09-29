#!/usr/bin/env python3
"""
stage6/detector.py -- the winning detector design, confirmed across
three separate training runs (all landed in the same stable range:
AUC ~0.92-0.95, G4 TPR@1%FPR ~41%):

  1. Detection happens on VAE-encoded latents, not raw pixels
     (following SERUM's score_model.py design -- same conv-stack
     shape, applied to a harder problem since this detector also
     recovers a bit, not just presence).
  2. Presence loss uses a margin objective instead of plain BCE, to
     push apart the near-threshold overlap that caps TPR at strict
     FPR despite a healthy AUC.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from diffusers import AutoencoderKL

from config import Config


class LatentJointDecoder(nn.Module):
    """Same conv-stack shape as SERUM's WatermarkScoreModel, extended
    with a second output head so it recovers the bit, not just
    presence."""

    def __init__(self, latent_channels: int = 4):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(latent_channels, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        flat_dim = 256 * 4 * 4  # 64x64 latent -> 4x4 after 4 pools
        self.presence_fc = nn.Sequential(nn.Linear(flat_dim, 512), nn.ReLU(inplace=True), nn.Linear(512, 1))
        self.bit_fc = nn.Sequential(nn.Linear(flat_dim, 512), nn.ReLU(inplace=True), nn.Linear(512, 1))

    def forward(self, latent: torch.Tensor):
        f = torch.flatten(self.conv_layers(latent), 1)
        return self.bit_fc(f).squeeze(1), self.presence_fc(f).squeeze(1)

    def save(self, path, extra: dict | None = None) -> None:
        payload = {"model_state_dict": self.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print("Saved detector to:", path)

    @staticmethod
    def load(config: Config, path=None, device: str | None = None) -> "LatentJointDecoder":
        path = path or config.detector_ckpt
        device = device or config.get_device()
        ck = torch.load(path, map_location="cpu")
        model = LatentJointDecoder(config.watermark.latent_channels).to(device)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        model.eval()
        print("Loaded detector:", path, " selection:", ck.get("best_selection", "?"))
        return model


def margin_presence_loss(presence_logit: torch.Tensor, presence: torch.Tensor, margin: float) -> torch.Tensor:
    """Pushes confidently-correct predictions further from the decision
    boundary -- directly targets the near-threshold overlap that caps
    TPR at strict FPR (this is NOT confirmed from SERUM's code -- it's
    a standard technique applied here as a new addition on top of the
    latent-space idea)."""
    target_sign = 2.0 * presence - 1.0  # 0 -> -1, 1 -> +1
    return torch.clamp(margin - presence_logit * target_sign, min=0.0).mean()


class VAEEncoder:
    """VAE-only wrapper (SERUM-confirmed idea: detect on latents, not
    pixels). Uses the community fp16-fix VAE -- SDXL's original VAE
    produces NaN in fp16 for a meaningful fraction of inputs."""

    def __init__(self, config: Config):
        device = config.get_device()
        print("Loading VAE (fp16-fix):", config.vae_fp16fix_id)
        self.vae = AutoencoderKL.from_pretrained(
            config.vae_fp16fix_id, torch_dtype=torch.float16
        ).to(device).eval()
        self.vae.requires_grad_(False)
        self.scaling_factor = self.vae.config.scaling_factor
        self.device = device

    @torch.no_grad()
    def encode_batch(self, img_batch_neg1_to_1: torch.Tensor) -> torch.Tensor:
        """img_batch: (B, 3, H, W) in [-1, 1] -> (B, 4, H/8, W/8) latent."""
        img_batch = img_batch_neg1_to_1.to(self.device, dtype=torch.float16)
        dist = self.vae.encode(img_batch).latent_dist
        latent = (dist.sample() * self.scaling_factor).float()
        if torch.isnan(latent).any() or torch.isinf(latent).any():
            raise RuntimeError(
                "VAE produced NaN/Inf latents. Confirm vae_fp16fix_id points to "
                "'madebyollin/sdxl-vae-fp16-fix', not the original SDXL VAE."
            )
        return latent
