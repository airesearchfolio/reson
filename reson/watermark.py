#!/usr/bin/env python3
"""
stage6/watermark.py -- the carrier and bit-injection logic.

Structured as an nn.Module, matching SERUM's watermark.py class shape,
so the whole watermark (carrier + settings) can be saved/loaded as one
object. UNLIKE SERUM's grid, this carrier is NOT trained by an
optimizer anywhere in this pipeline -- it is built once by
build_carrier() (smoothing + noise-blend fix) and then frozen. The
nn.Module wrapper is here for clean structure and save/load
convenience, not because gradients ever flow into self.grid.

Core idea:

    bit 0 -> initial latent = noise - alpha * carrier
    bit 1 -> initial latent = noise + alpha * carrier

CARRIER CONSTRUCTION:
    1. Start from raw Gaussian noise.
    2. Smooth it with 8x repeated 5x5 average pooling (broad
       low-frequency shape).
    3. Mean-center, variance-normalize.
    4. Blend with fresh white noise at ratio `blend` (1.0 = original/
       unfixed carrier -- causes visible grid artifacts at production
       alpha; ~0.40-0.50 = validated fix, clean images, real
       decodability tradeoff measured across this whole project).
    5. Renormalize back to the smoothed carrier's original energy.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class Watermark(nn.Module):
    """Carrier + injection, as a single saveable/loadable module."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        latent_shape = (
            config.watermark.latent_channels,
            config.watermark.latent_h,
            config.watermark.latent_w,
        )
        # Registered as a buffer, not nn.Parameter -- this carrier is
        # frozen by design (see module docstring). Using a buffer keeps
        # it correctly saved/loaded/moved-to-device via state_dict(),
        # exactly like a parameter would be, without implying it's
        # trainable.
        self.register_buffer("grid", torch.zeros(latent_shape))
        self._built = False

    # ------------------------------------------------------------------
    # Carrier construction
    # ------------------------------------------------------------------

    def build(self, force: bool = False) -> "Watermark":
        """Builds (or loads a cached) carrier into self.grid."""
        path = self.config.carrier_path
        blend = self.config.watermark.carrier_blend
        c, h, w = self.grid.shape

        if not force and path.exists():
            arr = np.load(path)
            if arr.shape == (c, h, w):
                self.grid.copy_(torch.tensor(arr, dtype=torch.float32))
                self._built = True
                print(f"Loaded existing carrier (blend={blend:.2f}): {path}")
                return self

        gen = torch.Generator(device="cpu").manual_seed(self.config.seed + 55555)
        carrier = torch.randn(1, c, h, w, generator=gen)

        for _ in range(8):
            carrier = F.avg_pool2d(carrier, kernel_size=5, stride=1, padding=2)

        carrier = carrier - carrier.mean(dim=(1, 2, 3), keepdim=True)
        carrier = carrier / (carrier.std() + 1e-8)
        smoothed_norm = carrier.norm()

        if blend < 1.0:
            noise_gen = torch.Generator(device="cpu").manual_seed(self.config.seed + 77777)
            noise = torch.randn(carrier.shape, generator=noise_gen)
            noise = noise * (smoothed_norm / (noise.norm() + 1e-12))

            mixed = float(blend) * carrier + (1.0 - float(blend)) * noise
            mixed = mixed * (smoothed_norm / (mixed.norm() + 1e-12))
            carrier = mixed
            print(f"Applied carrier fix: spatial-domain blend={blend:.2f}")

        self.grid.copy_(carrier[0])
        self._built = True

        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.grid.cpu().numpy())
        print(f"Built and saved carrier: blend={blend:.2f} -> {path}")
        return self

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    def inject(self, bit: int, seed: int, device: str, alpha: Optional[float] = None) -> torch.Tensor:
        """Returns a single (1, C, H, W) initial latent with the bit
        encoded, ready to feed directly into a diffusion pipeline as
        `latents=`."""
        if not self._built:
            raise RuntimeError("Call watermark.build() before inject().")
        if alpha is None:
            alpha = self.config.watermark.noise_mix_alpha

        generator = torch.Generator(device=device).manual_seed(int(seed))
        noise = torch.randn(1, *self.grid.shape, generator=generator, device=device, dtype=torch.float32)

        carrier = self.grid.to(device=device, dtype=torch.float32)[None, ...]
        sign = 1.0 if int(bit) == 1 else -1.0

        latent = (noise + sign * float(alpha) * carrier) / math.sqrt(1.0 + float(alpha) ** 2)
        return latent

    def null_latent(self, seed: int, device: str) -> torch.Tensor:
        """alpha=0 control -- plain noise, no carrier at all."""
        generator = torch.Generator(device=device).manual_seed(int(seed))
        return torch.randn(1, *self.grid.shape, generator=generator, device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Save / load (mirrors SERUM's Watermark.save()/load())
    # ------------------------------------------------------------------

    def save(self, filename: Optional[str] = None) -> str:
        if filename is None:
            filename = f"watermark_blend{self.config.watermark.carrier_blend:.2f}.pt"
        path = self.config.checkpoint_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Watermark saved to: {path}")
        return str(path)

    @staticmethod
    def load(config: Config, filename: Optional[str] = None) -> "Watermark":
        if filename is None:
            filename = f"watermark_blend{config.watermark.carrier_blend:.2f}.pt"
        path = config.checkpoint_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"No watermark found at {path}")

        model = Watermark(config)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model._built = True
        print(f"Watermark loaded from: {path}")
        return model
