#!/usr/bin/env python3
"""
stage6/diffusion.py -- pipeline loaders and generation functions for
SDXL (source generation), FLUX and SDXL-family img2img (regeneration
attacks / multi-hop lineage).
"""

from __future__ import annotations
import gc
import hashlib
from typing import List, Tuple

import torch
from PIL import Image

from config import Config


def cleanup_pipe(pipe) -> None:
    try:
        del pipe
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stable_seed(*parts) -> int:
    s = "|".join(map(str, parts))
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_sdxl_txt2img(config: Config):
    from diffusers import StableDiffusionXLPipeline
    kw = dict(torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(config.sdxl_model_id, **kw)
    except Exception:
        kw.pop("variant", None)
        pipe = StableDiffusionXLPipeline.from_pretrained(config.sdxl_model_id, **kw)
    pipe.to(config.get_device())
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_sdxl_img2img(model_id: str, device: str):
    from diffusers import AutoPipelineForImage2Image
    kw = dict(torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    try:
        pipe = AutoPipelineForImage2Image.from_pretrained(model_id, **kw)
    except Exception:
        kw.pop("variant", None)
        pipe = AutoPipelineForImage2Image.from_pretrained(model_id, **kw)
    pipe.to(device)
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_flux_img2img(model_id: str, device: str):
    from diffusers import FluxImg2ImgPipeline
    pipe = FluxImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.to(device)
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ---------------------------------------------------------------------------
# Generation (source, from an injected latent)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_from_latent(pipe, latent: torch.Tensor, prompt: str, config: Config) -> Image.Image:
    return pipe(
        prompt=str(prompt),
        latents=latent.half(),
        height=config.image_size,
        width=config.image_size,
        num_inference_steps=config.generation_steps,
        guidance_scale=config.guidance,
    ).images[0].convert("RGB")


# ---------------------------------------------------------------------------
# Regeneration attacks (img2img)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_flux_img2img(pipe, init_image: Image.Image, prompt: str, strength: float, seed: int, config: Config) -> Image.Image:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return pipe(
        prompt=str(prompt), image=init_image, strength=float(strength),
        width=config.image_size, height=config.image_size,
        num_inference_steps=config.eval.eval_steps,
        guidance_scale=config.eval.flux_guidance,
        max_sequence_length=config.eval.flux_max_sequence_length,
        generator=g,
    ).images[0].convert("RGB")


@torch.inference_mode()
def run_sdxl_img2img(pipe, init_image: Image.Image, prompt: str, strength: float, seed: int, config: Config) -> Image.Image:
    device = config.get_device()
    g = torch.Generator(device=device).manual_seed(int(seed))
    return pipe(
        prompt=str(prompt), image=init_image, strength=float(strength),
        width=config.image_size, height=config.image_size,
        num_inference_steps=config.eval.eval_steps,
        guidance_scale=config.eval.sdxl_guidance,
        generator=g,
    ).images[0].convert("RGB")


def run_attack(pipe, family: str, init_image: Image.Image, prompt: str, strength: float, seed: int, config: Config) -> Image.Image:
    if family == "flux":
        return run_flux_img2img(pipe, init_image, prompt, strength, seed, config)
    return run_sdxl_img2img(pipe, init_image, prompt, strength, seed, config)


def load_attack_pipe(family: str, model_id: str, config: Config):
    device = config.get_device()
    if family == "flux":
        return load_flux_img2img(model_id, device)
    return load_sdxl_img2img(model_id, device)
