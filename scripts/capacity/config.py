from pathlib import Path
from dataclasses import dataclass, field
import torch

@dataclass
class WatermarkConfig:
    n_identity_carriers:int=24
    n_payload_bits:int=32
    alpha_payload:float=0.155
    latent_channels:int=4
    latent_h:int=64
    latent_w:int=64

@dataclass
class TrainingConfig:
    epochs:int=30
    batch_size:int=32
    lr:float=5e-5
    presence_weight:float=1.0
    payload_weight:float=1.0
    margin:float=1.5

@dataclass
class Config:
    root:Path=Path("workspace/8/reson_payload_capacity")
    joint_root:Path=Path("workspace/8/reson_multiuser_32bit")
    seed:int=20260912
    watermark:WatermarkConfig=field(default_factory=WatermarkConfig)
    training:TrainingConfig=field(default_factory=TrainingConfig)
    image_size:int=512
    generation_steps:int=30
    guidance:float=7.0
    sdxl_model_id:str="stabilityai/stable-diffusion-xl-base-1.0"
    vae_fp16fix_id:str="madebyollin/sdxl-vae-fp16-fix"

    @property
    def joint_manifest(self): return self.joint_root/"source_images"/"generated_manifest.csv"
    @property
    def joint_carrier_bank(self): return self.joint_root/"carrier_bank_id24_payload32_blend0.40.npy"
    def exp_dir(self,k): return self.root/f"k{k:02d}"
    def source_dir(self,k): return self.exp_dir(k)/"source_images"
    def source_manifest(self,k): return self.source_dir(k)/"generated_manifest.csv"
    def checkpoint_dir(self,k): return self.exp_dir(k)/"checkpoints"
    def eval_dir(self,k): return self.exp_dir(k)/"evaluation"
    def get_device(self): return "cuda" if torch.cuda.is_available() else "cpu"
    def make_dirs(self,k=None):
        self.root.mkdir(parents=True,exist_ok=True)
        if k is not None:
            for p in [self.exp_dir(k),self.source_dir(k),self.checkpoint_dir(k),self.eval_dir(k)]:
                p.mkdir(parents=True,exist_ok=True)
