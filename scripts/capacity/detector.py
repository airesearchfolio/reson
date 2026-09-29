import torch, torch.nn as nn
from diffusers import AutoencoderKL

class PayloadDecoder(nn.Module):
    def __init__(self,c,n_bits):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(c,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,1,1),nn.BatchNorm2d(256),nn.ReLU(),nn.MaxPool2d(2))
        dim=256*4*4
        self.bit_fc=nn.Sequential(nn.Linear(dim,512),nn.ReLU(),nn.Linear(512,n_bits))
        self.pres_fc=nn.Sequential(nn.Linear(dim,512),nn.ReLU(),nn.Linear(512,1))
    def forward(self,z):
        f=torch.flatten(self.conv(z),1); return self.bit_fc(f),self.pres_fc(f).squeeze(1)

class VAEEncoder:
    def __init__(self,cfg):
        self.device=cfg.get_device()
        self.vae=AutoencoderKL.from_pretrained(cfg.vae_fp16fix_id,torch_dtype=torch.float16).to(self.device).eval()
        self.vae.requires_grad_(False); self.sf=self.vae.config.scaling_factor
    @torch.no_grad()
    def encode_batch(self,x):
        x=x.to(self.device,dtype=torch.float16)
        return (self.vae.encode(x).latent_dist.sample()*self.sf).float()

def margin_presence_loss(logit,pres,margin):
    return torch.clamp(margin-logit*(2*pres-1),min=0).mean()
