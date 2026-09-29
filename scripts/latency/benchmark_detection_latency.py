#!/usr/bin/env python3
# ===== START: RESON end-to-end detection latency benchmark =====
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torchvision.transforms.functional import pil_to_tensor
from diffusers import AutoencoderKL

ROOT=Path("workspace/reson_sd21_canonical_10k")
CKPT=ROOT/"final_detector/detector_best.pt"
CSV=ROOT/"final_detector/test_pairs.csv"
VAE_ID="sd2-community/stable-diffusion-2-1-base"

class LatentJointDecoder(nn.Module):
    """
    Exact architecture implied by the final RESON checkpoint keys:
      conv_layers: Conv-BN-ReLU-Pool x4, channels 4->32->64->128->256
      heads: 4096->512->1 for presence and bit prediction.

    For a 4x64x64 SD2.1 latent, four 2x2 pooling operations produce
    256x4x4 = 4096 features.
    """
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),   # 0
            nn.BatchNorm2d(32),                           # 1
            nn.ReLU(inplace=True),                        # 2
            nn.MaxPool2d(2),                              # 3

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 4
            nn.BatchNorm2d(64),                           # 5
            nn.ReLU(inplace=True),                        # 6
            nn.MaxPool2d(2),                              # 7

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 8
            nn.BatchNorm2d(128),                          # 9
            nn.ReLU(inplace=True),                        # 10
            nn.MaxPool2d(2),                              # 11

            nn.Conv2d(128, 256, kernel_size=3, padding=1),# 12
            nn.BatchNorm2d(256),                          # 13
            nn.ReLU(inplace=True),                        # 14
            nn.MaxPool2d(2),                              # 15
        )

        self.presence_fc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

        self.bit_fc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, z):
        h = self.conv_layers(z).flatten(1)
        return self.presence_fc(h), self.bit_fc(h)

def load_decoder(path,dev):
    obj=torch.load(path,map_location="cpu")
    if isinstance(obj,dict):
        for k in ("model_state_dict","state_dict","decoder_state_dict","model","decoder"):
            if k in obj and isinstance(obj[k],dict): obj=obj[k]; break
    sd={}
    for k,v in obj.items():
        for pre in ("module.","model.","decoder."):
            if k.startswith(pre): k=k[len(pre):]
        sd[k]=v
    m=LatentJointDecoder()
    try: m.load_state_dict(sd,strict=True)
    except RuntimeError as e:
        print("CHECKPOINT ARCHITECTURE MISMATCH.\nKeys:",*sd.keys(),sep="\n")
        raise e
    return m.to(dev).eval()

def path_column(df):
    for name in ("wm_path","watermarked_path","image_wm","wm_image","image_path","path","clean_path"):
        for c in df.columns:
            if c.lower()==name: return c
    for c in df.columns:
        if "path" in c.lower() or "image" in c.lower():
            if any(Path(str(x)).is_file() for x in df[c].dropna().head(20)): return c
    raise RuntimeError(f"No image path column. Columns={list(df.columns)}")

def prep(p):
    im=Image.open(p).convert("RGB")
    if im.size!=(512,512): im=im.resize((512,512),Image.Resampling.LANCZOS)
    return pil_to_tensor(im).float().div(255).mul(2).sub(1).unsqueeze(0)

@torch.inference_mode()
def run(xcpu,vae,dec,dev,sf):
    x=xcpu.to(dev)
    z=vae.encode(x).latent_dist.sample()*sf
    return dec(z)

def sync(dev): torch.cuda.synchronize(dev)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gpu",type=int,default=7)
    ap.add_argument("--n",type=int,default=200)
    ap.add_argument("--warmup",type=int,default=30)
    ap.add_argument("--checkpoint",type=Path,default=CKPT)
    ap.add_argument("--test_csv",type=Path,default=CSV)
    ap.add_argument("--output",type=Path,default=ROOT/"final_detector/detection_latency_benchmark.json")
    a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    dev=torch.device(f"cuda:{a.gpu}"); torch.cuda.set_device(dev)
    print("GPU:",torch.cuda.get_device_name(dev))
    print("Timing: CPU tensor transfer + SD2.1 VAE encode + RESON decoder; disk I/O excluded; batch=1")
    vae=AutoencoderKL.from_pretrained(VAE_ID,subfolder="vae",torch_dtype=torch.float32).to(dev).eval()
    dec=load_decoder(a.checkpoint,dev); sf=float(vae.config.scaling_factor)
    df=pd.read_csv(a.test_csv); col=path_column(df); print("Image column:",col)
    paths=list(dict.fromkeys(Path(str(x)) for x in df[col].dropna() if Path(str(x)).is_file()))
    need=max(a.n,a.warmup)
    if len(paths)<need: raise RuntimeError(f"Only {len(paths)} valid images; need {need}")
    xs=[prep(x) for x in paths[:need]]
    for i in range(a.warmup): run(xs[i%len(xs)],vae,dec,dev,sf)
    sync(dev)
    ms=[]
    for i in range(a.n):
        sync(dev); t=time.perf_counter()
        run(xs[i%len(xs)],vae,dec,dev,sf)
        sync(dev); ms.append((time.perf_counter()-t)*1000)
    z=np.asarray(ms)
    r={"gpu":torch.cuda.get_device_name(dev),"batch_size":1,"n":a.n,
       "mean_ms_per_image":float(z.mean()),"std_ms":float(z.std(ddof=1)),
       "median_ms_per_image":float(np.median(z)),"p95_ms_per_image":float(np.percentile(z,95)),
       "images_per_second":float(1000/z.mean()),
       "timed_pipeline":"CPU tensor transfer + SD2.1 VAE encode + RESON decoder",
       "disk_io_included":False}
    print(json.dumps(r,indent=2))
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(r,indent=2))
    print("Saved:",a.output)
if __name__=="__main__": main()
# ===== END: RESON end-to-end detection latency benchmark =====
