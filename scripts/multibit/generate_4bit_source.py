#!/usr/bin/env python3
"""
RESON SD2.1 4-bit payload experiment.

Design:
- Same SD2.1 source, prompt split, alpha=.155, beta=.40, 50 steps, CFG=7.5.
- Same clean latent z for clean/WM pair.
- Extracts the canonical RESON 1-bit carrier from the existing Watermark implementation,
  then deterministically constructs 4 orthogonal keyed carriers from spatial transforms.
- Aggregate payload carrier is energy-normalized, so increasing from 1 to 4 bits does
  NOT multiply watermark energy.
- Balanced 4-bit messages are assigned deterministically from sample_id (0..15 cycle).
- Resume safe. Saves clean and WM together.
- Use --max-samples 200 FIRST as the required visual-quality gate.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT="reson"
PROMPTS=f"{PROJECT}/prompt_split.csv"
OUT="workspace/paper_experiments/reson_sd21_4bit_alpha0155_beta040_10k"
MODEL="sd2-community/stable-diffusion-2-1-base"

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--project-root",default=PROJECT)
    p.add_argument("--prompt-csv",default=PROMPTS)
    p.add_argument("--out-dir",default=OUT)
    p.add_argument("--gpu",type=int,default=5)
    p.add_argument("--alpha",type=float,default=.155)
    p.add_argument("--beta",type=float,default=.40)
    p.add_argument("--steps",type=int,default=50)
    p.add_argument("--guidance",type=float,default=7.5)
    p.add_argument("--max-samples",type=int,default=200,
                   help="Default 200 for visual gate. Use 10000 only after visual approval.")
    p.add_argument("--overwrite",action="store_true")
    return p.parse_args()

def get(row,names,default=None):
    for n in names:
        if n in row and pd.notna(row[n]): return row[n]
    return default

def bits4(sample_id):
    # Exact balance over 10K: each 16-code message appears 625 times.
    m=int(sample_id)%16
    return [(m >> j) & 1 for j in range(4)]

def canonical_carrier(wm, seed, device, alpha):
    """Recover +1 canonical carrier algebraically from canonical project API."""
    z=wm.null_latent(seed=seed,device=device).float()
    zp=wm.inject(bit=1,seed=seed,device=device,alpha=alpha).float()
    c=(np.sqrt(1.0+alpha*alpha)*zp-z)/alpha
    return z,c

def make_four_carriers(c):
    """
    Build 4 deterministic transformed carriers, Gram-Schmidt orthogonalize,
    and match each carrier's L2 norm to the canonical carrier.
    """
    candidates=[
        c,
        torch.roll(c,shifts=(11,17),dims=(-2,-1)),
        torch.flip(torch.roll(c,shifts=(23,7),dims=(-2,-1)),dims=(-1,)),
        torch.flip(torch.roll(c,shifts=(5,29),dims=(-2,-1)),dims=(-2,)),
    ]
    target=torch.linalg.vector_norm(c.reshape(-1))
    basis=[]
    for x in candidates:
        v=x.reshape(-1).clone()
        for q in basis:
            v=v-torch.dot(v,q)*q
        nv=torch.linalg.vector_norm(v)
        if nv < 1e-8:
            raise RuntimeError("Degenerate carrier during Gram-Schmidt")
        q=v/nv
        basis.append(q)
    return [q.reshape_as(c)*target for q in basis]

def inject4(z, carriers, bits, alpha):
    signs=torch.tensor([2*b-1 for b in bits],device=z.device,dtype=z.dtype)
    u=sum(signs[j]*carriers[j].to(z.dtype) for j in range(4))/2.0  # /sqrt(4)
    # Finite-sample correction: keep aggregate carrier energy exactly matched.
    target=torch.linalg.vector_norm(carriers[0].to(z.dtype).reshape(-1))
    un=torch.linalg.vector_norm(u.reshape(-1))
    u=u*(target/un)
    return (z+alpha*u)/np.sqrt(1.0+alpha*alpha)

def main():
    a=args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    device="cuda:0"
    sys.path.insert(0,str(Path(a.project_root).resolve()))
    from config import Config
    from watermark import Watermark
    from diffusers import StableDiffusionPipeline

    cfg=Config()
    cfg.watermark.noise_mix_alpha=a.alpha
    cfg.watermark.carrier_blend=a.beta
    wm=Watermark(cfg).build()

    df=pd.read_csv(a.prompt_csv)
    if len(df)!=10000: raise RuntimeError(f"Expected 10000 prompt rows, got {len(df)}")
    df=df.iloc[:min(a.max_samples,len(df))].copy()

    out=Path(a.out_dir); img=out/"images"; img.mkdir(parents=True,exist_ok=True)
    manifest=out/"manifest.csv"
    config_path=out/"experiment_config.json"

    pipe=StableDiffusionPipeline.from_pretrained(
        MODEL,torch_dtype=torch.float16,safety_checker=None,requires_safety_checker=False
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # Carrier is key-derived and fixed across samples; derive once using a fixed extraction seed.
    _,c=canonical_carrier(wm,20260920,device,a.alpha)
    carriers=make_four_carriers(c)
    gram=np.array([[float(torch.dot(x.reshape(-1),y.reshape(-1))/
                          (torch.linalg.vector_norm(x.reshape(-1))*torch.linalg.vector_norm(y.reshape(-1))))
                    for y in carriers] for x in carriers])
    print("4-carrier cosine Gram matrix:\n",np.round(gram,6))

    rows=[]
    existing={}
    if manifest.exists() and not a.overwrite:
        old=pd.read_csv(manifest)
        existing={int(r.sample_id):r.to_dict() for _,r in old.iterrows()}

    for idx,r in tqdm(df.iterrows(),total=len(df),desc="RESON-4bit"):
        d=r.to_dict()
        sid=int(get(d,["sample_id","id","index"],idx))
        prompt=str(get(d,["prompt","text","caption"]))
        seed=int(get(d,["generation_seed","seed","gen_seed"]))
        split=str(get(d,["split","partition","subset"],""))
        bits=bits4(sid)
        clean=img/f"{sid:05d}.png"
        wmp=img/f"{sid:05d}_wm.png"

        if clean.exists() and wmp.exists() and not a.overwrite:
            pass
        else:
            z=wm.null_latent(seed=seed,device=device).float()
            zw=inject4(z,carriers,bits,a.alpha)
            with torch.inference_mode():
                ci=pipe(prompt=prompt,latents=z.half(),num_inference_steps=a.steps,
                        guidance_scale=a.guidance,height=512,width=512).images[0]
                wi=pipe(prompt=prompt,latents=zw.half(),num_inference_steps=a.steps,
                        guidance_scale=a.guidance,height=512,width=512).images[0]
            ct=clean.with_name(clean.stem+".tmp.png"); wt=wmp.with_name(wmp.stem+".tmp.png")
            ci.save(ct); wi.save(wt); os.replace(ct,clean); os.replace(wt,wmp)

        row={k:v for k,v in d.items()}
        row.update({
            "sample_id":sid,"prompt":prompt,"generation_seed":seed,"split":split,
            "bit0":bits[0],"bit1":bits[1],"bit2":bits[2],"bit3":bits[3],
            "message_int":sum(bits[j]<<j for j in range(4)),
            "clean_image_path":str(clean),"wm_image_path":str(wmp),
            "source_model":MODEL,"alpha":a.alpha,"beta":a.beta,
            "payload_bits":4,"steps":a.steps,"guidance":a.guidance,
            "carrier_scheme":"canonical carrier + deterministic transforms + Gram-Schmidt; aggregate /sqrt(K) and energy matched",
        })
        existing[sid]=row

        # crash-safe refresh every sample
        pd.DataFrame([existing[k] for k in sorted(existing)]).to_csv(manifest,index=False)

    cfgout={
        "model":MODEL,"alpha":a.alpha,"beta":a.beta,"payload_bits":4,
        "steps":a.steps,"guidance":a.guidance,"n_requested":len(df),
        "prompt_csv":a.prompt_csv,"carrier_cosine_gram":gram.tolist(),
        "message_assignment":"sample_id mod 16; little-endian bit0..bit3",
        "energy_rule":"aggregate signed carrier / sqrt(4), then exact L2 match to one canonical carrier",
        "visual_gate_required": True
    }
    config_path.write_text(json.dumps(cfgout,indent=2))
    print("\nDONE")
    print("Manifest:",manifest)
    print("Images:",img)
    print("IMPORTANT: visually inspect clean/WM pairs before --max-samples 10000.")

if __name__=="__main__":
    main()
