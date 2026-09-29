#!/usr/bin/env python3
"""
Tree-Ring fair N=300 canonical RESON lineage generator.

Canonical subset is defined internally from:
  workspace/paper_experiments/
  reson_sd21_canonical_10k/manifest.csv

Selection:
  split == test -> sort sample_id -> first 300.

G0 is baseline-native and must already exist in the configured G0 directories.
This script DOES NOT regenerate G0. It matches G0 images to the canonical sample IDs,
writes generation_0.csv, then generates:
  G1 FLUX.1-dev -> G2 RealVisXL -> G3 DreamShaperXL -> G4 FLUX.1-dev

No --source-manifest is required.
"""

from __future__ import annotations
import argparse, gc, hashlib, re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm

BASELINE = "tree_ring"
CANONICAL_MANIFEST = Path("workspace/reson_sd21_canonical_10k/manifest.csv")
DEFAULT_G0_WM = Path("workspace/fair_sd21_main_table/tree_ring_g0/watermarked")
DEFAULT_G0_CLEAN = Path("workspace/fair_sd21_main_table/tree_ring_g0/no_watermark")
DEFAULT_OUT = Path("workspace/fair_sd21_main_table/tree_ring_lineage_n300")

FLUX_MODEL = "black-forest-labs/FLUX.1-dev"
REALVIS_MODEL = "SG161222/RealVisXL_V5.0"
DREAMSHAPER_MODEL = "Lykon/dreamshaper-xl-1-0"
STRENGTH = 0.50
STEPS = 30
SDXL_GUIDANCE = 7.5
FLUX_GUIDANCE = 3.5
WIDTH = HEIGHT = 512
SEED_BASE = 20260820
CHAIN = [
    (1, "flux", "flux", FLUX_MODEL),
    (2, "realvis", "sdxl", REALVIS_MODEL),
    (3, "dreamshaper", "sdxl", DREAMSHAPER_MODEL),
    (4, "flux", "flux", FLUX_MODEL),
]

def natural_key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]

def images(folder):
    exts={".png",".jpg",".jpeg",".webp"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts], key=natural_key)

def transition_seed(aid, depth):
    # Intentionally baseline-independent: same transition randomness for every baseline.
    s=f"{SEED_BASE}|fair_main_table|audit={int(aid)}|depth={int(depth)}|strength={STRENGTH:.2f}"
    return int(hashlib.sha256(s.encode()).hexdigest()[:8],16)&0x7fffffff

def cleanup(pipe):
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"): torch.cuda.ipc_collect()

def load_flux():
    from diffusers import FluxImg2ImgPipeline
    p=FluxImg2ImgPipeline.from_pretrained(FLUX_MODEL,torch_dtype=torch.bfloat16)
    p.to("cuda"); p.set_progress_bar_config(disable=True)
    return p

def load_sdxl(mid):
    from diffusers import AutoPipelineForImage2Image
    kw=dict(torch_dtype=torch.float16,use_safetensors=True,variant="fp16")
    try: p=AutoPipelineForImage2Image.from_pretrained(mid,**kw)
    except Exception:
        kw.pop("variant",None); p=AutoPipelineForImage2Image.from_pretrained(mid,**kw)
    p.to("cuda"); p.set_progress_bar_config(disable=True)
    return p

@torch.inference_mode()
def gen(pipe,family,img,prompt,seed):
    g=torch.Generator(device="cpu" if family=="flux" else "cuda").manual_seed(int(seed))
    kw=dict(prompt=str(prompt),image=img,strength=STRENGTH,width=WIDTH,height=HEIGHT,
            num_inference_steps=STEPS,generator=g)
    if family=="flux":
        kw.update(guidance_scale=FLUX_GUIDANCE,max_sequence_length=512)
    else:
        kw.update(guidance_scale=SDXL_GUIDANCE)
    return pipe(**kw).images[0].convert("RGB")

def canonical_subset(n):
    d=pd.read_csv(CANONICAL_MANIFEST)
    req={"split","sample_id","prompt"}
    if not req.issubset(d.columns): raise RuntimeError(f"Canonical manifest missing {req-set(d.columns)}")
    d=d[d["split"].astype(str).str.lower().str.strip().eq("test")].copy()
    d=d.sort_values("sample_id").head(n).reset_index(drop=True)
    if len(d)!=n: raise RuntimeError(f"Need {n} canonical test rows; found {len(d)}")
    if d.sample_id.duplicated().any(): raise RuntimeError("Duplicate canonical sample_id")
    return d

def map_g0(canon, clean_dir, wm_dir):
    # First try exact sample-id filenames (e.g. 09000.png / audit_09000.png).
    cfiles=images(clean_dir); wfiles=images(wm_dir)
    def idmap(fs):
        m={}
        for p in fs:
            nums=re.findall(r"\d+",p.stem)
            if nums: m[int(nums[-1])]=p
        return m
    cm,wm=idmap(cfiles),idmap(wfiles)
    ids=canon.sample_id.astype(int).tolist()
    if all(i in cm and i in wm for i in ids):
        return [(cm[i],wm[i]) for i in ids]
    # Baseline legacy G0 may be stored sequentially. Only accept exact N-or-more paired ordering.
    if len(cfiles)>=len(canon) and len(wfiles)>=len(canon):
        print("WARNING: G0 filenames do not encode canonical sample_id; using natural-order first N pairing.")
        return list(zip(cfiles[:len(canon)],wfiles[:len(canon)]))
    missing=[i for i in ids if i not in cm or i not in wm][:10]
    raise RuntimeError(f"Cannot match baseline G0 to canonical IDs. Missing examples: {missing}")

def make_g0(canon, clean_dir, wm_dir, out):
    pairs=map_g0(canon,clean_dir,wm_dir)
    rows=[]
    for r,(cp,wp) in zip(canon.itertuples(),pairs):
        rows.append(dict(audit_id=int(r.sample_id),sample_id=int(r.sample_id),prompt=str(r.prompt),
                         baseline=BASELINE,depth=0,generator="source",model_id="baseline_native_sd21",
                         strength=0.0,attack_seed=np.nan,
                         wm_image_path=str(wp.resolve()),alpha0_image_path=str(cp.resolve())))
    x=pd.DataFrame(rows)
    mp=out/"manifests"/"generation_0.csv"; mp.parent.mkdir(parents=True,exist_ok=True); x.to_csv(mp,index=False)
    print("G0 pointer manifest:",mp,"rows:",len(x))
    return x

def regen(prev,out,depth,name,family,mid):
    wd=out/f"generation_{depth}_{name}"/"watermarked"
    cd=out/f"generation_{depth}_{name}"/"alpha0"
    wd.mkdir(parents=True,exist_ok=True); cd.mkdir(parents=True,exist_ok=True)
    pipe=load_flux() if family=="flux" else load_sdxl(mid)
    rows=[]
    for r in tqdm(prev.itertuples(),total=len(prev),desc=f"{BASELINE} G{depth} {name}"):
        aid=int(r.audit_id); seed=transition_seed(aid,depth)
        wp=wd/f"audit_{aid:05d}.png"; cp=cd/f"audit_{aid:05d}.png"
        if not wp.exists():
            with Image.open(r.wm_image_path) as im: init=im.convert("RGB").resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS)
            gen(pipe,family,init,r.prompt,seed).save(wp)
        if not cp.exists():
            with Image.open(r.alpha0_image_path) as im: init=im.convert("RGB").resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS)
            gen(pipe,family,init,r.prompt,seed).save(cp)
        rows.append(dict(audit_id=aid,sample_id=aid,prompt=str(r.prompt),baseline=BASELINE,depth=depth,
                         generator=name,model_id=mid,strength=STRENGTH,attack_seed=seed,
                         wm_image_path=str(wp.resolve()),alpha0_image_path=str(cp.resolve())))
    cleanup(pipe)
    x=pd.DataFrame(rows); mp=out/"manifests"/f"generation_{depth}_{name}.csv"; x.to_csv(mp,index=False)
    print("Saved:",mp,"rows:",len(x)); return x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=300)
    ap.add_argument("--g0-clean",type=Path,default=DEFAULT_G0_CLEAN)
    ap.add_argument("--g0-wm",type=Path,default=DEFAULT_G0_WM)
    ap.add_argument("--out-root",type=Path,default=DEFAULT_OUT)
    a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    canon=canonical_subset(a.n)
    print("Canonical manifest:",CANONICAL_MANIFEST)
    print("Canonical test subset:",len(canon),"IDs",int(canon.sample_id.min()),"..",int(canon.sample_id.max()))
    print("G0 clean:",a.g0_clean); print("G0 WM:",a.g0_wm)
    a.out_root.mkdir(parents=True,exist_ok=True)
    prev=make_g0(canon,a.g0_clean,a.g0_wm,a.out_root)
    for depth,name,family,mid in CHAIN: prev=regen(prev,a.out_root,depth,name,family,mid)
    print("DONE:",a.out_root)

if __name__=="__main__": main()
