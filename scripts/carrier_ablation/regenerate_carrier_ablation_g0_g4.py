#!/usr/bin/env python3
"""Matched carrier-ablation lineage worker.
Uses explicit G0 clean_path/wm_path from the supplied test manifest.
Canonical chain: FLUX -> RealVisXL -> DreamShaperXL -> FLUX.
"""
import os, argparse, gc, hashlib, json, sys
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

def stable_seed(base,*parts):
    h=hashlib.sha256("|".join(map(str,(base,)+parts)).encode()).digest()
    return int.from_bytes(h[:8],"big")%(2**31-1)

def atomic(img,p):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(".tmp.png"); img.save(q); os.replace(q,p)

def col(df,*names):
    for n in names:
        if n in df.columns: return n
    raise RuntimeError(f"Missing one of {names}. Columns={list(df.columns)}")

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--project-root",default="reson")
    a.add_argument("--manifest",required=True)
    a.add_argument("--out",required=True)
    a.add_argument("--expected-n",type=int,default=300)
    a.add_argument("--base-seed",type=int,default=20260903)
    a.add_argument("--strength",type=float,default=.50)
    x=a.parse_args()

    out=Path(x.out); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(x.manifest).sort_values("sample_id")
    if "split" in df.columns:
        z=df[df["split"].astype(str).str.lower().str.strip()=="test"].copy()
        if len(z): df=z
    if len(df)!=x.expected_n or df.sample_id.nunique()!=x.expected_n:
        raise RuntimeError(f"Expected {x.expected_n} unique pairs, got {len(df)}/{df.sample_id.nunique()}")
    pc=col(df,"prompt","caption")
    cc=col(df,"clean_path","clean_image_path")
    wc=col(df,"wm_path","wm_image_path")
    for c in [cc,wc]:
        miss=[p for p in df[c].astype(str) if not Path(p).exists()]
        if miss: raise FileNotFoundError(f"{c}: {len(miss)} missing; first={miss[0]}")

    sys.path.insert(0,x.project_root)
    import diffusion as D
    from config import Config
    cfg=Config()
    chain=[
      ("g1","flux","black-forest-labs/FLUX.1-dev","flux"),
      ("g2","sdxl","SG161222/RealVisXL_V5.0","realvis_xl"),
      ("g3","sdxl","Lykon/dreamshaper-xl-1-0","dreamshaper_xl"),
      ("g4","flux","black-forest-labs/FLUX.1-dev","flux"),
    ]
    (out/"run_config.json").write_text(json.dumps({
      "manifest":str(Path(x.manifest).resolve()),"n":len(df),"strength":x.strength,
      "base_seed":x.base_seed,"chain":chain,
      "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES")
    },indent=2))

    prevc={int(r.sample_id):Path(getattr(r,cc)) for r in df.itertuples(index=False)}
    prevw={int(r.sample_id):Path(getattr(r,wc)) for r in df.itertuples(index=False)}
    allrows=[]
    for depth,family,model_id,prefix in chain:
        print(f"\n=== {depth.upper()} | {model_id} | n={len(df)} ===",flush=True)
        od=out/depth; od.mkdir(exist_ok=True)
        pipe=D.load_attack_pipe(family,model_id,cfg)
        nc={}; nw={}; rows=[]
        for r in tqdm(df.itertuples(index=False),total=len(df),desc=depth):
            sid=int(r.sample_id); prompt=str(getattr(r,pc))
            cp=od/f"{prefix}_regen_{sid:05d}.png"
            wp=od/f"{prefix}_regen_{sid:05d}_wm.png"
            sd=stable_seed(x.base_seed,"evallineage",sid,depth,x.strength)
            if not cp.exists():
                co=D.run_attack(pipe,family,Image.open(prevc[sid]).convert("RGB"),
                                prompt,x.strength,sd,cfg); atomic(co,cp)
            if not wp.exists():
                wo=D.run_attack(pipe,family,Image.open(prevw[sid]).convert("RGB"),
                                prompt,x.strength,sd,cfg); atomic(wo,wp)
            nc[sid]=cp; nw[sid]=wp
            rows.append(dict(sample_id=sid,split="test",depth=depth,model_id=model_id,
                             prompt=prompt,transition_seed=sd,strength=x.strength,
                             clean_path=str(cp),wm_path=str(wp)))
        pd.DataFrame(rows).to_csv(out/f"manifest_{depth}.csv",index=False)
        allrows.extend(rows); prevc,prevw=nc,nw
        del pipe; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    pd.DataFrame(allrows).to_csv(out/"manifest_g1_g4.csv",index=False)
    print("DONE ->",out)

if __name__=="__main__": main()
