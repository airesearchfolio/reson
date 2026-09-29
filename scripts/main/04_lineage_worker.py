#!/usr/bin/env python3
"""Final RESON SD2.1 TEST-set regeneration worker (FIXED for src_large diffusion.py API).
G0 existing -> G1 FLUX -> G2 RealVisXL -> G3 DreamShaperXL -> G4 FLUX.
No detector. No G0 regeneration. Intended to be launched in independent GPU shards.
"""
import os
import argparse, gc, hashlib, json, sys
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

def stable_seed(base,*parts):
    h=hashlib.sha256("|".join(map(str,(base,)+parts)).encode()).digest()
    return int.from_bytes(h[:8],"big")%(2**31-1)

def g0(root,sid,wm):
    # canonical layout first; fallback allows images directly under root
    names=[root/"images"/(f"{sid:05d}_wm.png" if wm else f"{sid:05d}.png"),
           root/(f"{sid:05d}_wm.png" if wm else f"{sid:05d}.png")]
    for p in names:
        if p.exists(): return p
    raise FileNotFoundError(f"Missing G0 {'WM' if wm else 'clean'} for {sid}: {names}")

def atomic(img,p):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(".tmp.png"); img.save(q); os.replace(q,p)

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--project-root",default="reson")
    a.add_argument("--data-root",default="workspace/reson_sd21_canonical_10k")
    a.add_argument("--manifest",default=None)
    a.add_argument("--out",default=None)
    a.add_argument("--expected-n",type=int,default=1000)
    a.add_argument("--base-seed",type=int,default=20260903)
    a.add_argument("--strength",type=float,default=.50)
    args=a.parse_args()

    root=Path(args.data_root)
    out=Path(args.out) if args.out else root/"final_lineage_test"
    out.mkdir(parents=True,exist_ok=True)
    mf=Path(args.manifest) if args.manifest else root/"manifest.csv"

    df=pd.read_csv(mf)
    df=df[df["split"].astype(str).str.lower()=="test"].copy().sort_values("sample_id")
    if len(df)!=args.expected_n:
        raise RuntimeError(f"Expected {args.expected_n} test pairs, got {len(df)}")
    pc="prompt" if "prompt" in df.columns else ("caption" if "caption" in df.columns else None)
    if pc is None: raise RuntimeError(f"No prompt/caption column. Columns={list(df.columns)}")
    if df["sample_id"].duplicated().any(): raise RuntimeError("Duplicate sample_id values")

    # Fail early if canonical inputs are not all present.
    for sid in df.sample_id.astype(int):
        g0(root,sid,False); g0(root,sid,True)

    sys.path.insert(0,args.project_root)
    import diffusion as D
    from config import Config

    # IMPORTANT: use the exact project's Config so diffusion.py controls device,
    # img2img steps/guidance/etc exactly as in the established implementation.
    cfg=Config()

    chain=[
      ("g1","flux","black-forest-labs/FLUX.1-dev","flux"),
      ("g2","sdxl","SG161222/RealVisXL_V5.0","realvis_xl"),
      ("g3","sdxl","Lykon/dreamshaper-xl-1-0","dreamshaper_xl"),
      ("g4","flux","black-forest-labs/FLUX.1-dev","flux"),
    ]
    (out/"run_config.json").write_text(json.dumps({
        "data_root":str(root),"manifest":str(mf),"n":len(df),
        "base_seed":args.base_seed,"strength":args.strength,
        "chain":chain,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "api":"load_attack_pipe(family, model_id, config); run_attack(pipe, family, init_image, prompt, strength, seed, config)"
    },indent=2))

    prevc={int(s):g0(root,int(s),False) for s in df.sample_id}
    prevw={int(s):g0(root,int(s),True) for s in df.sample_id}
    allrows=[]

    for depth,family,model_id,prefix in chain:
        print(f"\n=== {depth.upper()} | {model_id} | n={len(df)} ===",flush=True)
        od=out/depth; od.mkdir(exist_ok=True)

        # Exact API reported from current diffusion.py
        pipe=D.load_attack_pipe(family,model_id,cfg)
        nc={}; nw={}; rows=[]

        for _,r in tqdm(df.iterrows(),total=len(df),desc=depth):
            sid=int(r["sample_id"]); prompt=str(r[pc])
            cp=od/f"{prefix}_regen_{sid:05d}.png"
            wp=od/f"{prefix}_regen_{sid:05d}_wm.png"

            # Same transition seed for clean and WM.
            seed=stable_seed(args.base_seed,"evallineage",sid,depth,args.strength)

            if not cp.exists():
                ci=Image.open(prevc[sid]).convert("RGB")
                co=D.run_attack(pipe,family,ci,prompt,args.strength,seed,cfg)
                atomic(co,cp)
            if not wp.exists():
                wi=Image.open(prevw[sid]).convert("RGB")
                wo=D.run_attack(pipe,family,wi,prompt,args.strength,seed,cfg)
                atomic(wo,wp)

            nc[sid]=cp; nw[sid]=wp
            rows.append(dict(sample_id=sid,split="test",depth=depth,
                model_id=model_id,prompt=prompt,transition_seed=seed,
                strength=args.strength,clean_path=str(cp),wm_path=str(wp)))

        pd.DataFrame(rows).to_csv(out/f"manifest_{depth}.csv",index=False)
        allrows.extend(rows)
        prevc,prevw=nc,nw

        del pipe
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    pd.DataFrame(allrows).to_csv(out/"manifest_g1_g4.csv",index=False)
    print(f"\nDONE: {len(df)} test pairs x 4 depths -> {out}",flush=True)

if __name__=="__main__":
    main()
