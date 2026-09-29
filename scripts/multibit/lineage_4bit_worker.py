#!/usr/bin/env python3
"""
RESON 4-bit held-out TEST lineage worker.
Uses the EXACT canonical regeneration chain/protocol used for the final 1-bit experiment:
G0 existing -> G1 FLUX -> G2 RealVisXL V5.0 -> G3 DreamShaperXL -> G4 FLUX.
No watermark reinsertion. No detector training. Resumable.
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

def resolve_path(root, value):
    p=Path(str(value))
    if p.is_absolute(): return p
    return root/p

def atomic(img,p):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(".tmp.png"); img.save(q); os.replace(q,p)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default="reson")
    ap.add_argument("--data-root",default="workspace/paper_experiments/exp_16_coco_fid/sd21_4bit_alpha0155_beta040_n10000")
    ap.add_argument("--manifest",default=None)
    ap.add_argument("--out",default=None)
    ap.add_argument("--expected-n",type=int,default=1000)
    ap.add_argument("--base-seed",type=int,default=20260903)
    ap.add_argument("--strength",type=float,default=.50)
    a=ap.parse_args()

    root=Path(a.data_root)
    mf=Path(a.manifest) if a.manifest else root/"manifest_final_10000.csv"
    out=Path(a.out) if a.out else root/"final_lineage_4bit"
    out.mkdir(parents=True,exist_ok=True)

    df=pd.read_csv(mf)
    df=df[df["split"].astype(str).str.lower().str.strip()=="test"].copy().sort_values("sample_id")
    if len(df)!=a.expected_n: raise RuntimeError(f"Expected {a.expected_n} test rows, got {len(df)}")
    if df.sample_id.duplicated().any(): raise RuntimeError("Duplicate sample_id")
    required={"sample_id","prompt","clean_path","wm4_path","bit0","bit1","bit2","bit3","message_int"}
    missing=required-set(df.columns)
    if missing: raise RuntimeError(f"Manifest missing columns: {sorted(missing)}")

    # Resolve and validate the exact G0 pairs from the 4-bit manifest.
    g0c={}; g0w={}
    for _,r in df.iterrows():
        sid=int(r.sample_id)
        cp=resolve_path(root,r.clean_path); wp=resolve_path(root,r.wm4_path)
        if not cp.exists(): raise FileNotFoundError(f"Missing clean G0: {cp}")
        if not wp.exists(): raise FileNotFoundError(f"Missing 4-bit G0: {wp}")
        g0c[sid]=cp; g0w[sid]=wp

    sys.path.insert(0,a.project_root)
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
      "experiment":"RESON 4-bit held-out multi-hop lineage",
      "manifest":str(mf),"n":len(df),"base_seed":a.base_seed,"strength":a.strength,
      "chain":chain,"watermark_reinserted_after_g0":False,
      "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
      "protocol_match":"final canonical 1-bit lineage worker"
    },indent=2))

    prevc,prevw=g0c,g0w
    allrows=[]
    for depth,family,model_id,prefix in chain:
        print(f"\n=== {depth.upper()} | {model_id} | n={len(df)} ===",flush=True)
        od=out/depth; od.mkdir(exist_ok=True)
        pipe=D.load_attack_pipe(family,model_id,cfg)
        nc={}; nw={}; rows=[]
        for _,r in tqdm(df.iterrows(),total=len(df),desc=depth):
            sid=int(r.sample_id); prompt=str(r.prompt)
            cp=od/f"{prefix}_regen_{sid:05d}.png"
            wp=od/f"{prefix}_regen_{sid:05d}_wm.png"
            seed=stable_seed(a.base_seed,"evallineage",sid,depth,a.strength)
            if not cp.exists():
                ci=Image.open(prevc[sid]).convert("RGB")
                atomic(D.run_attack(pipe,family,ci,prompt,a.strength,seed,cfg),cp)
            if not wp.exists():
                wi=Image.open(prevw[sid]).convert("RGB")
                atomic(D.run_attack(pipe,family,wi,prompt,a.strength,seed,cfg),wp)
            nc[sid]=cp; nw[sid]=wp
            rows.append({
              "sample_id":sid,"split":"test","depth":depth,"model_id":model_id,
              "prompt":prompt,"transition_seed":seed,"strength":a.strength,
              "clean_path":str(cp),"wm_path":str(wp),
              "message_int":int(r.message_int),
              "bit0":int(r.bit0),"bit1":int(r.bit1),"bit2":int(r.bit2),"bit3":int(r.bit3)
            })
        pd.DataFrame(rows).to_csv(out/f"manifest_{depth}.csv",index=False)
        allrows.extend(rows); prevc,prevw=nc,nw
        del pipe; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    pd.DataFrame(allrows).to_csv(out/"manifest_g1_g4.csv",index=False)
    print(f"\nDONE: {len(df)} held-out pairs x G1-G4 -> {out}",flush=True)

if __name__=="__main__":
    main()
