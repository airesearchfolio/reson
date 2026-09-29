#!/usr/bin/env python3
import os, sys, gc, json, hashlib, argparse
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

CHAINS = {
 "order_a":[
  ("g1","sdxl","Lykon/dreamshaper-xl-1-0","dreamshaper_xl"),
  ("g2","sdxl","SG161222/RealVisXL_V5.0","realvis_xl"),
  ("g3","flux","black-forest-labs/FLUX.1-dev","flux"),
  ("g4","sdxl","RunDiffusion/Juggernaut-XL-v8","juggernaut_xl")],
 "order_b":[
  ("g1","sdxl","SG161222/RealVisXL_V5.0","realvis_xl"),
  ("g2","sdxl","RunDiffusion/Juggernaut-XL-v8","juggernaut_xl"),
  ("g3","sdxl","Lykon/dreamshaper-xl-1-0","dreamshaper_xl"),
  ("g4","flux","black-forest-labs/FLUX.1-dev","flux")],
 "order_c":[
  ("g1","sdxl","RunDiffusion/Juggernaut-XL-v8","juggernaut_xl"),
  ("g2","flux","black-forest-labs/FLUX.1-dev","flux"),
  ("g3","sdxl","SG161222/RealVisXL_V5.0","realvis_xl"),
  ("g4","sdxl","Lykon/dreamshaper-xl-1-0","dreamshaper_xl")],
 "order_d":[
  ("g1","sdxl","stabilityai/stable-diffusion-xl-base-1.0","sdxl_base"),
  ("g2","sdxl","Lykon/dreamshaper-xl-1-0","dreamshaper_xl"),
  ("g3","sdxl","RunDiffusion/Juggernaut-XL-v8","juggernaut_xl"),
  ("g4","flux","black-forest-labs/FLUX.1-dev","flux")]
}

def seed(base,*parts):
 h=hashlib.sha256("|".join(map(str,(base,)+parts)).encode()).digest()
 return int.from_bytes(h[:8],"big")%(2**31-1)

def g0(root,sid,wm):
 fn=f"{sid:05d}_wm.png" if wm else f"{sid:05d}.png"
 for p in [root/"images"/fn,root/fn]:
  if p.exists(): return p
 raise FileNotFoundError(f"Missing G0 sid={sid} wm={wm}")

def save(img,p):
 p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_suffix(".tmp.png"); img.save(q); os.replace(q,p)

def main():
 a=argparse.ArgumentParser()
 a.add_argument("--chain",choices=sorted(CHAINS),required=True)
 a.add_argument("--project-root",default="reson")
 a.add_argument("--data-root",default="workspace/reson_sd21_canonical_10k")
 a.add_argument("--out-root",default=None)
 a.add_argument("--manifest",default=None)
 a.add_argument("--expected-n",type=int,default=1000)
 a.add_argument("--base-seed",type=int,default=20260903)
 a.add_argument("--strength",type=float,default=.50)
 x=a.parse_args()

 root=Path(x.data_root)
 mf=Path(x.manifest) if x.manifest else root/"manifest.csv"
 outbase=Path(x.out_root) if x.out_root else root/"alternative_lineages"
 out=outbase/x.chain; out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(mf)
 df=df[df["split"].astype(str).str.lower().str.strip()=="test"].copy().sort_values("sample_id")
 if len(df)!=x.expected_n: raise RuntimeError(f"Expected {x.expected_n}, got {len(df)}")
 if df.sample_id.duplicated().any(): raise RuntimeError("Duplicate sample IDs")
 pc="prompt" if "prompt" in df else "caption"
 for s in df.sample_id.astype(int): g0(root,s,False); g0(root,s,True)

 sys.path.insert(0,x.project_root)
 import diffusion as D
 from config import Config
 cfg=Config(); chain=CHAINS[x.chain]
 (out/"run_config.json").write_text(json.dumps({
  "chain":chain,"strength":x.strength,"base_seed":x.base_seed,
  "n":len(df),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
  "watermark_reinserted_after_g0":False},indent=2))

 prevc={int(s):g0(root,int(s),False) for s in df.sample_id}
 prevw={int(s):g0(root,int(s),True) for s in df.sample_id}
 allrows=[]
 for depth,family,model,prefix in chain:
  print(f"\\n=== {x.chain} {depth.upper()} {model} ===",flush=True)
  od=out/depth; od.mkdir(exist_ok=True)
  pipe=D.load_attack_pipe(family,model,cfg)
  nc={}; nw={}; rows=[]
  for _,r in tqdm(df.iterrows(),total=len(df),desc=f"{x.chain}:{depth}"):
   sid=int(r.sample_id); prompt=str(r[pc])
   cp=od/f"{prefix}_regen_{sid:05d}.png"
   wp=od/f"{prefix}_regen_{sid:05d}_wm.png"
   # Same seed convention as canonical experiment; same seed for clean/WM.
   sd=seed(x.base_seed,"evallineage",sid,depth,x.strength)
   if not cp.exists():
    save(D.run_attack(pipe,family,Image.open(prevc[sid]).convert("RGB"),prompt,x.strength,sd,cfg),cp)
   if not wp.exists():
    save(D.run_attack(pipe,family,Image.open(prevw[sid]).convert("RGB"),prompt,x.strength,sd,cfg),wp)
   nc[sid]=cp; nw[sid]=wp
   rows.append(dict(sample_id=sid,split="test",chain=x.chain,depth=depth,
    family=family,model_id=model,prompt=prompt,transition_seed=sd,
    strength=x.strength,clean_path=str(cp),wm_path=str(wp)))
  pd.DataFrame(rows).to_csv(out/f"manifest_{depth}.csv",index=False)
  allrows += rows; prevc,prevw=nc,nw
  del pipe; gc.collect()
  if torch.cuda.is_available(): torch.cuda.empty_cache()
 pd.DataFrame(allrows).to_csv(out/"manifest_g1_g4.csv",index=False)
 print(f"DONE -> {out}",flush=True)

if __name__=="__main__": main()
