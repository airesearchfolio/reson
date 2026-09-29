#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd, torch
from tqdm import tqdm
from config import Config
from watermark import Watermark
import diffusion as D

N={"train":800,"val":100,"test":100}

def select_pilot(df,seed):
    parts=[]
    for j,(s,n) in enumerate(N.items()):
        x=df[df["split"].astype(str)==s].copy()
        x=x.sample(n=n,random_state=seed+7000+j).sort_values("prompt_id")
        parts.append(x)
    return pd.concat(parts,ignore_index=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bits",type=int,required=True,choices=[1,2,4,8,16,32])
    ap.add_argument("--shard-id",type=int,default=0)
    ap.add_argument("--num-shards",type=int,default=1)
    ap.add_argument("--merge-only",action="store_true")
    a=ap.parse_args(); k=a.bits
    cfg=Config(); cfg.make_dirs(k)

    if a.merge_only:
        parts=[]
        for i in range(a.num_shards):
            p=cfg.source_dir(k)/f"manifest_shard{i}.csv"
            if not p.exists(): raise FileNotFoundError(p)
            parts.append(pd.read_csv(p))
        df=pd.concat(parts,ignore_index=True).sort_values(["split","prompt_id"])
        df.to_csv(cfg.source_manifest(k),index=False)
        print("Saved:",cfg.source_manifest(k),"rows:",len(df)); print(df["split"].value_counts())
        return

    if not torch.cuda.is_available(): raise RuntimeError("CUDA required.")
    src=pd.read_csv(cfg.joint_manifest); df=select_pilot(src,cfg.seed)
    bits=[f"bit{i}" for i in range(k)]
    miss=[c for c in bits if c not in df.columns]
    if miss: raise KeyError(f"Joint manifest missing {miss}")

    # Each K uses exactly the same prompt IDs/splits; only payload dimensionality changes.
    df=df.iloc[a.shard_id::a.num_shards].copy().reset_index(drop=True)
    wm=Watermark(cfg,k).build(); device=cfg.get_device()
    pipe=None; rows=[]; gen=0
    try:
        for _,r in tqdm(df.iterrows(),total=len(df),desc=f"K={k} shard {a.shard_id}"):
            folder=cfg.source_dir(k)/str(r["split"]); folder.mkdir(parents=True,exist_ok=True)
            out=folder/f"row{int(r.prompt_id):05d}_wm.png"
            if not out.exists():
                if pipe is None: pipe=D.load_sdxl_txt2img(cfg)
                lat=wm.inject([int(r[c]) for c in bits],int(r.generation_seed),device)
                D.generate_from_latent(pipe,lat,str(r.prompt),cfg).save(out); gen+=1
            row={"prompt_id":int(r.prompt_id),"prompt":str(r.prompt),"split":str(r["split"]),
                 "generation_seed":int(r.generation_seed),"n_bits":k,
                 "alpha_payload":cfg.watermark.alpha_payload,
                 "clean_image_path":str(Path(str(r.clean_image_path)).resolve()),
                 "wm_image_path":str(out.resolve())}
            for c in bits: row[c]=int(r[c])
            rows.append(row)
    finally:
        if pipe is not None: D.cleanup_pipe(pipe)
    mp=cfg.source_dir(k)/f"manifest_shard{a.shard_id}.csv"
    pd.DataFrame(rows).to_csv(mp,index=False)
    print("Generated:",gen,"Saved:",mp)

if __name__=="__main__": main()
