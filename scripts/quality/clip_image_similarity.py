#!/usr/bin/env python3
"""
Compute paired clean-vs-watermarked CLIP image similarity for RESON.
Multi-GPU launcher: shards pairs across the supplied physical GPUs, then merges.
Default target is the main SD2.1 canonical test set (1000 pairs if available).

CLIP Score = mean cosine similarity between normalized CLIP image embeddings
for each matched clean/watermarked pair.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

DEFAULT_ROOT=Path("workspace/reson_sd21_canonical_10k")
DEFAULT_MODEL="openai/clip-vit-large-patch14"

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--manifest",type=Path,default=None,
                   help="Optional paired manifest. Defaults to root/manifest.csv.")
    p.add_argument("--out-dir",type=Path,default=None)
    p.add_argument("--model-id",default=DEFAULT_MODEL)
    p.add_argument("--split",default="test")
    p.add_argument("--n",type=int,default=1000)
    p.add_argument("--gpus",default="4,5,6,7")
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    p.add_argument("--shard-id",type=int,default=None,help=argparse.SUPPRESS)
    p.add_argument("--num-shards",type=int,default=1,help=argparse.SUPPRESS)
    return p.parse_args()

def pick(df,names):
    for c in names:
        if c in df.columns:return c
    return None

def load_pairs(a):
    mp=a.manifest if a.manifest else a.root/"manifest.csv"
    if not mp.exists(): raise FileNotFoundError(mp)
    m=pd.read_csv(mp)
    if "split" in m.columns and a.split:
        m=m[m["split"].astype(str).str.lower()==a.split.lower()].copy()
    sid=pick(m,["sample_id","sid","id"])
    clean=pick(m,["clean_path","alpha0_image_path","clean_image_path"])
    wm=pick(m,["wm_path","wm_image_path","watermarked_image_path"])
    if not sid or not clean or not wm:
        raise RuntimeError(f"Need sample_id + clean/wm paths. Columns={list(m.columns)}")
    m=m.drop_duplicates(sid).sort_values(sid).head(a.n).copy()
    if len(m)<a.n:
        raise RuntimeError(f"Requested n={a.n}, only {len(m)} matched rows available")
    z=pd.DataFrame({"sample_id":m[sid].astype(int),"clean_path":m[clean].astype(str),"wm_path":m[wm].astype(str)})
    def resolve(x):
        p=Path(x)
        return p if p.is_absolute() else a.root/p
    z["clean_path"]=z.clean_path.map(resolve); z["wm_path"]=z.wm_path.map(resolve)
    miss=[str(p) for c in ["clean_path","wm_path"] for p in z[c] if not Path(p).exists()]
    if miss: raise FileNotFoundError(f"Missing images, first={miss[0]}")
    return z.reset_index(drop=True)

class PairDS(Dataset):
    def __init__(self,df,processor):
        self.df=df.reset_index(drop=True); self.processor=processor
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        with Image.open(r.clean_path) as im: c=im.convert("RGB").copy()
        with Image.open(r.wm_path) as im: w=im.convert("RGB").copy()
        return c,w,int(r.sample_id)

def collate(batch,processor):
    c,w,sid=zip(*batch)
    ci=processor(images=list(c),return_tensors="pt")
    wi=processor(images=list(w),return_tensors="pt")
    return ci["pixel_values"],wi["pixel_values"],torch.tensor(sid,dtype=torch.long)

@torch.no_grad()
def worker(a):
    pairs=load_pairs(a).iloc[a.shard_id::a.num_shards].reset_index(drop=True)
    device=torch.device("cuda")
    proc=CLIPProcessor.from_pretrained(a.model_id)
    model=CLIPModel.from_pretrained(a.model_id,torch_dtype=torch.float32).to(device).eval()
    ds=PairDS(pairs,proc)
    dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,
                  pin_memory=True,persistent_workers=(a.num_workers>0),
                  collate_fn=lambda b: collate(b,proc))
    rows=[]
    for cp,wp,sid in tqdm(dl,desc=f"CLIP shard {a.shard_id}"):
        cp=cp.to(device,non_blocking=True); wp=wp.to(device,non_blocking=True)
        ce=model.get_image_features(pixel_values=cp)
        we=model.get_image_features(pixel_values=wp)
        ce=ce/ce.norm(dim=-1,keepdim=True); we=we/we.norm(dim=-1,keepdim=True)
        sim=(ce*we).sum(-1).cpu().numpy()
        for s,v in zip(sid.numpy(),sim):
            rows.append({"sample_id":int(s),"clip_similarity":float(v)})
    a.out_dir.mkdir(parents=True,exist_ok=True)
    fp=a.out_dir/f"clip_shard_{a.shard_id}.csv"
    pd.DataFrame(rows).to_csv(fp,index=False)
    print(f"[DONE] shard={a.shard_id} n={len(rows)} -> {fp}",flush=True)

def launch(a):
    a.out_dir = a.out_dir or a.root/"clip_quality_reson"
    a.out_dir.mkdir(parents=True,exist_ok=True)
    gpus=[x.strip() for x in a.gpus.split(",") if x.strip()]
    procs=[]; logs=[]
    for sh,gpu in enumerate(gpus):
        log=a.out_dir/f"clip_gpu{gpu}_shard{sh}.log"; logs.append(log)
        cmd=[sys.executable,str(Path(__file__).resolve()),
             "--worker","--shard-id",str(sh),"--num-shards",str(len(gpus)),
             "--root",str(a.root),"--out-dir",str(a.out_dir),"--model-id",a.model_id,
             "--split",a.split,"--n",str(a.n),"--batch-size",str(a.batch_size),
             "--num-workers",str(a.num_workers)]
        if a.manifest: cmd += ["--manifest",str(a.manifest)]
        env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=gpu
        f=open(log,"w")
        print(f"[LAUNCH] shard {sh} -> GPU {gpu}; log={log}",flush=True)
        procs.append((subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env),f))
    codes=[]
    for p,f in procs:
        codes.append(p.wait()); f.close()
    if any(c!=0 for c in codes): raise RuntimeError(f"CLIP worker failure codes={codes}; inspect {logs}")
    dfs=[pd.read_csv(a.out_dir/f"clip_shard_{i}.csv") for i in range(len(gpus))]
    z=pd.concat(dfs,ignore_index=True).sort_values("sample_id")
    if len(z)!=a.n or z.sample_id.nunique()!=a.n:
        raise RuntimeError(f"Merge mismatch rows={len(z)} unique={z.sample_id.nunique()} expected={a.n}")
    z.to_csv(a.out_dir/"clip_per_image.csv",index=False)
    vals=z.clip_similarity.to_numpy(float)
    summary={
        "n_pairs":int(len(vals)),
        "clip_model":a.model_id,
        "mean_clip_score":float(vals.mean()),
        "std_clip_score":float(vals.std(ddof=1)),
        "median_clip_score":float(np.median(vals)),
        "min_clip_score":float(vals.min()),
        "max_clip_score":float(vals.max()),
    }
    (a.out_dir/"clip_summary.json").write_text(json.dumps(summary,indent=2))
    pd.DataFrame([summary]).to_csv(a.out_dir/"clip_summary.csv",index=False)
    print("\nFINAL RESON CLIP SCORE")
    print(pd.DataFrame([summary]).to_string(index=False))
    print("Saved:",a.out_dir/"clip_summary.csv")

def main():
    a=parse()
    if a.out_dir is None:a.out_dir=a.root/"clip_quality_reson"
    if a.worker: worker(a)
    else: launch(a)
if __name__=="__main__": main()
