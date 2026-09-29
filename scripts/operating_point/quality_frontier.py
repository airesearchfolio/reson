#!/usr/bin/env python3
"""
Quality analysis for RESON beta=.40 alpha x steps controlled sweep.
Uses OpenCLIP ViT-L/14 (OpenAI) prompt-image similarity.

Outputs:
  quality_per_pair.csv
  quality_summary.csv
  quality_frontier.csv
  worst_pairs_by_condition.csv

This is an N=200 controlled operating-point diagnostic, not a replacement
for the diverse 10K quality benchmark.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--detector-summary",default=None)
    ap.add_argument("--out-dir",default=None)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--batch-size",type=int,default=16)
    ap.add_argument("--worst-k",type=int,default=20)
    args=ap.parse_args()

    import open_clip
    m=pd.read_csv(args.manifest)
    need={"prompt_id","prompt","bit","alpha","generation_steps","clean_image_path","wm_image_path"}
    miss=need-set(m.columns)
    if miss: raise ValueError(f"Missing columns: {sorted(miss)}")
    print("Condition counts:")
    print(m.groupby(["alpha","generation_steps"]).size())

    out=Path(args.out_dir) if args.out_dir else Path(args.manifest).parent/"screening_quality"
    out.mkdir(parents=True,exist_ok=True)

    model,_,prep=open_clip.create_model_and_transforms("ViT-L-14",pretrained="openai",device=args.device)
    tok=open_clip.get_tokenizer("ViT-L-14")
    model.eval()

    clean_clip=[]; wm_clip=[]; clean_wm_cos=[]
    with torch.inference_mode():
        for st in tqdm(range(0,len(m),args.batch_size),desc="CLIP quality"):
            q=m.iloc[st:st+args.batch_size]
            text=tok(q.prompt.tolist()).to(args.device)
            tf=model.encode_text(text); tf=tf/tf.norm(dim=-1,keepdim=True)

            feats=[]; sims=[]
            for col in ["clean_image_path","wm_image_path"]:
                ims=torch.stack([prep(Image.open(p).convert("RGB")) for p in q[col]]).to(args.device)
                f=model.encode_image(ims); f=f/f.norm(dim=-1,keepdim=True)
                feats.append(f)
                sims.append((f*tf).sum(-1).cpu().numpy())
            clean_clip.extend(sims[0]); wm_clip.extend(sims[1])
            clean_wm_cos.extend((feats[0]*feats[1]).sum(-1).cpu().numpy())

    d=m.copy()
    d["clean_clip"]=clean_clip
    d["wm_clip"]=wm_clip
    d["delta_clip"]=d.wm_clip-d.clean_clip
    d["relative_clip_pct"]=100*d.delta_clip/d.clean_clip.replace(0,np.nan)
    d["clean_wm_clip_cosine"]=clean_wm_cos
    d.to_csv(out/"quality_per_pair.csv",index=False)

    rows=[]
    for (a,s),g in d.groupby(["alpha","generation_steps"],sort=True):
        delta=g.delta_clip.to_numpy()
        rows.append({
            "alpha":a,"steps":int(s),"n_pairs":len(g),
            "clean_clip_mean":g.clean_clip.mean(),
            "wm_clip_mean":g.wm_clip.mean(),
            "delta_clip_mean":g.delta_clip.mean(),
            "delta_clip_median":g.delta_clip.median(),
            "relative_clip_pct_mean":g.relative_clip_pct.mean(),
            "wm_better_fraction":(g.delta_clip>0).mean(),
            "delta_lt_m005":(g.delta_clip<-.05).mean(),
            "delta_lt_m010":(g.delta_clip<-.10).mean(),
            "delta_lt_m015":(g.delta_clip<-.15).mean(),
            "wm_clip_lt_010":(g.wm_clip<.10).mean(),
            "wm_clip_lt_015":(g.wm_clip<.15).mean(),
            "review_flag":((g.delta_clip<-.10)|(g.wm_clip<.10)).mean(),
            "clean_wm_cosine_mean":g.clean_wm_clip_cosine.mean(),
            "delta_q01":np.quantile(delta,.01),
            "delta_q05":np.quantile(delta,.05),
            "delta_q10":np.quantile(delta,.10),
            "delta_q25":np.quantile(delta,.25),
        })
    sm=pd.DataFrame(rows).sort_values(["steps","alpha"])
    sm.to_csv(out/"quality_summary.csv",index=False)

    worst=(d.sort_values(["alpha","generation_steps","delta_clip"])
             .groupby(["alpha","generation_steps"],group_keys=False).head(args.worst_k))
    worst.to_csv(out/"worst_pairs_by_condition.csv",index=False)

    frontier=sm.copy()
    if args.detector_summary:
        det=pd.read_csv(args.detector_summary)
        frontier=frontier.merge(det,on=["alpha","steps"],how="left")
    frontier.to_csv(out/"quality_frontier.csv",index=False)

    cols=["alpha","steps","clean_clip_mean","wm_clip_mean","delta_clip_mean",
          "relative_clip_pct_mean","delta_lt_m005","delta_lt_m010","review_flag",
          "clean_wm_cosine_mean"]
    if "TPR_at_1pct_FPR" in frontier:
        cols += ["roc_auc","TPR_at_1pct_FPR","bit_accuracy"]
    print("\n=== QUALITY / DETECTION FRONTIER ===")
    print(frontier[cols].to_string(index=False))
    print("\nSaved:",out/"quality_summary.csv")
    print("Saved:",out/"quality_frontier.csv")
    print("Saved:",out/"quality_per_pair.csv")
    print("Saved:",out/"worst_pairs_by_condition.csv")

if __name__=="__main__":
    main()
