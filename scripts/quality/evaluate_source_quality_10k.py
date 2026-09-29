#!/usr/bin/env python3
"""
Final RESON SD2.1 G0 quality gate.

Primary:
  - absolute prompt -> WM CLIP score and tail statistics

Secondary:
  - clean prompt CLIP
  - WM-clean prompt CLIP delta
  - clean<->WM CLIP-image cosine
  - Clean-WM FID (distribution-shift FID; NOT real-reference FID)

Outputs:
  evaluation_quality/
    quality_summary.json / .csv
    per_pair_metrics.csv
    lowest_30_wm_clip.csv
    largest_30_clip_drops.csv
    wm_clip_histogram.png
    prompt_clip_distribution.png
    clean_vs_wm_prompt_clip.png
    lowest_wm_clip_contact_sheet.jpg
    largest_clip_drop_contact_sheet.jpg
"""
import argparse, json, math, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--root", default="workspace/reson_sd21_canonical_10k")
    p.add_argument("--manifest", default=None)
    p.add_argument("--gpu", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    p.add_argument("--fid-batch-size", type=int, default=64)
    p.add_argument("--skip-fid", action="store_true")
    return p.parse_args()

def ci_mean(x):
    x=np.asarray(x,float); n=len(x)
    m=float(x.mean()); se=float(x.std(ddof=1)/math.sqrt(n))
    return m, m-1.96*se, m+1.96*se

def contact_sheet(df, out, title, n=30):
    rows=[]
    for _,r in df.head(n).iterrows():
        try:
            a=Image.open(r.clean_path).convert("RGB")
            b=Image.open(r.wm_path).convert("RGB")
            a.thumbnail((220,220)); b.thumbnail((220,220))
            cell=Image.new("RGB",(460,280),"white")
            cell.paste(a,(5,30)); cell.paste(b,(235,30))
            d=ImageDraw.Draw(cell)
            d.text((5,5),f"ID {int(r.sample_id)} | clean",fill="black")
            d.text((235,5),f"WM | CLIP {r.wm_prompt_clip:.3f}",fill="black")
            prompt=str(r.prompt).replace("\n"," ")
            d.text((5,255),prompt[:72],fill="black")
            rows.append(cell)
        except Exception:
            pass
    if not rows: return
    cols=2; rr=math.ceil(len(rows)/cols)
    sheet=Image.new("RGB",(cols*460,rr*280),"white")
    for i,im in enumerate(rows): sheet.paste(im,((i%cols)*460,(i//cols)*280))
    sheet.save(out,quality=92)

def main():
    a=args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    root=Path(a.root); manifest=Path(a.manifest) if a.manifest else root/"manifest.csv"
    out=root/"evaluation_quality"; out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(manifest)
    if len(df)!=10000: raise RuntimeError(f"Expected 10000 pairs, got {len(df)}")
    for c in ["sample_id","prompt","clean_path","wm_path","split","bit","generation_seed"]:
        if c not in df: raise RuntimeError(f"Missing manifest column: {c}")

    from transformers import CLIPModel, CLIPProcessor
    model=CLIPModel.from_pretrained(a.clip_model).to(device).eval()
    proc=CLIPProcessor.from_pretrained(a.clip_model)

    clean_prompt=[]; wm_prompt=[]; image_cos=[]
    for st in tqdm(range(0,len(df),a.batch_size),desc="CLIP quality"):
        q=df.iloc[st:st+a.batch_size]
        prompts=q.prompt.astype(str).tolist()
        clean=[Image.open(x).convert("RGB") for x in q.clean_path]
        wm=[Image.open(x).convert("RGB") for x in q.wm_path]
        with torch.inference_mode():
            ti=proc(text=prompts,return_tensors="pt",padding=True,truncation=True).to(device)
            t=model.get_text_features(**ti); t=t/t.norm(dim=-1,keepdim=True)
            ci=proc(images=clean,return_tensors="pt").to(device)
            wi=proc(images=wm,return_tensors="pt").to(device)
            cf=model.get_image_features(**ci); cf=cf/cf.norm(dim=-1,keepdim=True)
            wf=model.get_image_features(**wi); wf=wf/wf.norm(dim=-1,keepdim=True)
            clean_prompt.extend((cf*t).sum(-1).cpu().float().tolist())
            wm_prompt.extend((wf*t).sum(-1).cpu().float().tolist())
            image_cos.extend((cf*wf).sum(-1).cpu().float().tolist())

    df["clean_prompt_clip"]=clean_prompt
    df["wm_prompt_clip"]=wm_prompt
    df["delta_prompt_clip"]=df.wm_prompt_clip-df.clean_prompt_clip
    df["relative_delta_clip_pct"]=100*df.delta_prompt_clip/df.clean_prompt_clip.replace(0,np.nan)
    df["clean_wm_image_cosine"]=image_cos
    df.to_csv(out/"per_pair_metrics.csv",index=False)

    low=df.sort_values("wm_prompt_clip").head(30)
    drops=df.sort_values("delta_prompt_clip").head(30)
    low.to_csv(out/"lowest_30_wm_clip.csv",index=False)
    drops.to_csv(out/"largest_30_clip_drops.csv",index=False)
    contact_sheet(low,out/"lowest_wm_clip_contact_sheet.jpg","Lowest WM CLIP")
    contact_sheet(drops,out/"largest_clip_drop_contact_sheet.jpg","Largest drops")

    w=df.wm_prompt_clip.to_numpy(); c=df.clean_prompt_clip.to_numpy()
    wm_m,wm_lo,wm_hi=ci_mean(w)
    cl_m,cl_lo,cl_hi=ci_mean(c)
    summary={
      "n_pairs":len(df),
      "primary_metric":"absolute prompt-to-watermarked-image CLIP",
      "wm_prompt_clip_mean":wm_m,
      "wm_prompt_clip_mean_ci95":[wm_lo,wm_hi],
      "wm_prompt_clip_median":float(np.median(w)),
      "wm_prompt_clip_q01":float(np.quantile(w,.01)),
      "wm_prompt_clip_q05":float(np.quantile(w,.05)),
      "wm_prompt_clip_q10":float(np.quantile(w,.10)),
      "wm_prompt_clip_lt_0.10_pct":float(100*np.mean(w<.10)),
      "wm_prompt_clip_lt_0.15_pct":float(100*np.mean(w<.15)),
      "wm_prompt_clip_lt_0.20_pct":float(100*np.mean(w<.20)),
      "clean_prompt_clip_mean":cl_m,
      "clean_prompt_clip_mean_ci95":[cl_lo,cl_hi],
      "mean_delta_prompt_clip":float(np.mean(w-c)),
      "mean_relative_delta_clip_pct":float(np.nanmean(df.relative_delta_clip_pct)),
      "wm_higher_than_clean_pct":float(100*np.mean(w>c)),
      "clean_wm_image_cosine_mean":float(df.clean_wm_image_cosine.mean()),
      "clean_wm_fid":None,
      "fid_label":"Clean-WM distribution-shift FID (not real-reference FID)"
    }

    # FID uses standard Inception features via torchmetrics if installed.
    if not a.skip_fid:
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            fid=FrechetInceptionDistance(feature=2048,normalize=True).to(device)
            for st in tqdm(range(0,len(df),a.fid_batch_size),desc="Clean-WM FID"):
                q=df.iloc[st:st+a.fid_batch_size]
                ca=[]; wa=[]
                for cp,wp in zip(q.clean_path,q.wm_path):
                    ca.append(np.asarray(Image.open(cp).convert("RGB").resize((299,299)),dtype=np.uint8))
                    wa.append(np.asarray(Image.open(wp).convert("RGB").resize((299,299)),dtype=np.uint8))
                ct=torch.from_numpy(np.stack(ca)).permute(0,3,1,2).to(device).float()/255.
                wt=torch.from_numpy(np.stack(wa)).permute(0,3,1,2).to(device).float()/255.
                fid.update(ct,real=True); fid.update(wt,real=False)
            summary["clean_wm_fid"]=float(fid.compute().cpu())
        except Exception as e:
            summary["fid_error"]=f"{type(e).__name__}: {e}"

    with open(out/"quality_summary.json","w") as f: json.dump(summary,f,indent=2)
    pd.DataFrame([summary]).to_csv(out/"quality_summary.csv",index=False)

    plt.figure(figsize=(7,5)); plt.hist(w,bins=50)
    plt.xlabel("Prompt → WM image CLIP"); plt.ylabel("Count"); plt.title("RESON SD2.1: WM prompt-CLIP")
    plt.tight_layout(); plt.savefig(out/"wm_clip_histogram.png",dpi=180); plt.close()

    plt.figure(figsize=(7,5)); plt.hist(c,bins=50,alpha=.55,label="Clean"); plt.hist(w,bins=50,alpha=.55,label="WM")
    plt.xlabel("Prompt-image CLIP"); plt.ylabel("Count"); plt.legend(); plt.title("Prompt-CLIP distributions")
    plt.tight_layout(); plt.savefig(out/"prompt_clip_distribution.png",dpi=180); plt.close()

    plt.figure(figsize=(6,6)); plt.scatter(c,w,s=5,alpha=.25)
    mn=min(c.min(),w.min()); mx=max(c.max(),w.max()); plt.plot([mn,mx],[mn,mx])
    plt.xlabel("Clean prompt-CLIP"); plt.ylabel("WM prompt-CLIP"); plt.title("Clean vs WM semantic alignment")
    plt.tight_layout(); plt.savefig(out/"clean_vs_wm_prompt_clip.png",dpi=180); plt.close()

    print("\nFINAL QUALITY GATE")
    print("==================")
    for k,v in summary.items(): print(f"{k}: {v}")
    print(f"\nOutputs: {out}")

if __name__=="__main__":
    main()
