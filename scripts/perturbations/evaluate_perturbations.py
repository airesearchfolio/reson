#!/usr/bin/env python3
"""
Evaluate FINAL RESON SD2.1 canonical G0 perturbation robustness.

Input:
  canonical root/perturbations_test_manifest.csv
  canonical root/perturbations_test/  (44,000 PNGs; 22 conditions x 1000 pairs)

Protocol:
  * FINAL frozen detector checkpoint
  * FINAL frozen validation threshold = 0.500754654
  * NO threshold recalibration for primary metrics
  * clean images -> realized FPR
  * WM images -> TPR + payload recovery
  * ROC AUC is threshold-independent
  * Optional ROC-derived TPR@<=1% FPR is diagnostic ONLY

Outputs:
  perturbation_metrics_summary.csv
  perturbation_metrics_all.json
  perturbation_predictions.csv
  perturbation_metrics_paper.csv
"""
import argparse, json, math, random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

FROZEN_THRESHOLD=0.500754654

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="workspace/reson_sd21_canonical_10k")
    p.add_argument("--manifest",default=None)
    p.add_argument("--checkpoint",default=None)
    p.add_argument("--out-dir",default=None)
    p.add_argument("--device",default="cuda")
    p.add_argument("--threshold",type=float,default=FROZEN_THRESHOLD)
    p.add_argument("--model-id",default="sd2-community/stable-diffusion-2-1-base")
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--expected-pairs",type=int,default=1000)
    p.add_argument("--expected-conditions",type=int,default=22)
    p.add_argument("--seed",type=int,default=20260920)
    return p.parse_args()

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def wilson(k,n,z=1.959963984540054):
    if n<=0:return [float("nan"),float("nan")]
    q=k/n; den=1+z*z/n
    cen=(q+z*z/(2*n))/den
    h=z*math.sqrt(q*(1-q)/n+z*z/(4*n*n))/den
    return [max(0.,cen-h),min(1.,cen+h)]

class LatentJointDecoder(nn.Module):
    """Exact architecture used by final canonical detector."""
    def __init__(self):
        super().__init__()
        self.conv_layers=nn.Sequential(
            nn.Conv2d(4,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,1,1),nn.BatchNorm2d(256),nn.ReLU(inplace=True),nn.MaxPool2d(2))
        d=256*4*4
        self.presence_fc=nn.Sequential(nn.Linear(d,512),nn.ReLU(inplace=True),nn.Linear(512,1))
        self.bit_fc=nn.Sequential(nn.Linear(d,512),nn.ReLU(inplace=True),nn.Linear(512,1))
    def forward(self,x):
        f=torch.flatten(self.conv_layers(x),1)
        return self.bit_fc(f).squeeze(1),self.presence_fc(f).squeeze(1)

class ImgDS(Dataset):
    def __init__(self,rows):
        self.rows=rows.reset_index(drop=True)
        self.tf=transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize([.5]*3,[.5]*3)])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows.iloc[i]
        with Image.open(r["path"]) as im:
            x=self.tf(im.convert("RGB"))
        return x,i

def identify_columns(m):
    """Resolve perturbation manifest columns, including the canonical final manifest."""
    aliases = {
        "condition": ["condition", "condition_name", "variant"],
        "sample_id": ["sample_id", "sid", "id"],
        "bit": ["bit", "payload_bit", "label_bit"],
        "clean": ["clean_image_path", "clean_path", "image_clean", "clean"],
        "wm": ["wm_image_path", "watermarked_image_path", "wm_path", "image_wm", "watermarked"],
        "attack": ["attack", "attack_name", "perturbation"],
        "strength": ["strength", "attack_strength", "level"],
    }

    def pick(key, required=True):
        for c in aliases[key]:
            if c in m.columns:
                return c
        if required:
            raise RuntimeError(
                f"Could not identify required '{key}' manifest column.\n"
                f"Columns: {', '.join(m.columns)}"
            )
        return None

    condition = pick("condition")
    sample_id = pick("sample_id")
    bit = pick("bit")
    clean = pick("clean")
    wm = pick("wm")
    attack = pick("attack", required=False)
    strength = pick("strength", required=False)

    # Preserve the caller's expected 7-return-value interface.
    return condition, sample_id, bit, clean, wm, attack, strength

def build_rows(m):
    cond,sid,bit,clean,wm,attack,strength=identify_columns(m)
    out=[]
    for _,r in m.iterrows():
        common=dict(condition=str(r[cond]),sample_id=int(r[sid]),bit=int(r[bit]))
        if attack: common["attack"]=str(r[attack])
        if strength: common["strength"]=str(r[strength])
        out.append({**common,"kind":"clean","presence":0,"path":str(r[clean])})
        out.append({**common,"kind":"wm","presence":1,"path":str(r[wm])})
    return pd.DataFrame(out)

def resolve_paths(rows,root):
    # Manifest may store absolute or relative paths.
    vals=[]
    for x in rows.path:
        p=Path(x)
        if not p.is_absolute(): p=root/p
        vals.append(str(p))
    rows=rows.copy(); rows["path"]=vals
    return rows

@torch.no_grad()
def infer(rows,model,vae,device,bs,nw):
    ds=ImgDS(rows)
    dl=DataLoader(ds,batch_size=bs,shuffle=False,num_workers=nw,pin_memory=True,
                  persistent_workers=(nw>0))
    ps=np.empty(len(ds),np.float32); bp=np.empty(len(ds),np.float32)
    for x,idx in tqdm(dl,desc="VAE + frozen detector"):
        x=x.to(device,dtype=torch.float32,non_blocking=True)
        # Must match final detector training: posterior sample * scaling factor.
        lat=vae.encode(x).latent_dist.sample()*float(vae.config.scaling_factor)
        bl,pl=model(lat.float())
        ii=idx.numpy()
        ps[ii]=torch.sigmoid(pl).cpu().numpy()
        bp[ii]=torch.sigmoid(bl).cpu().numpy()
    z=rows.copy()
    z["presence_score"]=ps
    z["bit_prob"]=bp
    return z

def metrics(g,thr):
    y=g.presence.to_numpy(int); s=g.presence_score.to_numpy(float)
    if len(np.unique(y))<2: raise RuntimeError("Condition missing clean or WM examples")
    auc=float(roc_auc_score(y,s))
    wm=g[g.presence.eq(1)]; cl=g[g.presence.eq(0)]
    wp=wm.presence_score.to_numpy()>=thr
    cp=cl.presence_score.to_numpy()>=thr
    tp=int(wp.sum()); nwm=len(wm); fp=int(cp.sum()); ncl=len(cl)
    truth=wm.bit.to_numpy(int)
    bhat=(wm.bit_prob.to_numpy()>=.5).astype(int)
    good=bhat==truth; b0=truth==0; b1=truth==1
    fpr,tpr,ths=roc_curve(y,s)
    ok=np.where(fpr<=.01+1e-12)[0]
    j=ok[np.argmax(tpr[ok])]
    return {
      "n_pairs":nwm,
      "auc":auc,
      "frozen_threshold":thr,
      "tpr_frozen":tp/nwm,
      "tpr_ci95_low":wilson(tp,nwm)[0],"tpr_ci95_high":wilson(tp,nwm)[1],
      "realized_fpr_frozen":fp/ncl,
      "fpr_ci95_low":wilson(fp,ncl)[0],"fpr_ci95_high":wilson(fp,ncl)[1],
      "bit_accuracy":float(good.mean()),
      "bit_ci95_low":wilson(int(good.sum()),len(good))[0],
      "bit_ci95_high":wilson(int(good.sum()),len(good))[1],
      "bit0_accuracy":float(good[b0].mean()) if b0.any() else None,
      "bit1_accuracy":float(good[b1].mean()) if b1.any() else None,
      "mean_clean_score":float(cl.presence_score.mean()),
      "median_clean_score":float(cl.presence_score.median()),
      "mean_wm_score":float(wm.presence_score.mean()),
      "median_wm_score":float(wm.presence_score.median()),
      # Diagnostic only -- NEVER substitute this threshold for frozen threshold.
      "diagnostic_tpr_at_le_1pct_fpr":float(tpr[j]),
      "diagnostic_fpr":float(fpr[j]),
      "diagnostic_threshold":float(ths[j]),
    }

def main():
    a=parse(); seed_all(a.seed)
    root=Path(a.root)
    manifest=Path(a.manifest) if a.manifest else root/"perturbations_test_manifest.csv"
    ck=Path(a.checkpoint) if a.checkpoint else root/"final_detector"/"detector_best.pt"
    out=Path(a.out_dir) if a.out_dir else root/"perturbation_evaluation_final"
    out.mkdir(parents=True,exist_ok=True)

    print("Manifest :",manifest)
    print("Checkpoint:",ck)
    print("Threshold :",a.threshold,"(FROZEN; no attack recalibration)")
    m=pd.read_csv(manifest)
    cond,sid,bit,clean,wm,attack,strength=identify_columns(m)
    print("Manifest rows:",len(m),"conditions:",m[cond].nunique())
    if len(m)!=a.expected_pairs*a.expected_conditions:
        raise RuntimeError(f"Expected {a.expected_pairs*a.expected_conditions} manifest rows; got {len(m)}")
    if m[cond].nunique()!=a.expected_conditions:
        raise RuntimeError(f"Expected {a.expected_conditions} conditions; got {m[cond].nunique()}")
    counts=m.groupby(cond)[sid].nunique()
    if not (counts==a.expected_pairs).all():
        raise RuntimeError("Not every condition has expected unique pairs:\n"+str(counts))

    rows=resolve_paths(build_rows(m),root)
    missing=[x for x in rows.path if not Path(x).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} images missing. Examples:\n"+"\n".join(missing[:10]))
    print("Validated images:",len(rows),"(expected 44,000)")

    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU is required.")
    device=torch.device(a.device)
    print("Loading VAE:",a.model_id)
    vae=AutoencoderKL.from_pretrained(a.model_id,subfolder="vae",torch_dtype=torch.float32).to(device).eval()
    vae.requires_grad_(False)

    print("Loading final frozen detector:",ck)
    model=LatentJointDecoder().to(device)
    obj=torch.load(ck,map_location=device)
    # Final training checkpoint stores model_state_dict.
    state=obj["model_state_dict"] if isinstance(obj,dict) and "model_state_dict" in obj else obj
    model.load_state_dict(state,strict=True); model.eval()

    pred=infer(rows,model,vae,device,a.batch_size,a.num_workers)
    pred.to_csv(out/"perturbation_predictions.csv",index=False)

    results=[]
    for name,g in pred.groupby("condition",sort=False):
        r={"condition":name}
        # Preserve attack/strength labels if present.
        if "attack" in g: r["attack"]=str(g.attack.iloc[0])
        if "strength" in g: r["strength"]=str(g.strength.iloc[0])
        r.update(metrics(g,a.threshold))
        results.append(r)

    res=pd.DataFrame(results)
    # Put identity/none first when recognizable.
    order=res.condition.str.lower().map(lambda x: 0 if x in {"identity","none","clean","original"} or "identity" in x else 1)
    res=res.assign(_ord=order).sort_values(["_ord","condition"]).drop(columns="_ord").reset_index(drop=True)
    res.to_csv(out/"perturbation_metrics_summary.csv",index=False)
    (out/"perturbation_metrics_all.json").write_text(
        json.dumps({r["condition"]:r for r in res.to_dict("records")},indent=2))

    papercols=[x for x in ["condition","attack","strength","n_pairs","auc","tpr_frozen",
                           "realized_fpr_frozen","bit_accuracy","bit0_accuracy","bit1_accuracy",
                           "mean_clean_score","mean_wm_score"] if x in res]
    res[papercols].to_csv(out/"perturbation_metrics_paper.csv",index=False)

    print("\nFINAL PERTURBATION RESULTS")
    print(res[papercols].to_string(index=False))
    print("\nSaved:")
    for x in ["perturbation_metrics_summary.csv","perturbation_metrics_all.json",
              "perturbation_predictions.csv","perturbation_metrics_paper.csv"]:
        print(" ",out/x)

if __name__=="__main__":
    main()
