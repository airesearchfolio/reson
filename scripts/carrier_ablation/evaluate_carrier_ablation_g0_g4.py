#!/usr/bin/env python3
"""
RESON final SD2.1 lineage evaluator: G0 -> G4.

Evaluates every COMPLETE depth it can find. Safe to run while G4 is still generating:
G0-G3 can be evaluated now; rerun later and G4 will be added automatically.

Primary detection protocol:
  - final frozen detector checkpoint
  - frozen validation threshold = 0.500754654
  - NO per-depth recalibration

Metrics per depth:
  Presence: ROC-AUC, frozen-threshold TPR, realized FPR, 95% Wilson CIs
  Payload: overall/bit0/bit1 accuracy + 95% Wilson CI
  Scores: clean/WM mean + median
  Semantic: prompt->clean CLIP, prompt->WM CLIP, delta CLIP,
            clean<->WM CLIP image cosine
  Distribution: FID(Clean, WM), using pytorch-fid dims=2048
  Diagnostic only: ROC-derived TPR@<=1%FPR and its threshold

CLIP: OpenAI CLIP ViT-B/32 via transformers (openai/clip-vit-base-patch32).
FID: python -m pytorch_fid (standard Inception-v3 FID implementation).

The script can merge per-GPU shard manifests before the multi-GPU launcher itself
has finished. A depth is evaluated only if exactly 1000 unique test pairs exist
and every clean/WM image exists.
"""
import argparse, json, math, os, random, shutil, subprocess, sys, tempfile
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

FROZEN_THRESHOLD = 0.500754654

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root", default="workspace/reson_sd21_canonical_10k")
    p.add_argument("--lineage-root", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threshold", type=float, default=FROZEN_THRESHOLD)
    p.add_argument("--expected-n", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--clip-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--sd21-model-id", default="sd2-community/stable-diffusion-2-1-base")
    p.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    p.add_argument("--fid-device", default=None, help="Defaults to --device")
    p.add_argument("--skip-detection", action="store_true")
    p.add_argument("--skip-clip", action="store_true")
    p.add_argument("--skip-fid", action="store_true")
    p.add_argument("--depths", default="g0,g1,g2,g3,g4")
    p.add_argument("--seed", type=int, default=20260919)
    return p.parse_args()

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def wilson(k,n,z=1.959963984540054):
    if n==0: return [float("nan"),float("nan")]
    ph=k/n; den=1+z*z/n
    cen=(ph+z*z/(2*n))/den
    half=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    return [max(0.,cen-half),min(1.,cen+half)]

class Detector(nn.Module):
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

class VAE:
    def __init__(self,mid,device):
        self.device=device
        print("Loading SD2.1 VAE:",mid,flush=True)
        self.vae=AutoencoderKL.from_pretrained(mid,subfolder="vae",torch_dtype=torch.float32).to(device).eval()
        self.vae.requires_grad_(False)
        self.scale=float(self.vae.config.scaling_factor)
    @torch.no_grad()
    def encode(self,x):
        # Match final detector training exactly: posterior SAMPLE * scaling_factor.
        return (self.vae.encode(x.to(self.device,dtype=torch.float32)).latent_dist.sample()*self.scale).float()

class PairImages(Dataset):
    def __init__(self,df):
        rows=[]
        for _,r in df.iterrows():
            rows.append((int(r.sample_id),str(r.clean_path),0,int(r.bit),"clean"))
            rows.append((int(r.sample_id),str(r.wm_path),1,int(r.bit),"wm"))
        self.rows=rows
        self.tf=transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor(),
                                    transforms.Normalize([.5]*3,[.5]*3)])
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        sid,path,pres,bit,kind=self.rows[i]
        with Image.open(path) as im: x=self.tf(im.convert("RGB"))
        return x,bit,pres,i

@torch.no_grad()
def detector_predictions(df,model,vae,bs,nw,desc):
    ds=PairImages(df)
    dl=DataLoader(ds,batch_size=bs,shuffle=False,num_workers=nw,pin_memory=True,
                  persistent_workers=(nw>0))
    out=[]; model.eval()
    for x,b,p,idx in tqdm(dl,desc=desc):
        lat=vae.encode(x); bl,pl=model(lat)
        bp=torch.sigmoid(bl).cpu().numpy(); pp=torch.sigmoid(pl).cpu().numpy()
        for j,k in enumerate(idx.numpy()):
            sid,path,pres,bit,kind=ds.rows[int(k)]
            out.append(dict(sample_id=sid,kind=kind,presence=pres,bit=bit,
                            presence_score=float(pp[j]),bit_prob=float(bp[j])))
    return pd.DataFrame(out)

def detection_metrics(pred,thr):
    y=pred.presence.to_numpy(int); s=pred.presence_score.to_numpy(float)
    auc=float(roc_auc_score(y,s))
    pos=pred.presence.eq(1).to_numpy(); neg=~pos; yhat=s>=thr
    tp=int(yhat[pos].sum()); np_=int(pos.sum()); fp=int(yhat[neg].sum()); nn=int(neg.sum())
    wm=pred[pred.presence.eq(1)].copy()
    bp=(wm.bit_prob.to_numpy()>=.5).astype(int); bt=wm.bit.to_numpy(int)
    bok=(bp==bt); bk=int(bok.sum()); bn=len(bok)
    bit0=bt==0; bit1=bt==1
    fpr,tpr,ths=roc_curve(y,s); ok=np.where(fpr<=.01+1e-12)[0]; j=ok[np.argmax(tpr[ok])]
    return {
      "auc":auc,
      "tpr_frozen":tp/np_,"tpr_ci95":wilson(tp,np_),
      "fpr_frozen":fp/nn,"fpr_ci95":wilson(fp,nn),
      "bit_accuracy":bk/bn,"bit_ci95":wilson(bk,bn),
      "bit0_accuracy":float(bok[bit0].mean()) if bit0.any() else None,
      "bit1_accuracy":float(bok[bit1].mean()) if bit1.any() else None,
      "mean_clean_score":float(pred.loc[pred.presence.eq(0),"presence_score"].mean()),
      "median_clean_score":float(pred.loc[pred.presence.eq(0),"presence_score"].median()),
      "mean_wm_score":float(pred.loc[pred.presence.eq(1),"presence_score"].mean()),
      "median_wm_score":float(pred.loc[pred.presence.eq(1),"presence_score"].median()),
      "diagnostic_tpr_at_le_1pct_fpr":float(tpr[j]),
      "diagnostic_fpr":float(fpr[j]),
      "diagnostic_threshold":float(ths[j]),
    }

def g0_manifest(root,expected):
    m=pd.read_csv(root/"manifest.csv")
    if "split" in m.columns:
        z=m[m["split"].astype(str).str.lower().str.strip()=="test"].copy()
        if len(z): m=z
    m=m.sort_values("sample_id")
    pc="prompt" if "prompt" in m else "caption"
    cc=next((c for c in ["clean_path","clean_image_path"] if c in m),None)
    wc=next((c for c in ["wm_path","wm_image_path"] if c in m),None)
    if cc is None or wc is None:
        raise RuntimeError(f"G0 manifest must contain explicit clean/wm paths. Columns={list(m.columns)}")
    z=pd.DataFrame({
      "sample_id":m.sample_id.astype(int),
      "prompt":m[pc].astype(str),
      "bit":m["bit"].astype(int),
      "clean_path":m[cc].astype(str),
      "wm_path":m[wc].astype(str),
    })
    if len(z)!=expected or z.sample_id.nunique()!=expected:
        raise RuntimeError(f"G0 expected {expected}, got {len(z)}/{z.sample_id.nunique()} unique")
    return z

def lineage_manifest(root,lineage_root,depth,base,expected):
    candidates=[lineage_root/f"manifest_{depth}.csv"]
    rows=[]
    if candidates[0].exists():
        rows=[pd.read_csv(candidates[0])]
    else:
        shard_files=sorted((lineage_root/"shards").glob(f"shard_*/manifest_{depth}.csv"))
        rows=[pd.read_csv(x) for x in shard_files]
    if not rows: return None
    m=pd.concat(rows,ignore_index=True).drop_duplicates("sample_id").sort_values("sample_id")
    if len(m)!=expected or m.sample_id.nunique()!=expected:
        print(f"SKIP {depth}: only {len(m)}/{expected} unique completed manifest rows.",flush=True)
        return None
    bits=base[["sample_id","bit"]]
    m=m.merge(bits,on="sample_id",how="left",validate="one_to_one")
    m=m.rename(columns={"clean_path":"clean_path","wm_path":"wm_path"})
    return m[["sample_id","prompt","bit","clean_path","wm_path"]]

def validate_depth(df,depth,expected):
    if df is None or len(df)!=expected: return False
    missing=[]
    for c in ["clean_path","wm_path"]:
        for x in df[c]:
            if not Path(x).exists():
                missing.append(str(x))
                if len(missing)>=5: break
    if missing:
        print(f"SKIP {depth}: missing files, examples: {missing}",flush=True)
        return False
    return True

@torch.no_grad()
def clip_metrics(df,model_id,device,bs,nw,desc):
    from transformers import CLIPModel, CLIPProcessor
    print("Loading CLIP:",model_id,flush=True)
    model=CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device).eval()
    proc=CLIPProcessor.from_pretrained(model_id)
    vals_clean=[]; vals_wm=[]; vals_pair=[]
    # PIL loading here is deliberately simple and deterministic.
    for start in tqdm(range(0,len(df),bs),desc=desc):
        chunk=df.iloc[start:start+bs]
        clean=[Image.open(x).convert("RGB") for x in chunk.clean_path]
        wm=[Image.open(x).convert("RGB") for x in chunk.wm_path]
        prompts=chunk.prompt.astype(str).tolist()
        ti=proc(text=prompts,padding=True,truncation=True,return_tensors="pt").to(device)
        ci=proc(images=clean,return_tensors="pt").to(device)
        wi=proc(images=wm,return_tensors="pt").to(device)
        tf=model.get_text_features(**ti); cf=model.get_image_features(**ci); wf=model.get_image_features(**wi)
        tf=tf/tf.norm(dim=-1,keepdim=True); cf=cf/cf.norm(dim=-1,keepdim=True); wf=wf/wf.norm(dim=-1,keepdim=True)
        vals_clean.extend((tf*cf).sum(-1).cpu().numpy().tolist())
        vals_wm.extend((tf*wf).sum(-1).cpu().numpy().tolist())
        vals_pair.extend((cf*wf).sum(-1).cpu().numpy().tolist())
        for im in clean+wm: im.close()
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    c=np.asarray(vals_clean); w=np.asarray(vals_wm); q=np.asarray(vals_pair)
    return {
      "prompt_clean_clip_mean":float(c.mean()),
      "prompt_wm_clip_mean":float(w.mean()),
      "delta_clip_wm_minus_clean":float((w-c).mean()),
      "wm_higher_prompt_clip_fraction":float((w>c).mean()),
      "clean_wm_image_clip_cosine_mean":float(q.mean()),
      "prompt_clean_clip_median":float(np.median(c)),
      "prompt_wm_clip_median":float(np.median(w)),
    }

def symlink_fid_dirs(df,tmp):
    c=Path(tmp)/"clean"; w=Path(tmp)/"wm"; c.mkdir(); w.mkdir()
    for _,r in df.iterrows():
        sid=int(r.sample_id)
        os.symlink(os.path.abspath(r.clean_path),c/f"{sid:05d}.png")
        os.symlink(os.path.abspath(r.wm_path),w/f"{sid:05d}.png")
    return c,w

def fid_clean_wm(df,device):
    with tempfile.TemporaryDirectory(prefix="reson_fid_") as td:
        c,w=symlink_fid_dirs(df,td)
        cmd=[sys.executable,"-m","pytorch_fid",str(c),str(w),"--dims","2048","--device",device]
        print("FID command:"," ".join(cmd),flush=True)
        r=subprocess.run(cmd,capture_output=True,text=True)
        print(r.stdout,flush=True)
        if r.returncode:
            print(r.stderr,flush=True)
            raise RuntimeError("pytorch-fid failed. Install with: pip install pytorch-fid")
        # Last floating number in stdout is the FID.
        import re
        nums=re.findall(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?",r.stdout)
        if not nums: raise RuntimeError("Could not parse FID output: "+r.stdout)
        return float(nums[-1])

def main():
    a=args(); seed_all(a.seed)
    root=Path(a.data_root)
    lr=Path(a.lineage_root) if a.lineage_root else root/"final_lineage_test_multigpu"
    ck=Path(a.checkpoint) if a.checkpoint else root/"final_detector"/"detector_best.pt"
    out=Path(a.out_dir) if a.out_dir else root/"final_lineage_evaluation"
    out.mkdir(parents=True,exist_ok=True)
    requested=[x.strip().lower() for x in a.depths.split(",") if x.strip()]
    base=g0_manifest(root,a.expected_n)
    manifests={"g0":base}
    for d in ["g1","g2","g3","g4"]:
        manifests[d]=lineage_manifest(root,lr,d,base,a.expected_n)
    complete=[d for d in requested if d in manifests and validate_depth(manifests[d],d,a.expected_n)]
    if not complete: raise RuntimeError("No complete requested depths found.")
    print("COMPLETE DEPTHS TO EVALUATE:",complete,flush=True)

    dev=torch.device(a.device)
    model=vae=None
    if not a.skip_detection:
        if not torch.cuda.is_available(): raise RuntimeError("CUDA required for detector/VAE evaluation")
        model=Detector().to(dev)
        state=torch.load(ck,map_location=dev)
        model.load_state_dict(state["model_state_dict"],strict=True); model.eval()
        vae=VAE(a.sd21_model_id,dev)

    allres={}
    summary=[]
    for di,d in enumerate(complete):
        print("\n"+"="*72+f"\nEVALUATING {d.upper()}\n"+"="*72,flush=True)
        df=manifests[d].copy()
        df.to_csv(out/f"{d}_evaluation_manifest.csv",index=False)
        res={"depth":d,"n_pairs":len(df),"frozen_threshold":a.threshold}

        if not a.skip_detection:
            # Fixed seed per depth so repeated runs are reproducible.
            seed_all(a.seed+1000+di)
            pred=detector_predictions(df,model,vae,a.batch_size,a.num_workers,f"{d} detector")
            pred.to_csv(out/f"{d}_detector_predictions.csv",index=False)
            res.update(detection_metrics(pred,a.threshold))

        if not a.skip_clip:
            res.update(clip_metrics(df,a.clip_model_id,dev,a.clip_batch_size,a.num_workers,f"{d} CLIP"))

        if not a.skip_fid:
            fiddev=a.fid_device or a.device
            res["fid_clean_wm"]=fid_clean_wm(df,fiddev)

        allres[d]=res
        (out/f"{d}_metrics.json").write_text(json.dumps(res,indent=2))
        summary.append(res)
        pd.DataFrame(summary).to_csv(out/"lineage_metrics_summary.csv",index=False)
        (out/"lineage_metrics_all.json").write_text(json.dumps(allres,indent=2))
        print(json.dumps(res,indent=2),flush=True)

    # Retention/drop relative to G0 when detection is available.
    if "g0" in allres and "tpr_frozen" in allres["g0"]:
        g0=allres["g0"]["tpr_frozen"]
        for r in summary:
            if "tpr_frozen" in r:
                r["tpr_drop_vs_g0"]=r["tpr_frozen"]-g0
                r["tpr_retention_vs_g0"]=r["tpr_frozen"]/g0 if g0 else None
        pd.DataFrame(summary).to_csv(out/"lineage_metrics_summary.csv",index=False)
        (out/"lineage_metrics_all.json").write_text(json.dumps({r["depth"]:r for r in summary},indent=2))

    print("\nDONE. Results:",out,flush=True)
    print("Primary table:",out/"lineage_metrics_summary.csv",flush=True)
    print("Rerun this same command after G4 completes; completed earlier depths may be recomputed, or use --depths g4 for G4 only.",flush=True)

if __name__=="__main__":
    main()
