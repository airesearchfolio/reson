#!/usr/bin/env python3
"""
RESON matched canonical-300 G0-G4 detector evaluation.

Selects the first N held-out RESON test sample_ids (sorted), then filters every
lineage depth to exactly those IDs. No generation/retraining. Detection only.

Reports:
- ROC-AUC
- frozen-threshold TPR/FPR
- payload bit accuracy
- diagnostic TPR @ <=1% FPR
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

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",default="workspace/reson_sd21_canonical_10k")
    p.add_argument("--lineage-root",default="workspace/paper_experiments/reson_sd20_g0_canonical300_transfer")
    p.add_argument("--checkpoint",default=None)
    p.add_argument("--out-dir",default=None)
    p.add_argument("--n",type=int,default=300)
    p.add_argument("--threshold",type=float,default=FROZEN_THRESHOLD)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--device",default="cuda")
    p.add_argument("--sd21-model-id",default="sd2-community/stable-diffusion-2-1-base")
    p.add_argument("--seed",type=int,default=20260922)
    return p.parse_args()

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def wilson(k,n,z=1.959963984540054):
    if not n: return [float("nan"),float("nan")]
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
        print("Loading SD2.1 VAE:",mid,flush=True)
        self.device=device
        self.vae=AutoencoderKL.from_pretrained(mid,subfolder="vae",torch_dtype=torch.float32).to(device).eval()
        self.vae.requires_grad_(False)
        self.scale=float(self.vae.config.scaling_factor)
    @torch.no_grad()
    def encode(self,x):
        return (self.vae.encode(x.to(self.device,dtype=torch.float32)).latent_dist.sample()*self.scale).float()

class PairImages(Dataset):
    def __init__(self,df):
        self.rows=[]
        for _,r in df.iterrows():
            self.rows.append((int(r.sample_id),str(r.clean_path),0,int(r.bit),"clean"))
            self.rows.append((int(r.sample_id),str(r.wm_path),1,int(r.bit),"wm"))
        self.tf=transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor(),
                                    transforms.Normalize([.5]*3,[.5]*3)])
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        sid,path,pres,bit,kind=self.rows[i]
        with Image.open(path) as im: x=self.tf(im.convert("RGB"))
        return x,bit,pres,i

def pick_col(df,names):
    return next((x for x in names if x in df.columns),None)

def load_transfer(lineage_root, depth, n, ids=None):
    names={0:"generation_0.csv",1:"generation_1_flux.csv",2:"generation_2_realvis_xl.csv",3:"generation_3_dreamshaper_xl.csv",4:"generation_4_flux.csv"}
    mp=lineage_root/"manifests"/names[depth]
    if not mp.exists(): raise FileNotFoundError(mp)
    m=pd.read_csv(mp)
    if "sample_id" not in m: raise RuntimeError(f"{mp}: missing sample_id")
    m["sample_id"]=m.sample_id.astype(int)
    m=m.drop_duplicates("sample_id").sort_values("sample_id")
    if ids is None:
        m=m.head(n).copy()
        if len(m)!=n: raise RuntimeError(f"G{depth}: expected {n}, got {len(m)}")
    else:
        ids=list(map(int,ids)); m=m[m.sample_id.isin(ids)].copy()
        if len(m)!=len(ids): raise RuntimeError(f"G{depth}: expected {len(ids)}, got {len(m)}")
        m=m.set_index("sample_id").loc[ids].reset_index()
    clean=pick_col(m,["clean_path","alpha0_image_path","clean_image_path"])
    wm=pick_col(m,["wm_path","wm_image_path","watermarked_image_path"])
    bit=pick_col(m,["bit","payload_bit","watermark_bit"])
    pc=pick_col(m,["prompt","caption","text"])
    if not clean or not wm or not bit: raise RuntimeError(f"G{depth}: missing required path/bit columns")
    z=pd.DataFrame({"sample_id":m.sample_id.astype(int),"bit":m[bit].astype(int),
                    "prompt":m[pc].astype(str) if pc else "",
                    "clean_path":m[clean].astype(str),"wm_path":m[wm].astype(str)})
    for c in ["clean_path","wm_path"]:
        miss=[x for x in z[c] if not Path(str(x)).exists()]
        if miss: raise FileNotFoundError(f"G{depth}: missing {c}; first={miss[0]}")
    return z.reset_index(drop=True)

@torch.no_grad()
def predict(df,model,vae,bs,nw,desc):
    ds=PairImages(df)
    dl=DataLoader(ds,batch_size=bs,shuffle=False,num_workers=nw,pin_memory=True,
                  persistent_workers=(nw>0))
    rows=[]; model.eval()
    # tqdm counts images, not batches.
    bar=tqdm(total=len(ds),desc=desc,unit="img")
    for x,b,p,idx in dl:
        lat=vae.encode(x); bl,pl=model(lat)
        bp=torch.sigmoid(bl).cpu().numpy(); pp=torch.sigmoid(pl).cpu().numpy()
        for j,k in enumerate(idx.numpy()):
            sid,path,pres,bit,kind=ds.rows[int(k)]
            rows.append(dict(sample_id=sid,kind=kind,presence=pres,bit=bit,
                             presence_score=float(pp[j]),bit_prob=float(bp[j])))
        bar.update(len(idx))
    bar.close()
    return pd.DataFrame(rows)

def metrics(pred,thr):
    y=pred.presence.to_numpy(int); s=pred.presence_score.to_numpy(float)
    auc=float(roc_auc_score(y,s))
    pos=y==1; neg=y==0; yh=s>=thr
    tp=int(yh[pos].sum()); fp=int(yh[neg].sum()); np_=int(pos.sum()); nn=int(neg.sum())
    wm=pred[pred.presence.eq(1)]
    bt=wm.bit.to_numpy(int); bp=(wm.bit_prob.to_numpy()>=.5).astype(int); ok=bp==bt
    fpr,tpr,ths=roc_curve(y,s)
    good=np.where(fpr<=.01+1e-12)[0]; j=good[np.argmax(tpr[good])]
    return dict(
        auc=auc,frozen_threshold=thr,
        tpr_frozen=tp/np_,tpr_ci95=wilson(tp,np_),
        fpr_frozen=fp/nn,fpr_ci95=wilson(fp,nn),
        bit_accuracy=float(ok.mean()),bit0_accuracy=float(ok[bt==0].mean()) if (bt==0).any() else None,
        bit1_accuracy=float(ok[bt==1].mean()) if (bt==1).any() else None,
        diagnostic_tpr_at_le_1pct_fpr=float(tpr[j]),diagnostic_fpr=float(fpr[j]),
        diagnostic_threshold=float(ths[j]))

def main():
    a=parse_args(); seed_all(a.seed)
    root=Path(a.data_root)
    lr=Path(a.lineage_root)
    ck=Path(a.checkpoint) if a.checkpoint else root/"final_detector"/"detector_best.pt"
    out=Path(a.out_dir) if a.out_dir else lr/"evaluation_frozen_sd21_detector"
    out.mkdir(parents=True,exist_ok=True)
    if not ck.exists(): raise FileNotFoundError(f"Frozen checkpoint not found: {ck}")

    base=load_transfer(lr,0,a.n)
    ids=base.sample_id.tolist()
    print(f"SD2.0 transfer matched subset: N={len(ids)}, IDs {ids[0]}..{ids[-1]}",flush=True)
    print(f"Frozen checkpoint: {ck}",flush=True)
    print(f"Frozen threshold: {a.threshold}",flush=True)
    manifests={"g0":base}
    for i in range(1,5): manifests[f"g{i}"]=load_transfer(lr,i,a.n,ids)
    for d,df in manifests.items():
        assert df.sample_id.tolist()==ids, f"{d}: sample ordering mismatch"
        df.to_csv(out/f"{d}_evaluation_manifest.csv",index=False)
    print("Verified: identical sample IDs at G0-G4.",flush=True)

    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    dev=torch.device(a.device)
    model=Detector().to(dev)
    state=torch.load(ck,map_location=dev)
    sd=state["model_state_dict"] if isinstance(state,dict) and "model_state_dict" in state else state
    model.load_state_dict(sd,strict=True); model.eval()
    vae=VAE(a.sd21_model_id,dev)

    summary=[]
    for i,d in enumerate(["g0","g1","g2","g3","g4"]):
        seed_all(a.seed+1000+i)
        pred=predict(manifests[d],model,vae,a.batch_size,a.num_workers,
                     f"RESON {d.upper()} detector")
        pred.to_csv(out/f"{d}_predictions.csv",index=False)
        r={"depth":d.upper(),"n_pairs":a.n,**metrics(pred,a.threshold)}
        summary.append(r)
        (out/f"{d}_metrics.json").write_text(json.dumps(r,indent=2))
        print("\n"+json.dumps(r,indent=2),flush=True)

    s=pd.DataFrame(summary)
    s.to_csv(out/"reson_sd20_transfer_canonical300_g0_g4_summary.csv",index=False)
    (out/"reson_sd20_transfer_canonical300_g0_g4_all.json").write_text(json.dumps(summary,indent=2))
    print("\nFINAL SD2.0 TRANSFER MATCHED N=300 SUMMARY",flush=True)
    print(s[["depth","auc","tpr_frozen","fpr_frozen","bit_accuracy",
             "diagnostic_tpr_at_le_1pct_fpr","diagnostic_fpr"]].to_string(index=False),flush=True)
    print("\nSaved:",out/"reson_sd20_transfer_canonical300_g0_g4_summary.csv",flush=True)

if __name__=="__main__":
    main()
