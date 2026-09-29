#!/usr/bin/env python3
"""
Evaluate final RESON 4-bit detector on G0-G4.
Primary: validation-frozen threshold from final 4-bit results.
Secondary: per-depth ROC TPR at <=1% FPR.
No retraining and no threshold recalibration for the primary result.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from diffusers import AutoencoderKL

class Detector4Bit(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(4,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.presence = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096,512),
            nn.ReLU(),
            nn.Linear(512,1)
        )

        self.payload = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096,512),
            nn.ReLU(),
            nn.Linear(512,4)
        )

    def forward(self, z):
        h = self.features(z)
        return self.presence(h).squeeze(1), self.payload(h)
        
def load_rgb(p):
    return Image.open(p).convert("RGB").resize((512,512),Image.Resampling.LANCZOS)

def tensorize(img):
    a=np.asarray(img).astype(np.float32)/127.5-1.0
    return torch.from_numpy(a).permute(2,0,1)

def op_at_fpr(y,s,target=.01):
    fpr,tpr,thr=roc_curve(y,s)
    idx=np.where(fpr<=target)[0]
    j=idx[np.argmax(tpr[idx])]
    return float(tpr[j]),float(fpr[j]),float(thr[j])

@torch.no_grad()
def encode(vae,x):
    # Must match final trainer: posterior sample() * scaling_factor.
    return vae.encode(x).latent_dist.sample()*vae.config.scaling_factor

@torch.no_grad()
def infer_pairs(df,model,vae,device,batch):
    ys=[]; ps=[]; bit_logits=[]; bit_true=[]; kinds=[]
    records=[]
    items=[]
    for _,r in df.iterrows():
        bits=[int(r[f"bit{i}"]) for i in range(4)]
        items.append((r.clean_path,0,bits,int(r.sample_id),"clean"))
        items.append((r.wm_path,1,bits,int(r.sample_id),"wm"))
    for st in tqdm(range(0,len(items),batch),desc="detector"):
        chunk=items[st:st+batch]
        x=torch.stack([tensorize(load_rgb(z[0])) for z in chunk]).to(device)
        z=encode(vae,x)
        pl,bl=model(z)
        p=torch.sigmoid(pl).cpu().numpy()
        b=torch.sigmoid(bl).cpu().numpy()
        for q,pp,bb in zip(chunk,p,b):
            path,y,bits,sid,kind=q
            records.append({"sample_id":sid,"kind":kind,"label":y,"presence_score":float(pp),
                            **{f"true_bit{i}":bits[i] for i in range(4)},
                            **{f"bit_prob{i}":float(bb[i]) for i in range(4)}})
    return pd.DataFrame(records)

def metrics(pred,thr):
    y=pred.label.to_numpy(); s=pred.presence_score.to_numpy()
    auc=float(roc_auc_score(y,s))
    yp=s>=thr
    tpr=float(yp[y==1].mean()); fpr=float(yp[y==0].mean())
    wm=pred[pred.kind=="wm"].copy()
    acc=[]
    for i in range(4):
        bh=(wm[f"bit_prob{i}"].to_numpy()>=.5).astype(int)
        bt=wm[f"true_bit{i}"].to_numpy().astype(int)
        acc.append(float((bh==bt).mean()))
    exact=np.ones(len(wm),dtype=bool)
    for i in range(4):
        exact &= ((wm[f"bit_prob{i}"].to_numpy()>=.5).astype(int)==wm[f"true_bit{i}"].to_numpy().astype(int))
    dtpr,dfpr,dthr=op_at_fpr(y,s,.01)
    return {"n_pairs":int(len(wm)),"auc":auc,"frozen_threshold":thr,
            "tpr_frozen":tpr,"fpr_frozen":fpr,
            "per_bit_accuracy":{f"bit{i}":acc[i] for i in range(4)},
            "mean_bit_accuracy":float(np.mean(acc)),
            "exact_4bit_message_accuracy":float(exact.mean()),
            "diagnostic_tpr_at_le_1pct_fpr":dtpr,"diagnostic_fpr":dfpr,"diagnostic_threshold":dthr}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",default="workspace/paper_experiments/exp_16_coco_fid/sd21_4bit_alpha0155_beta040_n10000")
    ap.add_argument("--lineage-root",default=None)
    ap.add_argument("--checkpoint",default=None)
    ap.add_argument("--threshold",type=float,default=0.9833354949951172)
    ap.add_argument("--vae-id",default="sd2-community/stable-diffusion-2-1-base")
    ap.add_argument("--batch-size",type=int,default=16)
    ap.add_argument("--device",default="cuda")
    a=ap.parse_args()

    root=Path(a.data_root)
    lineage=Path(a.lineage_root) if a.lineage_root else root/"final_lineage_4bit_multigpu"
    ck=Path(a.checkpoint) if a.checkpoint else root/"final_detector_4bit"/"detector_best.pt"
    out=root/"final_lineage_4bit_evaluation"; out.mkdir(parents=True,exist_ok=True)
    dev=torch.device(a.device)

    model=Detector4Bit().to(dev)
    state=torch.load(ck,map_location=dev)
    # Support common checkpoint wrappers.
    sd=state.get("model_state_dict",state.get("model",state.get("state_dict",state))) if isinstance(state,dict) else state
    model.load_state_dict(sd,strict=True); model.eval()

    vae=AutoencoderKL.from_pretrained(a.vae_id,subfolder="vae",torch_dtype=torch.float32).to(dev).eval()

    base=pd.read_csv(root/"manifest_final_10000.csv")
    base=base[base.split.astype(str).str.lower().str.strip()=="test"].copy().sort_values("sample_id")
    # G0 uses manifest's exact clean_path / wm4_path.
    g0=base[["sample_id","clean_path","wm4_path","bit0","bit1","bit2","bit3"]].copy()
    g0=g0.rename(columns={"wm4_path":"wm_path"})
    depths={"g0":g0}
    for d in ["g1","g2","g3","g4"]:
        depths[d]=pd.read_csv(lineage/f"manifest_{d}.csv").sort_values("sample_id")

    results={}
    for d,df in depths.items():
        print("\n"+"="*72+f"\nEVALUATING {d.upper()}\n"+"="*72)
        pred=infer_pairs(df,model,vae,dev,a.batch_size)
        pred.to_csv(out/f"predictions_{d}.csv",index=False)
        m=metrics(pred,a.threshold); m["depth"]=d
        results[d]=m
        print(json.dumps(m,indent=2))

    rows=[]
    for d,m in results.items():
        rows.append({"depth":d,"n_pairs":m["n_pairs"],"auc":m["auc"],
          "frozen_tpr":m["tpr_frozen"],"frozen_fpr":m["fpr_frozen"],
          "bit0":m["per_bit_accuracy"]["bit0"],"bit1":m["per_bit_accuracy"]["bit1"],
          "bit2":m["per_bit_accuracy"]["bit2"],"bit3":m["per_bit_accuracy"]["bit3"],
          "mean_bit_accuracy":m["mean_bit_accuracy"],
          "exact_4bit_message_accuracy":m["exact_4bit_message_accuracy"],
          "diagnostic_tpr_at_le_1pct_fpr":m["diagnostic_tpr_at_le_1pct_fpr"],
          "diagnostic_fpr":m["diagnostic_fpr"]})
    pd.DataFrame(rows).to_csv(out/"lineage_4bit_metrics_summary.csv",index=False)
    (out/"results_g0_g4.json").write_text(json.dumps(results,indent=2))
    print("\nDONE:",out)

if __name__=="__main__":
    main()
