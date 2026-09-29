#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

def tpr_at_fpr(y, score, max_fpr):
    # Exact convention from the user's canonical evaluate.py
    fpr, tpr, thr = roc_curve(y, score)
    ok = fpr <= max_fpr + 1e-12
    if not ok.any():
        return 0.0, float("inf"), 0.0
    ids = np.where(ok)[0]
    k = ids[np.argmax(tpr[ids])]
    return float(tpr[k]), float(thr[k]), float(fpr[k])

class EvalDataset(Dataset):
    def __init__(self, df, image_size):
        self.df=df.reset_index(drop=True)
        self.tf=transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor()])
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        with Image.open(r.image_path) as im:
            x=self.tf(im.convert("RGB"))*2.0-1.0
        return x,torch.tensor(float(r.bit)),torch.tensor(float(r.presence)),i

@torch.inference_mode()
def infer(model,vae,df,cfg,batch_size):
    ds=EvalDataset(df,cfg.image_size)
    dl=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
    rows=[]
    for x,bit,presence,idx in tqdm(dl,desc="detector",leave=False):
        latent=vae.encode_batch(x).to(cfg.get_device())
        bit_logit,presence_logit=model(latent)
        bp=torch.sigmoid(bit_logit).cpu().numpy()
        pp=torch.sigmoid(presence_logit).cpu().numpy()
        for j in range(len(idx)):
            r=ds.df.iloc[int(idx[j])]
            y,b=int(presence[j]),int(bit[j])
            rows.append(dict(prompt_id=int(r.prompt_id),alpha=float(r.alpha),generation_steps=int(r.generation_steps),
                presence=y,bit=b,presence_prob=float(pp[j]),bit_prob=float(bp[j]),
                bit_correct=float((bp[j]>=.5)==bool(b)) if y==1 else np.nan))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--project-root",default="reson")
    ap.add_argument("--checkpoint",default="reson/checkpoints/detector_best.pt")
    ap.add_argument("--batch-size",type=int,default=16)
    ap.add_argument("--out-dir",default=None)
    args=ap.parse_args()
    sys.path.insert(0,args.project_root)
    from config import Config
    from detector import LatentJointDecoder,VAEEncoder

    m=pd.read_csv(args.manifest)
    print("Condition counts:")
    print(m.groupby(["alpha","generation_steps"]).size())
    if len(m)!=2400: print("WARNING: expected 2400 manifest rows, got",len(m))

    # Build exactly the positive/negative layout used by canonical evaluate.py.
    pos=m.copy(); pos["presence"]=1; pos["image_path"]=pos.wm_image_path
    neg=m.copy(); neg["presence"]=0; neg["image_path"]=neg.clean_image_path
    ev=pd.concat([pos,neg],ignore_index=True)

    cfg=Config()
    model=LatentJointDecoder.load(cfg,path=args.checkpoint,device=str(cfg.get_device()))
    model.eval()
    vae=VAEEncoder(cfg)
    per=infer(model,vae,ev,cfg,args.batch_size)

    out=Path(args.out_dir) if args.out_dir else Path(args.manifest).parent/"screening_detector_exact"
    out.mkdir(parents=True,exist_ok=True)
    per.to_csv(out/"per_image_detector_scores.csv",index=False)

    rows=[]
    for (a,st),g in per.groupby(["alpha","generation_steps"],sort=True):
        y=g.presence.to_numpy(int); s=g.presence_prob.to_numpy(float)
        tpr,thr,fpr=tpr_at_fpr(y,s,.01)
        po=g[g.presence==1]; ne=g[g.presence==0]
        rows.append(dict(alpha=a,steps=int(st),n_positive=len(po),n_negative=len(ne),
            roc_auc=float(roc_auc_score(y,s)),TPR_at_1pct_FPR=tpr,
            threshold_at_1pct_FPR=thr,realized_FPR_at_1pct=fpr,
            bit_accuracy=float(po.bit_correct.astype(float).mean()),
            mean_wm_presence_score=float(po.presence_prob.mean()),
            mean_clean_presence_score=float(ne.presence_prob.mean())))
    sm=pd.DataFrame(rows).sort_values(["steps","alpha"])
    sm.to_csv(out/"summary_detector.csv",index=False)
    print("\n=== DETECTOR SUMMARY ===")
    print(sm.to_string(index=False))
    print("\nSaved:",out/"summary_detector.csv")
    print("Saved:",out/"per_image_detector_scores.csv")

if __name__=="__main__": main()
