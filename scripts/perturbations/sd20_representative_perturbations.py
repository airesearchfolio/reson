#!/usr/bin/env python3
"""
RESON SD2.0 transfer: representative G0 perturbations + frozen detector evaluation.

Conditions:
  JPEG Q70
  Gaussian blur radius=1.0
  Gaussian noise sigma=0.03
  Brightness x1.20
  Rotation 5 degrees
  Crop-and-scale: keep 75% area (center crop), resize back to original size

The same transform is applied independently to matched clean and WM images.
Primary detector threshold stays frozen at 0.500754654.
Also reports ROC-derived TPR at <=1% FPR as a diagnostic.
Multi-GPU: conditions are distributed across GPUs 4,5,6,7.
"""
import argparse, json, math, os, random, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

FROZEN_THRESHOLD=0.500754654
TRANSFER_ROOT=Path("workspace/paper_experiments/reson_sd20_g0_canonical300_transfer")
TRAIN_ROOT=Path("workspace/reson_sd21_canonical_10k")
CONDITIONS=["jpeg_q70","blur_r1","gnoise_s003","brightness_120","rotation_5","crop_scale_075"]

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--transfer-root",type=Path,default=TRANSFER_ROOT)
    p.add_argument("--train-root",type=Path,default=TRAIN_ROOT,
                   help="Original RESON SD2.1 root containing frozen detector checkpoint")
    p.add_argument("--checkpoint",type=Path,default=None)
    p.add_argument("--out-dir",type=Path,default=None)
    p.add_argument("--gpus",default="4,5,6,7")
    p.add_argument("--n",type=int,default=300)
    p.add_argument("--threshold",type=float,default=FROZEN_THRESHOLD)
    p.add_argument("--model-id",default="sd2-community/stable-diffusion-2-1-base")
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--seed",type=int,default=20260924)
    p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    p.add_argument("--worker-id",type=int,default=None,help=argparse.SUPPRESS)
    p.add_argument("--num-workers-gpu",type=int,default=1,help=argparse.SUPPRESS)
    return p.parse_args()

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class LatentJointDecoder(nn.Module):
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
        self.tf=transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor(),
                                    transforms.Normalize([.5]*3,[.5]*3)])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows.iloc[i]
        with Image.open(r["path"]) as im:x=self.tf(im.convert("RGB"))
        return x,i

def wilson(k,n,z=1.959963984540054):
    if n<=0:return [float("nan"),float("nan")]
    q=k/n; den=1+z*z/n
    cen=(q+z*z/(2*n))/den
    h=z*math.sqrt(q*(1-q)/n+z*z/(4*n*n))/den
    return [max(0.,cen-h),min(1.,cen+h)]

def load_g0(a):
    mp=a.transfer_root/"manifests"/"generation_0.csv"
    if not mp.exists():raise FileNotFoundError(mp)
    m=pd.read_csv(mp).drop_duplicates("sample_id").sort_values("sample_id").head(a.n).copy()
    if len(m)!=a.n:raise RuntimeError(f"Expected {a.n} G0 pairs; got {len(m)}")
    clean="clean_path" if "clean_path" in m else "alpha0_image_path"
    wm="wm_path" if "wm_path" in m else "wm_image_path"
    need=["sample_id","bit",clean,wm]
    if any(c not in m for c in need):raise RuntimeError(f"Missing columns. Have {list(m.columns)}")
    return m[["sample_id","bit",clean,wm]].rename(columns={clean:"clean_path",wm:"wm_path"}).reset_index(drop=True)

def transform_image(im,cond,seed):
    im=im.convert("RGB")
    if cond=="jpeg_q70":
        # JPEG round-trip in memory.
        import io
        b=io.BytesIO(); im.save(b,format="JPEG",quality=70,subsampling=0); b.seek(0)
        return Image.open(b).convert("RGB").copy()
    if cond=="blur_r1":
        return im.filter(ImageFilter.GaussianBlur(radius=1.0))
    if cond=="gnoise_s003":
        rng=np.random.default_rng(seed)
        x=np.asarray(im).astype(np.float32)/255.0
        x=np.clip(x+rng.normal(0,0.03,x.shape).astype(np.float32),0,1)
        return Image.fromarray(np.round(x*255).astype(np.uint8))
    if cond=="brightness_120":
        return ImageEnhance.Brightness(im).enhance(1.20)
    if cond=="rotation_5":
        return im.rotate(5.0,resample=Image.Resampling.BICUBIC,expand=False)
    if cond=="crop_scale_075":
        # Keep 75% of image AREA, then resize to original size.
        w,h=im.size; frac=math.sqrt(0.75)
        nw=max(1,int(round(w*frac))); nh=max(1,int(round(h*frac)))
        l=(w-nw)//2; t=(h-nh)//2
        return im.crop((l,t,l+nw,t+nh)).resize((w,h),Image.Resampling.BICUBIC)
    raise ValueError(cond)

def make_condition(base,cond,out_dir,seed):
    idir=out_dir/"images"/cond; idir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for j,r in base.iterrows():
        sid=int(r.sample_id); bit=int(r.bit)
        cp=idir/f"{sid:06d}.png"; wp=idir/f"{sid:06d}_wm.png"
        # For stochastic noise use the SAME noise realization for clean/WM pair.
        pair_seed=seed+sid*1009
        if not cp.exists():
            with Image.open(r.clean_path) as im: transform_image(im,cond,pair_seed).save(cp)
        if not wp.exists():
            with Image.open(r.wm_path) as im: transform_image(im,cond,pair_seed).save(wp)
        rows.append({"condition":cond,"sample_id":sid,"bit":bit,
                     "clean_path":str(cp.resolve()),"wm_path":str(wp.resolve())})
    return pd.DataFrame(rows)

@torch.no_grad()
def infer(rows,model,vae,device,bs,nw):
    ds=ImgDS(rows)
    dl=DataLoader(ds,batch_size=bs,shuffle=False,num_workers=nw,pin_memory=True,
                  persistent_workers=(nw>0))
    ps=np.empty(len(ds),np.float32); bp=np.empty(len(ds),np.float32)
    for x,idx in tqdm(dl,desc="VAE + frozen detector"):
        x=x.to(device,dtype=torch.float32,non_blocking=True)
        lat=vae.encode(x).latent_dist.sample()*float(vae.config.scaling_factor)
        bl,pl=model(lat.float()); ii=idx.numpy()
        ps[ii]=torch.sigmoid(pl).cpu().numpy(); bp[ii]=torch.sigmoid(bl).cpu().numpy()
    z=rows.copy(); z["presence_score"]=ps; z["bit_prob"]=bp
    return z

def metrics(g,thr):
    y=g.presence.to_numpy(int); s=g.presence_score.to_numpy(float)
    auc=float(roc_auc_score(y,s))
    wm=g[g.presence.eq(1)]; cl=g[g.presence.eq(0)]
    wp=wm.presence_score.to_numpy()>=thr; cp=cl.presence_score.to_numpy()>=thr
    truth=wm.bit.to_numpy(int); bhat=(wm.bit_prob.to_numpy()>=.5).astype(int); good=bhat==truth
    fpr,tpr,ths=roc_curve(y,s); ok=np.where(fpr<=.01+1e-12)[0]; j=ok[np.argmax(tpr[ok])]
    return {"n_pairs":len(wm),"auc":auc,"frozen_threshold":thr,
            "tpr_frozen":float(wp.mean()),"realized_fpr_frozen":float(cp.mean()),
            "bit_accuracy":float(good.mean()),
            "bit0_accuracy":float(good[truth==0].mean()) if np.any(truth==0) else None,
            "bit1_accuracy":float(good[truth==1].mean()) if np.any(truth==1) else None,
            "diagnostic_tpr_at_le_1pct_fpr":float(tpr[j]),
            "diagnostic_fpr":float(fpr[j]),"diagnostic_threshold":float(ths[j])}

def worker(a):
    seed_all(a.seed+a.worker_id)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    base=load_g0(a)
    myconds=CONDITIONS[a.worker_id::a.num_workers_gpu]
    ck=a.checkpoint if a.checkpoint else a.train_root/"final_detector"/"detector_best.pt"
    if not ck.exists():raise FileNotFoundError(ck)
    device=torch.device("cuda")
    vae=AutoencoderKL.from_pretrained(a.model_id,subfolder="vae",torch_dtype=torch.float32).to(device).eval()
    vae.requires_grad_(False)
    model=LatentJointDecoder().to(device)
    obj=torch.load(ck,map_location=device)
    state=obj["model_state_dict"] if isinstance(obj,dict) and "model_state_dict" in obj else obj
    model.load_state_dict(state,strict=True); model.eval()
    results=[]; allpred=[]
    for cond in myconds:
        pm=make_condition(base,cond,a.out_dir,a.seed)
        rr=[]
        for _,r in pm.iterrows():
            rr.append({"condition":cond,"sample_id":int(r.sample_id),"bit":int(r.bit),
                       "kind":"clean","presence":0,"path":r.clean_path})
            rr.append({"condition":cond,"sample_id":int(r.sample_id),"bit":int(r.bit),
                       "kind":"wm","presence":1,"path":r.wm_path})
        pred=infer(pd.DataFrame(rr),model,vae,device,a.batch_size,a.num_workers)
        allpred.append(pred)
        q={"condition":cond}; q.update(metrics(pred,a.threshold)); results.append(q)
        print(f"[DONE] {cond}: TPR@<=1%FPR={q['diagnostic_tpr_at_le_1pct_fpr']:.4f} "
              f"AUC={q['auc']:.4f} bit={q['bit_accuracy']:.4f}",flush=True)
    pd.DataFrame(results).to_csv(a.out_dir/f"metrics_worker{a.worker_id}.csv",index=False)
    if allpred:pd.concat(allpred,ignore_index=True).to_csv(a.out_dir/f"predictions_worker{a.worker_id}.csv",index=False)

def launch(a):
    a.out_dir=a.out_dir or a.transfer_root/"sd20_g0_representative_perturbations"
    a.out_dir.mkdir(parents=True,exist_ok=True)
    gpus=[x.strip() for x in a.gpus.split(",") if x.strip()]
    procs=[]; logs=[]
    for i,gpu in enumerate(gpus):
        log=a.out_dir/f"worker_gpu{gpu}_{i}.log"; logs.append(log)
        cmd=[sys.executable,str(Path(__file__).resolve()),"--worker","--worker-id",str(i),
             "--num-workers-gpu",str(len(gpus)),"--transfer-root",str(a.transfer_root),
             "--train-root",str(a.train_root),"--out-dir",str(a.out_dir),
             "--n",str(a.n),"--threshold",str(a.threshold),"--model-id",a.model_id,
             "--batch-size",str(a.batch_size),"--num-workers",str(a.num_workers),"--seed",str(a.seed)]
        if a.checkpoint:cmd+=["--checkpoint",str(a.checkpoint)]
        env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=gpu
        f=open(log,"w"); print(f"[LAUNCH] worker {i} -> GPU {gpu}; log={log}",flush=True)
        procs.append((subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env),f))
    codes=[]
    for p,f in procs:codes.append(p.wait());f.close()
    if any(c!=0 for c in codes):raise RuntimeError(f"Worker failure codes={codes}; inspect {logs}")
    mdfs=[]; pdfs=[]
    for i in range(len(gpus)):
        mf=a.out_dir/f"metrics_worker{i}.csv"
        if mf.exists():mdfs.append(pd.read_csv(mf))
        pf=a.out_dir/f"predictions_worker{i}.csv"
        if pf.exists():pdfs.append(pd.read_csv(pf))
    res=pd.concat(mdfs,ignore_index=True)
    order={c:i for i,c in enumerate(CONDITIONS)}
    res["_ord"]=res.condition.map(order);res=res.sort_values("_ord").drop(columns="_ord")
    res.to_csv(a.out_dir/"sd20_representative_perturbation_metrics.csv",index=False)
    if pdfs:pd.concat(pdfs,ignore_index=True).to_csv(a.out_dir/"sd20_representative_perturbation_predictions.csv",index=False)
    print("\nFINAL SD2.0 REPRESENTATIVE PERTURBATIONS")
    cols=["condition","n_pairs","auc","diagnostic_tpr_at_le_1pct_fpr","tpr_frozen",
          "realized_fpr_frozen","bit_accuracy"]
    print(res[cols].to_string(index=False))
    print("Saved:",a.out_dir/"sd20_representative_perturbation_metrics.csv")

def main():
    a=parse()
    if a.out_dir is None:a.out_dir=a.transfer_root/"sd20_g0_representative_perturbations"
    if a.worker:worker(a)
    else:launch(a)
if __name__=="__main__":main()
