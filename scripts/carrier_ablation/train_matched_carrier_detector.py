#!/usr/bin/env python3
"""
RESON SD2.1 controlled detector test.

Purpose
-------
Reuse the previous RESON LatentJointDecoder architecture:
  Conv 4->32->64->128->256, BN+ReLU+MaxPool at every block,
  flatten 256*4*4,
  separate 4096->512->1 presence and bit heads.

Only source-specific encoder is changed to SD2.1 VAE.

IMPORTANT
---------
- Reuses EXISTING SD2.1 images; does not regenerate anything.
- Reuses the EXISTING pilot train/val/test pair CSVs, so this is a controlled
  architecture/training comparison against the first SD2.1 pilot.
- SD2.1 VAE uses posterior sampling, matching the previous RESON VAEEncoder
  behavior (rather than posterior mean).
- Presence uses margin loss by default, matching the previous detector design.
- Best checkpoint selected by validation presence AUC.
- Primary test TPR uses the threshold calibrated on validation and frozen.
- Test-set ROC @ <=1% FPR is diagnostic only.

Default run:
CUDA_VISIBLE_DEVICES=7 python train_reson_sd21_previous_detector_pilot.py \
  --manifest workspace/paper_experiments/reson_sd21_alpha0155_1k/manifest.csv \
  --split-dir workspace/paper_experiments/reson_sd21_alpha0155_1k/pilot_detector \
  --out-dir workspace/paper_experiments/reson_sd21_alpha0155_1k/previous_reson_detector_pilot
"""
import argparse, json, random
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--split-dir",required=True)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--sd21-model-id",default="sd2-community/stable-diffusion-2-1-base")
    p.add_argument("--device",default="cuda")
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--patience",type=int,default=8)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--weight-decay",type=float,default=1e-5)
    p.add_argument("--presence-weight",type=float,default=1.0)
    p.add_argument("--bit-weight",type=float,default=1.0)
    p.add_argument("--margin",type=float,default=2.0)
    p.add_argument("--target-fpr",type=float,default=.01)
    p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--seed",type=int,default=20260919)
    return p.parse_args()

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class Detector(nn.Module):
    # Previous RESON LatentJointDecoder architecture.
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

class SD21VAE:
    def __init__(self,mid,device):
        self.device=device
        print("Loading SD2.1 VAE:",mid)
        # float32 is deliberate for numerical stability.
        self.vae=AutoencoderKL.from_pretrained(mid,subfolder="vae",torch_dtype=torch.float32).to(device).eval()
        self.vae.requires_grad_(False)
        self.scale=float(self.vae.config.scaling_factor)
    @torch.no_grad()
    def encode(self,x):
        x=x.to(self.device,dtype=torch.float32)
        # Previous RESON VAEEncoder sampled posterior rather than taking mean.
        return (self.vae.encode(x).latent_dist.sample()*self.scale).float()

class PairRows(Dataset):
    def __init__(self,pairs):
        rows=[]
        for _,r in pairs.iterrows():
            cleanp = getattr(r,"clean_image_path",getattr(r,"clean_path",None))
            wmp = getattr(r,"wm_image_path",getattr(r,"wm_path",None))
            rows.append(dict(sample_id=r.sample_id,image_path=cleanp,presence=0,bit=int(r.bit),kind="clean"))
            rows.append(dict(sample_id=r.sample_id,image_path=wmp,presence=1,bit=int(r.bit),kind="wm"))
        self.df=pd.DataFrame(rows)
        self.tf=transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor(),
                                    transforms.Normalize([.5]*3,[.5]*3)])
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        with Image.open(r.image_path) as im: x=self.tf(im.convert("RGB"))
        return x,torch.tensor(float(r.bit)),torch.tensor(float(r.presence)),i

def loader(pairs,bs,nw,shuffle):
    return DataLoader(PairRows(pairs),batch_size=bs,shuffle=shuffle,num_workers=nw,
                      pin_memory=True,persistent_workers=(nw>0))

def margin_loss(logit,y,m):
    sign=2*y-1
    return torch.clamp(m-logit*sign,min=0).mean()

@torch.no_grad()
def infer(model,vae,pairs,bs,nw,desc):
    ds=PairRows(pairs); dl=DataLoader(ds,batch_size=bs,shuffle=False,num_workers=nw,pin_memory=True)
    out=[]
    model.eval()
    for x,b,p,idx in tqdm(dl,desc=desc,leave=False):
        lat=vae.encode(x)
        bl,pl=model(lat)
        bp=torch.sigmoid(bl).cpu().numpy(); pp=torch.sigmoid(pl).cpu().numpy()
        for j,k in enumerate(idx.numpy()):
            r=ds.df.iloc[int(k)]
            out.append(dict(sample_id=r.sample_id,kind=r.kind,presence=int(r.presence),bit=int(r.bit),
                            presence_score=float(pp[j]),bit_prob=float(bp[j])))
    return pd.DataFrame(out)

def op(df,target=.01):
    y=df.presence.to_numpy(int); s=df.presence_score.to_numpy(float)
    auc=float(roc_auc_score(y,s)); fpr,tpr,thr=roc_curve(y,s)
    ok=np.where(fpr<=target+1e-12)[0]; best=ok[np.argmax(tpr[ok])]
    return dict(auc=auc,tpr=float(tpr[best]),fpr=float(fpr[best]),threshold=float(thr[best]))

def metrics_frozen(df,thr):
    y=df.presence.to_numpy(int); s=df.presence_score.to_numpy(float)
    pos=df.presence.eq(1); neg=~pos
    pred=s>=thr
    bitdf=df[pos]
    bitacc=float(((bitdf.bit_prob.to_numpy()>=.5).astype(int)==bitdf.bit.to_numpy(int)).mean())
    return dict(auc=float(roc_auc_score(y,s)),
                tpr=float(pred[pos.to_numpy()].mean()),fpr=float(pred[neg.to_numpy()].mean()),
                bit_accuracy=bitacc,mean_clean=float(df.loc[neg,"presence_score"].mean()),
                mean_wm=float(df.loc[pos,"presence_score"].mean()))

def main():
    a=args(); seed_all(a.seed)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    dev=torch.device(a.device)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    split=Path(a.split_dir)
    train=pd.read_csv(split/"train_pairs.csv"); val=pd.read_csv(split/"val_pairs.csv"); test=pd.read_csv(split/"test_pairs.csv")
    print(f"Pairs train/val/test: {len(train)}/{len(val)}/{len(test)}")
    print("Using exact existing split CSVs; no split regeneration.")
    vae=SD21VAE(a.sd21_model_id,dev); model=Detector().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    bce=nn.BCEWithLogitsLoss()
    dl=loader(train,a.batch_size,a.num_workers,True)
    best_auc=-1; best_epoch=0; stale=0; history=[]
    ck=out/"detector_best.pt"
    for ep in range(1,a.epochs+1):
        model.train(); losses=[]
        bar=tqdm(dl,desc=f"epoch {ep:02d}",leave=True)
        for x,b,p,_ in bar:
            b=b.to(dev); p=p.to(dev); lat=vae.encode(x)
            bl,pl=model(lat)
            lp=margin_loss(pl,p,a.margin)
            mask=p>.5
            lb=bce(bl[mask],b[mask]) if mask.any() else torch.zeros((),device=dev)
            loss=a.presence_weight*lp+a.bit_weight*lb
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.item()); bar.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")
        vp=infer(model,vae,val,a.batch_size,a.num_workers,"validation")
        vo=op(vp,a.target_fpr)
        vwm=vp[vp.presence==1]
        vb=float(((vwm.bit_prob.to_numpy()>=.5).astype(int)==vwm.bit.to_numpy(int)).mean())
        history.append(dict(epoch=ep,loss=float(np.mean(losses)),val_auc=vo["auc"],
                            val_tpr=vo["tpr"],val_fpr=vo["fpr"],val_threshold=vo["threshold"],val_bit=vb))
        pd.DataFrame(history).to_csv(out/"history.csv",index=False)
        print(f"Epoch {ep:02d}: loss={np.mean(losses):.5f} AUC={vo['auc']:.4f} "
              f"TPR={vo['tpr']:.4f} FPR={vo['fpr']:.4f} bit={vb:.4f}")
        if vo["auc"]>best_auc+1e-6:
            best_auc=vo["auc"]; best_epoch=ep; stale=0
            torch.save({"model_state_dict":model.state_dict(),"epoch":ep,"val_auc":best_auc},ck)
        else:
            stale+=1
            if stale>=a.patience:
                print("Early stopping."); break
    state=torch.load(ck,map_location=dev); model.load_state_dict(state["model_state_dict"]); model.eval()
    # Reset RNG before final scoring to make posterior-sampling evaluation reproducible.
    seed_all(a.seed+1000)
    vp=infer(model,vae,val,a.batch_size,a.num_workers,"final validation"); vo=op(vp,a.target_fpr)
    tp=infer(model,vae,test,a.batch_size,a.num_workers,"held-out test")
    primary=metrics_frozen(tp,vo["threshold"]); diagnostic=op(tp,a.target_fpr)
    vp.to_csv(out/"val_predictions.csv",index=False); tp.to_csv(out/"test_predictions.csv",index=False)
    result={"best_epoch":best_epoch,"validation":vo,
            "test_frozen_validation_threshold":primary,"test_roc_diagnostic_only":diagnostic,
            "architecture":"previous RESON LatentJointDecoder: conv 4-32-64-128-256, flatten, 512-unit separate heads",
            "vae":"SD2.1 VAE posterior sample","presence_loss":f"margin={a.margin}",
            "split_counts":{"train_pairs":len(train),"val_pairs":len(val),"test_pairs":len(test)}}
    (out/"results.json").write_text(json.dumps(result,indent=2))
    print("\n================ CONTROLLED RESULTS ================")
    print("Best epoch                :",best_epoch)
    print(f"Validation AUC            : {vo['auc']:.6f}")
    print(f"Validation threshold      : {vo['threshold']:.6f}")
    print(f"Validation TPR/FPR        : {vo['tpr']:.4f} / {vo['fpr']:.4f}")
    print("--- HELD-OUT TEST, FROZEN VAL THRESHOLD ---")
    print(f"Test AUC                  : {primary['auc']:.6f}")
    print(f"Test TPR                  : {primary['tpr']:.4f}")
    print(f"Test realized FPR         : {primary['fpr']:.4f}")
    print(f"Test payload bit accuracy : {primary['bit_accuracy']:.4f}")
    print(f"Mean clean score          : {primary['mean_clean']:.6f}")
    print(f"Mean WM score             : {primary['mean_wm']:.6f}")
    print("--- TEST ROC DIAGNOSTIC (NOT PRIMARY) ---")
    print(f"TPR@<={100*a.target_fpr:.1f}%FPR             : {diagnostic['tpr']:.4f}")
    print(f"ROC threshold             : {diagnostic['threshold']:.6f}")
    print("====================================================")
    print("Saved:",out/"results.json")

if __name__=="__main__": main()
