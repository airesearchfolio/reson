#!/usr/bin/env python3
"""
Train/evaluate RESON 4-bit joint presence + payload decoder.

Final protocol:
  8000 train / 1000 val / 1000 held-out test pairs
  checkpoint selected only by validation presence AUC
  presence threshold calibrated only on validation at <=1% FPR
  held-out test uses frozen validation threshold
  reports per-bit accuracy, mean bit accuracy, and exact 4-bit message accuracy
"""
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd
from PIL import Image
import torch,torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL
from sklearn.metrics import roc_auc_score,roc_curve
from tqdm import tqdm

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",default="workspace/paper_experiments/reson_sd21_4bit_alpha0155_beta040_10k/manifest.csv")
    p.add_argument("--out-dir",default="workspace/paper_experiments/reson_sd21_4bit_alpha0155_beta040_10k/final_detector")
    p.add_argument("--model-id",default="sd2-community/stable-diffusion-2-1-base")
    p.add_argument("--device",default="cuda")
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--patience",type=int,default=8)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--weight-decay",type=float,default=1e-5)
    p.add_argument("--presence-weight",type=float,default=1.0)
    p.add_argument("--payload-weight",type=float,default=1.0)
    p.add_argument("--margin",type=float,default=2.0)
    p.add_argument("--target-fpr",type=float,default=.01)
    p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--seed",type=int,default=20260920)
    return p.parse_args()

def seedall(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)

class Detector4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(4,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,1,1),nn.BatchNorm2d(256),nn.ReLU(True),nn.MaxPool2d(2))
        d=256*4*4
        self.pres=nn.Sequential(nn.Linear(d,512),nn.ReLU(True),nn.Linear(512,1))
        self.bits=nn.Sequential(nn.Linear(d,512),nn.ReLU(True),nn.Linear(512,4))
    def forward(self,x):
        f=torch.flatten(self.conv(x),1)
        return self.bits(f),self.pres(f).squeeze(1)

class VAE:
    def __init__(self,mid,dev):
        self.dev=dev
        self.vae=AutoencoderKL.from_pretrained(mid,subfolder="vae",torch_dtype=torch.float32).to(dev).eval()
        self.vae.requires_grad_(False);self.scale=float(self.vae.config.scaling_factor)
    @torch.no_grad()
    def encode(self,x):
        return (self.vae.encode(x.to(self.dev,dtype=torch.float32)).latent_dist.sample()*self.scale).float()

class DS(Dataset):
    def __init__(self,pairs):
        rr=[]
        for _,r in pairs.iterrows():
            bits=[int(r[f"bit{i}"]) for i in range(4)]
            rr += [
              dict(sample_id=int(r.sample_id),path=r.clean_image_path,pres=0,bits=bits,kind="clean"),
              dict(sample_id=int(r.sample_id),path=r.wm_image_path,pres=1,bits=bits,kind="wm")]
        self.df=pd.DataFrame(rr)
        self.tf=transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor(),
                                    transforms.Normalize([.5]*3,[.5]*3)])
    def __len__(self):return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        with Image.open(r.path) as im:x=self.tf(im.convert("RGB"))
        return x,torch.tensor(r.bits,dtype=torch.float32),torch.tensor(float(r.pres)),i

def dl(pairs,bs,nw,shuffle):
    return DataLoader(DS(pairs),batch_size=bs,shuffle=shuffle,num_workers=nw,
                      pin_memory=True,persistent_workers=nw>0)

def margin_loss(logit,y,m):
    return torch.clamp(m-logit*(2*y-1),min=0).mean()

@torch.no_grad()
def infer(model,vae,pairs,bs,nw,desc):
    ds=DS(pairs); loader=DataLoader(ds,batch_size=bs,shuffle=False,num_workers=nw,
                                    pin_memory=True,persistent_workers=nw>0)
    out=[];model.eval()
    for x,b,p,idx in tqdm(loader,desc=desc,leave=False):
        bl,pl=model(vae.encode(x)); bp=torch.sigmoid(bl).cpu().numpy(); pp=torch.sigmoid(pl).cpu().numpy()
        for j,k in enumerate(idx.numpy()):
            r=ds.df.iloc[int(k)]
            z=dict(sample_id=int(r.sample_id),kind=r.kind,presence=int(r.pres),presence_score=float(pp[j]))
            for q in range(4):
                z[f"bit{q}"]=int(r.bits[q]);z[f"bit{q}_prob"]=float(bp[j,q])
            out.append(z)
    return pd.DataFrame(out)

def op(df,target=.01):
    y=df.presence.to_numpy(int);s=df.presence_score.to_numpy(float)
    auc=float(roc_auc_score(y,s));fpr,tpr,thr=roc_curve(y,s)
    ok=np.where(fpr<=target+1e-12)[0];k=ok[np.argmax(tpr[ok])]
    return dict(auc=auc,tpr=float(tpr[k]),fpr=float(fpr[k]),threshold=float(thr[k]))

def payload(df):
    w=df[df.presence==1].copy()
    per=[]
    pred=[];true=[]
    for q in range(4):
        t=w[f"bit{q}"].to_numpy(int);p=(w[f"bit{q}_prob"].to_numpy()>=.5).astype(int)
        per.append(float((p==t).mean()));pred.append(p);true.append(t)
    pred=np.stack(pred,1);true=np.stack(true,1)
    return {"per_bit_accuracy":per,"mean_bit_accuracy":float(np.mean(per)),
            "message_accuracy":float(np.all(pred==true,axis=1).mean())}

def main():
    a=args();seedall(a.seed)
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    dev=torch.device(a.device);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.manifest)
    req=["sample_id","split","clean_image_path","wm_image_path"]+[f"bit{i}" for i in range(4)]
    miss=[x for x in req if x not in df.columns]
    if miss:raise RuntimeError(f"Missing columns {miss}")
    df["split"]=df["split"].astype(str).str.lower().str.strip()
    tr=df[df.split=="train"];va=df[df.split.isin(["val","validation"])];te=df[df.split=="test"]
    if (len(tr),len(va),len(te))!=(8000,1000,1000):
        raise RuntimeError(f"Final training requires 8000/1000/1000 pairs; got {len(tr)}/{len(va)}/{len(te)}")
    print("pairs train/val/test:",len(tr),len(va),len(te))

    vae=VAE(a.model_id,dev);model=Detector4().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    bce=nn.BCEWithLogitsLoss();loader=dl(tr,a.batch_size,a.num_workers,True)
    best=-1;stale=0;bestep=0;ck=out/"detector4_best.pt";hist=[]

    for ep in range(1,a.epochs+1):
        model.train();losses=[]
        bar=tqdm(loader,desc=f"epoch {ep:02d}")
        for x,b,p,_ in bar:
            b=b.to(dev);p=p.to(dev);bl,pl=model(vae.encode(x))
            lp=margin_loss(pl,p,a.margin);mask=p>.5
            lb=bce(bl[mask],b[mask]) if mask.any() else torch.zeros((),device=dev)
            loss=a.presence_weight*lp+a.payload_weight*lb
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            losses.append(loss.item());bar.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")
        vp=infer(model,vae,va,a.batch_size,a.num_workers,"val");vo=op(vp,a.target_fpr);pay=payload(vp)
        rec={"epoch":ep,"loss":float(np.mean(losses)),"val_auc":vo["auc"],"val_tpr":vo["tpr"],
             "val_fpr":vo["fpr"],"val_threshold":vo["threshold"],
             "val_mean_bit_acc":pay["mean_bit_accuracy"],"val_message_acc":pay["message_accuracy"]}
        hist.append(rec);pd.DataFrame(hist).to_csv(out/"history.csv",index=False)
        print(f"Epoch {ep}: AUC={vo['auc']:.6f} TPR={vo['tpr']:.4f} FPR={vo['fpr']:.4f} "
              f"bit={pay['mean_bit_accuracy']:.4f} msg4={pay['message_accuracy']:.4f}")
        if vo["auc"]>best+1e-6:
            best=vo["auc"];bestep=ep;stale=0
            torch.save({"model_state_dict":model.state_dict(),"epoch":ep,"val_auc":best,
                        "payload_bits":4,"architecture":"RESON Detector4 shared conv + 4-logit payload head + presence head"},ck)
        else:
            stale+=1
            if stale>=a.patience:break

    state=torch.load(ck,map_location=dev);model.load_state_dict(state["model_state_dict"])
    seedall(a.seed+1000)
    vp=infer(model,vae,va,a.batch_size,a.num_workers,"FINAL val");vo=op(vp,a.target_fpr)
    tp=infer(model,vae,te,a.batch_size,a.num_workers,"HELD-OUT test")
    y=tp.presence.to_numpy(int);s=tp.presence_score.to_numpy(float);pred=s>=vo["threshold"]
    pos=y==1;neg=y==0
    primary={"auc":float(roc_auc_score(y,s)),"tpr":float(pred[pos].mean()),"fpr":float(pred[neg].mean())}
    primary.update(payload(tp));diag=op(tp,a.target_fpr)
    vp.to_csv(out/"val_predictions.csv",index=False);tp.to_csv(out/"test_predictions.csv",index=False)
    result={"best_epoch":bestep,"validation":vo,"test_frozen_validation_threshold":primary,
            "test_roc_diagnostic_only":diag,"payload_bits":4,
            "protocol":"8K train / 1K val / 1K held-out test; threshold selected only on validation"}
    (out/"results.json").write_text(json.dumps(result,indent=2))
    print("\nFINAL 4-BIT RESULTS")
    print("best epoch:",bestep)
    print("frozen threshold:",vo["threshold"])
    print("test AUC/TPR/FPR:",primary["auc"],primary["tpr"],primary["fpr"])
    print("per-bit accuracy:",primary["per_bit_accuracy"])
    print("mean bit accuracy:",primary["mean_bit_accuracy"])
    print("exact 4-bit message accuracy:",primary["message_accuracy"])
    print("results:",out/"results.json")

if __name__=="__main__":
    main()
