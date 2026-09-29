#!/usr/bin/env python3
"""
RESON SD2.1 FINAL detector training on canonical 10K pairs.

Protocol:
- 8,000 train / 1,000 validation / 1,000 held-out test pairs from manifest `split`
- exact successful previous RESON LatentJointDecoder architecture
- SD2.1 VAE posterior SAMPLE * scaling_factor
- margin presence loss + BCE payload loss
- checkpoint chosen ONLY by validation presence ROC-AUC
- threshold calibrated ONLY on validation at <=1% FPR, then frozen
- held-out test evaluated once with frozen validation threshold
- test ROC <=1% FPR is saved as diagnostic only
- no image generation, no threshold recalibration on test
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

def get_args():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",default="workspace/reson_sd21_canonical_10k/manifest.csv")
    p.add_argument("--out-dir",default="workspace/reson_sd21_canonical_10k/final_detector")
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
        self.vae=AutoencoderKL.from_pretrained(mid,subfolder="vae",torch_dtype=torch.float32).to(device).eval()
        self.vae.requires_grad_(False)
        self.scale=float(self.vae.config.scaling_factor)
    @torch.no_grad()
    def encode(self,x):
        x=x.to(self.device,dtype=torch.float32)
        return (self.vae.encode(x).latent_dist.sample()*self.scale).float()

def normalize_manifest(df):
    aliases={
        "clean_image_path":["clean_image_path","clean_path"],
        "wm_image_path":["wm_image_path","wm_path"],
    }
    for target,opts in aliases.items():
        if target not in df.columns:
            src=next((x for x in opts if x in df.columns),None)
            if src is None: raise RuntimeError(f"Missing {target}; columns={list(df.columns)}")
            df[target]=df[src]
    required=["sample_id","split","bit","clean_image_path","wm_image_path"]
    for c in required:
        if c not in df.columns: raise RuntimeError(f"Missing manifest column: {c}")
    return df

class PairRows(Dataset):
    def __init__(self,pairs):
        rows=[]
        for _,r in pairs.iterrows():
            rows.append(dict(sample_id=int(r.sample_id),image_path=r.clean_image_path,presence=0,bit=int(r.bit),kind="clean"))
            rows.append(dict(sample_id=int(r.sample_id),image_path=r.wm_image_path,presence=1,bit=int(r.bit),kind="wm"))
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
    ds=PairRows(pairs)
    dl=DataLoader(ds,batch_size=bs,shuffle=False,num_workers=nw,pin_memory=True,
                  persistent_workers=(nw>0))
    out=[]; model.eval()
    for x,b,p,idx in tqdm(dl,desc=desc,leave=False):
        lat=vae.encode(x); bl,pl=model(lat)
        bp=torch.sigmoid(bl).cpu().numpy(); pp=torch.sigmoid(pl).cpu().numpy()
        for j,k in enumerate(idx.numpy()):
            r=ds.df.iloc[int(k)]
            out.append(dict(sample_id=int(r.sample_id),kind=r.kind,presence=int(r.presence),
                            bit=int(r.bit),presence_score=float(pp[j]),bit_prob=float(bp[j])))
    return pd.DataFrame(out)

def operating_point(df,target=.01):
    y=df.presence.to_numpy(int); s=df.presence_score.to_numpy(float)
    auc=float(roc_auc_score(y,s)); fpr,tpr,thr=roc_curve(y,s)
    ok=np.where(fpr<=target+1e-12)[0]
    best=ok[np.argmax(tpr[ok])]
    return dict(auc=auc,tpr=float(tpr[best]),fpr=float(fpr[best]),threshold=float(thr[best]))

def frozen_metrics(df,thr):
    y=df.presence.to_numpy(int); s=df.presence_score.to_numpy(float)
    pos=df.presence.eq(1).to_numpy(); neg=~pos; pred=s>=thr
    bitdf=df[df.presence.eq(1)]
    bitacc=float(((bitdf.bit_prob.to_numpy()>=.5).astype(int)==bitdf.bit.to_numpy(int)).mean())
    return dict(auc=float(roc_auc_score(y,s)),tpr=float(pred[pos].mean()),fpr=float(pred[neg].mean()),
                bit_accuracy=bitacc,mean_clean=float(df.loc[df.presence.eq(0),"presence_score"].mean()),
                mean_wm=float(df.loc[df.presence.eq(1),"presence_score"].mean()))

def main():
    a=get_args(); seed_all(a.seed)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    dev=torch.device(a.device)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    df=normalize_manifest(pd.read_csv(a.manifest))
    if len(df)!=10000 or df.sample_id.nunique()!=10000:
        raise RuntimeError(f"Expected 10,000 unique pairs; rows={len(df)}, unique={df.sample_id.nunique()}")
    df["split"]=df["split"].astype(str).str.lower().str.strip()
    train=df[df.split=="train"].copy(); val=df[df.split.isin(["val","validation"])].copy(); test=df[df.split=="test"].copy()
    if (len(train),len(val),len(test))!=(8000,1000,1000):
        raise RuntimeError(f"Expected 8000/1000/1000, got {len(train)}/{len(val)}/{len(test)}")
    if set(train.sample_id)&set(val.sample_id) or set(train.sample_id)&set(test.sample_id) or set(val.sample_id)&set(test.sample_id):
        raise RuntimeError("Pair split leakage detected")
    print("FINAL split pairs train/val/test:",len(train),len(val),len(test))
    print("Images train/val/test:",2*len(train),2*len(val),2*len(test))
    print("Test set is held out from optimization and checkpoint selection.")
    train.to_csv(out/"train_pairs.csv",index=False); val.to_csv(out/"val_pairs.csv",index=False); test.to_csv(out/"test_pairs.csv",index=False)

    vae=SD21VAE(a.sd21_model_id,dev); model=Detector().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    bce=nn.BCEWithLogitsLoss(); dl=loader(train,a.batch_size,a.num_workers,True)
    best_auc=-1.; best_epoch=0; stale=0; history=[]; ck=out/"detector_best.pt"

    for ep in range(1,a.epochs+1):
        model.train(); losses=[]
        bar=tqdm(dl,desc=f"epoch {ep:02d}",leave=True)
        for x,b,p,_ in bar:
            b=b.to(dev); p=p.to(dev); lat=vae.encode(x); bl,pl=model(lat)
            lp=margin_loss(pl,p,a.margin); mask=p>.5
            lb=bce(bl[mask],b[mask]) if mask.any() else torch.zeros((),device=dev)
            loss=a.presence_weight*lp+a.bit_weight*lb
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.item()); bar.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")
        vp=infer(model,vae,val,a.batch_size,a.num_workers,"validation")
        vo=operating_point(vp,a.target_fpr)
        vwm=vp[vp.presence==1]
        vb=float(((vwm.bit_prob.to_numpy()>=.5).astype(int)==vwm.bit.to_numpy(int)).mean())
        history.append(dict(epoch=ep,loss=float(np.mean(losses)),val_auc=vo["auc"],val_tpr=vo["tpr"],
                            val_fpr=vo["fpr"],val_threshold=vo["threshold"],val_bit=vb))
        pd.DataFrame(history).to_csv(out/"history.csv",index=False)
        print(f"Epoch {ep:02d}: loss={np.mean(losses):.5f} AUC={vo['auc']:.6f} "
              f"TPR@<=1%FPR={vo['tpr']:.4f} FPR={vo['fpr']:.4f} bit={vb:.4f}")
        if vo["auc"]>best_auc+1e-6:
            best_auc=vo["auc"]; best_epoch=ep; stale=0
            torch.save({"model_state_dict":model.state_dict(),"epoch":ep,"val_auc":best_auc,
                        "architecture":"RESON LatentJointDecoder 4-32-64-128-256 / 512 heads",
                        "sd21_model_id":a.sd21_model_id},ck)
        else:
            stale+=1
            if stale>=a.patience:
                print("Early stopping."); break

    state=torch.load(ck,map_location=dev); model.load_state_dict(state["model_state_dict"]); model.eval()

    # One deterministic RNG reset for final validation threshold + held-out test scoring.
    seed_all(a.seed+1000)
    vp=infer(model,vae,val,a.batch_size,a.num_workers,"FINAL validation")
    vo=operating_point(vp,a.target_fpr)
    print(f"\nFROZEN validation threshold: {vo['threshold']:.9f} "
          f"(val TPR={vo['tpr']:.4f}, val FPR={vo['fpr']:.4f})")
    print("Now scoring held-out test once with this frozen threshold...")
    tp=infer(model,vae,test,a.batch_size,a.num_workers,"HELD-OUT test")
    primary=frozen_metrics(tp,vo["threshold"])
    diagnostic=operating_point(tp,a.target_fpr)

    vp.to_csv(out/"val_predictions.csv",index=False); tp.to_csv(out/"test_predictions.csv",index=False)
    result={
      "protocol":"FINAL canonical SD2.1 10K; validation-calibrated threshold frozen before held-out test",
      "best_epoch":best_epoch,"validation":vo,
      "test_frozen_validation_threshold":primary,
      "test_roc_diagnostic_only":diagnostic,
      "architecture":"previous RESON LatentJointDecoder: conv 4-32-64-128-256, flatten, separate 4096-512-1 heads",
      "vae":"SD2.1 VAE posterior sample * scaling_factor",
      "presence_loss":f"margin={a.margin}","payload_loss":"BCEWithLogits on WM rows only",
      "target_fpr":a.target_fpr,
      "split_counts":{"train_pairs":len(train),"val_pairs":len(val),"test_pairs":len(test)}
    }
    (out/"results.json").write_text(json.dumps(result,indent=2))

    print("\n================ FINAL G0 DETECTOR RESULTS ================")
    print("Best epoch                :",best_epoch)
    print(f"Validation AUC            : {vo['auc']:.6f}")
    print(f"Validation threshold      : {vo['threshold']:.9f}")
    print(f"Validation TPR/FPR        : {vo['tpr']:.4f} / {vo['fpr']:.4f}")
    print("--- HELD-OUT TEST, FROZEN VALIDATION THRESHOLD ---")
    print(f"Test AUC                  : {primary['auc']:.6f}")
    print(f"Test TPR                  : {primary['tpr']:.4f}")
    print(f"Test realized FPR         : {primary['fpr']:.4f}")
    print(f"Test payload bit accuracy : {primary['bit_accuracy']:.4f}")
    print(f"Mean clean score          : {primary['mean_clean']:.6f}")
    print(f"Mean WM score             : {primary['mean_wm']:.6f}")
    print("--- TEST ROC DIAGNOSTIC ONLY (NOT PRIMARY) ---")
    print(f"Test TPR@<=1%FPR          : {diagnostic['tpr']:.4f}")
    print(f"Diagnostic threshold      : {diagnostic['threshold']:.9f}")
    print("===========================================================")
    print("Checkpoint:",ck)
    print("Results   :",out/"results.json")

if __name__=="__main__":
    main()
