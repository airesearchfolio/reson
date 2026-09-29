#!/usr/bin/env python3
import argparse,random
import numpy as np,pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import roc_auc_score,roc_curve
from tqdm import tqdm
from config import Config
from detector import PayloadDecoder,VAEEncoder,margin_presence_loss

def seedall(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def build_df(m,split,bits):
    g=m[m["split"].astype(str)==split].copy()
    p=g[["prompt_id",*bits,"wm_image_path"]].copy(); p["presence"]=1; p["image_path"]=p.wm_image_path
    n=g[["prompt_id",*bits,"clean_image_path"]].copy(); n["presence"]=0; n["image_path"]=n.clean_image_path
    return pd.concat([p[["prompt_id",*bits,"presence","image_path"]],n[["prompt_id",*bits,"presence","image_path"]]],ignore_index=True)

class DS(Dataset):
    def __init__(self,df,size,bits,aug=False):
        self.df=df.reset_index(drop=True); self.bits=bits
        ops=[transforms.Resize((size,size))]
        if aug: ops.append(transforms.RandomHorizontalFlip(.5))
        ops.append(transforms.ToTensor()); self.tf=transforms.Compose(ops)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        with Image.open(r.image_path) as im: x=self.tf(im.convert("RGB"))*2-1
        b=torch.tensor([float(r[c]) for c in self.bits],dtype=torch.float32)
        return x,b,torch.tensor(float(r.presence))

def sampler(df):
    np_=int((df.presence==1).sum()); nn_=int((df.presence==0).sum())
    return WeightedRandomSampler([.5/np_ if int(p)==1 else .5/nn_ for p in df.presence],len(df),replacement=True)

def tpr1(y,s):
    f,t,_=roc_curve(y,s); ids=np.where(f<=.01+1e-12)[0]; return float(t[ids[np.argmax(t[ids])]])

@torch.inference_mode()
def evaluate(model,vae,df,cfg,bits,bs):
    dl=DataLoader(DS(df,cfg.image_size,bits),batch_size=bs,shuffle=False,num_workers=4,pin_memory=True)
    ys=[];ps=[];ba=[];ex=[];ham=[]
    for x,b,pres in dl:
        bl,pl=model(vae.encode_batch(x).to(cfg.get_device()))
        ys.extend(pres.numpy()); ps.extend(torch.sigmoid(pl).cpu().numpy()); pm=pres.to(cfg.get_device())>.5
        if pm.any():
            pr=torch.sigmoid(bl[pm])>=.5; tr=b.to(cfg.get_device())[pm]>=.5; c=pr==tr
            ba.extend(c.float().mean(1).cpu().numpy()); ex.extend(c.all(1).float().cpu().numpy()); ham.extend((~c).sum(1).float().cpu().numpy())
    y=np.asarray(ys,int);s=np.asarray(ps,float);mb=float(np.mean(ba))
    return {"auc":float(roc_auc_score(y,s)),"tpr1":tpr1(y,s),"mean_bit_acc":mb,"BER":1-mb,
            "exact_payload":float(np.mean(ex)),"mean_hamming":float(np.mean(ham))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--bits",type=int,required=True,choices=[1,2,4,8,16,32])
    ap.add_argument("--epochs",type=int,default=30);ap.add_argument("--batch-size",type=int,default=32);ap.add_argument("--lr",type=float,default=5e-5)
    a=ap.parse_args();k=a.bits;cfg=Config();cfg.make_dirs(k);seedall(cfg.seed+k)
    m=pd.read_csv(cfg.source_manifest(k));bits=[f"bit{i}" for i in range(k)]
    tr=build_df(m,"train",bits);va=build_df(m,"val",bits)
    vae=VAEEncoder(cfg);model=PayloadDecoder(cfg.watermark.latent_channels,k).to(cfg.get_device())
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5)
    dl=DataLoader(DS(tr,cfg.image_size,bits,True),batch_size=a.batch_size,sampler=sampler(tr),num_workers=4,pin_memory=True)
    hist=[];best=None
    for ep in range(1,a.epochs+1):
        model.train()
        for x,b,pres in tqdm(dl,desc=f"K{k} {ep}/{a.epochs}"):
            b=b.to(cfg.get_device());pres=pres.to(cfg.get_device())
            with torch.no_grad():z=vae.encode_batch(x).to(cfg.get_device())
            bl,pl=model(z);lp=margin_presence_loss(pl,pres,cfg.training.margin);pm=pres>.5
            lb=F.binary_cross_entropy_with_logits(bl[pm],b[pm]) if pm.any() else torch.zeros((),device=cfg.get_device())
            loss=cfg.training.presence_weight*lp+cfg.training.payload_weight*lb
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        met=evaluate(model,vae,va,cfg,bits,a.batch_size);hist.append({"epoch":ep,**met})
        pd.DataFrame(hist).to_csv(cfg.checkpoint_dir(k)/"history.csv",index=False)
        print(f"Epoch {ep:02d} AUC={met['auc']:.4f} TPR1={met['tpr1']:.4f} bit={met['mean_bit_acc']:.4f} BER={met['BER']:.4f} exactK={met['exact_payload']:.4f} Hamming={met['mean_hamming']:.3f}")
        key=(met["mean_bit_acc"],met["exact_payload"],met["tpr1"])
        if best is None or key>best[0]:
            best=(key,ep,met);torch.save({"model_state_dict":model.state_dict(),"epoch":ep,"metrics":met,"n_bits":k},cfg.checkpoint_dir(k)/"detector_best_payload.pt")
    print("BEST:",best)

if __name__=="__main__":main()
