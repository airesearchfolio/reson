#!/usr/bin/env python3
import argparse,numpy as np,pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score,roc_curve
from config import Config
from detector import PayloadDecoder,VAEEncoder

class DS(Dataset):
    def __init__(self,df,cfg,bits):
        self.df=df.reset_index(drop=True);self.bits=bits
        self.tf=transforms.Compose([transforms.Resize((cfg.image_size,cfg.image_size)),transforms.ToTensor()])
    def __len__(self):return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        with Image.open(r.image_path) as im:x=self.tf(im.convert("RGB"))*2-1
        return x,torch.tensor([float(r[c]) for c in self.bits]),torch.tensor(float(r.presence))
def tpr1(y,s):
    f,t,_=roc_curve(y,s);ids=np.where(f<=.01+1e-12)[0];return float(t[ids[np.argmax(t[ids])]])
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--bits",type=int,required=True,choices=[1,2,4,8,16,32]);ap.add_argument("--batch-size",type=int,default=32)
    a=ap.parse_args();k=a.bits;cfg=Config();cfg.make_dirs(k);bits=[f"bit{i}" for i in range(k)]
    m=pd.read_csv(cfg.source_manifest(k));g=m[m["split"].astype(str)=="test"]
    p=g[["prompt_id",*bits,"wm_image_path"]].copy();p["presence"]=1;p["image_path"]=p.wm_image_path
    n=g[["prompt_id",*bits,"clean_image_path"]].copy();n["presence"]=0;n["image_path"]=n.clean_image_path
    df=pd.concat([p[["prompt_id",*bits,"presence","image_path"]],n[["prompt_id",*bits,"presence","image_path"]]],ignore_index=True)
    ck=cfg.checkpoint_dir(k)/"detector_best_payload.pt";raw=torch.load(ck,map_location="cpu",weights_only=False)
    model=PayloadDecoder(cfg.watermark.latent_channels,k).to(cfg.get_device());model.load_state_dict(raw["model_state_dict"]);model.eval();vae=VAEEncoder(cfg)
    ys=[];ps=[];ba=[];ex=[];ham=[]
    with torch.no_grad():
        for x,b,pres in DataLoader(DS(df,cfg,bits),batch_size=a.batch_size,shuffle=False,num_workers=4,pin_memory=True):
            bl,pl=model(vae.encode_batch(x).to(cfg.get_device()));ys.extend(pres.numpy());ps.extend(torch.sigmoid(pl).cpu().numpy());pm=pres.to(cfg.get_device())>.5
            if pm.any():
                pr=torch.sigmoid(bl[pm])>=.5;tr=b.to(cfg.get_device())[pm]>=.5;c=pr==tr
                ba.extend(c.float().mean(1).cpu().numpy());ex.extend(c.all(1).float().cpu().numpy());ham.extend((~c).sum(1).float().cpu().numpy())
    y=np.asarray(ys,int);s=np.asarray(ps,float);mb=float(np.mean(ba))
    met={"n_bits":k,"n_test_wm":len(g),"auc":float(roc_auc_score(y,s)),"tpr1":tpr1(y,s),"mean_bit_acc":mb,"BER":1-mb,"exact_payload":float(np.mean(ex)),"mean_hamming":float(np.mean(ham))}
    print(met);pd.DataFrame([met]).to_csv(cfg.eval_dir(k)/"summary.csv",index=False)
if __name__=="__main__":main()
