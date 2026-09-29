#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL
from sklearn.metrics import roc_auc_score,roc_curve
from tqdm import tqdm

ROOT=Path("workspace/reson_sd21_canonical_10k")
CKPT=ROOT/"final_detector/detector_best.pt"
RESULTS=ROOT/"final_detector/results.json"

class Detector(nn.Module):
 def __init__(self):
  super().__init__(); L=[]; ch=[4,32,64,128,256]
  for a,b in zip(ch[:-1],ch[1:]): L += [nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(True),nn.MaxPool2d(2)]
  self.conv_layers=nn.Sequential(*L)
  self.presence_fc=nn.Sequential(nn.Linear(4096,512),nn.ReLU(True),nn.Linear(512,1))
  self.bit_fc=nn.Sequential(nn.Linear(4096,512),nn.ReLU(True),nn.Linear(512,1))
 def forward(self,x):
  h=self.conv_layers(x); h=h.flatten(1); return self.presence_fc(h).squeeze(1),self.bit_fc(h).squeeze(1)

class DS(Dataset):
 def __init__(self,df):
  self.r=[]
  for _,x in df.iterrows():
   for kind,col,label in [("clean","clean_path",0),("wm","wm_path",1)]:
    self.r.append((str(x[col]),label,int(x.bit),int(x.sample_id),kind))
  self.tf=transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])
 def __len__(self): return len(self.r)
 def __getitem__(self,i):
  p,y,b,s,k=self.r[i]; return self.tf(Image.open(p).convert("RGB")),y,b,s,k,p

def threshold_from_json(x):
 keys=["validation_threshold_at_le_1pct_fpr","val_threshold_at_le_1pct_fpr","frozen_threshold","validation_threshold","threshold"]
 for k in keys:
  if k in x and isinstance(x[k],(int,float)): return float(x[k]),k
 for k,v in x.items():
  if isinstance(v,dict):
   try:
    t,n=threshold_from_json(v); return t,k+"."+n
   except RuntimeError: pass
 raise RuntimeError("Frozen threshold not found; pass --frozen-threshold explicitly.")

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--attack-manifest",required=True)
 ap.add_argument("--checkpoint",default=str(CKPT)); ap.add_argument("--detector-results",default=str(RESULTS))
 ap.add_argument("--frozen-threshold",type=float); ap.add_argument("--batch-size",type=int,default=16)
 ap.add_argument("--num-workers",type=int,default=4); ap.add_argument("--device",default="cuda")
 a=ap.parse_args(); mf=Path(a.attack_manifest); df=pd.read_csv(mf)
 need={"sample_id","bit","clean_path","wm_path"}
 if not need<=set(df.columns): raise RuntimeError(f"Missing columns: {need-set(df.columns)}")
 missing=[p for c in ["clean_path","wm_path"] for p in df[c].astype(str) if not Path(p).exists()]
 if missing: raise RuntimeError(f"{len(missing)} images missing, first={missing[:3]}")
 if a.frozen_threshold is None:
  frozen,key=threshold_from_json(json.loads(Path(a.detector_results).read_text())); print("Frozen threshold source:",key)
 else: frozen=a.frozen_threshold
 dev=torch.device(a.device if torch.cuda.is_available() else "cpu"); print("device:",dev,"pairs:",len(df),"frozen:",frozen)
 vae=AutoencoderKL.from_pretrained("sd2-community/stable-diffusion-2-1-base",subfolder="vae",torch_dtype=torch.float32).to(dev).eval()
 model=Detector().to(dev)
 obj=torch.load(a.checkpoint,map_location="cpu")
 sd=obj.get("model_state_dict",obj.get("state_dict",obj)) if isinstance(obj,dict) else obj
 try: model.load_state_dict(sd,strict=True)
 except RuntimeError as e:
  print("checkpoint keys:",list(sd)[:40]); raise RuntimeError("Checkpoint architecture mismatch; stop and send this error.") from e
 model.eval(); rec=[]
 dl=DataLoader(DS(df),batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=dev.type=="cuda")
 with torch.inference_mode():
  for x,y,b,sid,kind,path in tqdm(dl):
   x=x.to(dev)*2-1
   z=vae.encode(x).latent_dist.sample()*vae.config.scaling_factor
   pl,bl=model(z); ps=torch.sigmoid(pl).cpu().numpy(); bp=torch.sigmoid(bl).cpu().numpy()
   for j in range(len(ps)):
    rec.append({"sample_id":int(sid[j]),"kind":kind[j],"label":int(y[j]),"bit":int(b[j]),"path":path[j],
                "presence_score":float(ps[j]),"bit_prob":float(bp[j]),"bit_pred":int(bp[j]>=.5)})
 pr=pd.DataFrame(rec); yy=pr.label.to_numpy(); ss=pr.presence_score.to_numpy()
 auc=float(roc_auc_score(yy,ss)); clean=pr.label==0; wm=pr.label==1
 tpr=float((pr.loc[wm,"presence_score"]>=frozen).mean()); fpr=float((pr.loc[clean,"presence_score"]>=frozen).mean())
 w=pr[wm].copy(); bitacc=float((w.bit_pred==w.bit).mean()); miss=w.presence_score<frozen; mn=int(miss.sum())
 mba=float((w.loc[miss,"bit_pred"].to_numpy()==w.loc[miss,"bit"].to_numpy()).mean()) if mn else None
 rf,rt,rh=roc_curve(yy,ss); ok=np.where(rf<=.01+1e-12)[0]; j=ok[np.argmax(rt[ok])]
 out={"attack_manifest":str(mf),"n_pairs":len(df),"auc":auc,"frozen_threshold":frozen,"frozen_tpr":tpr,"frozen_fpr":fpr,
      "bit_accuracy":bitacc,"presence_miss_n":mn,"bit_accuracy_among_presence_misses":mba,
      "diagnostic_tpr_at_le_1pct_fpr":float(rt[j]),"diagnostic_fpr":float(rf[j]),"diagnostic_threshold":float(rh[j]),
      "mean_clean_score":float(pr.loc[clean,"presence_score"].mean()),"mean_wm_score":float(pr.loc[wm,"presence_score"].mean()),
      "vae_latent":"posterior sample() * scaling_factor"}
 od=mf.parent/"evaluation"; od.mkdir(exist_ok=True); pr.to_csv(od/"attack_detector_predictions.csv",index=False)
 (od/"attack_metrics.json").write_text(json.dumps(out,indent=2)); pd.DataFrame([out]).to_csv(od/"attack_metrics.csv",index=False)
 print("\n=== RESULTS ===")
 for k,v in out.items():
  if k!="attack_manifest": print(k,":",v)
 print("saved:",od)
if __name__=="__main__": main()
