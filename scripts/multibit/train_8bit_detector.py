#!/usr/bin/env python3
"""
Train/evaluate final RESON 8-bit G0 detector.

Protocol matches the finalized 4-bit detector:
  8000 train / 1000 val / 1000 held-out test
  frozen SD2.1 VAE -> shared conv encoder -> presence + 8-bit payload heads
  checkpoint selected by validation AUC
  <=1% FPR threshold selected on validation and frozen for test
"""
import argparse, csv, json, random
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL
from tqdm import tqdm

def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_rows(root):
    # Preferred merged manifest, if already created.
    candidates = [
        root/"manifest_final_10000.csv",
        root/"manifest_8bit_final_10000.csv",
        root/"manifest_10000.csv",
    ]
    for p in candidates:
        if p.exists():
            with p.open(newline="") as f:
                rows=list(csv.DictReader(f))
            print("Manifest:", p, flush=True)
            return rows

    # Otherwise merge completed worker shards.
    shard_dir=root/"shards"
    files=sorted(shard_dir.glob("manifest_worker_*.csv"))
    if not files:
        raise FileNotFoundError(f"No 8-bit manifest or worker shards found under {root}")
    rows=[]
    for p in files:
        with p.open(newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    # Deduplicate and sort.
    by_id={int(float(r["sample_id"])):r for r in rows}
    rows=[by_id[k] for k in sorted(by_id)]
    if len(rows)!=10000:
        raise RuntimeError(f"Expected 10000 unique rows, found {len(rows)}")
    merged=root/"manifest_final_10000.csv"
    with merged.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("Merged manifest:", merged, flush=True)
    return rows

class PairDataset(Dataset):
    def __init__(self, rows):
        self.items=[]
        for r in rows:
            bits=np.array([int(float(r[f"bit{i}"])) for i in range(8)],dtype=np.float32)
            clean=r.get("clean_path") or r.get("clean_image_path")
            wm=r.get("wm8_path") or r.get("wm_image_path") or r.get("watermarked_image_path")
            if not clean or not wm:
                raise KeyError("Manifest needs clean_path/clean_image_path and wm8_path/wm_image_path.")
            self.items.append((clean,0.0,np.zeros(8,dtype=np.float32)))
            self.items.append((wm,1.0,bits))
        self.tf=transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3),
        ])
    def __len__(self): return len(self.items)
    def __getitem__(self,idx):
        p,pres,bits=self.items[idx]
        im=self.tf(Image.open(p).convert("RGB"))
        return im,torch.tensor(pres,dtype=torch.float32),torch.from_numpy(bits)

class Joint8BitDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(4,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),nn.MaxPool2d(2),
        )
        self.presence=nn.Sequential(nn.Flatten(),nn.Linear(4096,512),nn.ReLU(),nn.Linear(512,1))
        self.payload=nn.Sequential(nn.Flatten(),nn.Linear(4096,512),nn.ReLU(),nn.Linear(512,8))
    def forward(self,z):
        h=self.features(z)
        return self.presence(h).squeeze(1),self.payload(h)

@torch.no_grad()
def encode(vae,x,scaling):
    return (vae.encode(x).latent_dist.sample()*scaling).float()

def threshold_at_fpr(y,s,max_fpr=.01):
    fpr,tpr,thr=roc_curve(y,s)
    ok=np.where(fpr<=max_fpr)[0]
    j=ok[np.argmax(tpr[ok])]
    return float(thr[j]),float(tpr[j]),float(fpr[j])

@torch.no_grad()
def evaluate(model,vae,loader,device,scaling,desc):
    model.eval()
    ys=[]; ss=[]; bt=[]; bp=[]
    for x,y,b in tqdm(loader,desc=desc,leave=False,dynamic_ncols=True):
        x=x.to(device,non_blocking=True); y=y.to(device); b=b.to(device)
        z=encode(vae,x,scaling)
        pl,bl=model(z)
        ys.extend(y.cpu().numpy())
        ss.extend(torch.sigmoid(pl).cpu().numpy())
        mask=y>0.5
        if mask.any():
            bt.append(b[mask].cpu().numpy())
            bp.append(torch.sigmoid(bl[mask]).cpu().numpy())
    y=np.asarray(ys); s=np.asarray(ss)
    true=np.concatenate(bt); prob=np.concatenate(bp)
    pred=(prob>=.5).astype(np.int32); true=true.astype(np.int32)
    per=(pred==true).mean(0)
    return {
        "y":y,"score":s,"auc":float(roc_auc_score(y,s)),
        "per_bit":[float(v) for v in per],
        "mean_bit_accuracy":float((pred==true).mean()),
        "exact_message_accuracy":float(np.all(pred==true,axis=1).mean()),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="workspace/paper_experiments/exp_16_coco_fid/sd21_8bit_alpha0155_beta040_n10000")
    ap.add_argument("--vae",default="sd2-community/stable-diffusion-2-1-base")
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--epochs",type=int,default=30)
    ap.add_argument("--batch-size",type=int,default=32)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--weight-decay",type=float,default=1e-5)
    ap.add_argument("--payload-weight",type=float,default=1.0)
    ap.add_argument("--patience",type=int,default=8)
    ap.add_argument("--seed",type=int,default=20260922)
    ap.add_argument("--num-workers",type=int,default=4)
    args=ap.parse_args()

    seed_all(args.seed)
    root=Path(args.root)
    out=root/"final_detector_8bit"; out.mkdir(parents=True,exist_ok=True)
    rows=load_rows(root)

    # Normalize split spelling and verify exact canonical partition.
    for r in rows: r["split"]=str(r["split"]).lower().strip()
    splits={k:[r for r in rows if r["split"]==k] for k in ("train","val","test")}
    assert [len(splits[k]) for k in ("train","val","test")]==[8000,1000,1000], \
        {k:len(v) for k,v in splits.items()}
    print("Split sizes:",{k:len(v) for k,v in splits.items()},flush=True)

    loaders={}
    for k in splits:
        loaders[k]=DataLoader(
            PairDataset(splits[k]),batch_size=args.batch_size,
            shuffle=(k=="train"),num_workers=args.num_workers,pin_memory=True)

    device=torch.device(args.device)
    vae=AutoencoderKL.from_pretrained(
        args.vae,subfolder="vae",torch_dtype=torch.float32).to(device).eval()
    for p in vae.parameters(): p.requires_grad=False
    scaling=float(vae.config.scaling_factor)

    model=Joint8BitDecoder().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    bce=nn.BCEWithLogitsLoss()

    best_auc=-1.; bad=0
    best_path=out/"detector_best.pt"
    history=[]

    for ep in range(1,args.epochs+1):
        model.train(); losses=[]
        bar=tqdm(loaders["train"],desc=f"Epoch {ep}/{args.epochs}",dynamic_ncols=True)
        for x,y,b in bar:
            x=x.to(device,non_blocking=True); y=y.to(device); b=b.to(device)
            with torch.no_grad(): z=encode(vae,x,scaling)
            pl,bl=model(z)
            lp=bce(pl,y)
            mask=y>0.5
            lb=bce(bl[mask],b[mask]) if mask.any() else torch.zeros((),device=device)
            loss=lp+args.payload_weight*lb
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
            bar.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")

        val=evaluate(model,vae,loaders["val"],device,scaling,"Validation")
        thr,vtpr,vfpr=threshold_at_fpr(val["y"],val["score"])
        rec={
            "epoch":ep,"loss":float(np.mean(losses)),
            "val_auc":val["auc"],"val_tpr_at_1pct":vtpr,"val_fpr":vfpr,
            "val_mean_bit_acc":val["mean_bit_accuracy"],
            "val_exact8_acc":val["exact_message_accuracy"],
        }
        history.append(rec); print(rec,flush=True)

        if val["auc"]>best_auc:
            best_auc=val["auc"]; bad=0
            torch.save({"model":model.state_dict(),"epoch":ep,"val_auc":best_auc},best_path)
        else:
            bad+=1
            if bad>=args.patience:
                print("Early stopping.",flush=True); break

    ck=torch.load(best_path,map_location=device)
    model.load_state_dict(ck["model"])

    val=evaluate(model,vae,loaders["val"],device,scaling,"Final validation")
    threshold,val_tpr,val_fpr=threshold_at_fpr(val["y"],val["score"])

    test=evaluate(model,vae,loaders["test"],device,scaling,"Held-out test")
    pred=test["score"]>=threshold
    test_tpr=float(pred[test["y"]==1].mean())
    test_fpr=float(pred[test["y"]==0].mean())
    diag_thr,diag_tpr,diag_fpr=threshold_at_fpr(test["y"],test["score"])

    result={
        "protocol":"8k train / 1k validation / 1k held-out test",
        "checkpoint_epoch":int(ck["epoch"]),
        "selection_metric":"validation AUC",
        "validation_auc":val["auc"],
        "validation_threshold_at_le_1pct_fpr":threshold,
        "validation_tpr":val_tpr,"validation_fpr":val_fpr,
        "test_auc":test["auc"],
        "test_frozen_threshold_tpr":test_tpr,
        "test_frozen_threshold_fpr":test_fpr,
        "test_per_bit_accuracy":{f"bit{i}":test["per_bit"][i] for i in range(8)},
        "test_mean_bit_accuracy":test["mean_bit_accuracy"],
        "test_exact_8bit_message_accuracy":test["exact_message_accuracy"],
        "test_diagnostic_threshold_at_le_1pct_fpr":diag_thr,
        "test_diagnostic_tpr":diag_tpr,
        "test_diagnostic_fpr":diag_fpr,
        "vae_latent":"posterior sample() * scaling_factor",
        "payload_bits":8,
    }
    (out/"results.json").write_text(json.dumps(result,indent=2))
    (out/"history.json").write_text(json.dumps(history,indent=2))
    print("\nFINAL 8-BIT RESULTS",flush=True)
    print(json.dumps(result,indent=2),flush=True)
    print("Checkpoint:",best_path,flush=True)
    print("Results:",out/"results.json",flush=True)

if __name__=="__main__":
    main()
