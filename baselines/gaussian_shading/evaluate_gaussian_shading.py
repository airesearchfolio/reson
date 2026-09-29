#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,roc_curve

BASE=Path("workspace/fair_sd21_main_table")
REPO=Path("third_party/Gaussian-Shading")
sys.path.insert(0,str(REPO))
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from watermark import Gaussian_Shading_chacha
from image_utils import transform_img

def col(df,*names):
    for n in names:
        if n in df.columns:return n
    raise RuntimeError(f"Need one of {names}; columns={list(df.columns)}")

def find_manifest(root,d):
    for p in [root/f"generation_{d}.csv",root/"manifests"/f"generation_{d}.csv",
              root/f"manifest_g{d}.csv",root/"manifests"/f"manifest_g{d}.csv"]:
        if p.exists():return p
    raise FileNotFoundError(f"No G{d} manifest under {root}")

def g0_manifest(root,explicit):
    if explicit:return Path(explicit)
    for n in ["generation_0.csv","g0_manifest.csv","manifest.csv","canonical_subset_manifest.csv"]:
        p=root/n
        if p.exists():return p
    raise FileNotFoundError(f"No G0 manifest under {root}")

def load_g0(p):
    x=pd.read_csv(p)
    sid=col(x,"sample_id","audit_id","image_index")
    clean=col(x,"unwatermarked_image_path","clean_image_path","alpha0_image_path","clean_path")
    wm=col(x,"watermarked_image_path","wm_image_path","wm_path")
    key=col(x,"key_state_path")
    idx="image_index" if "image_index" in x else sid
    def vals(n,default,typ=float):
        if n not in x:return pd.Series([default]*len(x))
        return pd.to_numeric(x[n],errors="coerce").fillna(default).astype(typ)
    return pd.DataFrame({
      "sample_id":pd.to_numeric(x[sid]).astype(int),
      "image_index":pd.to_numeric(x[idx]).astype(int),
      "clean_path":x[clean].astype(str),"wm_path":x[wm].astype(str),
      "key_state_path":x[key].astype(str),
      "channel_copy":vals("channel_copy",1,int),"hw_copy":vals("hw_copy",8,int),
      "fpr":vals("fpr",1e-6,float),"user_number":vals("user_number",1000000,int)
    }).drop_duplicates("sample_id").sort_values("sample_id")

def load_depth(p):
    x=pd.read_csv(p); sid=col(x,"sample_id","audit_id","image_index")
    clean=col(x,"alpha0_image_path","clean_image_path","unwatermarked_image_path","clean_path")
    wm=col(x,"wm_image_path","watermarked_image_path","wm_path")
    return pd.DataFrame({"sample_id":pd.to_numeric(x[sid]).astype(int),
      "clean_path":x[clean].astype(str),"wm_path":x[wm].astype(str)
    }).drop_duplicates("sample_id").sort_values("sample_id")

def asbytes(x):
    if torch.is_tensor(x):x=x.detach().cpu().numpy()
    if isinstance(x,np.ndarray):return bytes(x.astype(np.uint8).tolist())
    if isinstance(x,bytearray):return bytes(x)
    return x

def restore(r,device):
    try:s=torch.load(r.key_state_path,map_location="cpu",weights_only=False)
    except TypeError:s=torch.load(r.key_state_path,map_location="cpu")
    if not isinstance(s,dict):raise RuntimeError(f"Bad key state: {r.key_state_path}")
    ch=int(s.get("channel_copy",s.get("ch",r.channel_copy)))
    hw=int(s.get("hw_copy",s.get("hw",r.hw_copy)))
    fp=float(s.get("fpr",r.fpr)); users=int(s.get("user_number",r.user_number))
    w=Gaussian_Shading_chacha(ch,hw,fp,users)
    key=asbytes(s.get("key",s.get("chacha_key")))
    nonce=asbytes(s.get("nonce",s.get("chacha_nonce")))
    mark=s.get("watermark",s.get("wm",s.get("watermark_tensor")))
    if key is None or nonce is None or mark is None:
        raise RuntimeError(f"Missing key/nonce/watermark in {r.key_state_path}; keys={list(s)}")
    if not torch.is_tensor(mark):mark=torch.as_tensor(mark)
    w.key=key;w.nonce=nonce;w.watermark=mark.to(device=device,dtype=torch.uint8)
    return w

def tpr1(y,s):
    fpr,tpr,thr=roc_curve(y,s); ii=np.where(fpr<=.01)[0]
    j=ii[np.argmax(tpr[ii])]
    return float(tpr[j]),float(fpr[j]),float(thr[j])

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--g0-root",default=str(BASE/"gaussian_shading_g0"))
    a.add_argument("--g0-manifest",default=None)
    a.add_argument("--lineage-root",default=str(BASE/"gaussian_shading_canonical_n300"))
    a.add_argument("--model-id",default="sd2-community/stable-diffusion-2-1-base")
    a.add_argument("--n",type=int,default=300)
    a.add_argument("--steps",type=int,default=50)
    a.add_argument("--resume",action="store_true")
    a.add_argument("--out-dir",default=None)
    z=a.parse_args()
    g0root=Path(z.g0_root); root=Path(z.lineage_root)
    out=Path(z.out_dir) if z.out_dir else root/"native_evaluation";out.mkdir(parents=True,exist_ok=True)
    perfile=out/"gaussian_shading_g0_g4_per_image.csv"
    sumfile=out/"gaussian_shading_g0_g4_summary.csv"

    base=load_g0(g0_manifest(g0root,z.g0_manifest))
    ds={1:load_depth(find_manifest(root,1))}
    ids=ds[1].sample_id.tolist()[:z.n]
    if len(ids)!=z.n:raise RuntimeError(f"G1 has {len(ids)}, requested {z.n}")
    base=base[base.sample_id.isin(ids)]
    if len(base)!=z.n:raise RuntimeError("G0/key manifest does not contain exact G1 IDs")
    ds[0]=base[["sample_id","clean_path","wm_path"]].copy()
    for d in [2,3,4]:
        ds[d]=load_depth(find_manifest(root,d))
        ds[d]=ds[d][ds[d].sample_id.isin(ids)]
        if set(ds[d].sample_id)!=set(ids):raise RuntimeError(f"G{d} ID mismatch")
    meta=base[["sample_id","image_index","key_state_path","channel_copy","hw_copy","fpr","user_number"]]
    for d in range(5):
        ds[d]=ds[d].merge(meta,on="sample_id").sort_values("sample_id")
        if len(ds[d])!=z.n:raise RuntimeError(f"G{d}: expected {z.n}, got {len(ds[d])}")
        for c in ["clean_path","wm_path","key_state_path"]:
            miss=[q for q in ds[d][c] if not Path(q).exists()]
            if miss:raise FileNotFoundError(f"G{d} missing {c}: {miss[0]}")

    print(f"Matched N={z.n} at G0-G4; IDs {min(ids)}..{max(ids)}")
    print("GPU:",torch.cuda.get_device_name(0))
    sch=DPMSolverMultistepScheduler.from_pretrained(z.model_id,subfolder="scheduler")
    pipe=InversableStableDiffusionPipeline.from_pretrained(
        z.model_id,scheduler=sch,torch_dtype=torch.float16).to("cuda")
    pipe.safety_checker=None;pipe.set_progress_bar_config(disable=True)
    te=pipe.get_text_embedding("")

    rec=[];done=set()
    if z.resume and perfile.exists():
        old=pd.read_csv(perfile);rec=old.to_dict("records")
        done=set(zip(old.depth.astype(int),old.sample_id.astype(int),old.kind.astype(str)))
        print("Resuming from",len(old),"images")
    for d in range(5):
        for _,r in tqdm(ds[d].iterrows(),total=z.n,desc=f"G{d} pairs"):
            obj=None
            for kind,label,path in [("clean",0,r.clean_path),("watermarked",1,r.wm_path)]:
                k=(d,int(r.sample_id),kind)
                if k in done:continue
                if obj is None:obj=restore(r,torch.device("cuda"))
                im=transform_img(Image.open(path).convert("RGB")).unsqueeze(0).to("cuda",dtype=te.dtype)
                with torch.inference_mode():
                    lat=pipe.get_image_latents(im,sample=False)
                    rev=pipe.forward_diffusion(latents=lat,text_embeddings=te,guidance_scale=1,
                                               num_inference_steps=z.steps)
                    score=float(obj.eval_watermark(rev))
                tau=float(obj.tau_onebit)
                rec.append({"depth":d,"sample_id":int(r.sample_id),"kind":kind,"label":label,
                    "score":score,"tau_onebit":tau,"official_detect":int(score>=tau),
                    "exact_message":int(score==1.0),"image_path":str(path),
                    "key_state_path":str(r.key_state_path)})
                done.add(k);pd.DataFrame(rec).to_csv(perfile,index=False)

    q=pd.DataFrame(rec);rows=[]
    for d in range(5):
        x=q[q.depth.astype(int)==d]; y=x.label.to_numpy(int);s=x.score.to_numpy(float)
        pos=x[x.label==1];neg=x[x.label==0];t,f,th=tpr1(y,s)
        rows.append({"depth":f"G{d}","n_clean":len(neg),"n_wm":len(pos),
          "auc":roc_auc_score(y,s),"tpr_at_le_1pct_fpr":t,"realized_fpr":f,
          "diagnostic_threshold":th,"mean_clean_score":neg.score.mean(),
          "mean_wm_score":pos.score.mean(),
          "official_wm_detection_rate":pos.official_detect.mean(),
          "official_clean_fpr":neg.official_detect.mean(),
          "exact_message_recovery":pos.exact_message.mean()})
    sm=pd.DataFrame(rows);sm.to_csv(sumfile,index=False)
    print("\n",sm.to_string(index=False));print("\nSaved:",perfile,"\n",sumfile)

if __name__=="__main__":main()
