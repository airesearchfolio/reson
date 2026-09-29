#!/usr/bin/env python3
"""
Evaluate already-generated SERUM canonical N=300 at G0-G4 with SERUM's
native frozen detector. No generation is performed.

Run from the SERUM repository root / in serum_official environment.
"""
from pathlib import Path
import argparse, re, subprocess, sys
import pandas as pd

SERUM_ROOT=Path("third_party/SERUM")
ROOT=Path("workspace/fair_sd21_main_table/serum_canonical_n300")
CONFIG=SERUM_ROOT/"configs/config_stage15_full.yaml"
CKPT=SERUM_ROOT/"results/SERUM_stage15_full/checkpoints/checkpoint_epoch_50.pt"

SUMMARY_RE=re.compile(
    r"([0-9.]+)% TPR @ ([0-9.]+)% FPR "
    r"\(ROC AUC: ([0-9.]+)\) Thr: ([0-9.eE+-]+) BalAcc: ([0-9.]+)"
)

def cols(df):
    wc=next((c for c in ["wm_image_path","watermarked_image_path","wm_path"] if c in df),None)
    cc=next((c for c in ["alpha0_image_path","unwatermarked_image_path","clean_image_path","clean_path"] if c in df),None)
    if not wc or not cc: raise RuntimeError(f"Cannot identify paths: {list(df.columns)}")
    return cc,wc

def manifest(depth):
    if depth==0: return ROOT/"generation_0.csv"
    for p in [ROOT/f"generation_{depth}.csv", *sorted((ROOT/"manifests").glob(f"generation_{depth}_*.csv"))]:
        if p.exists(): return p
    raise FileNotFoundError(f"No G{depth} manifest")

def stage_dirs(depth):
    if depth==0:
        # Native SERUM evaluator wants folders, so make a symlink-only staging set.
        m=manifest(0); df=pd.read_csv(m).sort_values("sample_id").head(300)
        cc,wc=cols(df)
        base=ROOT/"native_eval_g0_staging"; clean=base/"clean"; wm=base/"watermarked"
        clean.mkdir(parents=True,exist_ok=True); wm.mkdir(parents=True,exist_ok=True)
        for i,r in df.reset_index(drop=True).iterrows():
            for src,dst in [(Path(str(r[cc])),clean/f"{i:05d}.png"),
                            (Path(str(r[wc])),wm/f"{i:05d}.png")]:
                if not src.exists(): raise FileNotFoundError(src)
                if not dst.exists(): dst.symlink_to(src.resolve())
        return clean,wm
    names={1:"flux",2:"realvis",3:"dreamshaper",4:"flux"}
    base=ROOT/f"generation_{depth}_{names[depth]}"
    return base/"alpha0",base/"watermarked"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=300)
    ap.add_argument("--config",type=Path,default=CONFIG)
    ap.add_argument("--checkpoint",type=Path,default=CKPT)
    ap.add_argument("--out-dir",type=Path,default=ROOT/"native_evaluation")
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for d in range(5):
        clean,wm=stage_dirs(d)
        nc=len(list(clean.glob("*"))); nw=len(list(wm.glob("*")))
        if nc<a.n or nw<a.n: raise RuntimeError(f"G{d}: clean={nc}, wm={nw}, need {a.n}")
        cmd=[sys.executable,"-m","src.evaluation.eval",
             "--config",str(a.config),"--load-from-checkpoint",
             "--checkpoint-path",str(a.checkpoint),"--eval-clean",
             "--num-samples",str(a.n),"--clean-path",str(clean),
             "--watermarked-path",str(wm)]
        print("\n"+"="*72+f"\nSERUM G{d} N={a.n}\n"+"="*72,flush=True)
        q=subprocess.run(cmd,cwd=str(SERUM_ROOT),text=True,stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
        print(q.stdout,flush=True)
        (a.out_dir/f"G{d}.log").write_text(q.stdout)
        if q.returncode: raise RuntimeError(f"SERUM evaluator failed at G{d}")
        m=SUMMARY_RE.search(q.stdout)
        if m:
            tpr,fpr,auc,thr,bal=m.groups()
            rows.append(dict(depth=f"G{d}",n=a.n,tpr_percent=float(tpr),
                             realized_fpr_percent=float(fpr),roc_auc=float(auc),
                             threshold=float(thr),balanced_accuracy=float(bal)))
        else:
            rows.append(dict(depth=f"G{d}",n=a.n,tpr_percent=None,
                             realized_fpr_percent=None,roc_auc=None,
                             threshold=None,balanced_accuracy=None))
    out=pd.DataFrame(rows)
    out.to_csv(a.out_dir/"serum_canonical300_g0_g4_summary.csv",index=False)
    print("\n",out.to_string(index=False))
    print("\nSaved:",a.out_dir/"serum_canonical300_g0_g4_summary.csv")

if __name__=="__main__": main()
