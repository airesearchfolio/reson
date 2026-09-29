#!/usr/bin/env python3
"""
RESON deliberate watermark-removal attack generator.

Attacks:
  regen_vae : official WatermarkAttacker VAEWMAttacker
  regen_diff: official WatermarkAttacker DiffWMAttacker
  rinse     : repeated Regen-Diff (Rinse-NxDiff protocol)

Input is the FINAL canonical held-out G0 test set. Both clean and WM controls
are attacked so frozen-threshold FPR remains measurable.

This script does NOT train/recalibrate the RESON detector.
"""
import argparse, json, os, sys, shutil
from pathlib import Path
import pandas as pd
import torch

DEFAULT_CANON = Path("workspace/reson_sd21_canonical_10k")
DEFAULT_REPO = Path("third_party/WatermarkAttacker")
DEFAULT_OUT = Path("workspace/reson_sd21_canonical_10k/removal_attacks")

def resolve_col(df, names):
    for x in names:
        if x in df.columns: return x
    raise RuntimeError(f"Could not find any of columns {names}; got {list(df.columns)}")

def ensure_repo(repo):
    if not (repo/"wmattacker.py").exists() or not (repo/"regen_pipe.py").exists():
        raise FileNotFoundError(
            f"Official WatermarkAttacker repo not found at {repo}\n"
            "Run setup_watermarkattacker.sh first."
        )

def load_test_manifest(path):
    df=pd.read_csv(path)
    if "split" in df.columns:
        df=df[df["split"].astype(str).str.lower().str.strip()=="test"].copy()
    sid=resolve_col(df,["sample_id","id"])
    prompt=resolve_col(df,["prompt","caption"])
    clean=resolve_col(df,["clean_path","path_clean"])
    wm=resolve_col(df,["wm_path","watermarked_path","path_wm"])
    bit=resolve_col(df,["bit","payload_bit"])
    df=df.rename(columns={sid:"sample_id",prompt:"prompt",clean:"clean_path",wm:"wm_path",bit:"bit"})
    df=df.sort_values("sample_id")
    if len(df)!=1000 or df.sample_id.nunique()!=1000:
        raise RuntimeError(f"Expected exactly 1000 unique held-out test rows, got {len(df)} / {df.sample_id.nunique()}")
    return df

def make_paths(df, outdir, kind):
    d=outdir/kind; d.mkdir(parents=True,exist_ok=True)
    return [str(d/f"{int(s):05d}.png") for s in df.sample_id]

def pending(srcs, dsts):
    pairs=[(s,d) for s,d in zip(srcs,dsts) if not Path(d).exists()]
    return [p[0] for p in pairs],[p[1] for p in pairs]

def attack_vae(df, repo, outdir, quality, model_name, device):
    sys.path.insert(0,str(repo))
    from wmattacker import VAEWMAttacker
    atk=VAEWMAttacker(model_name=model_name,quality=quality,device=device)
    for kind,col in [("clean","clean_path"),("wm","wm_path")]:
        src=df[col].astype(str).tolist(); dst=make_paths(df,outdir,kind)
        s,d=pending(src,dst)
        print(f"{kind}: {len(s)} pending / {len(src)}")
        if s: atk.attack(s,d)

def load_diff_pipe(repo, model_id, device):
    sys.path.insert(0,str(repo))
    from regen_pipe import ReSDPipeline
    pipe=ReSDPipeline.from_pretrained(model_id,torch_dtype=torch.float16)
    pipe=pipe.to(device)
    try: pipe.set_progress_bar_config(disable=True)
    except Exception: pass
    return pipe

def captions_for(df, src_col):
    # Official attacker looks up caption by image basename without extension.
    return {Path(str(p)).stem:str(q) for p,q in zip(df[src_col],df.prompt)}

def attack_diff_once(df, repo, outdir, noise_step, batch_size, model_id, device, src_clean="clean_path", src_wm="wm_path"):
    sys.path.insert(0,str(repo))
    from wmattacker import DiffWMAttacker
    pipe=load_diff_pipe(repo,model_id,device)
    for kind,col in [("clean",src_clean),("wm",src_wm)]:
        src=df[col].astype(str).tolist(); dst=make_paths(df,outdir,kind)
        s,d=pending(src,dst)
        caps=captions_for(df,col)
        print(f"{kind}: {len(s)} pending / {len(src)}")
        if s:
            atk=DiffWMAttacker(pipe,batch_size=batch_size,noise_step=noise_step,captions=caps)
            atk.attack(s,d)
    del pipe
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def attack_rinse(df, repo, outdir, noise_step, rounds, batch_size, model_id, device):
    # Repeated Regen-Diff. Each round becomes the input to the next.
    current=df.copy()
    for r in range(1,rounds+1):
        rd=outdir/f"round_{r:02d}"
        print(f"\n===== RINSE ROUND {r}/{rounds} =====")
        attack_diff_once(current,repo,rd,noise_step,batch_size,model_id,device)
        current=current.copy()
        current["clean_path"]=[str(rd/"clean"/f"{int(s):05d}.png") for s in current.sample_id]
        current["wm_path"]=[str(rd/"wm"/f"{int(s):05d}.png") for s in current.sample_id]
    # Final stable aliases via manifest, no duplicate image copy.
    return current

def write_manifest(df, outdir, attack, strength):
    m=df[["sample_id","prompt","bit"]].copy()
    m["clean_path"]=[str(outdir/"clean"/f"{int(s):05d}.png") for s in df.sample_id]
    m["wm_path"]=[str(outdir/"wm"/f"{int(s):05d}.png") for s in df.sample_id]
    m["attack"]=attack; m["strength"]=strength
    m.to_csv(outdir/"attack_manifest.csv",index=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--attack",choices=["regen_vae","regen_diff","rinse"],required=True)
    ap.add_argument("--manifest",default=str(DEFAULT_CANON/"evaluation_manifest.csv"))
    ap.add_argument("--repo",default=str(DEFAULT_REPO))
    ap.add_argument("--out-root",default=str(DEFAULT_OUT))
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--limit",type=int,default=None,help="Visual/smoke gate only; e.g. 10")
    ap.add_argument("--vae-quality",type=int,default=4)
    ap.add_argument("--vae-model",default="bmshj2018-factorized")
    ap.add_argument("--noise-step",type=int,default=60)
    ap.add_argument("--rounds",type=int,default=2)
    ap.add_argument("--batch-size",type=int,default=4)
    ap.add_argument("--diff-model",default="CompVis/stable-diffusion-v1-4")
    a=ap.parse_args()

    repo=Path(a.repo); ensure_repo(repo)
    df=load_test_manifest(a.manifest)
    if a.limit: df=df.iloc[:a.limit].copy()

    if a.attack=="regen_vae":
        tag=f"regen_vae_q{a.vae_quality}"
    elif a.attack=="regen_diff":
        tag=f"regen_diff_t{a.noise_step}"
    else:
        tag=f"rinse_{a.rounds}x_t{a.noise_step}"
    out=Path(a.out_root)/tag
    out.mkdir(parents=True,exist_ok=True)

    cfg=vars(a)|{"resolved_tag":tag,"n":len(df),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES")}
    (out/"attack_config.json").write_text(json.dumps(cfg,indent=2))

    if a.attack=="regen_vae":
        attack_vae(df,repo,out,a.vae_quality,a.vae_model,a.device)
        write_manifest(df,out,a.attack,f"quality={a.vae_quality}")
    elif a.attack=="regen_diff":
        attack_diff_once(df,repo,out,a.noise_step,a.batch_size,a.diff_model,a.device)
        write_manifest(df,out,a.attack,f"noise_step={a.noise_step}")
    else:
        finaldf=attack_rinse(df,repo,out,a.noise_step,a.rounds,a.batch_size,a.diff_model,a.device)
        # point final manifest to final round
        final=out/f"round_{a.rounds:02d}"
        m=df[["sample_id","prompt","bit"]].copy()
        m["clean_path"]=[str(final/"clean"/f"{int(s):05d}.png") for s in df.sample_id]
        m["wm_path"]=[str(final/"wm"/f"{int(s):05d}.png") for s in df.sample_id]
        m["attack"]=a.attack; m["strength"]=f"{a.rounds}x noise_step={a.noise_step}"
        m.to_csv(out/"attack_manifest.csv",index=False)

    print(f"\nDONE -> {out}")
    print("Next: visually inspect clean/WM attacked images before full N=1000 evaluation.")

if __name__=="__main__":
    main()
