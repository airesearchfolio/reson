#!/usr/bin/env python3
"""
MASTER: matched carrier-mechanism ablation for RESON.

Scientific comparison (matched detector training):
  white-only       beta=0.0
  RESON blend      beta=0.4
  structured-only  beta=1.0

FAST protocol:
  3000 source pairs/condition total
    2400 train
     300 validation
     300 held-out lineage test
  alpha = 0.155 for all
  identical prompts/sample IDs/bits across all variants
  matched detector architecture/training for all 3 variants
  canonical lineage on the same 300 held-out IDs:
    SD2.1 -> FLUX -> RealVisXL -> DreamShaperXL -> FLUX

GPU allocation:
  GPU 5: beta=0.0 generation -> training -> lineage -> evaluation
  GPU 6: beta=1.0 generation -> training -> lineage -> evaluation
  GPU 7: beta=0.4 matched detector training -> lineage -> evaluation
The beta=.4 source images are reused from the canonical 10k dataset.

This intentionally retrains beta=.4 on the SAME 2400/300/300 ablation split,
rather than comparing new 2400-pair detectors against the stronger canonical
8000-pair detector.

Usage:
  python run_carrier_mechanism_ablation_3gpu.py --prepare-only
  # inspect prepared split + optional smoke images if desired
  nohup python run_carrier_mechanism_ablation_3gpu.py \
      > carrier_mechanism_master.log 2>&1 &

The script launches one independent pipeline per physical GPU concurrently.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

SRC=Path("reson")
EXP=Path("workspace/paper_experiments")
CANON=EXP/"exp_16_coco_fid"/"sd21_canonical_final_n10000"
OUT=EXP/"carrier_mechanism_matched_n3000"
HERE=Path(__file__).resolve().parent

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--canonical-root",type=Path,default=CANON)
    p.add_argument("--out-root",type=Path,default=OUT)
    p.add_argument("--project-root",type=Path,default=SRC)
    p.add_argument("--n-total",type=int,default=3000)
    p.add_argument("--n-train",type=int,default=2400)
    p.add_argument("--n-val",type=int,default=300)
    p.add_argument("--n-test",type=int,default=300)
    p.add_argument("--alpha",type=float,default=.155)
    p.add_argument("--carrier-seed",type=int,default=20260903+55555)
    p.add_argument("--generation-seed",type=int,default=20260903)
    p.add_argument("--gpus",default="5,6,7")
    p.add_argument("--prepare-only",action="store_true")
    return p.parse_args()

def choose_manifest(root):
    for n in ["manifest_final_10000.csv","manifest.csv"]:
        p=root/n
        if p.exists(): return p
    raise FileNotFoundError(f"No canonical manifest under {root}")

def choose_col(df,names):
    for n in names:
        if n in df: return n
    raise RuntimeError(f"Missing {names}; columns={list(df.columns)}")

def normalize(df):
    cc=choose_col(df,["clean_path","clean_image_path"])
    wc=choose_col(df,["wm_path","wm_image_path","reson_path"])
    pc=choose_col(df,["prompt","caption"])
    bc=choose_col(df,["bit","bit0"])
    z=pd.DataFrame({
      "sample_id":df.sample_id.astype(int),
      "prompt":df[pc].astype(str),
      "bit":df[bc].astype(int),
      "clean_path":df[cc].astype(str),
      "wm_path":df[wc].astype(str),
    })
    return z.sort_values("sample_id").drop_duplicates("sample_id")

def prep(a):
    a.out_root.mkdir(parents=True,exist_ok=True)
    c=normalize(pd.read_csv(choose_manifest(a.canonical_root)))
    if len(c)<a.n_total: raise RuntimeError(f"Canonical has {len(c)} < {a.n_total}")
    # Same fixed IDs for all three conditions.
    c=c.head(a.n_total).copy()
    # Existing generator encodes bit = sample_id % 2. Verify canonical agreement.
    bad=c[c.bit.astype(int)!=(c.sample_id.astype(int)%2)]
    if len(bad):
        raise RuntimeError(
          f"Canonical bit assignment differs from generator for {len(bad)} rows. "
          "Do not proceed: matched payloads would not be guaranteed."
        )
    if a.n_train+a.n_val+a.n_test!=a.n_total:
        raise RuntimeError("n_train+n_val+n_test must equal n_total")
    split=["train"]*a.n_train+["val"]*a.n_val+["test"]*a.n_test
    c["split"]=split
    c.to_csv(a.out_root/"canonical_selected_3000.csv",index=False)
    (a.out_root/"prompts_3000.txt").write_text("\n".join(c.prompt.tolist())+"\n")

    # beta=.4 condition reuses canonical G0.
    b=a.out_root/"beta040_blend"; b.mkdir(exist_ok=True)
    c.to_csv(b/"manifest.csv",index=False)

    for name in ["beta000_white","beta100_structured"]:
        (a.out_root/name).mkdir(exist_ok=True)

    (a.out_root/"experiment_config.json").write_text(json.dumps({
      "n_total":a.n_total,"train":a.n_train,"val":a.n_val,"test":a.n_test,
      "alpha":a.alpha,"betas":{"white":0.0,"blend":0.4,"structured":1.0},
      "carrier_seed":a.carrier_seed,"generation_seed":a.generation_seed,
      "lineage":"SD2.1 -> FLUX -> RealVisXL -> DreamShaperXL -> FLUX",
      "principle":"matched source IDs, prompts, payloads, alpha, detector protocol and lineage"
    },indent=2))
    return c

def make_splits(root, canonical_split):
    m=pd.read_csv(root/"manifest.csv")
    cc=choose_col(m,["clean_path","clean_image_path"])
    wc=choose_col(m,["wm_path","wm_image_path"])
    pc=choose_col(m,["prompt","caption"])
    z=pd.DataFrame({
      "sample_id":m.sample_id.astype(int),"prompt":m[pc].astype(str),
      "bit":m["bit"].astype(int),
      "clean_path":m[cc].astype(str),"wm_path":m[wc].astype(str)
    }).sort_values("sample_id")
    z=z.merge(canonical_split[["sample_id","split"]],on="sample_id",how="inner",validate="one_to_one")
    if len(z)!=len(canonical_split): raise RuntimeError(f"{root}: incomplete source set")
    z.to_csv(root/"manifest.csv",index=False)
    sd=root/"matched_splits"; sd.mkdir(exist_ok=True)
    for s in ["train","val","test"]:
        q=z[z.split==s].copy()
        q["clean_image_path"]=q.clean_path
        q["wm_image_path"]=q.wm_path
        q.to_csv(sd/f"{s}_pairs.csv",index=False)
    # exact 300-row G0 evaluation manifest
    td=root/"lineage_input"; td.mkdir(exist_ok=True)
    z[z.split=="test"].to_csv(td/"manifest.csv",index=False)

def sh(cmd,log,env=None):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("a") as f:
        f.write("\n+ "+" ".join(map(str,cmd))+"\n"); f.flush()
        r=subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,env=env)
    if r.returncode: raise RuntimeError(f"Command failed rc={r.returncode}; log={log}")

def pipeline(label,beta,gpu,a,canonical_split,reuse_source=False):
    root=a.out_root/label; logs=root/"logs"; logs.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu)

    if not reuse_source:
        # Existing exact RESON SD2.1 source generator. One beta condition per GPU.
        sh([sys.executable,a.project_root/"run_reson_sd21_multigpu.py",
            "--prompts",a.out_root/"prompts_3000.txt",
            "--out-root",root,"--limit",a.n_total,"--gpus",gpu,
            "--alpha",a.alpha,"--beta",beta,"--steps",50,"--guidance",7.5,
            "--batch-size",8,"--seed",a.generation_seed,
            "--carrier-seed",a.carrier_seed],
           logs/"01_source_generation.log",os.environ.copy())

    make_splits(root,canonical_split)

    # REQUIRED visual gate: create marker and stop unless user has approved it.
    gate=root/"VISUAL_CHECK_APPROVED"
    if not gate.exists():
        print(f"[{label}] SOURCE READY. Inspect clean vs WM images in {root}.",flush=True)
        print(f"[{label}] Then run: touch {gate}",flush=True)
        return "WAITING_VISUAL_GATE"

    # Matched detector training.
    det=root/"matched_detector"
    sh([sys.executable,HERE/"train_matched_carrier_detector.py",
        "--manifest",root/"manifest.csv","--split-dir",root/"matched_splits",
        "--out-dir",det,"--device","cuda","--batch-size",32,
        "--epochs",50,"--patience",8,"--seed",20260919],
       logs/"02_detector_training.log",env)

    # Read each variant's OWN validation-selected threshold.
    rr=json.loads((det/"results.json").read_text())
    thr=float(rr["validation"]["threshold"])

    # Full canonical lineage, only N=300 held-out pairs.
    lin=root/"lineage_n300"
    sh([sys.executable,HERE/"regenerate_carrier_ablation_g0_g4.py",
        "--project-root",a.project_root,
        "--manifest",root/"lineage_input"/"manifest.csv",
        "--out",lin,"--expected-n",a.n_test,"--strength",.50],
       logs/"03_lineage.log",env)

    # Evaluate G0-G4 with matched checkpoint + its frozen validation threshold.
    ev=root/"evaluation_n300"
    sh([sys.executable,HERE/"evaluate_carrier_ablation_g0_g4.py",
        "--data-root",root/"lineage_input","--lineage-root",lin,
        "--checkpoint",det/"detector_best.pt","--out-dir",ev,
        "--device","cuda","--fid-device","cuda",
        "--threshold",thr,"--expected-n",a.n_test,
        "--skip-clip","--skip-fid","--depths","g0,g1,g2,g3,g4"],
       logs/"04_evaluation.log",env)
    return "DONE"

def main():
    a=args()
    g=[x.strip() for x in a.gpus.split(",")]
    if len(g)!=3: raise RuntimeError("--gpus must contain exactly 3 physical GPU IDs")
    c=prep(a)
    if a.prepare_only:
        print("Prepared:",a.out_root); return

    # Run all 3 matched conditions concurrently.
    import multiprocessing as mp
    jobs=[
      ("beta000_white",0.0,g[0],False),
      ("beta100_structured",1.0,g[1],False),
      ("beta040_blend",0.4,g[2],True),
    ]
    q=mp.Queue()
    def go(label,beta,gpu,reuse):
        try: q.put((label,pipeline(label,beta,gpu,a,c,reuse),None))
        except Exception as e: q.put((label,"FAILED",repr(e)))
    ps=[]
    for z in jobs:
        p=mp.Process(target=go,args=z); p.start(); ps.append(p)
    results=[q.get() for _ in ps]
    for p in ps: p.join()
    print("\nRESULTS")
    for r in results: print(r)
    print("\nIf any condition says WAITING_VISUAL_GATE, inspect its G0 images, then:")
    print("  touch <condition>/VISUAL_CHECK_APPROVED")
    print("and rerun the same master command. Generation is reused; training/lineage continues.")

if __name__=="__main__": main()
