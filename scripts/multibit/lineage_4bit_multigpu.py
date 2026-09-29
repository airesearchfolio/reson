#!/usr/bin/env python3
"""Multi-GPU launcher for RESON final 4-bit G1->G4 lineage."""
import argparse, os, subprocess, sys, json
from pathlib import Path
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument("--gpus",required=True,help="Physical GPU IDs, e.g. 4,6. Check nvidia-smi first.")
ap.add_argument("--data-root",default="workspace/paper_experiments/exp_16_coco_fid/sd21_4bit_alpha0155_beta040_n10000")
ap.add_argument("--project-root",default="reson")
ap.add_argument("--worker",default=str(Path(__file__).with_name("lineage_4bit_worker.py")))
ap.add_argument("--strength",type=float,default=.50)
a=ap.parse_args()

root=Path(a.data_root)
mf=root/"manifest_final_10000.csv"
df=pd.read_csv(mf)
test=df[df["split"].astype(str).str.lower().str.strip()=="test"].copy().sort_values("sample_id")
assert len(test)==1000 and test.sample_id.nunique()==1000
gpus=[x.strip() for x in a.gpus.split(",") if x.strip()]
if not gpus: raise SystemExit("No GPUs supplied.")

base=root/"final_lineage_4bit_multigpu"
shards=base/"shards"; logs=base/"logs"
shards.mkdir(parents=True,exist_ok=True); logs.mkdir(parents=True,exist_ok=True)
parts=[test.iloc[i::len(gpus)].copy() for i in range(len(gpus))]
procs=[]

for i,(gpu,part) in enumerate(zip(gpus,parts)):
    sm=shards/f"test_shard_{i:02d}_of_{len(gpus):02d}.csv"
    part.to_csv(sm,index=False)
    out=shards/f"shard_{i:02d}"
    log=logs/f"gpu_{gpu}_shard_{i:02d}.log"
    cmd=[sys.executable,a.worker,"--project-root",a.project_root,"--data-root",a.data_root,
         "--manifest",str(sm),"--out",str(out),"--expected-n",str(len(part)),
         "--strength",str(a.strength)]
    env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=gpu
    fh=open(log,"a",buffering=1)
    print(f"Launching shard {i}: physical GPU {gpu}, n={len(part)}, log={log}",flush=True)
    procs.append((i,gpu,subprocess.Popen(cmd,env=env,stdout=fh,stderr=subprocess.STDOUT),fh))

failed=[]
for i,gpu,p,fh in procs:
    rc=p.wait(); fh.close()
    print(f"Shard {i} GPU {gpu} finished rc={rc}",flush=True)
    if rc: failed.append((i,gpu,rc))
if failed: raise SystemExit(f"FAILED: {failed}. Inspect logs; generation is resumable.")

for depth in ["g1","g2","g3","g4"]:
    fs=[pd.read_csv(shards/f"shard_{i:02d}"/f"manifest_{depth}.csv") for i in range(len(gpus))]
    m=pd.concat(fs,ignore_index=True).sort_values("sample_id")
    assert len(m)==1000 and m.sample_id.nunique()==1000
    m.to_csv(base/f"manifest_{depth}.csv",index=False)

allm=pd.concat([pd.read_csv(base/f"manifest_{d}.csv") for d in ["g1","g2","g3","g4"]],ignore_index=True)
assert len(allm)==4000
allm.to_csv(base/"manifest_g1_g4.csv",index=False)
(base/"multigpu_run.json").write_text(json.dumps({"gpus":gpus,"n_test":1000,"per_shard":[len(x) for x in parts],"strength":a.strength},indent=2))
print(f"DONE -> {base}")
