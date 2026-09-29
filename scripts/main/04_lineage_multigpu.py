#!/usr/bin/env python3
"""Multi-GPU launcher for FIXED RESON SD2.1 test lineage worker."""
import argparse, os, subprocess, sys, json
from pathlib import Path
import pandas as pd

a=argparse.ArgumentParser()
a.add_argument("--gpus",default="5,7")
a.add_argument("--data-root",default="workspace/reson_sd21_canonical_10k")
a.add_argument("--project-root",default="reson")
a.add_argument("--worker",default=str(Path(__file__).with_name("04_lineage_worker.py")))
a.add_argument("--strength",type=float,default=.50)
args=a.parse_args()

root=Path(args.data_root)
df=pd.read_csv(root/"manifest.csv")
test=df[df["split"].astype(str).str.lower()=="test"].copy().sort_values("sample_id")
assert len(test)==1000, f"Expected 1000 test rows, got {len(test)}"
gpus=[x.strip() for x in args.gpus.split(",") if x.strip()]

base=root/"final_lineage_test_multigpu"
shards=base/"shards"; logs=base/"logs"
shards.mkdir(parents=True,exist_ok=True); logs.mkdir(parents=True,exist_ok=True)
parts=[test.iloc[i::len(gpus)].copy() for i in range(len(gpus))]
procs=[]

for i,(gpu,part) in enumerate(zip(gpus,parts)):
    mf=shards/f"test_shard_{i:02d}_of_{len(gpus):02d}.csv"
    part.to_csv(mf,index=False)
    out=shards/f"shard_{i:02d}"
    log=logs/f"gpu_{gpu}_shard_{i:02d}.log"
    cmd=[sys.executable,args.worker,
         "--project-root",args.project_root,"--data-root",args.data_root,
         "--manifest",str(mf),"--out",str(out),
         "--expected-n",str(len(part)),"--strength",str(args.strength)]
    env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=gpu
    fh=open(log,"a",buffering=1)
    print(f"Launching shard {i}: physical GPU {gpu}, n={len(part)}, log={log}",flush=True)
    procs.append((i,gpu,subprocess.Popen(cmd,env=env,stdout=fh,stderr=subprocess.STDOUT),fh))

failed=[]
for i,gpu,p,fh in procs:
    rc=p.wait(); fh.close()
    print(f"Shard {i} GPU {gpu} finished rc={rc}",flush=True)
    if rc: failed.append((i,gpu,rc))
if failed: raise SystemExit(f"FAILED shards: {failed}. Inspect logs; completed images are resumable.")

for depth in ["g1","g2","g3","g4"]:
    fs=[pd.read_csv(shards/f"shard_{i:02d}"/f"manifest_{depth}.csv") for i in range(len(gpus))]
    m=pd.concat(fs,ignore_index=True).sort_values("sample_id")
    assert len(m)==1000 and m.sample_id.nunique()==1000
    m.to_csv(base/f"manifest_{depth}.csv",index=False)

allm=pd.concat([pd.read_csv(base/f"manifest_{d}.csv") for d in ["g1","g2","g3","g4"]],ignore_index=True)
assert len(allm)==4000
allm.to_csv(base/"manifest_g1_g4.csv",index=False)
(base/"multigpu_run.json").write_text(json.dumps({"gpus":gpus,"n_test":1000,"per_shard":[len(x) for x in parts]},indent=2))
print(f"DONE: 1000 test pairs x G1-G4 -> {base}")
