#!/usr/bin/env python3
import argparse, os, subprocess, sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--gpus",required=True)
p.add_argument("--chains",default="order_a,order_b,order_c,order_d")
p.add_argument("--worker",default=str(Path(__file__).with_name("regenerate_alternative_lineages.py")))
p.add_argument("--data-root",default="workspace/reson_sd21_canonical_10k")
p.add_argument("--project-root",default="reson")
p.add_argument("--strength",type=float,default=.50)
a=p.parse_args()
g=[z.strip() for z in a.gpus.split(",") if z.strip()]
c=[z.strip() for z in a.chains.split(",") if z.strip()]
if len(g)!=len(c): raise SystemExit(f"Need one GPU per chain: GPUs={g} chains={c}")
root=Path(a.data_root); alt=root/"alternative_lineages"; logs=alt/"logs"; logs.mkdir(parents=True,exist_ok=True)
ps=[]
for gpu,ch in zip(g,c):
 log=logs/f"{ch}_gpu{gpu}.log"; fh=open(log,"a",buffering=1)
 cmd=[sys.executable,a.worker,"--chain",ch,"--project-root",a.project_root,
      "--data-root",a.data_root,"--strength",str(a.strength)]
 env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=gpu
 print(f"Launching {ch} on GPU {gpu}: {log}",flush=True)
 ps.append((ch,gpu,subprocess.Popen(cmd,env=env,stdout=fh,stderr=subprocess.STDOUT),fh))
bad=[]
for ch,gpu,proc,fh in ps:
 rc=proc.wait(); fh.close(); print(f"{ch} GPU {gpu}: rc={rc}",flush=True)
 if rc: bad.append((ch,gpu,rc))
if bad: raise SystemExit(f"FAILED {bad}; outputs are resume-safe.")
print("All requested chains completed.")
