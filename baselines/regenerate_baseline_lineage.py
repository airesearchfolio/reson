#!/usr/bin/env python3
"""
Fair matched G0->G4 heterogeneous lineage generator for baseline reruns.

MAIN-TABLE FAIRNESS CONTRACT:
- Baseline G0 manifest MUST correspond exactly to the canonical RESON held-out test split.
- The script verifies prompt/sample alignment against RESON test_pairs.csv BEFORE regeneration.
- It aborts on mismatch instead of silently running an incomparable experiment.

Canonical chain:
  G0: supplied baseline source images (SD2.1-based experiment)
  G1: FLUX.1-dev @ 0.50
  G2: RealVisXL V5.0 @ 0.50
  G3: DreamShaper XL @ 0.50
  G4: FLUX.1-dev @ 0.50

This script ONLY regenerates images/manifests. It intentionally does NOT use the
old Stage-7E-v2 RESON/SDXL detector block from the supplied Stage-14 script.
Evaluate each baseline with its own official/native detector afterward.

Required source manifest:
  prompt
  plus one watermarked-path column and one clean/control-path column.

Accepted path aliases:
  WM:    wm_path, wm_image_path, watermarked_path, image_wm_path
  CLEAN: clean_path, alpha0_image_path, null_path, unwatermarked_path, image_clean_path

Optional:
  sample_id or audit_id (otherwise row index)
  bit (preserved if present)
"""
from __future__ import annotations
import argparse, gc, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm

FLUX_MODEL="black-forest-labs/FLUX.1-dev"
REALVIS_MODEL="SG161222/RealVisXL_V5.0"
DREAMSHAPER_MODEL="Lykon/dreamshaper-xl-1-0"

CHAIN=[
    (1,"flux","flux",FLUX_MODEL),
    (2,"realvis","sdxl",REALVIS_MODEL),
    (3,"dreamshaper","sdxl",DREAMSHAPER_MODEL),
    (4,"flux","flux",FLUX_MODEL),
]
WM_ALIASES=["wm_path","wm_image_path","watermarked_path","image_wm_path"]
CLEAN_ALIASES=["clean_path","alpha0_image_path","null_path","unwatermarked_path","image_clean_path"]

def choose_col(df,names,explicit=None):
    if explicit:
        if explicit not in df.columns: raise ValueError(f"Column {explicit!r} not found. Columns={list(df.columns)}")
        return explicit
    for n in names:
        if n in df.columns: return n
    raise ValueError(f"None of {names} found. Columns={list(df.columns)}")

def seed_for(sample_id,depth,strength,seed_base):
    s=f"{seed_base}|baseline-lineage|id={sample_id}|depth={depth}|strength={strength:.2f}"
    return int(hashlib.sha256(s.encode()).hexdigest()[:8],16)&0x7fffffff

def cleanup(pipe):
    del pipe; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def load_flux():
    from diffusers import FluxImg2ImgPipeline
    print("Loading:",FLUX_MODEL)
    p=FluxImg2ImgPipeline.from_pretrained(FLUX_MODEL,torch_dtype=torch.bfloat16)
    p.to("cuda")
    if hasattr(p,"enable_vae_slicing"): p.enable_vae_slicing()
    p.set_progress_bar_config(disable=True)
    return p

def load_sdxl(model_id):
    from diffusers import AutoPipelineForImage2Image
    print("Loading:",model_id)
    kw=dict(torch_dtype=torch.float16,use_safetensors=True,variant="fp16")
    try: p=AutoPipelineForImage2Image.from_pretrained(model_id,**kw)
    except Exception:
        kw.pop("variant",None); p=AutoPipelineForImage2Image.from_pretrained(model_id,**kw)
    p.to("cuda")
    if hasattr(p,"enable_vae_slicing"): p.enable_vae_slicing()
    p.set_progress_bar_config(disable=True)
    return p

@torch.inference_mode()
def run_flux(pipe,img,prompt,seed,strength,steps):
    g=torch.Generator(device="cpu").manual_seed(int(seed))
    return pipe(prompt=str(prompt),image=img,strength=strength,width=512,height=512,
                num_inference_steps=steps,guidance_scale=3.5,max_sequence_length=512,
                generator=g).images[0].convert("RGB")

@torch.inference_mode()
def run_sdxl(pipe,img,prompt,seed,strength,steps):
    g=torch.Generator(device="cuda").manual_seed(int(seed))
    return pipe(prompt=str(prompt),image=img,strength=strength,width=512,height=512,
                num_inference_steps=steps,guidance_scale=7.5,generator=g).images[0].convert("RGB")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--method",required=True,help="e.g. tree_ring, gaussian_shading, wind, serum")
    ap.add_argument("--source-manifest",required=True,
                    help="Baseline G0 manifest generated using the exact RESON held-out prompts")
    ap.add_argument("--reference-manifest",
                    default="workspace/reson_sd21_canonical_10k/final_detector/test_pairs.csv",
                    help="Canonical RESON held-out split used only to verify matched prompts/IDs")
    ap.add_argument("--out-root",required=True)
    ap.add_argument("--wm-col",default=None)
    ap.add_argument("--clean-col",default=None)
    ap.add_argument("--limit",type=int,default=1000)
    ap.add_argument("--strength",type=float,default=.50)
    ap.add_argument("--steps",type=int,default=30)
    ap.add_argument("--seed-base",type=int,default=20260820)
    args=ap.parse_args()

    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    src=pd.read_csv(args.source_manifest)
    ref=pd.read_csv(args.reference_manifest)
    if "prompt" not in src.columns: raise ValueError("source manifest must contain 'prompt'")
    if "prompt" not in ref.columns: raise ValueError("reference manifest must contain 'prompt'")

    # Fairness gate: compare the exact first N prompts in canonical order.
    n = args.limit if args.limit is not None else len(ref)
    ref = ref.head(n).copy().reset_index(drop=True)
    src = src.head(n).copy().reset_index(drop=True)
    if len(src) != len(ref):
        raise RuntimeError(f"FAIRNESS CHECK FAILED: baseline rows={len(src)}, reference rows={len(ref)}")
    norm=lambda x: " ".join(str(x).split())
    sp=src["prompt"].map(norm)
    rp=ref["prompt"].map(norm)
    bad=np.where(sp.to_numpy()!=rp.to_numpy())[0]
    if len(bad):
        i=int(bad[0])
        raise RuntimeError(
            f"FAIRNESS CHECK FAILED: prompt mismatch at row {i}.\n"
            f"baseline={src.loc[i,'prompt']!r}\nreference={ref.loc[i,'prompt']!r}"
        )
    # If both sides expose an ID, verify it too.
    src_id="sample_id" if "sample_id" in src.columns else ("audit_id" if "audit_id" in src.columns else None)
    ref_id="sample_id" if "sample_id" in ref.columns else ("audit_id" if "audit_id" in ref.columns else None)
    if src_id and ref_id:
        a=pd.to_numeric(src[src_id],errors="coerce").to_numpy()
        b=pd.to_numeric(ref[ref_id],errors="coerce").to_numpy()
        if not np.array_equal(a,b,equal_nan=True):
            raise RuntimeError(f"FAIRNESS CHECK FAILED: {src_id} does not match reference {ref_id}.")
    print(f"FAIRNESS CHECK PASSED: exact matched RESON prompts, N={len(ref)}")
    wmcol=choose_col(src,WM_ALIASES,args.wm_col)
    clcol=choose_col(src,CLEAN_ALIASES,args.clean_col)
    idcol="sample_id" if "sample_id" in src.columns else ("audit_id" if "audit_id" in src.columns else None)
    # already truncated and verified against canonical reference above

    out=Path(args.out_root)/args.method
    (out/"manifests").mkdir(parents=True,exist_ok=True)
    print("GPU:",torch.cuda.get_device_name(0))
    print("Method:",args.method)
    print("G0 source: SD2.1 baseline images supplied by manifest")
    print("Rows:",len(src),"WM column:",wmcol,"clean column:",clcol)
    print("Output:",out)

    rows=[]
    for j,(_,r) in enumerate(src.iterrows()):
        sid=int(r[idcol]) if idcol else j
        row={"sample_id":sid,"prompt":str(r.prompt),"depth":0,"generator":"source_sd21",
             "model_id":"source_sd21","strength":0.0,"attack_seed":np.nan,
             "wm_image_path":str(Path(str(r[wmcol])).resolve()),
             "clean_image_path":str(Path(str(r[clcol])).resolve())}
        if "bit" in src.columns: row["bit"]=int(r.bit)
        rows.append(row)
    prev=pd.DataFrame(rows)
    prev.to_csv(out/"manifests/generation_0.csv",index=False)

    for depth,name,family,model_id in CHAIN:
        od=out/f"generation_{depth}_{name}"
        wd=od/"watermarked"; cd=od/"clean"; wd.mkdir(parents=True,exist_ok=True); cd.mkdir(parents=True,exist_ok=True)
        pipe=load_flux() if family=="flux" else load_sdxl(model_id)
        rows=[]
        for _,r in tqdm(prev.iterrows(),total=len(prev),desc=f"{args.method} G{depth} {name}"):
            sid=int(r.sample_id); seed=seed_for(sid,depth,args.strength,args.seed_base)
            wp=wd/f"{sid:05d}.png"; cp=cd/f"{sid:05d}.png"
            for inp,op in [(r.wm_image_path,wp),(r.clean_image_path,cp)]:
                if not op.exists():
                    with Image.open(inp) as im:
                        init=im.convert("RGB").resize((512,512),Image.Resampling.LANCZOS)
                    img=run_flux(pipe,init,r.prompt,seed,args.strength,args.steps) if family=="flux" else run_sdxl(pipe,init,r.prompt,seed,args.strength,args.steps)
                    img.save(op)
            row={"sample_id":sid,"prompt":str(r.prompt),"depth":depth,"generator":name,"model_id":model_id,
                 "strength":args.strength,"attack_seed":seed,
                 "wm_image_path":str(wp.resolve()),"clean_image_path":str(cp.resolve())}
            if "bit" in prev.columns: row["bit"]=int(r.bit)
            rows.append(row)
        cleanup(pipe)
        prev=pd.DataFrame(rows)
        mp=out/"manifests"/f"generation_{depth}_{name}.csv"; prev.to_csv(mp,index=False)
        print("Saved",mp,"rows",len(prev))

    cfg={"method":args.method, "source_manifest":args.source_manifest,"reference_manifest":args.reference_manifest,
         "fairness_check":"exact prompt-order match passed","source_generator":"SD2.1",
         "chain":["SD2.1 source","FLUX.1-dev","RealVisXL_V5.0","DreamShaper XL","FLUX.1-dev"],
         "strength":args.strength,"steps":args.steps,"seed_base":args.seed_base,"n":len(src),
         "same_transition_seed_for_clean_and_wm":True}
    (out/"run_config.json").write_text(json.dumps(cfg,indent=2))
    print("\nDONE:",out)
    print("G0-G4 manifests:",out/"manifests")

if __name__=="__main__":
    main()
