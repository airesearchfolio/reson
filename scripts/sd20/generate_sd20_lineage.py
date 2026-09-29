#!/usr/bin/env python3
from __future__ import annotations
import argparse,gc,hashlib,json,os,subprocess,sys
from pathlib import Path
import pandas as pd

CANON_ROOT=Path('workspace/reson_sd21_canonical_10k')
CANON_MANIFEST=CANON_ROOT/'manifest.csv'
PROJECT_ROOT=Path('reson')
OUT_DEFAULT=Path('workspace/paper_experiments/reson_sd20_g0_canonical300_transfer')
SD20_MODEL='./models/sd2-base'
FLUX_MODEL='black-forest-labs/FLUX.1-dev'
REALVIS_MODEL='SG161222/RealVisXL_V5.0'
DREAM_MODEL='Lykon/dreamshaper-xl-1-0'
CHAIN={1:('flux','flux',FLUX_MODEL),2:('realvis_xl','sdxl',REALVIS_MODEL),3:('dreamshaper_xl','sdxl',DREAM_MODEL),4:('flux','flux',FLUX_MODEL)}
ALPHA=.155; BETA=.40; G0_STEPS=50; G0_GUIDANCE=7.5
REGEN_STRENGTH=.50; REGEN_STEPS=30; SDXL_GUIDANCE=7.5; FLUX_GUIDANCE=3.5
WIDTH=HEIGHT=512; TRANSITION_SEED_BASE=20260820

def cli():
 p=argparse.ArgumentParser(description='RESON SD2.0 canonical300 G0-G4 two-GPU generator')
 p.add_argument('--stage',choices=['g0','lineage','all','status','merge'],default='all')
 p.add_argument('--gpus',default='3,5',help='Exactly two physical GPU IDs')
 p.add_argument('--n',type=int,default=300); p.add_argument('--out-root',type=Path,default=OUT_DEFAULT)
 p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS); p.add_argument('--worker-stage',choices=['g0','g1','g2','g3','g4'],help=argparse.SUPPRESS)
 p.add_argument('--shard-id',type=int,choices=[0,1],help=argparse.SUPPRESS); p.add_argument('--overwrite',action='store_true')
 return p.parse_args()

def valid_png(p):
 if not p.exists(): return False
 try:
  from PIL import Image
  with Image.open(p) as im: im.verify()
  return True
 except: return False

def canonical_subset(n):
 d=pd.read_csv(CANON_MANIFEST); need={'split','sample_id','prompt'}; miss=need-set(d.columns)
 if miss: raise RuntimeError(f'Canonical manifest missing {sorted(miss)}')
 d=d[d.split.astype(str).str.lower().str.strip().eq('test')].sort_values('sample_id').head(n).reset_index(drop=True)
 if len(d)!=n or d.sample_id.duplicated().any(): raise RuntimeError(f'Bad canonical subset: {len(d)} rows')
 return d

def seed_bit(r):
 seed=None
 for c in ('generation_seed','seed'):
  if c in r.index and pd.notna(r[c]): seed=int(r[c]); break
 if seed is None: raise RuntimeError('Canonical manifest has no generation_seed/seed; refusing to invent seeds.')
 if 'bit' not in r.index or pd.isna(r['bit']): raise RuntimeError('Canonical manifest has no bit column.')
 return seed,int(r['bit'])

def transition_seed(sid,depth):
 s=f'{TRANSITION_SEED_BASE}|fair_main_table|audit={sid}|depth={depth}|strength={REGEN_STRENGTH:.2f}'
 return int(hashlib.sha256(s.encode()).hexdigest()[:8],16)&0x7fffffff

def shard_manifest(out,stage,shard): return out/'manifests'/'shards'/f'{stage}_shard{shard}.csv'
def merged_manifest(out,d): return out/'manifests'/('generation_0.csv' if d==0 else f'generation_{d}_{CHAIN[d][0]}.csv')
def stage_dirs(out,d):
 b=out/('generation_0_sd20' if d==0 else f'generation_{d}_{CHAIN[d][0]}'); c=b/'alpha0'; w=b/'watermarked'; c.mkdir(parents=True,exist_ok=True); w.mkdir(parents=True,exist_ok=True); return c,w

def merge_stage(out,stage,n):
 d=0 if stage=='g0' else int(stage[1:]); parts=[]
 for s in (0,1):
  p=shard_manifest(out,stage,s)
  if not p.exists(): raise FileNotFoundError(p)
  parts.append(pd.read_csv(p))
 x=pd.concat(parts,ignore_index=True).drop_duplicates('sample_id',keep='last').sort_values('sample_id')
 exp=set(canonical_subset(n).sample_id.astype(int)); got=set(x.sample_id.astype(int))
 if len(x)!=n or got!=exp: raise RuntimeError(f'{stage} merge mismatch rows={len(x)}')
 p=merged_manifest(out,d); p.parent.mkdir(parents=True,exist_ok=True); x.to_csv(p,index=False); print(f'[MERGE] {stage}: {len(x)} -> {p}',flush=True)

def run_g0(a):
 import torch
 from diffusers import StableDiffusionPipeline
 sys.path.insert(0,str(PROJECT_ROOT)); from config import Config; from watermark import Watermark
 cdir,wdir=stage_dirs(a.out_root,0); sm=shard_manifest(a.out_root,'g0',a.shard_id); sm.parent.mkdir(parents=True,exist_ok=True)
 d=canonical_subset(a.n).iloc[a.shard_id::2].copy(); cfg=Config(); cfg.watermark.noise_mix_alpha=ALPHA; cfg.watermark.carrier_blend=BETA; wm=Watermark(cfg).build()
 pipe=StableDiffusionPipeline.from_pretrained(SD20_MODEL,torch_dtype=torch.float32,safety_checker=None,requires_safety_checker=False).to('cuda'); pipe.set_progress_bar_config(disable=True)
 device=pipe._execution_device; dtype=next(pipe.unet.parameters()).dtype; rows=[]
 for k,(_,r) in enumerate(d.iterrows(),1):
  sid=int(r.sample_id); seed,bit=seed_bit(r); cp=cdir/f'{sid:05d}.png'; wp=wdir/f'{sid:05d}_wm.png'
  if a.overwrite or not(valid_png(cp) and valid_png(wp)):
   for p in (cp,wp):
    if p.exists(): p.unlink()
   z0=wm.null_latent(seed=seed,device=device).to(device=device,dtype=dtype); zw=wm.inject(bit=bit,seed=seed,device=device,alpha=ALPHA).to(device=device,dtype=dtype)
   common=dict(prompt=str(r.prompt),negative_prompt='',num_inference_steps=G0_STEPS,guidance_scale=G0_GUIDANCE,height=HEIGHT,width=WIDTH)
   with torch.inference_mode():
    pipe(latents=z0.clone(),**common).images[0].convert('RGB').save(cp); pipe(latents=zw,**common).images[0].convert('RGB').save(wp)
  rows.append(dict(audit_id=sid,sample_id=sid,split='test',prompt=str(r.prompt),bit=bit,generation_seed=seed,depth=0,generator='sd20',model_id=SD20_MODEL,alpha=ALPHA,beta=BETA,strength=0.,attack_seed='',alpha0_image_path=str(cp.resolve()),wm_image_path=str(wp.resolve()),clean_path=str(cp.resolve()),wm_path=str(wp.resolve()),shard_id=a.shard_id))
  if k%10==0 or k==len(d): pd.DataFrame(rows).sort_values('sample_id').to_csv(sm,index=False); print(f'[G0 shard {a.shard_id}] {k}/{len(d)}',flush=True)

def load_pipe(fam,mid):
 import torch
 if fam=='flux':
  from diffusers import FluxImg2ImgPipeline; p=FluxImg2ImgPipeline.from_pretrained(mid,torch_dtype=torch.bfloat16).to('cuda')
 else:
  from diffusers import AutoPipelineForImage2Image
  kw=dict(torch_dtype=torch.float16,use_safetensors=True,variant='fp16')
  try: p=AutoPipelineForImage2Image.from_pretrained(mid,**kw)
  except Exception: kw.pop('variant',None); p=AutoPipelineForImage2Image.from_pretrained(mid,**kw)
  p=p.to('cuda')
 p.set_progress_bar_config(disable=True); return p

def regen_one(pipe,fam,img,prompt,seed):
 import torch
 from PIL import Image
 img=img.convert('RGB').resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS); g=torch.Generator(device='cpu' if fam=='flux' else 'cuda').manual_seed(seed)
 kw=dict(prompt=str(prompt),image=img,strength=REGEN_STRENGTH,width=WIDTH,height=HEIGHT,num_inference_steps=REGEN_STEPS,generator=g)
 kw.update(guidance_scale=FLUX_GUIDANCE,max_sequence_length=512) if fam=='flux' else kw.update(guidance_scale=SDXL_GUIDANCE)
 with torch.inference_mode(): return pipe(**kw).images[0].convert('RGB')

def run_regen(a,d):
 import torch
 from PIL import Image
 prevp=merged_manifest(a.out_root,d-1)
 if not prevp.exists(): raise FileNotFoundError(f'{prevp}: merge previous depth first')
 prev=pd.read_csv(prevp).sort_values('sample_id').reset_index(drop=True).iloc[a.shard_id::2].copy(); name,fam,mid=CHAIN[d]; cdir,wdir=stage_dirs(a.out_root,d); sm=shard_manifest(a.out_root,f'g{d}',a.shard_id); sm.parent.mkdir(parents=True,exist_ok=True); pipe=load_pipe(fam,mid); rows=[]
 for k,r in enumerate(prev.itertuples(index=False),1):
  sid=int(r.sample_id); seed=transition_seed(sid,d); cp=cdir/f'{sid:05d}.png'; wp=wdir/f'{sid:05d}_wm.png'; sc=r.alpha0_image_path; sw=r.wm_image_path
  if a.overwrite or not valid_png(cp):
   with Image.open(sc) as im: regen_one(pipe,fam,im,r.prompt,seed).save(cp)
  if a.overwrite or not valid_png(wp):
   with Image.open(sw) as im: regen_one(pipe,fam,im,r.prompt,seed).save(wp)
  rows.append(dict(audit_id=sid,sample_id=sid,split='test',prompt=str(r.prompt),bit=getattr(r,'bit',''),generation_seed=getattr(r,'generation_seed',''),depth=d,generator=name,model_id=mid,alpha=ALPHA,beta=BETA,strength=REGEN_STRENGTH,attack_seed=seed,alpha0_image_path=str(cp.resolve()),wm_image_path=str(wp.resolve()),clean_path=str(cp.resolve()),wm_path=str(wp.resolve()),shard_id=a.shard_id))
  if k%10==0 or k==len(prev): pd.DataFrame(rows).sort_values('sample_id').to_csv(sm,index=False); print(f'[G{d} shard {a.shard_id}] {k}/{len(prev)}',flush=True)
 del pipe; gc.collect(); torch.cuda.empty_cache()

def launch(a,stage):
 gs=[x.strip() for x in a.gpus.split(',') if x.strip()]
 if len(gs)!=2: raise ValueError('--gpus requires exactly two IDs, e.g. 3,5')
 logs=a.out_root/'logs'; logs.mkdir(parents=True,exist_ok=True); ps=[]; hs=[]
 for sh,gpu in enumerate(gs):
  log=logs/f'{stage}_gpu{gpu}_shard{sh}.log'; h=open(log,'a',buffering=1); hs.append(h); cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--worker-stage',stage,'--shard-id',str(sh),'--n',str(a.n),'--out-root',str(a.out_root)];
  if a.overwrite: cmd.append('--overwrite')
  env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=gpu; print(f'[LAUNCH] {stage} shard {sh} -> physical GPU {gpu}; log={log}'); ps.append(subprocess.Popen(cmd,stdout=h,stderr=subprocess.STDOUT,env=env))
 codes=[p.wait() for p in ps]
 for h in hs: h.close()
 if any(c!=0 for c in codes): raise RuntimeError(f'{stage} failed codes={codes}; inspect {logs}')
 merge_stage(a.out_root,stage,a.n)

def status(a):
 c=canonical_subset(a.n); print(f'Canonical N={len(c)} IDs={int(c.sample_id.min())}..{int(c.sample_id.max())}')
 for d in range(5):
  p=merged_manifest(a.out_root,d)
  if not p.exists(): print(f'G{d}: pending'); continue
  x=pd.read_csv(p); good=sum(valid_png(Path(r.alpha0_image_path)) and valid_png(Path(r.wm_image_path)) for r in x.itertuples(index=False)); print(f'G{d}: rows={len(x)} valid_pairs={good} {p}')

def main():
 a=cli(); a.out_root.mkdir(parents=True,exist_ok=True)
 if a.worker:
  run_g0(a) if a.worker_stage=='g0' else run_regen(a,int(a.worker_stage[1:])); return
 (a.out_root/'run_config.json').write_text(json.dumps(dict(n=a.n,selection='split=test; sort sample_id; first N',canonical_manifest=str(CANON_MANIFEST),g0_model=SD20_MODEL,alpha=ALPHA,beta=BETA,lineage=['SD2.0',FLUX_MODEL,REALVIS_MODEL,DREAM_MODEL,FLUX_MODEL],gpus=a.gpus,watermark_reinserted_after_g0=False),indent=2))
 if a.stage=='status': status(a); return
 if a.stage=='merge':
  for s in ('g0','g1','g2','g3','g4'):
   try: merge_stage(a.out_root,s,a.n)
   except FileNotFoundError: pass
  status(a); return
 if a.stage in ('g0','all'):
  launch(a,'g0'); print('\nVISUAL-QUALITY GATE: inspect',a.out_root/'generation_0_sd20'); print('After approval run --stage lineage')
  return
 if a.stage=='lineage':
  if not merged_manifest(a.out_root,0).exists(): raise FileNotFoundError('Run --stage g0 first')
  for d in range(1,5): launch(a,f'g{d}')
  status(a)
if __name__=='__main__': main()
