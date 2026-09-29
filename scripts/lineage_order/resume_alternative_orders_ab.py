#!/usr/bin/env python3
"""
Resume RESON alternative lineage orders A/B without JuggernautXL.

Existing work is preserved:
  order_a: G1 DreamShaperXL, G2 RealVisXL, G3 FLUX -> generate G4 DreamShaperXL
  order_b: G1 RealVisXL -> generate G2 FLUX, G3 DreamShaperXL, G4 RealVisXL

This wrapper reuses the project's existing regeneration implementation rather
than duplicating its image-to-image semantics. It writes a temporary run_config
with the corrected chain, then invokes regenerate_alternative_lineages.py.

Usage:
  python resume_reson_alternative_orders_ab.py --order order_a --n 1000
  python resume_reson_alternative_orders_ab.py --order order_b --n 1000
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path("workspace/reson_sd21_canonical_10k")
ALT = ROOT / "alternative_lineages"
SRC = Path("reson")
REGEN = Path(__file__).resolve().with_name("regenerate_alternative_lineages.py")

CHAINS = {
    "order_a": [
        ["g1","sdxl","Lykon/dreamshaper-xl-1-0","dreamshaper_xl"],
        ["g2","sdxl","SG161222/RealVisXL_V5.0","realvis_xl"],
        ["g3","flux","black-forest-labs/FLUX.1-dev","flux"],
        ["g4","sdxl","Lykon/dreamshaper-xl-1-0","dreamshaper_xl"],
    ],
    "order_b": [
        ["g1","sdxl","SG161222/RealVisXL_V5.0","realvis_xl"],
        ["g2","flux","black-forest-labs/FLUX.1-dev","flux"],
        ["g3","sdxl","Lykon/dreamshaper-xl-1-0","dreamshaper_xl"],
        ["g4","sdxl","SG161222/RealVisXL_V5.0","realvis_xl"],
    ],
}

def rows(p):
    if not p.exists(): return 0
    with p.open("r", errors="ignore") as f:
        return max(sum(1 for _ in f)-1, 0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--order", choices=CHAINS, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--base-seed", type=int, default=20260903)
    ap.add_argument("--dry-run", action="store_true")
    a=ap.parse_args()

    od=ALT/a.order
    od.mkdir(parents=True, exist_ok=True)
    cfgp=od/"run_config.json"
    old={}
    if cfgp.exists():
        old=json.loads(cfgp.read_text())
        (od/"run_config_before_resume.json").write_text(json.dumps(old,indent=2))

    cfg={
        "chain": CHAINS[a.order],
        "strength": a.strength,
        "base_seed": a.base_seed,
        "n": a.n,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES",""),
        "watermark_reinserted_after_g0": False,
    }
    cfgp.write_text(json.dumps(cfg,indent=2))

    print("Corrected chain:")
    for x in cfg["chain"]: print("  ", x)
    print("\nCurrent manifests:")
    for g in ("g1","g2","g3","g4"):
        print(f"  {g}: {rows(od/f'manifest_{g}.csv')}/{a.n}")

    if a.dry_run:
        return

    # Discover CLI from --help so we do not silently invent flags.
    help_txt=subprocess.run([sys.executable,str(REGEN),"--help"],
                            capture_output=True,text=True).stdout
    cmd=[sys.executable,str(REGEN)]
    # Add only flags actually supported by the existing script.
    candidates=[
        ("--order",a.order),
        ("--n",str(a.n)),
        ("--strength",str(a.strength)),
        ("--base-seed",str(a.base_seed)),
        ("--root",str(ROOT)),
        ("--data-root",str(ROOT)),
        ("--out-root",str(ALT)),
        ("--alternative-root",str(ALT)),
    ]
    used=set()
    for flag,val in candidates:
        if flag in help_txt:
            # avoid supplying synonymous roots twice
            group="root" if flag in {"--root","--data-root","--out-root","--alternative-root"} else flag
            if group in used: continue
            cmd += [flag,val]; used.add(group)

    print("\nLaunching existing regenerator:")
    print(" ".join(cmd), flush=True)
    rc=subprocess.call(cmd,cwd=str(SRC))
    raise SystemExit(rc)

if __name__=="__main__":
    main()
