# RESON: Persistent Watermarking Across Heterogeneous Generative Lineages

[Project page](https://airesearchfolio.github.io/reson)

Official code for **RESON** (under review at ICLR 2027) · 

RESON is a generation-time latent watermark for **insert-once, detect-later** provenance. A keyed, message-bearing carrier is embedded **once** in the source diffusion noise. The watermark can then be detected directly from pixels after the image has been re-synthesized by a chain of different generators, with no reinsertion, no generator modification and no diffusion inversion.

```
SD2.1 (G0, watermark inserted) → FLUX.1-dev (G1) → RealVisXL V5.0 (G2) → DreamShaperXL (G3) → FLUX.1-dev (G4)
```

## Headline result (canonical lineage, SD2.1 source, 1,000 held-out pairs)

| Depth | G0 | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|
| TPR @ 1% FPR (%) | 100.00 | 100.00 | 98.30 | 93.10 | 92.50 |
| ROC-AUC | 1.0000 | 0.9996 | 0.9980 | 0.9967 | 0.9963 |

The detector is trained on source generations only. Its threshold is calibrated on the validation split at 1% FPR and then frozen for every depth, generator and attack.

## Installation

```bash
conda create -n reson python=3.10 -y
conda activate reson
pip install -r requirements.txt
huggingface-cli login   # FLUX.1-dev is gated: accept its license on Hugging Face first
```

Models used: `sd2-community/stable-diffusion-2-1-base` (source), `black-forest-labs/FLUX.1-dev`, `SG161222/RealVisXL_V5.0`, `Lykon/dreamshaper-xl-1-0`, and `RunDiffusion/Juggernaut-XL-v8` (lineage-order experiment). Experiments were run on NVIDIA A100 GPUs.

**Run every command from the repository root.** Scripts locate the core modules through `--project-root reson` and write to `workspace/`.

## Repository structure

```
reson/                      core method (loaded via --project-root reson)
  config.py                 default settings (alpha = 0.155, beta = 0.40, lineage chain)
  watermark.py              keyed carrier construction and signed latent injection
  detector.py               latent joint decoder (presence + payload heads)
  diffusion.py              pipeline loaders and img2img regeneration
  data.py, prepare_data.py  prompt split construction
  run_reson_sd21_multigpu.py  multi-GPU paired clean / RESON generation (used by ablations)
scripts/
  main/                     canonical pipeline: generate → train → lineage → evaluate  (Table 1)
  lineage_order/            alternative generator orders                               (Table 2)
  perturbations/            conventional perturbations, SD2.1 sweep and SD2.0 transfer (Fig. 5)
  removal_attacks/          VAE / diffusion regeneration / RINSE removal attacks       (Table 4)
  carrier_ablation/         matched carrier-structure ablation, beta ∈ {0, 0.4, 1}     (Table 5, Fig. 7)
  operating_point/          alpha × steps sweep and quality frontier                   (Fig. 3)
  capacity/                 source-level payload capacity, 1–32 bits                   (Table 8)
  multibit/                 4-bit and 8-bit detectors and lineages                     (Tables 10–11, Fig. 6)
  quality/                  CLIP / FID evaluation                                      (Tables 9, 13)
  sd20/                     SD2.0 source lineage                                       (Table 1, SD2.0)
  latency/                  detection latency benchmark
baselines/                  lineage generation and native-detector evaluation for
                            SERUM, Tree-Ring, Gaussian Shading and WIND
plots/                      figure scripts
```

## Canonical pipeline (Table 1)

```bash
# 1. Paired clean / RESON source generation (SD2.1, 512×512). Shard across GPUs with --shard-id / --num-shards.
python scripts/main/01_generate_source.py --prompt-csv reson/prompt_split.csv \
    --out-dir workspace/reson_sd21_canonical_10k --gpu 0 --shard-id 0 --num-shards 1
python scripts/main/02_merge_source_shards.py --out-dir workspace/reson_sd21_canonical_10k --expected 10000

# 2. Detector training on source data only (8,000 train / 1,000 validation / 1,000 test).
#    The checkpoint is selected by validation AUC; the threshold is calibrated on validation at 1% FPR.
python scripts/main/03_train_detector.py --manifest workspace/reson_sd21_canonical_10k/manifest.csv \
    --out-dir workspace/reson_sd21_canonical_10k/final_detector

# 3. Heterogeneous lineage G1–G4 on the held-out test split (no reinsertion, strength 0.50).
python scripts/main/04_lineage_multigpu.py --gpus 0,1 --data-root workspace/reson_sd21_canonical_10k

# 4. Evaluation at every depth with the frozen detector and threshold.
python scripts/main/05_evaluate_lineage.py --data-root workspace/reson_sd21_canonical_10k
```

`05_evaluate_lineage.py` reports ROC-AUC, frozen-threshold TPR and realized FPR with 95% Wilson intervals, payload bit accuracy, CLIP similarity and clean-vs-watermarked FID. Use `--skip-clip --skip-fid` for detection metrics only. Every script documents its options with `--help`.

## Other experiments

| Paper result | Scripts |
|---|---|
| Table 1, SD2.0 source | `scripts/sd20/` |
| Table 1, baselines | `baselines/regenerate_baseline_lineage.py`, then each method's evaluator in `baselines/` |
| Table 2, lineage order: permutations of the canonical models (N = 300) | `scripts/lineage_order/generate_order_permutations.py`, `evaluate_lineage_subset.py` |
| Table 2, lineage order: alternative generator chain | `scripts/lineage_order/launch_alternative_lineages.py`, `resume_alternative_orders_ab.py`, `evaluate_alternative_lineages.sh` |
| Table 4, removal attacks | `scripts/removal_attacks/setup_watermarkattacker.sh`, `run_removal_attacks.py`, `evaluate_removal_attacks.py` |
| Figure 5, perturbation sweeps | `scripts/perturbations/generate_perturbations.py`, `evaluate_perturbations.py` |
| Table 5, Figure 7, carrier ablation | `scripts/carrier_ablation/run_carrier_mechanism_ablation_3gpu.py` |
| Figure 3, operating point | `scripts/operating_point/` |
| Table 8, payload capacity | `scripts/capacity/` |
| Tables 10–11, Figure 6, multi-bit | `scripts/multibit/` |
| Tables 9 and 13, quality | `scripts/quality/` |
| Detection latency | `scripts/latency/benchmark_detection_latency.py` |

## Data

Prompts come from the public [Stable-Diffusion-Prompts](https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts) dataset; `reson/prompt_split.csv` fixes the prompts, seeds and train / validation / test split used in the paper. COCO 2014 validation captions and images are used only for FID. Generated images and checkpoints are not included in this repository.

## Baselines

Baselines use their official implementations, which are not redistributed here. Clone them into `third_party/`:
[Tree-Ring](https://github.com/YuxinWenRick/tree-ring-watermark), [Gaussian Shading](https://github.com/bsmhmmlf/Gaussian-Shading), [WIND](https://github.com/Kasraarabi/Hidden-in-the-Noise), [SERUM](https://github.com/Hubizon/SERUM), and [WatermarkAttacker](https://github.com/XuandongZhao/WatermarkAttacker) for the removal attacks. No pretrained SERUM detector is publicly available, so SERUM is trained with the authors' official code and default hyperparameters.

## Citation

```bibtex
@inproceedings{anonymous2027reson,
  title     = {RESON: Persistent Watermarking Across Heterogeneous Generative Lineages},
  author    = {Anonymous},
  booktitle = {Submitted to the International Conference on Learning Representations (ICLR)},
  year      = {2027},
  note      = {Under review}
}
```

## License

Released for research purposes. Model weights used by this code are subject to their own licenses, in particular the FLUX.1-dev non-commercial license.
