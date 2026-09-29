#!/usr/bin/env bash
set -euo pipefail
PROJECT=reson
DATA=workspace/reson_sd21_canonical_10k
EVAL="scripts/main/05_evaluate_lineage.py"
ALT="$DATA/alternative_lineages"
for CHAIN in order_a order_b order_c order_d; do
  mkdir -p "$ALT/$CHAIN/evaluation"
  python -u "$EVAL"     --data-root "$DATA"     --lineage-root "$ALT/$CHAIN"     --out-dir "$ALT/$CHAIN/evaluation"     --expected-n 1000     --threshold 0.500754654     --depths g0,g1,g2,g3,g4     --skip-clip --skip-fid     2>&1 | tee "$ALT/$CHAIN/evaluation.log"
done
