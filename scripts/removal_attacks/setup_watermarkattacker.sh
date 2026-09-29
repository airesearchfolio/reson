#!/usr/bin/env bash
set -euo pipefail
ROOT="third_party"
REPO="$ROOT/WatermarkAttacker"
mkdir -p "$ROOT"
if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/XuandongZhao/WatermarkAttacker.git "$REPO"
else
  echo "Repo already exists: $REPO"
fi
cd "$REPO"
echo "Official repo ready at: $REPO"
echo
echo "IMPORTANT: use a dedicated conda env if its requirements conflict with your RESON env."
echo "Official requirements are in: $REPO/requirements.txt"
