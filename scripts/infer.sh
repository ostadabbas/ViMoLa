#!/usr/bin/env bash
# Quick ViMoLa inference (Wan 2.2 TI2V + adapters).
# Prerequisites:
#   1. pip install -r requirements.txt
#   2. python scripts/download_weights.py
#   3. Download Wan2.2-TI2V-5B-Diffusers (see README)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WAN_DIR="${WAN_DIR:-./models/Wan2.2-TI2V-5B-Diffusers}"
CKPT_DIR="${CKPT_DIR:-./checkpoints}"
IMAGE="${1:-./assets/sisyphus_input.png}"
PROMPT="${2:-pushing a boulder up a steep hill}"
OUT="${OUT:-./outputs/demo}"

mkdir -p "$OUT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python -m vimola.inference.generate \
  --wan_dir        "$WAN_DIR" \
  --adapter_ckpt   "$CKPT_DIR/adapters.pt" \
  --tokenizer_ckpt "$CKPT_DIR/tokenizer.pt" \
  --t2m_ckpt       "$CKPT_DIR/t2m.pt" \
  --image          "$IMAGE" \
  --prompt         "$PROMPT" \
  --output_dir     "$OUT" \
  --adapter_strength 0.1 \
  --num_frames 49 \
  --height 480 \
  --width 832 \
  --seed 42

echo "Videos in $OUT"
