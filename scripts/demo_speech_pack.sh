#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="${USG_BIN:-cargo run --quiet --}"
OUT_DIR="${OUT_DIR:-out/demos/speech_pack}"
mkdir -p "$OUT_DIR"

$BIN speech-pack \
  --text "UGLY SOUND GENERATOR 404: PLEASE ENJOY THIS BEAUTIFUL MALFUNCTION." \
  --out-dir "$OUT_DIR" \
  --sample-rate 44100 \
  --rank-by balanced \
  --top 5 \
  --seed 5050

echo "Wrote speech pack report to $OUT_DIR/report.html"
