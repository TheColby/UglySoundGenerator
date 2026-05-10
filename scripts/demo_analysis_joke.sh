#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="${USG_BIN:-cargo run --quiet --}"
OUT_DIR="${OUT_DIR:-out/demos}"
mkdir -p "$OUT_DIR"

$BIN render \
  --output "$OUT_DIR/punish_for_joke_analysis.wav" \
  --duration "${DURATION:-3}" \
  --style punish \
  --seed 9090

$BIN analyze "$OUT_DIR/punish_for_joke_analysis.wav" --joke --json-output "$OUT_DIR/punish_for_joke_analysis.json"

echo "Wrote $OUT_DIR/punish_for_joke_analysis.json"
