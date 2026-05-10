#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="${USG_BIN:-cargo run --quiet --}"
OUT_DIR="${OUT_DIR:-out/demos}"
mkdir -p "$OUT_DIR"

$BIN piece \
  --output "$OUT_DIR/alarm_choir_714.wav" \
  --duration "${DURATION:-30}" \
  --layout 7.1.4 \
  --scene alarm-choir \
  --region high \
  --random-preset catastrophic \
  --manifest "$OUT_DIR/alarm_choir_714.manifest.json" \
  --seed 44004

echo "Wrote $OUT_DIR/alarm_choir_714.wav"
