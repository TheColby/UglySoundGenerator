#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="${USG_BIN:-cargo run --quiet --}"
OUT_DIR="${OUT_DIR:-out/demos}"
mkdir -p "$OUT_DIR"

$BIN piece \
  --output "$OUT_DIR/rising_ugliness_piece.wav" \
  --duration "${DURATION:-60}" \
  --layout stereo \
  --scene arcade-collapse \
  --random-preset feral \
  --ugliness-trajectory-json '{"version":1,"name":"demo-rise","interpolation":"linear","points":[{"t":0.0,"colbys":-700},{"t":0.35,"colbys":50},{"t":0.72,"colbys":650},{"t":1.0,"colbys":950}]}' \
  --manifest "$OUT_DIR/rising_ugliness_piece.manifest.json" \
  --seed 43003

echo "Wrote $OUT_DIR/rising_ugliness_piece.wav"
