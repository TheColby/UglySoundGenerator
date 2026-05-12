#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="${USG_BIN:-cargo run --quiet --}"
OUT_DIR="${OUT_DIR:-out/demos}"
mkdir -p "$OUT_DIR"

RAW="$OUT_DIR/rising_ugliness_piece.raw.wav"
FINAL="$OUT_DIR/rising_ugliness_piece.wav"

$BIN piece \
  --output "$RAW" \
  --duration "${DURATION:-60}" \
  --layout stereo \
  --styles glitch,punish,catastrophic,meltdown,distort,spank,steal \
  --events-per-second "${EVENTS_PER_SECOND:-18}" \
  --min-event-duration "${MIN_EVENT_DURATION:-0.012}" \
  --max-event-duration "${MAX_EVENT_DURATION:-0.16}" \
  --min-pan-width 0.15 \
  --max-pan-width 2.7 \
  --random-preset catastrophic \
  --randomness 1 \
  --timing-randomness 1.55 \
  --spectral-randomness 1.8 \
  --amplitude-randomness 1.45 \
  --density-randomness 1.7 \
  --spatial-randomness 1.5 \
  --ugliness-trajectory-json '{"version":1,"name":"mostly-awful-demo","interpolation":"step","points":[{"t":0.0,"colbys":350},{"t":0.18,"colbys":780},{"t":0.42,"colbys":950},{"t":0.68,"colbys":620},{"t":0.82,"colbys":1000},{"t":1.0,"colbys":900}]}' \
  --manifest "$OUT_DIR/rising_ugliness_piece.manifest.json" \
  --seed 43003

$BIN go "$RAW" \
  --output "$FINAL" \
  --type dissonance-expand \
  --level 950 \
  --level-contour-json '{"version":1,"interpolation":"linear","points":[{"t":0.0,"colbys":650},{"t":0.45,"colbys":1000},{"t":1.0,"colbys":900}]}' \
  --seed 43004

echo "Wrote $FINAL"
echo "Raw pre-dissonancized piece: $RAW"
