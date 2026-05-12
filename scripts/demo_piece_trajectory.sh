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
  --styles "${EVENT_STYLES:-pop,glitch,spank,steal,catastrophic}" \
  --sections "${SECTIONS:-6}" \
  --rest-probability "${REST_PROBABILITY:-0.38}" \
  --section-contrast "${SECTION_CONTRAST:-0.72}" \
  --return-probability "${RETURN_PROBABILITY:-0.18}" \
  --events-per-second "${EVENTS_PER_SECOND:-4.5}" \
  --min-event-duration "${MIN_EVENT_DURATION:-0.004}" \
  --max-event-duration "${MAX_EVENT_DURATION:-0.04}" \
  --min-pan-width 0.08 \
  --max-pan-width 3.1 \
  --ugliness-trajectory-json '{"version":1,"name":"sparse-clicky-awful-demo","interpolation":"step","points":[{"t":0.0,"colbys":-120},{"t":0.08,"colbys":880},{"t":0.16,"colbys":140},{"t":0.27,"colbys":960},{"t":0.39,"colbys":240},{"t":0.56,"colbys":1000},{"t":0.72,"colbys":80},{"t":0.84,"colbys":940},{"t":1.0,"colbys":300}]}' \
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
