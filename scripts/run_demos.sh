#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT/scripts/demo_piece_trajectory.sh"
"$ROOT/scripts/demo_atmos_piece.sh"
"$ROOT/scripts/demo_speech_pack.sh"
"$ROOT/scripts/demo_analysis_joke.sh"

echo "All demos complete. Outputs are under ${OUT_DIR:-out/demos}."
