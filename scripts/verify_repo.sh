#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

required_files=(
  README.md
  README_EXAMPLES.md
  docs/COMMANDS.md
  docs/METRICS.md
  docs/PSYCHOACOUSTICS.md
  scripts/generate_example_corpus.sh
)

for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || {
    echo "missing required file: $path" >&2
    exit 1
  }
done

wav_count="$(find examples/audio -maxdepth 1 -type f -name '*.wav' | wc -l | tr -d ' ')"
[[ "$wav_count" == "333" ]] || {
  echo "expected 333 example wavs, found $wav_count" >&2
  exit 1
}

grep -q '333 reproducible Git-LFS-tracked WAV files' README_EXAMPLES.md
bash -n scripts/generate_example_corpus.sh

echo "repo audit ok: docs present, corpus count = $wav_count"
