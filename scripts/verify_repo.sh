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
  scripts/install.sh
  scripts/run_demos.sh
  scripts/refresh_homebrew_formula.sh
  pyproject.toml
  packaging/homebrew/usg.rb
  packaging/homebrew/README.md
  packaging/pip/README.md
  python/uglysoundgenerator/__init__.py
  python/uglysoundgenerator/__main__.py
  python/uglysoundgenerator/cli.py
  python/uglysoundgenerator/installer.py
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
while IFS= read -r script; do
  bash -n "$script"
done < <(find scripts -type f -name '*.sh' | sort)
ruby -c packaging/homebrew/usg.rb >/dev/null
python3 -c 'import ast, pathlib, sys; [ast.parse(pathlib.Path(p).read_text(), filename=p) for p in sys.argv[1:]]' \
  python/uglysoundgenerator/__init__.py \
  python/uglysoundgenerator/__main__.py \
  python/uglysoundgenerator/cli.py \
  python/uglysoundgenerator/installer.py

echo "repo audit ok: docs present, corpus count = $wav_count"
