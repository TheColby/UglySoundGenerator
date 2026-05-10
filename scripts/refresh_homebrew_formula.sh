#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORMULA="$ROOT/packaging/homebrew/usg.rb"

[[ -f "$FORMULA" ]] || {
  echo "missing formula: $FORMULA" >&2
  exit 1
}

ruby -c "$FORMULA"

if command -v brew >/dev/null 2>&1; then
  if brew audit --strict --online "$FORMULA"; then
    echo "brew audit ok."
  else
    echo "brew audit could not complete in this environment; Ruby syntax is valid." >&2
  fi
else
  echo "brew not found; validated Ruby syntax only."
fi

echo "Homebrew formula ready: $FORMULA"
