#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-$HOME/.local}"
FEATURES="${FEATURES:-}"
PROFILE="${PROFILE:-release}"

usage() {
  cat <<'USAGE'
Install usg from this source checkout.

Environment:
  PREFIX=/path        Install root. Default: $HOME/.local
  FEATURES=metal     Optional Cargo feature list, for example "metal"
  PROFILE=release    Cargo profile. Default: release

Examples:
  ./scripts/install.sh
  PREFIX=/usr/local ./scripts/install.sh
  FEATURES=metal ./scripts/install.sh

Pip-oriented checkout flow:
  python3 -m pip install -e .
  usg-pip-install
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

command -v cargo >/dev/null 2>&1 || {
  echo "cargo is required; install Rust from https://rustup.rs first." >&2
  exit 1
}

cd "$ROOT"
mkdir -p "$PREFIX"

cargo_args=(install --locked --path "$ROOT" --root "$PREFIX")
if [[ -n "$FEATURES" ]]; then
  cargo_args+=(--features "$FEATURES")
fi
if [[ "$PROFILE" != "release" ]]; then
  cargo_args+=(--profile "$PROFILE")
fi

cargo "${cargo_args[@]}"

echo
echo "Installed: $PREFIX/bin/usg"
if [[ ":$PATH:" != *":$PREFIX/bin:"* ]]; then
  echo "Add this to your shell profile if needed:"
  echo "  export PATH=\"$PREFIX/bin:\$PATH\""
fi
