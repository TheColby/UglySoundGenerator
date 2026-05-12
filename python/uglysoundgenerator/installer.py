"""Installer helper for pip/editable checkout users."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "Cargo.toml").is_file() and (parent / "src").is_dir():
            return parent
    return None


def main() -> int:
    root = _repo_root()
    if root is None:
        print(
            "usg-pip-install needs an editable/source checkout so it can see Cargo.toml. "
            "Run: python3 -m pip install -e .",
            file=sys.stderr,
        )
        return 2

    cargo = shutil.which("cargo")
    if cargo is None:
        print("cargo is required; install Rust from https://rustup.rs first.", file=sys.stderr)
        return 127

    prefix = Path(os.environ.get("PREFIX", str(Path.home() / ".local"))).expanduser()
    args = [cargo, "install", "--locked", "--path", str(root), "--root", str(prefix)]

    features = os.environ.get("FEATURES", "")
    if features:
        args.extend(["--features", features])

    profile = os.environ.get("PROFILE", "release")
    if profile != "release":
        args.extend(["--profile", profile])

    result = subprocess.call(args)
    if result != 0:
        return result

    print(f"\nInstalled: {prefix / 'bin' / 'usg'}")
    print("Use the Python shim as `uglysoundgenerator ...` or call `usg ...` directly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
