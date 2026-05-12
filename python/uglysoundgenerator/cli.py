"""Console shim that delegates to the Rust ``usg`` binary."""

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


def _delegate_command(argv: list[str]) -> list[str] | None:
    env_bin = os.environ.get("USG_BIN")
    if env_bin:
        return [env_bin, *argv]

    path_bin = shutil.which("usg")
    if path_bin:
        return [path_bin, *argv]

    root = _repo_root()
    cargo = shutil.which("cargo")
    if root is not None and cargo:
        return [cargo, "run", "--quiet", "--manifest-path", str(root / "Cargo.toml"), "--", *argv]

    return None


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = _delegate_command(args)
    if command is None:
        print(
            "Unable to find a usg binary. Install it with ./scripts/install.sh, "
            "run usg-pip-install from an editable checkout, or set USG_BIN.",
            file=sys.stderr,
        )
        return 127
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
