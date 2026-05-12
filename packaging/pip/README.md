# Pip Support

USG is still a Rust DSP program. The pip package in this repo is a Python-facing shim for environments where `python3 -m pip install -e .` is the easiest way to put helper commands on `PATH`.

## Editable Checkout

```bash
python3 -m pip install -e .
usg-pip-install
uglysoundgenerator --help
```

`usg-pip-install` runs the same underlying Cargo install flow as `scripts/install.sh`:

```bash
cargo install --locked --path . --root "${PREFIX:-$HOME/.local}"
```

The helper honors the same environment variables:

- `PREFIX`: install root, defaulting to `$HOME/.local`
- `FEATURES`: optional Cargo feature list, for example `metal`
- `PROFILE`: optional Cargo profile, defaulting to `release`

## Runtime Delegation

The `uglysoundgenerator` console command resolves the audio engine in this order:

1. `USG_BIN`, if set
2. `usg` on `PATH`
3. `cargo run --manifest-path <checkout>/Cargo.toml -- ...`, when running from an editable checkout

That keeps Python workflows convenient without creating a parallel Python synthesizer.
