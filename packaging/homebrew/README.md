# Homebrew Support

The bundled formula installs USG directly from the GitHub `main` branch:

```bash
brew install --HEAD ./packaging/homebrew/usg.rb
```

For a personal tap, copy `usg.rb` into the tap's `Formula/` directory and install with:

```bash
brew install --HEAD your-tap/usg
```

This formula is intentionally HEAD-first until release tarballs and checksums are part of the regular release process. The formula test runs `usg --help` and renders a tiny WAV so both CLI startup and audio output get smoke-tested.

Formula maintenance helper:

```bash
./scripts/refresh_homebrew_formula.sh
```
