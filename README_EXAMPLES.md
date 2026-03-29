# USG Example Corpus

This corpus is intentionally small, reproducible, and Git-LFS-ready.
All commands are run from the repository root.

| File | Command | Notes |
| --- | --- | --- |
| `examples/audio/00_source_hum.wav` | `cargo run -- render --output examples/audio/00_source_hum.wav --duration 0.2 --style hum --sample-rate 44100` | Clean-ish source file for `go` examples. |
| `examples/audio/01_harsh.wav` | `cargo run -- render --output examples/audio/01_harsh.wav --duration 0.2 --style harsh` | Default float32/192 kHz render. |
| `examples/audio/02_punish.wav` | `cargo run -- render --output examples/audio/02_punish.wav --duration 0.2 --style punish` | Dense clipped punishment texture. |
| `examples/audio/03_int24_punish.wav` | `cargo run -- render --output examples/audio/03_int24_punish.wav --duration 0.2 --style punish --sample-format int --bit-depth 24` | Integer output encoding example. |
| `examples/audio/04_ps1_grit.wav` | `cargo run -- chain --preset ps1_grit --duration 0.2 --output examples/audio/04_ps1_grit.wav` | Built-in chain preset example. |
| `examples/audio/05_arcade_overheat.wav` | `cargo run -- chain --preset arcade_overheat --duration 0.2 --output examples/audio/05_arcade_overheat.wav` | Hotter preset chain example. |
| `examples/audio/06_go_punish.wav` | `cargo run -- go examples/audio/00_source_hum.wav --type punish --output examples/audio/06_go_punish.wav` | `go` defaults to 192 kHz float32. |
| `examples/audio/07_go_glitch_contour.wav` | `cargo run -- go examples/audio/00_source_hum.wav --type glitch --level-contour presets/go_contours/12_step_pattern_01.json --output examples/audio/07_go_glitch_contour.wav` | Versioned contour preset example. |
| `examples/audio/08_go_51.wav` | `cargo run -- go examples/audio/00_source_hum.wav --type punish --upmix 5.1 --output examples/audio/08_go_51.wav` | 5.1 upmix example at 192 kHz. |
| `examples/audio/09_rom_corruption.wav` | `cargo run -- chain --preset rom_corruption --duration 0.2 --output examples/audio/09_rom_corruption.wav` | More unstable chain recipe. |
| `examples/audio/10_binaural_nuisance.wav` | `cargo run -- chain --preset binaural_nuisance --duration 0.2 --output examples/audio/10_binaural_nuisance.wav` | Spatial discomfort preset chain. |
| `examples/audio/11_streamed_glitch.wav` | `USG_STREAM_THRESHOLD_FRAMES=64 cargo run -- render --output examples/audio/11_streamed_glitch.wav --duration 0.2 --style glitch --sample-rate 44100` | Forces streaming render path for regression coverage. |
