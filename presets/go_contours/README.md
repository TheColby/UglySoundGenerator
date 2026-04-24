# USG Go Contour Presets

This folder contains 33 JSON presets for `usg go --level-contour ...`, using `colbys` as the contour key..

Each preset is versioned and currently uses `"version": 1`. The CLI rejects unsupported contour versions instead of silently guessing.

Preset groups:

- `01..11`: `linear_curve_*` (smooth ramps and bends)
- `12..22`: `step_pattern_*` (blocky jumps and gated ugliness)
- `23..33`: `morph_pattern_*` (multi-point long-form morphs)

Use any preset like this:

```bash
cargo run -- go out/clean.wav --type punish --level-contour presets/go_contours/01_linear_curve_01.json --output out/clean_p01.go.wav
```

Inline JSON still works:

```bash
cargo run -- go out/clean.wav --type glitch --level-contour-json '{"version":1,"interpolation":"linear","points":[{"t":0.0,"colbys":100},{"t":0.5,"colbys":900},{"t":1.0,"colbys":250}]}' --output out/clean_inline.go.wav
```
