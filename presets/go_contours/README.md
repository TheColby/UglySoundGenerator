# USG Go Contour Presets

This folder contains 33 JSON presets for `usg go --level-contour ...`, using `colbys` as the contour key.

Each preset is versioned and currently uses `"version": 1`. The CLI rejects unsupported contour versions instead of silently guessing.

Contours are a compact way to make ugliness move over time instead of asking for one fixed target. That matters because a constant high score can turn into a static wall, while a moving target creates audible phrasing: lunges, collapses, recoveries, ramps, gates, fake-outs, and the occasional tiny vacation from sonic punishment.

Preset groups:

- `01..11`: `linear_curve_*` (smooth ramps and bends)
- `12..22`: `step_pattern_*` (blocky jumps and gated ugliness)
- `23..33`: `morph_pattern_*` (multi-point long-form morphs)

Use the linear curves when you want readable long gestures. Use the step patterns for rhythmic interruption, bad-machine behavior, and sudden jumps in perceived damage. Use the morph patterns when you want a file to feel like it is gradually discovering new bad ideas.

Use any preset like this:

```bash
cargo run -- go out/clean.wav --type punish --level-contour presets/go_contours/01_linear_curve_01.json --output out/clean_p01.go.wav
```

Higher-level targets do not guarantee every moment will measure at that exact Colby value; `go` maps the target into processing intensity, then the input material and selected uglification type determine the actual result. Treat presets as reproducible control gestures rather than laboratory-grade perceptual clamps.

Inline JSON still works:

```bash
cargo run -- go out/clean.wav --type glitch --level-contour-json '{"version":1,"interpolation":"linear","points":[{"t":0.0,"colbys":100},{"t":0.5,"colbys":900},{"t":1.0,"colbys":250}]}' --output out/clean_inline.go.wav
```

Preset authoring checklist:

- Keep `t` values normalized from `0.0` to `1.0`.
- Keep `colbys` values inside `-1000..1000`.
- Prefer a small number of points for readable behavior.
- Add dense points only when the gesture is intentionally twitchy.
- Preserve the `version` key so future parsers can distinguish old contour semantics from new ones.
