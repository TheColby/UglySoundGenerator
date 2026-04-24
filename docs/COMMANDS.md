# Command Guide

This file is the operational reference for `usg`.

## Defaults That Matter

Unless you override them explicitly:

- internal DSP runs in `f64`
- renders write `32-bit float WAV`
- render sample rate defaults to `192000 Hz`
- normalization targets `0 dBFS`

## Core Commands

### `usg render`

Synthesize from scratch.

```bash
cargo run -- render --output out/harsh.wav --duration 2.0 --style harsh
cargo run -- render --output out/glitch.wav --duration 5.0 --style glitch --seed 42
cargo run -- render --output out/int16.wav --duration 1.0 --style hum --sample-format int --bit-depth 16
cargo run -- render --output out/slow.wav --duration 20.0 --style meltdown --jobs 0
```

Key options:

- `--duration <SECONDS>`
- `--sample-rate <HZ>`
- `--style <STYLE>`
- `--sample-format <float|int>`
- `--bit-depth <16|24|32>`
- `--normalize-dbfs <DBFS>`
- `--backend <auto|cpu|metal|cuda>`
- `--gpu-drive`, `--gpu-crush-bits`, `--gpu-crush-mix`

### `usg piece`

Render a full piece made from many short ugly sounds spread across stereo or arbitrary channel counts.

```bash
cargo run -- piece --output out/piece.wav --duration 20 --channels 2 --events-per-second 7
cargo run -- piece --output out/quad.wav --duration 45 --channels 4 --styles glitch,punish,catastrophic
cargo run -- piece --output out/octo.wav --duration 60 --channels 8 --min-event-duration 0.02 --max-event-duration 0.18 --seed 42
```

Key options:

- `--duration <SECONDS>`
- `--channels <N>`
- `--sample-rate <HZ>`
- `--styles <STYLE,...>`
- `--events-per-second <RATE>`
- `--min-event-duration <SECONDS>`
- `--max-event-duration <SECONDS>`
- `--min-pan-width`, `--max-pan-width`
- `--seed <U64>`
- `--backend <auto|cpu|metal|cuda>`

### `usg chain`

Build multi-stage synthesis/effect pipelines.

```bash
cargo run -- chain --stages style:glitch,stutter,pop --duration 3.0 --output out/chain.wav
cargo run -- chain --preset ps1_grit --duration 4.0 --output out/ps1.wav
cargo run -- chain --stages style:buzz,crush,gate,smear --duration 12.0 --output out/long.wav
```

Use explicit prefixes when names overlap:

```bash
cargo run -- chain --stages style:pop,stutter,effect:pop --duration 2.0 --output out/pop-pop.wav
```

### `usg analyze`

Measure ugliness.

```bash
cargo run -- analyze out/chain.wav
cargo run -- analyze out/chain.wav --json
cargo run -- analyze out/chain.wav --model basic
cargo run -- analyze out/chain.wav --model psycho
cargo run -- analyze out/chain.wav --timeline --timeline-format csv --timeline-output out/chain.timeline.csv
```

The JSON output includes score profile metadata so downstream tooling can tell whether a score came from `usg-basic-v1` or `usg-psycho-v1`.

### `usg go`

Force an existing WAV toward a target Colbys value.

```bash
cargo run -- go out/input.wav --level 600 --type glitch --output out/input.go.wav
cargo run -- go out/input.wav --level -250 --type geek --output out/input.less-ugly.wav
cargo run -- go out/input.wav --level 850 --type punish --sample-rate 192000 --output out/input.maxed.wav
```

Important points:

- `--level` is always in **Colbys** (`-1000..1000`)
- the engine internally converts that target to a normalized intensity
- contour JSON may use `colbys`, and old presets using `level` still parse for backward compatibility

#### Contours

File-based:

```bash
cargo run -- go out/input.wav \
  --type glitch \
  --level-contour presets/go_contours/01_linear_curve_01.json \
  --output out/input.contour.wav
```

Inline JSON:

```bash
cargo run -- go out/input.wav \
  --level-contour-json '{"version":1,"interpolation":"linear","points":[{"t":0.0,"colbys":-300},{"t":0.5,"colbys":950},{"t":1.0,"colbys":100}]}' \
  --output out/input.inline.wav
```

#### Upmix + trajectory

```bash
cargo run -- go out/input.wav \
  --level 700 \
  --type punish \
  --upmix 7.1 \
  --coords cartesian \
  --locus 0.2,0.8,0.0 \
  --trajectory line:-0.8,0.1,0.0 \
  --output out/input.71.wav
```

## Support Commands

### `usg presets`

List built-in contour presets and inspect one.

```bash
cargo run -- presets
cargo run -- presets --show 01_linear_curve_01
```

### `usg backends`

Show backend availability and effective GPU post-FX defaults.

```bash
cargo run -- backends
```

### `usg benchmark`

Benchmark render throughput and optionally export structured results.

```bash
cargo run -- benchmark --duration 1.0
cargo run -- benchmark --duration 1.0 --json-output out/bench.json --csv-output out/bench.csv
```

## Power Tools

These are intentionally secondary surfaces.

### `usg mutate`

```bash
cargo run -- mutate out/input.wav --count 12 --out-dir out/mutate
```

### `usg normalize-pack`

```bash
cargo run -- normalize-pack in_wavs out_wavs --level 500
```

### `usg evolve`

```bash
cargo run -- evolve --generations 5 --population 12 --out-dir out/evolve
```

### `usg speech` and `usg speech-pack`

```bash
cargo run -- speech --text "warning warning" --chip ps1 --output out/ps1_speech.wav
cargo run -- speech-pack --text "do not touch that cable" --out-dir out/speech-pack
```

## Where To Go Next

- Scoring details: [METRICS.md](METRICS.md)
- Psychoacoustic equations and references: [PSYCHOACOUSTICS.md](PSYCHOACOUSTICS.md)
- Full reproducible example corpus: [../README_EXAMPLES.md](../README_EXAMPLES.md)
