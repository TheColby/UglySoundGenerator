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
cargo run -- piece --output out/atmos.wav --duration 30 --layout 7.1.4 --events-per-second 9
```

Key options:

- `--duration <SECONDS>`
- `--channels <N>`
- `--layout <mono|stereo|quad|5.1|5.1.2|5.1.4|7.1|7.1.2|7.1.4|9.1.6|custom:N>`
- `--sample-rate <HZ>`
- `--styles <STYLE,...>`
- `--events-per-second <RATE>`
- `--min-event-duration <SECONDS>`
- `--max-event-duration <SECONDS>`
- `--min-pan-width`, `--max-pan-width`
- `--seed <U64>`
- `--backend <auto|cpu|metal|cuda>`

When `--layout` is provided, it sets the channel count automatically and uses named speaker positions, including Atmos-style height layouts.

#### Named Randomness Recipes

Most generative commands accept the same low-level randomness controls:

- `--randomness`
- `--timing-randomness`
- `--spectral-randomness`
- `--amplitude-randomness`
- `--density-randomness`
- `--spatial-randomness`
- `--articulation-randomness`
- `--seed-offset`
- `--seed-salt`
- `--seed-rerolls`

The repo includes recipe presets in `presets/randomness/`, and the shared CLI surface can apply them directly with `--random-preset <stable|restless|feral|catastrophic>`.

| Preset | Intent | Equivalent flags |
| --- | --- | --- |
| `stable` | light deterministic drift | `--random-preset stable` |
| `restless` | active variation without full collapse | `--random-preset restless` |
| `feral` | aggressive timing, color, level, and density movement | `--random-preset feral` |
| `catastrophic` | maximum practical instability | `--random-preset catastrophic` |

Example:

```bash
cargo run -- piece \
  --output out/restless-piece.wav \
  --duration 35 \
  --layout 5.1.2 \
  --styles glitch,punish,catastrophic \
  --events-per-second 8 \
  --random-preset restless \
  --seed 42002
```

#### Piece Scene Recipes

Reusable piece-scene recipes live in `presets/piece_scenes/`, and `usg piece` can apply the same built-in scenes directly with `--scene`.

| Scene | Best for | Layout | Randomness |
| --- | --- | --- | --- |
| `drone-field` | slow wide spatial smear | `7.1.4` | `stable` |
| `failure-chamber` | dense midrange faults in a small room | `5.1.2` | `restless` |
| `arcade-collapse` | fast cabinet-noise montage | `quad` | `feral` |
| `alarm-choir` | large siren-like spatial chorus | `9.1.6` | `catastrophic` |

Example using the `arcade-collapse` scene:

```bash
cargo run -- piece \
  --output out/arcade-collapse.wav \
  --scene arcade-collapse \
  --random-preset feral \
  --manifest out/arcade-collapse.manifest.json \
  --seed 43003
```

`piece` can also follow an ugliness trajectory. The format is the same contour JSON used by `go`: `version`, optional `name`/`description`, `interpolation` (`linear` or `step`), and `points` with normalized time `t` plus `colbys`.

```bash
cargo run -- piece \
  --output out/rising_ugliness.wav \
  --duration 60 \
  --layout stereo \
  --scene arcade-collapse \
  --ugliness-trajectory-json '{"version":1,"name":"rising","interpolation":"linear","points":[{"t":0.0,"colbys":-700},{"t":0.5,"colbys":150},{"t":1.0,"colbys":950}]}' \
  --manifest out/rising_ugliness.manifest.json \
  --seed 43003
```

The trajectory shapes event style selection, layer density, duration, gain, and the per-event `target_colbys` / `ugliness_intensity` fields recorded in the manifest.

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

List built-in contour, chain, randomness, and piece-scene presets and inspect one.

```bash
cargo run -- presets
cargo run -- presets --show 01_linear_curve_01
cargo run -- presets --kind chain --show ps1_grit
cargo run -- presets --kind randomness --show feral
cargo run -- presets --kind piece-scene --show arcade-collapse --json
```

Current CLI preset discovery covers built-in `go` contours, `chain` stage presets, named randomness profiles, and reusable `piece` scene recipes.

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

### `usg speech`

Render chip-speech-inspired audio from inline text or a UTF-8 text file.

```bash
cargo run -- speech --text "warning warning" --profile c64-sam --output out/c64_speech.wav
cargo run -- speech --text "WARNING 404!" --profile tms5220 --output out/tms5220_warning.wav
cargo run -- speech --text-file docs/script.txt --input-mode paragraph --profile c64-sam --output out/script.wav
cargo run -- speech --text "do not touch that cable" --profile votrax-sc01 --primary-osc phoneme --secondary-osc buzz --tertiary-osc koch --output out/votrax.wav
```

The v0.5 speech surface is intentionally deeper than a single `--text -> wav` path:

- text input: `--text <TEXT>` or `--text-file <PATH>`
- normalization: enabled by default; `--no-normalize-text` disables digit expansion, quote/dash cleanup, whitespace cleanup, and automatic uppercasing
- input modes: `--input-mode auto|character|word|sentence|paragraph`
- profiles: `--profile votrax-sc01|tms5220|sp0256|mea8000|s14001a|c64-sam|arcadey90s|handheld-lcd|speak-and-spell|macintalk|yamaha-psg|amiga-narrator`
- profile backends in exports: `lpc`, `formant-grid`, `sam-vocal-tract`, `arcade-pcm`, `delta-modulation`, `klatt-cascade`, or `psg-formant`
- oscillator slots: `--primary-osc`, `--secondary-osc`, and `--tertiary-osc`
- oscillator choices: `sine`, `pulse`, `triangle`, `saw`, `noise`, `buzz`, `formant`, `vowel`, `ring`, `fold`, `organ`, `fm`, `sync`, `lfsr`, `grain`, `chirp`, `subharmonic`, `reed`, `click`, `comb`, `koch`, `mandelbrot`, `strange`, `phoneme`, `glottal`, `aspiration`, `nasal-buzz`, `robotic-vocoder`, `plosive-burst`, `whisper`, `lpc-pulse`, `arcade-delta`, `casio-formant`, `phase-distort`
- timing/prosody: `--units-per-second`, `--word-gap-ms`, `--sentence-gap-ms`, `--paragraph-gap-ms`, `--punctuation-gap-ms`, `--word-accent`, `--sentence-lilt`, `--paragraph-decline`, `--emphasis`, `--attack-ms`, `--release-ms`
- voice color: `--pitch-hz`, `--pitch-jitter`, `--vibrato-hz`, `--vibrato-depth`, `--formant-shift`, `--consonant-noise`, `--vowel-mix`, `--hiss`, `--buzz`, `--fold`, `--chaos`, `--robotize`, `--ring-mix`, `--sub-mix`, `--nasal`, `--throat`, `--drift`, `--resampler-grit`, `--excitation`, `--breathiness`, `--plosive-pop`, `--sibilance`, `--nasal-leak`, `--phoneme-slur`, `--coarticulation`, `--phrase-swing`
- chip grime: `--bitcrush-bits`, `--sample-hold-hz`, plus shared backend post-FX flags
- reproducibility: `--seed`, `--unit-rerolls`, and the shared randomness controls including `--random-preset`
- exports: `--analysis-json <PATH>` writes render metadata, psycho analysis, and intelligibility; `--timeline-json <PATH>` writes the parsed unit/phoneme timeline

Profile-to-backend mapping:

| Profiles | Backend family |
| --- | --- |
| `votrax-sc01`, `tms5220`, `sp0256` | `lpc` |
| `mea8000`, `s14001a` | `formant-grid` |
| `c64-sam` | `sam-vocal-tract` |
| `arcadey90s`, `handheld-lcd` | `arcade-pcm` |
| `speak-and-spell` | `delta-modulation` |
| `macintalk`, `amiga-narrator` | `klatt-cascade` |
| `yamaha-psg` | `psg-formant` |

Examples:

```bash
cargo run -- speech \
  --text "Deck 7 pressure is 42 percent. Evacuate?" \
  --profile sp0256 \
  --input-mode sentence \
  --primary-osc phoneme \
  --secondary-osc formant \
  --tertiary-osc lfsr \
  --analysis-json out/deck7.analysis.json \
  --timeline-json out/deck7.timeline.json \
  --output out/deck7.wav
```

```bash
cargo run -- speech \
  --text "bad radio paragraph one\n\nbad radio paragraph two" \
  --input-mode paragraph \
  --profile handheld-lcd \
  --units-per-second 8 \
  --paragraph-gap-ms 360 \
  --paragraph-decline 0.22 \
  --primary-osc phoneme \
  --secondary-osc noise \
  --tertiary-osc click \
  --output out/paragraph_lcd.wav
```

Timeline JSON is the main phoneme parser diagnostic. Each row includes the source token, normalized token, derived label, unit kind, token/phoneme indices, start/end/duration/gap timing, emphasis, pitch/formants, `voiced`, `noisy`, oscillator/excitation family, parse rule/confidence, unit seed, and `backend_kind`.

### `usg speech-pack`

Render all speech-chip profiles for the same input, analyze them, compute an intelligibility index, and write ranked reports.

```bash
cargo run -- speech-pack --text "do not touch that cable" --out-dir out/speech-pack
cargo run -- speech-pack --text "do not touch that cable" --rank-by ugliness --out-dir out/speech-pack-ugly
cargo run -- speech-pack --text-file docs/script.txt --rank-by intelligibility --summary out/speech-summary.json --csv out/speech-ranking.csv --html out/speech-report.html
```

Key options:

- `--rank-by <ugliness|intelligibility|balanced>`
- `--model <basic|psycho>`
- `--input-mode <auto|character|word|sentence|paragraph>`
- `--summary <PATH>`
- `--csv <PATH>`
- `--html <PATH>`
- `--top <N>`
- `--seed <U64>`
- `--seed-stride <U64>`
- shared output, backend, jobs, GPU post-FX, and randomness flags

`speech-pack` writes one WAV per chip profile plus:

- `summary.json`: full analysis objects, intelligibility breakdowns, seeds, backend info, and the ranking table
- `ranking.csv`: compact profile/output/Colbys/intelligibility/rank-score table
- `report.html`: browser-friendly comparison report

The rank modes let you choose the tradeoff: `ugliness` rewards higher Colbys, `intelligibility` rewards the speech intelligibility index, and `balanced` favors material that stays legible while still scoring ugly.

## Where To Go Next

- Scoring details: [METRICS.md](METRICS.md)
- Psychoacoustic equations and references: [PSYCHOACOUSTICS.md](PSYCHOACOUSTICS.md)
- Full reproducible example corpus: [../README_EXAMPLES.md](../README_EXAMPLES.md)
