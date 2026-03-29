# UglySoundGenerator (usg)

<p align="center">
  <img src="assets/usg-logo.svg" alt="usg glitch logo" width="380">
</p>

`usg` is a Rust command-line tool for creating and inspecting intentionally ugly sounds.

Render policy (v-next):

- Internal DSP uses 64-bit (`f64`) processing.
- Output WAV format defaults to 32-bit float PCM; use `--sample-format int --bit-depth 16|24|32` to override.
- Render sample rate defaults to `192000` Hz unless set via CLI.
- Peak normalization defaults to `0 dBFS` unless disabled or changed via CLI.
- Long renders use chunked streaming (constant-memory render path).
- Backend-aware execution supports `cpu`, `metal`, `cuda`, or `auto`.
- Pack rendering is massively parallel by default (core-count workers).

## Commands

- `usg render`: Generate an ugly WAV file.
- `usg analyze`: Analyze a WAV file and report ugliness metrics.
- `usg render-pack`: Render all styles + analyze + rank ugliness.
- `usg speech`: Render chiptune speech from inline text or text files.
- `usg go`: Force any input WAV to a target ugliness level (`1..1000`).
- `usg chain`: Chain synthesis/effects stages into one output file.
- `usg styles`: List available ugliness style profiles.
- `usg backends`: Show CPU/Metal/CUDA backend availability.
- `usg benchmark`: Compare backend render throughput and optionally export JSON/CSV.
- `usg presets`: List and inspect built-in ugliness contour presets.
- `usg marathon`: Bulk-generate large ugly sound libraries.

## Quick Start

```bash
cargo run -- render --output out/ugly.wav --duration 3 --style harsh
cargo run -- render --output out/catastrophic.wav --duration 3 --style catastrophic
cargo run -- speech --output out/voice.wav --text "HELLO FROM 1984" --profile sp0256 --primary-osc phoneme --secondary-osc mandelbrot --tertiary-osc strange
cargo run -- analyze out/ugly.wav
cargo run -- analyze out/ugly.wav --json
cargo run -- analyze out/ugly.wav --model psycho --fft-size 2048 --hop-size 512
cargo run -- render-pack --out-dir out/pack --model psycho --seed 12345
cargo run -- go out/clean.wav --level 800 --type punish --output out/clean.go.wav
cargo run -- go out/clean.wav --type punish --level-contour presets/go_contours/01_linear_curve_01.json --output out/clean_p01.go.wav
cargo run -- chain --stages glitch,stutter,pop --output out/chain.wav --play
cargo run -- styles
cargo run -- presets
cargo run -- presets --show 12_step_pattern_01
cargo run -- backends
cargo run -- benchmark --runs 5 --duration 1.0 --style glitch --jobs 12
cargo run -- benchmark --runs 5 --duration 1.0 --style glitch --json-output out/bench.json --csv-output out/bench.csv
cargo run -- render --output out/ugly_int24.wav --duration 3 --style punish --sample-format int --bit-depth 24
cargo run -- marathon --out-dir out/marathon --count 256 --min-duration 0.1 --max-duration 600 --backend auto --jobs 12
```

Working directory note:

- If your shell is already in `out/`, use `clean.wav` and `clean_preset.go.wav` instead of `out/clean.wav` and `out/clean_preset.go.wav`.
- Preset paths are resolved from your current directory, so from `out/` use `../presets/go_contours/...`.

## Ugly Recipes (Many CLI Examples)

`usg render` is now streamed for long renders, so you can generate very large files without allocating the whole waveform in RAM.

Duration limits:

- `usg render`: `0.1s` to `86400s` (24 hours)
- `usg chain`: `0.1s` to `86400s` with automatic streaming for long renders

### Fast short-hit ugliness

```bash
cargo run -- render --output out/hit_glitch.wav --duration 0.12 --style glitch
cargo run -- render --output out/hit_pop.wav --duration 0.20 --style pop --gain 1.0
cargo run -- render --output out/hit_punish.wav --duration 0.35 --style punish --normalize-dbfs -0.2
cargo run -- render --output out/hit_spank.wav --duration 0.25 --style spank --seed 77
cargo run -- render --output out/hit_distort.wav --duration 0.40 --style distort --backend cpu
```

### Mid-length brutal textures

```bash
cargo run -- render --output out/grind_buzz_30s.wav --duration 30 --style buzz --sample-rate 192000
cargo run -- render --output out/grind_rub_45s.wav --duration 45 --style rub --gain 0.95
cargo run -- render --output out/grind_wink_60s.wav --duration 60 --style wink --seed 555
cargo run -- render --output out/grind_steal_90s.wav --duration 90 --style steal --normalize-dbfs 0
cargo run -- render --output out/grind_lucky_120s.wav --duration 120 --style lucky --backend auto --jobs 12
```

### Multi-hour ugly exports

```bash
cargo run -- render --output out/ugly_1h_glitch.wav --duration 3600 --style glitch --backend auto --jobs 12
cargo run -- render --output out/ugly_2h_punish.wav --duration 7200 --style punish --seed 9999 --backend auto --jobs 16
cargo run -- render --output out/ugly_4h_distort.wav --duration 14400 --style distort --gain 0.9 --normalize-dbfs -0.3
cargo run -- render --output out/ugly_8h_lucky.wav --duration 28800 --style lucky --backend auto --jobs 20
cargo run -- render --output out/ugly_24h_harsh.wav --duration 86400 --style harsh --backend cpu --jobs 24
cargo run -- render --output out/ugly_30m_catastrophic.wav --duration 1800 --style catastrophic --backend auto --jobs 24
```

### Extreme chain examples (ugly and uglier)

```bash
cargo run -- chain --stages glitch,stutter,pop --output out/chain_gsp.wav --duration 8 --play
cargo run -- chain --stages style:punish,crush,gate,smear --output out/chain_pcgs.wav --duration 20
cargo run -- chain --stages style:buzz,stutter,stutter,pop,smear --output out/chain_bssps.wav --duration 45
cargo run -- chain --stages style:steal,gate,pop,crush --output out/chain_sgpc.wav --duration 60 --seed 42
cargo run -- chain --stages style:lucky,smear,stutter,effect:pop --output out/chain_lssp.wav --duration 90
cargo run -- chain --stages style:punish,stutter,crush,smear,pop --output out/chain_1h.wav --duration 3600 --backend auto --jobs 16
```

### Batch generate ugly libraries

```bash
for s in harsh digital meltdown glitch pop buzz rub hum distort spank punish steal catastrophic wink lucky; do
  cargo run -- render --output "out/library/${s}_10m.wav" --duration 600 --style "$s" --backend auto --jobs 12
done
```

### One-command mass generation (`usg marathon`)

```bash
cargo run -- marathon --out-dir out/marathon_1 --count 512 --min-duration 0.10 --max-duration 30 --backend auto --jobs 12
cargo run -- marathon --out-dir out/marathon_2 --count 200 --min-duration 30 --max-duration 600 --styles glitch,punish,distort,steal --seed 4242
cargo run -- marathon --out-dir out/marathon_3 --count 64 --min-duration 600 --max-duration 3600 --styles harsh,meltdown,buzz --backend cpu --jobs 20
```

Marathon writes a manifest file by default:

- `<out-dir>/manifest.json` with style, duration, seed, and output path per file

```bash
for i in $(seq 1 40); do
  cargo run -- render --output "out/micro/micro_${i}.wav" --duration 0.18 --style glitch --seed "$i"
done
```

```bash
for s in glitch punish lucky distort; do
  for i in $(seq 1 12); do
    cargo run -- render --output "out/cards/${s}_${i}.wav" --duration 15 --style "$s" --seed "$((1000+i))"
  done
done
```

### Render packs for ranking the ugliest candidates

```bash
cargo run -- render-pack --out-dir out/pack_full --duration 4 --model psycho --top 14 --backend auto --jobs 12
cargo run -- render-pack --out-dir out/pack_violent --styles glitch,punish,distort,spank,steal --duration 20 --model psycho --top 5
cargo run -- render-pack --out-dir out/pack_short --styles pop,glitch,lucky --duration 0.5 --model basic --top 3
```

### Benchmark and choose your backend

```bash
cargo run -- backends
cargo run -- benchmark --runs 8 --duration 1.0 --style punish --jobs 16
cargo run --features metal -- benchmark --runs 8 --duration 1.0 --style punish --jobs 16
cargo run --features cuda -- benchmark --runs 8 --duration 1.0 --style punish --jobs 16
```

### Disk planning for very long files (float32 mono)

- Rough size formula: `seconds * sample_rate * 4 bytes`
- At `192000 Hz`: about `768 KB/s`, about `2.76 GB/hour`
- 8 hours: about `22 GB`
- 24 hours: about `66 GB`

## Render Options

```text
usg render [OPTIONS]
```

- `-o, --output <PATH>`: Output WAV path (default: `ugly.wav`)
- `-d, --duration <SECONDS>`: Duration in seconds (range: `0.1` to `86400`, default: `3.0`)
- `-r, --sample-rate <HZ>`: Sample rate (default: `192000`)
- `--seed <U64>`: Fixed seed for repeatable ugliness
- `--style <...>`: Ugliness profile (default: `harsh`)
- `--gain <0.0..1.0>`: Output gain (default: `0.8`)
- `--normalize-dbfs <DBFS>`: Peak normalize target (default: `0.0`)
- `--no-normalize`: Disable normalization
- `--backend <auto|cpu|metal|cuda>`: Rendering backend (default: `auto`)
- `--jobs <N>`: Worker count (`0` means auto core count)
- `--gpu-drive <F64>`: GPU post-FX drive (overrides default/env)
- `--gpu-crush-bits <F64>`: GPU post-FX bitcrush depth in bits
- `--gpu-crush-mix <0..1>`: GPU post-FX dry/crushed blend

## Go (Input -> Uglier)

`usg go` takes an existing WAV file and pushes it to a target ugliness level from `1` to `1000`.

```text
usg go <INPUT.wav> [OPTIONS]
```

Important options:

- `-o, --output <PATH>`: Output WAV path (default: `<input-stem>.go.wav`)
- `--level <1..1000>`: Target ugliness intensity (default: `700`)
- `--type <glitch|stutter|puff|punish|geek|dissonance-ring|dissonance-expand|random|lucky>`: Optional flavor (`random` when omitted)
- `--upmix <mono|stereo|quad|5.1|7.1|custom:N>`: Optional surround upmix layout
- `--coords <cartesian|polar>`: Coordinate system for spatial arguments
- `--locus <a,b,c>`: Start source locus
- `--seed <U64>`: Deterministic uglification
- `--level-contour <PATH.json>`: Time-varying ugliness contour from JSON
- `--level-contour-json <JSON>`: Time-varying ugliness contour inline JSON
- `--trajectory <static|line:a,b,c|orbit:radius,turns>`: Source movement over file duration
- `--normalize-dbfs <DBFS>` or `--no-normalize`
- `--backend <auto|cpu|metal|cuda>`
- `--jobs <N>`
- `--gpu-drive <F64>`
- `--gpu-crush-bits <F64>`
- `--gpu-crush-mix <0..1>`

Contour JSON schema:

```json
{
  "interpolation": "linear",
  "points": [
    { "t": 0.0, "level": 120 },
    { "t": 0.35, "level": 900 },
    { "t": 1.0, "level": 300 }
  ]
}
```

- `t`: normalized time from `0.0` to `1.0`
- `level`: ugliness target from `1` to `1000`
- `interpolation`: `linear` or `step`

Contour preset library:

- 33 ready presets are included in `presets/go_contours/`
- open [presets/go_contours/README.md](/Users/cleider/dev/UglySoundGenerator/presets/go_contours/README.md) for usage

Examples:

```bash
cargo run -- go out/clean.wav --level 300 --output out/clean_l300.go.wav
cargo run -- go out/clean.wav --level 700 --type glitch --output out/clean_glitch.go.wav
cargo run -- go out/clean.wav --level 900 --type punish --backend auto --jobs 12
cargo run -- go out/clean.wav --level 1000 --type geek --normalize-dbfs -0.2
cargo run -- go out/clean.wav --level 800 --type lucky --seed 1337
cargo run -- go out/clean.wav --level 600 --type random --seed 2026
cargo run -- go out/clean.wav --level 850 --type dissonance-ring --output out/clean_ringed.go.wav
cargo run -- go out/clean.wav --level 780 --type dissonance-expand --output out/clean_expanded.go.wav
cargo run -- go out/clean.wav --level 800 --type glitch --upmix 5.1 --coords cartesian --locus 0,1,0 --trajectory line:1,0,0 --output out/clean_51_move.go.wav
cargo run -- go out/clean.wav --level 900 --type punish --upmix custom:12 --coords polar --locus 30,0,1.0 --trajectory orbit:1.2,4 --output out/clean_12ch_orbit.go.wav
cargo run -- go out/clean.wav --type punish --level-contour-json '{"interpolation":"linear","points":[{"t":0.0,"level":120},{"t":0.35,"level":900},{"t":1.0,"level":300}]}' --output out/clean_contour.go.wav
cargo run -- go out/clean.wav --type glitch --level-contour presets/go_contours/12_step_pattern_01.json --output out/clean_contour_file.go.wav
```

If you are currently in `out/`, the same `go` command looks like this:

```bash
cargo run -- go clean.wav --type punish --level-contour ../presets/go_contours/01_linear_curve_01.json --output clean_preset.go.wav
```

## Chiptune Speech

`usg speech` is the start of the next local subversion focus: text-to-chiptune speech synthesis modeled after classic speech-chip eras.

Current speech-chip-inspired profiles:

- `votrax-sc01`, `tms5220`, `sp0256`, `mea8000`
- `s14001a`, `c64-sam`, `arcadey90s`, `handheld-lcd`

Current speech oscillators:

- `pulse`, `triangle`, `saw`, `noise`
- `buzz`, `formant`, `ring`, `fold`
- `koch`, `mandelbrot`, `strange`, `phoneme`

Text input can be a single letter, a word, a sentence, or a full paragraph:

```bash
cargo run -- speech --output out/letter_a.wav --text "A" --input-mode character --profile votrax-sc01
cargo run -- speech --output out/word_robot.wav --text "ROBOT" --input-mode word --profile tms5220 --primary-osc pulse --secondary-osc formant
cargo run -- speech --output out/sentence.wav --text "HELLO WORLD. THIS CHIP IS VERY CROSS." --input-mode sentence --profile sp0256 --primary-osc phoneme --secondary-osc mandelbrot --tertiary-osc strange
cargo run -- speech --output out/paragraph.wav --text-file speech.txt --input-mode paragraph --profile c64-sam --primary-osc koch --secondary-osc buzz --tertiary-osc ring
```

The command intentionally exposes a lot of shaping parameters already. Highlights:

- text and segmentation: `--text`, `--text-file`, `--input-mode`
- chip voicing: `--profile`, `--pitch-hz`, `--pitch-jitter`, `--vibrato-hz`, `--vibrato-depth`, `--monotone`, `--glide`
- oscillator stack: `--primary-osc`, `--secondary-osc`, `--tertiary-osc`, `--duty-cycle`, `--ring-mix`, `--sub-mix`
- speech color: `--formant-shift`, `--vowel-mix`, `--consonant-noise`, `--nasal`, `--throat`, `--buzz`, `--hiss`
- chip destruction: `--bitcrush-bits`, `--sample-hold-hz`, `--fold`, `--chaos`, `--robotize`, `--resampler-grit`, `--drift`
- timing: `--units-per-second`, `--attack-ms`, `--release-ms`, `--word-gap-ms`, `--sentence-gap-ms`, `--paragraph-gap-ms`, `--punctuation-gap-ms`

Example "maximal nonsense" render:

```bash
cargo run -- speech --output out/maximal_speech.wav --text "UGLY SOUND GENERATOR NOW SPEAKS IN FRACTALS." --profile handheld-lcd --primary-osc phoneme --secondary-osc koch --tertiary-osc strange --pitch-hz 132 --pitch-jitter 0.12 --vibrato-hz 6.5 --vibrato-depth 0.08 --formant-shift 1.2 --consonant-noise 0.9 --buzz 0.45 --fold 3.8 --chaos 0.8 --robotize 0.55 --bitcrush-bits 5.5 --sample-hold-hz 6800 --drift 0.12 --resampler-grit 0.7
```

## Analyze Output

`usg analyze` reports:

- Peak and RMS level
- Crest factor
- Zero crossing rate
- Clipped sample percentage
- Harshness ratio
- Composite ugly index (`0..1000`)

Use `--json` to emit machine-readable output for scripting:

```bash
cargo run -- analyze out/ugly.wav --json
```

Analyze supports two scoring models:

- `--model basic`: Fast time-domain proxy (default)
- `--model psycho`: FFT-based psychoacoustic approximation with component breakdown

For stereo files, psycho analysis uses cross-channel peak interactions for `binaural_beat_norm`; mono files use a conservative close-partial fallback estimator.

The psycho report now includes harmonicity/inharmonicity, binaural-beat pressure, beat-conflict, tritone-tension, and wolf-fifth components in addition to clip/roughness/sharpness/dissonance/transients.

Psycho model options:

- `--fft-size <N>`: STFT window length (default: `2048`)
- `--hop-size <N>`: STFT hop length (default: `512`)

Render styles currently available:

- `harsh`, `digital`, `meltdown`, `catastrophic`
- `glitch`, `pop`, `buzz`, `rub`, `hum`, `distort`
- `spank`, `punish`, `steal`, `wink`, `lucky`

Chain effects currently available:

- `stutter`, `pop`, `crush`, `gate`, `smear`
- `dissonance-ring`: paper-derived spectral ring modulation using Kameoka-Kuriyagawa roughness spacing
- `dissonance-expand`: paper-derived spectral dynamics expansion that exaggerates beating envelopes inside Bark-like bands

Notes for chain stage parsing:

- unprefixed names prefer effects when ambiguous (`pop` resolves to effect)
- use `style:<name>` or `effect:<name>` for explicit control
- example: `--stages style:pop,stutter,effect:pop`

## Chain Pipeline

`usg chain` lets you compose synthesis and effect stages in order.

Example from your workflow:

```bash
cargo run -- chain --stages glitch,stutter,pop --output out/glitch_stutter_pop.wav --play
cargo run -- chain --stages style:hum,dissonance-ring,dissonance-expand --output out/paper_dissonancizer.wav --duration 6
```

Useful options:

- `--stages <A,B,C>`: ordered stage list (required)
- `--output <PATH>`
- `--duration <SECONDS>`
- `--sample-rate <HZ>`
- `--seed <U64>`
- `--gain <0.0..1.0>`
- `--normalize-dbfs <DBFS>`
- `--no-normalize`
- `--backend <auto|cpu|metal|cuda>`
- `--jobs <N>`
- `--gpu-drive <F64>`
- `--gpu-crush-bits <F64>`
- `--gpu-crush-mix <0..1>`
- `--play`: attempt automatic playback after rendering

Chain now streams automatically for long durations, so multi-hour chain exports do not require loading the full signal into memory.

### Paper-derived dissonancizers

Two chain/go processes are now modeled after Hoffman and Cook's DAFx-08 paper on real-time dissonance augmentation:

- `dissonance-ring`: splits the signal into third-octave-ish bands and ring-modulates each band near the Kameoka-Kuriyagawa maximum-roughness spacing.
- `dissonance-expand`: splits the signal into Bark-like bands and exaggerates fast amplitude modulation while re-normalizing slower dynamics.

Examples:

```bash
cargo run -- chain --stages style:hum,dissonance-ring --output out/hum_max_rough.wav --duration 4
cargo run -- chain --stages style:hum,dissonance-ring,dissonance-expand --output out/hum_hyper_rough.wav --duration 4
cargo run -- go out/clean.wav --type dissonance-ring --level 900 --output out/clean_max_rough.go.wav
cargo run -- go out/clean.wav --type dissonance-expand --level 800 --output out/clean_beat_bloated.go.wav
```

## How To Make It Even Uglier

If the default presets are only moderately offensive, the fastest path to true ugliness is layering multiple failure modes instead of just turning up one knob.

Practical escalation tactics:

- Start with `catastrophic` when you want the generator to skip the warmup and get straight to the bad decisions.
- Stack a hostile synth style with several effects: `catastrophic`, `punish`, `steal`, and `buzz` get worse when followed by `stutter`, `crush`, `gate`, and `smear`.
- The nastier styles now include named fractal ingredients: Koch-style quasi-oscillators, Mandelbrot escape-drive oscillators, and strange-attractor drift, so the waveform can spiral into self-similar little disasters instead of repeating one cleanly ugly cycle.
- Re-uglify an already ugly render with `usg go --level 900..1000` so the source material starts damaged before the contour logic makes new mistakes.
- Use step-pattern contours for abrupt jumps instead of smooth ramps; ugly sounds get more irritating when they cannot decide what they are doing.
- Favor high sample rates plus clipping/normalization pressure, because more bandwidth means more room for alias-ish drama, buzz, and transient shrapnel.
- Render a pack, analyze with `--model psycho`, then take the ugliest winner and feed it back through `go` or `chain` again.
- Add spatial motion with `--trajectory orbit:...` or a custom upmix so the ugliness is no longer politely centered in one speaker.

Examples:

```bash
cargo run -- render --output out/pre_ruined.wav --duration 12 --style catastrophic --seed 666 --sample-rate 192000
cargo run -- chain --stages style:steal,stutter,crush,gate,smear,effect:pop --output out/structurally_offensive.wav --duration 20 --seed 4242
cargo run -- go out/structurally_offensive.wav --level 1000 --type punish --level-contour presets/go_contours/22_step_pattern_11.json --output out/absolutely_not.wav
cargo run -- go out/pre_ruined.wav --level 970 --type geek --upmix 5.1 --coords polar --locus 35,0,1.0 --trajectory orbit:1.5,9 --output out/moving_problem.wav
cargo run -- render-pack --out-dir out/pack_dread --styles catastrophic,punish,steal,glitch,distort --duration 8 --model psycho --top 4
cargo run -- analyze out/absolutely_not.wav --model psycho --fft-size 4096 --hop-size 256
```

Suggested escalation ladder:

1. Render something ugly.
2. Chain effects until the waveform looks like it has personal grievances.
3. Run `go` at `--level 1000` with a step contour.
4. Analyze the result and keep whichever file scores closest to “audio war crime.”

## Render Pack

`usg render-pack` creates a full style pack in one shot:

- renders one WAV per style
- analyzes each output with `basic` or `psycho` model
- writes a JSON summary with full entries and a ranked leaderboard

Example:

```bash
cargo run -- render-pack --out-dir out/pack --model psycho --seed 12345 --top 8
cargo run -- render-pack --styles glitch,punish,lucky --out-dir out/pack_focus
```

Useful options:

- `--out-dir <PATH>`: output folder for WAV files
- `--summary <PATH>`: summary JSON path (default: `<out-dir>/summary.json`)
- `--csv <PATH>`: ranking CSV path (default: `<out-dir>/ranking.csv`)
- `--html <PATH>`: HTML listening report path (default: `<out-dir>/report.html`)
- `--duration <SECONDS>`
- `--sample-rate <HZ>`
- `--seed <U64>`: base seed for deterministic per-style seeds
- `--gain <0.0..1.0>`
- `--normalize-dbfs <DBFS>`
- `--no-normalize`
- `--styles <A,B,C>`: optional comma-separated subset of styles
- `--model <basic|psycho>`
- `--fft-size <N>` and `--hop-size <N>` for psycho analysis
- `--top <N>`: number of top ugliest entries printed to terminal
- `--backend <auto|cpu|metal|cuda>`
- `--jobs <N>`: parallel workers for render/analyze tasks
- `--gpu-drive <F64>`
- `--gpu-crush-bits <F64>`
- `--gpu-crush-mix <0..1>`

Example high-throughput pack render:

```bash
cargo run -- render-pack --out-dir out/pack_gpu --backend auto --jobs 24 --model psycho
```

## Backends And Parallelism

Backends are runtime-selectable:

- `cpu`: portable baseline
- `metal`: Apple Metal-targeted backend path (requires `--features metal` on macOS)
- `cuda`: NVIDIA CUDA-targeted backend path (requires `--features cuda`, CUDA driver, and NVRTC runtime)
- `auto`: picks Metal first, then CUDA, then CPU

Backend status:

```bash
cargo run -- backends
```

Build with backend feature flags:

```bash
cargo run --features metal -- render --style glitch --backend metal --jobs 16
cargo run --features cuda -- render-pack --backend cuda --jobs 32
```

`benchmark` command:

```bash
cargo run -- benchmark --runs 6 --duration 0.8 --style punish --jobs 16
```

Useful benchmark flags:

- `--json-output <PATH>`: write a structured benchmark report with per-run timings
- `--csv-output <PATH>`: write a flat ranking CSV for spreadsheets or plotting
- `--sample-format <float|int>` and `--bit-depth <16|24|32>`: benchmark alternate file encodings

- `--runs <N>`: repeated renders per backend
- `--duration <SECONDS>`
- `--sample-rate <HZ>`
- `--style <STYLE>`
- `--jobs <N>`
- `--gpu-drive <F64>`
- `--gpu-crush-bits <F64>`
- `--gpu-crush-mix <0..1>`

`marathon` command:

```bash
cargo run -- marathon --out-dir out/marathon --count 1000 --min-duration 0.1 --max-duration 120 --backend auto --jobs 24
```

Useful marathon flags:

- `--out-dir <PATH>`
- `--count <N>`
- `--min-duration <SECONDS>`
- `--max-duration <SECONDS>`
- `--styles <A,B,C>`
- `--seed <U64>`
- `--sample-rate <HZ>`
- `--gain <0.0..1.0>`
- `--normalize-dbfs <DBFS>` or `--no-normalize`
- `--backend <auto|cpu|metal|cuda>`
- `--jobs <N>`
- `--manifest <PATH>`

Current v-next backend note:

- command/API level backend selection is stable now
- DSP remains 64-bit and deterministic
- backend post-processing executes native device kernels when selected (`metal`/`cuda`) with GPU drive+bitcrush+clip post-FX
- CPU remains the reference path and fallback target

GPU post-FX tuning can be configured by CLI flags or environment variables:

- `USG_GPU_DRIVE`
- `USG_GPU_CRUSH_BITS`
- `USG_GPU_CRUSH_MIX`

Benchmarking:

- `usg benchmark` runs repeatable timed renders for each available backend
- output is average render time per backend, ranked fastest to slowest

## Ugliness Index

`ugly_index` is currently a fast psychoacoustic proxy in the range `0..1000`.

### Current formula (implemented)

Let \(x[n]\) be the mono waveform, \(N\) samples total:

```math
\mathrm{clipped\_pct}
=
\frac{100}{N}\sum_{n=0}^{N-1}\mathbf{1}\{|x[n]| \ge 0.98\}
```

```math
\mathrm{zero\_cross\_rate}
=
\frac{1}{N}\sum_{n=1}^{N-1}\mathbf{1}\{\operatorname{sign}(x[n]) \ne \operatorname{sign}(x[n-1])\}
```

```math
\mathrm{harshness\_ratio}
=
\sqrt{
\frac{\sum_{n=1}^{N-1}\left(x[n]-x[n-1]\right)^2}
{\max\left(\sum_{n=0}^{N-1}x[n]^2,\varepsilon\right)}
}
```

```math
\mathrm{ugly\_index}
=
\operatorname{clamp}\!\left(
10\left(1.6\,\mathrm{clipped\_pct}
+45\,\mathrm{harshness\_ratio}
+200\,\mathrm{zero\_cross\_rate}\right),
0,1000
\right)
```

This exact computation lives in [src/lib.rs](/Users/cleider/dev/UglySoundGenerator/src/lib.rs).

### Why these terms map to perceived ugliness

- `clipped_pct`: hard limiting/clipping creates high-order harmonics and audible crackle.
- `harshness_ratio`: frame-to-frame slope energy is a cheap proxy for broadband, buzzy, non-smooth content.
- `zero_cross_rate`: strongly correlated with high-frequency/noisy content and perceived “edge.”

In psychoacoustic terms, the metric intentionally rewards traits associated with roughness, sharpness, and distortion fatigue.

### Interpretation guide

- `0-200`: mostly tame
- `200-450`: noticeably abrasive
- `450-700`: aggressively ugly
- `700-1000`: catastrophic/noise-weapon territory

## Psychoacoustic Math Roadmap

`usg` now includes an initial psycho model (`--model psycho`) that approximates this framework using STFT-derived features.

The model computes these normalized components:

- `clip_norm`
- `harshness_norm`
- `roughness_norm`
- `sharpness_norm`
- `dissonance_norm`
- `transient_norm`
- `harmonicity_norm`
- `inharmonicity_norm`
- `binaural_beat_norm`
- `beat_conflict_norm`
- `tritone_tension_norm`
- `wolf_fifth_norm`

Then fuses them:

```math
\begin{aligned}
\mathrm{weighted\_sum} =\;& -4.05
+ 1.6\,\mathrm{clip\_norm}
+ 1.3\,\mathrm{roughness\_norm}
+ 1.0\,\mathrm{sharpness\_norm}
+ 1.0\,\mathrm{dissonance\_norm} \\
&+ 1.2\,\mathrm{transient\_norm}
+ 0.9\,\mathrm{harshness\_norm}
+ 1.25\,\mathrm{inharmonicity\_norm}
+ 0.85\,\mathrm{binaural\_beat\_norm} \\
&+ 1.05\,\mathrm{beat\_conflict\_norm}
+ 0.85\,\mathrm{tritone\_tension\_norm}
+ 0.75\,\mathrm{wolf\_fifth\_norm}
- 0.45\,\mathrm{harmonicity\_norm}
\end{aligned}
```

```math
\mathrm{ugly\_index\_psycho} = 1000\,\sigma(\mathrm{weighted\_sum})
```

The deeper research target remains critical-band analysis:

1. Time-frequency analysis:
   ```math
   X(k,m)=\operatorname{STFT}\{x[n]\}
   ```
2. Bark-band energy:
   ```math
   E_b(m)=\sum_k |X(k,m)|^2 H_b(k)
   ```
3. Specific loudness:
   ```math
   N'(z,m)=f(E_z(m)),\quad N(m)=\int N'(z,m)\,dz
   ```
4. Sharpness (high-frequency weighted loudness):
   ```math
   S(m)=\frac{\int z\,g(z)\,N'(z,m)\,dz}{\int N'(z,m)\,dz}
   ```
5. Roughness (modulation around ~30-150 Hz in each band):
   ```math
   R(m)=\sum_b w_b M_b(m)\exp\!\left(-\left(\frac{f_{\mathrm{mod},b}-70}{\sigma}\right)^2\right)
   ```
6. Tonal dissonance (pairwise partial beating):
   ```math
   D(m)=\sum_i\sum_j a_i a_j\left(e^{-\alpha\Delta f_{ij}}-e^{-\beta\Delta f_{ij}}\right)
   ```

Then a calibrated composite:

```math
U = 1000\,\sigma\!\left(
w_c\,\mathrm{clip\_norm}
+ w_r\,\mathrm{rough\_norm}
+ w_s\,\mathrm{sharp\_norm}
+ w_d\,\mathrm{dissonance\_norm}
+ w_t\,\mathrm{transient\_norm}
\right)
```

where each term is normalized against a reference corpus, and weights are fitted from listening tests.

## Joke Math: The UglierBasis Equation

This section is intentionally not the real model. It is a ceremonial overreaction for anyone who feels a normal ugliness score is insufficiently dramatic and also insufficiently covered in summation symbols.

```math
\begin{aligned}
\mathfrak{U}_{\mathrm{UglierBasis}}(x)=1000\,\sigma\!\Bigg(
&\frac{
\displaystyle\sum_{m=1}^{M}\sum_{b=1}^{B}\sum_{q=1}^{Q}
\left[
\frac{
\left(\alpha_{1} C^{q}+\alpha_{2} R_{b}^{q}+\alpha_{3} S_{b}^{q}+\alpha_{4} D_{b}^{q}+\alpha_{5} T_{m}^{q}\right)
\left(1+\alpha_{6} I_{b}+\alpha_{7} B_{b}+\alpha_{8} F_{b}+\alpha_{9} W_{b}\right)
}{
1+\alpha_{10}H_{b}+\alpha_{11}L_{m}+\alpha_{12}\left|X_{m}-O_{b}\right|
}
\right]
}{
\displaystyle
1+\sum_{m=1}^{M}\frac{1}{1+G_{m}+P_{m}+J_{m}}
}
\\[6pt]
&+
\frac{
\displaystyle\sum_{m=1}^{M}
\frac{
\sum_{b=1}^{B}(A_{b}+Z_{b}+Q_{b})^2
}{
1+\sum_{r=1}^{3}\frac{L_{m}^{\,r}}{1+r+N_{m}}
}
}{
\displaystyle
1+\sum_{m=1}^{M}\sum_{b=1}^{B}\frac{H_{b}}{1+E_{m}+Y_{m}}
}
\\[6pt]
&+
\frac{
\displaystyle\prod_{b=1}^{B}
\left(
1+\frac{\beta_{1} M_{b}+\beta_{2} V_{b}+\beta_{3} K_{b}}{1+\beta_{4} H_{b}}
\right)^{\frac{1}{B}}
-1
}{
\displaystyle
1+\prod_{m=1}^{M}\left(1+\frac{1}{1+G_{m}^{2}}\right)^{\frac{1}{M}}
}
\\[6pt]
&+
\sum_{m=1}^{M}
\frac{
\displaystyle\sum_{b=1}^{B}
\frac{
\gamma_{1} \sin\!\big(\omega_{1}(B_{b}+F_{b}+Y_{m})\big)
+\gamma_{2} \cos\!\big(\omega_{2}(A_{b}+Z_{b}+Q_{b})\big)
}{
1+\frac{H_{b}}{1+I_{b}}+\frac{L_{m}}{1+T_{m}}
}
}{
\displaystyle
1+\sum_{b=1}^{B}\frac{1}{1+R_{b}S_{b}D_{b}}
}
\\[6pt]
&+
\frac{
\displaystyle\sum_{m=1}^{M}\sum_{b=1}^{B}
\frac{
\left(\delta_{1} O_{b}+\delta_{2} A_{b}+\delta_{3} E_{m}+\delta_{4} N_{m}+\delta_{5} Y_{m}\right)
\left(\delta_{6} X_{m}+\delta_{7} J_{m}+\delta_{8} V_{b}+\delta_{9} K_{b}\right)
}{
1+\frac{H_{b}^{2}}{1+I_{b}}+\frac{T_{m}^{2}}{1+L_{m}}
}
}{
\displaystyle
1+\sum_{m=1}^{M}\sum_{b=1}^{B}
\frac{1}{1+\left(C+R_{b}+S_{b}+D_{b}\right)^2}
}
\\[6pt]
&+
\frac{
\displaystyle\sum_{u=1}^{U}
\prod_{v=1}^{V}
\left(
1+
\frac{
\kappa_{u v}^{(1)}C+\kappa_{u v}^{(2)}R+\kappa_{u v}^{(3)}S+\kappa_{u v}^{(4)}D+\kappa_{u v}^{(5)}A+\kappa_{u v}^{(6)}Z+\kappa_{u v}^{(7)}Q
}{
1+\kappa_{u v}^{(8)}H+\kappa_{u v}^{(9)}L+\kappa_{u v}^{(10)}T
}
\right)
}{
\displaystyle
1+\sum_{u=1}^{U}\sum_{v=1}^{V}\frac{1}{1+\kappa_{u v}^{(11)}M+\kappa_{u v}^{(12)}P+\kappa_{u v}^{(13)}J}
}
\\[6pt]
&-\lambda\,
\frac{
\displaystyle\sum_{b=1}^{B}H_{b}
}{
\displaystyle
1+\sum_{b=1}^{B}\frac{1}{1+I_{b}+W_{b}}
}
\Bigg)
\end{aligned}
```

### Radical explanation of the nonsense

The fake theorem is simple in spirit: a sound becomes truly ugly when it is clipped, rough, bright, unstable, dissonant, badly behaved in time, and somehow also smug about it.

Variable roster:

- `C`: clip arrogance, or how confidently the waveform slams into the rails.
- `R`: roughness, the sandpaper component.
- `S`: sharpness, the “why is this so pointy?” factor.
- `D`: dissonance, the intervallic argument.
- `T`: transient density, the number of tiny jump-scares per second.
- `H`: harmonicity, included mainly so the equation can punish anything too musically coherent.
- `I`: inharmonicity, where partials stop cooperating.
- `B`: binaural beat pressure, the left-right disagreement tax.
- `F`: beat conflict, because one pulse of annoyance is never enough.
- `W`: wolf-fifth tension, for the medieval “something is wrong here” sensation.
- `M`: modulation glare, describing wobble that should have stayed in the lab.
- `G`: gate surprise, the abruptness with which sound appears to be switched by an unreliable electrician.
- `P`: pop density, or crackle-per-second.
- `Q`: quantization shame, the moral cost of brutal bit depth reduction.
- `Z`: zipper noise, the sonic equivalent of a broken jacket.
- `L`: loudness lurch, measuring how much the level behaves like a staircase in an earthquake.
- `O`: overtone hostility, when upper partials start making eye contact.
- `A`: alias spray, the confetti cannon of spectral regret.
- `E`: envelope panic, when attack and decay both overreact.
- `N`: notch cruelty, the weirdly mean filtering term.
- `Y`: hysteresis squeal, a totally serious quantity that definitely exists because I wrote it down.
- `X`: stereo argument, tracking how hard the channels disagree about reality.
- `J`: jitter, for timing uncertainty with commitment issues.
- `V`: vibrato malpractice, when pitch motion becomes a legal matter.
- `K`: cadence collapse, which penalizes any ending that fails to end with dignity.

How the monster works:

- The first fraction triple-sums over time windows, Bark bands, and whatever `q` is supposed to mean this week, so the equation immediately establishes that nobody is leaving the building quickly.
- The second fraction exists to punish the cursed alliance of aliasing, zipper noise, and quantization shame using another fraction nested inside a fraction, which is the traditional sign of scientific confidence.
- The product term says ugliness compounds multiplicatively once modulation glare, vibrato malpractice, and cadence collapse begin cooperating in bad faith.
- The trigonometric block makes the ugliness oscillate, because static unpleasantness is amateur work; true ugliness should precess.
- The fourth fraction cross-couples overtone hostility, envelope panic, stereo argument, jitter, and notch cruelty so every subsystem can blame every other subsystem.
- The `\sum\prod` block is the ceremonial tensor bureaucracy layer. It adds absolutely no interpretability and therefore feels academically authentic.
- The final negative harmonicity term is the tiny civilized voice in the room: musical coherence still reduces the score a little, but it now has to survive six separate bureaucratic tribunals first.

Practical interpretation:

- If `C`, `R`, `S`, `D`, `A`, `Z`, and `Q` are all high, the sound is not just ugly; it is academically ugly.
- If `H` is high, the equation sighs, lowers the score a little, and then raises it again somewhere else out of spite.
- If every term is high at once, the sigmoid saturates near `1000`, which is the mathematically rigorous symbol for “please turn that off.”

## Appendix B

### B.1 Scope Of This Reading List

This project sits at the intersection of psychoacoustics, auditory preference, musical dissonance, and philosophical aesthetics. Direct literature using the exact phrase `auditory ugliness` is relatively sparse; where that phrase is absent, the references below are included because they bear directly on the perceptual ingredients most often associated with sonic ugliness in practice: roughness, beating, critical-band interactions, sharpness, aversive timbre, sensory dissonance, musical instability, and explicitly theorized ugliness in music.

### B.2 Foundational Literature (Psychoacoustics And Sensory Dissonance)

This section now deliberately mixes two adjacent literatures: first, the psychoacoustic work that explains why some sounds are perceived as rough, unstable, shrill, beating, or aversive; second, the smaller but more direct literature that explicitly talks about ugliness, ugly sound, harsh noise, and ugliness in music. The first group is the stronger empirical foundation. The second group is the more explicit vocabulary layer.

Core psychoacoustics and sensory dissonance:

- Helmholtz, H. (1863/1954). *On the Sensations of Tone*. The long historical starting point for modern discussions of beating, consonance, dissonance, and the physiological basis of musical tone relations. [Google Books](https://books.google.com/books/about/On_the_Sensations_of_Tone.html?id=VM5gNckZj5YC)
- Plomp, R., & Levelt, W. J. M. (1965). *Tonal consonance and critical bandwidth*. Journal of the Acoustical Society of America, 38(4), 548-560. This is still one of the central empirical anchors for critical-band accounts of sensory dissonance. [PubMed](https://pubmed.ncbi.nlm.nih.gov/5831012/) | [MPI summary](https://www.mpi.nl/publications/item66382/tonal-consonance-and-critical-bandwidth)
- Kameoka, A., & Kuriyagawa, M. (1969). *Consonance theory part I: consonance of dyads*. Journal of the Acoustical Society of America, 45(6), 1451-1459. A classic attempt to formalize sensory consonance for two-tone combinations. [PubMed](https://pubmed.ncbi.nlm.nih.gov/5803168/)
- Kameoka, A., & Kuriyagawa, M. (1969). *Consonance theory part II: consonance of complex tones and its calculation method*. Journal of the Acoustical Society of America, 45(6), 1460-1469. Extends the dyad model toward complex-tone dissonance calculation. [PubMed](https://pubmed.ncbi.nlm.nih.gov/5803169/)
- Terhardt, E. (1974). *Pitch, consonance, and harmony*. Journal of the Acoustical Society of America, 55(5), 1061-1069. Important for pitch-based and harmonic-template perspectives on consonance. [PubMed](https://pubmed.ncbi.nlm.nih.gov/4833699/)
- Parncutt, R. (1989). *Harmony: A Psychoacoustical Approach*. A major book-length account linking harmony, roughness, and psychoacoustic constraint. [Springer](https://link.springer.com/book/10.1007/978-3-642-74831-8)
- Zwicker, E., & Fastl, H. (2007 ed.). *Psychoacoustics: Facts and Models*. Still one of the standard references for loudness, roughness, sharpness, fluctuation strength, and practical psychoacoustic modeling. [Springer](https://link.springer.com/book/10.1007/978-3-540-68888-4)
- Sethares, W. A. (1993). *Local Consonance and the Relationship Between Timbre and Scale*. Foundational for timbre-sensitive dissonance curves and for moving beyond “good interval / bad interval” models that ignore spectrum. [Author page](https://sethares.engr.wisc.edu/papers/consance.html)

Reviews, syntheses, and modern empirical follow-ups:

- Harrison, P. M. C., & Pearce, M. T. (2020). *Simultaneous Consonance in Music Perception and Composition*. Psychological Review, 127(2), 216-244. A large review/comparison of computational consonance models and explanatory families. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7032667/)
- Di Stefano, N., & Spence, C. (2022). *Roughness perception: A multisensory/crossmodal perspective*. Especially useful here because it explicitly treats auditory roughness as relevant both to dissonance and to aversive vocal timbres such as screams. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9481510/)
- Eerola, T., & Lahdelma, I. (2022). *Register impacts perceptual consonance through roughness and sharpness*. Helpful for any ugliness model that treats low-end mud and top-end shrillness as distinct but interacting penalties. [PubMed](https://pubmed.ncbi.nlm.nih.gov/34921342/) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9166839/)
- Popescu, T., Neuser, M. P., Neuwirth, M., Bravo, F., Mende, W., Boneh, O., Moss, F. C., & Rohrmeier, M. (2019). *The pleasantness of sensory dissonance is mediated by musical style and expertise*. A useful reminder that dissonance is not simply “bad”; style, training, and enculturation modulate how roughness and instability are valued. [PubMed](https://pubmed.ncbi.nlm.nih.gov/30705379/) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6355932/)
- Milne, A. J., Smit, E. A., Sarvasy, H. S., & Dean, R. T. (2023). *Evidence for a universal association of auditory roughness with musical stability*. Useful empirical support for roughness as a cross-cultural predictor of unstable / unfinished musical affect. [PubMed](https://pubmed.ncbi.nlm.nih.gov/37729156/)
- Marjieh, R., Harrison, P. M. C., Lee, H., Deligiannaki, F., et al. (2024). *Timbral effects on consonance disentangle psychoacoustic mechanisms and suggest perceptual origins for musical scales*. Important newer evidence that timbre is not a side issue but part of the consonance mechanism itself. [Nature Communications](https://www.nature.com/articles/s41467-024-45812-z)
- Fishman, Y. I., Volkov, I. O., Noh, M. D., Garell, P. C., Arezzo, J. C., Howard, M. A., & Steinschneider, M. (2001). *Consonance and dissonance of musical chords: neural correlates in auditory cortex of monkeys and humans*. Useful if Appendix B wants a neurophysiological bridge from acoustic roughness to cortical response. [PubMed](https://pubmed.ncbi.nlm.nih.gov/11731536/)
- Trulla, L. L., Di Stefano, N., & Giuliani, A. (2018). *Computational Approach to Musical Consonance and Dissonance*. A modern computational framing of consonance/dissonance with explicit methodological discussion. [Frontiers](https://www.frontiersin.org/articles/10.3389/fpsyg.2018.00381/full) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5893895/)
- Bowling, D. L. (2021). *Harmonicity and Roughness in the Biology of Tonal Aesthetics*. Useful for the ongoing harmonicity-versus-roughness debate rather than treating “ugliness” as reducible to one scalar measure. [PubMed](https://pubmed.ncbi.nlm.nih.gov/34566250/) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8460127/)
- Hoffman, M., & Cook, P. (2008). *Real-Time Dissonancizers: Two Dissonance-Augmenting Audio Effects*. Directly relevant to this repo's `dissonance-ring` and `dissonance-expand` DSP because it turns sensory-dissonance theory into concrete audio processing. [Princeton PDF](https://soundlab.cs.princeton.edu/publications/diss_dafx2008.pdf) | [SciSpace mirror](https://scispace.com/pdf/real-time-dissonancizers-two-dissonance-augmenting-audio-2c9dtmn1k0.pdf)
- Leider, C. (2007). *Dissonance Theory of Sound Objects*. PhD dissertation, Princeton University. Included here in the main B.2 list because it extends dissonance thinking from interval/chord judgments toward sound objects and process-oriented sonic organization, which is especially relevant to this project. [Referenced in DAFx-08 bibliography](https://soundlab.cs.princeton.edu/publications/diss_dafx2008.pdf) | [University of Miami note on degree/dissertation](https://news.miami.edu/frost/_assets/pdf/the-score/2007SCORE.pdf)

Roughness, aversion, harsh timbre, and unpleasant sound:

- Pressnitzer, D., & McAdams, S. (1999). *Two phase effects in roughness perception*. Important if the appendix wants something more mechanistic than broad roughness summaries; it shows that perceived roughness depends on temporal structure, not just gross spectrum. [PubMed](https://pubmed.ncbi.nlm.nih.gov/10335629/)
- Pressnitzer, D., McAdams, S., Winsberg, S., & Fineberg, J. (2000). *Perception of musical tension for nontonal orchestral timbres and its relation to psychoacoustic roughness*. Especially relevant if “ugliness” is being connected to modernist or non-tonal tension rather than only interval roughness in isolated dyads. [PubMed](https://pubmed.ncbi.nlm.nih.gov/10703256/)
- De Baene, W., Vandierendonck, A., Leman, M., Widmann, A., & Tervaniemi, M. (2004). *Roughness perception in sounds: behavioral and ERP evidence*. Useful for linking perceived roughness to electrophysiological response. [PubMed](https://pubmed.ncbi.nlm.nih.gov/15294389/)
- Arnal, L. H., Flinker, A., Kleinschmidt, A., Giraud, A.-L., & Poeppel, D. (2015). *Human screams occupy a privileged niche in the communication soundscape*. Important because it ties roughness directly to alarm, fear, salience, and biologically potent unpleasantness. [PubMed](https://pubmed.ncbi.nlm.nih.gov/26190070/) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4562283/)
- Eddins, D. A., Kopf, L. M., & Shrivastav, R. (2015). *The psychophysics of roughness applied to dysphonic voice*. Useful if the repo’s “ugly” sound design wants a speech/voice-quality bridge rather than only a music-theory bridge. [PubMed](https://pubmed.ncbi.nlm.nih.gov/26723336/) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4691258/)
- Taffou, M., Suied, C., & Viaud-Delmon, I. (2021). *Auditory roughness elicits defense reactions*. Direct evidence that rough sounds do not just sound “bad” but also alter defensive spatial behavior. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7806762/)
- Ringer, H., Rösch, S. A., Roeber, U., Deller, J., Escera, C., & Grimm, S. (2024). *That sounds awful! Does sound unpleasantness modulate the mismatch negativity and its habituation?* Useful for the broader “awful sound” literature beyond specifically musical dissonance. [PubMed](https://pubmed.ncbi.nlm.nih.gov/37779371/)

Auditory preference, aversion, and sound quality:

- McDermott, J. H. (2012). *Auditory Preferences and Aesthetics: Music, Voices, and Everyday Sounds*. This is especially relevant if “auditory ugliness” is being treated as a broader family of aversive listening responses, not just musical dissonance. [ScienceDirect chapter page](https://www.sciencedirect.com/science/article/abs/pii/B9780123814319000206) | [MIT abstract/reprint page](https://mcdermottlab.mit.edu/bib2php/pubs/makeAbs.php?loc=mcdermott11a)
- Alluri, V., Toiviainen, P., Jääskeläinen, I. P., Glerean, E., Sams, M., & Brattico, E. (2017). *Global Sensory Qualities and Aesthetic Experience in Music*. Relevant as a bridge from low-level auditory descriptors to high-level aesthetic judgment. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5380758/)
- Heller, L. M., & Smith, J. M. (2022). *Identification of Everyday Sounds Affects Their Pleasantness*. Helpful if this reading list wants to acknowledge that perceived ugliness is not only in the waveform; source recognition changes pleasantness too. [PubMed](https://pubmed.ncbi.nlm.nih.gov/35936236/) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9347306/)

Ugliness, aesthetic negativity, and ugliness in music:

- Henderson, G. E. (2015). *Ugliness: A Cultural History*. Broad rather than psychoacoustic, but useful because it explicitly situates ugliness across media, including music and the senses. [Google Books](https://books.google.com/books/about/Ugliness.html?id=FpG-CwAAQBAJ) | [Smithsonian catalog](https://www.si.edu/object/siris_sil_1058572)
- Rosenkranz, K. (1853/2015 critical edition). *Aesthetics of Ugliness*. Historically indispensable as a general ugliness framework, even though it is not specifically auditory. [Bloomsbury](https://www.bloomsbury.com/us/aesthetics-of-ugliness-9781350022928/) | [PhilPapers](https://philpapers.org/rec/ROSAOU-2)
- Henderson, G. P. (1966). *The Concept of Ugliness*. A classic philosophical treatment of ugliness as an aesthetic category. [Oxford Academic](https://academic.oup.com/bjaesthetics/article/6/3/219/48167)
- Pickford, R. W. (1969). *The Psychology of Ugliness*. Useful if this appendix wants historical precedent for treating ugliness as a perceptual-psychological problem rather than only a moral or stylistic one. [Oxford Academic](https://academic.oup.com/bjaesthetics/article/9/3/258/11416)
- Carmichael, P. A. (1972). *The Sense of Ugliness*. Another compact philosophical reference worth citing when arguing that ugliness is its own mode of aesthetic experience rather than merely “failed beauty.” [Oxford Academic](https://academic.oup.com/jaac/article/30/4/495/6337054)
- Gracyk, T. A. (1986). *Sublimity, Ugliness, and Formlessness in Kant's Aesthetic Theory*. Helpful if the repo wants to place ugly sound somewhere between formal breakdown and the sublime. [Oxford Academic](https://academic.oup.com/jaac/article/45/1/49/6339900)
- Thomson, G. (1992). *Kant's Problems With Ugliness*. Useful for the negative-aesthetics background that later music-ugliness arguments often inherit. [Oxford Academic](https://academic.oup.com/jaac/article/50/2/107/6341386)
- McConnell, S. (2008). *How Kant Might Explain Ugliness*. Helpful for negative aesthetic judgment as a positive category rather than merely “failed beauty.” [Oxford Academic](https://academic.oup.com/bjaesthetics/article/48/2/205/24266)
- Cohen, A. (2013). *Kant on the Possibility of Ugliness*. Relevant if the theoretical frame distinguishes aesthetic negativity from mere displeasure. [Oxford Academic](https://academic.oup.com/bjaesthetics/article/53/2/199/28454)
- Coate, M. (2018). *Nothing but Nonsense: A Kantian Account of Ugliness*. A contemporary contribution to ugliness as a distinct judgment category. [Oxford Academic](https://academic.oup.com/bjaesthetics/article-abstract/58/1/51/4670995)
- Paris, P. (2017). *The Deformity-Related Conception of Ugliness*. Not musical, but useful for thinking through deformation, disfiguration, and negative form as aesthetic concepts. [Oxford Academic PDF page](https://academic.oup.com/bjaesthetics/article-pdf/57/2/139/19298202/ayw090.pdf)
- Park, Byung-Kyu. (2019). *A Study of Auditory Ugliness in David Lynch's Films*. One of the most directly on-topic references because it explicitly uses the phrase `auditory ugliness` and analyzes sound design through the aesthetics of ugliness. [Earticle](https://www.earticle.net/Article/A356695)
- Garratt, J. (2020). *Beyond Beauty: The Aesthetics of Ugliness in German Musical and Artistic Debates of the Mid-19th Century*. Directly relevant to the history of ugliness in musical discourse rather than ugliness in the abstract. [University of Manchester Research Explorer](https://research.manchester.ac.uk/en/publications/beyond-beauty-the-aesthetics-of-ugliness-in-german-musical-and-ar/)
- Fan, J. (2024). *The Embodiment of Ugliness in 20th Century Western Music: A Case Study of Schoenberg's "Prelude"*. Useful as a recent music-specific attempt to treat ugliness as compositional and historical method. [Article page](https://www.ewadirect.com/journal/ahr/article/view/12535)
- Radovanovic Suput, B. (2025). *Aesthetics of Ugliness – Sound and Image of Extreme Metal Music*. This is one of the most directly relevant music-specific ugliness items I found; it explicitly addresses ugliness as a sonic and visual category in extreme metal. [Article page](https://hrcak.srce.hr/en/clanak/491657)
- Guesde, C., & Nadrigny, P. (2018). *The Most Beautiful Ugly Sound in the World: À l'écoute de la noise*. Particularly relevant for harsh-noise aesthetics, ugly sound as a positive listening practice, and the education of ears toward discomfort. [Publisher page](https://www.editions-mf.com/produit/61/9782378040352/the-most-beautiful-ugly-sound-in-the-world)
- McNeilly, K. (1995). *Ugly Beauty: John Zorn and the Politics of Postmodern Music*. Included because it addresses “ugly” as an affirmative musical-aesthetic category rather than merely a failure mode. [Postmodern Culture](https://www.pomoculture.org/2013/09/24/ugly-beauty-john-zorn-and-the-politics-of-postmodern-music/)
- Orestig, J. (2019). *“Sluta spela fin och gå loss”: Original Dunder Zubbis som lallande polyfoni*. Included here as a music ugliness reference in the broader sociocultural/aesthetic sense, rather than as a psychoacoustic study. [Article page](https://publicera.kb.se/tfl/article/view/7285)
- Unger, M. P. (2016). *Sound, Symbol, Sociality: The Aesthetic Experience of Extreme Metal Music*. Not a direct “ugliness” title, but highly adjacent for sound, defilement, harshness, and the aesthetics of extreme musical experience. [Springer](https://link.springer.com/book/10.1057/978-1-137-47835-1)

Colby Leider and adjacent project-specific references:

- Leider, C. (2007). *Dissonance Theory of Sound Objects*. PhD dissertation, Princeton University. This is the most directly project-adjacent reference for the present repo because it connects dissonance theory to sound objects rather than only interval/chord judgment. [Referenced in DAFx-08 bibliography](https://soundlab.cs.princeton.edu/publications/diss_dafx2008.pdf) | [University of Miami note on degree/dissertation](https://news.miami.edu/frost/_assets/pdf/the-score/2007SCORE.pdf)
- Leider’s current biography and selected-writings page is also useful as a lightweight attribution source tying the dissertation title to Colby Leider’s broader research identity in DSP, sound synthesis, and tuning systems. [Colby Leider Acoustics](https://leider.org/about)

Working synthesis for this repo:

- The strongest psychoacoustic basis for “ugliness” remains roughness / beating / critical-band interaction, especially in the Plomp-Levelt, Kameoka-Kuriyagawa, Terhardt, Parncutt, Zwicker-Fastl, Sethares, Harrison-Pearce, Eerola-Lahdelma, and Popescu et al. line of work.
- The stronger claim that there is a fully established, standalone empirical literature specifically named `auditory ugliness` would be an overstatement.
- The broader ugliness-in-music claim is therefore an inference built by combining sensory-dissonance research, auditory preference/aversion literature, and philosophical / musicological ugliness literature.

## Why Rust

This project is implemented entirely in Rust, including CLI parsing, synthesis, WAV I/O, and analysis.
