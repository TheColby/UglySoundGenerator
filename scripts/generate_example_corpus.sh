#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="$ROOT/target/debug/usg"
AUDIO_DIR="$ROOT/examples/audio"
README="$ROOT/README_EXAMPLES.md"
EXPECTED_COUNT=333

mkdir -p "$AUDIO_DIR"
cargo build --quiet --manifest-path "$ROOT/Cargo.toml"

count=0
next_index=12

start_readme() {
  cat > "$README" <<'DOC'
# USG Example Corpus

This corpus contains 333 reproducible Git-LFS-tracked WAV files.
Regenerate it from the repository root with `./scripts/generate_example_corpus.sh`.
Unless a row says otherwise, renders default to 32-bit float at 192 kHz and normalize to 0 dBFS.

## Seed Set

| File | Command | Notes |
| --- | --- | --- |
DOC
}

start_section() {
  local title="$1"
  printf '\n## %s\n\n| File | Command | Notes |\n| --- | --- | --- |\n' "$title" >> "$README"
}

record_row() {
  local file="$1"
  local cmd_text="$2"
  local notes="$3"
  printf '| `%s` | `%s` | %s |\n' "$file" "$cmd_text" "$notes" >> "$README"
  count=$((count + 1))
}

run_and_record() {
  local file="$1"
  local cmd_text="$2"
  local notes="$3"
  shift 3
  "$@"
  record_row "$file" "$cmd_text" "$notes"
}

start_readme

run_and_record \
  "examples/audio/00_source_hum.wav" \
  "./target/debug/usg render --output examples/audio/00_source_hum.wav --duration 0.2 --style hum --sample-rate 44100" \
  "Clean-ish mono source for the go and upmix examples." \
  "$BIN" render --output examples/audio/00_source_hum.wav --duration 0.2 --style hum --sample-rate 44100

run_and_record \
  "examples/audio/01_harsh.wav" \
  "./target/debug/usg render --output examples/audio/01_harsh.wav --duration 0.2 --style harsh" \
  "Default float32/192 kHz harsh render." \
  "$BIN" render --output examples/audio/01_harsh.wav --duration 0.2 --style harsh

run_and_record \
  "examples/audio/02_punish.wav" \
  "./target/debug/usg render --output examples/audio/02_punish.wav --duration 0.2 --style punish" \
  "Dense clipped punishment texture." \
  "$BIN" render --output examples/audio/02_punish.wav --duration 0.2 --style punish

run_and_record \
  "examples/audio/03_int24_punish.wav" \
  "./target/debug/usg render --output examples/audio/03_int24_punish.wav --duration 0.2 --style punish --sample-format int --bit-depth 24" \
  "Integer output encoding example." \
  "$BIN" render --output examples/audio/03_int24_punish.wav --duration 0.2 --style punish --sample-format int --bit-depth 24

run_and_record \
  "examples/audio/04_ps1_grit.wav" \
  "./target/debug/usg chain --preset ps1_grit --duration 0.2 --output examples/audio/04_ps1_grit.wav" \
  "Built-in chain preset example." \
  "$BIN" chain --preset ps1_grit --duration 0.2 --output examples/audio/04_ps1_grit.wav

run_and_record \
  "examples/audio/05_arcade_overheat.wav" \
  "./target/debug/usg chain --preset arcade_overheat --duration 0.2 --output examples/audio/05_arcade_overheat.wav" \
  "Hotter chain preset example." \
  "$BIN" chain --preset arcade_overheat --duration 0.2 --output examples/audio/05_arcade_overheat.wav

run_and_record \
  "examples/audio/06_go_punish.wav" \
  "./target/debug/usg go examples/audio/00_source_hum.wav --type punish --output examples/audio/06_go_punish.wav" \
  "go defaults to 192 kHz float32." \
  "$BIN" go examples/audio/00_source_hum.wav --type punish --output examples/audio/06_go_punish.wav

run_and_record \
  "examples/audio/07_go_glitch_contour.wav" \
  "./target/debug/usg go examples/audio/00_source_hum.wav --type glitch --level-contour presets/go_contours/12_step_pattern_01.json --output examples/audio/07_go_glitch_contour.wav" \
  "Versioned contour preset example." \
  "$BIN" go examples/audio/00_source_hum.wav --type glitch --level-contour presets/go_contours/12_step_pattern_01.json --output examples/audio/07_go_glitch_contour.wav

run_and_record \
  "examples/audio/08_go_51.wav" \
  "./target/debug/usg go examples/audio/00_source_hum.wav --type punish --upmix 5.1 --output examples/audio/08_go_51.wav" \
  "5.1 upmix example at 192 kHz." \
  "$BIN" go examples/audio/00_source_hum.wav --type punish --upmix 5.1 --output examples/audio/08_go_51.wav

run_and_record \
  "examples/audio/09_rom_corruption.wav" \
  "./target/debug/usg chain --preset rom_corruption --duration 0.2 --output examples/audio/09_rom_corruption.wav" \
  "Unstable chain recipe." \
  "$BIN" chain --preset rom_corruption --duration 0.2 --output examples/audio/09_rom_corruption.wav

run_and_record \
  "examples/audio/10_binaural_nuisance.wav" \
  "./target/debug/usg chain --preset binaural_nuisance --duration 0.2 --output examples/audio/10_binaural_nuisance.wav" \
  "Spatial discomfort preset chain." \
  "$BIN" chain --preset binaural_nuisance --duration 0.2 --output examples/audio/10_binaural_nuisance.wav

USG_STREAM_THRESHOLD_FRAMES=64 "$BIN" render --output examples/audio/11_streamed_glitch.wav --duration 0.2 --style glitch --sample-rate 44100
record_row \
  "examples/audio/11_streamed_glitch.wav" \
  "USG_STREAM_THRESHOLD_FRAMES=64 ./target/debug/usg render --output examples/audio/11_streamed_glitch.wav --duration 0.2 --style glitch --sample-rate 44100" \
  "Forces the streaming render path for regression coverage."

styles=(harsh digital meltdown glitch pop buzz rub hum distort spank punish steal catastrophic wink lucky)
render_durations=(0.12 0.16 0.18 0.14 0.20 0.22 0.15 0.19)
render_gains=(0.78 0.86 0.92 0.70 0.84 0.96 0.74 0.88)
render_rates=(192000 192000 96000 48000 192000 192000 88200 192000)
render_formats=(float int float int float int float float)
render_bits=(32 24 32 16 32 32 32 32)
render_norms=(0.0 -0.1 -0.3 -0.5 0.0 -0.2 -0.4 0.0)

start_section "Render Grid"
for style in "${styles[@]}"; do
  for idx in "${!render_durations[@]}"; do
    file="examples/audio/$(printf '%03d' "$next_index")_render_${style}_v$(printf '%02d' $((idx + 1))).wav"
    duration="${render_durations[$idx]}"
    gain="${render_gains[$idx]}"
    rate="${render_rates[$idx]}"
    fmt="${render_formats[$idx]}"
    bits="${render_bits[$idx]}"
    norm="${render_norms[$idx]}"
    seed=$((1000 + next_index * 37 + idx))
    cmd_text="./target/debug/usg render --output ${file} --duration ${duration} --style ${style} --seed ${seed} --gain ${gain} --sample-rate ${rate} --sample-format ${fmt} --bit-depth ${bits} --normalize-dbfs=${norm}"
    "$BIN" render --output "$file" --duration "$duration" --style "$style" --seed "$seed" --gain "$gain" --sample-rate "$rate" --sample-format "$fmt" --bit-depth "$bits" "--normalize-dbfs=${norm}"
    record_row "$file" "$cmd_text" "${style} render variant $((idx + 1)) with deliberate format and rate variation."
    next_index=$((next_index + 1))
  done
done

chain_presets=(arcade_overheat atari2600_shred binaural_nuisance cassette_headache chip_broken_clock choir_collapse console_busfight n64_rattle ps1_grit rom_corruption wolf_fifth_panic)
chain_stage_variants=("" "stutter,pop" "crush,smear" "gate,dissonance-ring" "style:catastrophic,effect:stutter" "effect:dissonance-expand,effect:crush")
chain_durations=(0.18 0.20 0.22 0.24 0.26 0.28)
chain_gains=(0.80 0.84 0.88 0.76 0.92 0.86)
chain_rates=(192000 192000 192000 96000 192000 48000)
chain_formats=(float float int float int float)
chain_bits=(32 32 24 32 16 32)

start_section "Chain Grid"
for preset in "${chain_presets[@]}"; do
  for idx in "${!chain_stage_variants[@]}"; do
    file="examples/audio/$(printf '%03d' "$next_index")_chain_${preset}_v$(printf '%02d' $((idx + 1))).wav"
    stages="${chain_stage_variants[$idx]}"
    duration="${chain_durations[$idx]}"
    gain="${chain_gains[$idx]}"
    rate="${chain_rates[$idx]}"
    fmt="${chain_formats[$idx]}"
    bits="${chain_bits[$idx]}"
    seed=$((2000 + next_index * 41 + idx))
    if [[ -n "$stages" ]]; then
      cmd_text="./target/debug/usg chain --preset ${preset} --stages ${stages} --output ${file} --duration ${duration} --gain ${gain} --seed ${seed} --sample-rate ${rate} --sample-format ${fmt} --bit-depth ${bits}"
      "$BIN" chain --preset "$preset" --stages "$stages" --output "$file" --duration "$duration" --gain "$gain" --seed "$seed" --sample-rate "$rate" --sample-format "$fmt" --bit-depth "$bits"
      note="${preset} preset plus extra stages ${stages}."
    else
      cmd_text="./target/debug/usg chain --preset ${preset} --output ${file} --duration ${duration} --gain ${gain} --seed ${seed} --sample-rate ${rate} --sample-format ${fmt} --bit-depth ${bits}"
      "$BIN" chain --preset "$preset" --output "$file" --duration "$duration" --gain "$gain" --seed "$seed" --sample-rate "$rate" --sample-format "$fmt" --bit-depth "$bits"
      note="${preset} preset without appended stages."
    fi
    record_row "$file" "$cmd_text" "$note"
    next_index=$((next_index + 1))
  done
done

go_flavors=(glitch punish lucky)
go_layouts=("" stereo quad)
go_levels=(320 720 980)

mapfile -t contour_files < <(find presets/go_contours -maxdepth 1 -name '*.json' | sort)
start_section "Go Contours"
for contour in "${contour_files[@]}"; do
  contour_name="$(basename "$contour" .json)"
  for idx in "${!go_flavors[@]}"; do
    flavor="${go_flavors[$idx]}"
    layout="${go_layouts[$idx]}"
    level="${go_levels[$idx]}"
    file="examples/audio/$(printf '%03d' "$next_index")_go_${contour_name}_${flavor}.wav"
    cmd=("$BIN" go examples/audio/00_source_hum.wav --type "$flavor" --level "$level" --level-contour "$contour" --output "$file")
    cmd_text="./target/debug/usg go examples/audio/00_source_hum.wav --type ${flavor} --level ${level} --level-contour ${contour} --output ${file}"
    note="${contour_name} contour with ${flavor} flavor."
    if [[ -n "$layout" ]]; then
      cmd+=(--upmix "$layout")
      cmd_text+=" --upmix ${layout}"
      note+=" Upmixed to ${layout}."
    fi
    "${cmd[@]}"
    record_row "$file" "$cmd_text" "$note"
    next_index=$((next_index + 1))
  done
done

speech_profiles=(votrax-sc01 tms5220 sp0256 mea8000 s14001a c64-sam)
speech_oscillators=(pulse triangle buzz formant koch strange)

start_section "Speech Grid"
for profile in "${speech_profiles[@]}"; do
  for osc in "${speech_oscillators[@]}"; do
    file="examples/audio/$(printf '%03d' "$next_index")_speech_${profile}_${osc}.wav"
    text="UGLY ${profile} ${osc}"
    cmd_text="./target/debug/usg speech --text ${text} --profile ${profile} --primary-osc ${osc} --output ${file}"
    "$BIN" speech --text "$text" --profile "$profile" --primary-osc "$osc" --output "$file"
    record_row "$file" "$cmd_text" "Speech-chip render using ${profile} with ${osc} as the primary oscillator."
    next_index=$((next_index + 1))
  done
done

actual_count="$(find "$AUDIO_DIR" -maxdepth 1 -type f -name '*.wav' | wc -l | tr -d ' ')"
if [[ "$count" -ne "$EXPECTED_COUNT" ]]; then
  echo "generated README count $count but expected $EXPECTED_COUNT" >&2
  exit 1
fi
if [[ "$actual_count" -ne "$EXPECTED_COUNT" ]]; then
  echo "generated audio count $actual_count but expected $EXPECTED_COUNT" >&2
  exit 1
fi

echo "Generated $count example WAV files and refreshed README_EXAMPLES.md"
