use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use usg::{
    AnalysisReport, AnalyzeModel, AnalyzeOptions, ChainStage, CoordSystem, DEFAULT_GPU_CRUSH_BITS,
    DEFAULT_GPU_CRUSH_MIX, DEFAULT_GPU_DRIVE, GoFlavor, OutputEncoding, RenderBackend,
    RenderEngine, RenderOptions, SpatialGoOptions, SpeechChipProfile, SpeechInputMode,
    SpeechOscillator, SpeechRenderOptions, Style, SurroundLayout, Trajectory, UglinessContour,
    analyze_wav_with_options, available_effects, available_styles, backend_capabilities,
    backend_status_report, default_jobs, go_ugly_file_with_engine_contour_encoding,
    go_ugly_upmix_file_with_engine_contour_encoding, parse_chain_stage, point_to_xyz,
    render_chain_to_wav_with_engine, render_speech_to_wav_with_engine, render_to_wav_with_engine,
    resolve_backend_plan,
};

#[derive(Debug, Parser)]
#[command(
    name = "usg",
    version,
    about = "UglySoundGenerator: command-line ugliness in pure Rust"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Render an ugly sound to a WAV file.
    Render(RenderArgs),
    /// Render chiptune speech from text using classic speech-chip-inspired models.
    Speech(SpeechArgs),
    /// Analyze a WAV file and report ugliness metrics.
    Analyze(AnalyzeArgs),
    /// Render every style, analyze each output, and write a pack summary.
    RenderPack(RenderPackArgs),
    /// Take an input file and force it to a target ugliness level.
    Go(GoArgs),
    /// Build a chained synthesis/effects pipeline and render one output file.
    Chain(ChainArgs),
    /// List available render styles.
    Styles,
    /// Show backend availability for CPU/Metal/CUDA.
    Backends,
    /// List built-in contour presets.
    Presets(PresetsArgs),
    /// Benchmark render throughput by backend.
    Benchmark(BenchmarkArgs),
    /// Generate a large library of ugly files in one run.
    Marathon(MarathonArgs),
}

#[derive(Debug, Clone, Parser)]
struct RenderArgs {
    /// Output WAV path.
    #[arg(short, long, default_value = "ugly.wav")]
    output: PathBuf,

    /// Duration in seconds.
    #[arg(short = 'd', long, default_value_t = 3.0)]
    duration: f64,

    /// Sample rate in Hz.
    #[arg(short = 'r', long, default_value_t = 192_000)]
    sample_rate: u32,

    #[command(flatten)]
    output_format: OutputFormatArgs,

    /// Optional seed for repeatable chaos.
    #[arg(long)]
    seed: Option<u64>,

    /// Ugliness style profile.
    #[arg(long, value_enum, default_value_t = StyleArg::Harsh)]
    style: StyleArg,

    /// Output gain (0.0..1.0).
    #[arg(long, default_value_t = 0.8)]
    gain: f64,

    /// Normalize peak to this dBFS target (default: 0.0 dBFS).
    #[arg(long, default_value_t = 0.0)]
    normalize_dbfs: f64,

    /// Disable normalization.
    #[arg(long)]
    no_normalize: bool,

    /// GPU post-FX drive (backend post stage).
    #[arg(long)]
    gpu_drive: Option<f64>,

    /// GPU post-FX bitcrush depth in bits.
    #[arg(long)]
    gpu_crush_bits: Option<f64>,

    /// GPU post-FX blend between dry and crushed signal (0..1).
    #[arg(long)]
    gpu_crush_mix: Option<f64>,

    /// Rendering backend.
    #[arg(long, value_enum, default_value_t = RenderBackendArg::Auto)]
    backend: RenderBackendArg,

    /// Parallel worker count (0 = auto).
    #[arg(long, default_value_t = 0)]
    jobs: usize,
}

#[derive(Debug, Clone, Parser)]
struct SpeechArgs {
    /// Output WAV path.
    #[arg(short, long, default_value = "out/speech.wav")]
    output: PathBuf,

    /// Inline text to synthesize.
    #[arg(long, conflicts_with = "text_file")]
    text: Option<String>,

    /// Read text to synthesize from a UTF-8 file.
    #[arg(long, conflicts_with = "text")]
    text_file: Option<PathBuf>,

    /// Treat input as characters, words, sentences, or paragraphs.
    #[arg(long, value_enum, default_value_t = SpeechInputModeArg::Auto)]
    input_mode: SpeechInputModeArg,

    /// Speech-chip-inspired profile.
    #[arg(long, value_enum, default_value_t = SpeechChipProfileArg::Tms5220)]
    profile: SpeechChipProfileArg,

    /// Primary speech oscillator.
    #[arg(long, value_enum, default_value_t = SpeechOscillatorArg::Phoneme)]
    primary_osc: SpeechOscillatorArg,

    /// Secondary speech oscillator.
    #[arg(long, value_enum, default_value_t = SpeechOscillatorArg::Buzz)]
    secondary_osc: SpeechOscillatorArg,

    /// Tertiary speech oscillator.
    #[arg(long, value_enum, default_value_t = SpeechOscillatorArg::Koch)]
    tertiary_osc: SpeechOscillatorArg,

    /// Output sample rate in Hz.
    #[arg(short = 'r', long, default_value_t = 192_000)]
    sample_rate: u32,

    #[command(flatten)]
    output_format: OutputFormatArgs,

    /// Optional seed for repeatable speech.
    #[arg(long)]
    seed: Option<u64>,

    /// Units per second (characters/words/sentences depending on input mode).
    #[arg(long, default_value_t = 11.0)]
    units_per_second: f64,

    /// Base pitch in Hz.
    #[arg(long, default_value_t = 118.0)]
    pitch_hz: f64,

    /// Random pitch jitter amount.
    #[arg(long, default_value_t = 0.06)]
    pitch_jitter: f64,

    /// Vibrato rate in Hz.
    #[arg(long, default_value_t = 5.0)]
    vibrato_hz: f64,

    /// Vibrato depth.
    #[arg(long, default_value_t = 0.02)]
    vibrato_depth: f64,

    /// Pulse duty cycle for pulse-based speech oscillators.
    #[arg(long, default_value_t = 0.42)]
    duty_cycle: f64,

    /// Shift formants up/down.
    #[arg(long, default_value_t = 1.0)]
    formant_shift: f64,

    /// Consonant noise amount.
    #[arg(long, default_value_t = 0.4)]
    consonant_noise: f64,

    /// Vowel/formant bed mix.
    #[arg(long, default_value_t = 0.7)]
    vowel_mix: f64,

    /// Broadband hiss amount.
    #[arg(long, default_value_t = 0.05)]
    hiss: f64,

    /// Square-wave buzz amount.
    #[arg(long, default_value_t = 0.2)]
    buzz: f64,

    /// Wavefold amount.
    #[arg(long, default_value_t = 2.4)]
    fold: f64,

    /// Chaos/fractal modulation amount.
    #[arg(long, default_value_t = 0.35)]
    chaos: f64,

    /// Ring/modulator speech amount.
    #[arg(long, default_value_t = 0.25)]
    robotize: f64,

    /// Pitch glide amount between units.
    #[arg(long, default_value_t = 0.12)]
    glide: f64,

    /// Flatten character-level pitch differences.
    #[arg(long, default_value_t = 0.35)]
    monotone: f64,

    /// Extra dynamic emphasis on capitals and punctuation.
    #[arg(long, default_value_t = 0.25)]
    emphasis: f64,

    /// Gap after whitespace.
    #[arg(long, default_value_t = 42.0)]
    word_gap_ms: f64,

    /// Extra gap after sentence punctuation.
    #[arg(long, default_value_t = 110.0)]
    sentence_gap_ms: f64,

    /// Extra gap after paragraph breaks.
    #[arg(long, default_value_t = 220.0)]
    paragraph_gap_ms: f64,

    /// Generic punctuation pause.
    #[arg(long, default_value_t = 80.0)]
    punctuation_gap_ms: f64,

    /// Per-unit attack time.
    #[arg(long, default_value_t = 6.0)]
    attack_ms: f64,

    /// Per-unit release time.
    #[arg(long, default_value_t = 20.0)]
    release_ms: f64,

    /// Bit depth used by chipy speech down-quantization.
    #[arg(long, default_value_t = 7.5)]
    bitcrush_bits: f64,

    /// Sample-and-hold rate used by speech chip coloration.
    #[arg(long, default_value_t = 9_600.0)]
    sample_hold_hz: f64,

    /// Ring-mod mix.
    #[arg(long, default_value_t = 0.18)]
    ring_mix: f64,

    /// Sub-oscillator mix.
    #[arg(long, default_value_t = 0.12)]
    sub_mix: f64,

    /// Nasal resonance amount.
    #[arg(long, default_value_t = 0.16)]
    nasal: f64,

    /// Throat/noisy low-end amount.
    #[arg(long, default_value_t = 0.14)]
    throat: f64,

    /// Slow long-term tuning drift.
    #[arg(long, default_value_t = 0.03)]
    drift: f64,

    /// Sample-and-hold / resampler grime mix.
    #[arg(long, default_value_t = 0.25)]
    resampler_grit: f64,

    /// Output gain (0.0..1.0).
    #[arg(long, default_value_t = 0.85)]
    gain: f64,

    /// Normalize peak to this dBFS target.
    #[arg(long, default_value_t = -0.6)]
    normalize_dbfs: f64,

    /// Disable normalization.
    #[arg(long)]
    no_normalize: bool,

    /// GPU post-FX drive (backend post stage).
    #[arg(long)]
    gpu_drive: Option<f64>,

    /// GPU post-FX bitcrush depth in bits.
    #[arg(long)]
    gpu_crush_bits: Option<f64>,

    /// GPU post-FX blend between dry and crushed signal (0..1).
    #[arg(long)]
    gpu_crush_mix: Option<f64>,

    /// Rendering backend.
    #[arg(long, value_enum, default_value_t = RenderBackendArg::Auto)]
    backend: RenderBackendArg,

    /// Parallel worker count (0 = auto).
    #[arg(long, default_value_t = 0)]
    jobs: usize,

    /// Try to play the output file after rendering.
    #[arg(long)]
    play: bool,
}

#[derive(Debug, Clone, Parser)]
struct AnalyzeArgs {
    /// WAV file to inspect.
    input: PathBuf,

    /// Analysis model.
    #[arg(long, value_enum, default_value_t = AnalyzeModelArg::Basic)]
    model: AnalyzeModelArg,

    /// FFT size used by psycho model.
    #[arg(long, default_value_t = 2048)]
    fft_size: usize,

    /// Hop size used by psycho model.
    #[arg(long, default_value_t = 512)]
    hop_size: usize,

    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,

    /// Also compute the joke UglierBasis score and breakdown.
    #[arg(long)]
    joke: bool,
}

#[derive(Debug, Clone, Parser)]
struct GoArgs {
    /// Input WAV file.
    input: PathBuf,

    /// Output WAV file (default: <input-stem>.go.wav).
    #[arg(short, long)]
    output: Option<PathBuf>,

    #[command(flatten)]
    output_format: OutputFormatArgs,

    /// Output sample rate in Hz for the uglified file.
    #[arg(short = 'r', long, default_value_t = 192_000)]
    sample_rate: u32,

    /// Ugliness level (1..1000).
    #[arg(long, value_parser = clap::value_parser!(u16).range(1..=1000), default_value_t = 700)]
    level: u16,

    /// Optional uglification flavor.
    #[arg(long = "type", value_enum)]
    flavor: Option<GoFlavorArg>,

    /// Optional surround layout for upmix while uglifying.
    /// Examples: stereo,quad,5.1,7.1,custom:12
    #[arg(long)]
    upmix: Option<String>,

    /// Coordinate system for locus and trajectory endpoints.
    #[arg(long, value_enum, default_value_t = CoordSystemArg::Cartesian)]
    coords: CoordSystemArg,

    /// Initial source locus. Format: a,b,c where:
    /// cartesian => x,y,z ; polar => az_deg,el_deg,r
    #[arg(long, default_value = "0,1,0")]
    locus: String,

    /// Trajectory: static | line:a,b,c | orbit:radius,turns
    #[arg(long, default_value = "static")]
    trajectory: String,

    /// Optional seed for deterministic uglification.
    #[arg(long)]
    seed: Option<u64>,

    /// Ugliness contour JSON file (time-varying level envelope).
    /// Format: {"interpolation":"linear","points":[{"t":0.0,"level":200},{"t":1.0,"level":900}]}
    #[arg(long, conflicts_with = "level_contour_json")]
    level_contour: Option<PathBuf>,

    /// Ugliness contour JSON passed directly on the command line.
    /// Format: {"interpolation":"linear","points":[{"t":0.0,"level":200},{"t":1.0,"level":900}]}
    #[arg(long, conflicts_with = "level_contour")]
    level_contour_json: Option<String>,

    /// Normalize peak to this dBFS target (default: 0.0 dBFS).
    #[arg(long, default_value_t = 0.0)]
    normalize_dbfs: f64,

    /// Disable normalization.
    #[arg(long)]
    no_normalize: bool,

    /// GPU post-FX drive (backend post stage).
    #[arg(long)]
    gpu_drive: Option<f64>,

    /// GPU post-FX bitcrush depth in bits.
    #[arg(long)]
    gpu_crush_bits: Option<f64>,

    /// GPU post-FX blend between dry and crushed signal (0..1).
    #[arg(long)]
    gpu_crush_mix: Option<f64>,

    /// Rendering backend.
    #[arg(long, value_enum, default_value_t = RenderBackendArg::Auto)]
    backend: RenderBackendArg,

    /// Parallel worker count (0 = auto).
    #[arg(long, default_value_t = 0)]
    jobs: usize,
}

#[derive(Debug, Clone, Parser)]
struct RenderPackArgs {
    /// Directory to write the rendered WAV files and summary.
    #[arg(long, default_value = "out/pack")]
    out_dir: PathBuf,

    /// Output summary JSON file path (default: <out-dir>/summary.json).
    #[arg(long)]
    summary: Option<PathBuf>,

    /// Output CSV ranking path (default: <out-dir>/ranking.csv).
    #[arg(long)]
    csv: Option<PathBuf>,

    /// Output HTML report path (default: <out-dir>/report.html).
    #[arg(long)]
    html: Option<PathBuf>,

    /// Duration in seconds for each rendered style.
    #[arg(short = 'd', long, default_value_t = 1.5)]
    duration: f64,

    /// Sample rate in Hz.
    #[arg(short = 'r', long, default_value_t = 192_000)]
    sample_rate: u32,

    #[command(flatten)]
    output_format: OutputFormatArgs,

    /// Optional pack seed (deterministic per-style seeds are derived from this).
    #[arg(long)]
    seed: Option<u64>,

    /// Output gain (0.0..1.0).
    #[arg(long, default_value_t = 0.8)]
    gain: f64,

    /// Normalize peak to this dBFS target (default: 0.0 dBFS).
    #[arg(long, default_value_t = 0.0)]
    normalize_dbfs: f64,

    /// Disable normalization.
    #[arg(long)]
    no_normalize: bool,

    /// Analysis model for pack scoring.
    #[arg(long, value_enum, default_value_t = AnalyzeModelArg::Psycho)]
    model: AnalyzeModelArg,

    /// FFT size used by psycho model.
    #[arg(long, default_value_t = 2048)]
    fft_size: usize,

    /// Hop size used by psycho model.
    #[arg(long, default_value_t = 512)]
    hop_size: usize,

    /// Number of top ugliest entries to print.
    #[arg(long, default_value_t = 5)]
    top: usize,

    /// Optional style subset for pack rendering.
    /// Example: --styles glitch,punish,lucky
    #[arg(long, value_enum, value_delimiter = ',', num_args = 1..)]
    styles: Vec<StyleArg>,

    /// Rendering backend.
    #[arg(long, value_enum, default_value_t = RenderBackendArg::Auto)]
    backend: RenderBackendArg,

    /// Parallel worker count for pack jobs (0 = auto).
    #[arg(long, default_value_t = 0)]
    jobs: usize,

    /// GPU post-FX drive (backend post stage).
    #[arg(long)]
    gpu_drive: Option<f64>,

    /// GPU post-FX bitcrush depth in bits.
    #[arg(long)]
    gpu_crush_bits: Option<f64>,

    /// GPU post-FX blend between dry and crushed signal (0..1).
    #[arg(long)]
    gpu_crush_mix: Option<f64>,
}

#[derive(Debug, Clone, Parser)]
struct ChainArgs {
    /// Built-in chain preset name or JSON path.
    #[arg(long)]
    preset: Option<String>,

    /// Comma-separated chain stages. Example: glitch,stutter,pop
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    stages: Vec<String>,

    /// Output WAV path.
    #[arg(short, long, default_value = "out/chain.wav")]
    output: PathBuf,

    /// Duration in seconds.
    #[arg(short = 'd', long, default_value_t = 2.0)]
    duration: f64,

    /// Sample rate in Hz.
    #[arg(short = 'r', long, default_value_t = 192_000)]
    sample_rate: u32,

    #[command(flatten)]
    output_format: OutputFormatArgs,

    /// Optional seed for repeatable chain output.
    #[arg(long)]
    seed: Option<u64>,

    /// Final output gain (0.0..1.0).
    #[arg(long, default_value_t = 0.8)]
    gain: f64,

    /// Normalize peak to this dBFS target (default: 0.0 dBFS).
    #[arg(long, default_value_t = 0.0)]
    normalize_dbfs: f64,

    /// Disable normalization.
    #[arg(long)]
    no_normalize: bool,

    /// Try to play the output file after rendering.
    #[arg(long)]
    play: bool,

    /// Rendering backend.
    #[arg(long, value_enum, default_value_t = RenderBackendArg::Auto)]
    backend: RenderBackendArg,

    /// Parallel worker count (0 = auto).
    #[arg(long, default_value_t = 0)]
    jobs: usize,

    /// GPU post-FX drive (backend post stage).
    #[arg(long)]
    gpu_drive: Option<f64>,

    /// GPU post-FX bitcrush depth in bits.
    #[arg(long)]
    gpu_crush_bits: Option<f64>,

    /// GPU post-FX blend between dry and crushed signal (0..1).
    #[arg(long)]
    gpu_crush_mix: Option<f64>,
}

#[derive(Debug, Clone, Parser)]
struct BenchmarkArgs {
    /// Number of benchmark runs per backend.
    #[arg(long, default_value_t = 3)]
    runs: usize,

    /// Duration in seconds per render.
    #[arg(short = 'd', long, default_value_t = 1.0)]
    duration: f64,

    /// Sample rate in Hz.
    #[arg(short = 'r', long, default_value_t = 192_000)]
    sample_rate: u32,

    #[command(flatten)]
    output_format: OutputFormatArgs,

    /// Style used during benchmark.
    #[arg(long, value_enum, default_value_t = StyleArg::Glitch)]
    style: StyleArg,

    /// Optional seed base.
    #[arg(long)]
    seed: Option<u64>,

    /// Output gain (0.0..1.0).
    #[arg(long, default_value_t = 0.8)]
    gain: f64,

    /// Normalize peak to this dBFS target (default: 0.0 dBFS).
    #[arg(long, default_value_t = 0.0)]
    normalize_dbfs: f64,

    /// Disable normalization.
    #[arg(long)]
    no_normalize: bool,

    /// Parallel worker count (0 = auto).
    #[arg(long, default_value_t = 0)]
    jobs: usize,

    /// GPU post-FX drive (backend post stage).
    #[arg(long)]
    gpu_drive: Option<f64>,

    /// GPU post-FX bitcrush depth in bits.
    #[arg(long)]
    gpu_crush_bits: Option<f64>,

    /// GPU post-FX blend between dry and crushed signal (0..1).
    #[arg(long)]
    gpu_crush_mix: Option<f64>,

    /// Optional structured JSON benchmark report.
    #[arg(long)]
    json_output: Option<PathBuf>,

    /// Optional CSV benchmark ranking.
    #[arg(long)]
    csv_output: Option<PathBuf>,
}

#[derive(Debug, Clone, Parser)]
struct PresetsArgs {
    /// Which built-in preset family to inspect.
    #[arg(long, value_enum, default_value_t = PresetKindArg::All)]
    kind: PresetKindArg,

    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,

    /// Print absolute preset paths in the listing.
    #[arg(long)]
    paths: bool,

    /// Show one preset by file stem or file name.
    #[arg(long)]
    show: Option<String>,
}

#[derive(Debug, Clone, Parser)]
struct MarathonArgs {
    /// Output directory for generated WAV files.
    #[arg(long, default_value = "out/marathon")]
    out_dir: PathBuf,

    /// Number of files to generate.
    #[arg(long, default_value_t = 256)]
    count: usize,

    /// Minimum duration in seconds.
    #[arg(long, default_value_t = 0.10)]
    min_duration: f64,

    /// Maximum duration in seconds.
    #[arg(long, default_value_t = 900.0)]
    max_duration: f64,

    /// Sample rate in Hz.
    #[arg(short = 'r', long, default_value_t = 192_000)]
    sample_rate: u32,

    #[command(flatten)]
    output_format: OutputFormatArgs,

    /// Optional seed for deterministic output.
    #[arg(long)]
    seed: Option<u64>,

    /// Output gain (0.0..1.0).
    #[arg(long, default_value_t = 0.9)]
    gain: f64,

    /// Normalize peak to this dBFS target (default: 0.0 dBFS).
    #[arg(long, default_value_t = 0.0)]
    normalize_dbfs: f64,

    /// Disable normalization.
    #[arg(long)]
    no_normalize: bool,

    /// Optional style subset. Example: --styles glitch,punish,lucky
    #[arg(long, value_enum, value_delimiter = ',', num_args = 1..)]
    styles: Vec<StyleArg>,

    /// Rendering backend.
    #[arg(long, value_enum, default_value_t = RenderBackendArg::Auto)]
    backend: RenderBackendArg,

    /// Parallel worker count (0 = auto).
    #[arg(long, default_value_t = 0)]
    jobs: usize,

    /// GPU post-FX drive (backend post stage).
    #[arg(long)]
    gpu_drive: Option<f64>,

    /// GPU post-FX bitcrush depth in bits.
    #[arg(long)]
    gpu_crush_bits: Option<f64>,

    /// GPU post-FX blend between dry and crushed signal (0..1).
    #[arg(long)]
    gpu_crush_mix: Option<f64>,

    /// Optional manifest path (default: <out-dir>/manifest.json).
    #[arg(long)]
    manifest: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum StyleArg {
    Harsh,
    Digital,
    Meltdown,
    Glitch,
    Pop,
    Buzz,
    Rub,
    Hum,
    Distort,
    Spank,
    Punish,
    Steal,
    Catastrophic,
    Wink,
    Lucky,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SpeechChipProfileArg {
    VotraxSc01,
    Tms5220,
    Sp0256,
    Mea8000,
    S14001a,
    C64Sam,
    Arcadey90s,
    HandheldLcd,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SpeechOscillatorArg {
    Pulse,
    Triangle,
    Saw,
    Noise,
    Buzz,
    Formant,
    Ring,
    Fold,
    Koch,
    Mandelbrot,
    Strange,
    Phoneme,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SpeechInputModeArg {
    Auto,
    Character,
    Word,
    Sentence,
    Paragraph,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum AnalyzeModelArg {
    Basic,
    Psycho,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum GoFlavorArg {
    Glitch,
    Stutter,
    Puff,
    Punish,
    Geek,
    DissonanceRing,
    DissonanceExpand,
    Random,
    Lucky,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CoordSystemArg {
    Cartesian,
    Polar,
}

#[derive(Debug, Clone, Copy, Args)]
struct OutputFormatArgs {
    /// File sample format for rendered WAV output.
    #[arg(long, value_enum, default_value_t = SampleFormatArg::Float)]
    sample_format: SampleFormatArg,

    /// File bit depth for rendered WAV output.
    #[arg(long, default_value_t = 32)]
    bit_depth: u16,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SampleFormatArg {
    Float,
    Int,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PresetKindArg {
    All,
    Contour,
    Chain,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RenderBackendArg {
    Auto,
    Cpu,
    Metal,
    Cuda,
}

impl From<StyleArg> for Style {
    fn from(value: StyleArg) -> Self {
        match value {
            StyleArg::Harsh => Style::Harsh,
            StyleArg::Digital => Style::Digital,
            StyleArg::Meltdown => Style::Meltdown,
            StyleArg::Glitch => Style::Glitch,
            StyleArg::Pop => Style::Pop,
            StyleArg::Buzz => Style::Buzz,
            StyleArg::Rub => Style::Rub,
            StyleArg::Hum => Style::Hum,
            StyleArg::Distort => Style::Distort,
            StyleArg::Spank => Style::Spank,
            StyleArg::Punish => Style::Punish,
            StyleArg::Steal => Style::Steal,
            StyleArg::Catastrophic => Style::Catastrophic,
            StyleArg::Wink => Style::Wink,
            StyleArg::Lucky => Style::Lucky,
        }
    }
}

impl From<SpeechChipProfileArg> for SpeechChipProfile {
    fn from(value: SpeechChipProfileArg) -> Self {
        match value {
            SpeechChipProfileArg::VotraxSc01 => SpeechChipProfile::VotraxSc01,
            SpeechChipProfileArg::Tms5220 => SpeechChipProfile::Tms5220,
            SpeechChipProfileArg::Sp0256 => SpeechChipProfile::Sp0256,
            SpeechChipProfileArg::Mea8000 => SpeechChipProfile::Mea8000,
            SpeechChipProfileArg::S14001a => SpeechChipProfile::S14001a,
            SpeechChipProfileArg::C64Sam => SpeechChipProfile::C64Sam,
            SpeechChipProfileArg::Arcadey90s => SpeechChipProfile::Arcadey90s,
            SpeechChipProfileArg::HandheldLcd => SpeechChipProfile::HandheldLcd,
        }
    }
}

impl From<SpeechOscillatorArg> for SpeechOscillator {
    fn from(value: SpeechOscillatorArg) -> Self {
        match value {
            SpeechOscillatorArg::Pulse => SpeechOscillator::Pulse,
            SpeechOscillatorArg::Triangle => SpeechOscillator::Triangle,
            SpeechOscillatorArg::Saw => SpeechOscillator::Saw,
            SpeechOscillatorArg::Noise => SpeechOscillator::Noise,
            SpeechOscillatorArg::Buzz => SpeechOscillator::Buzz,
            SpeechOscillatorArg::Formant => SpeechOscillator::Formant,
            SpeechOscillatorArg::Ring => SpeechOscillator::Ring,
            SpeechOscillatorArg::Fold => SpeechOscillator::Fold,
            SpeechOscillatorArg::Koch => SpeechOscillator::Koch,
            SpeechOscillatorArg::Mandelbrot => SpeechOscillator::Mandelbrot,
            SpeechOscillatorArg::Strange => SpeechOscillator::Strange,
            SpeechOscillatorArg::Phoneme => SpeechOscillator::Phoneme,
        }
    }
}

impl From<SpeechInputModeArg> for SpeechInputMode {
    fn from(value: SpeechInputModeArg) -> Self {
        match value {
            SpeechInputModeArg::Auto => SpeechInputMode::Auto,
            SpeechInputModeArg::Character => SpeechInputMode::Character,
            SpeechInputModeArg::Word => SpeechInputMode::Word,
            SpeechInputModeArg::Sentence => SpeechInputMode::Sentence,
            SpeechInputModeArg::Paragraph => SpeechInputMode::Paragraph,
        }
    }
}

impl From<AnalyzeModelArg> for AnalyzeModel {
    fn from(value: AnalyzeModelArg) -> Self {
        match value {
            AnalyzeModelArg::Basic => AnalyzeModel::Basic,
            AnalyzeModelArg::Psycho => AnalyzeModel::Psycho,
        }
    }
}

impl From<RenderBackendArg> for RenderBackend {
    fn from(value: RenderBackendArg) -> Self {
        match value {
            RenderBackendArg::Auto => RenderBackend::Auto,
            RenderBackendArg::Cpu => RenderBackend::Cpu,
            RenderBackendArg::Metal => RenderBackend::Metal,
            RenderBackendArg::Cuda => RenderBackend::Cuda,
        }
    }
}

impl From<GoFlavorArg> for GoFlavor {
    fn from(value: GoFlavorArg) -> Self {
        match value {
            GoFlavorArg::Glitch => GoFlavor::Glitch,
            GoFlavorArg::Stutter => GoFlavor::Stutter,
            GoFlavorArg::Puff => GoFlavor::Puff,
            GoFlavorArg::Punish => GoFlavor::Punish,
            GoFlavorArg::Geek => GoFlavor::Geek,
            GoFlavorArg::DissonanceRing => GoFlavor::DissonanceRing,
            GoFlavorArg::DissonanceExpand => GoFlavor::DissonanceExpand,
            GoFlavorArg::Random => GoFlavor::Random,
            GoFlavorArg::Lucky => GoFlavor::Lucky,
        }
    }
}

impl From<CoordSystemArg> for CoordSystem {
    fn from(value: CoordSystemArg) -> Self {
        match value {
            CoordSystemArg::Cartesian => CoordSystem::Cartesian,
            CoordSystemArg::Polar => CoordSystem::Polar,
        }
    }
}

impl OutputFormatArgs {
    fn output_encoding(&self) -> Result<OutputEncoding> {
        match (self.sample_format, self.bit_depth) {
            (SampleFormatArg::Float, 32) => Ok(OutputEncoding::Float32),
            (SampleFormatArg::Int, 16) => Ok(OutputEncoding::Int16),
            (SampleFormatArg::Int, 24) => Ok(OutputEncoding::Int24),
            (SampleFormatArg::Int, 32) => Ok(OutputEncoding::Int32),
            (SampleFormatArg::Float, bits) => Err(anyhow::anyhow!(
                "float output only supports --bit-depth 32 (got {bits})"
            )),
            (SampleFormatArg::Int, bits) => Err(anyhow::anyhow!(
                "integer output supports --bit-depth 16, 24, or 32 (got {bits})"
            )),
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Render(args) => render(args),
        Command::Speech(args) => speech(args),
        Command::Analyze(args) => analyze(args),
        Command::RenderPack(args) => render_pack(args),
        Command::Go(args) => go(args),
        Command::Chain(args) => chain(args),
        Command::Styles => styles(),
        Command::Backends => backends(),
        Command::Presets(args) => presets(args),
        Command::Benchmark(args) => benchmark(args),
        Command::Marathon(args) => marathon(args),
    }
}

fn render(args: RenderArgs) -> Result<()> {
    let output_encoding = args.output_format.output_encoding()?;
    let engine = engine_from_args(
        args.backend,
        args.jobs,
        args.gpu_drive,
        args.gpu_crush_bits,
        args.gpu_crush_mix,
    );
    let options = RenderOptions {
        duration: args.duration,
        sample_rate: args.sample_rate,
        seed: args.seed,
        style: args.style.into(),
        gain: args.gain,
        normalize: !args.no_normalize,
        normalize_dbfs: args.normalize_dbfs,
        output_encoding,
    };
    let summary = render_to_wav_with_engine(&args.output, &options, &engine)
        .with_context(|| format!("failed to write {}", args.output.display()))?;

    println!(
        "Rendered {} ({} frames @ {} Hz, style={}, seed={}, format={}, backend={} -> {}, jobs={})",
        summary.output.display(),
        summary.frames,
        summary.sample_rate,
        summary.style,
        summary.seed,
        summary.output_encoding,
        summary.backend_requested,
        summary.backend_active,
        summary.jobs
    );
    Ok(())
}

fn speech(args: SpeechArgs) -> Result<()> {
    let output_encoding = args.output_format.output_encoding()?;
    let jobs = if args.jobs == 0 {
        default_jobs()
    } else {
        args.jobs
    };
    let engine = engine_from_backend(
        args.backend.into(),
        jobs,
        args.gpu_drive,
        args.gpu_crush_bits,
        args.gpu_crush_mix,
    );
    let text = if let Some(path) = args.text_file.as_ref() {
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?
    } else {
        args.text
            .clone()
            .unwrap_or_else(|| "HELLO FROM UGLY SOUND GENERATOR".to_string())
    };
    let opts = SpeechRenderOptions {
        text,
        input_mode: args.input_mode.into(),
        sample_rate: args.sample_rate,
        seed: args.seed,
        chip_profile: args.profile.into(),
        primary_osc: args.primary_osc.into(),
        secondary_osc: args.secondary_osc.into(),
        tertiary_osc: args.tertiary_osc.into(),
        gain: args.gain,
        normalize: !args.no_normalize,
        normalize_dbfs: args.normalize_dbfs,
        output_encoding,
        units_per_second: args.units_per_second,
        pitch_hz: args.pitch_hz,
        pitch_jitter: args.pitch_jitter,
        vibrato_hz: args.vibrato_hz,
        vibrato_depth: args.vibrato_depth,
        duty_cycle: args.duty_cycle,
        formant_shift: args.formant_shift,
        consonant_noise: args.consonant_noise,
        vowel_mix: args.vowel_mix,
        hiss: args.hiss,
        buzz: args.buzz,
        fold: args.fold,
        chaos: args.chaos,
        robotize: args.robotize,
        glide: args.glide,
        monotone: args.monotone,
        emphasis: args.emphasis,
        word_gap_ms: args.word_gap_ms,
        sentence_gap_ms: args.sentence_gap_ms,
        paragraph_gap_ms: args.paragraph_gap_ms,
        punctuation_gap_ms: args.punctuation_gap_ms,
        attack_ms: args.attack_ms,
        release_ms: args.release_ms,
        bitcrush_bits: args.bitcrush_bits,
        sample_hold_hz: args.sample_hold_hz,
        ring_mix: args.ring_mix,
        sub_mix: args.sub_mix,
        nasal: args.nasal,
        throat: args.throat,
        drift: args.drift,
        resampler_grit: args.resampler_grit,
    };
    let summary = render_speech_to_wav_with_engine(&args.output, &opts, &engine)?;
    println!(
        "Rendered speech to {} ({} frames @ {} Hz, profile={}, mode={}, units={}, format={})",
        args.output.display(),
        summary.frames,
        summary.sample_rate,
        summary.chip_profile,
        summary.input_mode,
        summary.units_rendered,
        summary.output_encoding
    );
    println!(
        "oscillators: primary={}, secondary={}, tertiary={}",
        opts.primary_osc, opts.secondary_osc, opts.tertiary_osc
    );
    println!("seed: {}", summary.seed);

    if args.play {
        let played = try_play_audio(&args.output);
        if !played {
            println!(
                "Could not auto-play audio. Open manually: {}",
                args.output.display()
            );
        }
    }
    Ok(())
}

fn analyze(args: AnalyzeArgs) -> Result<()> {
    let options = AnalyzeOptions {
        model: args.model.into(),
        fft_size: args.fft_size,
        hop_size: args.hop_size,
        joke: args.joke,
    };
    let report = analyze_wav_with_options(&args.input, &options)
        .with_context(|| format!("failed to analyze {}", args.input.display()))?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("Analysis for {}", args.input.display());
    println!("model: {}", report.model);
    println!("sample_rate_hz: {}", report.basic.sample_rate);
    println!("channels: {}", report.basic.channels);
    println!("duration_s: {:.3}", report.basic.duration_s);
    println!("peak_dbfs: {:.2}", report.basic.peak_dbfs);
    println!("rms_dbfs: {:.2}", report.basic.rms_dbfs);
    println!("crest_factor_db: {:.2}", report.basic.crest_factor_db);
    println!("zero_crossing_rate: {:.4}", report.basic.zero_crossing_rate);
    println!("clipped_pct: {:.2}%", report.basic.clipped_pct);
    println!("harshness_ratio: {:.3}", report.basic.harshness_ratio);
    println!("basic_ugly_index: {:.1}/1000", report.basic.ugly_index);

    if let Some(psycho) = &report.psycho {
        println!("psycho.clip_norm: {:.3}", psycho.clip_norm);
        println!("psycho.harshness_norm: {:.3}", psycho.harshness_norm);
        println!("psycho.roughness_norm: {:.3}", psycho.roughness_norm);
        println!("psycho.sharpness_norm: {:.3}", psycho.sharpness_norm);
        println!("psycho.dissonance_norm: {:.3}", psycho.dissonance_norm);
        println!("psycho.transient_norm: {:.3}", psycho.transient_norm);
        println!("psycho.harmonicity_norm: {:.3}", psycho.harmonicity_norm);
        println!(
            "psycho.inharmonicity_norm: {:.3}",
            psycho.inharmonicity_norm
        );
        println!(
            "psycho.binaural_beat_norm: {:.3}",
            psycho.binaural_beat_norm
        );
        println!(
            "psycho.beat_conflict_norm: {:.3}",
            psycho.beat_conflict_norm
        );
        println!(
            "psycho.tritone_tension_norm: {:.3}",
            psycho.tritone_tension_norm
        );
        println!("psycho.wolf_fifth_norm: {:.3}", psycho.wolf_fifth_norm);
        println!("psycho.weighted_sum: {:.3}", psycho.weighted_sum);
        println!("psycho_fft_size: {}", psycho.fft_size);
        println!("psycho_hop_size: {}", psycho.hop_size);
    }

    if let Some(joke) = &report.joke {
        println!("joke.uglierbasis_index: {:.1}/1000", joke.uglierbasis_index);
        println!("joke.verdict: {}", joke.verdict);
        println!(
            "joke.academic_cluster_norm: {:.3}",
            joke.academic_cluster_norm
        );
        println!(
            "joke.bureaucratic_overhead_norm: {:.3}",
            joke.bureaucratic_overhead_norm
        );
        println!("joke.all_high_bonus_norm: {:.3}", joke.all_high_bonus_norm);
        println!(
            "joke.harmonicity_relief_norm: {:.3}",
            joke.harmonicity_relief_norm
        );
        println!("joke.weighted_sum: {:.3}", joke.weighted_sum);
    }

    println!("ugly_index: {:.1}/1000", report.selected_ugly_index);
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkEntry {
    rank: usize,
    backend: String,
    average_ms: f64,
    min_ms: f64,
    max_ms: f64,
    realtime_factor: f64,
    runs: usize,
    run_ms: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkReport {
    generated_unix_s: u64,
    runs: usize,
    duration_s: f64,
    sample_rate_hz: u32,
    style: String,
    gain: f64,
    normalize: bool,
    normalize_dbfs: f64,
    output_encoding: String,
    jobs: usize,
    gpu_drive: f64,
    gpu_crush_bits: f64,
    gpu_crush_mix: f64,
    rows: Vec<BenchmarkEntry>,
}

#[derive(Debug, Clone, Serialize)]
struct PresetEntry {
    kind: String,
    version: u16,
    name: String,
    path: String,
    summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChainPreset {
    #[serde(default = "default_chain_preset_version")]
    version: u16,
    name: String,
    #[serde(default)]
    description: String,
    stages: Vec<String>,
}

fn default_chain_preset_version() -> u16 {
    1
}

#[derive(Debug, Clone, Serialize)]
struct PackEntry {
    style: String,
    seed: u64,
    output: String,
    ugly_index: f64,
    analysis: AnalysisReport,
}

#[derive(Debug, Clone, Serialize)]
struct PackRankingEntry {
    rank: usize,
    style: String,
    output: String,
    ugly_index: f64,
    seed: u64,
    basic_ugly_index: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PackSummary {
    generated_unix_s: u64,
    model: String,
    duration_s: f64,
    sample_rate_hz: u32,
    gain: f64,
    output_encoding: String,
    backend_requested: String,
    backend_active: String,
    jobs: usize,
    base_seed: u64,
    styles_requested: Vec<String>,
    styles_rendered: usize,
    entries: Vec<PackEntry>,
    ranking: Vec<PackRankingEntry>,
}

#[derive(Debug, Clone, Serialize)]
struct MarathonEntry {
    index: usize,
    style: String,
    duration_s: f64,
    seed: u64,
    output: String,
}

#[derive(Debug, Clone, Serialize)]
struct MarathonManifest {
    generated_unix_s: u64,
    count: usize,
    sample_rate_hz: u32,
    gain: f64,
    output_encoding: String,
    normalize: bool,
    normalize_dbfs: f64,
    backend_requested: String,
    backend_active: String,
    jobs: usize,
    base_seed: u64,
    min_duration_s: f64,
    max_duration_s: f64,
    styles: Vec<String>,
    entries: Vec<MarathonEntry>,
}

fn render_pack(args: RenderPackArgs) -> Result<()> {
    let output_encoding = args.output_format.output_encoding()?;
    let analyze_options = AnalyzeOptions {
        model: args.model.into(),
        fft_size: args.fft_size,
        hop_size: args.hop_size,
        joke: false,
    };
    let engine = engine_from_args(
        args.backend,
        args.jobs,
        args.gpu_drive,
        args.gpu_crush_bits,
        args.gpu_crush_mix,
    );
    let plan = resolve_backend_plan(&engine)?;
    fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("failed to create {}", args.out_dir.display()))?;

    let base_seed = args.seed.unwrap_or_else(seed_from_time);
    let styles_to_render = selected_pack_styles(&args.styles);
    let tasks: Vec<(usize, Style)> = styles_to_render.iter().copied().enumerate().collect();
    let pool = ThreadPoolBuilder::new()
        .num_threads(plan.jobs)
        .build()
        .context("failed to build pack worker pool")?;
    let rendered_entries: Vec<Result<(usize, PackEntry)>> = pool.install(|| {
        tasks
            .par_iter()
            .map(|(idx, style)| {
                let seed = style_seed(base_seed, *idx as u64);
                let output = args
                    .out_dir
                    .join(format!("{:02}_{}.wav", *idx + 1, style.as_str()));
                let render_options = RenderOptions {
                    duration: args.duration,
                    sample_rate: args.sample_rate,
                    seed: Some(seed),
                    style: *style,
                    gain: args.gain,
                    normalize: !args.no_normalize,
                    normalize_dbfs: args.normalize_dbfs,
                    output_encoding,
                };
                render_to_wav_with_engine(&output, &render_options, &engine)
                    .with_context(|| format!("failed to render {}", output.display()))?;
                let analysis = analyze_wav_with_options(&output, &analyze_options)
                    .with_context(|| format!("failed to analyze {}", output.display()))?;
                Ok((
                    *idx,
                    PackEntry {
                        style: style.as_str().to_string(),
                        seed,
                        output: output.display().to_string(),
                        ugly_index: analysis.selected_ugly_index,
                        analysis,
                    },
                ))
            })
            .collect()
    });
    let mut ordered_entries: Vec<(usize, PackEntry)> = Vec::with_capacity(rendered_entries.len());
    for entry in rendered_entries {
        ordered_entries.push(entry?);
    }
    ordered_entries.sort_by_key(|(idx, _)| *idx);
    let entries: Vec<PackEntry> = ordered_entries
        .into_iter()
        .map(|(_, entry)| entry)
        .collect();

    let mut ranked = entries.clone();
    ranked.sort_by(|a, b| b.ugly_index.total_cmp(&a.ugly_index));
    let ranking: Vec<PackRankingEntry> = ranked
        .iter()
        .enumerate()
        .map(|(i, e)| PackRankingEntry {
            rank: i + 1,
            style: e.style.clone(),
            output: e.output.clone(),
            ugly_index: e.ugly_index,
            seed: e.seed,
            basic_ugly_index: e.analysis.basic.ugly_index,
        })
        .collect();

    let summary = PackSummary {
        generated_unix_s: now_unix_s(),
        model: analyze_options.model.as_str().to_string(),
        duration_s: args.duration,
        sample_rate_hz: args.sample_rate,
        gain: args.gain,
        output_encoding: output_encoding.as_str().to_string(),
        backend_requested: plan.requested.as_str().to_string(),
        backend_active: plan.active.as_str().to_string(),
        jobs: plan.jobs,
        base_seed,
        styles_requested: styles_to_render
            .iter()
            .map(|s| s.as_str().to_string())
            .collect(),
        styles_rendered: entries.len(),
        entries,
        ranking,
    };

    let summary_path = args
        .summary
        .unwrap_or_else(|| args.out_dir.join("summary.json"));
    let csv_path = args.csv.unwrap_or_else(|| args.out_dir.join("ranking.csv"));
    let html_path = args
        .html
        .unwrap_or_else(|| args.out_dir.join("report.html"));
    write_json(&summary_path, &summary)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;
    write_pack_csv(&csv_path, &summary)
        .with_context(|| format!("failed to write {}", csv_path.display()))?;
    write_pack_html(&html_path, &summary)
        .with_context(|| format!("failed to write {}", html_path.display()))?;

    println!(
        "Rendered style pack: {} styles -> {} (backend={} -> {}, jobs={})",
        summary.styles_rendered,
        args.out_dir.display(),
        summary.backend_requested,
        summary.backend_active,
        summary.jobs
    );
    println!("Output encoding: {}", summary.output_encoding);
    println!("Summary: {}", summary_path.display());
    println!("CSV: {}", csv_path.display());
    println!("HTML: {}", html_path.display());
    println!("Top ugliest:");
    for row in summary.ranking.iter().take(args.top.max(1)) {
        println!(
            "  {:>2}. {:<8} {:>5.1}/1000 (basic {:>5.1})  {}",
            row.rank, row.style, row.ugly_index, row.basic_ugly_index, row.output
        );
    }
    Ok(())
}

fn go(args: GoArgs) -> Result<()> {
    let output_encoding = args.output_format.output_encoding()?;
    let engine = engine_from_args(
        args.backend,
        args.jobs,
        args.gpu_drive,
        args.gpu_crush_bits,
        args.gpu_crush_mix,
    );
    let contour = parse_ugliness_contour(&args)?;
    let output = args
        .output
        .unwrap_or_else(|| default_go_output_path(&args.input));
    let flavor = args.flavor.map(Into::into);

    let summary = if let Some(layout_text) = args.upmix.as_ref() {
        let layout = parse_surround_layout(layout_text)?;
        let coord_system: CoordSystem = args.coords.into();
        let locus_vals = parse_triplet_csv(&args.locus)
            .with_context(|| format!("invalid --locus '{}'", args.locus))?;
        let locus_xyz = point_to_xyz(coord_system, locus_vals[0], locus_vals[1], locus_vals[2]);
        let trajectory = parse_trajectory(&args.trajectory, coord_system)
            .with_context(|| format!("invalid --trajectory '{}'", args.trajectory))?;
        let spatial = SpatialGoOptions {
            layout,
            locus_xyz,
            trajectory,
        };
        go_ugly_upmix_file_with_engine_contour_encoding(
            &args.input,
            &output,
            args.level,
            flavor,
            args.seed,
            !args.no_normalize,
            args.normalize_dbfs,
            spatial,
            contour.as_ref(),
            Some(args.sample_rate),
            output_encoding,
            &engine,
        )
    } else {
        go_ugly_file_with_engine_contour_encoding(
            &args.input,
            &output,
            args.level,
            flavor,
            args.seed,
            !args.no_normalize,
            args.normalize_dbfs,
            contour.as_ref(),
            Some(args.sample_rate),
            output_encoding,
            &engine,
        )
    }
    .with_context(|| {
        format!(
            "failed to uglify {} -> {}",
            args.input.display(),
            output.display()
        )
    })?;

    println!(
        "GO complete: {} ({} frames @ {} Hz, channels={}, level={}, type={}, layout={}, seed={}, format={}, backend={} -> {}, jobs={})",
        summary.output.display(),
        summary.frames,
        summary.sample_rate,
        summary.channels,
        summary.level,
        summary.flavor,
        summary.layout.as_deref().unwrap_or("source"),
        summary.seed,
        summary.output_encoding,
        summary.backend_requested,
        summary.backend_active,
        summary.jobs
    );
    if contour.is_some() {
        println!("Applied contour: time-varying ugliness envelope from JSON");
    }
    Ok(())
}

fn chain(args: ChainArgs) -> Result<()> {
    let output_encoding = args.output_format.output_encoding()?;
    let engine = engine_from_args(
        args.backend,
        args.jobs,
        args.gpu_drive,
        args.gpu_crush_bits,
        args.gpu_crush_mix,
    );
    let base_seed = args.seed.unwrap_or_else(seed_from_time);
    let mut stage_tokens = Vec::new();
    if let Some(preset_name) = args.preset.as_ref() {
        let preset = load_chain_preset(preset_name)?;
        stage_tokens.extend(preset.stages);
    }
    stage_tokens.extend(args.stages.clone());

    let mut parsed = Vec::new();
    let mut invalid = Vec::new();
    for token in &stage_tokens {
        if let Some(stage) = parse_chain_stage(token) {
            parsed.push(stage);
        } else {
            invalid.push(token.clone());
        }
    }
    if !invalid.is_empty() {
        return Err(anyhow::anyhow!(
            "unknown chain stage(s): {}. valid styles: {}. valid effects: {}. use style:<name> or effect:<name> for explicit resolution.",
            invalid.join(", "),
            styles_csv(),
            effects_csv(),
        ));
    }

    let frames = render_chain_to_wav_with_engine(
        &args.output,
        &parsed,
        args.duration,
        args.sample_rate,
        output_encoding,
        args.gain,
        !args.no_normalize,
        args.normalize_dbfs,
        base_seed,
        &engine,
    )?;

    println!(
        "Rendered chain to {} ({} frames @ {} Hz, format={}, stages: {})",
        args.output.display(),
        frames,
        args.sample_rate,
        output_encoding,
        format_chain(&parsed)
    );
    println!("seed: {}", base_seed);

    if args.play {
        let played = try_play_audio(&args.output);
        if !played {
            println!(
                "Could not auto-play audio. Open manually: {}",
                args.output.display()
            );
        }
    }
    Ok(())
}

fn styles() -> Result<()> {
    for style in available_styles() {
        println!("{style}");
    }
    Ok(())
}

fn backends() -> Result<()> {
    let caps = backend_capabilities();
    let status = backend_status_report();
    let effective = engine_from_backend(RenderBackend::Auto, default_jobs(), None, None, None);
    let cuda_disabled = std::env::var_os("USG_DISABLE_CUDA").is_some();

    println!(
        "CPU:   {}",
        if status.cpu.available {
            "available"
        } else {
            "not available"
        }
    );
    println!(
        "Metal: {}",
        if caps.metal {
            "available (build with --features metal)"
        } else {
            "not available"
        }
    );
    println!(
        "CUDA:  {}",
        if caps.cuda {
            "available (build with --features cuda)"
        } else {
            "not available"
        }
    );
    println!("Default jobs: {}", default_jobs());
    println!(
        "GPU post-FX defaults: drive={:.2}, crush_bits={:.2}, crush_mix={:.2}",
        effective.gpu_drive, effective.gpu_crush_bits, effective.gpu_crush_mix
    );
    println!("GPU post-FX env vars: USG_GPU_DRIVE, USG_GPU_CRUSH_BITS, USG_GPU_CRUSH_MIX");
    println!("CPU detail: {}", status.cpu.detail);
    println!(
        "Metal detail: feature_built={}, target_os_macos={}, runtime_available={}",
        cfg!(feature = "metal"),
        cfg!(target_os = "macos"),
        caps.metal
    );
    println!("Metal probe: {}", status.metal.detail);
    println!(
        "CUDA detail: feature_built={}, disabled_by_env={}, runtime_available={}",
        cfg!(feature = "cuda"),
        cuda_disabled,
        caps.cuda
    );
    println!("CUDA probe: {}", status.cuda.detail);
    if !cfg!(feature = "metal") {
        println!("Metal note: rebuild with --features metal to enable Metal acceleration.");
    } else if !cfg!(target_os = "macos") {
        println!("Metal note: Metal rendering is only supported on macOS targets.");
    } else if !caps.metal {
        println!("Metal note: no usable Metal device/runtime was detected.");
    }
    if cuda_disabled {
        println!("CUDA note: USG_DISABLE_CUDA is set, so CUDA was intentionally disabled.");
    } else if !cfg!(feature = "cuda") {
        println!("CUDA note: rebuild with --features cuda to enable CUDA acceleration.");
    } else if !caps.cuda {
        println!("CUDA note: a CUDA-capable runtime/device was not detected at startup.");
    }
    Ok(())
}

fn presets(args: PresetsArgs) -> Result<()> {
    let entries = builtin_preset_entries(args.kind)?;

    if let Some(name) = args.show.as_ref() {
        let path = resolve_preset_path(name, &entries)?;
        let info = entries
            .iter()
            .find(|entry| entry.path == path.display().to_string())
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("internal error: resolved preset missing from index"))?;
        let text = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if args.json {
            if info.kind == "contour" {
                let contour: UglinessContour = serde_json::from_str(&text)
                    .with_context(|| format!("invalid contour JSON in {}", path.display()))?;
                println!("{}", serde_json::to_string_pretty(&contour)?);
            } else {
                let preset: ChainPreset = serde_json::from_str(&text)
                    .with_context(|| format!("invalid chain preset JSON in {}", path.display()))?;
                println!("{}", serde_json::to_string_pretty(&preset)?);
            }
            return Ok(());
        }
        println!("Preset: {}", info.name);
        println!("Kind: {}", info.kind);
        println!("Version: {}", info.version);
        println!("Path: {}", path.display());
        println!("Summary: {}", info.summary);
        println!();
        println!("{}", text);
        return Ok(());
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&entries)?);
        return Ok(());
    }

    println!("Built-in presets: {}", entries.len());
    for entry in &entries {
        if args.paths {
            println!(
                "  {:<8} v{:<2} {:<28} {:<44} {}",
                entry.kind, entry.version, entry.name, entry.summary, entry.path
            );
        } else {
            println!(
                "  {:<8} v{:<2} {:<28} {}",
                entry.kind, entry.version, entry.name, entry.summary
            );
        }
    }
    println!("Use 'usg presets --show <name>' to inspect one preset.");
    Ok(())
}

fn benchmark(args: BenchmarkArgs) -> Result<()> {
    let output_encoding = args.output_format.output_encoding()?;
    let runs = args.runs.max(1);
    let jobs = if args.jobs == 0 {
        default_jobs()
    } else {
        args.jobs
    };
    let base_seed = args.seed.unwrap_or_else(seed_from_time);
    let caps = backend_capabilities();

    let mut backends = vec![RenderBackend::Cpu];
    if caps.metal {
        backends.push(RenderBackend::Metal);
    }
    if caps.cuda {
        backends.push(RenderBackend::Cuda);
    }

    println!(
        "Benchmarking {} backend(s), runs={}, duration={:.2}s, sample_rate={} Hz, jobs={}, format={}",
        backends.len(),
        runs,
        args.duration,
        args.sample_rate,
        jobs,
        output_encoding
    );

    let mut rows: Vec<BenchmarkEntry> = Vec::new();
    for backend in backends {
        let engine = engine_from_backend(
            backend,
            jobs,
            args.gpu_drive,
            args.gpu_crush_bits,
            args.gpu_crush_mix,
        );
        let mut run_ms = Vec::with_capacity(runs);
        for run_idx in 0..runs {
            let out_path = std::env::temp_dir().join(format!(
                "usg_bench_{}_{}_{}.wav",
                backend.as_str(),
                std::process::id(),
                run_idx
            ));
            let opts = RenderOptions {
                duration: args.duration,
                sample_rate: args.sample_rate,
                seed: Some(style_seed(base_seed, run_idx as u64)),
                style: args.style.into(),
                gain: args.gain,
                normalize: !args.no_normalize,
                normalize_dbfs: args.normalize_dbfs,
                output_encoding,
            };
            let t0 = Instant::now();
            render_to_wav_with_engine(&out_path, &opts, &engine)
                .with_context(|| format!("benchmark render failed for {}", backend))?;
            run_ms.push(t0.elapsed().as_secs_f64() * 1_000.0);
            let _ = fs::remove_file(out_path);
        }
        let average_ms = run_ms.iter().sum::<f64>() / run_ms.len() as f64;
        let min_ms = run_ms
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap_or(average_ms);
        let max_ms = run_ms
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap_or(average_ms);
        rows.push(BenchmarkEntry {
            rank: 0,
            backend: backend.as_str().to_string(),
            average_ms,
            min_ms,
            max_ms,
            realtime_factor: if average_ms > 0.0 {
                args.duration * 1_000.0 / average_ms
            } else {
                f64::INFINITY
            },
            runs,
            run_ms,
        });
    }

    rows.sort_by(|a, b| a.average_ms.total_cmp(&b.average_ms));
    for (idx, row) in rows.iter_mut().enumerate() {
        row.rank = idx + 1;
    }

    let report = BenchmarkReport {
        generated_unix_s: now_unix_s(),
        runs,
        duration_s: args.duration,
        sample_rate_hz: args.sample_rate,
        style: Style::from(args.style).as_str().to_string(),
        gain: args.gain,
        normalize: !args.no_normalize,
        normalize_dbfs: args.normalize_dbfs,
        output_encoding: output_encoding.as_str().to_string(),
        jobs,
        gpu_drive: engine_from_backend(
            RenderBackend::Auto,
            jobs,
            args.gpu_drive,
            args.gpu_crush_bits,
            args.gpu_crush_mix,
        )
        .gpu_drive,
        gpu_crush_bits: engine_from_backend(
            RenderBackend::Auto,
            jobs,
            args.gpu_drive,
            args.gpu_crush_bits,
            args.gpu_crush_mix,
        )
        .gpu_crush_bits,
        gpu_crush_mix: engine_from_backend(
            RenderBackend::Auto,
            jobs,
            args.gpu_drive,
            args.gpu_crush_bits,
            args.gpu_crush_mix,
        )
        .gpu_crush_mix,
        rows,
    };

    println!("Average render time:");
    for row in &report.rows {
        println!(
            "  {:>2}. {:<5} {:>8.3} ms  min {:>8.3}  max {:>8.3}  x{:>6.2} realtime",
            row.rank, row.backend, row.average_ms, row.min_ms, row.max_ms, row.realtime_factor
        );
    }

    if let Some(path) = args.json_output.as_ref() {
        write_json(path, &report).with_context(|| format!("failed to write {}", path.display()))?;
        println!("JSON: {}", path.display());
    }
    if let Some(path) = args.csv_output.as_ref() {
        write_benchmark_csv(path, &report)
            .with_context(|| format!("failed to write {}", path.display()))?;
        println!("CSV: {}", path.display());
    }
    Ok(())
}

fn marathon(args: MarathonArgs) -> Result<()> {
    let output_encoding = args.output_format.output_encoding()?;
    if args.count == 0 {
        return Err(anyhow::anyhow!("count must be >= 1"));
    }
    if args.min_duration <= 0.0 || args.max_duration <= 0.0 {
        return Err(anyhow::anyhow!("durations must be > 0"));
    }
    if args.min_duration > args.max_duration {
        return Err(anyhow::anyhow!("min-duration must be <= max-duration"));
    }

    fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("failed to create {}", args.out_dir.display()))?;
    let styles = selected_pack_styles(&args.styles);
    if styles.is_empty() {
        return Err(anyhow::anyhow!("no styles selected"));
    }

    let base_seed = args.seed.unwrap_or_else(seed_from_time);
    let engine = engine_from_args(
        args.backend,
        args.jobs,
        args.gpu_drive,
        args.gpu_crush_bits,
        args.gpu_crush_mix,
    );
    let plan = resolve_backend_plan(&engine)?;
    let tasks: Vec<usize> = (0..args.count).collect();
    let pool = ThreadPoolBuilder::new()
        .num_threads(plan.jobs)
        .build()
        .context("failed to build marathon worker pool")?;

    let rendered: Vec<Result<(usize, MarathonEntry)>> = pool.install(|| {
        tasks
            .par_iter()
            .map(|idx| {
                let slot_seed = style_seed(base_seed, *idx as u64);
                let style = styles[*idx % styles.len()];
                let duration = duration_for_slot(slot_seed, args.min_duration, args.max_duration);
                let output = args.out_dir.join(format!(
                    "{:05}_{}_s{slot_seed}.wav",
                    idx + 1,
                    style.as_str()
                ));
                let opts = RenderOptions {
                    duration,
                    sample_rate: args.sample_rate,
                    seed: Some(slot_seed),
                    style,
                    gain: args.gain,
                    normalize: !args.no_normalize,
                    normalize_dbfs: args.normalize_dbfs,
                    output_encoding,
                };
                render_to_wav_with_engine(&output, &opts, &engine)
                    .with_context(|| format!("failed to render {}", output.display()))?;
                Ok((
                    *idx,
                    MarathonEntry {
                        index: *idx + 1,
                        style: style.as_str().to_string(),
                        duration_s: duration,
                        seed: slot_seed,
                        output: output.display().to_string(),
                    },
                ))
            })
            .collect()
    });

    let mut ordered: Vec<(usize, MarathonEntry)> = Vec::with_capacity(rendered.len());
    for row in rendered {
        ordered.push(row?);
    }
    ordered.sort_by_key(|(idx, _)| *idx);
    let entries: Vec<MarathonEntry> = ordered.into_iter().map(|(_, entry)| entry).collect();

    let manifest = MarathonManifest {
        generated_unix_s: now_unix_s(),
        count: entries.len(),
        sample_rate_hz: args.sample_rate,
        gain: args.gain,
        output_encoding: output_encoding.as_str().to_string(),
        normalize: !args.no_normalize,
        normalize_dbfs: args.normalize_dbfs,
        backend_requested: plan.requested.as_str().to_string(),
        backend_active: plan.active.as_str().to_string(),
        jobs: plan.jobs,
        base_seed,
        min_duration_s: args.min_duration,
        max_duration_s: args.max_duration,
        styles: styles.iter().map(|s| s.as_str().to_string()).collect(),
        entries,
    };

    let manifest_path = args
        .manifest
        .unwrap_or_else(|| args.out_dir.join("manifest.json"));
    write_json(&manifest_path, &manifest)
        .with_context(|| format!("failed to write {}", manifest_path.display()))?;

    println!(
        "Marathon complete: {} files in {} (backend={} -> {}, jobs={})",
        manifest.count,
        args.out_dir.display(),
        manifest.backend_requested,
        manifest.backend_active,
        manifest.jobs
    );
    println!("Output encoding: {}", manifest.output_encoding);
    println!("Manifest: {}", manifest_path.display());
    println!("Examples:");
    for row in manifest.entries.iter().take(5) {
        println!(
            "  {:>5}. {:<8} {:>7.2}s  {}",
            row.index, row.style, row.duration_s, row.output
        );
    }
    Ok(())
}

fn duration_for_slot(seed: u64, min_duration: f64, max_duration: f64) -> f64 {
    if (max_duration - min_duration).abs() <= f64::EPSILON {
        return min_duration;
    }
    let unit = ((seed >> 11) as f64) / ((1_u64 << 53) as f64);
    min_duration + (max_duration - min_duration) * unit.clamp(0.0, 1.0)
}

fn engine_from_args(
    backend: RenderBackendArg,
    jobs: usize,
    gpu_drive: Option<f64>,
    gpu_crush_bits: Option<f64>,
    gpu_crush_mix: Option<f64>,
) -> RenderEngine {
    engine_from_backend(
        backend.into(),
        if jobs == 0 { default_jobs() } else { jobs },
        gpu_drive,
        gpu_crush_bits,
        gpu_crush_mix,
    )
}

fn engine_from_backend(
    backend: RenderBackend,
    jobs: usize,
    gpu_drive: Option<f64>,
    gpu_crush_bits: Option<f64>,
    gpu_crush_mix: Option<f64>,
) -> RenderEngine {
    let drive = gpu_drive
        .or_else(|| env_f64("USG_GPU_DRIVE"))
        .unwrap_or(DEFAULT_GPU_DRIVE)
        .clamp(0.1, 16.0);
    let crush_bits = gpu_crush_bits
        .or_else(|| env_f64("USG_GPU_CRUSH_BITS"))
        .unwrap_or(DEFAULT_GPU_CRUSH_BITS)
        .clamp(0.0, 24.0);
    let crush_mix = gpu_crush_mix
        .or_else(|| env_f64("USG_GPU_CRUSH_MIX"))
        .unwrap_or(DEFAULT_GPU_CRUSH_MIX)
        .clamp(0.0, 1.0);

    RenderEngine {
        backend,
        jobs: jobs.max(1),
        gpu_drive: drive,
        gpu_crush_bits: crush_bits,
        gpu_crush_mix: crush_mix,
    }
}

fn env_f64(key: &str) -> Option<f64> {
    std::env::var(key).ok()?.trim().parse::<f64>().ok()
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let text = serde_json::to_string_pretty(value)?;
    fs::write(path, text)?;
    Ok(())
}

fn seed_from_time() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(dur) => dur.as_nanos() as u64,
        Err(_) => 0xBAD5_EED,
    }
}

fn now_unix_s() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(dur) => dur.as_secs(),
        Err(_) => 0,
    }
}

fn style_seed(base_seed: u64, idx: u64) -> u64 {
    let mut z = base_seed
        .wrapping_add(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(idx.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn format_chain(stages: &[ChainStage]) -> String {
    stages
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(" -> ")
}

fn styles_csv() -> String {
    available_styles()
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(",")
}

fn effects_csv() -> String {
    available_effects()
        .iter()
        .map(|e| e.as_str())
        .collect::<Vec<_>>()
        .join(",")
}

fn default_go_output_path(input: &Path) -> PathBuf {
    let parent = input.parent().unwrap_or_else(|| Path::new(""));
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("input");
    parent.join(format!("{stem}.go.wav"))
}

fn parse_ugliness_contour(args: &GoArgs) -> Result<Option<UglinessContour>> {
    if let Some(path) = args.level_contour.as_ref() {
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read contour JSON {}", path.display()))?;
        let contour: UglinessContour = serde_json::from_str(&text)
            .with_context(|| format!("invalid JSON in {}", path.display()))?;
        return Ok(Some(contour));
    }
    if let Some(json) = args.level_contour_json.as_ref() {
        let contour: UglinessContour =
            serde_json::from_str(json).context("invalid --level-contour-json payload")?;
        return Ok(Some(contour));
    }
    Ok(None)
}

fn parse_surround_layout(text: &str) -> Result<SurroundLayout> {
    let t = text.trim().to_ascii_lowercase();
    match t.as_str() {
        "mono" | "1.0" => Ok(SurroundLayout::Mono),
        "stereo" | "2.0" => Ok(SurroundLayout::Stereo),
        "quad" | "4.0" => Ok(SurroundLayout::Quad),
        "5.1" | "fiveone" => Ok(SurroundLayout::FiveOne),
        "7.1" | "sevenone" => Ok(SurroundLayout::SevenOne),
        _ => {
            if let Some(rest) = t.strip_prefix("custom:") {
                let n: u16 = rest
                    .parse()
                    .map_err(|_| anyhow::anyhow!("invalid custom layout channels: {rest}"))?;
                return Ok(SurroundLayout::Custom(n.max(1)));
            }
            Err(anyhow::anyhow!(
                "unknown upmix layout '{text}' (use mono|stereo|quad|5.1|7.1|custom:N)"
            ))
        }
    }
}

fn parse_triplet_csv(text: &str) -> Result<[f64; 3]> {
    let parts: Vec<&str> = text.split(',').map(|s| s.trim()).collect();
    if parts.len() != 3 {
        return Err(anyhow::anyhow!("expected three comma-separated values"));
    }
    let a = parts[0]
        .parse::<f64>()
        .map_err(|_| anyhow::anyhow!("invalid number '{}'", parts[0]))?;
    let b = parts[1]
        .parse::<f64>()
        .map_err(|_| anyhow::anyhow!("invalid number '{}'", parts[1]))?;
    let c = parts[2]
        .parse::<f64>()
        .map_err(|_| anyhow::anyhow!("invalid number '{}'", parts[2]))?;
    Ok([a, b, c])
}

fn parse_trajectory(text: &str, coords: CoordSystem) -> Result<Trajectory> {
    let t = text.trim();
    if t.eq_ignore_ascii_case("static") {
        return Ok(Trajectory::Static);
    }
    if let Some(rest) = t.strip_prefix("line:") {
        let end = parse_triplet_csv(rest)?;
        let end_xyz = point_to_xyz(coords, end[0], end[1], end[2]);
        return Ok(Trajectory::Line { end: end_xyz });
    }
    if let Some(rest) = t.strip_prefix("orbit:") {
        let parts: Vec<&str> = rest.split(',').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            return Err(anyhow::anyhow!(
                "orbit trajectory must be orbit:radius,turns"
            ));
        }
        let radius = parts[0]
            .parse::<f64>()
            .map_err(|_| anyhow::anyhow!("invalid orbit radius '{}'", parts[0]))?;
        let turns = parts[1]
            .parse::<f64>()
            .map_err(|_| anyhow::anyhow!("invalid orbit turns '{}'", parts[1]))?;
        return Ok(Trajectory::Orbit {
            radius: radius.abs(),
            turns,
        });
    }
    Err(anyhow::anyhow!(
        "unknown trajectory '{text}' (use static | line:a,b,c | orbit:radius,turns)"
    ))
}

fn try_play_audio(path: &Path) -> bool {
    #[cfg(target_os = "macos")]
    {
        if run_player("afplay", &[path]) {
            return true;
        }
    }
    if run_player(
        "ffplay",
        &[Path::new("-nodisp"), Path::new("-autoexit"), path],
    ) {
        return true;
    }
    run_player("aplay", &[path])
}

fn run_player(cmd: &str, args: &[&Path]) -> bool {
    let mut child = ProcessCommand::new(cmd);
    for arg in args {
        child.arg(arg);
    }
    child.status().map(|s| s.success()).unwrap_or(false)
}

fn selected_pack_styles(style_args: &[StyleArg]) -> Vec<Style> {
    if style_args.is_empty() {
        return available_styles().to_vec();
    }
    let mut out = Vec::new();
    for s in style_args {
        let style: Style = (*s).into();
        if !out.contains(&style) {
            out.push(style);
        }
    }
    out
}

fn preset_dir(kind: PresetKindArg) -> PathBuf {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("presets");
    match kind {
        PresetKindArg::Contour => base.join("go_contours"),
        PresetKindArg::Chain => base.join("chains"),
        PresetKindArg::All => base,
    }
}

fn builtin_preset_entries(kind: PresetKindArg) -> Result<Vec<PresetEntry>> {
    let mut entries = Vec::new();
    match kind {
        PresetKindArg::All => {
            entries.extend(read_preset_family(PresetKindArg::Contour)?);
            entries.extend(read_preset_family(PresetKindArg::Chain)?);
        }
        family => entries.extend(read_preset_family(family)?),
    }
    entries.sort_by(|a, b| a.name.cmp(&b.name).then(a.kind.cmp(&b.kind)));
    Ok(entries)
}

fn read_preset_family(kind: PresetKindArg) -> Result<Vec<PresetEntry>> {
    let dir = preset_dir(kind);
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut paths: Vec<PathBuf> = fs::read_dir(&dir)
        .with_context(|| format!("failed to read {}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
        .collect();
    paths.sort();
    let mut entries = Vec::with_capacity(paths.len());
    for path in paths {
        entries.push(preset_entry_from_path(kind, &path)?);
    }
    Ok(entries)
}

fn preset_entry_from_path(kind: PresetKindArg, path: &Path) -> Result<PresetEntry> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    match kind {
        PresetKindArg::Contour => {
            let contour: UglinessContour = serde_json::from_str(&text)
                .with_context(|| format!("invalid contour preset JSON in {}", path.display()))?;
            let first = contour
                .points
                .first()
                .ok_or_else(|| anyhow::anyhow!("preset {} has no points", path.display()))?;
            let min_level = contour
                .points
                .iter()
                .map(|point| point.level)
                .min()
                .unwrap_or(first.level);
            let max_level = contour
                .points
                .iter()
                .map(|point| point.level)
                .max()
                .unwrap_or(first.level);
            Ok(PresetEntry {
                kind: "contour".to_string(),
                version: contour.version,
                name: contour.name.unwrap_or_else(|| {
                    path.file_stem()
                        .and_then(|stem| stem.to_str())
                        .unwrap_or("preset")
                        .to_string()
                }),
                path: path.display().to_string(),
                summary: format!(
                    "{:?} points={} range={}..{}",
                    contour.interpolation,
                    contour.points.len(),
                    min_level,
                    max_level
                )
                .to_ascii_lowercase(),
            })
        }
        PresetKindArg::Chain => {
            let preset: ChainPreset = serde_json::from_str(&text)
                .with_context(|| format!("invalid chain preset JSON in {}", path.display()))?;
            Ok(PresetEntry {
                kind: "chain".to_string(),
                version: preset.version,
                name: preset.name.clone(),
                path: path.display().to_string(),
                summary: format!(
                    "stages={} {}",
                    preset.stages.len(),
                    preset.stages.join(" -> ")
                ),
            })
        }
        PresetKindArg::All => unreachable!(),
    }
}

fn resolve_preset_path(name: &str, entries: &[PresetEntry]) -> Result<PathBuf> {
    let normalized = name.trim();
    if let Some(entry) = entries.iter().find(|entry| entry.name == normalized) {
        return Ok(PathBuf::from(&entry.path));
    }
    if let Some(entry) = entries
        .iter()
        .find(|entry| format!("{}.json", entry.name) == normalized)
    {
        return Ok(PathBuf::from(&entry.path));
    }
    Err(anyhow::anyhow!(
        "unknown preset '{normalized}'. Run 'usg presets' to list built-ins."
    ))
}

fn load_chain_preset(name: &str) -> Result<ChainPreset> {
    let path = if name.ends_with(".json") || name.contains('/') {
        PathBuf::from(name)
    } else {
        let entries = builtin_preset_entries(PresetKindArg::Chain)?;
        resolve_preset_path(name, &entries)?
    };
    let text =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let preset: ChainPreset = serde_json::from_str(&text)
        .with_context(|| format!("invalid chain preset JSON in {}", path.display()))?;
    if preset.version != 1 {
        return Err(anyhow::anyhow!(
            "unsupported chain preset version {} in {}",
            preset.version,
            path.display()
        ));
    }
    if preset.stages.is_empty() {
        return Err(anyhow::anyhow!(
            "chain preset {} had no stages",
            path.display()
        ));
    }
    Ok(preset)
}

fn write_benchmark_csv(path: &Path, report: &BenchmarkReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut text = String::new();
    text.push_str("rank,backend,average_ms,min_ms,max_ms,realtime_factor,runs\n");
    for row in &report.rows {
        text.push_str(&format!(
            "{},{},{:.6},{:.6},{:.6},{:.6},{}\n",
            row.rank,
            csv_escape(&row.backend),
            row.average_ms,
            row.min_ms,
            row.max_ms,
            row.realtime_factor,
            row.runs,
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

fn write_pack_csv(path: &Path, summary: &PackSummary) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut text = String::new();
    text.push_str("rank,style,ugly_index,basic_ugly_index,seed,output\n");
    for row in &summary.ranking {
        text.push_str(&format!(
            "{},{},{:.6},{:.6},{},{}\n",
            row.rank,
            csv_escape(&row.style),
            row.ugly_index,
            row.basic_ugly_index,
            row.seed,
            csv_escape(&row.output),
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

fn write_pack_html(path: &Path, summary: &PackSummary) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut rows = String::new();
    for row in &summary.ranking {
        rows.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{:.1}</td><td>{:.1}</td><td>{}</td><td><audio controls preload=\"none\" src=\"{}\"></audio></td></tr>\n",
            row.rank,
            html_escape(&row.style),
            row.ugly_index,
            row.basic_ugly_index,
            row.seed,
            html_escape(file_name_or_path(&row.output)),
        ));
    }

    let html = format!(
        "<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>USG Pack Report</title>
  <style>
    :root {{
      --bg: #0f1116;
      --panel: #171b25;
      --text: #e9edf5;
      --muted: #9aa5bf;
      --accent: #ff5f45;
      --line: #2b3345;
    }}
    body {{
      margin: 0;
      font-family: \"IBM Plex Sans\", \"Avenir Next\", \"Segoe UI\", sans-serif;
      background: radial-gradient(circle at 20% 0%, #202636 0%, var(--bg) 45%);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1000px;
      margin: 0 auto;
      padding: 28px 18px 44px;
    }}
    .card {{
      background: linear-gradient(180deg, #1b2130, var(--panel));
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    h1 {{ margin: 0 0 6px; font-size: 1.5rem; }}
    p {{ margin: 4px 0; color: var(--muted); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 10px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: middle;
      font-size: 0.92rem;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      letter-spacing: 0.02em;
    }}
    td:nth-child(3) {{
      color: var(--accent);
      font-weight: 700;
    }}
    audio {{ width: 220px; height: 30px; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"card\">
      <h1>UglySoundGenerator Pack Report</h1>
      <p>Model: {model} | Styles: {styles_rendered} | Duration: {duration:.2}s | SR: {sample_rate} Hz | Gain: {gain:.2}</p>
      <p>Backend: {backend_requested} -> {backend_active} | Jobs: {jobs}</p>
      <p>Base seed: {base_seed}</p>
    </div>
    <div class=\"card\">
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Style</th>
            <th>Ugliness</th>
            <th>Basic</th>
            <th>Seed</th>
            <th>Listen</th>
          </tr>
        </thead>
        <tbody>
{rows}        </tbody>
      </table>
    </div>
  </div>
</body>
</html>",
        model = html_escape(&summary.model),
        styles_rendered = summary.styles_rendered,
        duration = summary.duration_s,
        sample_rate = summary.sample_rate_hz,
        gain = summary.gain,
        backend_requested = html_escape(&summary.backend_requested),
        backend_active = html_escape(&summary.backend_active),
        jobs = summary.jobs,
        base_seed = summary.base_seed,
        rows = rows
    );

    fs::write(path, html)?;
    Ok(())
}

fn csv_escape(input: &str) -> String {
    if input.contains(',') || input.contains('"') || input.contains('\n') {
        format!("\"{}\"", input.replace('"', "\"\""))
    } else {
        input.to_string()
    }
}

fn html_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn file_name_or_path(path: &str) -> &str {
    Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
}
