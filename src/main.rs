use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use usg::{
    AnalysisReport, AnalyzeModel, AnalyzeOptions, COLBYS_MAX, COLBYS_MIN, ChainStage, CoordSystem,
    DEFAULT_GPU_CRUSH_BITS, DEFAULT_GPU_CRUSH_MIX, DEFAULT_GPU_DRIVE, GoFlavor, OutputEncoding,
    PieceOptions, RenderBackend, RenderEngine, RenderOptions, SpatialGoOptions, SpeechChipProfile,
    SpeechInputMode, SpeechIntelligibility, SpeechOscillator, SpeechRenderOptions, Style,
    SurroundLayout, TimelineOptions, Trajectory, UglinessContour, analyze_wav_timeline,
    analyze_wav_with_options, available_effects, available_styles, backend_capabilities,
    backend_status_report, default_jobs, go_ugly_file_with_engine_contour_encoding,
    go_ugly_upmix_file_with_engine_contour_encoding, parse_chain_stage, point_to_xyz,
    render_chain_to_wav_with_engine, render_piece_to_wav_with_engine,
    render_speech_with_artifacts_to_wav_with_engine, render_to_wav_with_engine,
    resolve_backend_plan, score_speech_intelligibility,
};

mod cli_core_commands;

use cli_core_commands::{
    analyze, backends, benchmark, chain, go, marathon, piece, presets, render, render_pack, speech,
    speech_pack, styles,
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
    /// Render a multichannel piece made of many short ugly sounds.
    Piece(PieceArgs),
    /// Render chiptune speech from text using classic speech-chip-inspired models.
    Speech(SpeechArgs),
    /// Analyze a WAV file and report ugliness metrics.
    Analyze(AnalyzeArgs),
    /// Render every speech-chip profile for the same text, analyze each, and write a ranked summary.
    SpeechPack(SpeechPackArgs),
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
    /// Apply random ugly mutations to an input WAV and rank by ugliness delta.
    Mutate(MutateArgs),
    /// Force every WAV in a directory to a target ugliness level.
    NormalizePack(NormalizePackArgs),
    /// Breed uglier renders across generations using a genetic algorithm.
    Evolve(EvolveArgs),
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

    #[command(flatten)]
    randomness: RandomnessArgs,
}

#[derive(Debug, Clone, Parser)]
struct PieceArgs {
    /// Output WAV path.
    #[arg(short, long, default_value = "out/piece.wav")]
    output: PathBuf,

    /// Total piece duration in seconds.
    #[arg(short = 'd', long, default_value_t = 12.0)]
    duration: f64,

    /// Output channel count.
    #[arg(long, default_value_t = 2)]
    channels: u16,

    /// Sample rate in Hz.
    #[arg(short = 'r', long, default_value_t = 192_000)]
    sample_rate: u32,

    #[command(flatten)]
    output_format: OutputFormatArgs,

    /// Optional seed for reproducible pieces.
    #[arg(long)]
    seed: Option<u64>,

    /// Optional style subset. Example: --styles glitch,punish,catastrophic
    #[arg(long, value_enum, value_delimiter = ',', num_args = 1..)]
    styles: Vec<StyleArg>,

    /// Average number of short ugly events per second.
    #[arg(long, default_value_t = 5.0)]
    events_per_second: f64,

    /// Minimum event duration in seconds.
    #[arg(long, default_value_t = 0.03)]
    min_event_duration: f64,

    /// Maximum event duration in seconds.
    #[arg(long, default_value_t = 0.35)]
    max_event_duration: f64,

    /// Minimum spatial spread across channels.
    #[arg(long, default_value_t = 0.35)]
    min_pan_width: f64,

    /// Maximum spatial spread across channels.
    #[arg(long, default_value_t = 1.75)]
    max_pan_width: f64,

    /// Base output gain (0.0..1.0).
    #[arg(long, default_value_t = 0.7)]
    gain: f64,

    /// Normalize peak to this dBFS target.
    #[arg(long, default_value_t = -0.6)]
    normalize_dbfs: f64,

    /// Disable normalization.
    #[arg(long)]
    no_normalize: bool,

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

    #[command(flatten)]
    randomness: RandomnessArgs,
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

    /// Disable speech-text normalization before parsing.
    #[arg(long)]
    no_normalize_text: bool,

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

    /// Per-word accent shaping.
    #[arg(long, default_value_t = 0.18)]
    word_accent: f64,

    /// Per-sentence melodic lilt shaping.
    #[arg(long, default_value_t = 0.14)]
    sentence_lilt: f64,

    /// Per-paragraph downward contour amount.
    #[arg(long, default_value_t = 0.10)]
    paragraph_decline: f64,

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

    /// Optional JSON path for rendered speech metadata and analysis.
    #[arg(long)]
    analysis_json: Option<PathBuf>,

    /// Optional JSON path for phoneme timeline export.
    #[arg(long)]
    timeline_json: Option<PathBuf>,

    #[command(flatten)]
    randomness: RandomnessArgs,
}

#[derive(Debug, Clone, Parser)]
struct SpeechPackArgs {
    /// Directory to write the rendered WAV files and summary.
    #[arg(long, default_value = "out/speech_pack")]
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

    /// Inline text to synthesize across all profiles.
    #[arg(
        long,
        conflicts_with = "text_file",
        default_value = "UGLY SOUND GENERATOR"
    )]
    text: Option<String>,

    /// Read text from a UTF-8 file instead of --text.
    #[arg(long, conflicts_with = "text")]
    text_file: Option<PathBuf>,

    /// Input mode for text segmentation.
    #[arg(long, value_enum, default_value_t = SpeechInputModeArg::Auto)]
    input_mode: SpeechInputModeArg,

    /// Output sample rate in Hz.
    #[arg(short = 'r', long, default_value_t = 192_000)]
    sample_rate: u32,

    #[command(flatten)]
    output_format: OutputFormatArgs,

    /// Optional seed for repeatable renders.
    #[arg(long)]
    seed: Option<u64>,

    /// Base pitch in Hz.
    #[arg(long, default_value_t = 118.0)]
    pitch_hz: f64,

    /// Normalize peak to this dBFS target.
    #[arg(long, default_value_t = -0.6)]
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

    /// How to rank speech-pack entries.
    #[arg(long, value_enum, default_value_t = SpeechPackRankArg::Balanced)]
    rank_by: SpeechPackRankArg,

    /// Rendering backend.
    #[arg(long, value_enum, default_value_t = RenderBackendArg::Auto)]
    backend: RenderBackendArg,

    /// Parallel worker count (0 = auto).
    #[arg(long, default_value_t = 0)]
    jobs: usize,

    /// GPU post-FX drive.
    #[arg(long)]
    gpu_drive: Option<f64>,

    /// GPU post-FX bitcrush depth in bits.
    #[arg(long)]
    gpu_crush_bits: Option<f64>,

    /// GPU post-FX blend between dry and crushed signal (0..1).
    #[arg(long)]
    gpu_crush_mix: Option<f64>,

    /// Seed stride between adjacent profiles.
    #[arg(long, default_value_t = 1)]
    seed_stride: u64,

    #[command(flatten)]
    randomness: RandomnessArgs,
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

    /// Emit per-window ugliness timeline instead of whole-file analysis.
    #[arg(long)]
    timeline: bool,

    /// Timeline window length in milliseconds.
    #[arg(long, default_value_t = 50.0)]
    timeline_window_ms: f64,

    /// Timeline hop (step) between windows in milliseconds.
    #[arg(long, default_value_t = 25.0)]
    timeline_hop_ms: f64,

    /// Timeline output format.
    #[arg(long, value_enum, default_value_t = TimelineFormatArg::Json)]
    timeline_format: TimelineFormatArg,

    /// Write timeline to this file instead of stdout.
    #[arg(long)]
    timeline_output: Option<PathBuf>,
}

#[derive(Debug, Clone, Parser)]
struct MutateArgs {
    /// Input WAV file to mutate.
    input: PathBuf,

    /// Directory to write mutated variants and summary.
    #[arg(long, default_value = "out/mutate")]
    out_dir: PathBuf,

    /// Number of random mutations to generate.
    #[arg(long, default_value_t = 8)]
    count: usize,

    /// Minimum ugliness in Colbys (-1000..1000).
    #[arg(long, default_value_t = -200, allow_negative_numbers = true)]
    level_min: i32,

    /// Maximum ugliness in Colbys (-1000..1000).
    #[arg(long, default_value_t = 900, allow_negative_numbers = true)]
    level_max: i32,

    /// Optional seed for reproducible mutations.
    #[arg(long)]
    seed: Option<u64>,

    /// Analysis model for scoring mutations.
    #[arg(long, value_enum, default_value_t = AnalyzeModelArg::Psycho)]
    model: AnalyzeModelArg,

    /// FFT size for psycho model.
    #[arg(long, default_value_t = 2048)]
    fft_size: usize,

    /// Hop size for psycho model.
    #[arg(long, default_value_t = 512)]
    hop_size: usize,

    /// Normalize peak to this dBFS target.
    #[arg(long, default_value_t = -0.3)]
    normalize_dbfs: f64,

    #[command(flatten)]
    output_format: OutputFormatArgs,

    /// Rendering backend.
    #[arg(long, value_enum, default_value_t = RenderBackendArg::Auto)]
    backend: RenderBackendArg,

    /// Parallel worker count (0 = auto).
    #[arg(long, default_value_t = 0)]
    jobs: usize,
}

#[derive(Debug, Clone, Parser)]
struct NormalizePackArgs {
    /// Directory containing WAV files to normalize.
    #[arg(long)]
    in_dir: PathBuf,

    /// Directory to write normalized output files.
    #[arg(long, default_value = "out/normalized")]
    out_dir: PathBuf,

    /// Target ugliness in Colbys (-1000..1000).
    #[arg(long, default_value_t = 400, allow_negative_numbers = true)]
    level: i32,

    /// Go flavor to apply.
    #[arg(long = "type", value_enum)]
    flavor: Option<GoFlavorArg>,

    /// Optional seed.
    #[arg(long)]
    seed: Option<u64>,

    /// Analysis model for pre/post scoring.
    #[arg(long, value_enum, default_value_t = AnalyzeModelArg::Basic)]
    model: AnalyzeModelArg,

    /// Normalize peak to this dBFS target.
    #[arg(long, default_value_t = -0.3)]
    normalize_dbfs: f64,

    #[command(flatten)]
    output_format: OutputFormatArgs,

    /// Rendering backend.
    #[arg(long, value_enum, default_value_t = RenderBackendArg::Auto)]
    backend: RenderBackendArg,

    /// Parallel worker count (0 = auto).
    #[arg(long, default_value_t = 0)]
    jobs: usize,
}

#[derive(Debug, Clone, Parser)]
struct EvolveArgs {
    /// Output directory for all generation files and lineage.
    #[arg(long, default_value = "out/evolve")]
    out_dir: PathBuf,

    /// Render style to evolve (default: all styles rotated).
    #[arg(long, value_enum)]
    style: Option<StyleArg>,

    /// Duration in seconds per render.
    #[arg(short = 'd', long, default_value_t = 1.0)]
    duration: f64,

    /// Sample rate in Hz.
    #[arg(short = 'r', long, default_value_t = 44_100)]
    sample_rate: u32,

    /// Number of individuals per generation.
    #[arg(long, default_value_t = 8)]
    population: usize,

    /// Number of generations to run.
    #[arg(long, default_value_t = 5)]
    generations: usize,

    /// Analysis model for fitness scoring.
    #[arg(long, value_enum, default_value_t = AnalyzeModelArg::Psycho)]
    model: AnalyzeModelArg,

    /// FFT size for psycho model.
    #[arg(long, default_value_t = 2048)]
    fft_size: usize,

    /// Hop size for psycho model.
    #[arg(long, default_value_t = 512)]
    hop_size: usize,

    /// Base seed for generation 0.
    #[arg(long)]
    seed: Option<u64>,

    /// Output gain.
    #[arg(long, default_value_t = 0.8)]
    gain: f64,

    /// Rendering backend.
    #[arg(long, value_enum, default_value_t = RenderBackendArg::Auto)]
    backend: RenderBackendArg,

    /// Parallel worker count (0 = auto).
    #[arg(long, default_value_t = 0)]
    jobs: usize,

    #[command(flatten)]
    output_format: OutputFormatArgs,
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

    /// Target ugliness in Colbys (-1000..1000).
    #[arg(long, default_value_t = 400, allow_negative_numbers = true)]
    level: i32,

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

    #[command(flatten)]
    randomness: RandomnessArgs,
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

    /// Seed stride between adjacent styles.
    #[arg(long, default_value_t = 1)]
    seed_stride: u64,

    #[command(flatten)]
    randomness: RandomnessArgs,
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

    #[command(flatten)]
    randomness: RandomnessArgs,
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

    /// Seed stride between adjacent benchmark runs.
    #[arg(long, default_value_t = 1)]
    seed_stride: u64,

    #[command(flatten)]
    randomness: RandomnessArgs,
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

    /// Seed stride between adjacent marathon slots.
    #[arg(long, default_value_t = 1)]
    seed_stride: u64,

    #[command(flatten)]
    randomness: RandomnessArgs,
}

#[derive(Debug, Clone, Args)]
struct RandomnessArgs {
    /// Add a deterministic offset to the chosen seed before rendering.
    #[arg(long, default_value_t = 0)]
    seed_offset: u64,

    /// Mix an extra deterministic salt into the chosen seed.
    #[arg(long, default_value_t = 0)]
    seed_salt: u64,

    /// Re-mix the seed this many extra times.
    #[arg(long, default_value_t = 0)]
    seed_rerolls: u32,

    /// Master amount of parameter randomization (0 = off).
    #[arg(long, default_value_t = 0.0)]
    randomness: f64,

    /// How much randomized timing/pace variation to apply.
    #[arg(long, default_value_t = 1.0)]
    timing_randomness: f64,

    /// How much randomized pitch/formant/spectral variation to apply.
    #[arg(long, default_value_t = 1.0)]
    spectral_randomness: f64,

    /// How much randomized gain/drive/level variation to apply.
    #[arg(long, default_value_t = 1.0)]
    amplitude_randomness: f64,

    /// How much randomized density/probability variation to apply.
    #[arg(long, default_value_t = 1.0)]
    density_randomness: f64,
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
    Sine,
    Pulse,
    Triangle,
    Saw,
    Noise,
    Buzz,
    Formant,
    Vowel,
    Ring,
    Fold,
    Organ,
    Fm,
    Sync,
    Lfsr,
    Grain,
    Chirp,
    Subharmonic,
    Reed,
    Click,
    Comb,
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
enum TimelineFormatArg {
    Json,
    Csv,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SpeechPackRankArg {
    Ugliness,
    Intelligibility,
    Balanced,
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
            SpeechOscillatorArg::Sine => SpeechOscillator::Sine,
            SpeechOscillatorArg::Pulse => SpeechOscillator::Pulse,
            SpeechOscillatorArg::Triangle => SpeechOscillator::Triangle,
            SpeechOscillatorArg::Saw => SpeechOscillator::Saw,
            SpeechOscillatorArg::Noise => SpeechOscillator::Noise,
            SpeechOscillatorArg::Buzz => SpeechOscillator::Buzz,
            SpeechOscillatorArg::Formant => SpeechOscillator::Formant,
            SpeechOscillatorArg::Vowel => SpeechOscillator::Vowel,
            SpeechOscillatorArg::Ring => SpeechOscillator::Ring,
            SpeechOscillatorArg::Fold => SpeechOscillator::Fold,
            SpeechOscillatorArg::Organ => SpeechOscillator::Organ,
            SpeechOscillatorArg::Fm => SpeechOscillator::Fm,
            SpeechOscillatorArg::Sync => SpeechOscillator::Sync,
            SpeechOscillatorArg::Lfsr => SpeechOscillator::Lfsr,
            SpeechOscillatorArg::Grain => SpeechOscillator::Grain,
            SpeechOscillatorArg::Chirp => SpeechOscillator::Chirp,
            SpeechOscillatorArg::Subharmonic => SpeechOscillator::Subharmonic,
            SpeechOscillatorArg::Reed => SpeechOscillator::Reed,
            SpeechOscillatorArg::Click => SpeechOscillator::Click,
            SpeechOscillatorArg::Comb => SpeechOscillator::Comb,
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
        Command::Piece(args) => piece(args),
        Command::Speech(args) => speech(args),
        Command::SpeechPack(args) => speech_pack(args),
        Command::Analyze(args) => analyze(args),
        Command::RenderPack(args) => render_pack(args),
        Command::Go(args) => go(args),
        Command::Chain(args) => chain(args),
        Command::Styles => styles(),
        Command::Backends => backends(),
        Command::Presets(args) => presets(args),
        Command::Benchmark(args) => benchmark(args),
        Command::Marathon(args) => marathon(args),
        Command::Mutate(args) => mutate(args),
        Command::NormalizePack(args) => normalize_pack(args),
        Command::Evolve(args) => evolve(args),
    }
}

// ── mutate ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
struct MutateEntry {
    index: usize,
    flavor: String,
    level_co: i32,
    seed: u64,
    output: String,
    colbys: f64,
    ugly_delta: f64,
}

#[derive(Debug, Clone, Serialize)]
struct MutateSummary {
    generated_unix_s: u64,
    input: String,
    base_colbys: f64,
    model: String,
    count: usize,
    entries: Vec<MutateEntry>,
}

fn mutate(args: MutateArgs) -> Result<()> {
    let output_encoding = args.output_format.output_encoding()?;
    let engine = engine_from_args(args.backend, args.jobs, None, None, None);
    let plan = resolve_backend_plan(&engine)?;
    let analyze_opts = AnalyzeOptions {
        model: args.model.into(),
        fft_size: args.fft_size,
        hop_size: args.hop_size,
        joke: false,
    };

    fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("failed to create {}", args.out_dir.display()))?;

    let base_analysis = analyze_wav_with_options(&args.input, &analyze_opts)
        .with_context(|| format!("failed to analyze input {}", args.input.display()))?;
    let base_ugly = base_analysis.colbys;

    let base_seed = args.seed.unwrap_or_else(seed_from_time);
    let level_range = (args.level_max.max(args.level_min + 1) - args.level_min) as u64;
    let level_min_co = args.level_min.clamp(-1000, 1000);

    // Deterministic flavors (exclude Random/Lucky so every mutation is named)
    const FLAVORS: [GoFlavor; 7] = [
        GoFlavor::Glitch,
        GoFlavor::Stutter,
        GoFlavor::Puff,
        GoFlavor::Punish,
        GoFlavor::Geek,
        GoFlavor::DissonanceRing,
        GoFlavor::DissonanceExpand,
    ];

    let tasks: Vec<usize> = (0..args.count).collect();
    let pool = ThreadPoolBuilder::new()
        .num_threads(plan.jobs)
        .build()
        .context("failed to build mutate worker pool")?;

    let results: Vec<Result<MutateEntry>> = pool.install(|| {
        tasks
            .par_iter()
            .map(|&idx| {
                let mut rng = ChaCha8Rng::seed_from_u64(style_seed(base_seed, idx as u64));
                let flavor = FLAVORS[rng.gen_range(0..FLAVORS.len())];
                let level_co = level_min_co + (rng.gen_range(0..level_range) as i32);
                let seed = style_seed(base_seed, idx as u64 + 1000);
                let output = args.out_dir.join(format!(
                    "{:02}_{}_{}Co.wav",
                    idx + 1,
                    flavor.as_str(),
                    level_co
                ));
                go_ugly_file_with_engine_contour_encoding(
                    &args.input,
                    &output,
                    level_co,
                    Some(flavor),
                    Some(seed),
                    true,
                    args.normalize_dbfs,
                    None,
                    None,
                    output_encoding,
                    &engine,
                )
                .with_context(|| format!("failed to mutate to {}", output.display()))?;
                let analysis = analyze_wav_with_options(&output, &analyze_opts)
                    .with_context(|| format!("failed to analyze {}", output.display()))?;
                Ok(MutateEntry {
                    index: idx + 1,
                    flavor: flavor.as_str().to_string(),
                    level_co,
                    seed,
                    output: output.display().to_string(),
                    colbys: analysis.colbys,
                    ugly_delta: analysis.colbys - base_ugly,
                })
            })
            .collect()
    });

    let mut entries: Vec<MutateEntry> = results.into_iter().collect::<Result<_>>()?;
    entries.sort_by(|a, b| b.ugly_delta.total_cmp(&a.ugly_delta));

    let summary = MutateSummary {
        generated_unix_s: now_unix_s(),
        input: args.input.display().to_string(),
        base_colbys: base_ugly,
        model: analyze_opts.model.as_str().to_string(),
        count: entries.len(),
        entries,
    };

    let summary_path = args.out_dir.join("mutate_summary.json");
    write_json(&summary_path, &summary)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;

    println!(
        "Mutate: {} variants of {} (base ugly: {:.0} Co)",
        summary.count,
        args.input.display(),
        summary.base_colbys
    );
    println!("Top mutations by ugliness delta:");
    for e in summary.entries.iter().take(5) {
        println!(
            "  {:>2}. {:<16} l={:<4} {:>6.0} Co (Δ{:+.1})  {}",
            e.index, e.flavor, e.level_co, e.colbys, e.ugly_delta, e.output
        );
    }
    println!("Summary: {}", summary_path.display());
    Ok(())
}

// ── normalize-pack ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
struct NormalizePackEntry {
    filename: String,
    input: String,
    output: String,
    before_colbys: f64,
    after_colbys: f64,
    target_colbys: i32,
}

#[derive(Debug, Clone, Serialize)]
struct NormalizePackSummary {
    generated_unix_s: u64,
    in_dir: String,
    out_dir: String,
    target_colbys: i32,
    model: String,
    files_processed: usize,
    entries: Vec<NormalizePackEntry>,
}

fn normalize_pack(args: NormalizePackArgs) -> Result<()> {
    let output_encoding = args.output_format.output_encoding()?;
    let engine = engine_from_args(args.backend, args.jobs, None, None, None);
    let plan = resolve_backend_plan(&engine)?;
    let analyze_opts = AnalyzeOptions {
        model: args.model.into(),
        fft_size: 2048,
        hop_size: 512,
        joke: false,
    };

    fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("failed to create {}", args.out_dir.display()))?;

    let wavs: Vec<PathBuf> = fs::read_dir(&args.in_dir)
        .with_context(|| format!("failed to read {}", args.in_dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wav"))
        .collect();

    if wavs.is_empty() {
        println!("No .wav files found in {}", args.in_dir.display());
        return Ok(());
    }

    let base_seed = args.seed.unwrap_or_else(seed_from_time);
    let flavor: Option<GoFlavor> = args.flavor.map(Into::into);

    let pool = ThreadPoolBuilder::new()
        .num_threads(plan.jobs)
        .build()
        .context("failed to build normalize-pack worker pool")?;

    let results: Vec<Result<NormalizePackEntry>> = pool.install(|| {
        wavs.par_iter()
            .enumerate()
            .map(|(idx, input)| {
                let fname = input
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown.wav");
                let output = args.out_dir.join(fname);
                let seed = style_seed(base_seed, idx as u64);

                let before = analyze_wav_with_options(input, &analyze_opts)
                    .with_context(|| format!("failed to analyze {}", input.display()))?;

                go_ugly_file_with_engine_contour_encoding(
                    input,
                    &output,
                    args.level,
                    flavor,
                    Some(seed),
                    true,
                    args.normalize_dbfs,
                    None,
                    None,
                    output_encoding,
                    &engine,
                )
                .with_context(|| format!("failed to normalize {}", input.display()))?;

                let after = analyze_wav_with_options(&output, &analyze_opts)
                    .with_context(|| format!("failed to analyze {}", output.display()))?;

                Ok(NormalizePackEntry {
                    filename: fname.to_string(),
                    input: input.display().to_string(),
                    output: output.display().to_string(),
                    before_colbys: before.colbys,
                    after_colbys: after.colbys,
                    target_colbys: args.level,
                })
            })
            .collect()
    });

    let mut entries: Vec<NormalizePackEntry> = results.into_iter().collect::<Result<_>>()?;
    entries.sort_by(|a, b| a.filename.cmp(&b.filename));

    let summary = NormalizePackSummary {
        generated_unix_s: now_unix_s(),
        in_dir: args.in_dir.display().to_string(),
        out_dir: args.out_dir.display().to_string(),
        target_colbys: args.level,
        model: analyze_opts.model.as_str().to_string(),
        files_processed: entries.len(),
        entries,
    };

    let manifest_path = args.out_dir.join("normalize_manifest.json");
    write_json(&manifest_path, &summary)
        .with_context(|| format!("failed to write {}", manifest_path.display()))?;

    println!(
        "normalize-pack: {} files -> {} (target {} Co)",
        summary.files_processed,
        args.out_dir.display(),
        summary.target_colbys
    );
    for e in &summary.entries {
        println!(
            "  {} : {:.1} -> {:.0} Co",
            e.filename, e.before_colbys, e.after_colbys
        );
    }
    println!("Manifest: {}", manifest_path.display());
    Ok(())
}

// ── evolve ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
struct EvolveIndividual {
    generation: usize,
    index: usize,
    style: String,
    seed: u64,
    colbys: f64,
    output: String,
}

#[derive(Debug, Clone, Serialize)]
struct EvolveLineage {
    generations: usize,
    population: usize,
    champion_colbys: f64,
    champion_style: String,
    champion_seed: u64,
    champion_output: String,
    all_individuals: Vec<EvolveIndividual>,
}

fn evolve(args: EvolveArgs) -> Result<()> {
    let output_encoding = args.output_format.output_encoding()?;
    let engine = engine_from_args(args.backend, args.jobs, None, None, None);
    let plan = resolve_backend_plan(&engine)?;
    let analyze_opts = AnalyzeOptions {
        model: args.model.into(),
        fft_size: args.fft_size,
        hop_size: args.hop_size,
        joke: false,
    };

    fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("failed to create {}", args.out_dir.display()))?;

    let base_seed = args.seed.unwrap_or_else(seed_from_time);
    let styles: Vec<Style> = if let Some(s) = args.style {
        vec![s.into()]
    } else {
        Style::ALL.to_vec()
    };

    let pool = ThreadPoolBuilder::new()
        .num_threads(plan.jobs)
        .build()
        .context("failed to build evolve worker pool")?;

    let mut population: Vec<(Style, u64)> = (0..args.population)
        .map(|i| {
            let style = styles[i % styles.len()];
            let seed = style_seed(base_seed, i as u64);
            (style, seed)
        })
        .collect();

    let mut all_individuals: Vec<EvolveIndividual> = Vec::new();

    for generation in 0..args.generations {
        let gen_dir = args.out_dir.join(format!("gen{:02}", generation + 1));
        fs::create_dir_all(&gen_dir)?;

        let tasks: Vec<(usize, Style, u64)> = population
            .iter()
            .enumerate()
            .map(|(i, &(s, seed))| (i, s, seed))
            .collect();

        let mut scored: Vec<(usize, Style, u64, f64, String)> = pool.install(|| {
            tasks
                .par_iter()
                .map(|(idx, style, seed)| {
                    let output = gen_dir.join(format!(
                        "{:02}_{}_s{}.wav",
                        idx + 1,
                        style.as_str(),
                        seed % 99999
                    ));
                    let render_opts = RenderOptions {
                        duration: args.duration,
                        sample_rate: args.sample_rate,
                        seed: Some(*seed),
                        style: *style,
                        gain: args.gain,
                        normalize: true,
                        normalize_dbfs: -0.3,
                        output_encoding,
                    };
                    render_to_wav_with_engine(&output, &render_opts, &engine)
                        .with_context(|| format!("failed to render {}", output.display()))?;
                    let analysis = analyze_wav_with_options(&output, &analyze_opts)
                        .with_context(|| format!("failed to analyze {}", output.display()))?;
                    Ok((
                        *idx,
                        *style,
                        *seed,
                        analysis.colbys,
                        output.display().to_string(),
                    ))
                })
                .collect::<Vec<Result<_>>>()
                .into_iter()
                .collect::<Result<Vec<_>>>()
        })?;

        scored.sort_by(|a, b| b.3.total_cmp(&a.3));

        for (i, (idx, style, seed, ugly, output)) in scored.iter().enumerate() {
            all_individuals.push(EvolveIndividual {
                generation: generation + 1,
                index: idx + 1,
                style: style.as_str().to_string(),
                seed: *seed,
                colbys: *ugly,
                output: output.clone(),
            });
            println!(
                "gen{:02} {:>2}. {:<12} ugly={:.0} Co  seed={}",
                generation + 1,
                i + 1,
                style.as_str(),
                ugly,
                seed
            );
        }

        // Breed: keep top half, derive offspring by XOR-mutating seeds
        let elite_count = (args.population / 2).max(1);
        let elites: Vec<(Style, u64)> = scored[..elite_count]
            .iter()
            .map(|(_, s, seed, _, _)| (*s, *seed))
            .collect();

        let mut next_gen: Vec<(Style, u64)> = elites.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(style_seed(base_seed, generation as u64 + 9999));
        while next_gen.len() < args.population {
            let (parent_style, parent_seed) = elites[next_gen.len() % elites.len()];
            let mutation: u64 = rng.gen_range(1u64..=0xFFFF);
            let child_seed = parent_seed ^ mutation;
            // Occasionally mutate the style too
            let child_style = if rng.gen_bool(0.3) {
                styles[rng.gen_range(0..styles.len())]
            } else {
                parent_style
            };
            next_gen.push((child_style, child_seed));
        }
        population = next_gen;
    }

    let champion = all_individuals
        .iter()
        .max_by(|a, b| a.colbys.total_cmp(&b.colbys))
        .cloned()
        .expect("no individuals");

    let lineage = EvolveLineage {
        generations: args.generations,
        population: args.population,
        champion_colbys: champion.colbys,
        champion_style: champion.style.clone(),
        champion_seed: champion.seed,
        champion_output: champion.output.clone(),
        all_individuals,
    };

    let lineage_path = args.out_dir.join("lineage.json");
    write_json(&lineage_path, &lineage)
        .with_context(|| format!("failed to write {}", lineage_path.display()))?;

    println!(
        "\nChampion: {} seed={} ugly={:.0} Co",
        lineage.champion_style, lineage.champion_seed, lineage.champion_colbys
    );
    println!("  -> {}", lineage.champion_output);
    println!("Lineage: {}", lineage_path.display());
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
struct SpeechPackEntry {
    profile: String,
    seed: u64,
    output: String,
    colbys: f64,
    intelligibility: SpeechIntelligibility,
    analysis: AnalysisReport,
}

#[derive(Debug, Clone, Serialize)]
struct SpeechPackRankingEntry {
    rank: usize,
    profile: String,
    output: String,
    colbys: f64,
    intelligibility_index: f64,
    rank_score: f64,
    basic_colbys: f64,
    seed: u64,
}

#[derive(Debug, Clone, Serialize)]
struct SpeechPackSummary {
    generated_unix_s: u64,
    model: String,
    rank_by: String,
    text: String,
    sample_rate_hz: u32,
    backend_requested: String,
    backend_active: String,
    jobs: usize,
    base_seed: u64,
    profiles_rendered: usize,
    entries: Vec<SpeechPackEntry>,
    ranking: Vec<SpeechPackRankingEntry>,
}

fn speech_pack_rank_score(entry: &SpeechPackEntry, rank_by: SpeechPackRankArg) -> f64 {
    match rank_by {
        SpeechPackRankArg::Ugliness => entry.colbys,
        SpeechPackRankArg::Intelligibility => entry.intelligibility.intelligibility_index,
        SpeechPackRankArg::Balanced => {
            0.65 * entry.colbys + 0.35 * entry.intelligibility.intelligibility_index
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct PackEntry {
    style: String,
    seed: u64,
    output: String,
    colbys: f64,
    analysis: AnalysisReport,
}

#[derive(Debug, Clone, Serialize)]
struct PackRankingEntry {
    rank: usize,
    style: String,
    output: String,
    colbys: f64,
    seed: u64,
    basic_colbys: f64,
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

#[derive(Debug, Clone, Copy)]
struct RandomnessControls {
    seed_offset: u64,
    seed_salt: u64,
    seed_rerolls: u32,
    randomness: f64,
    timing_randomness: f64,
    spectral_randomness: f64,
    amplitude_randomness: f64,
    density_randomness: f64,
}

impl From<&RandomnessArgs> for RandomnessControls {
    fn from(value: &RandomnessArgs) -> Self {
        Self {
            seed_offset: value.seed_offset,
            seed_salt: value.seed_salt,
            seed_rerolls: value.seed_rerolls,
            randomness: value.randomness.max(0.0),
            timing_randomness: value.timing_randomness.max(0.0),
            spectral_randomness: value.spectral_randomness.max(0.0),
            amplitude_randomness: value.amplitude_randomness.max(0.0),
            density_randomness: value.density_randomness.max(0.0),
        }
    }
}

fn duration_for_slot(seed: u64, min_duration: f64, max_duration: f64) -> f64 {
    if (max_duration - min_duration).abs() <= f64::EPSILON {
        return min_duration;
    }
    let unit = ((seed >> 11) as f64) / ((1_u64 << 53) as f64);
    min_duration + (max_duration - min_duration) * unit.clamp(0.0, 1.0)
}

fn apply_seed_controls(seed: Option<u64>, randomness: RandomnessControls) -> u64 {
    let mut mixed = seed
        .unwrap_or_else(seed_from_time)
        .wrapping_add(randomness.seed_offset);
    mixed = style_seed(mixed, randomness.seed_salt);
    for reroll_idx in 0..randomness.seed_rerolls {
        mixed = style_seed(
            mixed,
            randomness.seed_salt ^ ((reroll_idx as u64 + 1).wrapping_mul(0x9E37_79B9)),
        );
    }
    mixed
}

fn derived_batch_seed(base_seed: u64, idx: u64, stride: u64) -> u64 {
    style_seed(base_seed, idx.wrapping_mul(stride.max(1)))
}

fn derived_seed_label(base_seed: u64, label: u64) -> u64 {
    style_seed(base_seed, label)
}

fn randomness_amount(master: f64, specific: f64) -> f64 {
    (master.max(0.0) * specific.max(0.0)).clamp(0.0, 1.0)
}

fn symmetric_factor(seed: u64, amount: f64, span: f64) -> f64 {
    if amount <= 0.0 {
        return 1.0;
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    1.0 + rng.gen_range(-span..=span) * amount
}

fn symmetric_add(seed: u64, amount: f64, span: f64) -> f64 {
    if amount <= 0.0 {
        return 0.0;
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.gen_range(-span..=span) * amount
}

fn randomize_duration(duration: f64, randomness: RandomnessControls, seed: u64) -> f64 {
    (duration
        * symmetric_factor(
            seed,
            randomness_amount(randomness.randomness, randomness.timing_randomness),
            0.35,
        ))
    .clamp(0.1, 86_400.0)
}

fn randomize_gain(gain: f64, randomness: RandomnessControls, seed: u64) -> f64 {
    (gain
        * symmetric_factor(
            seed,
            randomness_amount(randomness.randomness, randomness.amplitude_randomness),
            0.30,
        ))
    .clamp(0.0, 1.0)
}

fn randomize_normalize_dbfs(dbfs: f64, randomness: RandomnessControls, seed: u64) -> f64 {
    (dbfs
        + symmetric_add(
            seed,
            randomness_amount(randomness.randomness, randomness.amplitude_randomness),
            1.25,
        ))
    .clamp(-24.0, 0.0)
}

fn randomize_render_options(
    mut opts: RenderOptions,
    randomness: RandomnessControls,
) -> RenderOptions {
    let seed = opts.seed.unwrap_or_else(seed_from_time);
    opts.duration = randomize_duration(opts.duration, randomness, derived_seed_label(seed, 0x5101));
    opts.gain = randomize_gain(opts.gain, randomness, derived_seed_label(seed, 0x5102));
    opts.normalize_dbfs = randomize_normalize_dbfs(
        opts.normalize_dbfs,
        randomness,
        derived_seed_label(seed, 0x5103),
    );
    opts
}

fn randomize_speech_options(
    mut opts: SpeechRenderOptions,
    randomness: RandomnessControls,
) -> SpeechRenderOptions {
    let seed = opts.seed.unwrap_or_else(seed_from_time);
    let timing = randomness_amount(randomness.randomness, randomness.timing_randomness);
    let spectral = randomness_amount(randomness.randomness, randomness.spectral_randomness);
    let amplitude = randomness_amount(randomness.randomness, randomness.amplitude_randomness);
    let density = randomness_amount(randomness.randomness, randomness.density_randomness);

    opts.units_per_second = (opts.units_per_second
        * symmetric_factor(derived_seed_label(seed, 0x5201), timing, 0.45))
    .clamp(1.0, 40.0);
    opts.pitch_hz = (opts.pitch_hz
        * symmetric_factor(derived_seed_label(seed, 0x5202), spectral, 0.35))
    .clamp(20.0, 2400.0);
    opts.pitch_jitter = (opts.pitch_jitter
        + symmetric_add(derived_seed_label(seed, 0x5203), timing, 0.18))
    .clamp(0.0, 1.0);
    opts.vibrato_hz = (opts.vibrato_hz
        * symmetric_factor(derived_seed_label(seed, 0x5204), timing, 0.4))
    .clamp(0.0, 20.0);
    opts.vibrato_depth = (opts.vibrato_depth
        + symmetric_add(derived_seed_label(seed, 0x5205), timing, 0.12))
    .clamp(0.0, 1.0);
    opts.formant_shift = (opts.formant_shift
        + symmetric_add(derived_seed_label(seed, 0x5206), spectral, 0.6))
    .clamp(0.2, 3.5);
    opts.consonant_noise = (opts.consonant_noise
        + symmetric_add(derived_seed_label(seed, 0x5207), density, 0.35))
    .clamp(0.0, 2.0);
    opts.vowel_mix = (opts.vowel_mix
        + symmetric_add(derived_seed_label(seed, 0x5208), spectral, 0.28))
    .clamp(0.0, 1.5);
    opts.hiss = (opts.hiss + symmetric_add(derived_seed_label(seed, 0x5209), amplitude, 0.18))
        .clamp(0.0, 2.5);
    opts.buzz = (opts.buzz + symmetric_add(derived_seed_label(seed, 0x5210), amplitude, 0.35))
        .clamp(0.0, 2.5);
    opts.fold = (opts.fold * symmetric_factor(derived_seed_label(seed, 0x5211), spectral, 0.45))
        .clamp(0.5, 16.0);
    opts.chaos = (opts.chaos + symmetric_add(derived_seed_label(seed, 0x5212), density, 0.45))
        .clamp(0.0, 2.0);
    opts.robotize = (opts.robotize
        + symmetric_add(derived_seed_label(seed, 0x5213), spectral, 0.3))
    .clamp(0.0, 1.5);
    opts.glide =
        (opts.glide + symmetric_add(derived_seed_label(seed, 0x5214), timing, 0.2)).clamp(0.0, 1.5);
    opts.emphasis = (opts.emphasis
        + symmetric_add(derived_seed_label(seed, 0x5215), amplitude, 0.35))
    .clamp(0.0, 2.0);
    opts.word_accent = (opts.word_accent
        + symmetric_add(derived_seed_label(seed, 0x5216), timing, 0.25))
    .clamp(0.0, 1.0);
    opts.sentence_lilt = (opts.sentence_lilt
        + symmetric_add(derived_seed_label(seed, 0x5217), timing, 0.25))
    .clamp(0.0, 1.0);
    opts.paragraph_decline = (opts.paragraph_decline
        + symmetric_add(derived_seed_label(seed, 0x5218), timing, 0.2))
    .clamp(0.0, 1.0);
    opts.word_gap_ms = (opts.word_gap_ms
        * symmetric_factor(derived_seed_label(seed, 0x5219), timing, 0.5))
    .clamp(0.0, 500.0);
    opts.sentence_gap_ms = (opts.sentence_gap_ms
        * symmetric_factor(derived_seed_label(seed, 0x5220), timing, 0.6))
    .clamp(0.0, 1500.0);
    opts.paragraph_gap_ms = (opts.paragraph_gap_ms
        * symmetric_factor(derived_seed_label(seed, 0x5221), timing, 0.6))
    .clamp(0.0, 3000.0);
    opts.attack_ms = (opts.attack_ms
        * symmetric_factor(derived_seed_label(seed, 0x5222), timing, 0.45))
    .clamp(0.0, 200.0);
    opts.release_ms = (opts.release_ms
        * symmetric_factor(derived_seed_label(seed, 0x5223), timing, 0.45))
    .clamp(0.0, 500.0);
    opts.bitcrush_bits = (opts.bitcrush_bits
        + symmetric_add(derived_seed_label(seed, 0x5224), spectral, 2.0))
    .clamp(1.0, 24.0);
    opts.sample_hold_hz = (opts.sample_hold_hz
        * symmetric_factor(derived_seed_label(seed, 0x5225), density, 0.55))
    .clamp(500.0, 96_000.0);
    opts.ring_mix = (opts.ring_mix
        + symmetric_add(derived_seed_label(seed, 0x5226), amplitude, 0.18))
    .clamp(0.0, 1.0);
    opts.sub_mix = (opts.sub_mix
        + symmetric_add(derived_seed_label(seed, 0x5227), amplitude, 0.15))
    .clamp(0.0, 1.0);
    opts.nasal = (opts.nasal + symmetric_add(derived_seed_label(seed, 0x5228), spectral, 0.15))
        .clamp(0.0, 1.0);
    opts.throat = (opts.throat + symmetric_add(derived_seed_label(seed, 0x5229), amplitude, 0.15))
        .clamp(0.0, 1.0);
    opts.drift = (opts.drift + symmetric_add(derived_seed_label(seed, 0x5230), timing, 0.16))
        .clamp(0.0, 1.0);
    opts.resampler_grit = (opts.resampler_grit
        + symmetric_add(derived_seed_label(seed, 0x5231), density, 0.25))
    .clamp(0.0, 1.0);
    opts.gain = randomize_gain(opts.gain, randomness, derived_seed_label(seed, 0x5232));
    opts.normalize_dbfs = randomize_normalize_dbfs(
        opts.normalize_dbfs,
        randomness,
        derived_seed_label(seed, 0x5233),
    );
    opts
}

fn randomize_go_colbys(target_colbys: i32, randomness: RandomnessControls, seed: u64) -> i32 {
    let delta = symmetric_add(
        derived_seed_label(seed, 0x5301),
        randomness_amount(randomness.randomness, randomness.amplitude_randomness),
        320.0,
    );
    (target_colbys as f64 + delta)
        .round()
        .clamp(COLBYS_MIN as f64, COLBYS_MAX as f64) as i32
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
            let min_colbys = contour
                .points
                .iter()
                .map(|point| point.colbys)
                .min()
                .unwrap_or(first.colbys);
            let max_colbys = contour
                .points
                .iter()
                .map(|point| point.colbys)
                .max()
                .unwrap_or(first.colbys);
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
                    "{:?} points={} range={}..{} Co",
                    contour.interpolation,
                    contour.points.len(),
                    min_colbys,
                    max_colbys
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
    text.push_str("rank,style,colbys,basic_colbys,seed,output\n");
    for row in &summary.ranking {
        text.push_str(&format!(
            "{},{},{:.6},{:.6},{},{}\n",
            row.rank,
            csv_escape(&row.style),
            row.colbys,
            row.basic_colbys,
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
            row.colbys,
            row.basic_colbys,
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

fn write_speech_pack_csv(path: &Path, summary: &SpeechPackSummary) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut text = String::new();
    text.push_str(
        "rank,profile,colbys,intelligibility_index,rank_score,basic_colbys,seed,output\n",
    );
    for row in &summary.ranking {
        text.push_str(&format!(
            "{},{},{:.6},{:.6},{:.6},{:.6},{},{}\n",
            row.rank,
            csv_escape(&row.profile),
            row.colbys,
            row.intelligibility_index,
            row.rank_score,
            row.basic_colbys,
            row.seed,
            csv_escape(&row.output),
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

fn write_speech_pack_html(path: &Path, summary: &SpeechPackSummary) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut rows = String::new();
    for row in &summary.ranking {
        rows.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{:.1}</td><td>{:.1}</td><td>{:.1}</td><td>{:.1}</td><td>{}</td><td><audio controls preload=\"none\" src=\"{}\"></audio></td></tr>\n",
            row.rank,
            html_escape(&row.profile),
            row.colbys,
            row.intelligibility_index,
            row.rank_score,
            row.basic_colbys,
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
  <title>USG Speech Pack Report</title>
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
    .wrap {{ max-width: 1000px; margin: 0 auto; padding: 28px 18px 44px; }}
    .card {{
      background: linear-gradient(180deg, #1b2130, var(--panel));
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    h1 {{ margin: 0 0 6px; font-size: 1.5rem; }}
    p {{ margin: 4px 0; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 10px; }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: middle;
      font-size: 0.92rem;
    }}
    th {{ color: var(--muted); font-weight: 600; letter-spacing: 0.02em; }}
    td:nth-child(3) {{ color: var(--accent); font-weight: 700; }}
    audio {{ width: 220px; height: 30px; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"card\">
      <h1>UglySoundGenerator Speech Pack Report</h1>
      <p>Text: &ldquo;{text}&rdquo;</p>
      <p>Model: {model} | Ranking: {rank_by} | Profiles: {profiles_rendered} | SR: {sample_rate} Hz</p>
      <p>Backend: {backend_requested} -&gt; {backend_active} | Jobs: {jobs}</p>
      <p>Base seed: {base_seed}</p>
    </div>
    <div class=\"card\">
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Profile</th>
            <th>Ugliness</th>
            <th>Intelligibility</th>
            <th>Rank Score</th>
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
        text = html_escape(&summary.text),
        model = html_escape(&summary.model),
        rank_by = html_escape(&summary.rank_by),
        profiles_rendered = summary.profiles_rendered,
        sample_rate = summary.sample_rate_hz,
        backend_requested = html_escape(&summary.backend_requested),
        backend_active = html_escape(&summary.backend_active),
        jobs = summary.jobs,
        base_seed = summary.base_seed,
        rows = rows
    );

    fs::write(path, html)?;
    Ok(())
}
