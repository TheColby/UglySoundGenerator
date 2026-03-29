use std::f64::consts::PI;
use std::fmt;
use std::io::{Seek, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{fs, result};

use anyhow::{Result, anyhow};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex64};
use serde::{Deserialize, Serialize};

const EPS64: f64 = 1e-12;
const BARK_BANDS: usize = 24;
const MAX_RENDER_DURATION_S: f64 = 86_400.0;
const STREAM_RENDER_THRESHOLD_FRAMES: u64 = 192_000 * 120;
const STREAM_CHUNK_FRAMES: usize = 262_144;
pub const DEFAULT_GPU_DRIVE: f64 = 1.35;
pub const DEFAULT_GPU_CRUSH_BITS: f64 = 8.0;
pub const DEFAULT_GPU_CRUSH_MIX: f64 = 0.22;
const PAPER_BARK_BOUNDARIES_HZ: [f64; 26] = [
    20.0, 100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0,
    2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12_000.0,
    15_500.0, 22_050.0,
];

#[cfg(feature = "cuda")]
mod backend_cuda;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod backend_metal;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Style {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    Stutter,
    Pop,
    Crush,
    Gate,
    Smear,
    DissonanceRing,
    DissonanceExpand,
}

impl Style {
    pub const ALL: [Style; 15] = [
        Style::Harsh,
        Style::Digital,
        Style::Meltdown,
        Style::Glitch,
        Style::Pop,
        Style::Buzz,
        Style::Rub,
        Style::Hum,
        Style::Distort,
        Style::Spank,
        Style::Punish,
        Style::Steal,
        Style::Catastrophic,
        Style::Wink,
        Style::Lucky,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Style::Harsh => "harsh",
            Style::Digital => "digital",
            Style::Meltdown => "meltdown",
            Style::Glitch => "glitch",
            Style::Pop => "pop",
            Style::Buzz => "buzz",
            Style::Rub => "rub",
            Style::Hum => "hum",
            Style::Distort => "distort",
            Style::Spank => "spank",
            Style::Punish => "punish",
            Style::Steal => "steal",
            Style::Catastrophic => "catastrophic",
            Style::Wink => "wink",
            Style::Lucky => "lucky",
        }
    }
}

impl Effect {
    pub const ALL: [Effect; 7] = [
        Effect::Stutter,
        Effect::Pop,
        Effect::Crush,
        Effect::Gate,
        Effect::Smear,
        Effect::DissonanceRing,
        Effect::DissonanceExpand,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Effect::Stutter => "stutter",
            Effect::Pop => "pop",
            Effect::Crush => "crush",
            Effect::Gate => "gate",
            Effect::Smear => "smear",
            Effect::DissonanceRing => "dissonance-ring",
            Effect::DissonanceExpand => "dissonance-expand",
        }
    }
}

impl fmt::Display for Style {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl SpeechChipProfile {
    pub const ALL: [SpeechChipProfile; 8] = [
        SpeechChipProfile::VotraxSc01,
        SpeechChipProfile::Tms5220,
        SpeechChipProfile::Sp0256,
        SpeechChipProfile::Mea8000,
        SpeechChipProfile::S14001a,
        SpeechChipProfile::C64Sam,
        SpeechChipProfile::Arcadey90s,
        SpeechChipProfile::HandheldLcd,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            SpeechChipProfile::VotraxSc01 => "votrax-sc01",
            SpeechChipProfile::Tms5220 => "tms5220",
            SpeechChipProfile::Sp0256 => "sp0256",
            SpeechChipProfile::Mea8000 => "mea8000",
            SpeechChipProfile::S14001a => "s14001a",
            SpeechChipProfile::C64Sam => "c64-sam",
            SpeechChipProfile::Arcadey90s => "arcadey90s",
            SpeechChipProfile::HandheldLcd => "handheld-lcd",
        }
    }
}

impl fmt::Display for SpeechChipProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl SpeechOscillator {
    pub const ALL: [SpeechOscillator; 12] = [
        SpeechOscillator::Pulse,
        SpeechOscillator::Triangle,
        SpeechOscillator::Saw,
        SpeechOscillator::Noise,
        SpeechOscillator::Buzz,
        SpeechOscillator::Formant,
        SpeechOscillator::Ring,
        SpeechOscillator::Fold,
        SpeechOscillator::Koch,
        SpeechOscillator::Mandelbrot,
        SpeechOscillator::Strange,
        SpeechOscillator::Phoneme,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            SpeechOscillator::Pulse => "pulse",
            SpeechOscillator::Triangle => "triangle",
            SpeechOscillator::Saw => "saw",
            SpeechOscillator::Noise => "noise",
            SpeechOscillator::Buzz => "buzz",
            SpeechOscillator::Formant => "formant",
            SpeechOscillator::Ring => "ring",
            SpeechOscillator::Fold => "fold",
            SpeechOscillator::Koch => "koch",
            SpeechOscillator::Mandelbrot => "mandelbrot",
            SpeechOscillator::Strange => "strange",
            SpeechOscillator::Phoneme => "phoneme",
        }
    }
}

impl fmt::Display for SpeechOscillator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl SpeechInputMode {
    pub fn as_str(self) -> &'static str {
        match self {
            SpeechInputMode::Auto => "auto",
            SpeechInputMode::Character => "character",
            SpeechInputMode::Word => "word",
            SpeechInputMode::Sentence => "sentence",
            SpeechInputMode::Paragraph => "paragraph",
        }
    }
}

impl fmt::Display for SpeechInputMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainStage {
    Style(Style),
    Effect(Effect),
}

impl ChainStage {
    pub fn as_str(self) -> &'static str {
        match self {
            ChainStage::Style(s) => s.as_str(),
            ChainStage::Effect(e) => e.as_str(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalyzeModel {
    Basic,
    Psycho,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SpeechChipProfile {
    VotraxSc01,
    Tms5220,
    Sp0256,
    Mea8000,
    S14001a,
    C64Sam,
    Arcadey90s,
    HandheldLcd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SpeechOscillator {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SpeechInputMode {
    Auto,
    Character,
    Word,
    Sentence,
    Paragraph,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GoFlavor {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ContourInterpolation {
    Step,
    Linear,
}

impl Default for ContourInterpolation {
    fn default() -> Self {
        Self::Linear
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UglinessContourPoint {
    pub t: f64,
    pub level: u16,
}

fn default_contour_version() -> u16 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UglinessContour {
    #[serde(default = "default_contour_version")]
    pub version: u16,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub interpolation: ContourInterpolation,
    pub points: Vec<UglinessContourPoint>,
}

impl GoFlavor {
    pub fn as_str(self) -> &'static str {
        match self {
            GoFlavor::Glitch => "glitch",
            GoFlavor::Stutter => "stutter",
            GoFlavor::Puff => "puff",
            GoFlavor::Punish => "punish",
            GoFlavor::Geek => "geek",
            GoFlavor::DissonanceRing => "dissonance-ring",
            GoFlavor::DissonanceExpand => "dissonance-expand",
            GoFlavor::Random => "random",
            GoFlavor::Lucky => "lucky",
        }
    }
}

impl fmt::Display for GoFlavor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RenderBackend {
    Auto,
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CoordSystem {
    Cartesian,
    Polar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SurroundLayout {
    Mono,
    Stereo,
    Quad,
    FiveOne,
    SevenOne,
    Custom(u16),
}

impl SurroundLayout {
    pub fn as_str(self) -> String {
        match self {
            SurroundLayout::Mono => "mono".to_string(),
            SurroundLayout::Stereo => "stereo".to_string(),
            SurroundLayout::Quad => "quad".to_string(),
            SurroundLayout::FiveOne => "5.1".to_string(),
            SurroundLayout::SevenOne => "7.1".to_string(),
            SurroundLayout::Custom(n) => format!("custom:{n}"),
        }
    }

    pub fn channels(self) -> u16 {
        match self {
            SurroundLayout::Mono => 1,
            SurroundLayout::Stereo => 2,
            SurroundLayout::Quad => 4,
            SurroundLayout::FiveOne => 6,
            SurroundLayout::SevenOne => 8,
            SurroundLayout::Custom(n) => n.max(1),
        }
    }
}

impl fmt::Display for SurroundLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_str())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Trajectory {
    Static,
    Line { end: [f64; 3] },
    Orbit { radius: f64, turns: f64 },
}

#[derive(Debug, Clone, Copy)]
pub struct SpatialGoOptions {
    pub layout: SurroundLayout,
    pub locus_xyz: [f64; 3],
    pub trajectory: Trajectory,
}

impl RenderBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            RenderBackend::Auto => "auto",
            RenderBackend::Cpu => "cpu",
            RenderBackend::Metal => "metal",
            RenderBackend::Cuda => "cuda",
        }
    }
}

impl fmt::Display for RenderBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RenderEngine {
    pub backend: RenderBackend,
    pub jobs: usize,
    pub gpu_drive: f64,
    pub gpu_crush_bits: f64,
    pub gpu_crush_mix: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputEncoding {
    Float32,
    Int16,
    Int24,
    Int32,
}

impl OutputEncoding {
    pub fn as_str(self) -> &'static str {
        match self {
            OutputEncoding::Float32 => "float32",
            OutputEncoding::Int16 => "int16",
            OutputEncoding::Int24 => "int24",
            OutputEncoding::Int32 => "int32",
        }
    }

    pub fn bits_per_sample(self) -> u16 {
        match self {
            OutputEncoding::Float32 => 32,
            OutputEncoding::Int16 => 16,
            OutputEncoding::Int24 => 24,
            OutputEncoding::Int32 => 32,
        }
    }

    pub fn sample_format(self) -> SampleFormat {
        match self {
            OutputEncoding::Float32 => SampleFormat::Float,
            OutputEncoding::Int16 | OutputEncoding::Int24 | OutputEncoding::Int32 => {
                SampleFormat::Int
            }
        }
    }
}

impl Default for OutputEncoding {
    fn default() -> Self {
        Self::Float32
    }
}

impl fmt::Display for OutputEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Default for RenderEngine {
    fn default() -> Self {
        Self {
            backend: RenderBackend::Auto,
            jobs: default_jobs(),
            gpu_drive: DEFAULT_GPU_DRIVE,
            gpu_crush_bits: DEFAULT_GPU_CRUSH_BITS,
            gpu_crush_mix: DEFAULT_GPU_CRUSH_MIX,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BackendCapabilities {
    pub cpu: bool,
    pub metal: bool,
    pub cuda: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackendPlan {
    pub requested: RenderBackend,
    pub active: RenderBackend,
    pub jobs: usize,
    pub gpu_drive: f64,
    pub gpu_crush_bits: f64,
    pub gpu_crush_mix: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl AnalyzeModel {
    pub fn as_str(self) -> &'static str {
        match self {
            AnalyzeModel::Basic => "basic",
            AnalyzeModel::Psycho => "psycho",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnalyzeOptions {
    pub model: AnalyzeModel,
    pub fft_size: usize,
    pub hop_size: usize,
}

impl Default for AnalyzeOptions {
    fn default() -> Self {
        Self {
            model: AnalyzeModel::Basic,
            fft_size: 2048,
            hop_size: 512,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RenderOptions {
    pub duration: f64,
    pub sample_rate: u32,
    pub seed: Option<u64>,
    pub style: Style,
    pub gain: f64,
    pub normalize: bool,
    pub normalize_dbfs: f64,
    pub output_encoding: OutputEncoding,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            duration: 3.0,
            sample_rate: 192_000,
            seed: None,
            style: Style::Harsh,
            gain: 0.8,
            normalize: true,
            normalize_dbfs: 0.0,
            output_encoding: OutputEncoding::Float32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpeechRenderOptions {
    pub text: String,
    pub input_mode: SpeechInputMode,
    pub sample_rate: u32,
    pub seed: Option<u64>,
    pub chip_profile: SpeechChipProfile,
    pub primary_osc: SpeechOscillator,
    pub secondary_osc: SpeechOscillator,
    pub tertiary_osc: SpeechOscillator,
    pub gain: f64,
    pub normalize: bool,
    pub normalize_dbfs: f64,
    pub output_encoding: OutputEncoding,
    pub units_per_second: f64,
    pub pitch_hz: f64,
    pub pitch_jitter: f64,
    pub vibrato_hz: f64,
    pub vibrato_depth: f64,
    pub duty_cycle: f64,
    pub formant_shift: f64,
    pub consonant_noise: f64,
    pub vowel_mix: f64,
    pub hiss: f64,
    pub buzz: f64,
    pub fold: f64,
    pub chaos: f64,
    pub robotize: f64,
    pub glide: f64,
    pub monotone: f64,
    pub emphasis: f64,
    pub word_gap_ms: f64,
    pub sentence_gap_ms: f64,
    pub paragraph_gap_ms: f64,
    pub punctuation_gap_ms: f64,
    pub attack_ms: f64,
    pub release_ms: f64,
    pub bitcrush_bits: f64,
    pub sample_hold_hz: f64,
    pub ring_mix: f64,
    pub sub_mix: f64,
    pub nasal: f64,
    pub throat: f64,
    pub drift: f64,
    pub resampler_grit: f64,
}

impl Default for SpeechRenderOptions {
    fn default() -> Self {
        Self {
            text: "ugly speech generator".to_string(),
            input_mode: SpeechInputMode::Auto,
            sample_rate: 44_100,
            seed: None,
            chip_profile: SpeechChipProfile::Tms5220,
            primary_osc: SpeechOscillator::Phoneme,
            secondary_osc: SpeechOscillator::Buzz,
            tertiary_osc: SpeechOscillator::Koch,
            gain: 0.85,
            normalize: true,
            normalize_dbfs: -0.6,
            output_encoding: OutputEncoding::Float32,
            units_per_second: 11.0,
            pitch_hz: 118.0,
            pitch_jitter: 0.06,
            vibrato_hz: 5.0,
            vibrato_depth: 0.02,
            duty_cycle: 0.42,
            formant_shift: 1.0,
            consonant_noise: 0.4,
            vowel_mix: 0.7,
            hiss: 0.05,
            buzz: 0.2,
            fold: 2.4,
            chaos: 0.35,
            robotize: 0.25,
            glide: 0.12,
            monotone: 0.35,
            emphasis: 0.25,
            word_gap_ms: 42.0,
            sentence_gap_ms: 110.0,
            paragraph_gap_ms: 220.0,
            punctuation_gap_ms: 80.0,
            attack_ms: 6.0,
            release_ms: 20.0,
            bitcrush_bits: 7.5,
            sample_hold_hz: 9_600.0,
            ring_mix: 0.18,
            sub_mix: 0.12,
            nasal: 0.16,
            throat: 0.14,
            drift: 0.03,
            resampler_grit: 0.25,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RenderSummary {
    pub output: PathBuf,
    pub frames: usize,
    pub sample_rate: u32,
    pub style: Style,
    pub seed: u64,
    pub output_encoding: OutputEncoding,
    pub backend_requested: RenderBackend,
    pub backend_active: RenderBackend,
    pub jobs: usize,
}

#[derive(Debug, Clone)]
pub struct SpeechSummary {
    pub output: PathBuf,
    pub frames: usize,
    pub sample_rate: u32,
    pub chip_profile: SpeechChipProfile,
    pub input_mode: SpeechInputMode,
    pub text_len: usize,
    pub units_rendered: usize,
    pub seed: u64,
    pub output_encoding: OutputEncoding,
    pub backend_requested: RenderBackend,
    pub backend_active: RenderBackend,
    pub jobs: usize,
}

#[derive(Debug, Clone)]
pub struct GoSummary {
    pub output: PathBuf,
    pub frames: usize,
    pub sample_rate: u32,
    pub channels: u16,
    pub level: u16,
    pub flavor: GoFlavor,
    pub seed: u64,
    pub output_encoding: OutputEncoding,
    pub layout: Option<String>,
    pub backend_requested: RenderBackend,
    pub backend_active: RenderBackend,
    pub jobs: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Analysis {
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_s: f64,
    pub peak_dbfs: f64,
    pub rms_dbfs: f64,
    pub crest_factor_db: f64,
    pub zero_crossing_rate: f64,
    pub clipped_pct: f64,
    pub harshness_ratio: f64,
    pub ugly_index: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PsychoAnalysis {
    pub clip_norm: f64,
    pub harshness_norm: f64,
    pub roughness_norm: f64,
    pub sharpness_norm: f64,
    pub dissonance_norm: f64,
    pub transient_norm: f64,
    pub harmonicity_norm: f64,
    pub inharmonicity_norm: f64,
    pub binaural_beat_norm: f64,
    pub beat_conflict_norm: f64,
    pub tritone_tension_norm: f64,
    pub wolf_fifth_norm: f64,
    pub weighted_sum: f64,
    pub ugly_index: f64,
    pub fft_size: usize,
    pub hop_size: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnalysisReport {
    pub model: String,
    pub selected_ugly_index: f64,
    pub basic: Analysis,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub psycho: Option<PsychoAnalysis>,
}

pub fn available_styles() -> &'static [Style] {
    &Style::ALL
}

pub fn available_effects() -> &'static [Effect] {
    &Effect::ALL
}

pub fn available_speech_profiles() -> &'static [SpeechChipProfile] {
    &SpeechChipProfile::ALL
}

pub fn available_speech_oscillators() -> &'static [SpeechOscillator] {
    &SpeechOscillator::ALL
}

pub fn default_jobs() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

pub fn backend_capabilities() -> BackendCapabilities {
    BackendCapabilities {
        cpu: true,
        metal: metal_supported(),
        cuda: cuda_supported(),
    }
}

pub fn resolve_backend_plan(engine: &RenderEngine) -> Result<BackendPlan> {
    let caps = backend_capabilities();
    let jobs = engine.jobs.max(1);
    let gpu_drive = engine.gpu_drive.clamp(0.1, 16.0);
    let gpu_crush_bits = engine.gpu_crush_bits.clamp(0.0, 24.0);
    let gpu_crush_mix = engine.gpu_crush_mix.clamp(0.0, 1.0);

    let mk_plan =
        |requested: RenderBackend, active: RenderBackend, note: Option<String>| BackendPlan {
            requested,
            active,
            jobs,
            gpu_drive,
            gpu_crush_bits,
            gpu_crush_mix,
            note,
        };

    match engine.backend {
        RenderBackend::Cpu => Ok(mk_plan(RenderBackend::Cpu, RenderBackend::Cpu, None)),
        RenderBackend::Metal => {
            if caps.metal {
                Ok(mk_plan(
                    RenderBackend::Metal,
                    RenderBackend::Metal,
                    Some("Metal backend active".to_string()),
                ))
            } else {
                Err(anyhow!(
                    "metal backend requested, but this build/runtime does not support it (hint: macOS target + --features metal)"
                ))
            }
        }
        RenderBackend::Cuda => {
            if caps.cuda {
                Ok(mk_plan(
                    RenderBackend::Cuda,
                    RenderBackend::Cuda,
                    Some("CUDA backend active".to_string()),
                ))
            } else {
                Err(anyhow!(
                    "cuda backend requested, but this build/runtime does not support it (hint: build with --features cuda on a CUDA-capable host)"
                ))
            }
        }
        RenderBackend::Auto => {
            if caps.metal {
                Ok(mk_plan(
                    RenderBackend::Auto,
                    RenderBackend::Metal,
                    Some("auto-selected metal backend".to_string()),
                ))
            } else if caps.cuda {
                Ok(mk_plan(
                    RenderBackend::Auto,
                    RenderBackend::Cuda,
                    Some("auto-selected cuda backend".to_string()),
                ))
            } else {
                Ok(mk_plan(
                    RenderBackend::Auto,
                    RenderBackend::Cpu,
                    Some("auto-selected cpu backend".to_string()),
                ))
            }
        }
    }
}

fn metal_supported() -> bool {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        backend_metal::available()
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        false
    }
}

fn cuda_supported() -> bool {
    if std::env::var_os("USG_DISABLE_CUDA").is_some() {
        return false;
    }
    #[cfg(feature = "cuda")]
    {
        backend_cuda::available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

pub fn parse_style_name(name: &str) -> Option<Style> {
    match name.trim().to_ascii_lowercase().as_str() {
        "harsh" => Some(Style::Harsh),
        "digital" => Some(Style::Digital),
        "meltdown" => Some(Style::Meltdown),
        "glitch" => Some(Style::Glitch),
        "pop" => Some(Style::Pop),
        "buzz" => Some(Style::Buzz),
        "rub" => Some(Style::Rub),
        "hum" => Some(Style::Hum),
        "distort" => Some(Style::Distort),
        "spank" => Some(Style::Spank),
        "punish" => Some(Style::Punish),
        "steal" => Some(Style::Steal),
        "catastrophic" => Some(Style::Catastrophic),
        "wink" => Some(Style::Wink),
        "lucky" => Some(Style::Lucky),
        _ => None,
    }
}

pub fn parse_effect_name(name: &str) -> Option<Effect> {
    match name.trim().to_ascii_lowercase().as_str() {
        "stutter" => Some(Effect::Stutter),
        "pop" => Some(Effect::Pop),
        "crush" => Some(Effect::Crush),
        "gate" => Some(Effect::Gate),
        "smear" => Some(Effect::Smear),
        "dissonance-ring" | "spectral-ring" => Some(Effect::DissonanceRing),
        "dissonance-expand" | "dynamics-expand" => Some(Effect::DissonanceExpand),
        _ => None,
    }
}

pub fn parse_chain_stage(name: &str) -> Option<ChainStage> {
    let trimmed = name.trim();
    if let Some(rest) = trimmed.strip_prefix("style:") {
        return parse_style_name(rest).map(ChainStage::Style);
    }
    if let Some(rest) = trimmed.strip_prefix("effect:") {
        return parse_effect_name(rest).map(ChainStage::Effect);
    }
    if let Some(effect) = parse_effect_name(trimmed) {
        return Some(ChainStage::Effect(effect));
    }
    parse_style_name(trimmed).map(ChainStage::Style)
}

pub fn render_to_wav(output: &Path, opts: &RenderOptions) -> Result<RenderSummary> {
    render_to_wav_with_engine(output, opts, &RenderEngine::default())
}

pub fn render_to_wav_with_engine(
    output: &Path,
    opts: &RenderOptions,
    engine: &RenderEngine,
) -> Result<RenderSummary> {
    validate_render_options(opts)?;
    let plan = resolve_backend_plan(engine)?;
    let seed = opts.seed.unwrap_or_else(seed_from_time);
    let total_frames = duration_to_frames(opts.duration, opts.sample_rate)?;
    let frames_usize = usize::try_from(total_frames)
        .map_err(|_| anyhow!("requested render is too large for this platform"))?;

    if total_frames > STREAM_RENDER_THRESHOLD_FRAMES {
        render_to_wav_streaming(output, opts, &plan, seed, total_frames)?;
    } else {
        let mut samples = render_samples_with_plan(
            frames_usize,
            opts.sample_rate as f64,
            opts.style,
            opts.gain,
            seed,
            &plan,
        )?;
        if opts.normalize {
            normalize_peak_dbfs(&mut samples, opts.normalize_dbfs);
        }
        write_wav_mono(output, opts.sample_rate, &samples, opts.output_encoding)?;
    }
    Ok(RenderSummary {
        output: output.to_path_buf(),
        frames: frames_usize,
        sample_rate: opts.sample_rate,
        style: opts.style,
        seed,
        output_encoding: opts.output_encoding,
        backend_requested: plan.requested,
        backend_active: plan.active,
        jobs: plan.jobs,
    })
}

pub fn render_speech_to_wav(output: &Path, opts: &SpeechRenderOptions) -> Result<SpeechSummary> {
    render_speech_to_wav_with_engine(output, opts, &RenderEngine::default())
}

pub fn render_speech_to_wav_with_engine(
    output: &Path,
    opts: &SpeechRenderOptions,
    engine: &RenderEngine,
) -> Result<SpeechSummary> {
    validate_speech_options(opts)?;
    let plan = resolve_backend_plan(engine)?;
    let seed = opts.seed.unwrap_or_else(seed_from_time);
    let mut samples = render_speech_samples_with_plan(opts, seed, &plan)?;
    let frames = samples.len();
    if opts.normalize {
        normalize_peak_dbfs(&mut samples, opts.normalize_dbfs);
    }
    write_wav_mono(output, opts.sample_rate, &samples, opts.output_encoding)?;
    Ok(SpeechSummary {
        output: output.to_path_buf(),
        frames,
        sample_rate: opts.sample_rate,
        chip_profile: opts.chip_profile,
        input_mode: opts.input_mode,
        text_len: opts.text.chars().count(),
        units_rendered: speech_units_for_mode(&opts.text, opts.input_mode).len(),
        seed,
        output_encoding: opts.output_encoding,
        backend_requested: plan.requested,
        backend_active: plan.active,
        jobs: plan.jobs,
    })
}

pub fn render_samples(
    frames: usize,
    sample_rate: f64,
    style: Style,
    gain: f64,
    seed: u64,
) -> Result<Vec<f64>> {
    if !(0.0..=1.0).contains(&gain) {
        return Err(anyhow!("gain must be between 0.0 and 1.0"));
    }
    if !(8_000.0..=192_000.0).contains(&sample_rate) {
        return Err(anyhow!("sample rate must be between 8000 and 192000"));
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Ok(synth_ugly(frames, sample_rate, style, gain, &mut rng))
}

fn render_samples_with_plan(
    frames: usize,
    sample_rate: f64,
    style: Style,
    gain: f64,
    seed: u64,
    plan: &BackendPlan,
) -> Result<Vec<f64>> {
    let mut samples = render_samples(frames, sample_rate, style, gain, seed)?;
    apply_backend_post(samples.as_mut_slice(), plan)?;
    Ok(samples)
}

#[derive(Clone)]
struct SpeechUnit {
    ch: char,
    kind: SpeechSymbolKind,
    duration_s: f64,
    gap_s: f64,
    emphasis: f64,
}

fn render_speech_samples_with_plan(
    opts: &SpeechRenderOptions,
    seed: u64,
    plan: &BackendPlan,
) -> Result<Vec<f64>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let units = speech_units_for_mode(&opts.text, opts.input_mode);
    if units.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    let sample_rate = opts.sample_rate as f64;
    let tuning = speech_profile_tuning(opts.chip_profile);
    let mut voiced_phase = 0.0_f64;
    let mut secondary_phase = 0.0_f64;
    let mut tertiary_phase = 0.0_f64;
    let mut sample_hold_counter = 0.0_f64;
    let mut held = 0.0_f64;
    let mut koch_t = 0.0_f64;
    let mut mandel_t = 0.0_f64;
    let mut strange_a = StrangeAttractorState::new(0.13, -0.21);
    let mut strange_b = StrangeAttractorState::new(-0.33, 0.17);

    for (idx, unit) in units.iter().enumerate() {
        let phoneme = speech_symbol_params(unit.ch, unit.kind);
        let pace_scale = 11.0 / opts.units_per_second.max(1.0);
        let frames = (unit.duration_s * pace_scale * sample_rate).max(1.0) as usize;
        let attack_s = opts.attack_ms * 0.001;
        let release_s = opts.release_ms * 0.001;
        let unit_center = idx as f64 / units.len().max(1) as f64;
        let punct_emphasis = if matches!(
            unit.kind,
            SpeechSymbolKind::Punctuation | SpeechSymbolKind::ParagraphBreak
        ) {
            0.0
        } else {
            unit.emphasis
        };

        for frame in 0..frames {
            let t = frame as f64 / sample_rate;
            let progress = frame as f64 / frames.max(1) as f64;
            let attack = if attack_s > 0.0 {
                (t / attack_s).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let release = if release_s > 0.0 {
                ((unit.duration_s - t) / release_s).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let env = attack.min(release);
            let vibrato = 1.0
                + opts.vibrato_depth
                    * (2.0 * PI * opts.vibrato_hz * (idx as f64 * unit.duration_s + t)).sin();
            let jitter = 1.0 + opts.pitch_jitter * rng.gen_range(-1.0_f64..1.0_f64) * 0.25;
            let drift =
                1.0 + opts.drift * ((2.0 * PI * 0.17 * unit_center).sin() * 0.5 + progress - 0.5);
            let monotone_mix = 1.0 - opts.monotone * (phoneme.pitch_mul - 1.0);
            let pitch = opts.pitch_hz
                * tuning.pitch_mul
                * phoneme.pitch_mul
                * vibrato
                * jitter
                * drift
                * monotone_mix.max(0.2);
            let target_formant1 = phoneme.formant1 * opts.formant_shift * tuning.formant_mul;
            let target_formant2 = phoneme.formant2 * opts.formant_shift * tuning.formant_mul;
            let intensity = (1.0 + opts.emphasis * punct_emphasis).max(0.2);
            let primary = speech_oscillator_sample(
                opts.primary_osc,
                pitch,
                target_formant1,
                target_formant2,
                phoneme.voiced,
                opts.duty_cycle,
                opts.chaos + phoneme.brightness * 0.2,
                opts.fold,
                t + idx as f64 * unit.duration_s,
                &mut voiced_phase,
                &mut koch_t,
                &mut mandel_t,
                &mut strange_a,
                &mut rng,
            );
            let secondary = speech_oscillator_sample(
                opts.secondary_osc,
                pitch * (1.0 + 0.17 * opts.glide),
                target_formant1 * 1.08,
                target_formant2 * 0.94,
                phoneme.voiced,
                (opts.duty_cycle * 0.91).clamp(0.05, 0.95),
                opts.chaos + 0.1,
                opts.fold + 0.6,
                t,
                &mut secondary_phase,
                &mut koch_t,
                &mut mandel_t,
                &mut strange_b,
                &mut rng,
            );
            let tertiary = speech_oscillator_sample(
                opts.tertiary_osc,
                pitch * 0.5,
                target_formant1 * 0.5,
                target_formant2 * 0.65,
                true,
                0.5,
                opts.chaos + 0.2,
                opts.fold + 1.2,
                t,
                &mut tertiary_phase,
                &mut koch_t,
                &mut mandel_t,
                &mut strange_a,
                &mut rng,
            );
            let noise = if phoneme.noisy {
                rng.gen_range(-1.0_f64..1.0_f64)
                    * opts.consonant_noise
                    * phoneme.noise_mul
                    * tuning.noise_mul
            } else {
                0.0
            };
            let formant_bed = opts.vowel_mix
                * phoneme.vowel_mul
                * ((2.0 * PI * target_formant1 * t).sin() * 0.6
                    + (2.0 * PI * target_formant2 * t).sin() * 0.4);
            let ring = opts.ring_mix * primary * secondary;
            let sub = opts.sub_mix * (2.0 * PI * pitch * 0.5 * t).sin();
            let nasal = opts.nasal * (2.0 * PI * target_formant1 * 0.42 * t).sin();
            let throat = opts.throat * saw((pitch * 0.25).max(20.0), t);
            let robot = opts.robotize
                * (2.0 * PI * (tuning.robot_carrier_hz + 40.0 * phoneme.brightness) * t).sin()
                * env;
            let voiced_gate = if phoneme.voiced { 1.0 } else { 0.55 };
            let mut sample = env
                * intensity
                * voiced_gate
                * (0.62 * primary
                    + 0.24 * secondary
                    + 0.14 * tertiary
                    + formant_bed
                    + ring
                    + sub
                    + nasal
                    + throat
                    + robot)
                + noise
                + rng.gen_range(-1.0_f64..1.0_f64) * opts.hiss * tuning.hiss_mul;
            sample += opts.buzz * tuning.buzz_mul * square((pitch * 0.5).max(20.0), t) * env * 0.2;
            sample =
                apply_speech_chip_fx(sample, opts, &tuning, &mut sample_hold_counter, &mut held);
            out.push(sample);
        }

        let gap_ms = match unit.kind {
            SpeechSymbolKind::Whitespace => opts.word_gap_ms,
            SpeechSymbolKind::Punctuation => {
                opts.punctuation_gap_ms
                    + if matches!(unit.ch, '.' | '!' | '?') {
                        opts.sentence_gap_ms
                    } else {
                        0.0
                    }
            }
            SpeechSymbolKind::ParagraphBreak => opts.paragraph_gap_ms,
            _ => unit.gap_s * 1000.0,
        };
        let gap_frames = (gap_ms * 0.001 * sample_rate).round() as usize;
        out.extend(std::iter::repeat_n(0.0, gap_frames));
    }

    apply_backend_post(out.as_mut_slice(), plan)?;
    Ok(out)
}

fn render_to_wav_streaming(
    output: &Path,
    opts: &RenderOptions,
    plan: &BackendPlan,
    seed: u64,
    total_frames: u64,
) -> Result<()> {
    create_parent_dir(output)?;
    let sample_rate = opts.sample_rate as f64;
    let target_peak = 10.0_f64.powf(opts.normalize_dbfs / 20.0).abs();
    let mut global_peak = 0.0_f64;

    if opts.normalize {
        let mut frame_cursor = 0_u64;
        let mut chunk_idx = 0_u64;
        while frame_cursor < total_frames {
            let frames = (total_frames - frame_cursor).min(STREAM_CHUNK_FRAMES as u64) as usize;
            let mut chunk = render_samples_with_plan(
                frames,
                sample_rate,
                opts.style,
                opts.gain,
                derive_seed(seed, chunk_idx),
                plan,
            )?;
            if !chunk.is_empty() {
                let peak = chunk.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
                if peak > global_peak {
                    global_peak = peak;
                }
            }
            // Release chunk memory aggressively between passes.
            chunk.clear();
            frame_cursor += frames as u64;
            chunk_idx += 1;
        }
    }

    let scale = if opts.normalize && global_peak > EPS64 && target_peak > EPS64 {
        target_peak / global_peak
    } else {
        1.0
    };

    let mut writer =
        WavWriter::create(output, wav_spec(1, opts.sample_rate, opts.output_encoding))?;

    let mut frame_cursor = 0_u64;
    let mut chunk_idx = 0_u64;
    while frame_cursor < total_frames {
        let frames = (total_frames - frame_cursor).min(STREAM_CHUNK_FRAMES as u64) as usize;
        let mut chunk = render_samples_with_plan(
            frames,
            sample_rate,
            opts.style,
            opts.gain,
            derive_seed(seed, chunk_idx),
            plan,
        )?;
        if scale != 1.0 {
            for x in &mut chunk {
                *x *= scale;
            }
        }
        for sample in chunk {
            write_encoded_sample(&mut writer, sample, opts.output_encoding)?;
        }
        frame_cursor += frames as u64;
        chunk_idx += 1;
    }
    writer.finalize()?;
    Ok(())
}

fn duration_to_frames(duration: f64, sample_rate: u32) -> Result<u64> {
    if !duration.is_finite() {
        return Err(anyhow!("duration must be finite"));
    }
    let frames_f64 = (duration * sample_rate as f64).floor();
    if frames_f64 <= 0.0 {
        return Err(anyhow!("duration is too small for at least one sample"));
    }
    if frames_f64 > u64::MAX as f64 {
        return Err(anyhow!("requested duration is too large"));
    }
    Ok(frames_f64 as u64)
}

pub fn render_chain(
    stages: &[ChainStage],
    duration_s: f64,
    sample_rate: u32,
    gain: f64,
    normalize: bool,
    normalize_dbfs: f64,
    base_seed: u64,
) -> Result<Vec<f64>> {
    render_chain_with_engine(
        stages,
        duration_s,
        sample_rate,
        gain,
        normalize,
        normalize_dbfs,
        base_seed,
        &RenderEngine::default(),
    )
}

pub fn render_chain_with_engine(
    stages: &[ChainStage],
    duration_s: f64,
    sample_rate: u32,
    gain: f64,
    normalize: bool,
    normalize_dbfs: f64,
    base_seed: u64,
    engine: &RenderEngine,
) -> Result<Vec<f64>> {
    if stages.is_empty() {
        return Err(anyhow!("at least one stage is required"));
    }
    let plan = resolve_backend_plan(engine)?;
    let render_opts = RenderOptions {
        duration: duration_s,
        sample_rate,
        seed: Some(base_seed),
        style: Style::Harsh,
        gain,
        normalize,
        normalize_dbfs,
        output_encoding: OutputEncoding::Float32,
    };
    validate_render_options(&render_opts)?;
    let total_frames = duration_to_frames(duration_s, sample_rate)?;
    let frames = usize::try_from(total_frames)
        .map_err(|_| anyhow!("requested chain render is too large for this platform"))?;
    let mut out = render_chain_chunk_with_plan(
        stages,
        frames,
        sample_rate as f64,
        gain,
        base_seed,
        0,
        0,
        &plan,
    )?;
    if normalize {
        normalize_peak_dbfs(&mut out, normalize_dbfs);
    }
    Ok(out)
}

pub fn render_chain_to_wav_with_engine(
    output: &Path,
    stages: &[ChainStage],
    duration_s: f64,
    sample_rate: u32,
    output_encoding: OutputEncoding,
    gain: f64,
    normalize: bool,
    normalize_dbfs: f64,
    base_seed: u64,
    engine: &RenderEngine,
) -> Result<usize> {
    if stages.is_empty() {
        return Err(anyhow!("at least one stage is required"));
    }
    let plan = resolve_backend_plan(engine)?;
    let render_opts = RenderOptions {
        duration: duration_s,
        sample_rate,
        seed: Some(base_seed),
        style: Style::Harsh,
        gain,
        normalize,
        normalize_dbfs,
        output_encoding,
    };
    validate_render_options(&render_opts)?;
    let total_frames = duration_to_frames(duration_s, sample_rate)?;
    let frames = usize::try_from(total_frames)
        .map_err(|_| anyhow!("requested chain render is too large for this platform"))?;

    if total_frames > STREAM_RENDER_THRESHOLD_FRAMES {
        render_chain_to_wav_streaming(
            output,
            stages,
            sample_rate,
            output_encoding,
            gain,
            normalize,
            normalize_dbfs,
            base_seed,
            &plan,
            total_frames,
        )?;
    } else {
        let mut out = render_chain_chunk_with_plan(
            stages,
            frames,
            sample_rate as f64,
            gain,
            base_seed,
            0,
            0,
            &plan,
        )?;
        if normalize {
            normalize_peak_dbfs(&mut out, normalize_dbfs);
        }
        write_wav_mono(output, sample_rate, &out, output_encoding)?;
    }
    Ok(frames)
}

fn render_chain_to_wav_streaming(
    output: &Path,
    stages: &[ChainStage],
    sample_rate: u32,
    output_encoding: OutputEncoding,
    gain: f64,
    normalize: bool,
    normalize_dbfs: f64,
    base_seed: u64,
    plan: &BackendPlan,
    total_frames: u64,
) -> Result<()> {
    create_parent_dir(output)?;
    let sample_rate_f64 = sample_rate as f64;
    let target_peak = 10.0_f64.powf(normalize_dbfs / 20.0).abs();
    let mut global_peak = 0.0_f64;

    if normalize {
        let mut frame_cursor = 0_u64;
        let mut chunk_idx = 0_u64;
        while frame_cursor < total_frames {
            let frames = (total_frames - frame_cursor).min(STREAM_CHUNK_FRAMES as u64) as usize;
            let chunk = render_chain_chunk_with_plan(
                stages,
                frames,
                sample_rate_f64,
                gain,
                base_seed,
                chunk_idx,
                frame_cursor,
                plan,
            )?;
            let peak = chunk.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
            global_peak = global_peak.max(peak);
            frame_cursor += frames as u64;
            chunk_idx += 1;
        }
    }

    let scale = if normalize && global_peak > EPS64 && target_peak > EPS64 {
        target_peak / global_peak
    } else {
        1.0
    };
    let mut writer = WavWriter::create(output, wav_spec(1, sample_rate, output_encoding))?;

    let mut frame_cursor = 0_u64;
    let mut chunk_idx = 0_u64;
    while frame_cursor < total_frames {
        let frames = (total_frames - frame_cursor).min(STREAM_CHUNK_FRAMES as u64) as usize;
        let mut chunk = render_chain_chunk_with_plan(
            stages,
            frames,
            sample_rate_f64,
            gain,
            base_seed,
            chunk_idx,
            frame_cursor,
            plan,
        )?;
        if scale != 1.0 {
            for x in &mut chunk {
                *x *= scale;
            }
        }
        for s in chunk {
            writer.write_sample(s.clamp(-1.0, 1.0) as f32)?;
        }
        frame_cursor += frames as u64;
        chunk_idx += 1;
    }
    writer.finalize()?;
    Ok(())
}

fn render_chain_chunk_with_plan(
    stages: &[ChainStage],
    frames: usize,
    sample_rate: f64,
    gain: f64,
    base_seed: u64,
    chunk_idx: u64,
    frame_offset: u64,
    plan: &BackendPlan,
) -> Result<Vec<f64>> {
    let mut current: Option<Vec<f64>> = None;
    let style_jobs = plan.jobs.max(1);
    let style_specs: Vec<(usize, Style, u64)> = stages
        .iter()
        .enumerate()
        .filter_map(|(idx, stage)| match stage {
            ChainStage::Style(style) => {
                Some((idx, *style, stage_chunk_seed(base_seed, idx, chunk_idx)))
            }
            ChainStage::Effect(_) => None,
        })
        .collect();
    let mut style_layers = render_style_layers(&style_specs, frames, sample_rate, style_jobs)?;

    for (idx, stage) in stages.iter().enumerate() {
        match stage {
            ChainStage::Style(_) => {
                let layer = style_layers
                    .remove(&idx)
                    .ok_or_else(|| anyhow!("internal error: missing pre-rendered style layer"))?;
                if current.is_none() {
                    current = Some(layer);
                } else if let Some(buf) = current.as_mut()
                    && let ChainStage::Style(style) = stage
                {
                    apply_style_layer(buf, &layer, *style);
                }
            }
            ChainStage::Effect(effect) => {
                let stage_seed = stage_chunk_seed(base_seed, idx, chunk_idx);
                let buf = current.get_or_insert_with(|| vec![0.0; frames]);
                apply_effect(buf, *effect, sample_rate, stage_seed, frame_offset);
            }
        }
    }

    let mut out = current.unwrap_or_else(|| vec![0.0; frames]);
    for x in &mut out {
        *x = soft_clip(*x * gain * 1.2);
    }
    apply_backend_post(out.as_mut_slice(), plan)?;
    Ok(out)
}

fn stage_chunk_seed(base_seed: u64, stage_idx: usize, chunk_idx: u64) -> u64 {
    let chunk_seed = derive_seed(base_seed, chunk_idx.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    derive_seed(chunk_seed, stage_idx as u64)
}

pub fn write_samples_to_wav(path: &Path, sample_rate: u32, samples: &[f64]) -> Result<()> {
    write_wav_mono(path, sample_rate, samples, OutputEncoding::Float32)
}

pub fn go_ugly_file(
    input: &Path,
    output: &Path,
    level: u16,
    flavor: Option<GoFlavor>,
    seed: Option<u64>,
    normalize: bool,
    normalize_dbfs: f64,
) -> Result<GoSummary> {
    go_ugly_file_with_engine_contour(
        input,
        output,
        level,
        flavor,
        seed,
        normalize,
        normalize_dbfs,
        None,
        &RenderEngine::default(),
    )
}

pub fn go_ugly_file_with_engine(
    input: &Path,
    output: &Path,
    level: u16,
    flavor: Option<GoFlavor>,
    seed: Option<u64>,
    normalize: bool,
    normalize_dbfs: f64,
    engine: &RenderEngine,
) -> Result<GoSummary> {
    go_ugly_file_with_engine_contour(
        input,
        output,
        level,
        flavor,
        seed,
        normalize,
        normalize_dbfs,
        None,
        engine,
    )
}

pub fn go_ugly_file_with_engine_contour(
    input: &Path,
    output: &Path,
    level: u16,
    flavor: Option<GoFlavor>,
    seed: Option<u64>,
    normalize: bool,
    normalize_dbfs: f64,
    contour: Option<&UglinessContour>,
    engine: &RenderEngine,
) -> Result<GoSummary> {
    go_ugly_file_with_engine_contour_encoding(
        input,
        output,
        level,
        flavor,
        seed,
        normalize,
        normalize_dbfs,
        contour,
        None,
        OutputEncoding::Float32,
        engine,
    )
}

pub fn go_ugly_file_with_engine_contour_encoding(
    input: &Path,
    output: &Path,
    level: u16,
    flavor: Option<GoFlavor>,
    seed: Option<u64>,
    normalize: bool,
    normalize_dbfs: f64,
    contour: Option<&UglinessContour>,
    output_sample_rate: Option<u32>,
    output_encoding: OutputEncoding,
    engine: &RenderEngine,
) -> Result<GoSummary> {
    if let Some(c) = contour {
        validate_ugliness_contour(c)?;
    }

    go_ugly_file_with_engine_impl(
        input,
        output,
        level,
        flavor,
        seed,
        normalize,
        normalize_dbfs,
        engine,
        contour,
        output_sample_rate,
        output_encoding,
    )
}

fn go_ugly_file_with_engine_impl(
    input: &Path,
    output: &Path,
    level: u16,
    flavor: Option<GoFlavor>,
    seed: Option<u64>,
    normalize: bool,
    normalize_dbfs: f64,
    engine: &RenderEngine,
    contour: Option<&UglinessContour>,
    output_sample_rate: Option<u32>,
    output_encoding: OutputEncoding,
) -> Result<GoSummary> {
    if !(1..=1000).contains(&level) {
        return Err(anyhow!("level must be between 1 and 1000"));
    }
    if normalize_dbfs > 0.0 {
        return Err(anyhow!("normalize-dbfs must be <= 0.0"));
    }

    let (mut channels, input_sample_rate, channel_count) = read_wav_channels_f64(input)?;
    if channels.is_empty() || channels[0].is_empty() {
        return Err(anyhow!("input had no audio samples"));
    }
    let frames = channels.iter().map(|c| c.len()).min().unwrap_or(0);
    if frames == 0 {
        return Err(anyhow!("input had no audio samples"));
    }
    for ch in &mut channels {
        ch.truncate(frames);
    }

    let sample_rate = output_sample_rate.unwrap_or(input_sample_rate);
    if sample_rate != input_sample_rate {
        channels = resample_channels_linear(&channels, input_sample_rate, sample_rate);
    }

    let plan = resolve_backend_plan(engine)?;
    let base_seed = seed.unwrap_or_else(seed_from_time);
    let chosen = flavor.unwrap_or(GoFlavor::Random);
    let sample_rate_f64 = sample_rate as f64;

    for (ch_idx, channel) in channels.iter_mut().enumerate() {
        let ch_seed = derive_seed(base_seed, ch_idx as u64);
        apply_go_ugly_to_channel(channel, sample_rate_f64, level, chosen, ch_seed, contour);
        apply_backend_post(channel.as_mut_slice(), &plan)?;
    }

    if normalize {
        normalize_peak_dbfs_channels(&mut channels, normalize_dbfs);
    }
    write_wav_channels(output, sample_rate, &channels, output_encoding)?;

    Ok(GoSummary {
        output: output.to_path_buf(),
        frames,
        sample_rate,
        channels: channel_count,
        level,
        flavor: chosen,
        seed: base_seed,
        output_encoding,
        layout: None,
        backend_requested: plan.requested,
        backend_active: plan.active,
        jobs: plan.jobs,
    })
}

pub fn go_ugly_upmix_file_with_engine(
    input: &Path,
    output: &Path,
    level: u16,
    flavor: Option<GoFlavor>,
    seed: Option<u64>,
    normalize: bool,
    normalize_dbfs: f64,
    spatial: SpatialGoOptions,
    engine: &RenderEngine,
) -> Result<GoSummary> {
    go_ugly_upmix_file_with_engine_contour(
        input,
        output,
        level,
        flavor,
        seed,
        normalize,
        normalize_dbfs,
        spatial,
        None,
        engine,
    )
}

pub fn go_ugly_upmix_file_with_engine_contour(
    input: &Path,
    output: &Path,
    level: u16,
    flavor: Option<GoFlavor>,
    seed: Option<u64>,
    normalize: bool,
    normalize_dbfs: f64,
    spatial: SpatialGoOptions,
    contour: Option<&UglinessContour>,
    engine: &RenderEngine,
) -> Result<GoSummary> {
    go_ugly_upmix_file_with_engine_contour_encoding(
        input,
        output,
        level,
        flavor,
        seed,
        normalize,
        normalize_dbfs,
        spatial,
        contour,
        None,
        OutputEncoding::Float32,
        engine,
    )
}

pub fn go_ugly_upmix_file_with_engine_contour_encoding(
    input: &Path,
    output: &Path,
    level: u16,
    flavor: Option<GoFlavor>,
    seed: Option<u64>,
    normalize: bool,
    normalize_dbfs: f64,
    spatial: SpatialGoOptions,
    contour: Option<&UglinessContour>,
    output_sample_rate: Option<u32>,
    output_encoding: OutputEncoding,
    engine: &RenderEngine,
) -> Result<GoSummary> {
    if let Some(c) = contour {
        validate_ugliness_contour(c)?;
    }

    if !(1..=1000).contains(&level) {
        return Err(anyhow!("level must be between 1 and 1000"));
    }
    if normalize_dbfs > 0.0 {
        return Err(anyhow!("normalize-dbfs must be <= 0.0"));
    }

    let (channel_data, input_sample_rate, _channels) = read_wav_channels_f64(input)?;
    if channel_data.is_empty() || channel_data[0].is_empty() {
        return Err(anyhow!("input had no audio samples"));
    }
    let mut mono = mixdown_mono(&channel_data);
    if mono.is_empty() {
        return Err(anyhow!("input had no audio samples"));
    }

    let sample_rate = output_sample_rate.unwrap_or(input_sample_rate);
    if sample_rate != input_sample_rate {
        mono = resample_linear(&mono, input_sample_rate, sample_rate);
    }

    let plan = resolve_backend_plan(engine)?;
    let base_seed = seed.unwrap_or_else(seed_from_time);
    let chosen = flavor.unwrap_or(GoFlavor::Random);
    apply_go_ugly_to_channel(
        mono.as_mut_slice(),
        sample_rate as f64,
        level,
        chosen,
        base_seed,
        contour,
    );

    let mut channels_out = spatialize_mono(
        &mono,
        sample_rate as f64,
        spatial.layout,
        spatial.locus_xyz,
        spatial.trajectory,
    );
    for (idx, ch) in channels_out.iter_mut().enumerate() {
        apply_backend_post(ch.as_mut_slice(), &plan)
            .map_err(|e| anyhow!("backend post failed on channel {idx}: {e}"))?;
    }

    if normalize {
        normalize_peak_dbfs_channels(&mut channels_out, normalize_dbfs);
    }
    write_wav_channels(output, sample_rate, &channels_out, output_encoding)?;

    let frames = channels_out.iter().map(|c| c.len()).min().unwrap_or(0);
    Ok(GoSummary {
        output: output.to_path_buf(),
        frames,
        sample_rate,
        channels: spatial.layout.channels(),
        level,
        flavor: chosen,
        seed: base_seed,
        output_encoding,
        layout: Some(spatial.layout.as_str()),
        backend_requested: plan.requested,
        backend_active: plan.active,
        jobs: plan.jobs,
    })
}

pub fn analyze_wav(path: &Path) -> Result<Analysis> {
    let (channel_data, sample_rate, channels) = read_wav_channels_f64(path)?;
    if channel_data.is_empty() || channel_data[0].is_empty() {
        return Err(anyhow!("input had no audio samples"));
    }
    let mono = mixdown_mono(&channel_data);
    Ok(analyze_samples_base(&mono, sample_rate, channels))
}

pub fn analyze_wav_with_options(path: &Path, opts: &AnalyzeOptions) -> Result<AnalysisReport> {
    validate_analyze_options(opts)?;
    let (channel_data, sample_rate, channels) = read_wav_channels_f64(path)?;
    if channel_data.is_empty() || channel_data[0].is_empty() {
        return Err(anyhow!("input had no audio samples"));
    }
    let mono = mixdown_mono(&channel_data);
    let stereo_pair = if channels >= 2 {
        Some((channel_data[0].as_slice(), channel_data[1].as_slice()))
    } else {
        None
    };

    let basic = analyze_samples_base(&mono, sample_rate, channels);
    let (selected_ugly_index, psycho) = match opts.model {
        AnalyzeModel::Basic => (basic.ugly_index, None),
        AnalyzeModel::Psycho => {
            let psycho = analyze_samples_psycho(&mono, sample_rate, &basic, opts, stereo_pair);
            (psycho.ugly_index, Some(psycho))
        }
    };

    Ok(AnalysisReport {
        model: opts.model.as_str().to_string(),
        selected_ugly_index,
        basic,
        psycho,
    })
}

pub fn validate_render_options(opts: &RenderOptions) -> Result<()> {
    if !(0.1..=MAX_RENDER_DURATION_S).contains(&opts.duration) {
        return Err(anyhow!(
            "duration must be between 0.1 and {} seconds",
            MAX_RENDER_DURATION_S as u64
        ));
    }
    if !(8_000..=192_000).contains(&opts.sample_rate) {
        return Err(anyhow!("sample-rate must be between 8000 and 192000"));
    }
    if !(0.0..=1.0).contains(&opts.gain) {
        return Err(anyhow!("gain must be between 0.0 and 1.0"));
    }
    if opts.normalize_dbfs > 0.0 {
        return Err(anyhow!("normalize-dbfs must be <= 0.0"));
    }
    Ok(())
}

pub fn validate_speech_options(opts: &SpeechRenderOptions) -> Result<()> {
    if opts.text.trim().is_empty() {
        return Err(anyhow!("speech text must not be empty"));
    }
    if !(8_000..=192_000).contains(&opts.sample_rate) {
        return Err(anyhow!("sample-rate must be between 8000 and 192000"));
    }
    if !(0.0..=1.0).contains(&opts.gain) {
        return Err(anyhow!("gain must be between 0.0 and 1.0"));
    }
    if opts.normalize_dbfs > 0.0 {
        return Err(anyhow!("normalize-dbfs must be <= 0.0"));
    }
    if !(1.0..=40.0).contains(&opts.units_per_second) {
        return Err(anyhow!("units-per-second must be between 1.0 and 40.0"));
    }
    if !(20.0..=2400.0).contains(&opts.pitch_hz) {
        return Err(anyhow!("pitch-hz must be between 20 and 2400"));
    }
    if !(0.0..=1.0).contains(&opts.pitch_jitter) {
        return Err(anyhow!("pitch-jitter must be between 0.0 and 1.0"));
    }
    if !(0.0..=20.0).contains(&opts.vibrato_hz) {
        return Err(anyhow!("vibrato-hz must be between 0.0 and 20.0"));
    }
    if !(0.0..=1.0).contains(&opts.vibrato_depth) {
        return Err(anyhow!("vibrato-depth must be between 0.0 and 1.0"));
    }
    if !(0.05..=0.95).contains(&opts.duty_cycle) {
        return Err(anyhow!("duty-cycle must be between 0.05 and 0.95"));
    }
    if !(0.2..=3.5).contains(&opts.formant_shift) {
        return Err(anyhow!("formant-shift must be between 0.2 and 3.5"));
    }
    if !(0.0..=2.0).contains(&opts.consonant_noise) {
        return Err(anyhow!("consonant-noise must be between 0.0 and 2.0"));
    }
    if !(0.0..=1.5).contains(&opts.vowel_mix) {
        return Err(anyhow!("vowel-mix must be between 0.0 and 1.5"));
    }
    if !(0.0..=2.5).contains(&opts.hiss)
        || !(0.0..=2.5).contains(&opts.buzz)
        || !(0.0..=1.5).contains(&opts.robotize)
        || !(0.0..=1.5).contains(&opts.glide)
        || !(0.0..=1.0).contains(&opts.monotone)
        || !(0.0..=2.0).contains(&opts.emphasis)
        || !(0.0..=1.0).contains(&opts.ring_mix)
        || !(0.0..=1.0).contains(&opts.sub_mix)
        || !(0.0..=1.0).contains(&opts.nasal)
        || !(0.0..=1.0).contains(&opts.throat)
        || !(0.0..=1.0).contains(&opts.drift)
        || !(0.0..=1.0).contains(&opts.resampler_grit)
    {
        return Err(anyhow!(
            "one or more speech mix parameters are out of range"
        ));
    }
    if !(0.5..=16.0).contains(&opts.fold) {
        return Err(anyhow!("fold must be between 0.5 and 16.0"));
    }
    if !(0.0..=2.0).contains(&opts.chaos) {
        return Err(anyhow!("chaos must be between 0.0 and 2.0"));
    }
    if !(0.0..=500.0).contains(&opts.word_gap_ms)
        || !(0.0..=1500.0).contains(&opts.sentence_gap_ms)
        || !(0.0..=3000.0).contains(&opts.paragraph_gap_ms)
        || !(0.0..=1000.0).contains(&opts.punctuation_gap_ms)
        || !(0.0..=200.0).contains(&opts.attack_ms)
        || !(0.0..=500.0).contains(&opts.release_ms)
    {
        return Err(anyhow!(
            "one or more speech timing parameters are out of range"
        ));
    }
    if !(1.0..=24.0).contains(&opts.bitcrush_bits) {
        return Err(anyhow!("bitcrush-bits must be between 1.0 and 24.0"));
    }
    if !(500.0..=96_000.0).contains(&opts.sample_hold_hz) {
        return Err(anyhow!("sample-hold-hz must be between 500 and 96000"));
    }
    Ok(())
}

pub fn validate_analyze_options(opts: &AnalyzeOptions) -> Result<()> {
    if opts.fft_size < 128 {
        return Err(anyhow!("fft-size must be >= 128"));
    }
    if opts.hop_size == 0 {
        return Err(anyhow!("hop-size must be > 0"));
    }
    if opts.hop_size > opts.fft_size {
        return Err(anyhow!("hop-size must be <= fft-size"));
    }
    Ok(())
}

pub fn validate_ugliness_contour(contour: &UglinessContour) -> Result<()> {
    if contour.version != 1 {
        return Err(anyhow!(
            "unsupported contour version {} (expected 1)",
            contour.version
        ));
    }
    if contour.points.is_empty() {
        return Err(anyhow!("contour must include at least one point"));
    }
    let mut prev_t = -1.0_f64;
    for (idx, p) in contour.points.iter().enumerate() {
        if !p.t.is_finite() || !(0.0..=1.0).contains(&p.t) {
            return Err(anyhow!(
                "contour point {idx} has invalid t={} (expected 0.0..=1.0)",
                p.t
            ));
        }
        if !(1..=1000).contains(&p.level) {
            return Err(anyhow!(
                "contour point {idx} has invalid level={} (expected 1..=1000)",
                p.level
            ));
        }
        if p.t < prev_t {
            return Err(anyhow!(
                "contour times must be non-decreasing (point {idx} has t={}, previous was {})",
                p.t,
                prev_t
            ));
        }
        prev_t = p.t;
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum SpeechSymbolKind {
    Vowel,
    Consonant,
    Digit,
    Whitespace,
    Punctuation,
    ParagraphBreak,
}

#[derive(Clone, Copy)]
struct SpeechSymbolParams {
    voiced: bool,
    noisy: bool,
    pitch_mul: f64,
    formant1: f64,
    formant2: f64,
    brightness: f64,
    noise_mul: f64,
    vowel_mul: f64,
}

#[derive(Clone, Copy)]
struct SpeechProfileTuning {
    pitch_mul: f64,
    formant_mul: f64,
    noise_mul: f64,
    buzz_mul: f64,
    hiss_mul: f64,
    robot_carrier_hz: f64,
    default_hold_hz: f64,
    default_bitcrush_bits: f64,
    default_fold: f64,
}

fn speech_profile_tuning(profile: SpeechChipProfile) -> SpeechProfileTuning {
    match profile {
        SpeechChipProfile::VotraxSc01 => SpeechProfileTuning {
            pitch_mul: 1.08,
            formant_mul: 0.95,
            noise_mul: 0.9,
            buzz_mul: 0.9,
            hiss_mul: 0.7,
            robot_carrier_hz: 620.0,
            default_hold_hz: 8_000.0,
            default_bitcrush_bits: 6.5,
            default_fold: 2.0,
        },
        SpeechChipProfile::Tms5220 => SpeechProfileTuning {
            pitch_mul: 0.94,
            formant_mul: 1.0,
            noise_mul: 0.85,
            buzz_mul: 0.85,
            hiss_mul: 0.55,
            robot_carrier_hz: 520.0,
            default_hold_hz: 10_000.0,
            default_bitcrush_bits: 7.5,
            default_fold: 2.3,
        },
        SpeechChipProfile::Sp0256 => SpeechProfileTuning {
            pitch_mul: 0.98,
            formant_mul: 1.05,
            noise_mul: 0.9,
            buzz_mul: 1.0,
            hiss_mul: 0.55,
            robot_carrier_hz: 710.0,
            default_hold_hz: 9_500.0,
            default_bitcrush_bits: 7.0,
            default_fold: 2.5,
        },
        SpeechChipProfile::Mea8000 => SpeechProfileTuning {
            pitch_mul: 1.02,
            formant_mul: 1.08,
            noise_mul: 0.75,
            buzz_mul: 0.75,
            hiss_mul: 0.45,
            robot_carrier_hz: 480.0,
            default_hold_hz: 12_000.0,
            default_bitcrush_bits: 8.5,
            default_fold: 2.0,
        },
        SpeechChipProfile::S14001a => SpeechProfileTuning {
            pitch_mul: 1.12,
            formant_mul: 0.88,
            noise_mul: 1.05,
            buzz_mul: 1.1,
            hiss_mul: 0.8,
            robot_carrier_hz: 860.0,
            default_hold_hz: 7_600.0,
            default_bitcrush_bits: 5.8,
            default_fold: 2.8,
        },
        SpeechChipProfile::C64Sam => SpeechProfileTuning {
            pitch_mul: 1.18,
            formant_mul: 0.92,
            noise_mul: 0.95,
            buzz_mul: 1.05,
            hiss_mul: 0.65,
            robot_carrier_hz: 910.0,
            default_hold_hz: 7_200.0,
            default_bitcrush_bits: 5.5,
            default_fold: 3.0,
        },
        SpeechChipProfile::Arcadey90s => SpeechProfileTuning {
            pitch_mul: 1.0,
            formant_mul: 1.15,
            noise_mul: 0.8,
            buzz_mul: 1.2,
            hiss_mul: 0.4,
            robot_carrier_hz: 350.0,
            default_hold_hz: 14_000.0,
            default_bitcrush_bits: 9.0,
            default_fold: 1.8,
        },
        SpeechChipProfile::HandheldLcd => SpeechProfileTuning {
            pitch_mul: 1.26,
            formant_mul: 0.84,
            noise_mul: 1.1,
            buzz_mul: 1.25,
            hiss_mul: 0.9,
            robot_carrier_hz: 1_120.0,
            default_hold_hz: 6_800.0,
            default_bitcrush_bits: 4.8,
            default_fold: 3.3,
        },
    }
}

fn speech_units_for_mode(text: &str, mode: SpeechInputMode) -> Vec<SpeechUnit> {
    let mode = if matches!(mode, SpeechInputMode::Auto) {
        if text.contains("\n\n") {
            SpeechInputMode::Paragraph
        } else if text.contains('.') || text.contains('!') || text.contains('?') {
            SpeechInputMode::Sentence
        } else if text.contains(char::is_whitespace) {
            SpeechInputMode::Word
        } else {
            SpeechInputMode::Character
        }
    } else {
        mode
    };

    let mut units = Vec::new();
    for ch in text.chars() {
        let kind = classify_speech_symbol(ch);
        let duration_s = speech_symbol_duration(ch, kind, mode);
        let gap_s = speech_symbol_gap(ch, kind, mode);
        let emphasis = if ch.is_uppercase() || matches!(ch, '!' | '?') {
            1.0
        } else if ch.is_ascii_digit() {
            0.55
        } else {
            0.2
        };
        units.push(SpeechUnit {
            ch,
            kind,
            duration_s,
            gap_s,
            emphasis,
        });
    }
    units
}

fn classify_speech_symbol(ch: char) -> SpeechSymbolKind {
    if ch == '\n' {
        SpeechSymbolKind::ParagraphBreak
    } else if ch.is_whitespace() {
        SpeechSymbolKind::Whitespace
    } else if ch.is_ascii_digit() {
        SpeechSymbolKind::Digit
    } else if matches!(
        ch,
        '.' | ',' | ';' | ':' | '!' | '?' | '\'' | '"' | '-' | '(' | ')'
    ) {
        SpeechSymbolKind::Punctuation
    } else if matches!(ch.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u' | 'y') {
        SpeechSymbolKind::Vowel
    } else {
        SpeechSymbolKind::Consonant
    }
}

fn speech_symbol_duration(ch: char, kind: SpeechSymbolKind, mode: SpeechInputMode) -> f64 {
    let base = match kind {
        SpeechSymbolKind::Vowel => 0.085,
        SpeechSymbolKind::Consonant => 0.055,
        SpeechSymbolKind::Digit => 0.075,
        SpeechSymbolKind::Whitespace => 0.0,
        SpeechSymbolKind::Punctuation => 0.02,
        SpeechSymbolKind::ParagraphBreak => 0.0,
    };
    let mode_scale = match mode {
        SpeechInputMode::Character => 1.15,
        SpeechInputMode::Word => 1.0,
        SpeechInputMode::Sentence => 0.95,
        SpeechInputMode::Paragraph => 0.9,
        SpeechInputMode::Auto => 1.0,
    };
    let char_scale = if ch.is_uppercase() { 1.1 } else { 1.0 };
    base * mode_scale * char_scale
}

fn speech_symbol_gap(ch: char, kind: SpeechSymbolKind, mode: SpeechInputMode) -> f64 {
    match kind {
        SpeechSymbolKind::Whitespace => {
            if matches!(mode, SpeechInputMode::Character) {
                0.02
            } else {
                0.05
            }
        }
        SpeechSymbolKind::Punctuation => match ch {
            '.' | '!' | '?' => 0.12,
            ',' | ';' | ':' => 0.07,
            _ => 0.03,
        },
        SpeechSymbolKind::ParagraphBreak => 0.24,
        _ => 0.004,
    }
}

fn speech_symbol_params(ch: char, kind: SpeechSymbolKind) -> SpeechSymbolParams {
    match kind {
        SpeechSymbolKind::Vowel => {
            let (f1, f2, pitch_mul) = match ch.to_ascii_lowercase() {
                'a' => (820.0, 1_250.0, 0.96),
                'e' => (560.0, 1_850.0, 1.04),
                'i' => (320.0, 2_220.0, 1.12),
                'o' => (480.0, 960.0, 0.9),
                'u' => (360.0, 780.0, 0.84),
                _ => (520.0, 1_480.0, 1.0),
            };
            SpeechSymbolParams {
                voiced: true,
                noisy: false,
                pitch_mul,
                formant1: f1,
                formant2: f2,
                brightness: 0.45,
                noise_mul: 0.0,
                vowel_mul: 1.0,
            }
        }
        SpeechSymbolKind::Consonant => {
            let lower = ch.to_ascii_lowercase();
            let fricative = matches!(lower, 's' | 'z' | 'f' | 'v' | 'h' | 'x' | 'c');
            let nasal = matches!(lower, 'm' | 'n');
            SpeechSymbolParams {
                voiced: !matches!(lower, 'p' | 't' | 'k' | 's' | 'f' | 'h' | 'x'),
                noisy: fricative || matches!(lower, 'p' | 't' | 'k' | 'q' | 'g'),
                pitch_mul: if nasal { 0.82 } else { 1.08 },
                formant1: if nasal { 280.0 } else { 620.0 },
                formant2: if fricative { 2_600.0 } else { 1_420.0 },
                brightness: if fricative { 0.9 } else { 0.55 },
                noise_mul: if fricative { 1.0 } else { 0.55 },
                vowel_mul: 0.35,
            }
        }
        SpeechSymbolKind::Digit => SpeechSymbolParams {
            voiced: true,
            noisy: false,
            pitch_mul: 1.02 + (ch as u32 % 3) as f64 * 0.04,
            formant1: 460.0,
            formant2: 1_520.0,
            brightness: 0.5,
            noise_mul: 0.1,
            vowel_mul: 0.8,
        },
        SpeechSymbolKind::Whitespace | SpeechSymbolKind::ParagraphBreak => SpeechSymbolParams {
            voiced: false,
            noisy: false,
            pitch_mul: 1.0,
            formant1: 400.0,
            formant2: 1_200.0,
            brightness: 0.0,
            noise_mul: 0.0,
            vowel_mul: 0.0,
        },
        SpeechSymbolKind::Punctuation => SpeechSymbolParams {
            voiced: false,
            noisy: true,
            pitch_mul: 1.0,
            formant1: 720.0,
            formant2: 2_400.0,
            brightness: 0.95,
            noise_mul: 0.4,
            vowel_mul: 0.0,
        },
    }
}

fn seed_from_time() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(dur) => dur.as_nanos() as u64,
        Err(_) => 0x5EED_F00D,
    }
}

#[derive(Clone, Copy)]
struct StrangeAttractorState {
    x: f64,
    y: f64,
}

impl StrangeAttractorState {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

fn synth_ugly(
    frames: usize,
    sample_rate: f64,
    style: Style,
    gain: f64,
    rng: &mut ChaCha8Rng,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(frames);
    let params = style_params(style);
    let mut glitch_samples_left = 0usize;
    let mut pop_env = 0.0_f64;
    let mut rub_state = 0.0_f64;
    let mut spank_env = 0.0_f64;
    let mut steal_hold = 0.0_f64;
    let mut steal_countdown = 0usize;
    let mut catastrophic_hold = 0.0_f64;
    let mut catastrophic_countdown = 0usize;
    let mut catastrophic_env = 0.0_f64;
    let mut glitch_attractor = StrangeAttractorState::new(0.11, -0.23);
    let mut punish_attractor = StrangeAttractorState::new(-0.37, 0.19);
    let mut catastrophic_attractor = StrangeAttractorState::new(0.29, 0.41);
    let mut lucky_mode = 0usize;
    let mut lucky_mode_samples_left = 0usize;

    for i in 0..frames {
        let t = i as f64 / sample_rate;
        let phase_wobble = (2.0 * PI * 0.27 * t).sin() * 4.0;

        let base = match style {
            Style::Harsh => {
                let saw_a = saw(90.0 + phase_wobble + 55.0 * (2.0 * PI * 1.7 * t).sin(), t);
                let sq_b = square(180.0 + 40.0 * (2.0 * PI * 3.1 * t).sin(), t);
                0.65 * saw_a + 0.45 * sq_b
            }
            Style::Digital => {
                let tri = triangle(150.0 + 90.0 * (2.0 * PI * 11.0 * t).sin(), t);
                let ring = ((2.0 * PI * 400.0 * t).sin()) * ((2.0 * PI * 1250.0 * t).sin());
                0.5 * tri + 0.8 * ring
            }
            Style::Meltdown => {
                let fm_carrier =
                    (2.0 * PI * (110.0 + 550.0 * (2.0 * PI * 0.9 * t).sin()) * t).sin();
                let sub = saw(55.0 + 20.0 * (2.0 * PI * 0.2 * t).sin(), t);
                0.75 * fm_carrier + 0.35 * sub
            }
            Style::Glitch => {
                let staircase = ((i / 6) % 48) as f64 / 24.0 - 1.0;
                let fold = (2.0 * PI * (60.0 + 420.0 * (2.0 * PI * 0.4 * t).sin().abs()) * t)
                    .sin()
                    .signum();
                let koch =
                    koch_quasi_oscillator(73.0 + 280.0 * (2.0 * PI * 0.31 * t).sin().abs(), t, 4);
                let mandelbrot = mandelbrot_quasi_oscillator(41.0 + 0.12 * i as f64, t, 14);
                let attractor = strange_attractor_oscillator(
                    &mut glitch_attractor,
                    0.92 + 0.25 * (2.0 * PI * 0.17 * t).sin().abs(),
                );
                let quasi = fractal_quasi_oscillator(koch, mandelbrot, attractor, 0.72, 2.8);
                0.52 * staircase + 0.42 * fold + 0.38 * quasi
            }
            Style::Pop => {
                if rng.gen_bool(0.004) {
                    pop_env = rng.gen_range(0.7_f64..1.3_f64);
                }
                pop_env *= 0.988;
                let body = (2.0 * PI * (80.0 + 2400.0 * pop_env) * t).sin();
                let spray = rng.gen_range(-1.0_f64..1.0_f64) * pop_env;
                0.9 * pop_env * body + 0.35 * spray
            }
            Style::Buzz => {
                let f = 96.0 + 20.0 * (2.0 * PI * 0.3 * t).sin();
                let mut harmonics = 0.0;
                for h in 1..=9 {
                    harmonics += saw(f * h as f64, t) / h as f64;
                }
                0.75 * harmonics + 0.25 * square(f * 2.0, t)
            }
            Style::Rub => {
                let white = rng.gen_range(-1.0_f64..1.0_f64);
                rub_state = 0.985 * rub_state + 0.12 * white;
                let friction = rub_state * (0.45 + 0.55 * (2.0 * PI * 5.0 * t).sin().abs());
                let scrape = saw(35.0 + 12.0 * (2.0 * PI * 0.7 * t).sin(), t) * 0.25;
                1.1 * friction + scrape
            }
            Style::Hum => {
                let mains = 0.8 * (2.0 * PI * 60.0 * t).sin()
                    + 0.32 * (2.0 * PI * 120.0 * t).sin()
                    + 0.15 * (2.0 * PI * 180.0 * t).sin();
                mains + 0.05 * (2.0 * PI * 7.0 * t).sin()
            }
            Style::Distort => {
                let dry = 0.65 * (2.0 * PI * 140.0 * t).sin()
                    + 0.45 * saw(280.0, t)
                    + 0.25 * square(35.0, t);
                (dry * 3.8).tanh()
            }
            Style::Spank => {
                if rng.gen_bool(0.006) {
                    spank_env = rng.gen_range(0.8_f64..1.4_f64);
                }
                spank_env *= 0.976;
                let snap = (2.0 * PI * (190.0 + 4200.0 * spank_env) * t).sin();
                let tail = saw(70.0, t) * spank_env * 0.4;
                1.1 * spank_env * snap + tail
            }
            Style::Punish => {
                let fm = (2.0 * PI * (90.0 + 1400.0 * (2.0 * PI * 2.4 * t).sin()) * t).sin();
                let sq = square(43.0, t);
                let koch =
                    koch_quasi_oscillator(95.0 + 480.0 * (2.0 * PI * 1.9 * t).sin().abs(), t, 5);
                let mandelbrot = mandelbrot_quasi_oscillator(83.0 + 0.33 * i as f64, t, 18);
                let attractor = strange_attractor_oscillator(
                    &mut punish_attractor,
                    1.18 + 0.31 * (2.0 * PI * 0.23 * t).sin().abs(),
                );
                let quasi = fractal_quasi_oscillator(koch, mandelbrot, attractor, 1.05, 3.4);
                let brutal =
                    (fm + 0.7 * sq + 0.55 * quasi + 0.4 * rng.gen_range(-1.0_f64..1.0_f64)) * 4.2;
                brutal.tanh()
            }
            Style::Steal => {
                if steal_countdown == 0 {
                    steal_countdown = rng.gen_range(3..80);
                    steal_hold = saw(120.0 + 65.0 * (2.0 * PI * 0.2 * t).sin(), t)
                        + 0.5 * rng.gen_range(-1.0_f64..1.0_f64);
                }
                steal_countdown -= 1;
                let ghost = square(40.0 + 170.0 * steal_hold.abs(), t) * 0.25;
                0.85 * steal_hold + ghost
            }
            Style::Catastrophic => {
                if catastrophic_countdown == 0 {
                    catastrophic_countdown = rng.gen_range(2..28);
                    catastrophic_hold = rng.gen_range(-1.0_f64..1.0_f64);
                    catastrophic_env = rng.gen_range(0.6_f64..1.5_f64);
                }
                catastrophic_countdown -= 1;
                catastrophic_env = (catastrophic_env * 0.989).max(0.15);
                if rng.gen_bool(0.009) {
                    catastrophic_env = rng.gen_range(0.8_f64..1.8_f64);
                }

                let staircase = (((i / 3) % 24) as f64 / 12.0) - 1.0;
                let fm = (2.0
                    * PI
                    * (120.0
                        + 1800.0 * (2.0 * PI * 3.7 * t).sin()
                        + 420.0 * catastrophic_hold.abs())
                    * t)
                    .sin();
                let shriek = saw(
                    420.0
                        + 4200.0 * (0.5 + 0.5 * (2.0 * PI * 0.43 * t).sin())
                        + 600.0 * catastrophic_env,
                    t,
                );
                let koch = koch_quasi_oscillator(
                    160.0
                        + 2400.0 * (0.5 + 0.5 * (2.0 * PI * 0.61 * t).sin())
                        + 320.0 * catastrophic_env,
                    t,
                    6,
                );
                let mandelbrot = mandelbrot_quasi_oscillator(
                    137.0 + 0.61 * i as f64 + 41.0 * catastrophic_env,
                    t,
                    24,
                );
                let attractor = strange_attractor_oscillator(
                    &mut catastrophic_attractor,
                    1.36 + 0.48 * catastrophic_env,
                );
                let quasi = fractal_quasi_oscillator(koch, mandelbrot, attractor, 1.28, 4.3);
                let sub = square(27.0 + 90.0 * catastrophic_hold.abs(), t);
                let noise = rng.gen_range(-1.0_f64..1.0_f64) * (0.3 + 0.7 * catastrophic_env);
                let raw = 0.55 * fm
                    + 0.45 * staircase
                    + 0.45 * shriek
                    + 0.65 * quasi
                    + 0.35 * sub
                    + 0.6 * catastrophic_hold
                    + 0.4 * noise;
                ((raw * 5.6).sin() + (raw * 4.4).tanh() + 0.55 * raw.signum()) / 1.9
            }
            Style::Wink => {
                let gate = if (2.0 * PI * 2.8 * t).sin() >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                let chirp_freq = 250.0 + 2400.0 * (t * 1.6).fract();
                let chirp = (2.0 * PI * chirp_freq * t).sin();
                0.7 * chirp * gate + 0.35 * triangle(90.0, t)
            }
            Style::Lucky => {
                if lucky_mode_samples_left == 0 {
                    lucky_mode_samples_left = (sample_rate as usize / 5).max(200);
                    lucky_mode = rng.gen_range(0..6);
                }
                lucky_mode_samples_left -= 1;
                match lucky_mode {
                    0 => 0.8 * saw(80.0 + 20.0 * (2.0 * PI * 1.5 * t).sin(), t),
                    1 => 0.7 * square(42.0, t) + 0.4 * triangle(620.0, t),
                    2 => 0.6 * (2.0 * PI * (220.0 + 600.0 * (2.0 * PI * 0.8 * t).sin()) * t).sin(),
                    3 => 0.5 * saw(140.0, t) + 0.5 * rng.gen_range(-1.0_f64..1.0_f64),
                    4 => 0.9 * (((i / 8) % 32) as f64 / 16.0 - 1.0),
                    _ => 0.6 * triangle(300.0 + 120.0 * (2.0 * PI * 4.0 * t).sin(), t),
                }
            }
        };

        let hiss = rng.gen_range(-1.0_f64..1.0_f64) * params.hiss_amp;
        let click = if rng.gen_bool(params.click_prob) {
            rng.gen_range(-1.0_f64..1.0_f64) * params.click_amp
        } else {
            0.0
        };

        if glitch_samples_left == 0 && rng.gen_bool(params.glitch_prob) {
            glitch_samples_left = rng.gen_range(50..1600);
        }
        let glitch = if glitch_samples_left > 0 {
            glitch_samples_left -= 1;
            let held = ((i / 24) % 32) as f64 / 31.0;
            (held * 2.0 - 1.0) * params.glitch_amp
        } else {
            0.0
        };

        let raw = base + hiss + click + glitch;
        let levels = 2.0_f64.powf(params.bit_depth);
        let crushed = (raw * levels).round() / levels;
        out.push(soft_clip(crushed * gain * params.drive));
    }

    out
}

#[derive(Clone, Copy)]
struct StyleParams {
    hiss_amp: f64,
    click_prob: f64,
    click_amp: f64,
    glitch_prob: f64,
    glitch_amp: f64,
    bit_depth: f64,
    drive: f64,
}

fn style_params(style: Style) -> StyleParams {
    match style {
        Style::Harsh => StyleParams {
            hiss_amp: 0.22,
            click_prob: 0.0015,
            click_amp: 1.4,
            glitch_prob: 0.0020,
            glitch_amp: 0.75,
            bit_depth: 6.0,
            drive: 1.35,
        },
        Style::Digital => StyleParams {
            hiss_amp: 0.16,
            click_prob: 0.0012,
            click_amp: 1.2,
            glitch_prob: 0.0015,
            glitch_amp: 0.55,
            bit_depth: 4.0,
            drive: 1.3,
        },
        Style::Meltdown => StyleParams {
            hiss_amp: 0.25,
            click_prob: 0.0017,
            click_amp: 1.5,
            glitch_prob: 0.0023,
            glitch_amp: 0.9,
            bit_depth: 5.0,
            drive: 1.45,
        },
        Style::Glitch => StyleParams {
            hiss_amp: 0.09,
            click_prob: 0.0030,
            click_amp: 1.4,
            glitch_prob: 0.0050,
            glitch_amp: 1.2,
            bit_depth: 3.0,
            drive: 1.55,
        },
        Style::Pop => StyleParams {
            hiss_amp: 0.06,
            click_prob: 0.0060,
            click_amp: 1.8,
            glitch_prob: 0.0010,
            glitch_amp: 0.4,
            bit_depth: 7.0,
            drive: 1.25,
        },
        Style::Buzz => StyleParams {
            hiss_amp: 0.12,
            click_prob: 0.0010,
            click_amp: 0.9,
            glitch_prob: 0.0012,
            glitch_amp: 0.5,
            bit_depth: 6.0,
            drive: 1.5,
        },
        Style::Rub => StyleParams {
            hiss_amp: 0.18,
            click_prob: 0.0015,
            click_amp: 0.8,
            glitch_prob: 0.0009,
            glitch_amp: 0.4,
            bit_depth: 6.0,
            drive: 1.3,
        },
        Style::Hum => StyleParams {
            hiss_amp: 0.04,
            click_prob: 0.0003,
            click_amp: 0.5,
            glitch_prob: 0.0005,
            glitch_amp: 0.25,
            bit_depth: 9.0,
            drive: 1.1,
        },
        Style::Distort => StyleParams {
            hiss_amp: 0.1,
            click_prob: 0.0011,
            click_amp: 1.1,
            glitch_prob: 0.0013,
            glitch_amp: 0.55,
            bit_depth: 5.0,
            drive: 1.7,
        },
        Style::Spank => StyleParams {
            hiss_amp: 0.08,
            click_prob: 0.0038,
            click_amp: 1.6,
            glitch_prob: 0.0018,
            glitch_amp: 0.65,
            bit_depth: 6.0,
            drive: 1.45,
        },
        Style::Punish => StyleParams {
            hiss_amp: 0.28,
            click_prob: 0.0032,
            click_amp: 1.9,
            glitch_prob: 0.0038,
            glitch_amp: 1.3,
            bit_depth: 3.0,
            drive: 1.95,
        },
        Style::Steal => StyleParams {
            hiss_amp: 0.14,
            click_prob: 0.0022,
            click_amp: 1.2,
            glitch_prob: 0.0028,
            glitch_amp: 0.95,
            bit_depth: 4.0,
            drive: 1.55,
        },
        Style::Catastrophic => StyleParams {
            hiss_amp: 0.34,
            click_prob: 0.0065,
            click_amp: 2.2,
            glitch_prob: 0.0072,
            glitch_amp: 1.6,
            bit_depth: 2.0,
            drive: 2.35,
        },
        Style::Wink => StyleParams {
            hiss_amp: 0.07,
            click_prob: 0.0010,
            click_amp: 0.9,
            glitch_prob: 0.0011,
            glitch_amp: 0.45,
            bit_depth: 7.0,
            drive: 1.2,
        },
        Style::Lucky => StyleParams {
            hiss_amp: 0.2,
            click_prob: 0.0024,
            click_amp: 1.4,
            glitch_prob: 0.0027,
            glitch_amp: 1.0,
            bit_depth: 5.0,
            drive: 1.5,
        },
    }
}

fn apply_style_layer(buffer: &mut [f64], layer: &[f64], style: Style) {
    let p = style_params(style);
    let dry = 0.6;
    let wet = (0.35 + (p.drive - 1.0) * 0.2).clamp(0.2, 0.85);
    for (x, y) in buffer.iter_mut().zip(layer.iter()) {
        *x = soft_clip(dry * *x + wet * *y);
    }
}

fn render_style_layers(
    style_specs: &[(usize, Style, u64)],
    frames: usize,
    sample_rate: f64,
    jobs: usize,
) -> Result<std::collections::BTreeMap<usize, Vec<f64>>> {
    use std::collections::BTreeMap;

    if style_specs.is_empty() {
        return Ok(BTreeMap::new());
    }

    if jobs <= 1 || style_specs.len() == 1 {
        let mut out = BTreeMap::new();
        for (idx, style, seed) in style_specs {
            let layer = render_samples(frames, sample_rate, *style, 1.0, *seed)?;
            out.insert(*idx, layer);
        }
        return Ok(out);
    }

    let pool = ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .map_err(|e| anyhow!("failed to build render pool: {e}"))?;

    let rendered: Vec<(usize, Result<Vec<f64>>)> = pool.install(|| {
        style_specs
            .par_iter()
            .map(|(idx, style, seed)| {
                (
                    *idx,
                    render_samples(frames, sample_rate, *style, 1.0, *seed),
                )
            })
            .collect()
    });

    let mut out = BTreeMap::new();
    for (idx, layer_res) in rendered {
        out.insert(idx, layer_res?);
    }
    Ok(out)
}

fn apply_effect(
    buffer: &mut [f64],
    effect: Effect,
    sample_rate: f64,
    seed: u64,
    frame_offset: u64,
) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    match effect {
        Effect::Stutter => effect_stutter(buffer, sample_rate, &mut rng),
        Effect::Pop => effect_pop(buffer, sample_rate, &mut rng),
        Effect::Crush => effect_crush(buffer, sample_rate, &mut rng),
        Effect::Gate => effect_gate(buffer, sample_rate, frame_offset),
        Effect::Smear => effect_smear(buffer, sample_rate),
        Effect::DissonanceRing => effect_dissonance_ring(buffer, sample_rate, 0.85, frame_offset),
        Effect::DissonanceExpand => effect_dissonance_expand(buffer, sample_rate, 0.8),
    }
}

fn effect_stutter(buffer: &mut [f64], sample_rate: f64, rng: &mut ChaCha8Rng) {
    if buffer.len() < 8 {
        return;
    }
    let min_win = (0.01 * sample_rate).max(8.0) as usize;
    let max_win = (0.07 * sample_rate).max(min_win as f64 + 1.0) as usize;
    let mut i = 0usize;
    while i + min_win < buffer.len() {
        if rng.gen_bool(0.22) {
            let win = rng.gen_range(min_win..=max_win).min(buffer.len() - i);
            let reps = rng.gen_range(2..=5);
            let chunk = buffer[i..i + win].to_vec();
            for r in 1..reps {
                let dst = i + r * win;
                if dst + win > buffer.len() {
                    break;
                }
                buffer[dst..dst + win].copy_from_slice(&chunk);
            }
            i += win * reps;
        } else {
            i += rng.gen_range(min_win..=max_win).min(buffer.len() - i);
        }
    }
}

fn effect_pop(buffer: &mut [f64], sample_rate: f64, rng: &mut ChaCha8Rng) {
    if buffer.is_empty() {
        return;
    }
    let min_len = (0.003 * sample_rate).max(8.0) as usize;
    let max_len = (0.02 * sample_rate).max(min_len as f64 + 1.0) as usize;
    let mut i = 0usize;
    while i < buffer.len() {
        if rng.gen_bool(0.0035) {
            let len = rng.gen_range(min_len..=max_len).min(buffer.len() - i);
            let amp = rng.gen_range(0.35_f64..1.0_f64);
            for j in 0..len {
                let env = (1.0 - j as f64 / len as f64).powf(2.2);
                let phase = j as f64 / sample_rate;
                let tone = (2.0 * PI * (700.0 + 4200.0 * env) * phase).sin();
                let noise = rng.gen_range(-1.0_f64..1.0_f64);
                buffer[i + j] = soft_clip(buffer[i + j] + amp * env * (0.75 * tone + 0.25 * noise));
            }
            i += len;
        } else {
            i += 1;
        }
    }
}

fn effect_crush(buffer: &mut [f64], sample_rate: f64, rng: &mut ChaCha8Rng) {
    if buffer.is_empty() {
        return;
    }
    let hold = rng
        .gen_range((0.0008 * sample_rate) as usize..=(0.004 * sample_rate) as usize)
        .max(1);
    let levels = 2.0_f64.powf(rng.gen_range(3.0_f64..6.0_f64));
    let mut i = 0usize;
    while i < buffer.len() {
        let src = buffer[i];
        let held = (src * levels).round() / levels;
        let end = (i + hold).min(buffer.len());
        for x in &mut buffer[i..end] {
            *x = held;
        }
        i = end;
    }
}

fn effect_gate(buffer: &mut [f64], sample_rate: f64, frame_offset: u64) {
    if buffer.is_empty() {
        return;
    }
    let lfo_hz = 7.5;
    for (i, x) in buffer.iter_mut().enumerate() {
        let t = (frame_offset + i as u64) as f64 / sample_rate;
        let g = if (2.0 * PI * lfo_hz * t).sin() > 0.0 {
            1.0
        } else {
            0.2
        };
        *x *= g;
    }
}

fn effect_smear(buffer: &mut [f64], sample_rate: f64) {
    if buffer.is_empty() {
        return;
    }
    let delay = (0.018 * sample_rate).max(1.0) as usize;
    let fb = 0.55;
    for i in delay..buffer.len() {
        let delayed = buffer[i - delay];
        buffer[i] = soft_clip(buffer[i] + delayed * fb);
    }
}

fn kameoka_max_roughness_gap(freq_hz: f64) -> f64 {
    2.27 * freq_hz.max(1.0).powf(0.477)
}

fn third_octave_bands(sample_rate: f64) -> Vec<(f64, f64)> {
    let nyquist = (sample_rate * 0.5 * 0.98).max(40.0);
    let edge_ratio = 2.0_f64.powf(1.0 / 6.0);
    let mut center = 31.5_f64;
    let mut bands = Vec::new();

    while center / edge_ratio < nyquist {
        let low = (center / edge_ratio).max(20.0);
        let high = (center * edge_ratio).min(nyquist);
        if high > low * 1.05 {
            bands.push((low, high));
        }
        center *= 2.0_f64.powf(1.0 / 3.0);
    }

    bands
}

fn paper_bark_bands(sample_rate: f64) -> Vec<(f64, f64, f64)> {
    let nyquist = sample_rate * 0.5;
    let mut bands = Vec::new();

    for window in PAPER_BARK_BOUNDARIES_HZ.windows(2) {
        let low = window[0];
        let high = window[1].min(nyquist * 0.995);
        if high > low * 1.02 {
            bands.push((low, high, (low * high).sqrt()));
        }
    }

    bands
}

fn lowpass_one_pole_alpha(cutoff_hz: f64, sample_rate: f64) -> f64 {
    let cutoff_hz = cutoff_hz.clamp(1.0, sample_rate * 0.45);
    (1.0 - (-2.0 * PI * cutoff_hz / sample_rate.max(1.0)).exp()).clamp(0.0, 1.0)
}

fn lowpass_one_pole_in_place(buffer: &mut [f64], cutoff_hz: f64, sample_rate: f64) {
    if buffer.is_empty() {
        return;
    }

    let alpha = lowpass_one_pole_alpha(cutoff_hz, sample_rate);
    let mut y = buffer[0];
    for x in buffer.iter_mut() {
        y += alpha * (*x - y);
        *x = y;
    }
}

fn highpass_one_pole_in_place(buffer: &mut [f64], cutoff_hz: f64, sample_rate: f64) {
    if buffer.is_empty() {
        return;
    }

    let cutoff_hz = cutoff_hz.clamp(1.0, sample_rate * 0.45);
    let dt = 1.0 / sample_rate.max(1.0);
    let rc = 1.0 / (2.0 * PI * cutoff_hz);
    let alpha = (rc / (rc + dt)).clamp(0.0, 1.0);
    let mut prev_x = buffer[0];
    let mut prev_y = 0.0_f64;

    for x in buffer.iter_mut() {
        let y = alpha * (prev_y + *x - prev_x);
        prev_x = *x;
        prev_y = y;
        *x = y;
    }
}

fn bandpass_filter_samples(input: &[f64], low_hz: f64, high_hz: f64, sample_rate: f64) -> Vec<f64> {
    let mut band = input.to_vec();
    if low_hz > 20.0 {
        highpass_one_pole_in_place(&mut band, low_hz, sample_rate);
    }
    lowpass_one_pole_in_place(&mut band, high_hz, sample_rate);
    band
}

fn band_focus_weight(center_hz: f64) -> f64 {
    if center_hz < 90.0 {
        0.25
    } else if center_hz < 180.0 {
        0.55
    } else if center_hz > 12_000.0 {
        0.8
    } else {
        1.0
    }
}

fn envelope_follower_step(input_abs: f64, prev: f64, decay_coeff: f64) -> f64 {
    if input_abs > prev {
        input_abs
    } else {
        decay_coeff * prev + (1.0 - decay_coeff) * input_abs
    }
}

fn scale_decay_from_44k(decay_coeff_44k: f64, sample_rate: f64) -> f64 {
    decay_coeff_44k
        .clamp(0.0, 0.999_999)
        .powf(44_100.0 / sample_rate.max(1.0))
}

fn effect_dissonance_ring(buffer: &mut [f64], sample_rate: f64, impact: f64, frame_offset: u64) {
    if buffer.len() < 8 {
        return;
    }

    let input = buffer.to_vec();
    let mut delta = vec![0.0_f64; buffer.len()];

    for (low_hz, high_hz) in third_octave_bands(sample_rate) {
        let center_hz = 0.2 * low_hz + 0.8 * high_hz;
        let band_impact = (impact * band_focus_weight((low_hz * high_hz).sqrt())).clamp(0.0, 1.0);
        if band_impact <= EPS64 {
            continue;
        }

        let band = bandpass_filter_samples(&input, low_hz, high_hz, sample_rate);
        let mod_hz = (0.5 * kameoka_max_roughness_gap(center_hz)).min(high_hz * 0.9);
        for (i, &sample) in band.iter().enumerate() {
            let t = (frame_offset + i as u64) as f64 / sample_rate;
            let modulator = (1.0 - band_impact) + band_impact * (2.0 * PI * mod_hz * t).sin();
            delta[i] += sample * modulator - sample;
        }
    }

    for (sample, change) in buffer.iter_mut().zip(delta.into_iter()) {
        *sample = soft_clip(*sample + change);
    }
}

fn effect_dissonance_expand(buffer: &mut [f64], sample_rate: f64, impact: f64) {
    if buffer.len() < 8 {
        return;
    }

    let input = buffer.to_vec();
    let mut delta = vec![0.0_f64; buffer.len()];

    for (low_hz, high_hz, center_hz) in paper_bark_bands(sample_rate) {
        let band_impact = (impact * band_focus_weight(center_hz)).clamp(0.0, 1.0);
        if band_impact <= EPS64 {
            continue;
        }

        let band = bandpass_filter_samples(&input, low_hz, high_hz, sample_rate);
        let fast_decay_44k = (1.0 - 0.000_837_34 * center_hz.sqrt()).clamp(0.0, 0.999_9);
        let fast_decay = scale_decay_from_44k(fast_decay_44k, sample_rate);
        let slow_decay = fast_decay.powf(1.0 / 50.0).clamp(0.0, 0.999_999);

        let mut fast_env = 0.0_f64;
        let mut slow_in = 0.0_f64;
        let mut slow_out = 0.0_f64;

        for (i, &sample) in band.iter().enumerate() {
            let rectified = sample.abs();
            fast_env = envelope_follower_step(rectified, fast_env, fast_decay);
            let emphasized = sample * fast_env.max(EPS64).powf(band_impact);
            slow_in = envelope_follower_step(rectified, slow_in, slow_decay);
            slow_out = envelope_follower_step(emphasized.abs(), slow_out, slow_decay);
            let normalized = emphasized * (slow_in / slow_out.max(1e-6));
            delta[i] += normalized - sample;
        }
    }

    for (sample, change) in buffer.iter_mut().zip(delta.into_iter()) {
        *sample = soft_clip(*sample + change);
    }
}

fn derive_seed(base_seed: u64, idx: u64) -> u64 {
    let mut z = base_seed
        .wrapping_add(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(idx.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn saw(freq: f64, t: f64) -> f64 {
    let phase = (freq * t).fract();
    2.0 * phase - 1.0
}

fn square(freq: f64, t: f64) -> f64 {
    if (freq * t).fract() < 0.5 { 1.0 } else { -1.0 }
}

fn triangle(freq: f64, t: f64) -> f64 {
    (2.0 * ((freq * t).fract()) - 1.0).abs() * 2.0 - 1.0
}

fn koch_quasi_oscillator(freq: f64, t: f64, depth: usize) -> f64 {
    let depth = depth.clamp(1, 8);
    let phase = (freq * t).fract();
    let mut out = 0.0_f64;
    let mut amp = 1.0_f64;
    let mut norm = 0.0_f64;

    for layer in 0..depth {
        let segs = 3.0_f64.powi(layer as i32 + 1);
        let p = (phase * segs).fract();
        let ridge = 1.0 - 4.0 * (p - 0.5).abs();
        let cusp = ridge.signum() * ridge.abs().powf(0.72 + 0.03 * layer as f64);
        out += amp * cusp;
        norm += amp;
        amp *= 0.61;
    }

    soft_clip(out / norm.max(EPS64))
}

fn mandelbrot_quasi_oscillator(freq: f64, t: f64, iterations: usize) -> f64 {
    let iterations = iterations.clamp(4, 32);
    let phase = (freq * 0.0031 * t).fract();
    let c_re = -0.84 + 1.58 * phase;
    let c_im = -0.42 + 0.84 * (2.0 * PI * (0.09 * freq + 1.0) * t).sin();
    let mut z_re = 0.0_f64;
    let mut z_im = 0.0_f64;
    let mut accum = 0.0_f64;

    for iter in 0..iterations {
        let next_re = z_re * z_re - z_im * z_im + c_re;
        let next_im = 2.0 * z_re * z_im + c_im;
        z_re = next_re.clamp(-2.4, 2.4);
        z_im = next_im.clamp(-2.4, 2.4);
        let radius = (z_re * z_re + z_im * z_im).sqrt();
        let orbit = (z_im.atan2(z_re) / PI).sin();
        accum += orbit / (1.0 + 0.35 * iter as f64 + radius);
        if radius > 2.0 {
            accum += 1.0 / (1.0 + iter as f64);
            break;
        }
    }

    soft_clip(accum * 2.6)
}

fn strange_attractor_oscillator(state: &mut StrangeAttractorState, drive: f64) -> f64 {
    let drive = drive.clamp(0.2, 2.0);
    let a = -1.4 + 0.18 * drive;
    let b = 1.56 + 0.11 * drive;
    let c = 1.22 - 0.07 * drive;
    let d = -0.91 + 0.05 * drive;
    let next_x = (a * state.y).sin() - (b * state.x).cos();
    let next_y = (c * state.x).sin() - (d * state.y).cos();
    state.x = next_x.clamp(-2.0, 2.0);
    state.y = next_y.clamp(-2.0, 2.0);
    soft_clip(0.6 * state.x + 0.4 * state.y)
}

fn fractal_quasi_oscillator(
    koch: f64,
    mandelbrot: f64,
    strange: f64,
    chaos: f64,
    fold: f64,
) -> f64 {
    let chaos = chaos.clamp(0.0, 1.5);
    let fold = fold.clamp(0.25, 5.0);
    let blend = koch + (0.85 + 0.25 * chaos) * mandelbrot + (0.9 + 0.2 * chaos) * strange;
    soft_clip(
        (((blend * fold).sin()) + 0.45 * (koch * mandelbrot) + 0.3 * (strange * blend).tanh())
            / 1.75,
    )
}

fn speech_oscillator_sample(
    osc: SpeechOscillator,
    pitch: f64,
    formant1: f64,
    formant2: f64,
    voiced: bool,
    duty_cycle: f64,
    chaos: f64,
    fold: f64,
    t: f64,
    phase: &mut f64,
    koch_t: &mut f64,
    mandel_t: &mut f64,
    attractor: &mut StrangeAttractorState,
    rng: &mut ChaCha8Rng,
) -> f64 {
    let freq = pitch.max(20.0);
    *phase = (*phase + freq / 44_100.0).fract();
    *koch_t += 1.0 / 44_100.0;
    *mandel_t += 1.0 / 44_100.0;
    match osc {
        SpeechOscillator::Pulse => {
            if *phase < duty_cycle {
                1.0
            } else {
                -1.0
            }
        }
        SpeechOscillator::Triangle => triangle(freq, t),
        SpeechOscillator::Saw => saw(freq, t),
        SpeechOscillator::Noise => rng.gen_range(-1.0_f64..1.0_f64),
        SpeechOscillator::Buzz => {
            let mut s = 0.0;
            for h in 1..=5 {
                s += saw(freq * h as f64, t) / h as f64;
            }
            soft_clip(s)
        }
        SpeechOscillator::Formant => {
            0.62 * (2.0 * PI * formant1 * t).sin() + 0.38 * (2.0 * PI * formant2 * t).sin()
        }
        SpeechOscillator::Ring => {
            (2.0 * PI * freq * t).sin() * (2.0 * PI * (formant2 * 0.5) * t).sin()
        }
        SpeechOscillator::Fold => {
            let carrier = (2.0 * PI * freq * t).sin() + 0.6 * saw(freq * 2.0, t);
            soft_clip((carrier * fold).sin())
        }
        SpeechOscillator::Koch => koch_quasi_oscillator(freq * (1.0 + 0.25 * chaos), *koch_t, 5),
        SpeechOscillator::Mandelbrot => {
            mandelbrot_quasi_oscillator(freq * (1.0 + 0.3 * chaos), *mandel_t, 18)
        }
        SpeechOscillator::Strange => strange_attractor_oscillator(attractor, 0.8 + chaos * 0.6),
        SpeechOscillator::Phoneme => {
            let source = if voiced {
                0.7 * (2.0 * PI * freq * t).sin() + 0.3 * saw(freq * 2.0, t)
            } else {
                rng.gen_range(-1.0_f64..1.0_f64) * 0.8
            };
            soft_clip(
                source
                    + 0.42 * (2.0 * PI * formant1 * t).sin()
                    + 0.28 * (2.0 * PI * formant2 * t).sin(),
            )
        }
    }
}

fn apply_speech_chip_fx(
    sample: f64,
    opts: &SpeechRenderOptions,
    tuning: &SpeechProfileTuning,
    sample_hold_counter: &mut f64,
    held: &mut f64,
) -> f64 {
    let hold_hz = opts.sample_hold_hz.min(tuning.default_hold_hz).max(500.0);
    *sample_hold_counter += hold_hz / opts.sample_rate as f64;
    if *sample_hold_counter >= 1.0 {
        *sample_hold_counter -= 1.0;
        *held = sample;
    }
    let held_mix = opts.resampler_grit.clamp(0.0, 1.0);
    let crushed_input = sample * (1.0 - held_mix) + *held * held_mix;
    let crush_bits = opts
        .bitcrush_bits
        .min(tuning.default_bitcrush_bits)
        .max(1.0);
    let levels = 2.0_f64.powf(crush_bits);
    let crushed = (crushed_input * levels).round() / levels;
    let folded = soft_clip(crushed * (opts.fold + tuning.default_fold) * 0.5);
    soft_clip(0.7 * crushed + 0.3 * folded)
}

fn soft_clip(x: f64) -> f64 {
    x.tanh()
}

fn wav_spec(channels: u16, sample_rate: u32, output_encoding: OutputEncoding) -> WavSpec {
    WavSpec {
        channels,
        sample_rate,
        bits_per_sample: output_encoding.bits_per_sample(),
        sample_format: output_encoding.sample_format(),
    }
}

fn quantize_signed(sample: f64, peak: f64) -> i32 {
    let clamped = sample.clamp(-1.0, 1.0);
    let scaled = (clamped * peak).round();
    scaled.clamp(-peak - 1.0, peak) as i32
}

fn write_encoded_sample<W>(
    writer: &mut WavWriter<W>,
    sample: f64,
    output_encoding: OutputEncoding,
) -> Result<()>
where
    W: Write + Seek,
{
    match output_encoding {
        OutputEncoding::Float32 => writer.write_sample(sample.clamp(-1.0, 1.0) as f32)?,
        OutputEncoding::Int16 => {
            writer.write_sample(quantize_signed(sample, i16::MAX as f64) as i16)?
        }
        OutputEncoding::Int24 => writer.write_sample(quantize_signed(sample, 8_388_607.0))?,
        OutputEncoding::Int32 => writer.write_sample(quantize_signed(sample, i32::MAX as f64))?,
    }
    Ok(())
}

fn write_wav_mono(
    path: &Path,
    sample_rate: u32,
    samples: &[f64],
    output_encoding: OutputEncoding,
) -> Result<()> {
    create_parent_dir(path)?;
    let mut writer = WavWriter::create(path, wav_spec(1, sample_rate, output_encoding))?;
    for &sample in samples {
        write_encoded_sample(&mut writer, sample, output_encoding)?;
    }
    writer.finalize()?;
    Ok(())
}

fn write_wav_channels(
    path: &Path,
    sample_rate: u32,
    channels: &[Vec<f64>],
    output_encoding: OutputEncoding,
) -> Result<()> {
    if channels.is_empty() {
        return Err(anyhow!("no channels to write"));
    }
    let frames = channels.iter().map(|c| c.len()).min().unwrap_or(0);
    if frames == 0 {
        return Err(anyhow!("no frames to write"));
    }

    create_parent_dir(path)?;
    let mut writer = WavWriter::create(
        path,
        wav_spec(channels.len() as u16, sample_rate, output_encoding),
    )?;
    for i in 0..frames {
        for ch in channels {
            write_encoded_sample(&mut writer, ch[i], output_encoding)?;
        }
    }
    writer.finalize()?;
    Ok(())
}

fn resample_linear(samples: &[f64], input_sample_rate: u32, output_sample_rate: u32) -> Vec<f64> {
    if input_sample_rate == output_sample_rate || samples.len() <= 1 {
        return samples.to_vec();
    }
    let ratio = output_sample_rate as f64 / input_sample_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let idx = src_pos.floor() as usize;
        let frac = src_pos - idx as f64;
        let a = samples[idx.min(samples.len() - 1)];
        let b = samples[(idx + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

fn resample_channels_linear(
    channels: &[Vec<f64>],
    input_sample_rate: u32,
    output_sample_rate: u32,
) -> Vec<Vec<f64>> {
    channels
        .iter()
        .map(|channel| resample_linear(channel, input_sample_rate, output_sample_rate))
        .collect()
}

fn estimate_true_peak(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut peak = samples.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    for window in samples.windows(2) {
        let a = window[0];
        let b = window[1];
        for step in 1..=4 {
            let t = step as f64 / 5.0;
            let v = a + (b - a) * t;
            peak = peak.max(v.abs());
        }
    }
    peak
}

fn estimate_true_peak_channels(channels: &[Vec<f64>]) -> f64 {
    channels
        .iter()
        .map(|channel| estimate_true_peak(channel))
        .fold(0.0_f64, f64::max)
}

fn normalize_peak_dbfs(samples: &mut [f64], target_dbfs: f64) {
    let target_peak = 10.0_f64.powf(target_dbfs / 20.0).abs();
    let peak = estimate_true_peak(samples);
    if peak <= EPS64 || target_peak <= EPS64 {
        return;
    }
    let scale = target_peak / peak;
    for x in samples.iter_mut() {
        *x *= scale;
    }
    let corrected_peak = estimate_true_peak(samples);
    if corrected_peak > target_peak + 1e-9 {
        let correction = target_peak / corrected_peak;
        for x in samples.iter_mut() {
            *x *= correction;
        }
    }
}

fn normalize_peak_dbfs_channels(channels: &mut [Vec<f64>], target_dbfs: f64) {
    let target_peak = 10.0_f64.powf(target_dbfs / 20.0).abs();
    if target_peak <= EPS64 {
        return;
    }
    let peak = estimate_true_peak_channels(channels);
    if peak <= EPS64 {
        return;
    }
    let scale = target_peak / peak;
    for ch in channels.iter_mut() {
        for x in ch.iter_mut() {
            *x *= scale;
        }
    }
    let corrected_peak = estimate_true_peak_channels(channels);
    if corrected_peak > target_peak + 1e-9 {
        let correction = target_peak / corrected_peak;
        for ch in channels.iter_mut() {
            for x in ch.iter_mut() {
                *x *= correction;
            }
        }
    }
}

fn apply_backend_post(samples: &mut [f64], plan: &BackendPlan) -> Result<()> {
    match plan.active {
        RenderBackend::Cpu | RenderBackend::Auto => Ok(()),
        RenderBackend::Metal | RenderBackend::Cuda => {
            if samples.len() < 2048 || plan.jobs <= 1 {
                return Ok(());
            }
            match plan.active {
                RenderBackend::Metal => {
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    {
                        backend_metal::post_fx_in_place(
                            samples,
                            plan.gpu_drive,
                            plan.gpu_crush_bits,
                            plan.gpu_crush_mix,
                            plan.jobs,
                        )?;
                        Ok(())
                    }
                    #[cfg(not(all(feature = "metal", target_os = "macos")))]
                    {
                        Err(anyhow!(
                            "metal backend requested but this build does not include Metal support"
                        ))
                    }
                }
                RenderBackend::Cuda => {
                    #[cfg(feature = "cuda")]
                    {
                        backend_cuda::post_fx_in_place(
                            samples,
                            plan.gpu_drive,
                            plan.gpu_crush_bits,
                            plan.gpu_crush_mix,
                            plan.jobs,
                        )?;
                        Ok(())
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        Err(anyhow!(
                            "cuda backend requested but this build does not include CUDA support"
                        ))
                    }
                }
                _ => Ok(()),
            }
        }
    }
}

fn create_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn read_wav_channels_f64(path: &Path) -> Result<(Vec<Vec<f64>>, u32, u16)> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels.max(1);
    let sample_rate = spec.sample_rate;

    let interleaved: Vec<f64> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f64 / i16::MAX as f64))
            .collect::<result::Result<Vec<_>, _>>()?,
        (SampleFormat::Int, 24 | 32) => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f64 / i32::MAX as f64))
            .collect::<result::Result<Vec<_>, _>>()?,
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .map(|s| s.map(|v| v as f64))
            .collect::<result::Result<Vec<_>, _>>()?,
        _ => {
            return Err(anyhow!(
                "unsupported WAV format: {:?} {}-bit",
                spec.sample_format,
                spec.bits_per_sample
            ));
        }
    };

    let mut per_channel = vec![Vec::new(); channels as usize];
    for frame in interleaved.chunks(channels as usize) {
        for (ch, &sample) in frame.iter().enumerate() {
            per_channel[ch].push(sample);
        }
        if frame.len() < channels as usize {
            let last = *frame.last().unwrap_or(&0.0);
            for ch_buf in per_channel.iter_mut().skip(frame.len()) {
                ch_buf.push(last);
            }
        }
    }
    Ok((per_channel, sample_rate, channels))
}

fn mixdown_mono(channel_data: &[Vec<f64>]) -> Vec<f64> {
    if channel_data.is_empty() {
        return Vec::new();
    }
    if channel_data.len() == 1 {
        return channel_data[0].clone();
    }
    let frames = channel_data.iter().map(|c| c.len()).min().unwrap_or(0);
    let ch_count = channel_data.len() as f64;
    let mut mono = Vec::with_capacity(frames);
    for i in 0..frames {
        let sum: f64 = channel_data.iter().map(|ch| ch[i]).sum();
        mono.push(sum / ch_count);
    }
    mono
}

fn analyze_samples_base(samples: &[f64], sample_rate: u32, channels: u16) -> Analysis {
    let len = samples.len().max(1) as f64;
    let mut peak = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut clipped_count = 0usize;
    let mut zero_crossings = 0usize;
    let mut prev = samples[0];
    let mut diff_energy = 0.0_f64;

    for &x in samples {
        let ax = x.abs();
        if ax > peak {
            peak = ax;
        }
        sum_sq += x * x;
        if ax >= 0.98 {
            clipped_count += 1;
        }
        if (x >= 0.0 && prev < 0.0) || (x < 0.0 && prev >= 0.0) {
            zero_crossings += 1;
        }
        let d = x - prev;
        diff_energy += d * d;
        prev = x;
    }

    let rms = (sum_sq / len).sqrt().max(EPS64);
    let peak_safe = peak.max(EPS64);
    let duration_s = samples.len() as f64 / sample_rate as f64;
    let peak_dbfs = 20.0 * peak_safe.log10();
    let rms_dbfs = 20.0 * rms.log10();
    let crest_factor_db = 20.0 * (peak_safe / rms).log10();
    let zero_crossing_rate = zero_crossings as f64 / len;
    let clipped_pct = (clipped_count as f64 / len) * 100.0;
    let harshness_ratio = (diff_energy / sum_sq.max(EPS64)).sqrt();
    let ugly_index = ((clipped_pct * 1.6 + harshness_ratio * 45.0 + zero_crossing_rate * 200.0)
        * 10.0)
        .clamp(0.0, 1000.0);

    Analysis {
        sample_rate,
        channels,
        duration_s,
        peak_dbfs,
        rms_dbfs,
        crest_factor_db,
        zero_crossing_rate,
        clipped_pct,
        harshness_ratio,
        ugly_index,
    }
}

fn analyze_samples_psycho(
    samples: &[f64],
    sample_rate: u32,
    basic: &Analysis,
    opts: &AnalyzeOptions,
    stereo_pair: Option<(&[f64], &[f64])>,
) -> PsychoAnalysis {
    let spectra = stft_mag_spectra(samples, opts.fft_size, opts.hop_size);
    let nyquist = sample_rate as f64 / 2.0;
    let stereo_spectra = stereo_pair.map(|(l, r)| {
        (
            stft_mag_spectra(l, opts.fft_size, opts.hop_size),
            stft_mag_spectra(r, opts.fft_size, opts.hop_size),
        )
    });

    let sharpness_norm = compute_sharpness_norm(&spectra, sample_rate);
    let roughness_norm = compute_roughness_norm(&spectra, sample_rate);
    let dissonance_norm = compute_dissonance_norm(&spectra, sample_rate);
    let transient_norm = compute_transient_norm(&spectra);
    let harmonicity_norm = compute_harmonicity_norm(&spectra, sample_rate);
    let inharmonicity_norm = (1.0 - harmonicity_norm).clamp(0.0, 1.0);
    let binaural_beat_norm = compute_binaural_beat_norm(
        &spectra,
        stereo_spectra
            .as_ref()
            .map(|(left, right)| (left.as_slice(), right.as_slice())),
        sample_rate,
    );
    let beat_conflict_norm = compute_beat_conflict_norm(&spectra, sample_rate);
    let tritone_tension_norm = compute_tritone_tension_norm(&spectra, sample_rate);
    let wolf_fifth_norm = compute_wolf_fifth_norm(&spectra, sample_rate);
    let clip_norm = (basic.clipped_pct / 10.0).clamp(0.0, 1.0);
    let harshness_norm = (basic.harshness_ratio / (0.28 + 0.00001 * nyquist)).clamp(0.0, 1.0);

    let weighted_sum = -4.05
        + 1.6 * clip_norm
        + 1.3 * roughness_norm
        + 1.0 * sharpness_norm
        + 1.0 * dissonance_norm
        + 1.2 * transient_norm
        + 0.9 * harshness_norm
        + 1.25 * inharmonicity_norm
        + 0.85 * binaural_beat_norm
        + 1.05 * beat_conflict_norm
        + 0.85 * tritone_tension_norm
        + 0.75 * wolf_fifth_norm
        - 0.45 * harmonicity_norm;

    let ugly_index = 1000.0 * sigmoid(weighted_sum);

    PsychoAnalysis {
        clip_norm,
        harshness_norm,
        roughness_norm,
        sharpness_norm,
        dissonance_norm,
        transient_norm,
        harmonicity_norm,
        inharmonicity_norm,
        binaural_beat_norm,
        beat_conflict_norm,
        tritone_tension_norm,
        wolf_fifth_norm,
        weighted_sum,
        ugly_index: ugly_index.clamp(0.0, 1000.0),
        fft_size: opts.fft_size,
        hop_size: opts.hop_size,
    }
}

fn stft_mag_spectra(samples: &[f64], fft_size: usize, hop_size: usize) -> Vec<Vec<f64>> {
    if samples.is_empty() || fft_size == 0 || hop_size == 0 {
        return Vec::new();
    }

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let window: Vec<f64> = (0..fft_size)
        .map(|n| {
            let x = 2.0 * PI * n as f64 / (fft_size.saturating_sub(1) as f64).max(1.0);
            0.5 - 0.5 * x.cos()
        })
        .collect();

    let mut starts = Vec::new();
    let mut start = 0usize;
    while start < samples.len() {
        starts.push(start);
        if start + fft_size >= samples.len() {
            break;
        }
        start += hop_size;
    }

    if starts.is_empty() {
        starts.push(0);
    }

    let mut frame = vec![Complex64::new(0.0, 0.0); fft_size];
    let mut spectra = Vec::with_capacity(starts.len());

    for start in starts {
        frame.fill(Complex64::new(0.0, 0.0));
        let end = (start + fft_size).min(samples.len());
        for (i, &x) in samples[start..end].iter().enumerate() {
            frame[i].re = x * window[i];
        }

        fft.process(&mut frame);
        let bins = fft_size / 2 + 1;
        let mags = frame.iter().take(bins).map(|c| c.norm_sqr()).collect();
        spectra.push(mags);
    }

    spectra
}

fn compute_sharpness_norm(spectra: &[Vec<f64>], sample_rate: u32) -> f64 {
    if spectra.is_empty() {
        return 0.0;
    }
    let nyquist = sample_rate as f64 / 2.0;
    let mut centroid_acc = 0.0_f64;
    let mut hf_ratio_acc = 0.0_f64;
    let mut frames = 0.0_f64;

    for mags in spectra {
        if mags.is_empty() {
            continue;
        }
        let mut sum_mag = 0.0;
        let mut centroid_num = 0.0;
        let mut hf_mag = 0.0;
        for (k, &m) in mags.iter().enumerate() {
            let freq = (k as f64 / (mags.len() as f64 - 1.0).max(1.0)) * nyquist;
            sum_mag += m;
            centroid_num += freq * m;
            if freq >= 3000.0 {
                hf_mag += m;
            }
        }
        let centroid = centroid_num / sum_mag.max(EPS64);
        let centroid_norm = (centroid / nyquist.max(EPS64)).clamp(0.0, 1.0);
        let hf_ratio = (hf_mag / sum_mag.max(EPS64)).clamp(0.0, 1.0);
        centroid_acc += centroid_norm;
        hf_ratio_acc += hf_ratio;
        frames += 1.0;
    }

    let raw = 0.45 * (centroid_acc / frames.max(1.0)) + 0.55 * (hf_ratio_acc / frames.max(1.0));
    finite01(raw)
}

fn compute_roughness_norm(spectra: &[Vec<f64>], sample_rate: u32) -> f64 {
    if spectra.len() < 2 || spectra[0].is_empty() {
        return 0.0;
    }
    let bins = spectra[0].len();
    let nyquist = sample_rate as f64 / 2.0;
    let mut band_bin_idx: Vec<Vec<usize>> = vec![Vec::new(); BARK_BANDS];

    for k in 0..bins {
        let freq = (k as f64 / (bins as f64 - 1.0).max(1.0)) * nyquist;
        let band = bark_hz(freq).floor() as usize;
        if band < BARK_BANDS {
            band_bin_idx[band].push(k);
        }
    }

    let mut band_energy = vec![vec![0.0_f64; spectra.len()]; BARK_BANDS];
    for (t, mags) in spectra.iter().enumerate() {
        for b in 0..BARK_BANDS {
            if band_bin_idx[b].is_empty() {
                continue;
            }
            let e = band_bin_idx[b].iter().map(|&k| mags[k]).sum::<f64>();
            band_energy[b][t] = e;
        }
    }

    let mut weighted = 0.0;
    let mut wsum = 0.0;
    for (b, series) in band_energy.iter().enumerate() {
        let center_bark = b as f64 + 0.5;
        let w = ((-(center_bark - 11.0).powi(2)) / (2.0 * 5.0_f64.powi(2))).exp() + 0.15;
        let mut sum_diff = 0.0;
        for t in 1..series.len() {
            let a = (1.0 + series[t]).ln();
            let p = (1.0 + series[t - 1]).ln();
            sum_diff += (a - p).abs();
        }
        let mean_diff = sum_diff / (series.len() as f64 - 1.0).max(1.0);
        weighted += w * mean_diff;
        wsum += w;
    }
    let rough_raw = weighted / wsum.max(EPS64);
    finite01(rough_raw / 0.17)
}

fn compute_dissonance_norm(spectra: &[Vec<f64>], sample_rate: u32) -> f64 {
    if spectra.is_empty() || spectra[0].len() < 4 {
        return 0.0;
    }
    let nyquist = sample_rate as f64 / 2.0;
    let mut frame_sum = 0.0_f64;
    let mut used_frames = 0.0_f64;

    for mags in spectra {
        let mut peaks: Vec<(usize, f64)> = Vec::new();
        for k in 2..mags.len() - 2 {
            let m = mags[k];
            if m > mags[k - 1] && m >= mags[k + 1] {
                peaks.push((k, m.sqrt()));
            }
        }
        peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
        peaks.truncate(12);
        if peaks.len() < 2 {
            continue;
        }

        let amp_sum: f64 = peaks.iter().map(|p| p.1).sum();
        let mut diss = 0.0;
        for i in 0..peaks.len() {
            for j in i + 1..peaks.len() {
                let fi = (peaks[i].0 as f64 / (mags.len() as f64 - 1.0).max(1.0)) * nyquist;
                let fj = (peaks[j].0 as f64 / (mags.len() as f64 - 1.0).max(1.0)) * nyquist;
                let dz = (bark_hz(fi) - bark_hz(fj)).abs();
                let curve = (-3.5 * dz).exp() - (-5.75 * dz).exp();
                if curve > 0.0 {
                    diss += (peaks[i].1 * peaks[j].1).sqrt() * curve;
                }
            }
        }
        frame_sum += diss / amp_sum.max(EPS64);
        used_frames += 1.0;
    }

    if used_frames <= 0.0 {
        return 0.0;
    }
    finite01((frame_sum / used_frames) / 0.15)
}

fn frame_peaks_hz(mags: &[f64], sample_rate: u32, max_peaks: usize) -> Vec<(f64, f64)> {
    if mags.len() < 5 || max_peaks == 0 {
        return Vec::new();
    }
    let nyquist = sample_rate as f64 / 2.0;
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    for k in 2..mags.len() - 2 {
        let m = mags[k];
        if m > mags[k - 1] && m >= mags[k + 1] {
            let freq = (k as f64 / (mags.len() as f64 - 1.0).max(1.0)) * nyquist;
            peaks.push((freq, m.sqrt()));
        }
    }
    peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
    peaks.truncate(max_peaks);
    peaks
}

fn compute_harmonicity_norm(spectra: &[Vec<f64>], sample_rate: u32) -> f64 {
    if spectra.is_empty() {
        return 0.0;
    }

    let mut score_sum = 0.0_f64;
    let mut frames = 0.0_f64;

    for mags in spectra {
        let peaks = frame_peaks_hz(mags, sample_rate, 16);
        if peaks.len() < 3 {
            continue;
        }
        let f0 = peaks[0].0.max(35.0);
        let amp_sum: f64 = peaks.iter().map(|(_, a)| *a).sum();
        if amp_sum <= EPS64 {
            continue;
        }

        let mut frame_score = 0.0;
        for (freq, amp) in &peaks {
            let n = (*freq / f0).round().max(1.0);
            let harmonic_freq = f0 * n;
            let cents = 1200.0 * ((*freq / harmonic_freq.max(EPS64)).log2()).abs();
            let in_tune = (-(cents / 32.0).powi(2)).exp();
            frame_score += (*amp / amp_sum) * in_tune;
        }
        score_sum += frame_score.clamp(0.0, 1.0);
        frames += 1.0;
    }

    finite01(score_sum / frames.max(1.0))
}

fn compute_binaural_beat_norm(
    mono_spectra: &[Vec<f64>],
    stereo_spectra: Option<(&[Vec<f64>], &[Vec<f64>])>,
    sample_rate: u32,
) -> f64 {
    match stereo_spectra {
        Some((left, right)) => {
            let frames_n = left.len().min(right.len());
            if frames_n == 0 {
                return 0.0;
            }
            let mut acc = 0.0_f64;
            let mut used = 0.0_f64;

            for idx in 0..frames_n {
                let peaks_l = frame_peaks_hz(&left[idx], sample_rate, 10)
                    .into_iter()
                    .filter(|(f, _)| (80.0..=2000.0).contains(f))
                    .collect::<Vec<_>>();
                let peaks_r = frame_peaks_hz(&right[idx], sample_rate, 10)
                    .into_iter()
                    .filter(|(f, _)| (80.0..=2000.0).contains(f))
                    .collect::<Vec<_>>();
                if peaks_l.is_empty() || peaks_r.is_empty() {
                    continue;
                }
                let mut weighted = 0.0_f64;
                let mut wsum = 0.0_f64;
                for (fl, al) in &peaks_l {
                    for (fr, ar) in &peaks_r {
                        let df = (fl - fr).abs();
                        if !(0.5..=40.0).contains(&df) {
                            continue;
                        }
                        let pair_w = (al * ar).sqrt();
                        let beat_band = (-(df - 7.0).powi(2) / (2.0 * 5.0_f64.powi(2))).exp();
                        weighted += pair_w * beat_band;
                        wsum += pair_w;
                    }
                }
                if wsum > EPS64 {
                    acc += weighted / wsum;
                    used += 1.0;
                }
            }
            finite01((acc / used.max(1.0)) / 0.36)
        }
        None => {
            if mono_spectra.is_empty() {
                return 0.0;
            }
            // Mono fallback: estimate binaural-beat potential from narrow close-frequency pairs.
            let mut acc = 0.0_f64;
            let mut used = 0.0_f64;
            for mags in mono_spectra {
                let peaks = frame_peaks_hz(mags, sample_rate, 12)
                    .into_iter()
                    .filter(|(f, _)| (80.0..=2000.0).contains(f))
                    .collect::<Vec<_>>();
                if peaks.len() < 2 {
                    continue;
                }
                let mut weighted = 0.0_f64;
                let mut wsum = 0.0_f64;
                for i in 0..peaks.len() {
                    for j in i + 1..peaks.len() {
                        let df = (peaks[i].0 - peaks[j].0).abs();
                        if !(0.5..=40.0).contains(&df) {
                            continue;
                        }
                        let pair_w = (peaks[i].1 * peaks[j].1).sqrt();
                        let beat_band = (-(df - 7.0).powi(2) / (2.0 * 5.5_f64.powi(2))).exp();
                        weighted += pair_w * beat_band;
                        wsum += pair_w;
                    }
                }
                if wsum > EPS64 {
                    acc += (weighted / wsum) * 0.65;
                    used += 1.0;
                }
            }
            finite01((acc / used.max(1.0)) / 0.30)
        }
    }
}

fn compute_beat_conflict_norm(spectra: &[Vec<f64>], sample_rate: u32) -> f64 {
    if spectra.is_empty() {
        return 0.0;
    }

    let mut acc = 0.0_f64;
    let mut frames = 0.0_f64;

    for mags in spectra {
        let peaks = frame_peaks_hz(mags, sample_rate, 12);
        if peaks.len() < 2 {
            continue;
        }

        let mut pair_energy = 0.0;
        let mut beat_energy = 0.0;
        for i in 0..peaks.len() {
            for j in i + 1..peaks.len() {
                let df = (peaks[i].0 - peaks[j].0).abs();
                if !(0.4..=20.0).contains(&df) {
                    continue;
                }
                let pair_w = (peaks[i].1 * peaks[j].1).sqrt();
                // Most objectionable beating tends to sit in the low single-digit Hz.
                let beat_weight = (-(df - 4.0).powi(2) / (2.0 * 2.8_f64.powi(2))).exp();
                beat_energy += pair_w * beat_weight;
                pair_energy += pair_w;
            }
        }
        if pair_energy > EPS64 {
            acc += beat_energy / pair_energy;
            frames += 1.0;
        }
    }

    finite01((acc / frames.max(1.0)) / 0.42)
}

fn compute_tritone_tension_norm(spectra: &[Vec<f64>], sample_rate: u32) -> f64 {
    if spectra.is_empty() {
        return 0.0;
    }

    let mut acc = 0.0_f64;
    let mut frames = 0.0_f64;
    for mags in spectra {
        let peaks = frame_peaks_hz(mags, sample_rate, 12);
        if peaks.len() < 2 {
            continue;
        }
        let mut weighted = 0.0;
        let mut wsum = 0.0;
        for i in 0..peaks.len() {
            for j in i + 1..peaks.len() {
                let lo = peaks[i].0.min(peaks[j].0).max(EPS64);
                let hi = peaks[i].0.max(peaks[j].0);
                let mut cents = 1200.0 * (hi / lo).log2();
                while cents > 1200.0 {
                    cents -= 1200.0;
                }
                let pair_w = (peaks[i].1 * peaks[j].1).sqrt();
                let close_to_tritone = (-(cents - 600.0).powi(2) / (2.0 * 26.0_f64.powi(2))).exp();
                weighted += pair_w * close_to_tritone;
                wsum += pair_w;
            }
        }
        if wsum > EPS64 {
            acc += weighted / wsum;
            frames += 1.0;
        }
    }
    finite01((acc / frames.max(1.0)) / 0.34)
}

fn compute_wolf_fifth_norm(spectra: &[Vec<f64>], sample_rate: u32) -> f64 {
    if spectra.is_empty() {
        return 0.0;
    }

    let mut acc = 0.0_f64;
    let mut frames = 0.0_f64;
    for mags in spectra {
        let peaks = frame_peaks_hz(mags, sample_rate, 12);
        if peaks.len() < 2 {
            continue;
        }
        let mut weighted = 0.0;
        let mut wsum = 0.0;
        for i in 0..peaks.len() {
            for j in i + 1..peaks.len() {
                let lo = peaks[i].0.min(peaks[j].0).max(EPS64);
                let hi = peaks[i].0.max(peaks[j].0);
                let mut cents = 1200.0 * (hi / lo).log2();
                while cents > 1200.0 {
                    cents -= 1200.0;
                }
                let pair_w = (peaks[i].1 * peaks[j].1).sqrt();
                // Center around the infamous wolf fifth neighborhood (~678 cents).
                let close_to_wolf = (-(cents - 678.0).powi(2) / (2.0 * 28.0_f64.powi(2))).exp();
                weighted += pair_w * close_to_wolf;
                wsum += pair_w;
            }
        }
        if wsum > EPS64 {
            acc += weighted / wsum;
            frames += 1.0;
        }
    }
    finite01((acc / frames.max(1.0)) / 0.31)
}

fn compute_transient_norm(spectra: &[Vec<f64>]) -> f64 {
    if spectra.len() < 2 || spectra[0].is_empty() {
        return 0.0;
    }
    let mut flux_acc = 0.0_f64;
    let mut frames = 0.0_f64;

    for t in 1..spectra.len() {
        let curr = &spectra[t];
        let prev = &spectra[t - 1];
        let mut flux = 0.0;
        let mut energy = 0.0;
        for k in 0..curr.len().min(prev.len()) {
            let d = curr[k] - prev[k];
            if d > 0.0 {
                flux += d;
            }
            energy += curr[k];
        }
        flux_acc += flux / energy.max(EPS64);
        frames += 1.0;
    }

    finite01((flux_acc / frames.max(1.0)) / 0.45)
}

fn apply_go_ugly_to_channel(
    buffer: &mut [f64],
    sample_rate: f64,
    level: u16,
    flavor: GoFlavor,
    seed: u64,
    contour: Option<&UglinessContour>,
) {
    if buffer.is_empty() {
        return;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let intensity = (level as f64 / 1000.0).clamp(0.001, 1.0);
    for x in buffer.iter_mut() {
        *x = soft_clip(*x * (1.0 + 1.5 * intensity));
    }

    let resolved = match flavor {
        GoFlavor::Random => random_go_mode(&mut rng),
        GoFlavor::Lucky => {
            if rng.gen_bool(0.5) {
                GoFlavor::Glitch
            } else {
                GoFlavor::Geek
            }
        }
        other => other,
    };

    match resolved {
        GoFlavor::Glitch => {
            effect_crush(buffer, sample_rate, &mut rng);
            effect_stutter(buffer, sample_rate, &mut rng);
            go_dropouts(buffer, sample_rate, intensity, &mut rng);
        }
        GoFlavor::Stutter => {
            let reps = if intensity > 0.7 { 3 } else { 2 };
            for _ in 0..reps {
                effect_stutter(buffer, sample_rate, &mut rng);
            }
        }
        GoFlavor::Puff => {
            effect_pop(buffer, sample_rate, &mut rng);
            go_puffs(buffer, sample_rate, intensity, &mut rng);
        }
        GoFlavor::Punish => {
            effect_gate(buffer, sample_rate, 0);
            effect_crush(buffer, sample_rate, &mut rng);
            effect_smear(buffer, sample_rate);
            for x in buffer.iter_mut() {
                *x = soft_clip(*x * (2.6 + 2.2 * intensity));
            }
        }
        GoFlavor::Geek => {
            go_sample_hold(buffer, sample_rate, intensity, &mut rng);
            go_ring_mod(buffer, sample_rate, intensity);
            effect_crush(buffer, sample_rate, &mut rng);
        }
        GoFlavor::DissonanceRing => {
            effect_dissonance_ring(buffer, sample_rate, intensity, 0);
            if intensity > 0.55 {
                effect_crush(buffer, sample_rate, &mut rng);
            }
        }
        GoFlavor::DissonanceExpand => {
            effect_dissonance_expand(buffer, sample_rate, intensity);
            if intensity > 0.72 {
                effect_smear(buffer, sample_rate);
            }
        }
        GoFlavor::Lucky => {}
        GoFlavor::Random => {}
    }

    if intensity > 0.75 {
        effect_pop(buffer, sample_rate, &mut rng);
    }
    if let Some(c) = contour {
        apply_ugliness_contour(buffer, sample_rate, c, derive_seed(seed, 0xC07E_0001));
    }
}

fn random_go_mode(rng: &mut ChaCha8Rng) -> GoFlavor {
    match rng.gen_range(0..7) {
        0 => GoFlavor::Glitch,
        1 => GoFlavor::Stutter,
        2 => GoFlavor::Puff,
        3 => GoFlavor::Punish,
        4 => GoFlavor::Geek,
        5 => GoFlavor::DissonanceRing,
        _ => GoFlavor::DissonanceExpand,
    }
}

fn contour_level_at(contour: &UglinessContour, t_norm: f64) -> f64 {
    if contour.points.is_empty() {
        return 1.0;
    }
    if contour.points.len() == 1 {
        return contour.points[0].level as f64;
    }
    let t = t_norm.clamp(0.0, 1.0);
    if t <= contour.points[0].t {
        return contour.points[0].level as f64;
    }
    for pair in contour.points.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        if t <= b.t {
            if (b.t - a.t).abs() <= EPS64 {
                return b.level as f64;
            }
            match contour.interpolation {
                ContourInterpolation::Step => return a.level as f64,
                ContourInterpolation::Linear => {
                    let alpha = ((t - a.t) / (b.t - a.t)).clamp(0.0, 1.0);
                    return a.level as f64 + alpha * (b.level as f64 - a.level as f64);
                }
            }
        }
    }
    contour.points.last().map(|p| p.level as f64).unwrap_or(1.0)
}

fn apply_ugliness_contour(
    buffer: &mut [f64],
    sample_rate: f64,
    contour: &UglinessContour,
    seed: u64,
) {
    if buffer.is_empty() {
        return;
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let denom = (buffer.len() - 1).max(1) as f64;
    for (i, x) in buffer.iter_mut().enumerate() {
        let t_norm = i as f64 / denom;
        let level = contour_level_at(contour, t_norm).clamp(1.0, 1000.0);
        let intensity = (level / 1000.0).clamp(0.001, 1.0);
        let t = i as f64 / sample_rate.max(1.0);

        let drive = 1.0 + 5.5 * intensity;
        let wobble = 1.0 + 0.22 * intensity * (2.0 * PI * (2.5 + 27.0 * intensity) * t).sin();
        let bits = (16.0 - 12.0 * intensity).clamp(3.0, 16.0);
        let levels = 2.0_f64.powf(bits);
        let mut y = soft_clip(*x * drive * wobble);
        if rng.gen_bool((0.0002 + 0.02 * intensity).clamp(0.0, 0.3)) {
            y = soft_clip(y + rng.gen_range(-1.0_f64..1.0_f64) * 0.18 * intensity);
        }
        y = (y * levels).round() / levels;
        *x = soft_clip(y);
    }
}

fn go_dropouts(buffer: &mut [f64], sample_rate: f64, intensity: f64, rng: &mut ChaCha8Rng) {
    let events = ((buffer.len() as f64 / sample_rate) * (4.0 + 26.0 * intensity)).round() as usize;
    if events == 0 {
        return;
    }
    let max_len = (sample_rate * (0.008 + 0.06 * intensity)).max(8.0) as usize;
    for _ in 0..events {
        if buffer.len() <= 8 {
            break;
        }
        let start = rng.gen_range(0..buffer.len() - 1);
        let len = rng.gen_range(8..=max_len).min(buffer.len() - start);
        let held = if rng.gen_bool(0.65) {
            0.0
        } else {
            buffer[start]
        };
        for x in &mut buffer[start..start + len] {
            *x = held;
        }
    }
}

fn go_sample_hold(buffer: &mut [f64], sample_rate: f64, intensity: f64, rng: &mut ChaCha8Rng) {
    if buffer.is_empty() {
        return;
    }
    let hold = rng
        .gen_range((0.0004 * sample_rate) as usize..=(0.0045 * sample_rate) as usize)
        .max((1.0 + intensity * 5.0) as usize);
    let mut i = 0usize;
    while i < buffer.len() {
        let held = buffer[i];
        let end = (i + hold).min(buffer.len());
        for x in &mut buffer[i..end] {
            *x = held;
        }
        i = end;
    }
}

fn go_ring_mod(buffer: &mut [f64], sample_rate: f64, intensity: f64) {
    let f1 = 37.0 + 420.0 * intensity;
    let f2 = 111.0 + 900.0 * intensity;
    for (i, x) in buffer.iter_mut().enumerate() {
        let t = i as f64 / sample_rate;
        let modulator = 0.6 * (2.0 * PI * f1 * t).sin() + 0.4 * (2.0 * PI * f2 * t).sin();
        *x = soft_clip(*x * (1.0 + modulator));
    }
}

fn go_puffs(buffer: &mut [f64], sample_rate: f64, intensity: f64, rng: &mut ChaCha8Rng) {
    let events = ((buffer.len() as f64 / sample_rate) * (3.0 + 18.0 * intensity)).round() as usize;
    if events == 0 {
        return;
    }
    let max_len = (sample_rate * (0.02 + 0.11 * intensity)).max(16.0) as usize;
    for _ in 0..events {
        if buffer.len() <= 16 {
            break;
        }
        let start = rng.gen_range(0..buffer.len() - 1);
        let len = rng.gen_range(16..=max_len).min(buffer.len() - start);
        let amp = rng.gen_range(0.15_f64..(0.45 + 0.9 * intensity));
        for j in 0..len {
            let env = (1.0 - j as f64 / len as f64).powf(2.7);
            let noise = rng.gen_range(-1.0_f64..1.0_f64);
            let tone = (2.0 * PI * (50.0 + 240.0 * env) * (j as f64 / sample_rate)).sin();
            buffer[start + j] =
                soft_clip(buffer[start + j] + amp * env * (0.65 * noise + 0.35 * tone));
        }
    }
}

pub fn point_to_xyz(coord_system: CoordSystem, a: f64, b: f64, c: f64) -> [f64; 3] {
    match coord_system {
        CoordSystem::Cartesian => [a, b, c],
        CoordSystem::Polar => {
            let az = a.to_radians();
            let el = b.to_radians();
            let r = c.max(0.0);
            let x = r * el.cos() * az.sin();
            let y = r * el.cos() * az.cos();
            let z = r * el.sin();
            [x, y, z]
        }
    }
}

fn spatialize_mono(
    mono: &[f64],
    sample_rate: f64,
    layout: SurroundLayout,
    locus_xyz: [f64; 3],
    trajectory: Trajectory,
) -> Vec<Vec<f64>> {
    let (speaker_pos, lfe_idx) = speaker_positions(layout);
    let channels = speaker_pos.len();
    let frames = mono.len();
    let mut out = vec![vec![0.0_f64; frames]; channels];

    for (i, &m) in mono.iter().enumerate() {
        let t = if frames <= 1 {
            0.0
        } else {
            i as f64 / (frames - 1) as f64
        };
        let src = source_position_at(locus_xyz, trajectory, t);

        let mut weights = vec![0.0_f64; channels];
        let mut wsum = 0.0_f64;
        for (ch, spk) in speaker_pos.iter().enumerate() {
            if Some(ch) == lfe_idx {
                continue;
            }
            let d = distance3(src, *spk);
            let w = 1.0 / (0.1 + d.powf(1.7));
            weights[ch] = w;
            wsum += w;
        }
        let inv = if wsum > EPS64 { 1.0 / wsum } else { 0.0 };
        for ch in 0..channels {
            if Some(ch) == lfe_idx {
                continue;
            }
            out[ch][i] = m * weights[ch] * inv;
        }
    }

    if let Some(lfe) = lfe_idx {
        fill_lfe_channel(mono, &mut out[lfe], sample_rate);
    }
    out
}

fn source_position_at(locus: [f64; 3], trajectory: Trajectory, t: f64) -> [f64; 3] {
    match trajectory {
        Trajectory::Static => locus,
        Trajectory::Line { end } => [
            locus[0] + (end[0] - locus[0]) * t,
            locus[1] + (end[1] - locus[1]) * t,
            locus[2] + (end[2] - locus[2]) * t,
        ],
        Trajectory::Orbit { radius, turns } => {
            let ph = 2.0 * PI * turns * t;
            [
                locus[0] + radius * ph.cos(),
                locus[1] + radius * ph.sin(),
                locus[2],
            ]
        }
    }
}

fn speaker_positions(layout: SurroundLayout) -> (Vec<[f64; 3]>, Option<usize>) {
    match layout {
        SurroundLayout::Mono => (vec![[0.0, 1.0, 0.0]], None),
        SurroundLayout::Stereo => (vec![azimuth_deg(-30.0), azimuth_deg(30.0)], None),
        SurroundLayout::Quad => (
            vec![
                azimuth_deg(-45.0),
                azimuth_deg(45.0),
                azimuth_deg(-135.0),
                azimuth_deg(135.0),
            ],
            None,
        ),
        SurroundLayout::FiveOne => (
            vec![
                azimuth_deg(-30.0),  // L
                azimuth_deg(30.0),   // R
                azimuth_deg(0.0),    // C
                [0.0, 0.5, 0.0],     // LFE virtual
                azimuth_deg(-110.0), // Ls
                azimuth_deg(110.0),  // Rs
            ],
            Some(3),
        ),
        SurroundLayout::SevenOne => (
            vec![
                azimuth_deg(-30.0),  // L
                azimuth_deg(30.0),   // R
                azimuth_deg(0.0),    // C
                [0.0, 0.5, 0.0],     // LFE
                azimuth_deg(-90.0),  // Lss
                azimuth_deg(90.0),   // Rss
                azimuth_deg(-150.0), // Lrs
                azimuth_deg(150.0),  // Rrs
            ],
            Some(3),
        ),
        SurroundLayout::Custom(n) => {
            let n = n.max(1) as usize;
            if n == 1 {
                return (vec![[0.0, 1.0, 0.0]], None);
            }
            let mut pos = Vec::with_capacity(n);
            for i in 0..n {
                let az = -180.0 + 360.0 * (i as f64 / n as f64);
                pos.push(azimuth_deg(az));
            }
            (pos, None)
        }
    }
}

fn azimuth_deg(az_deg: f64) -> [f64; 3] {
    let az = az_deg.to_radians();
    [az.sin(), az.cos(), 0.0]
}

fn distance3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn fill_lfe_channel(mono: &[f64], lfe: &mut [f64], sample_rate: f64) {
    let fc = 120.0_f64;
    let alpha = (2.0 * PI * fc / sample_rate.max(1000.0)).min(0.45);
    let mut y = 0.0_f64;
    for (i, &x) in mono.iter().enumerate() {
        y += alpha * (x - y);
        lfe[i] = y * 0.75;
    }
}

fn bark_hz(freq_hz: f64) -> f64 {
    13.0 * (0.00076 * freq_hz).atan() + 3.5 * ((freq_hz / 7500.0).powi(2)).atan()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn finite01(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn styles_are_exposed() {
        let styles = available_styles();
        assert_eq!(styles.len(), 15);
        assert!(styles.contains(&Style::Harsh));
        assert!(styles.contains(&Style::Digital));
        assert!(styles.contains(&Style::Meltdown));
        assert!(styles.contains(&Style::Glitch));
        assert!(styles.contains(&Style::Pop));
        assert!(styles.contains(&Style::Buzz));
        assert!(styles.contains(&Style::Rub));
        assert!(styles.contains(&Style::Hum));
        assert!(styles.contains(&Style::Distort));
        assert!(styles.contains(&Style::Spank));
        assert!(styles.contains(&Style::Punish));
        assert!(styles.contains(&Style::Steal));
        assert!(styles.contains(&Style::Catastrophic));
        assert!(styles.contains(&Style::Wink));
        assert!(styles.contains(&Style::Lucky));
    }

    #[test]
    fn catastrophic_style_is_uglier_than_hum() {
        let hum = render_samples(16_384, 44_100.0, Style::Hum, 0.8, 7).expect("hum");
        let catastrophic =
            render_samples(16_384, 44_100.0, Style::Catastrophic, 0.8, 7).expect("catastrophic");
        let hum_report = analyze_samples_base(&hum, 44_100, 1);
        let catastrophic_report = analyze_samples_base(&catastrophic, 44_100, 1);
        assert!(catastrophic_report.ugly_index > hum_report.ugly_index + 150.0);
        assert!(catastrophic_report.harshness_ratio > hum_report.harshness_ratio);
    }

    #[test]
    fn named_fractal_oscillators_are_finite_and_bounded() {
        let mut attractor = StrangeAttractorState::new(0.2, -0.1);
        for i in 0..512 {
            let t = i as f64 / 44_100.0;
            let koch = koch_quasi_oscillator(137.0, t, 5);
            let mandelbrot = mandelbrot_quasi_oscillator(91.0 + i as f64 * 0.2, t, 20);
            let strange = strange_attractor_oscillator(&mut attractor, 1.15);
            let y = fractal_quasi_oscillator(koch, mandelbrot, strange, 1.1, 3.2);
            assert!(koch.is_finite());
            assert!(mandelbrot.is_finite());
            assert!(strange.is_finite());
            assert!(y.is_finite());
            assert!(y.abs() <= 1.0 + 1e-9, "sample out of bounds: {y}");
        }
    }

    #[test]
    fn speech_render_generates_non_silent_audio() {
        let opts = SpeechRenderOptions {
            text: "HELLO 1984 CHIP VOICE".to_string(),
            primary_osc: SpeechOscillator::Phoneme,
            secondary_osc: SpeechOscillator::Mandelbrot,
            tertiary_osc: SpeechOscillator::Strange,
            chip_profile: SpeechChipProfile::Sp0256,
            ..SpeechRenderOptions::default()
        };
        let out = render_speech_to_wav(&std::env::temp_dir().join("usg_test_speech.wav"), &opts)
            .expect("speech render");
        assert!(out.frames > 1_000);
    }

    #[test]
    fn deterministic_render_with_same_seed() {
        let a = render_samples(512, 44_100.0, Style::Digital, 0.8, 42).expect("render a");
        let b = render_samples(512, 44_100.0, Style::Digital, 0.8, 42).expect("render b");
        assert_eq!(a, b);
    }

    #[test]
    fn chain_parser_handles_style_and_effect() {
        let s = parse_chain_stage("glitch").expect("glitch");
        let e = parse_chain_stage("stutter").expect("stutter");
        let explicit_style = parse_chain_stage("style:pop").expect("style pop");
        let explicit_effect = parse_chain_stage("effect:pop").expect("effect pop");
        let ring = parse_chain_stage("dissonance-ring").expect("dissonance ring");
        let expand = parse_chain_stage("effect:dissonance-expand").expect("dissonance expand");
        assert!(matches!(s, ChainStage::Style(Style::Glitch)));
        assert!(matches!(e, ChainStage::Effect(Effect::Stutter)));
        assert!(matches!(explicit_style, ChainStage::Style(Style::Pop)));
        assert!(matches!(explicit_effect, ChainStage::Effect(Effect::Pop)));
        assert!(matches!(ring, ChainStage::Effect(Effect::DissonanceRing)));
        assert!(matches!(
            expand,
            ChainStage::Effect(Effect::DissonanceExpand)
        ));
    }

    #[test]
    fn paper_dissonancizers_increase_roughness_on_clean_material() {
        let sample_rate = 44_100.0;
        let source: Vec<f64> = (0..44_100)
            .map(|i| {
                let t = i as f64 / sample_rate;
                0.28 * (2.0 * PI * 220.0 * t).sin() + 0.24 * (2.0 * PI * 330.0 * t).sin()
            })
            .collect();

        let basic_before = analyze_samples_base(&source, sample_rate as u32, 1);
        let psycho_before = analyze_samples_psycho(
            &source,
            sample_rate as u32,
            &basic_before,
            &AnalyzeOptions {
                model: AnalyzeModel::Psycho,
                fft_size: 1024,
                hop_size: 256,
            },
            None,
        );

        let mut ringed = source.clone();
        effect_dissonance_ring(&mut ringed, sample_rate, 1.0, 0);
        let basic_ringed = analyze_samples_base(&ringed, sample_rate as u32, 1);
        let psycho_ringed = analyze_samples_psycho(
            &ringed,
            sample_rate as u32,
            &basic_ringed,
            &AnalyzeOptions {
                model: AnalyzeModel::Psycho,
                fft_size: 1024,
                hop_size: 256,
            },
            None,
        );

        let beating_source: Vec<f64> = (0..44_100)
            .map(|i| {
                let t = i as f64 / sample_rate;
                0.25 * (2.0 * PI * 440.0 * t).sin() + 0.25 * (2.0 * PI * 466.0 * t).sin()
            })
            .collect();
        let basic_beating = analyze_samples_base(&beating_source, sample_rate as u32, 1);
        let psycho_beating = analyze_samples_psycho(
            &beating_source,
            sample_rate as u32,
            &basic_beating,
            &AnalyzeOptions {
                model: AnalyzeModel::Psycho,
                fft_size: 1024,
                hop_size: 256,
            },
            None,
        );

        let mut expanded = beating_source.clone();
        effect_dissonance_expand(&mut expanded, sample_rate, 1.0);
        let basic_expanded = analyze_samples_base(&expanded, sample_rate as u32, 1);
        let psycho_expanded = analyze_samples_psycho(
            &expanded,
            sample_rate as u32,
            &basic_expanded,
            &AnalyzeOptions {
                model: AnalyzeModel::Psycho,
                fft_size: 1024,
                hop_size: 256,
            },
            None,
        );

        assert!(
            psycho_ringed.roughness_norm > psycho_before.roughness_norm
                || psycho_ringed.dissonance_norm > psycho_before.dissonance_norm
        );
        assert!(psycho_expanded.roughness_norm >= psycho_beating.roughness_norm);
    }

    #[test]
    fn silence_metrics_are_stable() {
        let silence = vec![0.0_f64; 4_410];
        let report = analyze_samples_base(&silence, 44_100, 1);
        assert!(report.peak_dbfs < -100.0);
        assert!(report.ugly_index >= 0.0 && report.ugly_index <= 1000.0);
        assert_eq!(report.clipped_pct, 0.0);
    }

    #[test]
    fn psycho_silence_is_low() {
        let silence = vec![0.0_f64; 44_100];
        let basic = analyze_samples_base(&silence, 44_100, 1);
        let psycho = analyze_samples_psycho(
            &silence,
            44_100,
            &basic,
            &AnalyzeOptions {
                model: AnalyzeModel::Psycho,
                fft_size: 1024,
                hop_size: 256,
            },
            None,
        );
        assert!(psycho.ugly_index < 200.0);
    }

    #[test]
    fn render_chain_generates_buffer() {
        let stages = vec![
            ChainStage::Style(Style::Glitch),
            ChainStage::Effect(Effect::Stutter),
            ChainStage::Effect(Effect::Pop),
        ];
        let out = render_chain(&stages, 0.4, 44_100, 0.8, true, 0.0, 777).expect("render chain");
        assert_eq!(out.len(), (0.4_f64 * 44_100.0) as usize);
        let peak = out.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(peak > 0.05);
    }

    #[test]
    fn chain_supports_durations_above_120s() {
        let stages = vec![
            ChainStage::Style(Style::Glitch),
            ChainStage::Effect(Effect::Gate),
            ChainStage::Effect(Effect::Crush),
        ];
        let out = render_chain_with_engine(
            &stages,
            121.0,
            8_000,
            0.7,
            true,
            0.0,
            991,
            &RenderEngine {
                backend: RenderBackend::Cpu,
                jobs: 2,
                ..RenderEngine::default()
            },
        )
        .expect("render chain >120s");
        assert_eq!(out.len(), (121.0_f64 * 8_000.0) as usize);
    }

    #[test]
    fn contour_interpolation_level_lookup_is_correct() {
        let linear = UglinessContour {
            version: 1,
            name: None,
            description: None,
            interpolation: ContourInterpolation::Linear,
            points: vec![
                UglinessContourPoint { t: 0.0, level: 100 },
                UglinessContourPoint { t: 1.0, level: 900 },
            ],
        };
        let step = UglinessContour {
            version: 1,
            name: None,
            description: None,
            interpolation: ContourInterpolation::Step,
            points: vec![
                UglinessContourPoint { t: 0.0, level: 100 },
                UglinessContourPoint { t: 1.0, level: 900 },
            ],
        };
        assert!((contour_level_at(&linear, 0.5) - 500.0).abs() < 1e-6);
        assert_eq!(contour_level_at(&step, 0.5), 100.0);
    }

    #[test]
    fn cpu_backend_plan_resolves() {
        let plan = resolve_backend_plan(&RenderEngine {
            backend: RenderBackend::Cpu,
            jobs: 3,
            ..RenderEngine::default()
        })
        .expect("cpu backend");
        assert_eq!(plan.active, RenderBackend::Cpu);
        assert_eq!(plan.jobs, 3);
    }

    #[test]
    fn chain_parallel_style_prerender_works() {
        let stages = vec![
            ChainStage::Style(Style::Glitch),
            ChainStage::Style(Style::Buzz),
            ChainStage::Effect(Effect::Smear),
        ];
        let out = render_chain_with_engine(
            &stages,
            0.3,
            44_100,
            0.7,
            true,
            0.0,
            4242,
            &RenderEngine {
                backend: RenderBackend::Cpu,
                jobs: 4,
                ..RenderEngine::default()
            },
        )
        .expect("render chain with engine");
        assert_eq!(out.len(), (0.3_f64 * 44_100.0) as usize);
    }

    #[test]
    fn psycho_model_returns_component_breakdown() {
        let out = std::env::temp_dir().join(format!("usg_test_{}_psycho.wav", std::process::id()));
        let opts = RenderOptions {
            duration: 0.35,
            sample_rate: 44_100,
            seed: Some(987),
            style: Style::Digital,
            gain: 0.9,
            normalize: true,
            normalize_dbfs: 0.0,
            output_encoding: OutputEncoding::Float32,
        };
        render_to_wav(&out, &opts).expect("render");

        let report = analyze_wav_with_options(
            &out,
            &AnalyzeOptions {
                model: AnalyzeModel::Psycho,
                fft_size: 2048,
                hop_size: 512,
            },
        )
        .expect("analyze psycho");

        assert_eq!(report.model, "psycho");
        assert!(report.selected_ugly_index >= 0.0 && report.selected_ugly_index <= 1000.0);
        assert!(report.psycho.is_some());
        let psycho = report.psycho.expect("psycho");
        assert!((0.0..=1.0).contains(&psycho.harmonicity_norm));
        assert!((0.0..=1.0).contains(&psycho.inharmonicity_norm));
        assert!((0.0..=1.0).contains(&psycho.binaural_beat_norm));
        assert!((0.0..=1.0).contains(&psycho.beat_conflict_norm));
        assert!((0.0..=1.0).contains(&psycho.tritone_tension_norm));
        assert!((0.0..=1.0).contains(&psycho.wolf_fifth_norm));
        let _ = fs::remove_file(out);
    }

    #[test]
    fn render_to_wav_and_analyze_round_trip() {
        let out =
            std::env::temp_dir().join(format!("usg_test_{}_roundtrip.wav", std::process::id()));
        let opts = RenderOptions {
            duration: 0.2,
            sample_rate: 44_100,
            seed: Some(123),
            style: Style::Harsh,
            gain: 0.8,
            normalize: true,
            normalize_dbfs: 0.0,
            output_encoding: OutputEncoding::Float32,
        };
        let render = render_to_wav(&out, &opts).expect("render");
        assert_eq!(
            render.frames,
            (opts.duration * opts.sample_rate as f64) as usize
        );
        let analysis = analyze_wav(&out).expect("analyze");
        assert!(analysis.duration_s > 0.15 && analysis.duration_s < 0.25);
        let _ = fs::remove_file(out);
    }

    #[test]
    fn long_render_duration_is_valid() {
        let opts = RenderOptions {
            duration: 3_600.0,
            sample_rate: 192_000,
            seed: Some(1),
            style: Style::Harsh,
            gain: 0.8,
            normalize: true,
            normalize_dbfs: 0.0,
            output_encoding: OutputEncoding::Float32,
        };
        validate_render_options(&opts).expect("long duration should validate");
    }

    #[test]
    fn go_ugly_file_generates_output() {
        let in_path =
            std::env::temp_dir().join(format!("usg_test_{}_go_in.wav", std::process::id()));
        let out_path =
            std::env::temp_dir().join(format!("usg_test_{}_go_out.wav", std::process::id()));

        let render_opts = RenderOptions {
            duration: 0.25,
            sample_rate: 44_100,
            seed: Some(42),
            style: Style::Harsh,
            gain: 0.8,
            normalize: true,
            normalize_dbfs: 0.0,
            output_encoding: OutputEncoding::Float32,
        };
        render_to_wav(&in_path, &render_opts).expect("input render");
        let summary = go_ugly_file(
            &in_path,
            &out_path,
            8,
            Some(GoFlavor::Glitch),
            Some(1234),
            true,
            0.0,
        )
        .expect("go uglify");

        assert_eq!(summary.level, 8);
        assert_eq!(summary.flavor, GoFlavor::Glitch);
        assert!(summary.frames > 1000);
        let analyzed = analyze_wav(&out_path).expect("analyze go output");
        assert!(analyzed.duration_s > 0.2);
        let _ = fs::remove_file(in_path);
        let _ = fs::remove_file(out_path);
    }
}
