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
pub const COLBYS_MIN: i32 = -1000;
pub const COLBYS_NEUTRAL: i32 = 0;
pub const COLBYS_MAX: i32 = 1000;

fn stream_render_threshold_frames() -> u64 {
    std::env::var("USG_STREAM_THRESHOLD_FRAMES")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(STREAM_RENDER_THRESHOLD_FRAMES)
}
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

impl SpeechBackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SpeechBackendKind::Lpc => "lpc",
            SpeechBackendKind::FormantGrid => "formant-grid",
            SpeechBackendKind::SamVocalTract => "sam-vocal-tract",
            SpeechBackendKind::ArcadePcm => "arcade-pcm",
        }
    }
}

impl fmt::Display for SpeechBackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl SpeechOscillator {
    pub const ALL: [SpeechOscillator; 24] = [
        SpeechOscillator::Sine,
        SpeechOscillator::Pulse,
        SpeechOscillator::Triangle,
        SpeechOscillator::Saw,
        SpeechOscillator::Noise,
        SpeechOscillator::Buzz,
        SpeechOscillator::Formant,
        SpeechOscillator::Vowel,
        SpeechOscillator::Ring,
        SpeechOscillator::Fold,
        SpeechOscillator::Organ,
        SpeechOscillator::Fm,
        SpeechOscillator::Sync,
        SpeechOscillator::Lfsr,
        SpeechOscillator::Grain,
        SpeechOscillator::Chirp,
        SpeechOscillator::Subharmonic,
        SpeechOscillator::Reed,
        SpeechOscillator::Click,
        SpeechOscillator::Comb,
        SpeechOscillator::Koch,
        SpeechOscillator::Mandelbrot,
        SpeechOscillator::Strange,
        SpeechOscillator::Phoneme,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            SpeechOscillator::Sine => "sine",
            SpeechOscillator::Pulse => "pulse",
            SpeechOscillator::Triangle => "triangle",
            SpeechOscillator::Saw => "saw",
            SpeechOscillator::Noise => "noise",
            SpeechOscillator::Buzz => "buzz",
            SpeechOscillator::Formant => "formant",
            SpeechOscillator::Vowel => "vowel",
            SpeechOscillator::Ring => "ring",
            SpeechOscillator::Fold => "fold",
            SpeechOscillator::Organ => "organ",
            SpeechOscillator::Fm => "fm",
            SpeechOscillator::Sync => "sync",
            SpeechOscillator::Lfsr => "lfsr",
            SpeechOscillator::Grain => "grain",
            SpeechOscillator::Chirp => "chirp",
            SpeechOscillator::Subharmonic => "subharmonic",
            SpeechOscillator::Reed => "reed",
            SpeechOscillator::Click => "click",
            SpeechOscillator::Comb => "comb",
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
pub enum SpeechBackendKind {
    Lpc,
    FormantGrid,
    SamVocalTract,
    ArcadePcm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SpeechOscillator {
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
    #[serde(alias = "level")]
    pub colbys: i32,
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
    FiveOneTwo,
    FiveOneFour,
    SevenOne,
    SevenOneTwo,
    SevenOneFour,
    NineOneSix,
    Custom(u16),
}

impl SurroundLayout {
    pub fn as_str(self) -> String {
        match self {
            SurroundLayout::Mono => "mono".to_string(),
            SurroundLayout::Stereo => "stereo".to_string(),
            SurroundLayout::Quad => "quad".to_string(),
            SurroundLayout::FiveOne => "5.1".to_string(),
            SurroundLayout::FiveOneTwo => "5.1.2".to_string(),
            SurroundLayout::FiveOneFour => "5.1.4".to_string(),
            SurroundLayout::SevenOne => "7.1".to_string(),
            SurroundLayout::SevenOneTwo => "7.1.2".to_string(),
            SurroundLayout::SevenOneFour => "7.1.4".to_string(),
            SurroundLayout::NineOneSix => "9.1.6".to_string(),
            SurroundLayout::Custom(n) => format!("custom:{n}"),
        }
    }

    pub fn channels(self) -> u16 {
        match self {
            SurroundLayout::Mono => 1,
            SurroundLayout::Stereo => 2,
            SurroundLayout::Quad => 4,
            SurroundLayout::FiveOne => 6,
            SurroundLayout::FiveOneTwo => 8,
            SurroundLayout::FiveOneFour => 10,
            SurroundLayout::SevenOne => 8,
            SurroundLayout::SevenOneTwo => 10,
            SurroundLayout::SevenOneFour => 12,
            SurroundLayout::NineOneSix => 16,
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
pub struct BackendStatus {
    pub available: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackendStatusReport {
    pub cpu: BackendStatus,
    pub metal: BackendStatus,
    pub cuda: BackendStatus,
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
    pub joke: bool,
}

impl Default for AnalyzeOptions {
    fn default() -> Self {
        Self {
            model: AnalyzeModel::Basic,
            fft_size: 2048,
            hop_size: 512,
            joke: false,
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
pub struct PieceOptions {
    pub duration: f64,
    pub sample_rate: u32,
    pub seed: Option<u64>,
    pub styles: Vec<Style>,
    pub layout: Option<SurroundLayout>,
    pub channels: u16,
    pub gain: f64,
    pub normalize: bool,
    pub normalize_dbfs: f64,
    pub output_encoding: OutputEncoding,
    pub events_per_second: f64,
    pub min_event_duration: f64,
    pub max_event_duration: f64,
    pub min_pan_width: f64,
    pub max_pan_width: f64,
}

impl Default for PieceOptions {
    fn default() -> Self {
        Self {
            duration: 12.0,
            sample_rate: 192_000,
            seed: None,
            styles: available_styles().to_vec(),
            layout: Some(SurroundLayout::Stereo),
            channels: 2,
            gain: 0.7,
            normalize: true,
            normalize_dbfs: -0.6,
            output_encoding: OutputEncoding::Float32,
            events_per_second: 5.0,
            min_event_duration: 0.03,
            max_event_duration: 0.35,
            min_pan_width: 0.35,
            max_pan_width: 1.75,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpeechRenderOptions {
    pub text: String,
    pub input_mode: SpeechInputMode,
    pub sample_rate: u32,
    pub seed: Option<u64>,
    pub normalize_text: bool,
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
    pub word_accent: f64,
    pub sentence_lilt: f64,
    pub paragraph_decline: f64,
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
            normalize_text: true,
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
            word_accent: 0.18,
            sentence_lilt: 0.14,
            paragraph_decline: 0.1,
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
pub struct PieceSummary {
    pub output: PathBuf,
    pub frames: usize,
    pub sample_rate: u32,
    pub channels: u16,
    pub layout: Option<String>,
    pub events: usize,
    pub seed: u64,
    pub output_encoding: OutputEncoding,
    pub backend_requested: RenderBackend,
    pub backend_active: RenderBackend,
    pub jobs: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpeechSummary {
    pub output: PathBuf,
    pub frames: usize,
    pub sample_rate: u32,
    pub chip_profile: SpeechChipProfile,
    pub backend_kind: SpeechBackendKind,
    pub input_mode: SpeechInputMode,
    pub text_len: usize,
    pub normalized_text: String,
    pub units_rendered: usize,
    pub phonemes_rendered: usize,
    pub seed: u64,
    pub output_encoding: OutputEncoding,
    pub backend_requested: RenderBackend,
    pub backend_active: RenderBackend,
    pub jobs: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpeechTimelineEntry {
    pub index: usize,
    pub label: String,
    pub source: String,
    pub kind: String,
    pub start_s: f64,
    pub end_s: f64,
    pub gap_after_s: f64,
    pub emphasis: f64,
    pub pitch_hz: f64,
    pub formant1_hz: f64,
    pub formant2_hz: f64,
    pub voiced: bool,
    pub noisy: bool,
    pub backend_kind: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpeechIntelligibility {
    pub intelligibility_index: f64,
    pub articulation_norm: f64,
    pub timing_norm: f64,
    pub voicing_norm: f64,
    pub chip_clarity_norm: f64,
    pub ugliness_penalty_norm: f64,
    pub weighted_sum: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpeechRenderArtifacts {
    pub summary: SpeechSummary,
    pub timeline: Vec<SpeechTimelineEntry>,
}

#[derive(Debug, Clone)]
pub struct GoSummary {
    pub output: PathBuf,
    pub frames: usize,
    pub sample_rate: u32,
    pub channels: u16,
    pub target_colbys: i32,
    pub target_intensity: f64,
    pub flavor: GoFlavor,
    pub seed: u64,
    pub output_encoding: OutputEncoding,
    pub layout: Option<String>,
    pub backend_requested: RenderBackend,
    pub backend_active: RenderBackend,
    pub jobs: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoreMetadata {
    pub unit: &'static str,
    pub min: i32,
    pub neutral: i32,
    pub max: i32,
    pub profile: &'static str,
    pub version: &'static str,
    pub calibrated_from_listening_tests: bool,
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
    /// Ugliness in Colbys: -1000 (cleanest) to +1000 (most ugly), 0 = neutral.
    pub colbys: f64,
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
    /// Ugliness in Colbys: -1000 (cleanest) to +1000 (most ugly), 0 = neutral.
    pub colbys: f64,
    pub fft_size: usize,
    pub hop_size: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnalysisReport {
    pub model: String,
    pub score_metadata: ScoreMetadata,
    /// Ugliness in Colbys: -1000 (cleanest) to +1000 (most ugly), 0 = neutral.
    pub colbys: f64,
    pub basic: Analysis,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub psycho: Option<PsychoAnalysis>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub joke: Option<JokeAnalysis>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JokeAnalysis {
    pub uglierbasis_index: f64,
    pub verdict: String,
    pub academic_cluster_norm: f64,
    pub bureaucratic_overhead_norm: f64,
    pub all_high_bonus_norm: f64,
    pub harmonicity_relief_norm: f64,
    pub weighted_sum: f64,
    pub clip_arrogance: f64,
    pub roughness: f64,
    pub sharpness: f64,
    pub dissonance: f64,
    pub transient_density: f64,
    pub harmonicity: f64,
    pub inharmonicity: f64,
    pub binaural_beat_pressure: f64,
    pub beat_conflict: f64,
    pub wolf_fifth_tension: f64,
    pub modulation_glare: f64,
    pub gate_surprise: f64,
    pub pop_density: f64,
    pub quantization_shame: f64,
    pub zipper_noise: f64,
    pub loudness_lurch: f64,
    pub overtone_hostility: f64,
    pub alias_spray: f64,
    pub envelope_panic: f64,
    pub notch_cruelty: f64,
    pub hysteresis_squeal: f64,
    pub stereo_argument: f64,
    pub jitter: f64,
    pub vibrato_malpractice: f64,
    pub cadence_collapse: f64,
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

pub fn backend_status_report() -> BackendStatusReport {
    BackendStatusReport {
        cpu: BackendStatus {
            available: true,
            detail: "Always available in this build".to_string(),
        },
        metal: BackendStatus {
            available: metal_supported(),
            detail: metal_status_detail(),
        },
        cuda: BackendStatus {
            available: cuda_supported(),
            detail: cuda_status_detail(),
        },
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

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_status_detail() -> String {
    backend_metal::availability_detail()
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_status_detail() -> String {
    if !cfg!(feature = "metal") {
        "Metal feature not built; rebuild with --features metal".to_string()
    } else {
        "Metal rendering is only supported on macOS targets".to_string()
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

#[cfg(feature = "cuda")]
fn cuda_status_detail() -> String {
    if std::env::var_os("USG_DISABLE_CUDA").is_some() {
        "CUDA disabled by USG_DISABLE_CUDA".to_string()
    } else {
        backend_cuda::availability_detail()
    }
}

#[cfg(not(feature = "cuda"))]
fn cuda_status_detail() -> String {
    "CUDA feature not built; rebuild with --features cuda".to_string()
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

pub fn render_piece_to_wav(output: &Path, opts: &PieceOptions) -> Result<PieceSummary> {
    render_piece_to_wav_with_engine(output, opts, &RenderEngine::default())
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

    if total_frames > stream_render_threshold_frames() {
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

pub fn render_piece_to_wav_with_engine(
    output: &Path,
    opts: &PieceOptions,
    engine: &RenderEngine,
) -> Result<PieceSummary> {
    render_piece_to_wav_with_engine_progress(output, opts, engine, |_done, _total, _style| {})
}

pub fn render_piece_to_wav_with_engine_progress<F>(
    output: &Path,
    opts: &PieceOptions,
    engine: &RenderEngine,
    mut progress: F,
) -> Result<PieceSummary>
where
    F: FnMut(usize, usize, Style),
{
    validate_piece_options(opts)?;
    let plan = resolve_backend_plan(engine)?;
    let seed = opts.seed.unwrap_or_else(seed_from_time);
    let layout = opts
        .layout
        .unwrap_or_else(|| SurroundLayout::Custom(opts.channels));
    let total_frames = duration_to_frames(opts.duration, opts.sample_rate)?;
    let frames_usize = usize::try_from(total_frames)
        .map_err(|_| anyhow!("requested piece render is too large for this platform"))?;
    let channel_count = usize::from(layout.channels());
    let event_count = (opts.duration * opts.events_per_second).round().max(1.0) as usize;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut channels = vec![vec![0.0_f64; frames_usize]; channel_count];
    let (speaker_pos, lfe_idx) = speaker_positions(layout);

    let mut last_style = None;
    for event_idx in 0..event_count {
        let style = pick_piece_style(&opts.styles, last_style, &mut rng);
        last_style = Some(style);
        let event_duration_s = rng.gen_range(opts.min_event_duration..=opts.max_event_duration);
        let event_frames = usize::try_from(duration_to_frames(event_duration_s, opts.sample_rate)?)
            .map_err(|_| anyhow!("piece event is too large for this platform"))?
            .max(1)
            .min(frames_usize);
        let max_start = frames_usize.saturating_sub(event_frames);
        let start = if max_start == 0 {
            0
        } else {
            rng.gen_range(0..=max_start)
        };
        let event_seed = rng.r#gen::<u64>();
        let event_gain = (opts.gain * rng.gen_range(0.45_f64..1.0_f64)).clamp(0.0, 1.0);
        let mut event = render_samples_with_plan(
            event_frames,
            opts.sample_rate as f64,
            style,
            event_gain,
            event_seed,
            &plan,
        )?;
        if rng.gen_bool(0.72) && opts.styles.len() > 1 {
            let alt_style = pick_piece_style(&opts.styles, Some(style), &mut rng);
            let alt_seed = rng.r#gen::<u64>();
            let alt_gain = (event_gain * rng.gen_range(0.25_f64..0.7_f64)).clamp(0.0, 1.0);
            let alt = render_samples_with_plan(
                event_frames,
                opts.sample_rate as f64,
                alt_style,
                alt_gain,
                alt_seed,
                &plan,
            )?;
            mix_piece_layer(&mut event, &alt, rng.gen_range(0.22_f64..0.58_f64));
        }
        if rng.gen_bool(0.26) && opts.styles.len() > 2 {
            let third_style = pick_piece_style(&opts.styles, Some(style), &mut rng);
            let third = render_samples_with_plan(
                event_frames,
                opts.sample_rate as f64,
                third_style,
                (event_gain * rng.gen_range(0.15_f64..0.45_f64)).clamp(0.0, 1.0),
                rng.r#gen::<u64>(),
                &plan,
            )?;
            mix_piece_layer(&mut event, &third, rng.gen_range(0.14_f64..0.35_f64));
        }
        diversify_piece_event(
            &mut event,
            opts.sample_rate as f64,
            event_seed ^ start as u64 ^ event_idx as u64,
        );
        apply_event_envelope(&mut event, opts.sample_rate);

        let source = random_piece_source_position(layout, &mut rng);
        let width = rng
            .gen_range(opts.min_pan_width..=opts.max_pan_width)
            .max(0.05);
        let weights = piece_spatial_weights(&speaker_pos, lfe_idx, source, width);

        for (ch_idx, channel) in channels.iter_mut().enumerate() {
            let weight = weights[ch_idx];
            if weight <= EPS64 {
                continue;
            }
            for (frame_idx, &sample) in event.iter().enumerate() {
                channel[start + frame_idx] += sample * weight;
            }
        }

        progress(event_idx + 1, event_count, style);
    }

    if opts.normalize {
        normalize_peak_dbfs_channels(&mut channels, opts.normalize_dbfs);
    }
    write_wav_channels(output, opts.sample_rate, &channels, opts.output_encoding)?;

    Ok(PieceSummary {
        output: output.to_path_buf(),
        frames: frames_usize,
        sample_rate: opts.sample_rate,
        channels: layout.channels(),
        layout: Some(layout.as_str()),
        events: event_count,
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

pub fn render_speech_with_artifacts_to_wav(
    output: &Path,
    opts: &SpeechRenderOptions,
) -> Result<SpeechRenderArtifacts> {
    render_speech_with_artifacts_to_wav_with_engine(output, opts, &RenderEngine::default())
}

pub fn render_speech_to_wav_with_engine(
    output: &Path,
    opts: &SpeechRenderOptions,
    engine: &RenderEngine,
) -> Result<SpeechSummary> {
    Ok(render_speech_with_artifacts_to_wav_with_engine(output, opts, engine)?.summary)
}

pub fn render_speech_with_artifacts_to_wav_with_engine(
    output: &Path,
    opts: &SpeechRenderOptions,
    engine: &RenderEngine,
) -> Result<SpeechRenderArtifacts> {
    validate_speech_options(opts)?;
    let plan = resolve_backend_plan(engine)?;
    let seed = opts.seed.unwrap_or_else(seed_from_time);
    let backend_kind = speech_backend_for_profile(opts.chip_profile);
    let normalized_text = normalized_speech_text(&opts.text, opts.normalize_text);
    let units = speech_units_for_mode(&normalized_text, opts.input_mode);
    let (mut samples, timeline) =
        render_speech_samples_with_plan(opts, &units, seed, backend_kind, &plan)?;
    let frames = samples.len();
    if opts.normalize {
        normalize_peak_dbfs(&mut samples, opts.normalize_dbfs);
    }
    write_wav_mono(output, opts.sample_rate, &samples, opts.output_encoding)?;
    Ok(SpeechRenderArtifacts {
        summary: SpeechSummary {
            output: output.to_path_buf(),
            frames,
            sample_rate: opts.sample_rate,
            chip_profile: opts.chip_profile,
            backend_kind,
            input_mode: opts.input_mode,
            text_len: opts.text.chars().count(),
            normalized_text,
            units_rendered: units.len(),
            phonemes_rendered: timeline
                .iter()
                .filter(|entry| entry.kind == "phoneme")
                .count(),
            seed,
            output_encoding: opts.output_encoding,
            backend_requested: plan.requested,
            backend_active: plan.active,
            jobs: plan.jobs,
        },
        timeline,
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

fn try_backend_render_style(
    frames: usize,
    sample_rate: f64,
    style: Style,
    gain: f64,
    seed: u64,
    plan: &BackendPlan,
) -> Result<Option<Vec<f64>>> {
    #[cfg(not(any(all(feature = "metal", target_os = "macos"), feature = "cuda")))]
    let _ = (sample_rate, style, gain, seed);

    if frames < 8_192 || plan.jobs <= 1 {
        return Ok(None);
    }
    match plan.active {
        RenderBackend::Metal => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                return backend_metal::render_style(
                    style,
                    frames,
                    sample_rate,
                    gain,
                    seed,
                    plan.gpu_drive,
                    plan.gpu_crush_bits,
                    plan.gpu_crush_mix,
                )
                .map(Some);
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                return Ok(None);
            }
        }
        RenderBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                return backend_cuda::render_style(
                    style,
                    frames,
                    sample_rate,
                    gain,
                    seed,
                    plan.gpu_drive,
                    plan.gpu_crush_bits,
                    plan.gpu_crush_mix,
                )
                .map(Some);
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Ok(None);
            }
        }
        _ => Ok(None),
    }
}

fn render_samples_with_plan(
    frames: usize,
    sample_rate: f64,
    style: Style,
    gain: f64,
    seed: u64,
    plan: &BackendPlan,
) -> Result<Vec<f64>> {
    if let Some(samples) = try_backend_render_style(frames, sample_rate, style, gain, seed, plan)? {
        return Ok(samples);
    }
    let mut samples = render_samples(frames, sample_rate, style, gain, seed)?;
    apply_backend_post(samples.as_mut_slice(), plan)?;
    Ok(samples)
}

#[derive(Clone)]
struct SpeechUnit {
    label: String,
    source: String,
    kind: SpeechSymbolKind,
    duration_s: f64,
    gap_s: f64,
    emphasis: f64,
    params: SpeechSymbolParams,
    word_index: usize,
    sentence_index: usize,
    paragraph_index: usize,
}

fn render_speech_samples_with_plan(
    opts: &SpeechRenderOptions,
    units: &[SpeechUnit],
    seed: u64,
    backend_kind: SpeechBackendKind,
    plan: &BackendPlan,
) -> Result<(Vec<f64>, Vec<SpeechTimelineEntry>)> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    if units.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut out = Vec::new();
    let mut timeline = Vec::with_capacity(units.len());
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
        let phoneme = unit.params;
        let pace_scale = 11.0 / opts.units_per_second.max(1.0);
        let frames = (unit.duration_s * pace_scale * sample_rate).max(1.0) as usize;
        let attack_s = opts.attack_ms * 0.001;
        let release_s = opts.release_ms * 0.001;
        let unit_center = idx as f64 / units.len().max(1) as f64;
        let sentence_progress = unit.sentence_index as f64 / (units.len().max(1) as f64);
        let paragraph_drop = 1.0 - opts.paragraph_decline * unit.paragraph_index as f64 * 0.06;
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
            let word_shape = if unit.kind.is_phoneme() && unit.word_index % 2 == 0 {
                1.0 + opts.word_accent * (1.0 - progress).max(0.0) * 0.35
            } else {
                1.0 + opts.word_accent * progress * 0.15
            };
            let sentence_shape = 1.0
                + opts.sentence_lilt
                    * (sentence_progress - 0.5)
                    * if unit.source.ends_with('?') { 1.4 } else { 1.0 };
            let pitch = opts.pitch_hz
                * tuning.pitch_mul
                * phoneme.pitch_mul
                * vibrato
                * jitter
                * drift
                * monotone_mix.max(0.2)
                * word_shape
                * sentence_shape
                * paragraph_drop.max(0.55);
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
            sample = apply_speech_chip_fx(
                sample,
                opts,
                &tuning,
                backend_kind,
                &mut sample_hold_counter,
                &mut held,
            );
            out.push(sample);
        }

        let gap_ms = match unit.kind {
            SpeechSymbolKind::Whitespace => opts.word_gap_ms,
            SpeechSymbolKind::Punctuation => {
                opts.punctuation_gap_ms
                    + if matches!(unit.source.as_str(), "." | "!" | "?") {
                        opts.sentence_gap_ms
                    } else {
                        0.0
                    }
            }
            SpeechSymbolKind::ParagraphBreak => opts.paragraph_gap_ms,
            _ => unit.gap_s * 1000.0,
        };
        let gap_frames = (gap_ms * 0.001 * sample_rate).round() as usize;
        let end_s = out.len() as f64 / sample_rate;
        timeline.push(SpeechTimelineEntry {
            index: idx,
            label: unit.label.clone(),
            source: unit.source.clone(),
            kind: unit.kind.as_str().to_string(),
            start_s: (end_s - frames as f64 / sample_rate).max(0.0),
            end_s,
            gap_after_s: gap_frames as f64 / sample_rate,
            emphasis: unit.emphasis,
            pitch_hz: opts.pitch_hz * tuning.pitch_mul * phoneme.pitch_mul,
            formant1_hz: phoneme.formant1 * opts.formant_shift * tuning.formant_mul,
            formant2_hz: phoneme.formant2 * opts.formant_shift * tuning.formant_mul,
            voiced: phoneme.voiced,
            noisy: phoneme.noisy,
            backend_kind: backend_kind.as_str().to_string(),
        });
        out.extend(std::iter::repeat_n(0.0, gap_frames));
    }

    apply_backend_post(out.as_mut_slice(), plan)?;
    Ok((out, timeline))
}

fn apply_event_envelope(samples: &mut [f64], sample_rate: u32) {
    if samples.len() < 2 {
        return;
    }
    let ramp_frames = ((sample_rate as f64 * 0.004).round() as usize)
        .clamp(8, samples.len().max(2) / 2)
        .max(1);
    for i in 0..ramp_frames.min(samples.len()) {
        let w = i as f64 / ramp_frames as f64;
        samples[i] *= w;
        let tail_idx = samples.len() - 1 - i;
        samples[tail_idx] *= w;
    }
}

fn pick_piece_style(styles: &[Style], previous: Option<Style>, rng: &mut ChaCha8Rng) -> Style {
    if styles.len() <= 1 {
        return styles[0];
    }
    for _ in 0..8 {
        let style = styles[rng.gen_range(0..styles.len())];
        if Some(style) != previous || rng.gen_bool(0.18) {
            return style;
        }
    }
    styles[rng.gen_range(0..styles.len())]
}

fn mix_piece_layer(base: &mut [f64], layer: &[f64], mix: f64) {
    let wet = mix.clamp(0.0, 1.0);
    let dry = (1.0 - 0.45 * wet).clamp(0.35, 1.0);
    for (dst, src) in base.iter_mut().zip(layer.iter()) {
        *dst = soft_clip(*dst * dry + *src * wet);
    }
}

fn diversify_piece_event(event: &mut [f64], sample_rate: f64, seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let fx_count = if rng.gen_bool(0.22) {
        3
    } else if rng.gen_bool(0.58) {
        2
    } else {
        1
    };
    for fx_idx in 0..fx_count {
        let effect = pick_piece_effect(&mut rng);
        apply_effect(
            event,
            effect,
            sample_rate,
            derive_seed(seed, 0xEFFE_0000 + fx_idx as u64),
            0,
        );
    }

    if rng.gen_bool(0.46) {
        piece_slice_scramble(event, &mut rng);
    }
    if rng.gen_bool(0.31) {
        piece_glitch_gate(event, &mut rng);
    }
    if rng.gen_bool(0.19) {
        event.reverse();
    }
    if rng.gen_bool(0.64) {
        let wobble_hz = rng.gen_range(7.0_f64..48.0_f64);
        for (i, sample) in event.iter_mut().enumerate() {
            let t = i as f64 / sample_rate;
            let wobble = 0.58 + 0.42 * (2.0 * PI * wobble_hz * t).sin().abs();
            *sample = soft_clip(*sample * wobble);
        }
    }
}

fn pick_piece_effect(rng: &mut ChaCha8Rng) -> Effect {
    let roll = rng.gen_range(0.0_f64..1.0_f64);
    if roll < 0.24 {
        Effect::Stutter
    } else if roll < 0.42 {
        Effect::Crush
    } else if roll < 0.58 {
        Effect::Gate
    } else if roll < 0.72 {
        Effect::Pop
    } else if roll < 0.84 {
        Effect::Smear
    } else if roll < 0.93 {
        Effect::DissonanceRing
    } else {
        Effect::DissonanceExpand
    }
}

fn piece_slice_scramble(event: &mut [f64], rng: &mut ChaCha8Rng) {
    if event.len() < 128 {
        return;
    }
    let slice = rng.gen_range(24..192).min(event.len().max(1));
    let mut idx = 0usize;
    while idx + slice * 2 < event.len() {
        if rng.gen_bool(0.28) {
            let a = idx;
            let b = (idx + slice).min(event.len() - slice);
            for off in 0..slice {
                event.swap(a + off, b + off);
            }
        }
        idx += slice;
    }
}

fn piece_glitch_gate(event: &mut [f64], rng: &mut ChaCha8Rng) {
    if event.is_empty() {
        return;
    }
    let window = rng.gen_range(8..128).min(event.len().max(1));
    let mut idx = 0usize;
    while idx < event.len() {
        let gate = if rng.gen_bool(0.22) {
            0.0
        } else if rng.gen_bool(0.35) {
            rng.gen_range(-0.35_f64..0.35_f64)
        } else {
            1.0
        };
        let end = (idx + window).min(event.len());
        for sample in &mut event[idx..end] {
            *sample = soft_clip(*sample * gate);
        }
        idx += window;
    }
}

fn piece_spatial_weights(
    speaker_pos: &[[f64; 3]],
    lfe_idx: Option<usize>,
    source: [f64; 3],
    width: f64,
) -> Vec<f64> {
    let sigma = width.max(0.05);
    let mut weights = Vec::with_capacity(speaker_pos.len());
    let mut sum = 0.0_f64;
    for (idx, spk) in speaker_pos.iter().enumerate() {
        if Some(idx) == lfe_idx {
            weights.push(0.0);
            continue;
        }
        let dist = distance3(source, *spk) / sigma;
        let w = (-0.5 * dist * dist).exp();
        weights.push(w);
        sum += w;
    }
    if sum <= EPS64 {
        return vec![1.0 / speaker_pos.len().max(1) as f64; speaker_pos.len()];
    }
    for weight in &mut weights {
        *weight /= sum;
    }
    if let Some(lfe) = lfe_idx {
        let height_energy = source[2].max(0.0);
        let lfe_weight = (0.06 + 0.14 * height_energy).clamp(0.0, 0.18);
        let keep = (1.0 - lfe_weight).clamp(0.0, 1.0);
        for (idx, weight) in weights.iter_mut().enumerate() {
            if idx == lfe {
                continue;
            }
            *weight *= keep;
        }
        weights[lfe] = lfe_weight;
    }
    weights
}

fn random_piece_source_position(layout: SurroundLayout, rng: &mut ChaCha8Rng) -> [f64; 3] {
    let az = rng.gen_range(-180.0_f64..180.0_f64).to_radians();
    let radius = rng.gen_range(0.45_f64..1.35_f64);
    let z = match layout {
        SurroundLayout::FiveOneTwo
        | SurroundLayout::FiveOneFour
        | SurroundLayout::SevenOneTwo
        | SurroundLayout::SevenOneFour
        | SurroundLayout::NineOneSix => rng.gen_range(0.0_f64..1.0_f64),
        _ => rng.gen_range(-0.1_f64..0.15_f64),
    };
    [az.sin() * radius, az.cos() * radius, z]
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

    if total_frames > stream_render_threshold_frames() {
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
    target_colbys: i32,
    flavor: Option<GoFlavor>,
    seed: Option<u64>,
    normalize: bool,
    normalize_dbfs: f64,
) -> Result<GoSummary> {
    go_ugly_file_with_engine_contour(
        input,
        output,
        target_colbys,
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
    target_colbys: i32,
    flavor: Option<GoFlavor>,
    seed: Option<u64>,
    normalize: bool,
    normalize_dbfs: f64,
    engine: &RenderEngine,
) -> Result<GoSummary> {
    go_ugly_file_with_engine_contour(
        input,
        output,
        target_colbys,
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
    target_colbys: i32,
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
        target_colbys,
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
    target_colbys: i32,
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
        target_colbys,
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
    target_colbys: i32,
    flavor: Option<GoFlavor>,
    seed: Option<u64>,
    normalize: bool,
    normalize_dbfs: f64,
    engine: &RenderEngine,
    contour: Option<&UglinessContour>,
    output_sample_rate: Option<u32>,
    output_encoding: OutputEncoding,
) -> Result<GoSummary> {
    if !(COLBYS_MIN..=COLBYS_MAX).contains(&target_colbys) {
        return Err(anyhow!(
            "target Colbys must be between {} and {}",
            COLBYS_MIN,
            COLBYS_MAX
        ));
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
    let target_intensity = colbys_to_intensity(target_colbys);

    for (ch_idx, channel) in channels.iter_mut().enumerate() {
        let ch_seed = derive_seed(base_seed, ch_idx as u64);
        apply_go_ugly_to_channel(
            channel,
            sample_rate_f64,
            target_colbys,
            chosen,
            ch_seed,
            contour,
        );
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
        target_colbys,
        target_intensity,
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
    target_colbys: i32,
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
        target_colbys,
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
    target_colbys: i32,
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
        target_colbys,
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
    target_colbys: i32,
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

    if !(COLBYS_MIN..=COLBYS_MAX).contains(&target_colbys) {
        return Err(anyhow!(
            "target Colbys must be between {} and {}",
            COLBYS_MIN,
            COLBYS_MAX
        ));
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
    let target_intensity = colbys_to_intensity(target_colbys);
    apply_go_ugly_to_channel(
        mono.as_mut_slice(),
        sample_rate as f64,
        target_colbys,
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
        target_colbys,
        target_intensity,
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
    let (colbys, psycho) = match opts.model {
        AnalyzeModel::Basic => (basic.colbys, None),
        AnalyzeModel::Psycho => {
            let psycho = analyze_samples_psycho(&mono, sample_rate, &basic, opts, stereo_pair);
            (psycho.colbys, Some(psycho))
        }
    };
    let joke = if opts.joke {
        let psycho_for_joke = psycho.clone().unwrap_or_else(|| {
            analyze_samples_psycho(&mono, sample_rate, &basic, opts, stereo_pair)
        });
        Some(compute_joke_analysis(&basic, &psycho_for_joke))
    } else {
        None
    };

    Ok(AnalysisReport {
        model: opts.model.as_str().to_string(),
        score_metadata: match opts.model {
            AnalyzeModel::Basic => basic_score_metadata(),
            AnalyzeModel::Psycho => psycho_score_metadata(),
        },
        colbys,
        basic,
        psycho,
        joke,
    })
}

/// Options for windowed ugliness timeline analysis.
#[derive(Debug, Clone)]
pub struct TimelineOptions {
    /// Analysis window length in milliseconds.
    pub window_ms: f64,
    /// Hop (step) between windows in milliseconds.
    pub hop_ms: f64,
}

impl Default for TimelineOptions {
    fn default() -> Self {
        Self {
            window_ms: 50.0,
            hop_ms: 25.0,
        }
    }
}

/// One windowed ugliness measurement at a point in time.
#[derive(Debug, Clone, Serialize)]
pub struct TimelineFrame {
    /// Start time of this window in seconds.
    pub time_s: f64,
    /// Ugliness of this window in Colbys (-1000..+1000).
    pub colbys: f64,
    /// Clipped sample percentage in this window.
    pub clipped_pct: f64,
    /// Harshness ratio in this window.
    pub harshness_ratio: f64,
    /// Zero-crossing rate in this window.
    pub zero_crossing_rate: f64,
}

/// Compute per-window ugliness over the file.
pub fn analyze_wav_timeline(path: &Path, opts: &TimelineOptions) -> Result<Vec<TimelineFrame>> {
    if opts.window_ms < 1.0 {
        return Err(anyhow!("timeline window_ms must be >= 1.0"));
    }
    if opts.hop_ms < 1.0 {
        return Err(anyhow!("timeline hop_ms must be >= 1.0"));
    }
    let (channel_data, sample_rate, channels) = read_wav_channels_f64(path)?;
    if channel_data.is_empty() || channel_data[0].is_empty() {
        return Err(anyhow!("input had no audio samples"));
    }
    let mono = mixdown_mono(&channel_data);
    let sr = sample_rate as f64;
    let win_frames = ((opts.window_ms * 0.001 * sr) as usize).max(2);
    let hop_frames = ((opts.hop_ms * 0.001 * sr) as usize).max(1);
    let total = mono.len();
    let mut frames = Vec::new();
    let mut pos = 0usize;
    while pos < total {
        let end = (pos + win_frames).min(total);
        let window = &mono[pos..end];
        let analysis = analyze_samples_base(window, sample_rate, channels);
        frames.push(TimelineFrame {
            time_s: pos as f64 / sr,
            colbys: analysis.colbys,
            clipped_pct: analysis.clipped_pct,
            harshness_ratio: analysis.harshness_ratio,
            zero_crossing_rate: analysis.zero_crossing_rate,
        });
        pos += hop_frames;
    }
    Ok(frames)
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

pub fn validate_piece_options(opts: &PieceOptions) -> Result<()> {
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
    if opts.channels == 0 || opts.channels > 64 {
        return Err(anyhow!("channels must be between 1 and 64"));
    }
    if let Some(layout) = opts.layout
        && opts.channels != layout.channels()
    {
        return Err(anyhow!(
            "channels ({}) must match layout {} ({})",
            opts.channels,
            layout,
            layout.channels()
        ));
    }
    if opts.styles.is_empty() {
        return Err(anyhow!("piece requires at least one style"));
    }
    if !(0.1..=10_000.0).contains(&opts.events_per_second) {
        return Err(anyhow!("events-per-second must be between 0.1 and 10000.0"));
    }
    if !(0.005..=opts.duration).contains(&opts.min_event_duration) {
        return Err(anyhow!(
            "min-event-duration must be at least 0.005 seconds and no greater than duration"
        ));
    }
    if !(opts.min_event_duration..=opts.duration).contains(&opts.max_event_duration) {
        return Err(anyhow!(
            "max-event-duration must be at least min-event-duration and no greater than duration"
        ));
    }
    if !(0.05..=64.0).contains(&opts.min_pan_width) {
        return Err(anyhow!("min-pan-width must be between 0.05 and 64.0"));
    }
    if !(opts.min_pan_width..=64.0).contains(&opts.max_pan_width) {
        return Err(anyhow!(
            "max-pan-width must be at least min-pan-width and no greater than 64.0"
        ));
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
        || !(0.0..=1.0).contains(&opts.word_accent)
        || !(0.0..=1.0).contains(&opts.sentence_lilt)
        || !(0.0..=1.0).contains(&opts.paragraph_decline)
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

pub fn score_speech_intelligibility(
    summary: &SpeechSummary,
    timeline: &[SpeechTimelineEntry],
) -> SpeechIntelligibility {
    let phonemes: Vec<&SpeechTimelineEntry> = timeline
        .iter()
        .filter(|entry| entry.kind == "phoneme")
        .collect();
    let phoneme_count = phonemes.len().max(1) as f64;
    let articulation_norm = (phonemes.iter().filter(|p| p.label.len() >= 2).count() as f64
        / phoneme_count)
        .clamp(0.0, 1.0);
    let voicing_norm =
        (phonemes.iter().filter(|p| p.voiced).count() as f64 / phoneme_count).clamp(0.0, 1.0);
    let avg_gap = if timeline.is_empty() {
        0.0
    } else {
        timeline.iter().map(|entry| entry.gap_after_s).sum::<f64>() / timeline.len() as f64
    };
    let timing_norm = (1.0 - (avg_gap - 0.045).abs() / 0.12).clamp(0.0, 1.0);
    let avg_formant_sep = phonemes
        .iter()
        .map(|p| (p.formant2_hz - p.formant1_hz).abs())
        .sum::<f64>()
        / phoneme_count;
    let chip_clarity_norm = ((avg_formant_sep - 700.0) / 1_700.0).clamp(0.0, 1.0);
    let ugliness_penalty_norm = ((summary.phonemes_rendered as f64
        / summary.units_rendered.max(1) as f64)
        * (1.0 - chip_clarity_norm * 0.25))
        .clamp(0.0, 1.0);
    let weighted_sum = 1.15 * articulation_norm
        + 0.95 * timing_norm
        + 0.75 * voicing_norm
        + 0.9 * chip_clarity_norm
        - 0.55 * ugliness_penalty_norm;
    SpeechIntelligibility {
        intelligibility_index: (1000.0 * sigmoid(weighted_sum - 1.25)).clamp(0.0, 1000.0),
        articulation_norm,
        timing_norm,
        voicing_norm,
        chip_clarity_norm,
        ugliness_penalty_norm,
        weighted_sum,
    }
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
        if !(COLBYS_MIN..=COLBYS_MAX).contains(&p.colbys) {
            return Err(anyhow!(
                "contour point {idx} has invalid colbys={} (expected {}..={})",
                p.colbys,
                COLBYS_MIN,
                COLBYS_MAX
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

pub fn colbys_to_intensity(colbys: i32) -> f64 {
    let clamped = colbys.clamp(COLBYS_MIN, COLBYS_MAX) as f64;
    ((clamped - COLBYS_MIN as f64) / (COLBYS_MAX - COLBYS_MIN) as f64).clamp(0.0, 1.0)
}

fn basic_score_metadata() -> ScoreMetadata {
    ScoreMetadata {
        unit: "Colbys",
        min: COLBYS_MIN,
        neutral: COLBYS_NEUTRAL,
        max: COLBYS_MAX,
        profile: "usg-basic-v1",
        version: "1",
        calibrated_from_listening_tests: false,
    }
}

fn psycho_score_metadata() -> ScoreMetadata {
    ScoreMetadata {
        unit: "Colbys",
        min: COLBYS_MIN,
        neutral: COLBYS_NEUTRAL,
        max: COLBYS_MAX,
        profile: "usg-psycho-v1",
        version: "1",
        calibrated_from_listening_tests: false,
    }
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

impl SpeechSymbolKind {
    fn as_str(self) -> &'static str {
        match self {
            SpeechSymbolKind::Vowel => "phoneme",
            SpeechSymbolKind::Consonant => "phoneme",
            SpeechSymbolKind::Digit => "digit",
            SpeechSymbolKind::Whitespace => "whitespace",
            SpeechSymbolKind::Punctuation => "punctuation",
            SpeechSymbolKind::ParagraphBreak => "paragraph-break",
        }
    }

    fn is_phoneme(self) -> bool {
        matches!(
            self,
            SpeechSymbolKind::Vowel | SpeechSymbolKind::Consonant | SpeechSymbolKind::Digit
        )
    }
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
    backend: SpeechBackendKind,
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
            backend: SpeechBackendKind::Lpc,
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
            backend: SpeechBackendKind::Lpc,
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
            backend: SpeechBackendKind::Lpc,
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
            backend: SpeechBackendKind::FormantGrid,
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
            backend: SpeechBackendKind::FormantGrid,
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
            backend: SpeechBackendKind::SamVocalTract,
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
            backend: SpeechBackendKind::ArcadePcm,
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
            backend: SpeechBackendKind::ArcadePcm,
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

fn speech_backend_for_profile(profile: SpeechChipProfile) -> SpeechBackendKind {
    speech_profile_tuning(profile).backend
}

fn normalized_speech_text(text: &str, normalize: bool) -> String {
    if !normalize {
        return text.replace("\r\n", "\n");
    }

    let replaced = text
        .replace("\r\n", "\n")
        .replace(['“', '”'], "\"")
        .replace(['‘', '’'], "'")
        .replace(['–', '—'], "-")
        .replace('\t', " ");
    let mut out = String::new();
    let mut prev_space = false;
    for ch in replaced.chars() {
        if ch == '\n' {
            if !out.ends_with('\n') {
                out.push('\n');
            }
            prev_space = false;
            continue;
        }
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
            continue;
        }
        prev_space = false;
        if ch.is_ascii_digit() {
            let word = match ch {
                '0' => "ZERO",
                '1' => "ONE",
                '2' => "TWO",
                '3' => "THREE",
                '4' => "FOUR",
                '5' => "FIVE",
                '6' => "SIX",
                '7' => "SEVEN",
                '8' => "EIGHT",
                '9' => "NINE",
                _ => "",
            };
            if !out.ends_with([' ', '\n']) && !out.is_empty() {
                out.push(' ');
            }
            out.push_str(word);
            out.push(' ');
        } else {
            out.push(ch.to_ascii_uppercase());
        }
    }
    out.lines()
        .map(str::trim)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn speech_units_for_mode(text: &str, mode: SpeechInputMode) -> Vec<SpeechUnit> {
    let mode = if matches!(mode, SpeechInputMode::Auto) {
        if text.contains("\n\n") || text.lines().count() > 1 {
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

    if matches!(mode, SpeechInputMode::Character) {
        return speech_units_from_characters(text, mode);
    }

    speech_units_from_tokens(text, mode)
}

fn speech_units_from_characters(text: &str, mode: SpeechInputMode) -> Vec<SpeechUnit> {
    let mut units = Vec::new();
    let mut word_index = 0;
    let mut sentence_index = 0;
    let mut paragraph_index = 0;
    for ch in text.chars() {
        let kind = classify_speech_symbol(ch);
        let duration_s = speech_symbol_duration(ch, kind, mode);
        let gap_s = speech_symbol_gap(ch, kind, mode);
        let emphasis = speech_source_emphasis(ch.to_string().as_str());
        let label = ch.to_string();
        units.push(SpeechUnit {
            label: label.clone(),
            source: label,
            kind,
            duration_s,
            gap_s,
            emphasis,
            params: speech_symbol_params(ch, kind),
            word_index,
            sentence_index,
            paragraph_index,
        });
        match kind {
            SpeechSymbolKind::Whitespace => word_index += 1,
            SpeechSymbolKind::Punctuation if matches!(ch, '.' | '!' | '?') => {
                sentence_index += 1;
                word_index += 1;
            }
            SpeechSymbolKind::ParagraphBreak => {
                paragraph_index += 1;
                sentence_index += 1;
                word_index += 1;
            }
            _ => {}
        }
    }
    units
}

fn speech_units_from_tokens(text: &str, mode: SpeechInputMode) -> Vec<SpeechUnit> {
    let mut units = Vec::new();
    let mut token = String::new();
    let mut word_index = 0;
    let mut sentence_index = 0;
    let mut paragraph_index = 0;

    let flush_word = |token: &mut String,
                      units: &mut Vec<SpeechUnit>,
                      word_index: usize,
                      sentence_index: usize,
                      paragraph_index: usize| {
        if token.is_empty() {
            return;
        }
        let source = token.clone();
        for label in approximate_phonemes(token) {
            let kind = phoneme_kind(&label);
            let params = speech_phoneme_params(&label);
            let duration_s = speech_unit_duration_for_params(kind, mode, &label);
            units.push(SpeechUnit {
                label,
                source: source.clone(),
                kind,
                duration_s,
                gap_s: 0.004,
                emphasis: speech_source_emphasis(&source),
                params,
                word_index,
                sentence_index,
                paragraph_index,
            });
        }
        token.clear();
    };

    for ch in text.chars() {
        if ch == '\n' {
            flush_word(
                &mut token,
                &mut units,
                word_index,
                sentence_index,
                paragraph_index,
            );
            units.push(SpeechUnit {
                label: "PARA".to_string(),
                source: "\n".to_string(),
                kind: SpeechSymbolKind::ParagraphBreak,
                duration_s: 0.0,
                gap_s: speech_symbol_gap('\n', SpeechSymbolKind::ParagraphBreak, mode),
                emphasis: 0.0,
                params: speech_symbol_params('\n', SpeechSymbolKind::ParagraphBreak),
                word_index,
                sentence_index,
                paragraph_index,
            });
            paragraph_index += 1;
            sentence_index += 1;
            word_index += 1;
        } else if ch.is_whitespace() {
            flush_word(
                &mut token,
                &mut units,
                word_index,
                sentence_index,
                paragraph_index,
            );
            units.push(SpeechUnit {
                label: "SPACE".to_string(),
                source: " ".to_string(),
                kind: SpeechSymbolKind::Whitespace,
                duration_s: 0.0,
                gap_s: speech_symbol_gap(' ', SpeechSymbolKind::Whitespace, mode),
                emphasis: 0.0,
                params: speech_symbol_params(' ', SpeechSymbolKind::Whitespace),
                word_index,
                sentence_index,
                paragraph_index,
            });
            word_index += 1;
        } else if matches!(classify_speech_symbol(ch), SpeechSymbolKind::Punctuation) {
            flush_word(
                &mut token,
                &mut units,
                word_index,
                sentence_index,
                paragraph_index,
            );
            let kind = SpeechSymbolKind::Punctuation;
            units.push(SpeechUnit {
                label: ch.to_string(),
                source: ch.to_string(),
                kind,
                duration_s: speech_symbol_duration(ch, kind, mode),
                gap_s: speech_symbol_gap(ch, kind, mode),
                emphasis: speech_source_emphasis(ch.to_string().as_str()),
                params: speech_symbol_params(ch, kind),
                word_index,
                sentence_index,
                paragraph_index,
            });
            if matches!(ch, '.' | '!' | '?') {
                sentence_index += 1;
            }
            word_index += 1;
        } else {
            token.push(ch);
        }
    }

    flush_word(
        &mut token,
        &mut units,
        word_index,
        sentence_index,
        paragraph_index,
    );

    units
}

fn approximate_phonemes(word: &str) -> Vec<String> {
    let upper = word.to_ascii_uppercase();
    if upper.is_empty() {
        return Vec::new();
    }
    let bytes = upper.as_bytes();
    let mut i = 0usize;
    let mut out = Vec::new();
    while i < bytes.len() {
        let rem = &upper[i..];
        let (step, label) = if rem.starts_with("TH") {
            (2, "TH")
        } else if rem.starts_with("SH") {
            (2, "SH")
        } else if rem.starts_with("CH") {
            (2, "CH")
        } else if rem.starts_with("PH") {
            (2, "F")
        } else if rem.starts_with("NG") {
            (2, "NG")
        } else if rem.starts_with("QU") {
            (2, "KW")
        } else if rem.starts_with("EE") {
            (2, "IY")
        } else if rem.starts_with("OO") {
            (2, "UW")
        } else if rem.starts_with("AI") || rem.starts_with("AY") {
            (2, "EY")
        } else if rem.starts_with("OW") || rem.starts_with("OU") {
            (2, "AW")
        } else if rem.starts_with("OI") || rem.starts_with("OY") {
            (2, "OY")
        } else if rem.starts_with("AR") {
            (2, "AR")
        } else if rem.starts_with("ER") || rem.starts_with("IR") || rem.starts_with("UR") {
            (2, "ER")
        } else if rem.starts_with("OR") {
            (2, "OR")
        } else {
            let ch = rem.chars().next().unwrap_or(' ');
            let label = match ch {
                'A' => "AH",
                'E' => "EH",
                'I' => "IH",
                'O' => "OH",
                'U' => "UH",
                'Y' => "Y",
                'B' => "B",
                'C' => "K",
                'D' => "D",
                'F' => "F",
                'G' => "G",
                'H' => "HH",
                'J' => "JH",
                'K' => "K",
                'L' => "L",
                'M' => "M",
                'N' => "N",
                'P' => "P",
                'Q' => "K",
                'R' => "R",
                'S' => "S",
                'T' => "T",
                'V' => "V",
                'W' => "W",
                'X' => "KS",
                'Z' => "Z",
                _ => "UH",
            };
            (1, label)
        };
        for sub in label.split_whitespace() {
            out.push(sub.to_string());
        }
        if label == "KW" {
            out.pop();
            out.push("K".to_string());
            out.push("W".to_string());
        } else if label == "KS" {
            out.pop();
            out.push("K".to_string());
            out.push("S".to_string());
        }
        i += step;
    }
    out
}

fn phoneme_kind(label: &str) -> SpeechSymbolKind {
    match label {
        "AH" | "EH" | "IH" | "OH" | "UH" | "IY" | "UW" | "EY" | "AW" | "OY" | "AR" | "ER"
        | "OR" => SpeechSymbolKind::Vowel,
        _ => SpeechSymbolKind::Consonant,
    }
}

fn speech_source_emphasis(source: &str) -> f64 {
    if source.chars().all(|ch| ch.is_uppercase()) || source.contains('!') || source.contains('?') {
        1.0
    } else if source.chars().any(|ch| ch.is_ascii_digit()) {
        0.55
    } else {
        0.2
    }
}

fn speech_unit_duration_for_params(
    kind: SpeechSymbolKind,
    mode: SpeechInputMode,
    label: &str,
) -> f64 {
    let seed = label.chars().next().unwrap_or('A');
    speech_symbol_duration(seed, kind, mode)
        * if matches!(kind, SpeechSymbolKind::Vowel) {
            1.08
        } else {
            1.0
        }
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

fn speech_phoneme_params(label: &str) -> SpeechSymbolParams {
    match label {
        "AH" => SpeechSymbolParams {
            voiced: true,
            noisy: false,
            pitch_mul: 0.98,
            formant1: 760.0,
            formant2: 1_280.0,
            brightness: 0.46,
            noise_mul: 0.0,
            vowel_mul: 1.0,
        },
        "EH" => SpeechSymbolParams {
            voiced: true,
            noisy: false,
            pitch_mul: 1.02,
            formant1: 560.0,
            formant2: 1_840.0,
            brightness: 0.48,
            noise_mul: 0.0,
            vowel_mul: 1.0,
        },
        "IH" | "IY" => SpeechSymbolParams {
            voiced: true,
            noisy: false,
            pitch_mul: 1.08,
            formant1: 340.0,
            formant2: 2_180.0,
            brightness: 0.52,
            noise_mul: 0.0,
            vowel_mul: 1.0,
        },
        "OH" | "OR" => SpeechSymbolParams {
            voiced: true,
            noisy: false,
            pitch_mul: 0.92,
            formant1: 480.0,
            formant2: 980.0,
            brightness: 0.4,
            noise_mul: 0.0,
            vowel_mul: 0.95,
        },
        "UH" | "UW" => SpeechSymbolParams {
            voiced: true,
            noisy: false,
            pitch_mul: 0.86,
            formant1: 360.0,
            formant2: 820.0,
            brightness: 0.36,
            noise_mul: 0.0,
            vowel_mul: 0.92,
        },
        "EY" | "AW" | "OY" | "AR" | "ER" => SpeechSymbolParams {
            voiced: true,
            noisy: false,
            pitch_mul: 1.0,
            formant1: 540.0,
            formant2: 1_520.0,
            brightness: 0.5,
            noise_mul: 0.0,
            vowel_mul: 1.0,
        },
        "TH" | "SH" | "S" | "Z" | "F" | "V" | "HH" => SpeechSymbolParams {
            voiced: matches!(label, "Z" | "V"),
            noisy: true,
            pitch_mul: 1.04,
            formant1: 640.0,
            formant2: 2_700.0,
            brightness: 0.92,
            noise_mul: 1.0,
            vowel_mul: 0.22,
        },
        "CH" | "JH" => SpeechSymbolParams {
            voiced: matches!(label, "JH"),
            noisy: true,
            pitch_mul: 1.03,
            formant1: 620.0,
            formant2: 2_300.0,
            brightness: 0.86,
            noise_mul: 0.8,
            vowel_mul: 0.24,
        },
        "M" | "N" | "NG" => SpeechSymbolParams {
            voiced: true,
            noisy: false,
            pitch_mul: 0.84,
            formant1: 280.0,
            formant2: 1_180.0,
            brightness: 0.32,
            noise_mul: 0.05,
            vowel_mul: 0.44,
        },
        "L" | "R" | "W" | "Y" => SpeechSymbolParams {
            voiced: true,
            noisy: false,
            pitch_mul: 1.02,
            formant1: 420.0,
            formant2: 1_420.0,
            brightness: 0.42,
            noise_mul: 0.03,
            vowel_mul: 0.55,
        },
        "P" | "T" | "K" | "B" | "D" | "G" => SpeechSymbolParams {
            voiced: matches!(label, "B" | "D" | "G"),
            noisy: true,
            pitch_mul: 1.01,
            formant1: 540.0,
            formant2: 1_520.0,
            brightness: 0.62,
            noise_mul: 0.56,
            vowel_mul: 0.3,
        },
        _ => SpeechSymbolParams {
            voiced: true,
            noisy: false,
            pitch_mul: 1.0,
            formant1: 520.0,
            formant2: 1_480.0,
            brightness: 0.45,
            noise_mul: 0.08,
            vowel_mul: 0.85,
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

fn pulse(freq: f64, t: f64, duty: f64) -> f64 {
    if (freq * t).fract() < duty.clamp(0.01, 0.99) {
        1.0
    } else {
        -1.0
    }
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
        SpeechOscillator::Sine => (2.0 * PI * freq * t).sin(),
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
        SpeechOscillator::Vowel => soft_clip(
            0.55 * (2.0 * PI * freq * t).sin()
                + 0.3 * (2.0 * PI * formant1 * t).sin()
                + 0.22 * (2.0 * PI * formant2 * t).sin(),
        ),
        SpeechOscillator::Ring => {
            (2.0 * PI * freq * t).sin() * (2.0 * PI * (formant2 * 0.5) * t).sin()
        }
        SpeechOscillator::Fold => {
            let carrier = (2.0 * PI * freq * t).sin() + 0.6 * saw(freq * 2.0, t);
            soft_clip((carrier * fold).sin())
        }
        SpeechOscillator::Organ => soft_clip(
            (2.0 * PI * freq * t).sin()
                + 0.5 * (2.0 * PI * freq * 2.0 * t).sin()
                + 0.25 * (2.0 * PI * freq * 3.0 * t).sin(),
        ),
        SpeechOscillator::Fm => {
            let modulator = (2.0 * PI * (freq * (1.7 + 0.6 * chaos)) * t).sin();
            (2.0 * PI * freq * t + modulator * (1.5 + chaos)).sin()
        }
        SpeechOscillator::Sync => soft_clip(saw(freq, t) + 0.6 * pulse(freq * 2.0, t, 0.35)),
        SpeechOscillator::Lfsr => {
            let hash = (((*phase * 65535.0) as u32)
                .wrapping_mul(1103515245)
                .wrapping_add(12345))
                & 0xffff;
            (hash as f64 / 32767.5) - 1.0
        }
        SpeechOscillator::Grain => {
            let grain_phase = ((*phase * (4.0 + chaos * 8.0)).fract() - 0.5).abs();
            let env = (1.0 - grain_phase * 2.0).max(0.0);
            env * (2.0 * PI * (freq * (1.0 + chaos * 0.25)) * t).sin()
        }
        SpeechOscillator::Chirp => {
            let sweep = freq * (1.0 + chaos * (*phase - 0.5) * 1.8);
            (2.0 * PI * sweep.max(20.0) * t).sin()
        }
        SpeechOscillator::Subharmonic => {
            0.5 * (2.0 * PI * freq * t).sin()
                + 0.3 * (2.0 * PI * freq * 0.5 * t).sin()
                + 0.2 * (2.0 * PI * freq * 0.25 * t).sin()
        }
        SpeechOscillator::Reed => soft_clip(0.72 * saw(freq, t) + 0.28 * triangle(freq * 0.5, t)),
        SpeechOscillator::Click => {
            if *phase < 0.035 {
                1.0 - (*phase / 0.035)
            } else {
                -0.08 * (2.0 * PI * freq * t).sin()
            }
        }
        SpeechOscillator::Comb => {
            let carrier = (2.0 * PI * freq * t).sin();
            soft_clip(carrier + 0.45 * (2.0 * PI * (freq + formant1 * 0.03) * (t - 0.0015)).sin())
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
    backend_kind: SpeechBackendKind,
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
    let shaped = match backend_kind {
        SpeechBackendKind::Lpc => soft_clip(0.74 * crushed + 0.26 * folded),
        SpeechBackendKind::FormantGrid => soft_clip(0.68 * crushed + 0.32 * (folded * 1.15).sin()),
        SpeechBackendKind::SamVocalTract => {
            soft_clip(0.62 * crushed + 0.22 * folded + 0.16 * square(110.0, *sample_hold_counter))
        }
        SpeechBackendKind::ArcadePcm => {
            let coarse = (crushed * 32.0).round() / 32.0;
            soft_clip(0.58 * crushed + 0.2 * folded + 0.22 * coarse)
        }
    };
    soft_clip(shaped)
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

struct BasicScoreProfile {
    clip_pct_weight: f64,
    harshness_weight: f64,
    zero_crossing_weight: f64,
    scale: f64,
    offset: f64,
}

const BASIC_SCORE_PROFILE_V1: BasicScoreProfile = BasicScoreProfile {
    clip_pct_weight: 1.6,
    harshness_weight: 45.0,
    zero_crossing_weight: 200.0,
    scale: 20.0,
    offset: -1000.0,
};

struct PsychoScoreProfile {
    bias: f64,
    clip_weight: f64,
    roughness_weight: f64,
    sharpness_weight: f64,
    dissonance_weight: f64,
    transient_weight: f64,
    harshness_weight: f64,
    inharmonicity_weight: f64,
    binaural_beat_weight: f64,
    beat_conflict_weight: f64,
    tritone_tension_weight: f64,
    wolf_fifth_weight: f64,
    harmonicity_relief_weight: f64,
}

const PSYCHO_SCORE_PROFILE_V1: PsychoScoreProfile = PsychoScoreProfile {
    bias: -4.05,
    clip_weight: 1.6,
    roughness_weight: 1.3,
    sharpness_weight: 1.0,
    dissonance_weight: 1.0,
    transient_weight: 1.2,
    harshness_weight: 0.9,
    inharmonicity_weight: 1.25,
    binaural_beat_weight: 0.85,
    beat_conflict_weight: 1.05,
    tritone_tension_weight: 0.85,
    wolf_fifth_weight: 0.75,
    harmonicity_relief_weight: 0.45,
};

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
    let basic_profile = BASIC_SCORE_PROFILE_V1;
    let colbys = ((clipped_pct * basic_profile.clip_pct_weight
        + harshness_ratio * basic_profile.harshness_weight
        + zero_crossing_rate * basic_profile.zero_crossing_weight)
        * basic_profile.scale
        + basic_profile.offset)
        .clamp(COLBYS_MIN as f64, COLBYS_MAX as f64);

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
        colbys,
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

    let psycho_profile = PSYCHO_SCORE_PROFILE_V1;
    let weighted_sum = psycho_profile.bias
        + psycho_profile.clip_weight * clip_norm
        + psycho_profile.roughness_weight * roughness_norm
        + psycho_profile.sharpness_weight * sharpness_norm
        + psycho_profile.dissonance_weight * dissonance_norm
        + psycho_profile.transient_weight * transient_norm
        + psycho_profile.harshness_weight * harshness_norm
        + psycho_profile.inharmonicity_weight * inharmonicity_norm
        + psycho_profile.binaural_beat_weight * binaural_beat_norm
        + psycho_profile.beat_conflict_weight * beat_conflict_norm
        + psycho_profile.tritone_tension_weight * tritone_tension_norm
        + psycho_profile.wolf_fifth_weight * wolf_fifth_norm
        - psycho_profile.harmonicity_relief_weight * harmonicity_norm;

    let colbys =
        (2000.0 * sigmoid(weighted_sum) - 1000.0).clamp(COLBYS_MIN as f64, COLBYS_MAX as f64);

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
        colbys,
        fft_size: opts.fft_size,
        hop_size: opts.hop_size,
    }
}

fn mean_norm(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        finite01(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn compute_joke_analysis(basic: &Analysis, psycho: &PsychoAnalysis) -> JokeAnalysis {
    let c = psycho.clip_norm;
    let r = psycho.roughness_norm;
    let s = psycho.sharpness_norm;
    let d = psycho.dissonance_norm;
    let t = psycho.transient_norm;
    let h = psycho.harmonicity_norm;
    let i = psycho.inharmonicity_norm;
    let b = psycho.binaural_beat_norm;
    let f = psycho.beat_conflict_norm;
    let w = psycho.wolf_fifth_norm;

    let crest_norm = finite01(basic.crest_factor_db.max(0.0) / 18.0);
    let harsh_base = finite01(basic.harshness_ratio / 0.4);
    let m = mean_norm(&[r, f, t]);
    let g = finite01(0.55 * t + 0.45 * c);
    let p = finite01(0.7 * t + 0.3 * c);
    let q = finite01(0.55 * s + 0.45 * c);
    let z = finite01(0.65 * s + 0.35 * f);
    let l = finite01(0.55 * t + 0.45 * crest_norm);
    let o = finite01(0.55 * s + 0.45 * i);
    let a = finite01(0.5 * s + 0.35 * i + 0.15 * c);
    let e = finite01(0.6 * t + 0.4 * harsh_base);
    let n = finite01(0.5 * d + 0.5 * i);
    let y = finite01(0.5 * r + 0.3 * s + 0.2 * b);
    let x = finite01(0.65 * b + 0.35 * f);
    let j = finite01(0.5 * t + 0.5 * f);
    let v = finite01(0.45 * f + 0.3 * b + 0.25 * r);
    let k = finite01(0.45 * t + 0.35 * d + 0.2 * i);

    let academic_cluster = mean_norm(&[c, r, s, d, a, z, q]);
    let bureaucracy = mean_norm(&[t, i, b, f, w, m, g, p, l, o, e, n, y, x, j, v, k]);
    let all_terms = [
        c, r, s, d, t, h, i, b, f, w, m, g, p, q, z, l, o, a, e, n, y, x, j, v, k,
    ];
    let min_term = all_terms
        .iter()
        .copied()
        .fold(1.0_f64, |acc, value| acc.min(value));
    let all_mean = mean_norm(&all_terms);
    let all_high_bonus = finite01(0.55 * min_term + 0.45 * all_mean);
    let harmonicity_relief = h;

    let weighted_sum = -4.2
        + 2.6 * academic_cluster
        + 1.35 * bureaucracy
        + 2.0 * all_high_bonus
        + 0.25 * i
        + 0.2 * c
        - 0.55 * harmonicity_relief
        + 0.35 * harmonicity_relief * academic_cluster;
    let uglierbasis_index = 1000.0 * sigmoid(weighted_sum);

    let verdict = if uglierbasis_index >= 995.0 {
        "please turn that off"
    } else if academic_cluster >= 0.78 {
        "academically ugly"
    } else if harmonicity_relief >= 0.72 && uglierbasis_index < 500.0 {
        "briefly spared by harmonicity"
    } else {
        "bureaucratically offensive"
    };

    JokeAnalysis {
        uglierbasis_index: uglierbasis_index.clamp(0.0, 1000.0),
        verdict: verdict.to_string(),
        academic_cluster_norm: academic_cluster,
        bureaucratic_overhead_norm: bureaucracy,
        all_high_bonus_norm: all_high_bonus,
        harmonicity_relief_norm: harmonicity_relief,
        weighted_sum,
        clip_arrogance: c,
        roughness: r,
        sharpness: s,
        dissonance: d,
        transient_density: t,
        harmonicity: h,
        inharmonicity: i,
        binaural_beat_pressure: b,
        beat_conflict: f,
        wolf_fifth_tension: w,
        modulation_glare: m,
        gate_surprise: g,
        pop_density: p,
        quantization_shame: q,
        zipper_noise: z,
        loudness_lurch: l,
        overtone_hostility: o,
        alias_spray: a,
        envelope_panic: e,
        notch_cruelty: n,
        hysteresis_squeal: y,
        stereo_argument: x,
        jitter: j,
        vibrato_malpractice: v,
        cadence_collapse: k,
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
    target_colbys: i32,
    flavor: GoFlavor,
    seed: u64,
    contour: Option<&UglinessContour>,
) {
    if buffer.is_empty() {
        return;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let intensity = colbys_to_intensity(target_colbys).clamp(0.001, 1.0);
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
        return COLBYS_NEUTRAL as f64;
    }
    if contour.points.len() == 1 {
        return contour.points[0].colbys as f64;
    }
    let t = t_norm.clamp(0.0, 1.0);
    if t <= contour.points[0].t {
        return contour.points[0].colbys as f64;
    }
    for pair in contour.points.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        if t <= b.t {
            if (b.t - a.t).abs() <= EPS64 {
                return b.colbys as f64;
            }
            match contour.interpolation {
                ContourInterpolation::Step => return a.colbys as f64,
                ContourInterpolation::Linear => {
                    let alpha = ((t - a.t) / (b.t - a.t)).clamp(0.0, 1.0);
                    return a.colbys as f64 + alpha * (b.colbys as f64 - a.colbys as f64);
                }
            }
        }
    }
    contour
        .points
        .last()
        .map(|p| p.colbys as f64)
        .unwrap_or(COLBYS_NEUTRAL as f64)
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
        let target_colbys =
            contour_level_at(contour, t_norm).clamp(COLBYS_MIN as f64, COLBYS_MAX as f64);
        let intensity = colbys_to_intensity(target_colbys.round() as i32).clamp(0.001, 1.0);
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
        SurroundLayout::FiveOneTwo => (
            vec![
                azimuth_deg(-30.0),  // L
                azimuth_deg(30.0),   // R
                azimuth_deg(0.0),    // C
                [0.0, 0.5, 0.0],     // LFE
                azimuth_deg(-110.0), // Ls
                azimuth_deg(110.0),  // Rs
                [-0.45, 0.35, 0.85], // Ltf
                [0.45, 0.35, 0.85],  // Rtf
            ],
            Some(3),
        ),
        SurroundLayout::FiveOneFour => (
            vec![
                azimuth_deg(-30.0),   // L
                azimuth_deg(30.0),    // R
                azimuth_deg(0.0),     // C
                [0.0, 0.5, 0.0],      // LFE
                azimuth_deg(-110.0),  // Ls
                azimuth_deg(110.0),   // Rs
                [-0.45, 0.35, 0.85],  // Ltf
                [0.45, 0.35, 0.85],   // Rtf
                [-0.45, -0.35, 0.85], // Ltr
                [0.45, -0.35, 0.85],  // Rtr
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
        SurroundLayout::SevenOneTwo => (
            vec![
                azimuth_deg(-30.0),  // L
                azimuth_deg(30.0),   // R
                azimuth_deg(0.0),    // C
                [0.0, 0.5, 0.0],     // LFE
                azimuth_deg(-90.0),  // Lss
                azimuth_deg(90.0),   // Rss
                azimuth_deg(-150.0), // Lrs
                azimuth_deg(150.0),  // Rrs
                [-0.45, 0.35, 0.85], // Ltf
                [0.45, 0.35, 0.85],  // Rtf
            ],
            Some(3),
        ),
        SurroundLayout::SevenOneFour => (
            vec![
                azimuth_deg(-30.0),   // L
                azimuth_deg(30.0),    // R
                azimuth_deg(0.0),     // C
                [0.0, 0.5, 0.0],      // LFE
                azimuth_deg(-90.0),   // Lss
                azimuth_deg(90.0),    // Rss
                azimuth_deg(-150.0),  // Lrs
                azimuth_deg(150.0),   // Rrs
                [-0.45, 0.35, 0.85],  // Ltf
                [0.45, 0.35, 0.85],   // Rtf
                [-0.45, -0.35, 0.85], // Ltr
                [0.45, -0.35, 0.85],  // Rtr
            ],
            Some(3),
        ),
        SurroundLayout::NineOneSix => (
            vec![
                azimuth_deg(-30.0),   // L
                azimuth_deg(30.0),    // R
                azimuth_deg(0.0),     // C
                [0.0, 0.5, 0.0],      // LFE
                azimuth_deg(-60.0),   // Lw
                azimuth_deg(60.0),    // Rw
                azimuth_deg(-90.0),   // Lss
                azimuth_deg(90.0),    // Rss
                azimuth_deg(-150.0),  // Lrs
                azimuth_deg(150.0),   // Rrs
                [-0.55, 0.65, 0.85],  // Ltf
                [0.55, 0.65, 0.85],   // Rtf
                [0.0, 0.75, 0.9],     // Tm
                [-0.55, -0.55, 0.85], // Ltr
                [0.55, -0.55, 0.85],  // Rtr
                [0.0, -0.75, 0.9],    // Trm
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
        assert!(catastrophic_report.colbys > hum_report.colbys + 300.0);
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
                joke: false,
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
                joke: false,
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
                joke: false,
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
                joke: false,
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
        assert!(report.colbys >= -1000.0 && report.colbys <= 1000.0);
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
                joke: false,
            },
            None,
        );
        assert!(psycho.colbys < -600.0);
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
                UglinessContourPoint {
                    t: 0.0,
                    colbys: -800,
                },
                UglinessContourPoint {
                    t: 1.0,
                    colbys: 800,
                },
            ],
        };
        let step = UglinessContour {
            version: 1,
            name: None,
            description: None,
            interpolation: ContourInterpolation::Step,
            points: vec![
                UglinessContourPoint {
                    t: 0.0,
                    colbys: -800,
                },
                UglinessContourPoint {
                    t: 1.0,
                    colbys: 800,
                },
            ],
        };
        assert!(contour_level_at(&linear, 0.5).abs() < 1e-6);
        assert_eq!(contour_level_at(&step, 0.5), -800.0);
    }

    #[test]
    fn contour_json_supports_legacy_level_and_new_colbys_keys() {
        let legacy = r#"{
            "version": 1,
            "interpolation": "linear",
            "points": [
                { "t": 0.0, "level": -250 },
                { "t": 1.0, "level": 900 }
            ]
        }"#;
        let modern = r#"{
            "version": 1,
            "interpolation": "linear",
            "points": [
                { "t": 0.0, "colbys": -250 },
                { "t": 1.0, "colbys": 900 }
            ]
        }"#;

        let legacy_contour: UglinessContour = serde_json::from_str(legacy).expect("legacy contour");
        let modern_contour: UglinessContour = serde_json::from_str(modern).expect("modern contour");
        assert_eq!(legacy_contour.points[0].colbys, -250);
        assert_eq!(modern_contour.points[1].colbys, 900);
        validate_ugliness_contour(&legacy_contour).expect("legacy contour valid");
        validate_ugliness_contour(&modern_contour).expect("modern contour valid");
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
                joke: false,
            },
        )
        .expect("analyze psycho");

        assert_eq!(report.model, "psycho");
        assert!(report.colbys >= -1000.0 && report.colbys <= 1000.0);
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
    fn joke_analysis_returns_bounded_score_and_breakdown() {
        let out = std::env::temp_dir().join(format!("usg_test_{}_joke.wav", std::process::id()));
        let opts = RenderOptions {
            duration: 0.35,
            sample_rate: 44_100,
            seed: Some(31337),
            style: Style::Punish,
            gain: 0.9,
            normalize: true,
            normalize_dbfs: 0.0,
            output_encoding: OutputEncoding::Float32,
        };
        render_to_wav(&out, &opts).expect("render");

        let report = analyze_wav_with_options(
            &out,
            &AnalyzeOptions {
                model: AnalyzeModel::Basic,
                fft_size: 1024,
                hop_size: 256,
                joke: true,
            },
        )
        .expect("analyze joke");

        let joke = report.joke.expect("joke analysis");
        assert!((0.0..=1000.0).contains(&joke.uglierbasis_index));
        assert!((0.0..=1.0).contains(&joke.academic_cluster_norm));
        assert!((0.0..=1.0).contains(&joke.bureaucratic_overhead_norm));
        assert!((0.0..=1.0).contains(&joke.all_high_bonus_norm));
        assert!((0.0..=1.0).contains(&joke.harmonicity_relief_norm));
        assert!(!joke.verdict.is_empty());
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

        assert_eq!(summary.target_colbys, 8);
        assert_eq!(summary.flavor, GoFlavor::Glitch);
        assert!(summary.frames > 1000);
        let analyzed = analyze_wav(&out_path).expect("analyze go output");
        assert!(analyzed.duration_s > 0.2);
        let _ = fs::remove_file(in_path);
        let _ = fs::remove_file(out_path);
    }
}
