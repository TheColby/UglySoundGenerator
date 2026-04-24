use std::fs;

use hound::{SampleFormat, WavReader};
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_usg")
}

fn temp_dir(label: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("usg_cli_{label}_{now}"));
    fs::create_dir_all(&dir).expect("mkdir");
    dir
}

#[test]
fn analyze_reports_scores_as_out_of_1000() {
    let dir = temp_dir("analyze");
    let wav = dir.join("input.wav");

    let render = Command::new(bin())
        .args([
            "render",
            "--output",
            wav.to_str().expect("wav path"),
            "--duration",
            "0.1",
            "--style",
            "hum",
        ])
        .output()
        .expect("render command");
    assert!(render.status.success(), "render failed");

    let analyze = Command::new(bin())
        .args(["analyze", wav.to_str().expect("wav path")])
        .output()
        .expect("analyze command");
    assert!(analyze.status.success(), "analyze failed");
    let stdout = String::from_utf8_lossy(&analyze.stdout);
    assert!(stdout.contains(" Co"), "stdout was:\n{stdout}");
}

#[test]
fn analyze_supports_joke_mode() {
    let dir = temp_dir("analyze_joke");
    let wav = dir.join("input.wav");

    let render = Command::new(bin())
        .args([
            "render",
            "--output",
            wav.to_str().expect("wav path"),
            "--duration",
            "0.1",
            "--style",
            "punish",
        ])
        .output()
        .expect("render command");
    assert!(render.status.success(), "render failed");

    let analyze = Command::new(bin())
        .args(["analyze", wav.to_str().expect("wav path"), "--joke"])
        .output()
        .expect("analyze joke command");
    assert!(analyze.status.success(), "analyze --joke failed");
    let stdout = String::from_utf8_lossy(&analyze.stdout);
    assert!(
        stdout.contains("joke.uglierbasis_index"),
        "stdout was:\n{stdout}"
    );
    assert!(stdout.contains("joke.verdict"), "stdout was:\n{stdout}");
}

#[test]
fn analyze_json_includes_score_metadata() {
    let dir = temp_dir("analyze_json_meta");
    let wav = dir.join("input.wav");

    let render = Command::new(bin())
        .args([
            "render",
            "--output",
            wav.to_str().expect("wav path"),
            "--duration",
            "0.1",
            "--style",
            "hum",
        ])
        .output()
        .expect("render command");
    assert!(render.status.success(), "render failed");

    let analyze = Command::new(bin())
        .args(["analyze", wav.to_str().expect("wav path"), "--json"])
        .output()
        .expect("analyze json command");
    assert!(analyze.status.success(), "analyze --json failed");
    let stdout = String::from_utf8_lossy(&analyze.stdout);
    assert!(
        stdout.contains("\"score_metadata\""),
        "stdout was:\n{stdout}"
    );
    assert!(stdout.contains("\"profile\""), "stdout was:\n{stdout}");
    assert!(
        stdout.contains("\"calibrated_from_listening_tests\": false"),
        "stdout was:\n{stdout}"
    );
}

#[test]
fn render_pack_reports_scores_as_out_of_1000() {
    let dir = temp_dir("pack");
    let pack = dir.join("pack");
    let pack_str = pack.to_str().expect("pack path");

    let out = Command::new(bin())
        .args([
            "render-pack",
            "--out-dir",
            pack_str,
            "--duration",
            "0.1",
            "--top",
            "2",
        ])
        .output()
        .expect("render-pack command");
    assert!(out.status.success(), "render-pack failed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains(" Co"), "stdout was:\n{stdout}");
}

#[test]
fn go_help_includes_contour_flags() {
    let help = Command::new(bin())
        .args(["go", "--help"])
        .output()
        .expect("go help");
    assert!(help.status.success(), "go --help failed");
    let stdout = String::from_utf8_lossy(&help.stdout);
    assert!(stdout.contains("--level-contour"));
    assert!(stdout.contains("--level-contour-json"));
}

#[test]
fn backends_reports_gpu_post_fx_defaults() {
    let out = Command::new(bin())
        .args(["backends"])
        .output()
        .expect("backends");
    assert!(out.status.success(), "backends failed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("GPU post-FX defaults"));
    assert!(stdout.contains("USG_GPU_DRIVE"));
    assert!(stdout.contains("Metal probe"));
    assert!(stdout.contains("CUDA probe"));
}

#[test]
fn render_supports_int24_output() {
    let dir = temp_dir("int24");
    let wav = dir.join("int24.wav");

    let render = Command::new(bin())
        .args([
            "render",
            "--output",
            wav.to_str().expect("wav path"),
            "--duration",
            "0.1",
            "--style",
            "punish",
            "--sample-format",
            "int",
            "--bit-depth",
            "24",
        ])
        .output()
        .expect("render command");
    assert!(
        render.status.success(),
        "render failed: {}",
        String::from_utf8_lossy(&render.stderr)
    );

    let reader = WavReader::open(&wav).expect("wav reader");
    let spec = reader.spec();
    assert_eq!(spec.sample_format, SampleFormat::Int);
    assert_eq!(spec.bits_per_sample, 24);
}

#[test]
fn piece_generates_requested_channel_count() {
    let dir = temp_dir("piece");
    let wav = dir.join("piece.wav");

    let render = Command::new(bin())
        .args([
            "piece",
            "--output",
            wav.to_str().expect("wav path"),
            "--duration",
            "0.6",
            "--channels",
            "4",
            "--events-per-second",
            "8",
            "--min-event-duration",
            "0.02",
            "--max-event-duration",
            "0.08",
            "--seed",
            "42",
        ])
        .output()
        .expect("piece command");
    assert!(
        render.status.success(),
        "piece failed: {}",
        String::from_utf8_lossy(&render.stderr)
    );

    let reader = WavReader::open(&wav).expect("wav reader");
    let spec = reader.spec();
    assert_eq!(spec.channels, 4);
    assert_eq!(spec.sample_format, SampleFormat::Float);
    assert_eq!(spec.bits_per_sample, 32);
}

#[test]
fn benchmark_can_export_json_and_csv() {
    let dir = temp_dir("bench_export");
    let json_path = dir.join("bench.json");
    let csv_path = dir.join("bench.csv");

    let out = Command::new(bin())
        .args([
            "benchmark",
            "--runs",
            "1",
            "--duration",
            "0.1",
            "--json-output",
            json_path.to_str().expect("json path"),
            "--csv-output",
            csv_path.to_str().expect("csv path"),
        ])
        .output()
        .expect("benchmark command");
    assert!(
        out.status.success(),
        "benchmark failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json = fs::read_to_string(&json_path).expect("json output");
    let csv = fs::read_to_string(&csv_path).expect("csv output");
    assert!(
        json.contains("\"rows\""),
        "json was:
{json}"
    );
    assert!(
        csv.contains("rank,backend,average_ms"),
        "csv was:
{csv}"
    );
}

#[test]
fn presets_command_lists_and_shows_builtin_presets() {
    let list = Command::new(bin())
        .args(["presets"])
        .output()
        .expect("presets list");
    assert!(list.status.success(), "presets list failed");
    let stdout = String::from_utf8_lossy(&list.stdout);
    assert!(
        stdout.contains("01_linear_curve_01"),
        "stdout was:
{stdout}"
    );

    let show = Command::new(bin())
        .args(["presets", "--show", "01_linear_curve_01", "--json"])
        .output()
        .expect("presets show");
    assert!(show.status.success(), "presets show failed");
    let stdout = String::from_utf8_lossy(&show.stdout);
    assert!(
        stdout.contains("\"points\""),
        "stdout was:
{stdout}"
    );
}

#[test]
fn speech_command_renders_text_to_wav() {
    let dir = temp_dir("speech");
    let wav = dir.join("speech.wav");
    let timeline = dir.join("speech.timeline.json");
    let analysis = dir.join("speech.analysis.json");

    let out = Command::new(bin())
        .args([
            "speech",
            "--output",
            wav.to_str().expect("wav path"),
            "--text",
            "HELLO WORLD FROM 1980",
            "--profile",
            "sp0256",
            "--primary-osc",
            "phoneme",
            "--secondary-osc",
            "mandelbrot",
            "--tertiary-osc",
            "strange",
            "--timeline-json",
            timeline.to_str().expect("timeline path"),
            "--analysis-json",
            analysis.to_str().expect("analysis path"),
        ])
        .output()
        .expect("speech command");
    assert!(
        out.status.success(),
        "speech failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let reader = WavReader::open(&wav).expect("speech wav");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1);
    assert!(reader.duration() > 1_000);
    assert!(timeline.exists(), "timeline json missing");
    assert!(analysis.exists(), "analysis json missing");
}

#[test]
fn speech_defaults_to_192khz_float32_output() {
    let dir = temp_dir("speech_defaults");
    let wav = dir.join("speech_defaults.wav");

    let out = Command::new(bin())
        .args([
            "speech",
            "--output",
            wav.to_str().expect("wav path"),
            "--text",
            "UGLY DEFAULT CHECK",
        ])
        .output()
        .expect("speech command");
    assert!(
        out.status.success(),
        "speech failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let reader = WavReader::open(&wav).expect("speech wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 192_000);
    assert_eq!(spec.sample_format, SampleFormat::Float);
    assert_eq!(spec.bits_per_sample, 32);
}

#[test]
fn go_supports_paper_dissonancizer_flavors() {
    let dir = temp_dir("dissonancizer_go");
    let input = dir.join("input.wav");
    let output = dir.join("ringed.go.wav");

    let render = Command::new(bin())
        .args([
            "render",
            "--output",
            input.to_str().expect("input path"),
            "--duration",
            "0.15",
            "--style",
            "hum",
        ])
        .output()
        .expect("render input");
    assert!(
        render.status.success(),
        "render failed: {}",
        String::from_utf8_lossy(&render.stderr)
    );

    let go = Command::new(bin())
        .args([
            "go",
            input.to_str().expect("input path"),
            "--output",
            output.to_str().expect("output path"),
            "--type",
            "dissonance-ring",
            "--level",
            "850",
        ])
        .output()
        .expect("go dissonance-ring");
    assert!(
        go.status.success(),
        "go failed: {}",
        String::from_utf8_lossy(&go.stderr)
    );

    let reader = WavReader::open(&output).expect("go output wav");
    assert!(reader.duration() > 1_000);
}

#[test]
fn go_defaults_to_192khz_for_upmix_output() {
    let dir = temp_dir("go_upmix_sr");
    let input = dir.join("input.wav");
    let output = dir.join("upmix.go.wav");

    let render = Command::new(bin())
        .args([
            "render",
            "--output",
            input.to_str().expect("input path"),
            "--duration",
            "0.1",
            "--style",
            "hum",
            "--sample-rate",
            "44100",
        ])
        .output()
        .expect("render input");
    assert!(
        render.status.success(),
        "render failed: {}",
        String::from_utf8_lossy(&render.stderr)
    );

    let go = Command::new(bin())
        .args([
            "go",
            input.to_str().expect("input path"),
            "--output",
            output.to_str().expect("output path"),
            "--upmix",
            "5.1",
            "--type",
            "punish",
        ])
        .output()
        .expect("go output");
    assert!(
        go.status.success(),
        "go failed: {}",
        String::from_utf8_lossy(&go.stderr)
    );

    let reader = WavReader::open(&output).expect("go wav");
    let spec = reader.spec();
    assert_eq!(spec.channels, 6);
    assert_eq!(spec.sample_rate, 192_000);
}

#[test]
fn chain_supports_builtin_preset() {
    let dir = temp_dir("chain_preset");
    let output = dir.join("preset_chain.wav");

    let out = Command::new(bin())
        .args([
            "chain",
            "--preset",
            "ps1_grit",
            "--duration",
            "0.1",
            "--output",
            output.to_str().expect("output path"),
        ])
        .output()
        .expect("chain preset");
    assert!(
        out.status.success(),
        "chain preset failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(output.exists());
}

#[test]
fn presets_support_chain_family_and_versioned_contours() {
    let list = Command::new(bin())
        .args(["presets", "--kind", "chain"])
        .output()
        .expect("chain preset list");
    assert!(list.status.success(), "chain preset list failed");
    let stdout = String::from_utf8_lossy(&list.stdout);
    assert!(
        stdout.contains("ps1_grit"),
        "stdout was:
{stdout}"
    );

    let show = Command::new(bin())
        .args(["presets", "--show", "01_linear_curve_01", "--json"])
        .output()
        .expect("contour show");
    assert!(show.status.success(), "contour show failed");
    let stdout = String::from_utf8_lossy(&show.stdout);
    assert!(
        stdout.contains("\"version\": 1"),
        "stdout was:
{stdout}"
    );
}

#[test]
fn render_can_force_streaming_path_for_regression_coverage() {
    let dir = temp_dir("streaming");
    let wav = dir.join("streaming.wav");

    let out = Command::new(bin())
        .env("USG_STREAM_THRESHOLD_FRAMES", "64")
        .args([
            "render",
            "--output",
            wav.to_str().expect("wav path"),
            "--duration",
            "0.1",
            "--style",
            "glitch",
            "--sample-rate",
            "44100",
        ])
        .output()
        .expect("streaming render");
    assert!(
        out.status.success(),
        "streaming render failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(wav.exists());
}

#[test]
fn speech_pack_renders_all_profiles_and_writes_outputs() {
    let dir = temp_dir("speech_pack");

    let out = Command::new(bin())
        .args([
            "speech-pack",
            "--out-dir",
            dir.to_str().expect("dir path"),
            "--text",
            "CHIP",
            "--sample-rate",
            "22050",
        ])
        .output()
        .expect("speech-pack command");
    assert!(
        out.status.success(),
        "speech-pack failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("ugly "), "stdout was:\n{stdout}");
    assert!(stdout.contains("intel "), "stdout was:\n{stdout}");
    assert!(dir.join("summary.json").exists(), "summary.json missing");
    assert!(dir.join("ranking.csv").exists(), "ranking.csv missing");
    assert!(dir.join("report.html").exists(), "report.html missing");

    let csv = fs::read_to_string(dir.join("ranking.csv")).expect("csv");
    assert!(
        csv.contains("rank,profile,colbys,intelligibility_index,rank_score,"),
        "csv header wrong:\n{csv}"
    );
}

#[test]
fn go_accepts_zero_level_in_contour_json() {
    let dir = temp_dir("contour_zero");
    let input = dir.join("input.wav");
    let output = dir.join("output.go.wav");

    let render = Command::new(bin())
        .args([
            "render",
            "--output",
            input.to_str().expect("input path"),
            "--duration",
            "0.1",
            "--style",
            "hum",
        ])
        .output()
        .expect("render input");
    assert!(
        render.status.success(),
        "render failed: {}",
        String::from_utf8_lossy(&render.stderr)
    );

    let go = Command::new(bin())
        .args([
            "go",
            input.to_str().expect("input path"),
            "--output",
            output.to_str().expect("output path"),
            "--type",
            "glitch",
            "--level-contour-json",
            "{\"version\":1,\"interpolation\":\"linear\",\"points\":[{\"t\":0.0,\"level\":0},{\"t\":1.0,\"level\":900}]}",
        ])
        .output()
        .expect("go contour");
    assert!(
        go.status.success(),
        "go failed: {}",
        String::from_utf8_lossy(&go.stderr)
    );
    assert!(output.exists());
}

#[test]
fn analyze_timeline_outputs_csv_with_header() {
    let dir = temp_dir("timeline");
    let wav = dir.join("input.wav");

    let render = Command::new(bin())
        .args([
            "render",
            "--output",
            wav.to_str().expect("wav"),
            "--duration",
            "0.5",
            "--style",
            "harsh",
            "--sample-rate",
            "44100",
        ])
        .output()
        .expect("render");
    assert!(
        render.status.success(),
        "{}",
        String::from_utf8_lossy(&render.stderr)
    );

    let timeline_path = dir.join("timeline.csv");
    let out = Command::new(bin())
        .args([
            "analyze",
            wav.to_str().expect("wav"),
            "--timeline",
            "--timeline-format",
            "csv",
            "--timeline-output",
            timeline_path.to_str().expect("path"),
        ])
        .output()
        .expect("analyze --timeline");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(timeline_path.exists(), "timeline csv missing");
    let csv = fs::read_to_string(&timeline_path).expect("csv read");
    assert!(csv.starts_with("time_s,"), "csv header wrong:\n{csv}");
    assert!(csv.lines().count() > 2, "csv too short:\n{csv}");
}

#[test]
fn mutate_generates_variants_and_summary() {
    let dir = temp_dir("mutate");
    let input = dir.join("input.wav");

    let render = Command::new(bin())
        .args([
            "render",
            "--output",
            input.to_str().expect("input"),
            "--duration",
            "0.2",
            "--style",
            "hum",
            "--sample-rate",
            "22050",
        ])
        .output()
        .expect("render");
    assert!(
        render.status.success(),
        "{}",
        String::from_utf8_lossy(&render.stderr)
    );

    let out_dir = dir.join("mutate_out");
    let out = Command::new(bin())
        .args([
            "mutate",
            input.to_str().expect("input"),
            "--out-dir",
            out_dir.to_str().expect("out_dir"),
            "--count",
            "3",
            "--model",
            "basic",
        ])
        .output()
        .expect("mutate");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out_dir.join("mutate_summary.json").exists(),
        "summary missing"
    );
    let summary = fs::read_to_string(out_dir.join("mutate_summary.json")).expect("summary read");
    assert!(
        summary.contains("ugly_delta"),
        "summary missing ugly_delta:\n{summary}"
    );
}

#[test]
fn normalize_pack_normalizes_a_batch_of_wavs() {
    let dir = temp_dir("normalize_pack");
    let in_dir = dir.join("input");
    let out_dir = dir.join("output");
    fs::create_dir_all(&in_dir).expect("mkdir");

    for i in 0..3 {
        let wav = in_dir.join(format!("item{i}.wav"));
        let render = Command::new(bin())
            .args([
                "render",
                "--output",
                wav.to_str().expect("wav"),
                "--duration",
                "0.15",
                "--style",
                "hum",
                "--sample-rate",
                "22050",
            ])
            .output()
            .expect("render");
        assert!(
            render.status.success(),
            "{}",
            String::from_utf8_lossy(&render.stderr)
        );
    }

    let out = Command::new(bin())
        .args([
            "normalize-pack",
            "--in-dir",
            in_dir.to_str().expect("in_dir"),
            "--out-dir",
            out_dir.to_str().expect("out_dir"),
            "--level",
            "800",
            "--model",
            "basic",
        ])
        .output()
        .expect("normalize-pack");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out_dir.join("normalize_manifest.json").exists(),
        "manifest missing"
    );
    let manifest = fs::read_to_string(out_dir.join("normalize_manifest.json")).expect("manifest");
    assert!(
        manifest.contains("\"files_processed\": 3"),
        "count wrong:\n{manifest}"
    );
}

#[test]
fn evolve_runs_generations_and_writes_lineage() {
    let dir = temp_dir("evolve");

    let out = Command::new(bin())
        .args([
            "evolve",
            "--out-dir",
            dir.to_str().expect("dir"),
            "--generations",
            "2",
            "--population",
            "4",
            "--duration",
            "0.1",
            "--sample-rate",
            "22050",
            "--style",
            "harsh",
            "--model",
            "basic",
        ])
        .output()
        .expect("evolve");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(dir.join("lineage.json").exists(), "lineage.json missing");
    let lineage = fs::read_to_string(dir.join("lineage.json")).expect("lineage read");
    assert!(
        lineage.contains("champion_colbys"),
        "lineage wrong:\n{lineage}"
    );
    assert!(dir.join("gen01").exists(), "gen01 dir missing");
    assert!(dir.join("gen02").exists(), "gen02 dir missing");
}
